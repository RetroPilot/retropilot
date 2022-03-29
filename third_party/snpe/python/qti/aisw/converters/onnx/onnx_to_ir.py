# ==============================================================================
#
#  Copyright (c) 2018-2022 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import sys
import traceback
from packaging import version

from qti.aisw.converters.common.utils import code_to_message

try:
    import onnx
except ImportError as e:
    raise Exception(code_to_message.get_error_message("ERROR_ONNX_NOT_FOUND")(str(e), str(sys.path)))

from qti.aisw.converters.common.converter_ir import op_policies
from qti.aisw.converters.common.converter_ir.axis_tracker import AxisOrders
from qti.aisw.converters.common.converter_base import ConverterFrontend
from .util import *
from . import onnx_translations


# ------------------------------------------------------------------------------
#   The Converter Class
# ------------------------------------------------------------------------------
class OnnxConverterFrontend(ConverterFrontend):
    class ArgParser(ConverterFrontend.ArgParser):
        def __init__(self, **kwargs):
            super(OnnxConverterFrontend.ArgParser, self).__init__(**kwargs)
            # add command-line options custom to onnx converter
            self.add_optional_argument("--dry_run", type=str, nargs='?', const='info', default=None,
                                       help='Evaluates the model without actually converting any ops, and '
                                            'returns unsupported ops/attributes as well as unused inputs and/or '
                                            'outputs if any. Leave empty or specify "info" to see dry run as a '
                                            'table, or specify "debug" to show more detailed messages only"')
            self.add_optional_argument('-d', '--input_dim', nargs=2, action='append',
                                       metavar=('INPUT_NAME', 'INPUT_DIM'),
                                       help="The name and dimension of all the input buffers to the network specified in\n"
                                            "the format [input_name comma-separated-dimensions],\n"
                                            "for example: 'data' 1,224,224,3. \n"
                                            "Note that the quotes should always be included in order to handle special\n"
                                            "characters, spaces, etc.\n"
                                            "NOTE: This feature works only with Onnx 1.6.0 and above")
            self.add_optional_argument('-n', '--no_simplification', action='store_true', default=False,
                                       help="Do not attempt to simplify the model automatically. This may prevent some models from properly converting \n"
                                            "when sequences of unsupported static operations are present.")
            self.add_optional_argument('--dump_inferred_model', action='store_true', default=False,
                                       help=argparse.SUPPRESS)
            self.add_optional_argument('--dump_value_info', action='store_true', default=False,
                                       help=argparse.SUPPRESS)

    def __init__(self, args, *, custom_op_factory=None):
        super(OnnxConverterFrontend, self).__init__(args,
                                                    naming_policy=OnnxNamePolicy(),
                                                    shape_inference_policy=OnnxShapeInferencePolicy(),
                                                    axis_order=AxisOrders.ONNX,
                                                    custom_op_factory=custom_op_factory)
        self.translations = onnx_translations.OnnxTranslations
        self.dry_run = args.dry_run
        self.no_simplification = args.no_simplification
        self.dump_inferred_model = args.dump_inferred_model
        self.dump_value_info = args.dump_value_info
        self.op_info = onnx_translations.OpVersionInfo()
        if args.input_dim is not None:
            (in_names, in_dims) = list(zip(*args.input_dim))
            self.input_names = in_names
            self.input_dims = in_dims
        else:
            self.input_names = None
            self.input_dims = None

        # We can't run simplification and quantization overrides/custom ops as the simplification process
        # could possibly squash layers preventing the custom ops or quantization overrides from being used
        if not self.no_simplification and (args.quantization_overrides or args.custom_op_config_paths):
            self.no_simplification = True
            log_warning("Can't simplify the model when custom ops or quantization overrides are specified, converting without simplification.")

    def evaluate(self, model):
        """
        Performs a dry-run of the Onnx Model without actually converting it, highlighting potential issues with
        attributes, inputs/outputs or opset versions.
        :param model: An Onnx model
        :return:
        """
        from qti.aisw.converters.onnx import model_evaluator
        try:
            onnx.checker.check_model(model)
        except Exception as e:
            log_warning("Potential errors found in {} as per Onnx's in-built checker tool".format(self.input_model_path))
            log_warning("{}: {}", type(e), str(e))
        log_info('Proceeding with model evaluation...................................\n')
        model_evaluator.setup_dry_run(model, self.dry_run)

    def convert(self):
        model = onnx.load(self.input_model_path)

        # Try to simplify the model first
        if not self.no_simplification:
            try:
                import onnxsim
                try:
                    if self.input_names and self.input_dims:
                        dims_dict = {}
                        for i in range(len(self.input_names)):
                            dims_dict[self.input_names[i]] = [int(k) for k in self.input_dims[i].split(',')]
                        model_optimized, check_ok = onnxsim.simplify(model, input_shapes=dims_dict, dynamic_input_shape=True)
                    else:
                        model_optimized, check_ok = onnxsim.simplify(model)
                    if check_ok:
                        log_debug1("Successfully simplified the onnx model!")
                        model = model_optimized
                    else:
                        log_warning("Couldn't simplify the model, attempting normal conversion")
                except Exception as e:
                    log_warning("Onnx model simplification failed, trying unsimplified model. ({}: {})", type(e), str(e))
            except ImportError as e:
                log_warning("Couldn't import onnx-simplifier. ({}: {})", type(e), str(e))
                log_warning("Install the onnx-simplifier for better model compatibility: \"pip3 install onnx-simplifier\"")
            except Exception as e:
                log_warning("Unknown error ({}: {}) during import of onnx simplifier", type(e), str(e))

        self.op_info.set_global_op_ver(model)

        if self.dry_run:
            self.evaluate(model)
            sys.exit(0)

        # Attempt to run shape inference on the full ONNX model so that we gain access to all shape info
        try:
            from onnx import shape_inference
            model = shape_inference.infer_shapes(model)
        except:
            if self.dump_inferred_model:
                log_error("Unable to dump inferred model since ONNX shape inference failed.")
            else:
                log_warning("ONNX shape inference failed.")

        if self.input_dims and self.input_names:
            self._update_input_node(model)

        self.graph.weights = WeightProvider(model)
        self.graph.tensor_to_np_dtype = self._track_tensor_type(model.graph)

        if self.output_names:
            # Trims the existing graph to the output nodes specified
            self._update_output_nodes(model)
        elif model.graph.output:
            # Add the Onnx model outputs to IR Graph
            for value_info in model.graph.output:
                self.graph.output_names.append(str(value_info.name))

        # Dumps the trimmed and inferred model, if it was requested
        if self.dump_inferred_model:
            inferred_model_filename = self.input_model_path.split('.')[0] + "_inferred.onnx"
            onnx.save(model, inferred_model_filename)

        # Dumps the value_info field of the ONNX graph after trimming, for debugging purposes
        if self.dump_value_info and model.graph.value_info:
            original_stdout = sys.stdout
            with open(self.input_model_path.split('.')[0] + "_value_info.info", "w") as file:
                sys.stdout = file
                print(model.graph.value_info)
                sys.stdout = original_stdout
        elif self.dump_value_info:
            log_warning("Unable to dump value info because field is not populated.")

        # populate custom op nodes if config paths are provided; condition is checked in function
        self.populate_custom_op_collection(model, 'onnx')

        # extract inputs
        parameter_names = set()
        for tensor in model.graph.initializer:
            parameter_names.add(str(tensor.name))

        for value_info in model.graph.input:
            name = str(value_info.name)
            if name in parameter_names:
                # weights are usually listed as inputs too.
                continue
            self.translations.apply_method_to_op(converter_type("input", "onnx"),
                                                 onnx_translations.OnnxTranslationBase.ADD_INPUT_OP, value_info, self.graph)

        # extract parameters, infer shapes, etc.
        for i, src_op in enumerate(model.graph.node):
            log_debug(code_to_message.get_debugging_message("DEBUG_CONVERTING_NODE")(i, src_op.op_type))
            src_type = converter_type(src_op.op_type, "onnx")

            try:
                # first check if layer is a registered custom op in an op collection.
                # If so, the layer is added and the outer loop continues.
                if self.custom_op_factory and src_op.op_type in self.custom_op_factory.op_collection:
                    src_type = converter_type('custom', "onnx")
                    node = self.translations.apply_method_to_op(src_type,
                                                                onnx_translations.OnnxTranslationBase.ADD_OP,
                                                                src_op,
                                                                self.graph)
                    self.graph.add_src_op_info(node.op.name, [i for i in src_op.input], [o for o in src_op.output])

                else:
                    # If the op is not a custom operation, check the version and use the
                    # native converter translation
                    supported_version = self.translations.apply_method_to_op(src_type,
                                                                             onnx_translations.OnnxTranslationBase.SUPPORTED_VERSION,
                                                                             src_op.op_type)
                    self.op_info.validate_op_ver(src_op, supported_version)

                    self.translations.apply_method_to_op(src_type,
                                                         onnx_translations.OnnxTranslationBase.ADD_OP,
                                                         src_op,
                                                         self.graph)
            except Exception as e:
                if self.debug:
                    traceback.print_exc()
                log_error("Node %s: %s" % (src_op.name, e))
                sys.exit(-1)

        self.graph.eval_macs_params()
        return self.graph

    def _track_tensor_type(self, graph):
        tensor_to_np_dtype = {}

        for value_info in graph.input:
            tensor_to_np_dtype[value_info.name] = onnx_to_np_dtype.get(value_info.type.tensor_type.elem_type)

        for value_info in graph.value_info:
            tensor_to_np_dtype[value_info.name] = onnx_to_np_dtype.get(value_info.type.tensor_type.elem_type)

        for value_info in graph.output:
            tensor_to_np_dtype[value_info.name] = onnx_to_np_dtype.get(value_info.type.tensor_type.elem_type)

        return tensor_to_np_dtype

    def _update_input_node(self, model):
        graph = model.graph
        if version.parse(onnx.version.version) < version.parse('1.6.0'):
            raise ValueError("--input_dim command not supported with ONNX versions < 1.6.0")
        input_names = list(self.input_names)
        input_dims = list(self.input_dims)
        initializers = [node.name for node in graph.initializer]
        original_inputs = {node.name : node for node in graph.input}
        new_inputs = {name: dim for name, dim in zip(input_names, input_dims)}

        # Step 1: remove original graph inputs
        for node_name in original_inputs:
            if node_name not in initializers:
                graph.input.remove(original_inputs[node_name])

        # Step 2: If input specified is part of graph inputs, update its dimensions
        for name in new_inputs:
            if name in initializers:
                raise ValueError("--input_dim command not supported with initializer " + name)
            elif name in original_inputs:
                dim = new_inputs[name]
                dims = tuple(map(int, dim.split(',')))
                input_new = onnx.helper.make_tensor_value_info(name,
                    onnx.TensorProto.FLOAT, dims)
                graph.input.insert(0, input_new)
                input_names.remove(name)
                input_dims.remove(dim)
            else:
                continue

        # Check if all inputs are accounted for, if Yes nothing more to be done. Return
        if len(input_names) == 0 and len(input_dims) == 0:
            return

        # Get the type of each model input
        input_types = {}
        for input_name in input_names:
            input_found, input_type, _ = get_type_dims_info(model.graph.input, input_name)
            input_types[input_name] = input_type if input_found else onnx.TensorProto.FLOAT

        # Step 3: If input specified is intermittent graph output,
        #         a.  Add this buffer to a list for removal later
        #         b.  Create input TensorProto with this name and dimension
        bufs_to_remove = set()
        for i, src_op in enumerate(graph.node):
            for output_buf_name in src_op.output:
                if output_buf_name in input_names:
                    position = input_names.index(output_buf_name)
                    dim = input_dims[position]
                    dims = tuple(map(int, dim.split(',')))
                    input_new = onnx.helper.make_tensor_value_info(output_buf_name, input_types[output_buf_name], dims)
                    graph.input.insert(0, input_new)
                    bufs_to_remove.add(output_buf_name)
                    input_names.remove(output_buf_name)
                    input_dims.remove(dim)

        # Check if all inputs specified are accounted for
        if len(input_names) != 0 and len(input_dims) != 0:
            invalid_names = ", ".join(input_names)
            raise ValueError("--input_dim command input name(s) not found: {}".format(invalid_names))

        # Step 4: Find all nodes to be removed from the graph. These include:
        #   a. Nodes that produce the buffers cached for removal
        #   b. All nodes that precede them in the graph
        nodes_to_remove = []
        while bufs_to_remove:
            buf_name = bufs_to_remove.pop()
            if buf_name in original_inputs or buf_name in initializers:
                # This was already removed or does not need to be handled
                continue

            # Find node that produces the buffer or is named after the buffer
            node_list = [node for node in graph.node if buf_name in node.output]
            if not node_list:
                raise KeyError("Node that produces {} not found".format(buf_name))
            elif len(node_list) != 1:
                raise KeyError("Multiple nodes {} found for output buffer {}".format(node_list, buf_name))

            node = node_list[0]
            # Add all inputs of this node as also to be removed
            bufs_to_remove.update(set(node.input))
            # Add this node to be removed if not already added
            if node not in nodes_to_remove:
                nodes_to_remove.append(node)

        # Step 5: Remove the nodes marked in Step 4
        # Check that all buffers in a slice were specified, if not Throw Error
        remaining_nodes = [node for node in graph.node if node not in nodes_to_remove]
        remaining_buffers = set()
        for remaining_node in remaining_nodes:
            remaining_buffers.update(remaining_node.input)
        for node in nodes_to_remove:
            for output in node.output:
                if output in remaining_buffers and output not in self.input_names:
                    raise ValueError("Cannot disconnect node with outputs: {} as output buffer"
                                     ": {} is still in use and was not specified as input to the Model".format
                                     (str(node.output), str(output)))
            graph.node.remove(node)

    def _update_output_nodes(self, model):

        # Determine which nodes should be retained
        nodes_to_retain = []
        queue = list(self.output_names)
        visited = set(queue)
        while queue:
            input_name = queue.pop(0)
            preceding_nodes = [node for node in model.graph.node if input_name in node.output]
            for node in preceding_nodes:
                nodes_to_retain.append(node)
                for input_name in node.input:
                    if input_name in visited:
                        continue
                    queue.append(input_name)
                    visited.add(input_name)

        # Remove nodes that are not retained
        for node in [node for node in model.graph.node if node not in nodes_to_retain]:
            model.graph.node.remove(node)

        # Get the output dimensions of the new output nodes
        new_output_value_infos = []
        for output_name in self.output_names:
            # First check the graph outputs for info on outputs
            output_found, output_type, output_dims = get_type_dims_info(model.graph.output, output_name)

            # Fallback to using optional value_info field for info on new outputs
            if not output_found and model.graph.value_info:
                output_found, output_type, output_dims = get_type_dims_info(model.graph.value_info, output_name)

            # Finally, fallback to using graph inputs for info on new outputs
            if not output_found:
                output_found, output_type, output_dims = get_type_dims_info(model.graph.input, output_name)

            if not output_found:
                raise ValueError("Could not find type/dim info for output {} specified on command line".format(
                    output_name))

            output_value_info = onnx.helper.make_tensor_value_info(output_name, output_type, output_dims)
            new_output_value_infos.append(output_value_info)

        # Remove old output nodes
        for output_node in [_ for _ in model.graph.output]:
            model.graph.output.remove(output_node)

        # Add new output info
        model.graph.output.extend(new_output_value_infos)


# ------------------------------------------------------------------------------
#   Policies
# ------------------------------------------------------------------------------
class OnnxNamePolicy(op_policies.ConversionNamePolicy):
    def __init__(self):
        op_policies.ConversionNamePolicy.__init__(self)

    def get_op_name(self, op):
        count = self.type_count.get(op.type, 0)
        self.type_count[op.type] = count + 1
        if op.name:
            return str(op.name)
        elif op.type == 'custom':
            return "%s_%s_%d" % (str(op.custom_type).lower(), op.type, count)
        else:
            return "%s_%d" % (op.type, count)


class OnnxShapeInferencePolicy(op_policies.ConversionShapeInferencePolicy):

    def infer_shape(self, op, input_shapes):
        return onnx_translations.OnnxTranslations.apply_method_to_op(op.type,
                                                                     onnx_translations.OnnxTranslationBase.INFER_SHAPE,
                                                                     op,
                                                                     input_shapes)
