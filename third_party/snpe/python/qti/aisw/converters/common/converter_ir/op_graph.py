# ==============================================================================
#
#  Copyright (c) 2018-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
import inspect
import json
import numpy as np

from qti.aisw.converters.common.converter_ir import op_adapter
from qti.aisw.converters.common.converter_ir.axis_tracker import AxisTracker
from qti.aisw.converters.common.utils.converter_utils import *
from qti.aisw.converters.common.utils import code_to_message, translation_utils


class OpNode(object):
    def __init__(self, op, input_names, output_names, axis_order=None):
        self.op = op
        self.input_names = input_names
        self.output_names = output_names
        self.axis_order = axis_order

    def __repr__(self):
        return self.op.name

    def __str__(self):
        return self.op.name

    def is_equal(self, other_node):
        if not isinstance(other_node, self.__class__):
            return False
        node_vars = dict(self.__dict__)
        other_node_vars = dict(other_node.__dict__)
        for var in list(node_vars.keys()):
            if node_vars[var] != other_node_vars[var]:
                return False, "Node for {} does not match current node. \nExpected node: {}. \nGot {}".\
                    format(other_node.op.name, node_vars.items(), other_node_vars.items())
        return True, "Node for {} matches current Node instance.".format(other_node.op.name)

    def encode(self, graph):
        data = {"name": self.op.name, "type": self.op.type, "inputs": self.input_names}
        output_dict = {}
        for name in self.output_names:
            output_dict[name] = list(graph.get_buffer(name).shape)
        data["outputs"] = output_dict
        return data


class Buffer(object):
    def __init__(self, name, shape: list, producer, axis_format=None):
        self.name = name
        self.producer = producer
        self.consumers = set()
        self.shape = shape
        if axis_format is None:
            self.axis_format = AxisTracker.AxisFormat.NOT_YET_DEFINED
        else:
            self.axis_format = axis_format

    def rank(self):
        return len(self.shape)

    def get_buf_dims(self):
        return self.shape

    def set_buf_dims(self, shape):
        self.shape = shape

    def get_axis_format(self):
        return self.axis_format

    def populate_axis_format(self, axis_order, encodings):
        # get_axis_format returns format with implicit assumption for image input when buf is 4D
        # and format for time-series style network when buf is 3D. Depending on InputEncoding provided by user
        # we need to account for non-trivial case
        input_in_encodings = [input_encoding[0] for input_encoding in encodings]
        if (InputEncodings.OTHER in input_in_encodings and (len(self.get_buf_dims()) in [3, 4])) \
                or (InputEncodings.TIME_SERIES not in input_in_encodings and len(self.get_buf_dims()) == 3):
            self.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
        else:
            self.axis_format = axis_order.get_axis_format(self.rank())

    def get_axis_annotations(self):
        """Translate AxisFormat enum to modeltools axis order list"""
        if self.axis_format == 'NSC':
            return [AxisTracker.AxisAnnotations.BATCH, AxisTracker.AxisAnnotations.HEIGHT,
                    AxisTracker.AxisAnnotations.WIDTH, AxisTracker.AxisAnnotations.CHANNEL]
        elif self.axis_format == 'NCS':
            return [AxisTracker.AxisAnnotations.BATCH, AxisTracker.AxisAnnotations.CHANNEL,
                    AxisTracker.AxisAnnotations.HEIGHT, AxisTracker.AxisAnnotations.WIDTH]
        elif self.axis_format == 'FEATURE':
            return [AxisTracker.AxisAnnotations.BATCH, AxisTracker.AxisAnnotations.FEATURE]
        elif self.axis_format == 'BTF':
            return [AxisTracker.AxisAnnotations.BATCH, AxisTracker.AxisAnnotations.TIME,
                    AxisTracker.AxisAnnotations.FEATURE]
        elif self.axis_format == 'TBF':
            return [AxisTracker.AxisAnnotations.TIME, AxisTracker.AxisAnnotations.BATCH,
                    AxisTracker.AxisAnnotations.FEATURE]
        elif self.axis_format == 'NONTRIVIAL':
            return [AxisTracker.AxisAnnotations.NONTRIVIAL]
        elif self.axis_format == 'ANY':
            return [AxisTracker.AxisAnnotations.ANY for _ in range(len(self.shape))]
        elif self.axis_format == 'HWIO':
            return [AxisTracker.AxisAnnotations.HEIGHT, AxisTracker.AxisAnnotations.WIDTH,
                    AxisTracker.AxisAnnotations.INPUT_CHANNELS, AxisTracker.AxisAnnotations.OUTPUT_CHANNELS]
        elif self.axis_format == 'OIHW':
            return [AxisTracker.AxisAnnotations.OUTPUT_CHANNELS, AxisTracker.AxisAnnotations.INPUT_CHANNELS,
                    AxisTracker.AxisAnnotations.HEIGHT, AxisTracker.AxisAnnotations.WIDTH]
        else:
            raise ValueError("Encountered unexpected axis format for get_axis_annotations: %s" % self.axis_format)

    def is_equal(self, other_buffer):
        """
        Compares another buf instance to current one based attribute matching
        :param other_buffer: a Buffer object
        :return: bool, msg. True if type and attr/params match, False otherwise. Plus message detailing what was
                            different
        """
        # attr/param list equality check
        other_buf_params = dict(other_buffer.__dict__)
        current_buf_params = dict(self.__dict__)
        for attr_ in list(current_buf_params.keys()):
            if attr_ == "producer":  # node comparison
                is_attr_eq = current_buf_params[attr_].is_equal(other_buf_params[attr_])
            elif attr_ == "consumers":
                cur_consumers = current_buf_params[attr_]
                other_consumers = other_buf_params[attr_]
                if len(cur_consumers) != len(other_consumers):
                    return False, "Attribute match error for Buffer:{}. Consumers length mismatch. Expected {} Got {}."\
                        .format(str(other_buffer.name), len(cur_consumers), len(other_consumers))
                is_attr_eq = True
                for cur_cons, other_cons in zip(cur_consumers, other_consumers):
                    is_attr_eq = cur_cons.is_equal(other_cons)
                    if not is_attr_eq:
                        break
            else:
                is_attr_eq = translation_utils.compare_values(other_buf_params[attr_], current_buf_params[attr_])
            if not is_attr_eq:
                return False, "Attribute match error for Buffer:{} Attribute: {}. Expected {}, Got {} ".format(
                    str(other_buffer.name), attr_, str(current_buf_params[attr_]), str(other_buf_params[attr_]))

        return True, "Buffer {} is equal to current Buffer instance".format(other_buffer)


class BufferCriteria(object):
    """
    Class(enum) to use for setting buffer criteria on inputs/outputs for validating matched node sequences
    """
    # to be used for individual buffers
    ALL = "ALL"  # all the buffer(s) must be this same expected op_type
    ANY = "ANY"  # There can be one or more of this op_type as buffer(s)
    NONE = "NONE"  # None of the buffer(s) should be of this type

    # to be used for set of buffers
    MATCH_NUM_BUFS = "MATCH_NUM_BUFS"  # the expected number of buffers must be same length as matched buffers
    FLEXIBLE_NUM_BUFS = "FLEXIBLE_NUM_BUFS"  # the expected number of buffers doesnt need to be equal to matched buffers
    MATCH_BUFS_AT_INDEX = "MATCH_BUFS_AT_INDEX" # matches the buffers at the specified index


class InputType(object):
    """
    Contains supported input types. This will be used by DSP to determine quantization
    """
    IMAGE = "image"  # input is float between 0-255 and the input's mean is 0.0f and the input's max is 255.0f
    DEFAULT = "default"  # pass the input as floats to the dsp directly and the DSP will quantize it
    OPAQUE = "opaque"  # assumes input is float because the consumer layer(i.e next layer) requires it as float,
    # therefore it won't be quantized by DSP

    @classmethod
    def get_supported_types(cls):
        return [cls.IMAGE, cls.DEFAULT, cls.OPAQUE]

    @classmethod
    def is_valid_type(cls, input_type):
        return input_type in cls.get_supported_types()


class InputEncodings(object):
    """
    Contains supported input encodings
    """
    BGR = "bgr"
    RGB = "rgb"
    RGBA = "rgba"
    ARGB32 = "argb32"
    NV21 = "nv21"
    NV12 = "nv12"
    TIME_SERIES = "time_series"
    OTHER = "other"

    valid_transformations = [
        (ARGB32, RGB),
        (ARGB32, BGR),
        (RGBA, RGB),
        (RGBA, BGR),
        (NV21, RGB),
        (NV21, BGR),
        (NV12, RGB),
        (NV12, BGR),
    ]

    @classmethod
    def get_supported_encodings(cls):
        return [cls.BGR, cls.RGB, cls.RGBA, cls.ARGB32, cls.NV21, cls.NV12, cls.TIME_SERIES, cls.OTHER]

    @classmethod
    def is_valid_encoding(cls, input_encoding):
        return input_encoding in cls.get_supported_encodings()

    @classmethod
    def is_valid_transformation(cls, input_encoding, out_encoding):
        return (input_encoding, out_encoding) in cls.valid_transformations


class QuantParams(object):
    """
    Contains supported quantization params
    """
    BN_PARAMS = "bn_params"
    OUTPUT_ENCODINGS = "output_encodings"
    PARAM_ENCODINGS = "param_encodings"

    @classmethod
    def get_supported_quant_params(cls):
        return [cls.BN_PARAMS, cls.OUTPUT_ENCODINGS, cls.PARAM_ENCODINGS]

    @classmethod
    def is_valid_quant_param(cls, param):
        return param in cls.get_supported_quant_params()


class IROpGraph(object):
    def __init__(self,
                 naming_policy,
                 shape_inference_policy,
                 input_types,
                 input_dtypes,
                 input_encodings,
                 src_axis_order,
                 quantization_overrides=None,
                 keep_quant_nodes=False,
                 output_nodes=[]):
        self.naming_policy = naming_policy
        self.shape_inference_policy = shape_inference_policy
        self.src_axis_order = src_axis_order
        self.inputs_type_dict = self._create_input_types_dict(input_types)
        self.input_dtypes_dict = self._create_input_dtypes_dict(input_dtypes)
        self.inputs_encoding_dict = self._create_input_encodings_dict(input_encodings)
        self.nodes_by_name = {}
        self.nodes_in_order = []
        self.buffers = {}
        self.output_names = output_nodes
        self.quantization_params = {}
        self.src_graph_op_info = {}
        self.keep_quant_nodes = keep_quant_nodes
        self.user_quantization_overrides = {}
        self.total_macs = 0
        self.total_params_count = 0

        if quantization_overrides:
            log_info('Processing user provided quantization encodings: ', quantization_overrides)
            f = open(quantization_overrides)
            self.user_quantization_overrides = json.load(f)

    def __iter__(self):
        return iter(self.nodes_in_order)

    def dump_json(self, filename):
        graph = self

        class Encoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, OpNode):
                    return obj.encode(graph)
                elif isinstance(obj, np.int32):
                    return int(obj)
                elif isinstance(obj, np.int64):
                    return int(obj)
                else:
                    try:
                        import tvm
                        if isinstance(obj, tvm.tir.expr.IntImm):
                            return int(obj)
                    except ImportError:
                        pass
                    # Let the base class default method raise the TypeError
                    return json.JSONEncoder.default(self, obj)

        if filename[-5:] != ".json":
            filename = filename + ".json"
        with open(filename, "w") as f:
            # Create a dict order by nodes_in_order instead of node_by_name to debug insert order issues
            dict_in_order = {n.op.name: n for n in self.nodes_in_order}
            json.dump(dict_in_order, f, cls=Encoder,
                      indent=4, separators=(',', ': '))

    @staticmethod
    def _create_input_types_dict(input_types):
        log_assert(all(InputType.is_valid_type(type_) for _, type_ in input_types),
                   code_to_message.get_error_message("ERROR_UNSUPPORTED_INPUT_TYPE")(InputType.get_supported_types()))
        return {input_name: input_type for input_name, input_type in input_types}

    @staticmethod
    def _create_input_dtypes_dict(input_dtypes):
        return {input_name: np.dtype(input_dtype) for input_name, input_dtype in input_dtypes}

    @staticmethod
    def _create_input_encodings_dict(input_encodings):

        # Populate input encodings out with default encoding of BGR
        for i, input_encoding in enumerate(input_encodings):
            if len(input_encoding) == 2:
                input_encodings[i].append(InputEncodings.BGR)

        log_assert(all(InputEncodings.is_valid_encoding(encoding_in) and InputEncodings.is_valid_encoding(encoding_out)
                       for _, encoding_in, encoding_out in input_encodings),
                   code_to_message.get_error_message("ERROR_UNSUPPORTED_INPUT_ENCODING")
                   (input_encodings, InputEncodings.get_supported_encodings()))
        # Return dictionary of the form - input_name: (input_encoding_in, input_encoding_out)
        return {input_encoding[0]: tuple(input_encoding[1:]) for input_encoding in input_encodings}

    def get_input_type(self, input_name):
        # use input_type: default as the default for all inputs
        return self.inputs_type_dict.get(input_name, InputType.DEFAULT)

    def get_input_dtype(self, input_name):
        return self.input_dtypes_dict.get(input_name, np.dtype("float32"))

    def get_input_types(self):
        return self.inputs_type_dict.values()

    def get_input_encoding(self, input_name):
        # use input_encoding: bgr as the default for all inputs
        # tensor name can follow (colon):num which is stripped to check for command-line name
        input_name_ = input_name
        if ":" in input_name and input_name not in self.inputs_type_dict:
            input_name_ = input_name[0: input_name.index(':')]
        return self.inputs_encoding_dict.get(input_name_, (InputEncodings.BGR, InputEncodings.BGR))

    def get_input_encodings(self):
        return self.inputs_encoding_dict.values()

    def add_src_op_info(self, op_name, inputs: list = None, outputs: list = None):
        """
        Adds a mapping of the original op's inputs and outputs to the original op name

        :param op_name: The name of the op whose inputs and outputs were added
        :param inputs: The inputs in the original source framework for the given op (params)
        :param outputs: The outputs in the original source framework for the given op
        """
        if inputs is None:
            inputs = []
        if outputs is None:
            outputs = []
        self.src_graph_op_info[op_name] = {'inputs': inputs,
                                           'outputs': outputs}

    def add_quantization_params(self, op_name, **kwargs):
        """
        Adds quantization params to an IR graph object for a given op_name. The dictionary provided
        is expected to contain one/all of output_encodings, param_encodings or bn_params as a key(s).

        :param op_name: The name of the op whose quantization params will be added.
        :param kwargs: The dictionary containing the output encodings, param encodings and bn_params for that op
        :raises: An assertion error if the quantization params are not valid.
        """
        log_assert(all(QuantParams.is_valid_quant_param(param) for param, _ in kwargs.items()),
                   code_to_message.get_error_message("ERROR_UNSUPPORTED_QUANT_PARAM")
                   (QuantParams.get_supported_quant_params()))

        bn_params = {}
        output_encodings = []
        param_encodings = []
        if op_name in self.quantization_params:
            # update defaults to existing params if we are updating a layer
            bn_params = self.quantization_params[op_name][QuantParams.BN_PARAMS]
            output_encodings = self.quantization_params[op_name][QuantParams.OUTPUT_ENCODINGS]
            param_encodings = self.quantization_params[op_name][QuantParams.PARAM_ENCODINGS]

        new_output_encs = translation_utils.to_list(kwargs.get(QuantParams.OUTPUT_ENCODINGS, []))
        new_param_encs = translation_utils.to_list(kwargs.get(QuantParams.PARAM_ENCODINGS, []))

        # Replace exist encodings if found, or Extend existing lists(if exist) for encodings
        for o in new_output_encs:
            found_encoding = False
            for i,current_enc in enumerate(output_encodings):
                if o['name'] == current_enc['name']:
                    output_encodings[i] = o
                    found_encoding = True
            if not found_encoding:
                output_encodings.extend(translation_utils.to_list(o))

        for p in new_param_encs:
            found_encoding = False
            for i,current_enc in enumerate(param_encodings):
                if p['name'] == current_enc['name']:
                    param_encodings[i] = p
                    found_encoding = True
            if not found_encoding:
                param_encodings.extend(translation_utils.to_list(p))

        self.quantization_params.update({
            op_name: {
                QuantParams.BN_PARAMS: kwargs.get(QuantParams.BN_PARAMS, bn_params),
                QuantParams.OUTPUT_ENCODINGS: output_encodings,
                QuantParams.PARAM_ENCODINGS: param_encodings
            }
        })

    def has_quantization_param(self, layer_name):
        return layer_name in self.quantization_params

    def remove_quantization_params(self, layer_name):
        if self.has_quantization_param(layer_name):
            del self.quantization_params[layer_name]

    def get_layer_quantization_param(self, layer_name):
        log_assert(layer_name in self.quantization_params,
                   code_to_message.get_error_message("ERROR_LAYER_NOT_FOUND_IN_QUANT_PARAM")(layer_name))
        return self.quantization_params[layer_name]

    def merge_quantization_params(self, source_op_name, destination_op_name, pre_merge_dest_tensor_name,
                                  post_merge_dest_tensor_name, encoding_type=QuantParams.OUTPUT_ENCODINGS):
        """
        Merges the output encodings for source op to destination
        :param source_op_name: the layer/op name for the source Op
        :param destination_op_name: the layer/op name for the destination Op
        :param pre_merge_dest_tensor_name: the output tensor name for the destination op before merging source op
        :param post_merge_dest_tensor_name: the output tensor name for the destination op after merging source op
        :param encoding_type: The type of encoding to update
        """

        if source_op_name in self.quantization_params:
            # verify this is proper merging. i.e. source op with more than one output/param is not allowed
            if len(self.quantization_params[source_op_name][encoding_type]) != 1:
                log_debug2('Can only merge 1 encoding for src op: {} w/tensor: {}, but found {}'.format(
                       source_op_name, pre_merge_dest_tensor_name, len(self.quantization_params[source_op_name][encoding_type])))
                return

            source_encodings = self.quantization_params[source_op_name][encoding_type]

            overridden_encodings = []
            dest_encodings = { QuantParams.BN_PARAMS: {}, QuantParams.PARAM_ENCODINGS: [], QuantParams.OUTPUT_ENCODINGS: [] }
            if destination_op_name in self.quantization_params:
                overridden_encoding = self.quantization_params[destination_op_name][encoding_type]
                dest_encodings = {
                    QuantParams.BN_PARAMS : self.quantization_params[destination_op_name][QuantParams.BN_PARAMS],
                    QuantParams.PARAM_ENCODINGS : self.quantization_params[destination_op_name][QuantParams.PARAM_ENCODINGS],
                    QuantParams.OUTPUT_ENCODINGS : self.quantization_params[destination_op_name][QuantParams.OUTPUT_ENCODINGS]
                }
                # remove the entry for the destination encoding as that will be replaced with the source op's encoding
                for i, encodings in enumerate(overridden_encodings):
                    if pre_merge_dest_tensor_name == encodings["name"]:
                        del self.quantization_params[destination_op_name][encoding_type][i]

            # Note: only need output encoding from source since the weights/bias will be merged into destination op
            source_encodings[0]["name"] = post_merge_dest_tensor_name  # replace output tensor name
            overridden_encodings.extend(source_encodings)
            dest_encodings[encoding_type] = overridden_encodings
            self.quantization_params.update({ destination_op_name: dest_encodings })

            # remove quantization entry for source op as the op will be merged
            del self.quantization_params[source_op_name]

    def eval_macs_params(self):
        """ Evaluates macs and params count for each Op in graph and adds to total"""
        self.reset_macs_params()
        for node in self.list_nodes():
            input_shapes = self.get_input_shapes(node)
            output_shapes = self.get_output_shapes(node)
            node.op.set_macs_params(input_shapes, output_shapes, self.src_axis_order)
            self.total_macs += node.op.macs
            self.total_params_count += node.op.params_count

    def reeval_macs_params(self):
        """ Re Calculates total macs and params count for graph"""
        self.reset_macs_params()
        for node in self.list_nodes():
            self.total_macs += node.op.macs
            self.total_params_count += node.op.params_count

    def reset_macs_params(self):
        """ Resets total macs and params count for graph"""
        self.total_macs = 0
        self.total_params_count = 0

    def __insert_node(self, node, output_shapes, axis_formats=None, idx=-1):
        """Insert a node into the graph's internal data structures.

        node: Node to be inserted
        output_shapes: shapes of the node's output buffers, which must be created.
        axis_formats: List of axis_format to override for each output Buffer
        idx: index in nodes_in_order at which to insert. By default, appends to
             the list.
        """
        for i, (name, shape) in enumerate(zip(node.output_names, output_shapes)):
            if self.has_buffer(name):
                raise ValueError("Duplicate buffer name, {} already exists".format(name))

            if axis_formats is None:
                buf = Buffer(name, shape, node)
                node.op.populate_axis_format(buf, self.src_axis_order, self.get_input_encodings())
            else:
                buf = Buffer(name, shape, node, axis_formats[i])
            self.buffers[name] = buf
            log_debug1("Added buffer named {0} of shape {1}", name, shape)

        for name in node.input_names:
            self.buffers[name].consumers.add(node)

        if node in self.nodes_in_order:
            raise IndexError("Node by name {} already exists at index {}".format(
                node.op.name, self.nodes_in_order.index(node)))
        if node.op.name in self.nodes_by_name:
            raise ValueError("Duplicate node name, {} already exists".format(node.op.name))

        self.nodes_by_name[node.op.name] = node
        if idx == -1:
            self.nodes_in_order.append(node)
        else:
            self.nodes_in_order.insert(idx, node)

    def add(self, op, input_names, output_names, axis_formats=None, idx=-1):
        """
        Adds op to graph by creating a node and corresponding buffer, as well as update
        input and output buffers for node.
        :param op: an operation from op_adapter class
        :param input_names: inputs to node. (This will be the buffer input names)
        :param output_names: output buffer names of node
        :param axis_formats: axis format of output buffers
        :param idx: index in nodes_in_order at which to insert. By default, appends to the list.
        :return: The created node for op.
        """
        op.name = self.naming_policy.get_op_name(op)

        if not isinstance(input_names, list):
            input_names = [input_names]
        input_names = self.naming_policy.get_input_names(op, input_names)

        input_shapes = []
        for name in input_names:
            if name not in self.buffers:
                raise KeyError("Graph has no buffer %s, referred to as input for %s" % (name, op.name))
            input_shapes.append(self.buffers[name].shape)

        if not isinstance(output_names, list):
            output_names = [output_names]
        output_names = self.naming_policy.get_output_names(op, output_names)

        # TODO: Move output_shapes, input_shapes inside OpNode.
        node = OpNode(op, input_names, output_names, self.src_axis_order)
        log_debug1("\n")
        log_debug1("Added OpNode with name {0}, in_names {1}, out_names {2}".format(op.name, input_names, output_names))
        try:
            input_buffers = [self.get_buffer(in_name) for in_name in input_names]
            input_axis_formats = [buf.axis_format for buf in input_buffers]
            output_shapes = op.infer_shape(input_shapes, input_axis_formats, len(output_names), self.src_axis_order)
        except NotImplementedError as e:
            if self.shape_inference_policy:
                try:
                    output_shapes = self.shape_inference_policy.infer_shape(op, input_shapes)
                except KeyError as e:
                    log_error("Node %s: %s", op.name, e)
                    raise e
            else:
                log_error("Node %s: %s", op.name, e)
                raise e

        if len(output_shapes) != len(output_names):
            raise ValueError("Op %s: produced %d output shapes, but have %d outputs" % (op.name, len(output_shapes),
                                                                                        len(output_names)))

        # at this point everything should be error free, so it's fine to actually
        # touch the data structures
        self.__insert_node(node, output_shapes, axis_formats, idx=idx)

        op.update_param_quant_overrides(self, node)

        # return the added node
        return node

    def remove_node_as_consumer(self, node, buffer_name):
        """
        Removes node as consumer from buffer with name buffer_name. Removes all instances of buffer_name from node's
        input_name list
        :param node: the node that should be removed as a consumer of the buffer specified by buffer_name
        :param buffer_name: the name of the buffer that this node is no longer a consumer of
        """
        if not isinstance(node, OpNode):
            raise TypeError("Passed node is not an instance of class OpNode. Passed type: {}".format(type(node)))
        buffer = self.get_buffer(buffer_name)
        if node in buffer.consumers:
            buffer.consumers.remove(node)
        # Note: Removes all instances of buffer_name from node's input_names
        while buffer_name in node.input_names:
            node.input_names.remove(buffer_name)
        # Prune the producing node if node was the only consumer of buffer buffer_name
        if not len(buffer.consumers):
            self.prune(buffer.producer)

    def replace(self, old_op, new_op):
        old_node = self.nodes_by_name[old_op.name]
        input_buffers = self.get_input_buffers(old_node)
        output_buffers = self.get_output_buffers(old_node)
        input_names = [buf.name for buf in input_buffers]
        output_names = [buf.name for buf in output_buffers]

        # Create OpNode for the new op
        new_op.name = self.naming_policy.get_op_name(new_op)
        new_node = OpNode(new_op, input_names, output_names, self.src_axis_order)

        # Replace the op in buffers
        # loop through as set to support scenarios where a node is listed as input more than once
        input_shapes = []
        for buf in uniques(input_buffers):
            buf.consumers.remove(old_node)
            buf.consumers.add(new_node)
            input_shapes.append(buf.shape)

        try:
            input_axis_formats = [buf.axis_format for buf in input_buffers]
            output_shapes = new_op.infer_shape(input_shapes, input_axis_formats, len(output_names), self.src_axis_order)
        except NotImplementedError as e:
            if self.shape_inference_policy:
                try:
                    output_shapes = self.shape_inference_policy.infer_shape(new_op, input_shapes)
                except KeyError as e:
                    log_error("Node %s: %s", new_op.name, e)
                    raise e
            else:
                log_error("Node %s: %s", new_op.name, e)
                raise e

        for i, buf in enumerate(output_buffers):
            buf.producer = new_node
            buf.shape = output_shapes[i]
            new_op.populate_axis_format(buf, self.src_axis_order, self.get_input_encodings())

        # Replace the op in op-lists
        idx = self.nodes_in_order.index(old_node)
        self.nodes_by_name[new_op.name] = new_node
        if idx == -1:
            self.nodes_in_order.append(new_node)
        else:
            self.nodes_in_order.insert(idx, new_node)

        del self.nodes_by_name[old_node.op.name]
        self.nodes_in_order.remove(old_node)

    def add_input(self, name, shape, axis_format=None, input_type=None, input_dtype=None):
        if not input_type:
            input_type = self.get_input_type(name)

        if not input_dtype:
            input_dtype = self.get_input_dtype(name)

        input_encoding_in, input_encoding_out = self.get_input_encoding(name)

        if input_encoding_in == InputEncodings.TIME_SERIES:
            log_assert(len(shape) == 3,
                       code_to_message.get_error_message("ERROR_TIMESERIES_UNEXPECTED_RANK")
                       (name, len(shape)))
        if input_encoding_in == InputEncodings.OTHER:
            axis_format = AxisTracker.AxisFormat.NONTRIVIAL

        # validate color transformation for image based encodings
        if input_encoding_in != input_encoding_out and \
                input_encoding_in not in [InputEncodings.TIME_SERIES, InputEncodings.OTHER]:

            log_assert(len(shape) == 4 and
                       self.src_axis_order.extract_spatial_dims(shape)[-1] == 3,
                       code_to_message.get_error_message("ERROR_INVALID_COLOR_TRANSFORM_INPUT_SHAPE")
                       (shape, name, "[b,h,w,3]"))

            log_assert(InputEncodings.is_valid_transformation(input_encoding_in, input_encoding_out),
                       code_to_message.get_error_message("ERROR_UNSUPPORTED_COLOR_TRANSFORMATION")
                       (input_encoding_in, input_encoding_out))

        op = op_adapter.InputOp(name, shape,
                                input_encoding_in=input_encoding_in,
                                input_encoding_out=input_encoding_out,
                                input_type=input_type,
                                input_dtype=input_dtype)
        output_names = self.naming_policy.get_output_names(op, [name])

        node = OpNode(op, [], output_names, self.src_axis_order)
        axis_formats = None if axis_format is None else [axis_format]
        self.__insert_node(node, [shape], axis_formats)

        # return the added input node
        return node

    def inject(self, op, input_name, output_name, consumer_names=None, axis_format=None):
        op.name = self.naming_policy.get_op_name(op)
        if input_name not in self.buffers:
            raise KeyError("Cannot inject op %s onto nonexistent buffer %s" % (op.name, input_name))

        input_buffer = self.buffers[input_name]
        if consumer_names is None:
            old_consumers = list(input_buffer.consumers)
            input_buffer.consumers.clear()
        else:
            old_consumers = []
            for name in consumer_names:
                if name not in self.nodes_by_name:
                    raise KeyError("Cannot inject op %s with nonexistent consumer %s" % (op.name, name))
                consumer = self.nodes_by_name[name]
                if consumer not in input_buffer.consumers:
                    raise KeyError("Cannot inject op %s, specified consumer %s does not actually consume input"
                                   " buffer %s" % (op.name, name, input_name))

                old_consumers.append(consumer)
                input_buffer.consumers.remove(consumer)

        output_name = self.naming_policy.get_output_names(op, [output_name])[0]
        producer_idx = self.nodes_in_order.index(input_buffer.producer)
        try:
            output_shapes = op.infer_shape([input_buffer.shape], [input_buffer.axis_format], 1, self.src_axis_order)
        except NotImplementedError as e:
            if self.shape_inference_policy:
                try:
                    output_shapes = self.shape_inference_policy.infer_shape(op, [input_buffer.shape])
                except KeyError as e:
                    log_error("Node %s: %s", op.name, e)
                    raise e
            else:
                log_error("Node %s: %s", op.name, e)
                raise e
        node = OpNode(op, [input_name], [output_name], self.src_axis_order)
        axis_formats = None if axis_format is None else [axis_format]
        self.__insert_node(node, output_shapes, axis_formats, producer_idx+1)

        output_buffer = self.buffers[output_name]
        for consumer in old_consumers:
            output_buffer.consumers.add(consumer)
            for i, name in enumerate(consumer.input_names):
                if name == input_name:
                    consumer.input_names[i] = output_name
        return node

    @staticmethod
    def get_implicit_permute_node_name(input_name, op_name, target_format):
        return str(input_name + '_' + op_name + '.' + target_format.lower())

    def inject_implicit_permute(self, input_name, op_name, target_format, permute_order, consumers=None):
        permute_name = self.get_implicit_permute_node_name(input_name, op_name, target_format)
        input_buf = self.get_buffer(input_name)
        log_assert(input_buf.rank() == len(permute_order),
                   "Error: length of buf to permute({}) does not match length of permute order({})"
                   " for input name: {}",
                   input_buf.rank(), len(permute_order), input_name)
        implicit_permute = op_adapter.PermuteOp(permute_name, permute_order)
        # since the implicit permute won't be visited in this pass, go
        # ahead and set the correct order for its buffer here.
        self.inject(implicit_permute, input_name, permute_name, consumers, axis_format=target_format)

    def prune(self, node, force_remove=False):
        """Remove a node and its output buffers from the graph completely.
        Will raise an exception if force_remove is False and the node has any successors."""

        # Disconnect output nodes
        output_buffers = self.get_output_buffers(node)
        consumers = []
        for buf in output_buffers:
            consumers.extend(buf.consumers)

        if len(consumers) > 0:
            if force_remove:
                for buf in output_buffers:
                    for c in buf.consumers:
                        try:
                            c.input_names.remove(buf.name)
                        except Exception as e:
                            log_error("Buffer {} not found in consumers for node {}".format(buf.name,
                                                                                            c.op.name))
                            raise e
            else:
                consumer_name_list = [c.op.name for c in consumers]
                raise RuntimeError("Cannot prune node %s, which has the following successors: %s"
                                   % (node.op.name, consumer_name_list))

        for buf in output_buffers:
            del self.buffers[buf.name]
            self.naming_policy.remove_output_name(buf.name)

        # Disconnect input nodes
        # loop through as set to support scenarios where a node is listed as input more than once
        for buf in set(self.get_input_buffers(node)):
            buf.consumers.remove(node)
        del self.nodes_by_name[node.op.name]
        self.nodes_in_order.remove(node)

    def squash_noop(self, node: OpNode):
        """
        Squashes Noop node into the input buffer. Basically change the output name of the previous op to be
        the output of current opnode. Since we are only changing output name, there is no need to check for number
        of consumers or other restrictions
        :param node: OpNode of type NoOp, Identity
        :return: throws Error if node is not of type Noop, Identity, otherwise return nothing
        """
        if not node.op.type in [op_adapter.NoopOp.TRANSLATION_KEY,
                                op_adapter.IdentityOp.TRANSLATION_KEY]:
            raise TypeError("squash_noop should be called only with Noop or Identity Op")
        if len(node.input_names) != 1:
            raise ValueError("Op {} expected to have only 1 input, has {}".format(node.op.type, len(node.input_names)))
        if len(node.output_names) != 1:
            raise ValueError("Op {} expected to have only 1 output, has {}".format(node.op.type, len(node.output_names)))
        input_name = node.input_names[0]
        output_name = node.output_names[0]

        # Change the input names of input buffer's consumers to consume output_name instead.
        # (ideally consumers other than current node,
        # but its ok to change here because the output name was cached above)
        #
        # No need to change the input_buffer's consumers set since the buffer will be deleted anyway
        input_buffer = self.get_buffer(input_name)
        for consumer in input_buffer.consumers:
            input_idx = consumer.input_names.index(input_name)
            consumer.input_names[input_idx] = output_name

        # Change the output name of prev_node
        prev_node = input_buffer.producer
        output_idx = prev_node.output_names.index(input_name)
        prev_node.output_names[output_idx] = output_name

        # Change the producer of output buffer to prev_node
        output_buffer = self.get_buffer(output_name)
        output_buffer.producer = prev_node

        # Delete dangling buffer and nodes
        del self.buffers[input_name]
        del self.nodes_by_name[node.op.name]
        self.nodes_in_order.remove(node)

    def squash(self, node: OpNode, input_name: str, *, squash_into_next: bool = False, no_op = False) -> bool:
        """
        Squashes OpNode node into the buffer specified by buffer_name. OpNode will no longer exist. If squash_into_next
        is True, then input_name is the preserved buffer, else output_name is the preserved buffer.
        :param node: the node to be squashed
        :param input_name: the name of the input buffer coming from the previous node
        :param squash_into_next: True specifies a squash into next should happen, else squash into previous. Will
        attempt the alternative option if the function detects the specified strategy is not possible
        :param no_op The merge is a no-op, eg the merge doesn't affect the output of the op it's being merged into
        :return returns False on failure, else True
        """

        def get_buffer(op_node: OpNode, buffer_name: str, is_input: bool) -> Buffer:
            node_name = op_node.op.name
            if is_input:
                if buffer_name not in op_node.input_names:
                    raise ValueError("Buffer {} is not a valid input of node {}.".format(buffer_name, node_name))
            else:
                if buffer_name not in op_node.output_names:
                    raise ValueError("Buffer {} is not a valid output of node {}.".format(buffer_name, node_name))

            buffer = self.buffers.get(buffer_name)
            if buffer is None:
                raise KeyError("Buffer {} does not exist, cannot squash {} node {}.".format(input_name, op_node.op.type,
                                                                                            node_name))
            return buffer

        if len(node.output_names) != 1:
            raise ValueError("Node {} must have exactly one output to be squashed. Got {}".format(
                node.op.name, len(node.output_names)))

        output_name = node.output_names[0]
        input_buffer = get_buffer(node, input_name, is_input=True)
        output_buffer = get_buffer(node, output_name, is_input=False)

        previous_node = input_buffer.producer
        next_op_nodes = output_buffer.consumers

        if not squash_into_next:
            # Squash into previous not possible
            if previous_node.op.type == op_adapter.InputOp.TRANSLATION_KEY or len(input_buffer.consumers) != 1:
                log_warning("Cannot squash node {} into previous. Will attempt squash into next.".format(
                    node.op.name))
                squash_into_next = True
                # Squash into next is not possible
                if not len(next_op_nodes):
                    log_warning("Squashing node {} is not possible.".format(node.op.name))
                    return False
        else:
            # Squash into next is not possible
            if not len(next_op_nodes):
                log_warning("Cannot squash node {} into next. Will attempt squash into previous.".format(
                    node.op.name))
                squash_into_next = False
                # Squash into previous is not possible
                if previous_node.op.type == op_adapter.InputOp.TRANSLATION_KEY or len(input_buffer.consumers) != 1:
                    log_warning("Squashing node {} is not possible.".format(node.op.name))
                    return False

        if not squash_into_next:

            # If we aren't performing a squash_into_next, perform a squash into previous
            output_idx = previous_node.output_names.index(input_name)
            previous_node.output_names[output_idx] = output_name
            output_buffer.producer = previous_node
            if not no_op:
              self.merge_quantization_params(node.op.name, previous_node.op.name, input_name, output_name)

            # Generate other_input_names as set in case input names are listed twice
            other_input_names = set([in_name for in_name in node.input_names if in_name != input_name])
            for in_buf_name in other_input_names:
                in_buf = self.get_buffer(in_buf_name)
                in_producer = self.get_producer_node(in_buf_name)
                # Simply prune if producing node has exactly one consumer, the node we are squashing
                if len(in_buf.consumers) == 1 and len(in_producer.output_names) == 1:
                    self.prune(in_producer, True)
                else:
                    # Remove node to be squashed from input buffer consumers
                    in_buf.consumers.remove(node)
                    # If this input buffer has no more consumers after removing producing node, then remove it
                    if not len(in_buf.consumers):
                        in_producer.output_names.remove(in_buf_name)
                        del self.buffers[in_buf_name]
                        self.naming_policy.remove_output_name(in_buf_name)

            # Remove the input buffer from the graph
            del self.buffers[input_name]
            self.naming_policy.remove_output_name(input_name)
        else:
            for next_op_node in next_op_nodes:
                input_idx = next_op_node.input_names.index(output_name)
                next_op_node.input_names[input_idx] = input_name
                input_buffer.consumers.add(next_op_node)
            del self.buffers[output_name]
            self.naming_policy.remove_output_name(output_name)
            self.remove_quantization_params(node.op.name)
            input_buffer.consumers.remove(node)

        # Remove the squashed node from the graph
        del self.nodes_by_name[node.op.name]
        self.nodes_in_order.remove(node)
        return True

    def get_matched_nodes(self, sequence, validator=None, ignore_constants=False):
        """
        Traverses each node in graph to find the requested pattern
        :param sequence: list[tuples] a list of node translation keys with their inputs and outputs. i.e:
                         each tuple contains ("opdapter.<op_name>.TRANSLATION_KEY", ([inputs]), ([outputs]))
                         The tuple for inputs/outputs should state BufferCriteria to verify list length; additionally,
                         each input/output should state specific BufferCriteria to determine how many(if any) of the
                         buffer should be in the matched sequence.
             E.g for format:
             sequence = [
                   # node type A
                   (op_adapter.<op_name>.TRANSLATION_KEY,
                       # inputs
                       (BufferCriteria.<criteria>, [(op_adapter.<op_name>.TRANSLATION_KEY, BufferCriteria.<criteria>)
                                                    (op_adapter.<op_name>.TRANSLATION_KEY, BufferCriteria.<criteria>)
                                                    ...]),
                       # outputs
                       (BufferCriteria.<criteria>, [(op_adapter.<op_name>.TRANSLATION_KEY, BufferCriteria.<criteria>)
                                                    (op_adapter.<op_name>.TRANSLATION_KEY, BufferCriteria.<criteria>)
                                                    ...])
                   ),
                   # node type B
                   (op_adapter.<op_name>.TRANSLATION_KEY,
                       # inputs
                       (),
                       # outputs
                       ()
                   ),
                   ...
             ]
             E.g (Channel Shuffle). Note: we can pass strings instead of class.xxx for convenience,
                                          this function handles both.
             sequence = [
                        ("reshape",
                            (),
                            ("MATCH_NUM_BUFS", [("permute", "ALL")])
                        ),
                        ("permute",
                            (),
                            ("MATCH_NUM_BUFS", [("reshape", "ALL")])
                        ),
                        ("reshape",
                            (),
                            ()
                        )
                       ]
             Note 1: both inputs and outputs should also be translation keys
             Note 2: BufferCriteria can either be one of the BufferCriteria Enums or an INT to match a specific index
             Note 3: it is not required to have inputs or outputs, they can be left empty.
        :param validator: function to run if a match is found based on sequence. The matched sequence will be passed as
                          {"node_tuples": (nodes_matched)}
                          If not provided, function will return based on only matching the sequence as criteria.
        :param ignore_constants: if constant nodes need to be filtered during matching, this flag will be set to True.
        :return: list of node tuples that match the sequence provided, where each tuple contains the corresponding nodes
                 for each TRANSLATION_KEY in the sequence.
        """

        matched_nodes = []
        requested_types_seq = [entry[0].lower() for entry in sequence]
        start = 0
        end = len(sequence)
        nodes_list = self.list_nodes()

        if ignore_constants:
            nodes_list = [node for node in nodes_list if node.op.type != 'constant']

        log_debug2("Evaluating to match Sequence {}...", requested_types_seq)

        # we want to allow use of strings for op translation_keys(i.e op_types) to make sequence length minimal
        # so validate user has asked to match op_types that are supported in op_adapter
        log_assert(self.verify_op_types_exist(requested_types_seq) is True,
                   code_to_message.get_error_message("ERROR_UNKNOWN_OP_TYPE(S)_FOUND")(requested_types_seq))

        while end <= len(nodes_list):
            nodes_tuple = tuple(nodes_list[start:end])  # get number of nodes based on length of sequence
            current_types_seq = [node.op.type for node in nodes_tuple]
            if (current_types_seq == requested_types_seq and self._validate_nodes_topology(nodes_tuple, sequence)) and \
                    (validator is None or validator(nodes_tuple)):
                matched_nodes.append(nodes_tuple)
                start = end  # start next node by skipping over the length of the sequence matched
                end += len(sequence)
            else:
                start += 1
                end = start + len(sequence)

        log_debug2("Found {} match(es)", len(matched_nodes))

        return matched_nodes

    def _validate_nodes_topology(self, nodes_tuple, sequence):
        """
        validates the input and output buffers for each matched node sequence in graph

        :param nodes_tuple: a tuple of matched nodes based on pattern
        :param sequence: the original list of sequences provided by user
        :return: True if each node's input and output buffer match the expected ones in sequence, False otherwise
        :raises: AssertionError if length and node types of node_list and sequence do not match
        """

        log_assert(len(nodes_tuple) == len(sequence), "Matched node list length must be same as requested sequence. "
                                                      "Expected {}, Got {}", len(nodes_tuple), len(sequence))

        for i in range(0, len(nodes_tuple)):
            node_type_actual = nodes_tuple[i].op.type
            node_type_expected = sequence[i][0]
            log_assert(node_type_actual == node_type_expected,
                       "Cannot validate topology for nodes of different types. Expected {}, Got{}",
                       node_type_expected, node_type_actual)

            inputs_actual = self.get_input_op_types(nodes_tuple[i])
            outputs_actual = self.get_output_op_types(nodes_tuple[i])
            inputs_expected, outputs_expected = sequence[i][1:]

            # providing inputs_expected and outputs_expected is not required from user
            # since user might just care to match a sequence of node types for any given inputs/outputs
            if (len(inputs_expected) and not self._validate_buffers(inputs_expected, inputs_actual)) or \
               (len(outputs_expected) and not self._validate_buffers(outputs_expected, outputs_actual)):
                log_debug2("Sequence pattern {} matched, but not input/output buffers for node {} of type {} in "
                           "sequence.", [entry[0] for entry in sequence], nodes_tuple[i].op.name,
                           nodes_tuple[i].op.type)
                return False

        return True

    def _validate_buffers(self, expected_buffers, actual_buffers):
        """
        validates the actual buffers(inputs or outputs of nodes) against the criteria set in the expected buffers
        :param expected_buffers: a tuple with BufferCriteria for matching the list of buffers, list of tuple pairs
                                 with each tuple containing the type of op and a buffer criteria
                        (BufferCriteria.<criteria>, [(op_adapter.<op_name>.TRANSLATION_KEY, BufferCriteria.<criteria>)
                                                    (op_adapter.<op_name>.TRANSLATION_KEY, BufferCriteria.<criteria>)
                                                    ...])
        :param actual_buffers: list of actual buffer types for the current node being evaluated
        :return: true if actual buffers pass criteria set in the expected buffers, False otherwise

        raises Assertion error: if unknown buffer criteria,
               Value error: if ALL criteria given and there exists more expected inputs
        """

        # remove matching criteria from expected buffers and validate
        matching_criteria, expected_buffers = expected_buffers
        matching_criteria = matching_criteria.upper()
        log_assert(matching_criteria in [BufferCriteria.MATCH_NUM_BUFS, BufferCriteria.FLEXIBLE_NUM_BUFS, BufferCriteria.MATCH_BUFS_AT_INDEX],
                   code_to_message.get_error_message("ERROR_UNKNOWN_MATCHING_CRITERIA")
                   ([BufferCriteria.MATCH_NUM_BUFS, BufferCriteria.FLEXIBLE_NUM_BUFS, BufferCriteria.MATCH_BUFS_AT_INDEX], matching_criteria))

        if matching_criteria == BufferCriteria.MATCH_NUM_BUFS and len(expected_buffers) != len(actual_buffers):
            return False

        for op_type, buf_criteria in expected_buffers:
            op_type = op_type.lower()
            log_assert(self.verify_op_types_exist(op_type) is True,
                       code_to_message.get_error_message("ERROR_UNKNOWN_OP_TYPE(S)_FOUND")(op_type))

            if type(buf_criteria) == int:
                if matching_criteria == BufferCriteria.MATCH_NUM_BUFS:
                    # User knows the number of input/output buffers to expect, hence it is an error to request
                    # an out-of-range index
                    log_assert(buf_criteria < len(actual_buffers),
                               code_to_message.get_error_message("ERROR_BUFFER_CRITERIA_INDEX")
                               (op_type, buf_criteria, len(actual_buffers)))

                    if actual_buffers[buf_criteria] != op_type:
                        return False

                elif matching_criteria == BufferCriteria.MATCH_BUFS_AT_INDEX:
                    # In this case, user doesnt know/care for the number of input/output buffers of a node but want to
                    # match ops that fit a certain criteria e.g. when the 2nd input is a particular op type;
                    # in this instance an out-of-range index is not an error.

                    if buf_criteria >= len(actual_buffers) or actual_buffers[buf_criteria] != op_type:
                        return False
                elif matching_criteria == BufferCriteria.FLEXIBLE_NUM_BUFS:
                    # In this case, user knows exactly how many of this type to expect but does not care
                    # about the position in the inputs
                    op_type_count = len([actual_op_type for actual_op_type in actual_buffers
                                         if actual_op_type == op_type])
                    if op_type_count != buf_criteria:
                        return False
            elif buf_criteria.upper() == BufferCriteria.ALL:
                if len(expected_buffers) != 1:
                    raise ValueError(code_to_message.get_error_message("ERROR_BUFFER_CRITERIA_ALL")
                                     (op_type, len(expected_buffers)))
                if not all(buf == op_type for buf in actual_buffers):
                    return False

            elif buf_criteria.upper() == BufferCriteria.ANY:
                if not any(buf == op_type for buf in actual_buffers):
                    return False

            elif buf_criteria.upper() == BufferCriteria.NONE:
                if any(buf == op_type for buf in actual_buffers):
                    return False

            # Unknown buffer criteria, so raise error
            else:
                raise ValueError(code_to_message.get_error_message("ERROR_UNKNOWN_BUFFER_CRITERIA")
                                 (op_type, ["ALL", "ANY", "NONE"], buf_criteria))

        return True

    @staticmethod
    def verify_op_types_exist(op_list):
        if type(op_list) is not list:
            op_list = [op_list]
        # get all supported op_types in op_adapter module
        supported_op_list = [class_[1].TRANSLATION_KEY if hasattr(class_[1], 'TRANSLATION_KEY') else ''
                             for class_ in inspect.getmembers(op_adapter, inspect.isclass)]
        return all(op in supported_op_list for op in op_list)

    def get_input_buffers(self, node):
        return [self.buffers[name] for name in node.input_names]

    def get_output_buffers(self, node):
        return [self.buffers[name] for name in node.output_names]

    def get_input_shapes(self, node):
        return [self.buffers[name].shape for name in node.input_names]

    def get_output_shapes(self, node):
        return [self.buffers[name].shape for name in node.output_names]

    def add_output_buffer(self, node, name: str, shape: list, axis_format=None):
        buf = Buffer(name, shape, node, axis_format=axis_format)
        self.buffers[name] = buf
        node.output_names.append(name)

    def get_op_output_nodes(self, node):
        output_nodes = set()
        for buf in self.get_output_buffers(node):
            output_nodes.update(buf.consumers)
        return list(output_nodes)

    def get_op_input_nodes(self, node):
        return [buf.producer for buf in self.get_input_buffers(node)]

    def get_input_op_types(self, node):
        return [self.buffers[name].producer.op.type for name in node.input_names]

    def get_output_op_types(self, node):
        consumer_nodes = []
        consumer_nodes_types = []
        for name in node.output_names:
            for consumer in self.buffers[name].consumers:
                # consumer already existing in our list can happen if one consumer takes 2 or more outputs of a node.
                # e.g: if node_a has buf_1, buf_2 as outputs and next layer(node_b) has both of these buffers as input,
                # both buf_1 and buf_2 will list node_b as consumers so we don't want to have [node_b, node_b]
                # for outputs
                if consumer not in consumer_nodes:
                    consumer_nodes.append(consumer)
                    consumer_nodes_types.append(consumer.op.type)
        return consumer_nodes_types

    def get_buffer(self, buffer_name):
        return self.buffers[buffer_name]

    def has_buffer(self, buffer_name):
        return buffer_name in self.buffers

    def add_buffer(self, buf):
        if self.has_buffer(buf.name):
            raise ValueError("Duplicate buffer name, {} already exists".format(buf.name))
        self.buffers[buf.name] = buf

    def delete_buffer(self, buffer_name):
        if not self.has_buffer(buffer_name):
            raise ValueError("Buffer ({}) requested for deletion is not found.".format(buffer_name))
        del self.buffers[buffer_name]

    def get_producer_node(self, buffer_name):
        return self.buffers[buffer_name].producer

    def get_producer_op(self, buffer_name):
        return self.buffers[buffer_name].producer.op

    def get_input_nodes_to_graph(self):
        input_nodes = []
        for node in self.list_nodes():
            if node.op.TRANSLATION_KEY == op_adapter.InputOp.TRANSLATION_KEY:
                input_nodes.append(node)
        return input_nodes

    def get_output_nodes_of_graph(self):
        output_nodes = []
        passed_output_node_names = self.output_names[:]
        # matches list of given output node names to graph node names
        for node in self.list_nodes():
            for output_node_name in self.output_names:
                if ((output_node_name + ":0") in node.output_names) or (output_node_name in node.output_names):
                    output_nodes.append(node)
                    passed_output_node_names.remove(output_node_name)

        for name in passed_output_node_names:
            log_warning("Output node {}, specified via command line, does not exist in graph.".format(name))

        return output_nodes

    def is_output_node(self, node):
        return node in self.get_output_nodes_of_graph()

    def list_nodes(self):
        return self.nodes_in_order[:]

    def list_buffers(self):
        return list(self.buffers.values())

    def has_op(self, op):
        nodes = self.list_nodes()
        for node in nodes:
            if node.op == op:
                return True
        return False

    def has_node(self, node):
        nodes = self.list_nodes()
        for node_ in nodes:
            is_node_equal, _ = node_.is_equal(node)
            if is_node_equal:
                return True
        return False

    def get_quantizable_tensors(self):
        params = {}
        for node in self.list_nodes():
            if node in self.list_nodes():
                tmp = {'type' : node.op.type}
                for kv in node.attrs.items():
                    if isinstance(kv[1], np.ndarray) and kv[1].dtype == np.float32 and \
                       node.op.hasattr('quantizable') and node.op['quantizable']:
                        tmp[kv[0]] = kv[1]
                if len(tmp) > 1:
                    params[node.name] = tmp
        return params

    def get_overridden_encoding(self, name, is_param=True):
        """
        Returns quantization encoding if provided externally for a given buffer name
        :param graph: IR graph object
        :param name: name of the buffer whose quantization encoding is to be retrieved
        :param is_param: True if given buffer is a parameter such as weights and bias, False for activation buffer
        :return: list of encoding dictionary for the given buffer, if any, otherwise return None
        """
        if self.user_quantization_overrides:
            if is_param:
                param_encodings = self.user_quantization_overrides['param_encodings']
                if name in param_encodings.keys():
                    return param_encodings[name]
            else:
                act_encodings = self.user_quantization_overrides['activation_encodings']
                if name in act_encodings.keys():
                    return act_encodings[name]
        return None

    def remove_overridden_encoding(self, name, is_param=True):
        """
        Removes user provided quantization encoding for a given buffer name
        :param name: name of the buffer whose quantization encoding is to be deleted
        :param is_param: True if given buffer is a parameter such as weights and bias, False for activation buffer
        """
        if is_param:
            del self.user_quantization_overrides['param_encodings'][name]
        else:
            del self.user_quantization_overrides['activation_encodings'][name]
