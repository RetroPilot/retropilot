# ==============================================================================
#
#  Copyright (c) 2019-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
from qti.aisw.converters.common.converter_ir import op_adapter
from qti.aisw.converters.common.utils import code_to_message
from qti.aisw.converters.backend.custom_ops.op_factory import OpFactory
from qti.aisw.converters.backend.custom_ops.core import *
from qti.aisw.converters.tensorflow.common import LayerBuilder, LayerDescriptor, LayerResolver
import numpy as np


class CustomTfOp(BackendCustomOp):
    def __init__(self, src_op, input_tensor_info, output_tensor_info, param_info, model):
        output_tensor_info = self.set_output_dims(src_op, output_tensor_info, model)
        self.src_op = src_op
        self.param_info = param_info
        super(CustomTfOp, self).__init__(src_op.type,
                                         name=src_op.name,
                                         src_op=src_op,
                                         input_tensors=input_tensor_info,
                                         output_tensors=output_tensor_info,
                                         param_info=param_info)

    @classmethod
    def extract_attrs(cls, src_op, param_infos):
        # check if attribute can be extracted
        def is_iterable(attr_value):
            try:
                iter(attr_value)
            except TypeError:
                return False
            return True

        attrs = dict()
        type_attrs = [attr.name for attr in src_op.op_def.attr if attr.type == 'type']
        for param_info in param_infos:
            name = param_info.name

            # type_attrs are not extracted since we don't use them
            if name in type_attrs:
                continue

            if name in src_op.node_def.attr:
                attr_value = src_op.get_attr(name)
            elif param_info.static:
                attr_value = []
            elif param_info.default_value is not None:
                attr_value = param_info.default_value
            else:
                raise KeyError(code_to_message.get_error_message('ERROR_MISSING_ATTRIBUTE')(param_info.name,
                                                                                            src_op.type))
            if is_iterable(attr_value):
                iterable = True
            elif type(attr_value) is bool:
                attr_value = int(attr_value)
                iterable = False
            elif isinstance(attr_value, (int, float)):
                iterable = False
            else:
                raise TypeError("Type: {} for attr: {} is not recognized".format(type(attr_value), name))

            if not iterable:
                attrs[name] = Param(name, ParamTypes.SCALAR, ScalarParam(attr_value))
            else:
                if isinstance(attr_value, (str, bytes)):
                    if isinstance(attr_value, bytes):
                        # assuming unicode or bytes and utf-8 encoding
                        attr_value = attr_value.decode('utf-8') + '\0'
                    attrs[name] = Param(name, ParamTypes.STRING, StringParam(attr_value))
                else:
                    attrs[name] = Param(name, ParamTypes.TENSOR,
                                        TensorParam(attr_value, param_info))
        return attrs

    def infer_output_shapes(self, node, **kwargs):
        """
        Extracts the shape from the tensorflow output nodes
        """
        output_dims = []
        for tensor in node.outputs:
            if not any(str(dim.value) == "?" or dim.value is None for dim in tensor.shape.dims):
                output_dims.append(tensor.get_shape().as_list())
            else:
                output_dims.append([])
        return output_dims

    def set_tensor_data_types(self, node):
        for i, output in enumerate(self.outputs):
            tensor = node.outputs[i]
            dtype = tensor.dtype.as_numpy_dtype
            output.data_type = convert_to_backend_type_from_numpy(dtype)

        for i, input_ in enumerate(self.inputs):
            tensor = node.inputs[i]
            dtype = tensor.dtype.as_numpy_dtype
            input_.data_type = convert_to_backend_type_from_numpy(dtype)

    def validate(self, *args):
        self.validate_params(self.src_op, self.param_info)

    @staticmethod
    def validate_params(src_op, param_info):
        """
        Validate params in the src_op with respect to param_infos defined in the config spec.
        Note that unlike tensors, params must be named in the config spec.
        If the param is not present in the op, a KeyError is raised. Likewise,
        if a param not provided in the config spec is included, a warning is issued
        :param src_op: The onnx op containing the params
        :param param_info: The list of param information as defined in the config spec.
        :raises: a KeyError if the param is missing or an param is present in the op.
        """
        for param in param_info:
            if param.name not in src_op.node_def.attr \
                    and not param.static and param.default_value is None:
                raise KeyError(code_to_message.get_error_message('ERROR_MISSING_ATTRIBUTE')(param.name,
                                                                                            src_op.type))

        # some attributes are included simply to denote the type of an input
        type_attrs = [attr.name for attr in src_op.op_def.attr if attr.type == 'type']

        # some attributes only indicate the amount of expected inputs or outputs
        number_attr_string = "N"

        for attr in src_op.node_def.attr:
            if attr not in (param.name for param in param_info):
                if attr in type_attrs or str(attr) == number_attr_string:
                    continue
                log_debug("Attribute: {} was found in the op: {} but has not been defined"
                            " in the op config. "
                            "The attribute will be ignored!".format(attr, src_op.name))


class CustomLayerResolver(LayerResolver, object):
    class Descriptor(LayerDescriptor):
        def __init__(self, name, nodes, custom_op_instance):
            super(CustomLayerResolver.Descriptor, self).__init__('custom', name, nodes)
            self.custom_op = custom_op_instance
            self.src_op = self.child_ops[-1]

    def __init__(self):
        self.sequence = None

    def resolve_layer(self, graph_matcher, graph_helper):
        descriptors = []
        for graph_node in graph_matcher.graph:
            for node_type in graph_node.node_types:
                if node_type in graph_helper.op_collection:
                    original_tf_op = graph_node.original_node
                    target_custom_tf_op = graph_helper.op_collection.get_op_by_name(node_type,
                                                                                    original_tf_op.name)
                    const_input_ops = set()
                    input_ops = []
                    consumable_types = ['Const', 'Identity', 'Variable', 'Fill']

                    consumable_inputs = [input for input in original_tf_op.inputs
                                         if input.op.type in consumable_types]
                    # resolve missing dims that require further evaluation
                    for i, dim in enumerate(target_custom_tf_op.output_dims):
                        if not dim:
                            tensor = original_tf_op.outputs[i]
                            output = graph_helper.evaluate_tensor_output(tensor)
                            assert all(shape for shape in output.shape)
                            target_custom_tf_op.output_dims[i] = list(output.shape)

                    for input in consumable_inputs:
                        candidate_input_ops = []
                        # The idea here is to first determine if the input to an op has a
                        # constant origin, which implies it may be part of an input sequence that
                        # should either be consumed as a series of child ops or resolved by the
                        # converter individually (as either ignored or a legitimate op). Note:
                        # Check_tensor_const_origin may take a while as it may propagate to the
                        # top-level node in the worst case.
                        if graph_helper.check_tensor_const_origin(input)[0]:
                            candidate_input_ops = graph_helper.get_consumable_input_sequence(input)

                        for op in candidate_input_ops:
                            if op.type not in consumable_types:
                                # Here we have determined that there is a input op which is part
                                # of an input sequence however the op type cannot be trivially
                                # consumed as a child op. Print a warning here to let the user
                                # know that the op may cause an error if it cannot be resolved.
                                log_warning("Cannot resolve non-const sequence of input ops as "
                                            "child operations of Custom Op. Converter will fail "
                                            "if Op: {} cannot "
                                            "be resolved independently or as part of another "
                                            "sequence. "
                                            .format(op.type))
                            else:
                                const_input_ops.add(op)
                    input_ops.extend(const_input_ops)
                    input_ops.append(original_tf_op)  # always append the original op last
                    custom_op_descriptor = CustomLayerResolver.Descriptor(original_tf_op.name,
                                                                          input_ops,
                                                                          target_custom_tf_op)
                    descriptors.append(custom_op_descriptor)
        return descriptors

    def is_final_resolution(self):
        return True


class CustomLayerBuilder(LayerBuilder):
    def build_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):
        custom_op = descriptor.custom_op
        local_graph_helper = converter_context.graph_helper
        package_name = OpFactory.get_package_name(custom_op.op_type)

        # Because descriptors may have been merged into the child ops, the original src op
        # is retrieved as a class member. All other ops are considered to be unconsumed input ops.
        src_op = descriptor.src_op
        unconsumed_input_ops = descriptor.child_ops[0:-1]

        for name, custom_param in custom_op.params.items():
            param = custom_param.param
            if custom_param.param_type == ParamTypes.TENSOR and param.data is None:
                if not param.static:
                    raise ValueError(
                        code_to_message.get_error_message("ERROR_CUSTOM_OP_PARAM_NO_DATA")(name, custom_op.op_type))
                else:
                    tensor = local_graph_helper.get_tensor_by_name(custom_param.name)
                    tensor_idx = [i for i, input in enumerate(src_op.inputs) if input.name == custom_param.name]
                    consumed_op = [op for op in unconsumed_input_ops for output in op.outputs if
                                   output.name == tensor.name]
                    if consumed_op and tensor_idx:
                        try:
                            attr_name = getattr(src_op.op_def.input_arg[tensor_idx[0]], 'name')
                        except IndexError:
                            # try to split the string on the assumption that the tensor name will
                            # always include the param name before a ':'. This is purely an
                            # assumption based on tensorflow models seen so far.
                            attr_name = (str(tensor.name).split('/')[-1]).split(':')[0]

                            # make sure that name we have split exists in the input arg otherwise
                            # we give up
                            if not any(attr_name == input_arg.name for input_arg in src_op.op_def.input_arg):
                                raise LookupError(
                                    code_to_message.get_error_message("ERROR_CANNOT_INGEST_STATIC_INPUT")
                                    (str(name)))

                        output_tensor = local_graph_helper.evaluate_tensor_output(tensor)
                        param.data = np.asarray(output_tensor)  # ensure array is numpy
                        param.data_type = get_internal_dtype(output_tensor, param)
                        param.rank = len(output_tensor)
                        param.dimensions = list(output_tensor.shape)

                    else:
                        raise LookupError(code_to_message.get_error_message("ERROR_CANNOT_INGEST_STATIC_INPUT")
                                          (str(name)))

        inputs, outputs, scalar_params, tensor_params = custom_op.as_dict()
        input_names = self.get_input_names(converter_context, descriptor, input_descriptors)
        output_names = descriptor.output_names

        # since merging may have occurred, output names may have been updated.
        # we need to add the axis order for the new output to the custom_op axis orders dict
        new_output_names = [output_name for output_name in output_names if
                            output_name not in custom_op.axis_orders.keys()]
        if new_output_names:
            for output_name in new_output_names:
                custom_op.axis_orders[output_name] = 'NOT_YET_DEFINED'
        ir_graph.add(op_adapter.CustomOp(name=descriptor.layer_name,
                                         package_name=package_name,
                                         output_dims=custom_op.output_dims,
                                         custom_type=src_op.type,
                                         axis_orders=custom_op.axis_orders,
                                         inputs=inputs,
                                         outputs=outputs,
                                         scalar_params=scalar_params,
                                         tensor_params=tensor_params), input_names, output_names)
