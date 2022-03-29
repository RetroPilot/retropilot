# ==============================================================================
#
#  Copyright (c) 2020 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
from qti.aisw.converters.common.utils import code_to_message
from qti.aisw.converters.common.utils.converter_utils import log_warning
from qti.aisw.converters.backend.custom_ops.core import BackendCustomOp as CustomOp
from qti.aisw.converters.backend.custom_ops.core import Param, ScalarParam, TensorParam, \
    StringParam, ParamTypes, convert_to_backend_type_from_numpy
import onnx

try:
    from onnx import version_converter

    version_converter_available = True
except ImportError:
    version_converter = None
    version_converter_available = False


class CustomOnnxOp(CustomOp):
    """
    A subclass of the CustomOp interface which implements framework specific methods defined in
    CustomOp. Calling this class requires that an Onnx module can be imported. Note that an
    instance of this class has further requirements than a generic CustomOp, in that the output
    dims must be determinable or provided. Additionally, the parameters must be extractable from
    the op. See CustomOp for all methods that will be called when a CustomOnnxOp is instantiated
    """

    def __init__(self, src_op, input_tensor_info, output_tensor_info, param_info, model):
        self.model = model
        self.src_op = src_op
        self.param_info = param_info
        input_tensors = input_tensor_info
        output_tensors = self.set_output_dims(src_op, output_tensor_info, model)
        super(CustomOnnxOp, self).__init__(src_op.op_type, src_op=src_op,
                                           input_tensors=input_tensors,
                                           output_tensors=output_tensors, param_info=param_info)

    @classmethod
    def extract_attrs(cls, src_op, param_infos):
        """
        This method extracts attributes from the provided onnx src_op, using a list of param_infos
        that have been obtained from the operator spec.

        :param src_op: The onnx src_op
        :param param_infos: The list of parameter information which should be a list of
        CustomTensorInfo objects.
        :return: a dictionary of attributes, where the key is the attribute name and the value
        is a CustomParam object
        """
        attrs = dict()
        parm = None
        extract_tensor = onnx.numpy_helper.to_array

        def is_iterable(attr_value):
            try:
                iter(attr_value)
            except TypeError:
                return False
            return True

        for param_info in param_infos:

            name = param_info.name
            result = [x for x in src_op.attribute if name == x.name]
            if not len(result):
                if param_info.default_value is not None:
                    attr_value = param_info.default_value
                    if not is_iterable(attr_value):
                        if isinstance(attr_value, (int, float, bool)):
                            if isinstance(attr_value, bool):
                                attr_value = int(attr_value)
                            parm = Param(name, ParamTypes.SCALAR,
                                         ScalarParam(attr_value))
                    else:
                        if isinstance(attr_value, str):
                            parm = Param(name, ParamTypes.SCALAR, StringParam(attr_value))
                        else:
                            parm = Param(name, ParamTypes.TENSOR,
                                         TensorParam(attr_value, param_info))
                    attrs[name] = parm
                    continue
                else:
                    raise KeyError(
                        code_to_message.get_error_message('ERROR_MISSING_ATTRIBUTE')
                        (param_info.name, src_op.op_type))
            attr = result[0]
            code = attr.type
            # TODO: Use the actual datatype from the value and validate against it

            if code == 1:
                parm = Param(name, ParamTypes.SCALAR, ScalarParam(attr.f))
            elif code == 2:
                parm = Param(name, ParamTypes.SCALAR, ScalarParam(attr.i))
            elif code == 3:
                string = str(attr.s).decode('utf-8')
                parm = Param(name, ParamTypes.SCALAR, StringParam(string))
            elif code == 4:
                parm = Param(name, ParamTypes.TENSOR,
                             TensorParam(extract_tensor(attr.t), param_info))
            elif code == 6:
                parm = Param(name, ParamTypes.TENSOR, TensorParam(list(attr.floats), param_info))
            elif code == 7:
                parm = Param(name, ParamTypes.TENSOR, TensorParam(list(attr.ints), param_info))
            elif code == 8:
                parm = Param(name, ParamTypes.TENSOR, TensorParam(list(attr.strings), param_info))
            elif code == 9:
                parm = Param(name, ParamTypes.TENSOR,
                             TensorParam(list(map(extract_tensor, attr.tensors)),
                                         param_info(name)))

            attrs[name] = parm

        return attrs

    @staticmethod
    def get_all_tensors_in_model(model):
        tensors = set()
        for node in model.graph.node:
            list(map(tensors.add, [output for output in node.output]))
            list(map(tensors.add, [input for input in node.input]))
        return tensors

    def infer_output_shapes(self, node, model=None, perform_shape_inference=False):
        """
         This method infers the shape of an Onnx NodeProto's output tensors using the node itself,
         a user provided model containing the node and optionally
         Onnx's in-built shape inference function.

        :param node: The onnx NodeProto object
        :param model: A required field which should be an Onnx ModelProto object
        :param perform_shape_inference: if set to True, the method will call Onnx's shape inference
        method, otherwise, the method will assume that the value info contains shape information
        for all output tensors.
        :return: a list of lists which contains output dimensions for each output tensor
        in the Onnx NodeProto.
        """
        output_dims = []
        if not model:
            raise RuntimeError(code_to_message.get_error_message("ERROR_MODEL_NOT_VALID"))
        if perform_shape_inference:
            inferred_model = self.up_convert_infer_shapes(model)
            self.model = inferred_model
        else:
            inferred_model = model

        # for each output in the node, this loop checks for a corresponding entry in the graph
        # value info or in the list of outputs. Note that Onnx's shape inference will return
        # nothing if shape inference is not available for the queried output so a value info must
        # be verified as non-empty.If the output name is present in either structure then the
        # boolean "found" is set to True, otherwise, an Exception is raised.
        for j in range(len(node.output)):
            found = False

            for value_info in inferred_model.graph.output:
                if node.output[j] == value_info.name:
                    output_dims.append(
                        [int(dim.dim_value) for dim in value_info.type.tensor_type.shape.dim])
                    found = True
                    break

            if found:
                continue
            else:
                for value_info in inferred_model.graph.value_info:
                    if not value_info:
                        break
                    elif value_info.name == node.output[j]:
                        output_dims.append(
                            [int(dim.dim_value) for dim in value_info.type.tensor_type.shape.dim])
                        found = True
                        break

            if not found:
                raise Exception(code_to_message.get_error_message('ERROR_INFER_OUTPUT_SHAPES')(
                    node.output[j]))
        return output_dims

    def set_tensor_data_types(self, node):
        onnx_numpy_map = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE
        for output in node.output:
            onnx_op_output = [tensor for tensor in self.outputs if tensor.name == output][0]
            onnx_dtype = None
            for output_info in self.model.graph.output:
                if onnx_op_output.name == output_info.name:
                    onnx_dtype = output_info.type.tensor_type.elem_type
                    break
            if onnx_dtype is None:
                for output_value_info in self.model.graph.value_info:
                    if onnx_op_output.name == output_value_info.name:
                        onnx_dtype = output_value_info.type.tensor_type.elem_type
                        break
            if not onnx_dtype:
                raise TypeError("Could not resolve datatype for custom op "
                                "output tensor: {}".format(output))
            onnx_op_output.data_type = convert_to_backend_type_from_numpy(onnx_numpy_map[onnx_dtype])

        # setting inputs for now, although these are not really used by converter
        for input_ in node.input:
            onnx_op_inputs = [tensor for tensor in self.inputs if tensor.name == input_]

            # if the input name is not found, that means it could be a static input which has been
            # moved into the list of params. If so, the input is skipped otherwise an error is
            # returned
            if not onnx_op_inputs:
                if any(param_name == input_ for param_name in self.params):
                    continue
                raise TypeError("Could not resolve datatype for custom op "
                                "input tensor: {} ".format(input_))

            onnx_op_input = onnx_op_inputs[0]
            onnx_dtype = None
            for input_info in self.model.graph.input:
                if onnx_op_input.name == input_info.name:
                    onnx_dtype = input_info.type.tensor_type.elem_type
                    break
            if onnx_dtype is None:
                for input_value_info in self.model.graph.value_info:
                    if onnx_op_input.name == input_value_info.name:
                        onnx_dtype = input_value_info.type.tensor_type.elem_type
                        break
            if not onnx_dtype:
                raise TypeError("Could not resolve datatype for custom op "
                                "input tensor: {} ".format(input_))
            onnx_op_input.data_type = convert_to_backend_type_from_numpy(onnx_numpy_map[onnx_dtype])

    def validate(self, *args, **kwargs):
        self.validate_params(self.src_op, self.param_info)

    @staticmethod
    def validate_params(src_op, param_info):
        """
        Validate params in the src_op with respect to param_infos defined in the config spec. Note
        that unlike tensors,
        params must be named in the config spec. If the param is not present in the op, a KeyError
         is raised. Likewise, if a param not provided in the config spec is included,
         the error is also raised.
        :param src_op: The onnx op containing the params
        :param param_info: The list of param information as defined in the config spec.
        :raises: a KeyError if the param is missing or an param is present in the op.
        """
        for param in param_info:
            if param.name not in (attr.name for attr in src_op.attribute) \
                    and not param.static and param.default_value is None:
                raise KeyError(
                    code_to_message.get_error_message('ERROR_MISSING_ATTRIBUTE')(param.name,
                                                                                 src_op.op_type))
        for attr in src_op.attribute:
            if attr.name not in (param.name for param in param_info):
                log_warning("Attribute: {} was found in the op: {} but has not been defined in "
                            "the op config. The attribute will be ignored!",
                            attr.name, src_op.op_type)

    def up_convert_infer_shapes(self, model):

        def check_all_shapes(model: onnx.ModelProto):
            if len(model.graph.value_info) + len(model.graph.input) + len(model.graph.output) < \
                    len(self.get_all_tensors_in_model(model)):
                return False
            return True

        try:
            from onnx import shape_inference
        except ImportError:
            raise ImportError(
                "Could not import Onnx shape inference module "
                "which is needed to infer output shapes for custom ops")

        # only perform shape inference if the value info field is not populated at all
        # or the number of inferred shapes does not match the total number of model tensors
        if not model.graph.value_info or not check_all_shapes(model):
            inferred_model = shape_inference.infer_shapes(model)
        else:
            inferred_model = model

        if not check_all_shapes(inferred_model):
            # Onnx shape inference had known issues before opset version 6 and although shape
            # inference is not guaranteed to be complete, opset versions greater than 6
            # are usually correct unless there is an unknown issue. A warning is displayed in
            # those cases as the missing shapes may not actually impact conversion.
            if inferred_model.opset_import[0].version > 6:
                log_warning("ONNX_CUSTOM_OP_INFER_SHAPES: Could not infer shapes for "
                            "all model tensors. This may cause issues during conversion")
            elif version_converter_available:
                version_converted_model = version_converter.convert_version(model, 6)
                inferred_model = shape_inference.infer_shapes(version_converted_model)
            else:
                raise RuntimeError(
                    "Could not infer shapes for this model, a"
                    "s the opset version is too low. Expected > {}, "
                    "instead got > {}".format("6", model.opset_import[0].version))
        return inferred_model
