# ==============================================================================
#
#  Copyright (c) 2020 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from qti.aisw.converters.backend.custom_ops.core import BackendCustomOp as CustomOp
from qti.aisw.converters.backend.custom_ops.core import Param, \
    ScalarParam, TensorParam, StringParam, ParamTypes, convert_to_backend_type_from_numpy
from qti.aisw.converters.common.utils import code_to_message
import numpy as np
import google.protobuf.pyext as proto
import caffe.proto.caffe_pb2 as caffe_pb2
from collections import OrderedDict


class CustomCaffeOp(CustomOp):
    """
    A subclass of the CustomOp interface which implements framework specific methods defined in
    CustomOp. Calling this class requires that a caffe module can be imported. Note that an instance
    of this class has further requirements than a generic CustomOp, in that the output dims must
    be determinable or provided. Additionally, the parameters must be extractable from the op.
    See CustomOp for all methods that will be called when a CustomCaffeOp is instantiated.
    """

    def __init__(self, src_op, input_tensor_info, output_tensor_info, param_info, model):
        self.model = model  # set model here since it may be needed (there may be a better location)
        self.param_info = param_info
        self.input_tensor_info = input_tensor_info
        self.src_op = src_op
        output_tensors = self.set_output_dims(src_op, output_tensor_info, model)
        super(CustomCaffeOp, self).__init__(src_op.type,
                                            src_op=src_op,
                                            input_tensors=input_tensor_info,
                                            output_tensors=output_tensors,
                                            param_info=param_info,
                                            name=src_op.name)
        self.num_inputs = len(self.inputs)
        self.num_outputs = len(self.outputs)

    @classmethod
    def extract_attrs(cls, src_op, param_infos):
        """
        This method extracts attributes from the caffe src_op, using a list of param_infos that have been
        obtained from the operator spec.

        :param src_op: The caffe src_op
        :param param_infos: The list of parameter information which should be a list of TensorInfo objects.
        :return: a dictionary of attributes, where the key is the attribute name and the value is a Param object
        """
        attrs = OrderedDict()
        param_obj = cls.get_param_object(src_op)
        nested_param_obj = cls.get_nested_params(param_obj)

        attr_value = None
        for param in param_infos:
            if hasattr(src_op, param.name):
                attr_value = getattr(src_op, param.name)
            elif param_obj and hasattr(param_obj, param.name):
                attr_value = getattr(param_obj, param.name)

                # attr value may be a protobuf scalar container or a filler parameter,
                # in both cases it is turned into a list. Note: the filler parameter
                # will be contained in the blobs object and filled during the translation.
                if isinstance(attr_value, proto._message.RepeatedScalarContainer):
                    attr_value = list(attr_value)
                    # if the attr_value is empty and the param is not static, then we don't
                    # bother adding it to the list of attributes as it there is no data to
                    # serialize.
                    if not attr_value and not param.static:
                        if param.default_value:
                            attr_value = param.default_value
                        else:
                            continue
                elif isinstance(attr_value, caffe_pb2.FillerParameter):
                    attr_value = []
            # this is to extract the nested parameters
            # in case of ops like DetectionOutput, nms params are nested
            elif nested_param_obj and hasattr(nested_param_obj, param.name):
                attr_value = getattr(nested_param_obj, param.name)
            elif param.default_value:
                attr_value = param.default_value
            elif not param.static:
                raise RuntimeError('Could not extract parameter: {} from src_op: {} '.format(
                    param.name, src_op.name))

            if isinstance(attr_value, str):
                parm =Param(param.name, ParamTypes.STRING, StringParam(attr_value))
            elif isinstance(attr_value, bool):
                parm = Param(param.name, ParamTypes.SCALAR, ScalarParam(int(attr_value)))
            elif isinstance(attr_value, (list, tuple, np.ndarray)):
                parm = Param(param.name, ParamTypes.TENSOR,
                             TensorParam(attr_value, param))
            elif isinstance(attr_value, (int, float)):
                parm = Param(param.name, ParamTypes.SCALAR, ScalarParam(attr_value))
            else:
                raise RuntimeError('Could not determine parameter type for: {} from src_op: {} '
                                   ''.format(param.name, src_op.name))

            attrs[param.name] = parm

        return attrs

    def infer_output_shapes(self, node, model=None, **kwargs):
        """
         This method infers the shape of a Caffe NodeProto's output tensors using the model and
         node information.

        :param node: The LayerParameter object
        :param model: The NetParameter object
        """
        output_dims = []
        if model:
            for top in node.top:

                if hasattr(model.blobs[top], 'shape'):
                    shape = model.blobs[top].shape
                elif hasattr(model.blobs[top], 'data'):
                    shape = model.blobs[top].data.shape
                else:
                    raise KeyError('Caffe blob:{} is missing shape parameter'.format(str(top)))

                output_dims.append(list([dim for dim in shape]))
        else:
            raise RuntimeError(code_to_message.get_error_message("ERROR_MODEL_NOT_VALID"))

        return output_dims

    def set_tensor_data_types(self, node, validate=False, **kwargs):
        for output in node.top:
            caffe_op_outputs = [tensor for tensor in self.outputs if tensor.name == output]

            caffe_op_output = caffe_op_outputs[0]
            caffe_blob = self.model.blobs[output]
            caffe_op_output.data_type = convert_to_backend_type_from_numpy(caffe_blob.data.dtype)

        for input_ in node.bottom:
            caffe_op_inputs = [tensor for tensor in self.inputs if tensor.name == input_]

            # if the input name is not found, that means it could be a static input which has been
            # moved into the list of params. If so, the input is skipped otherwise an error is
            # returned
            if not caffe_op_inputs:
                if any(param_name == input_ for param_name in self.params):
                    continue
                raise TypeError("Could not map datatype for tensor: {} in node: {}".format(input_,
                                                                                           node.name))
            caffe_op_input = caffe_op_inputs[0]
            caffe_blob = self.model.blobs[input_]
            caffe_op_input.data_type = convert_to_backend_type_from_numpy(caffe_blob.data.dtype)

    def validate(self):
        self.validate_params(self.src_op, self.param_info, self.input_tensor_info)

    @staticmethod
    def validate_params(src_op, param_info, input_tensor_info):
        """
        Validate params in the src_op with respect to param_infos defined in the config spec. Note
        that unlike tensors, params must be named in the config spec. If the param is not known to
        the custom op definition, a KeyError is raised. Likewise, if a param not provided in the
        config spec is included, the error is also raised.
        :param src_op: The onnx op containing the params
        :param param_info: The list of param information as defined in the config spec.
        :param input_tensor_info: The list of input tensors
        :raises: a KeyError if the param is missing or an param is present in the op.
        """
        param_obj = CustomCaffeOp.get_param_object(src_op)
        nested_param_obj = CustomCaffeOp.get_nested_params(param_obj)

        def _is_missing_param_obj(param_name, obj):
            return param_name not in (attr[0].name for attr in obj.ListFields()) and not \
                    param.static and param.default_value is None

        for param in param_info:
            # This checks if an attribute present in the custom Op description is also
            # present in the model. If the attribute is not present but is listed as either static:
            # meaning it will found in the inputs, or has a default value, then no error is raised.
            # Otherwise, a missing attribute error is raised.
            valid_param = True
            if _is_missing_param_obj(param.name, param_obj):
                valid_param = False
            # if the parent param is valid, check nested params
            elif nested_param_obj and _is_missing_param_obj(param.name, nested_param_obj):
                valid_param = False

            if not valid_param:
                raise KeyError(
                    code_to_message.get_error_message('ERROR_MISSING_ATTRIBUTE')(param.name,
                                                                                     src_op.type))
        for attr in param_obj.ListFields():
            valid_attr = True
            checked_attr_name = attr[0].name

            # First check if the attribute name has been defined in the parameter info
            if checked_attr_name not in (param.name for param in param_info):
                # for caffe, it is unclear if filler attributes are listed as attributes or inputs
                # we need to check that if an attribute is unknown, then we look in the list of
                # inputs
                if checked_attr_name in (tensor.name for tensor in input_tensor_info):
                    continue
                valid_attr = False

            # Check nested attributes if any provided the parent attr name is valid
            elif len(attr) > 1 and hasattr(attr[1], 'DESCRIPTOR'):
                for nested_attr in attr[1].ListFields():
                    checked_attr_name = nested_attr[0].name
                    if checked_attr_name not in (param.name for param in param_info):
                        valid_attr = False
                        break

            # if the attribute is invalid for any reason
            if not valid_attr:
                raise KeyError(
                    code_to_message.get_error_message("ERROR_CUSTOM_OP_ATTRIBUTE_NOT_SUPPORTED")
                    (checked_attr_name, src_op.type,
                     [param.name for param in param_info]))

    @staticmethod
    def get_param_object(src_op):
        param_obj = None
        # identify parameter object, as caffe parameter are usually grouped as op_name_param
        # if we cannot find it that way, then look to see if any of the registered descriptors
        # matches the expected class name: op_typeParameter.
        if hasattr(src_op, (str(src_op.type).lower() + '_param')):
            param_obj = getattr(src_op, str(src_op.type).lower() + '_param')
        else:
            try:
                for potential_param in src_op.ListFields():
                    if hasattr(potential_param[1], 'DESCRIPTOR'):
                        if potential_param[1].DESCRIPTOR.name == str(src_op.type) + 'Parameter':
                            param_obj = potential_param[1]
                            break
                    else:
                        continue
            except Exception:
                raise TypeError('Could not identify attributes from Caffe src_op:{} of '
                                'type: {}'.format(src_op.name, src_op.type))
        return param_obj

    @staticmethod
    def get_nested_params(param_obj):
        nested_param_obj = None
        for attr in param_obj.ListFields():
            if hasattr(attr[1], 'DESCRIPTOR'):
                if attr[1].DESCRIPTOR.name.endswith('Parameter'):
                    nested_param_obj = attr[1]
        return nested_param_obj

