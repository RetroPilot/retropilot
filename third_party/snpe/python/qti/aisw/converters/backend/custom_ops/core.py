# ==============================================================================
#
#  Copyright (c) 2020 - 2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from collections import OrderedDict
from qti.aisw.converters.common.custom_ops.core import *
from qti.aisw.converters.backend.custom_ops.helpers.udo_module_helpers import *
from qti.aisw.converters.common.custom_ops.utils.config_helpers import *


# ------------------------------------------------------------------------------
#   SNPE UDO config Core Classes
# ------------------------------------------------------------------------------
class TensorInfo(CustomTensorInfo):
    shape = property_type('shape', str)  # string for now, should be something interpretable

    def __init__(self, **tensor):
        super().__init__(**tensor)
        self.data = None
        self.quant_type = None

        # snpe needs a param type for all objects
        self.param_type = "SNPE_UDO_PARAMTYPE_TENSOR"

    def from_dict(self, tensor_dict, name=''):
        self.name = tensor_dict.get('name', name)
        self.dimensions = tensor_dict.get('dims', [])
        self.static = tensor_dict.get('static', False)
        self.default_value = tensor_dict.get('default_value', None)
        self.data = tensor_dict.get('data', [])
        self.layout = get_internal_tensor_layout(tensor_dict.get('tensor_layout', "NHWC"),
                                                 self.name)
        # udo only allows a single datatype to be specified per tensor
        self.data_type = get_internal_data_type(tensor_dict.get("data_type", "FLOAT_32"), self.name)
        self.allowed_data_types = [self.data_type]
        # this is not really needed, but DSP templates still use it
        # TODO: Remove once its confirmed that this value is always TF anyway
        self.quant_type = SnpeUdoConstants.SNPE_UDO_QUANT_TYPES['TF']

    @staticmethod
    def create_tensor_infos(operator_dict, dict_type):
        tensor_infos = list()
        for tensor_dict in operator_dict.get(str(dict_type), list()):
            tensor_info = TensorInfo()
            tensor_info.from_dict(tensor_dict)
            tensor_infos.append(tensor_info)

        return tensor_infos

    @staticmethod
    def create_per_core_tensor_infos(tensor_dict, type, core_types=None):
        if core_types is None:
            core_types = SnpeUdoConstants.SNPE_UDO_CORETYPES.keys()
        tensor_infos = list()

        for tensor in tensor_dict.get(str(type), list()):
            tensor_info = TensorInfo()
            tensor_info.from_dict(tensor)
            setattr(tensor_info, "repeated", tensor.get("repeated", False))
            setattr(tensor_info, "static", tensor.get("static", False))
            if 'per_core_data_types' not in tensor:
                # set per core data types based on data_type to maintain backward compatibility
                # if a user sets data_type only it will be replicated for each runtime
                data_types = [tensor.get("data_type")] * len(core_types)
                per_core_dict = get_per_core_data_types(dict(zip(core_types, data_types)))
                setattr(tensor_info, "per_core_data_types", per_core_dict)

                # In this case, it should always be assumed that static data always has an allowed
                # data type of float32 since the CPU core type was not defined
                if tensor_info.static:
                    tensor_info.allowed_data_types.append("SNPE_UDO_DATATYPE_FLOAT_32")
            else:
                per_core_dict = get_per_core_data_types(tensor.get("per_core_data_types"))
                tensor_info.allowed_data_types = list(per_core_dict.keys())
                if "SNPE_UDO_CORETYPE_CPU" in per_core_dict:
                    # if a core type has been defined for CPU explicitly, the default
                    # datatype should be changed
                    tensor_info.data_type = per_core_dict["SNPE_UDO_CORETYPE_CPU"]
                setattr(tensor_info, "per_core_data_types", per_core_dict)
            tensor_infos.append(tensor_info)

        return tensor_infos

    def as_dict(self):
        temp_dict = super(TensorInfo, self).as_dict()
        temp_dict.update(param_type=self.param_type)
        temp_dict.update(per_core_data_types=getattr(self, "per_core_data_types", None))
        return temp_dict


class ScalarParam(CustomScalarParam):
    def __init__(self, data, data_type=None):
        if data_type is None:
            data_type = get_snpe_type(data)  # assign datatype based on data
        super().__init__(data, data_type)
        self.param_type = 'SNPE_UDO_PARAMTYPE_SCALAR'

    def as_dict(self):
        temp_dict = super(ScalarParam, self).as_dict()
        temp_dict.update(param_type=self.param_type)
        return temp_dict


class TensorParam(CustomTensorParam):
    def __init__(self, data, tensor_info):
        super().__init__(data, tensor_info)
        # now set datatype based on data, or use existing datatype if data is deliberately empty
        self.data_type = get_snpe_type(data) if data else tensor_info.data_type
        self.param_type = 'SNPE_UDO_PARAMTYPE_TENSOR'

    def as_dict(self):
        temp_dict = super(TensorParam, self).as_dict()
        temp_dict.update(param_type=self.param_type)
        temp_dict.update(per_core_data_types=getattr(self, "per_core_data_types", None))
        return temp_dict


class StringParam(ScalarParam):
    def __init__(self, value):
        super().__init__(value, 'SNPE_UDO_DATATYPE_UINT_8')
        self.param_type = 'SNPE_UDO_PARAMTYPE_STRING'

    def as_dict(self):
        temp_dict = super(StringParam, self).as_dict()
        temp_dict.update(param_type=self.param_type)
        return temp_dict


class Operator(CustomOperator):
    """
    This object describes an operation provided in the config spec, using inputs, outputs, tensor_params and scalar_params.
    The metaclass ensures that the certain types are valid. The udo_property method ensures that those fields cannot be
    set directly, and is essentially an accessor to view the operator's members.
    """
    input = aggregate_property('input', TensorInfo)
    output = aggregate_property('output', TensorInfo)
    param = aggregate_property('param', TensorInfo)

    def __init__(self, type_name, core_types=None, *, dsp_arch_types=None):
        super().__init__(type_name)
        if core_types is not None:
            self.core_types = get_internal_core_types(core_types)
        else:
            self.core_types = [SnpeUdoConstants.SNPE_UDO_CORETYPES["CPU"]]
        self.dsp_arch_types = dsp_arch_types
        self.__param_types = dict()

    @staticmethod
    def from_dict(op_dict):
        try:
            core_types = op_dict['core_types']
            self = Operator(op_dict['type'], core_types)
            self.inputs(TensorInfo.create_per_core_tensor_infos(op_dict, 'inputs', core_types))
            self.outputs(TensorInfo.create_per_core_tensor_infos(op_dict, 'outputs', core_types))
        except KeyError as e:
            raise KeyError(
                "Required operator field: {} was not found in config".format(str(e).split(':')[-1]))

        # Create params as generic tensor info and then set param type manually
        scalar_params = TensorInfo.create_tensor_infos(op_dict, 'scalar_params')
        for param in scalar_params:
            setattr(param, "param_type", "SNPE_UDO_PARAMTYPE_SCALAR")
        self.params(scalar_params)

        tensor_params = TensorInfo.create_tensor_infos(op_dict, 'tensor_params')
        self.params(tensor_params)

        self.dsp_arch_types = op_dict['dsp_arch_types'] if 'dsp_arch_types' in op_dict else []

        return self

    @property
    def scalar_param(self):
        return [param for param in self.param if param.param_type == "SNPE_UDO_PARAMTYPE_SCALAR"]

    @property
    def tensor_param(self):
        return [param for param in self.param if param.param_type == "SNPE_UDO_PARAMTYPE_TENSOR"]

    def __copy__(self):
        new_operator = Operator(self.name, self.core_types)
        new_operator.inputs(self.input)
        new_operator.outputs(self.output)
        new_operator.params(self.params)
        new_operator.dsp_arch_types = self.dsp_arch_types
        new_operator.__param_types = self.__param_types


class Param(CustomParam):
    param_type = property_type('param_type', ParamTypes)
    param = union_property('param', [type(None), ScalarParam, TensorParam])

    def __init__(self, name, param_type, param=None):
        super(Param, self).__init__(name, param_type, param)


class SnpeUdoCustomOp(CustomOp):
    __metaclass__ = ABCMeta
    methods = dict()
    inputs = aggregate_property('inputs', TensorInfo)
    outputs = aggregate_property('outputs', TensorInfo)
    param = aggregate_property('params', Param)

    def __init__(self,
                 op_type: str,
                 input_tensors: List[TensorInfo],
                 output_tensors: List[TensorInfo], *,
                 params: Optional[List[Param]] = None,
                 param_info: Optional[List[TensorInfo]] = None,
                 src_op=None,
                 infer_output_shapes=None,
                 name: Optional[str] = ""):
        super().__init__(op_type, input_tensors,
                         output_tensors, params, param_info, src_op, name)
        if infer_output_shapes is not None:
            self.infer_output_shapes = infer_output_shapes
        # set backend specific arguments
        self.set_axis_orders(self.inputs, tensor_layouts=SnpeUdoConstants.SNPE_UDO_TENSOR_LAYOUT)
        self.set_axis_orders(self.outputs, tensor_layouts=SnpeUdoConstants.SNPE_UDO_TENSOR_LAYOUT)

    def as_dict(self):
        tensor_params = {param.name: param.param.as_dict() for _, param in self.params.items()
                         if param.param_type == ParamTypes.TENSOR}
        scalar_params = {param.name: param.param.as_dict() for _, param in self.params.items()
                         if param.param_type == ParamTypes.SCALAR or
                         param.param_type == ParamTypes.STRING}
        inputs = OrderedDict()
        outputs = OrderedDict()
        for input_ in self.inputs:
            inputs[input_.name] = input_.as_dict()
        for output in self.outputs:
            outputs[output.name] = output.as_dict()

        return inputs, outputs, scalar_params, tensor_params

    @classmethod
    @abstractmethod
    def extract_attrs(cls, src_op, param_info: Dict[str, TensorInfo]):
        """
        The intention of this method is to extract param_info from a framework src_op and return a dictionary of
        Param objects, such that "attr_name": "Param". This must be implemented, as it is called during
        initialization
        :param src_op: Framework src_op
        :param param_info: Parameter info
        :return: A dictionary of Params
        """

    @abstractmethod
    def infer_output_shapes(self, node, **kwargs):
        """
        This method recieves a framework node and returns the output shapes
        :param node:
        :param kwargs:
        :return: a list of lists which contain output dimensions for each output tensor
        """

    @abstractmethod
    def set_tensor_data_types(self, node):
        """
        Sets the datatype for each input and output tensor based on the operation instance
        :param node : The source framework node
        :raises An error if data_type cannot be set
        :returns
        """

    def set_static_tensor_to_param(self, tensors):
        """
        Sets a static tensor to a param. This method is called by the base class, meaning instances of this class
        are expected to have static tensors become params. This method takes a single tensor, and changes it to a
        param object. Note that a static tensor must have a data field defined.
        :param tensors: The tensor to be made a param.
        """
        local_tensor = []
        for tensor_info in tensors:
            if tensor_info.static:
                log_debug('Static custom input tensor: {} found for op: {}. '
                          'Note this tensor will be stored in the model output'
                          .format(tensor_info.name, self.op_type))
                self.params[tensor_info['name']] = Param(tensor_info['name'], ParamTypes.TENSOR,
                                                         TensorParam(None, tensor_info))
            else:
                local_tensor.append(tensor_info)

        return local_tensor


BackendCustomOp = SnpeUdoCustomOp
