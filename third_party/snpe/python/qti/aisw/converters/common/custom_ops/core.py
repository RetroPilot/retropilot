# ==============================================================================
#
#  Copyright (c) 2020-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
import collections
from typing import List, Optional, Dict

from qti.aisw.converters.common.custom_ops.utils.config_helpers import *


# ------------------------------------------------------------------------------
#   CustomOp Module Core Classes
# ------------------------------------------------------------------------------

class ParamTypes(Enum):
    """
    Defines the allowed parameter types
    """
    TENSOR = 0,
    SCALAR = 1,
    STRING = 1


class CustomTensorInfo:
    # TODO: repeated, constraints, tensor_layout enum
    def __init__(self, **tensor):
        self.name = tensor.get('name')
        self.allowed_data_types = tensor.get('allowed_data_types')
        self.allowed_values = tensor.get("allowed_values", [])
        self.shape = tensor.get('shape', "")
        self.rank = tensor.get('rank', 1)
        self.default_value = tensor.get('default_value', None)
        self.layout = tensor.get('layout', "NHWC")
        self.repeated = tensor.get('repeated', False)
        self.dimensions = []
        self.static = tensor.get("static", False)
        # tensor infos do not have an explicit type unless tied to a model instance
        self.data_type = None

    def get(self, item, default=None):
        if item in self.__dict__:
            return self.__dict__[item]
        else:
            return default

    def __getitem__(self, name):
        return self.__dict__[name]

    def __repr__(self):
        return str(self.as_dict()).replace('\\', '')

    def __iter__(self):
        return self.as_dict()

    def as_dict(self):
        return dict(name=self.name,
                    allowed_data_types=self.allowed_data_types,
                    shape=self.shape,
                    default_value=self.default_value,
                    layout=self.layout,
                    dimensions=self.dimensions,
                    repeated=self.repeated,
                    static=self.static,
                    rank=self.rank,
                    data_type=self.data_type)


class CustomScalarParam:

    def __init__(self, data, data_type, allowed_values=None):
        self.data_type = data_type
        self.data = data
        self.allowed_values = allowed_values if allowed_values is not None else []

    def __repr__(self):
        return str(self.data_type)

    def as_dict(self):
        return dict(data_type=self.data_type, data=self.data)


class CustomTensorParam(CustomTensorInfo):
    def __init__(self, data, tensor_info):
        super().__init__(**tensor_info.as_dict())
        if data is not None:
            self.data = numpy.asarray(data)
            self.dimensions = list(self.data.shape)
            self.rank = len(self.dimensions)
        else:
            self.data = data

    def as_dict(self):
        new_dict = super(CustomTensorParam, self).as_dict()
        new_dict.update(data=self.data)
        return new_dict


class CustomStringParam(CustomScalarParam):
    def __init__(self, value):
        super().__init__(value, [])


class CustomParam:

    def __init__(self, name, param_type, param=None):
        self.name = name
        self.param_type = param_type
        self.param = param

    def get(self, item, default=None):
        if item in self.__dict__:
            return self.__dict__[item]
        else:
            return default

    def __getitem__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]

    def __repr__(self):
        return str(self.as_dict()).replace('\\', '')

    def as_dict(self):
        return dict(name=self.name, param_type=self.param_type, param=self.param.as_dict())

class CustomOperator:
    """
    This object describes an operation provided in the config spec, using inputs, outputs,
    params. The aggregate_property method ensures that those fields must be the correct type and
    cannot be set directly. The fields are only an accessor to view the operator's members.
    """

    input = aggregate_property('input', CustomTensorInfo)
    output = aggregate_property('output', CustomTensorInfo)
    param = aggregate_property('param', CustomTensorInfo)

    # setting package name statically for now, ideally it would be done from headers
    def __init__(self, type_name, package_name="qti.aisw", name="", use_default_translation=False):
        self.name = name
        self.package_name = package_name
        self.type_name = type_name
        self.use_default_translation = use_default_translation

    def inputs(self, input_tensors):
        if hasattr(self, '_input'):
            for tensor in input_tensors:
                self._input.append(tensor)
        else:
            setattr(self, '_input', [tensor for tensor in input_tensors])

    def outputs(self, output_tensors):
        if hasattr(self, '_output'):
            for tensor in output_tensors:
                self._output.append(tensor)
        else:
            setattr(self, '_output', [tensor for tensor in output_tensors])

    def params(self, params):
        if hasattr(self, '_param'):
            for tensor in params:
                self._param.append(tensor)
        else:
            setattr(self, '_param', [tensor for tensor in params])

    def get(self, attr, default=None):
        return getattr(self, attr, default)

    def __getitem__(self, item):
        try:
            return self.__dict__[item]
        except KeyError:
            raise KeyError("Operator field: {} was not found".format(str(item)))

    def __contains__(self, item):
        return hasattr(self, item)

    def __copy__(self):
        new_operator = CustomOperator(name=self.name, type_name=self.type_name,
                                      package_name=self.package_name)
        new_operator.inputs(self.input)
        new_operator.outputs(self.output)
        new_operator.params(self.param)

    def __repr__(self):
        return str(dict(type=self.type_name,
                        params=list(map(lambda x: x.__repr__(), self.param)),
                        inputs=list(map(lambda x: x.__repr__(), self.input)),
                        outputs=list(map(lambda x: x.__repr__(), self.output)),
                        use_default_translation=True)).replace('\\', '')


class CustomOp(object):
    """
    Abstract class which describes a Custom Operation in terms of its inputs, outputs
    and parameters. The intended usage is to create an Op that can be consumed by converters.
    """
    __metaclass__ = ABCMeta

    def __init__(self,
                 op_type: str,
                 input_tensors,
                 output_tensors,
                 params=None,
                 param_info=None,
                 src_op=None,
                 name: Optional[str] = ""):
        """
        This method initializes a CustomOp with the args provided, and as well as other members
        which depend on the provided arguments.
        :param op_type: The type of the CustomOp
        :param input_tensors: A list of CustomTensorInfo or CustomTensor objects.
        :param output_tensors: A list of CustomTensorInfo or CustomTensor objects.
        :param params: A dictionary of params such that "name": CustomParam are the key-value pairs.
        :param param_info: An optional argument, which can define params as a list of
        CustomTensorInfos. Note that initialization will attempt to call extract attrs on this
        argument.
        :param src_op: An optional argument, which is a framework node, or any object upon which a
        call to extract attrs has a well-defined behavior.
        :param name: Optional string indicating the name of the op
        """
        self.name = name
        self.op_type = op_type
        self.outputs = output_tensors
        self.output_dims = [tensor['dimensions'] for tensor in output_tensors]
        self.params = params if params else self.extract_attrs(src_op, param_info)
        self.inputs = self.set_static_tensor_to_param(input_tensors)
        self.axis_orders = dict()
        self.validate()

    @classmethod
    @abstractmethod
    def extract_attrs(cls, src_op, param_info):
        """
        The intention of this method is to extract param_info from a framework src_op and return
        a dictionary of Param objects, such that "attr_name": "Param". This must be implemented,
        as it is called during initialization
        :param src_op: Framework src_op
        :param param_info: Parameter info
        :return: A dictionary of Params
        """

    @abstractmethod
    def infer_output_shapes(self, node, **kwargs):
        """
        This method receives a framework node and returns the output shapes
        :param node:
        :param kwargs:
        :return: a list of lists which contain output dimensions for each output tensor
        """

    @abstractmethod
    def set_tensor_data_types(self, node):
        """
        Sets the datatype for each input and output tensor based on the operation instance
        against its tensor info allowed types
        :param node : The source framework node
        :raises An backend defined error if datatype cannot be set.
        :returns
        """

    def set_output_dims(self, src_op, output_tensor_info: List[CustomTensorInfo], model):
        """
        Creates an output tensor from a tensor info object. An output tensor must have a valid
        dimension, or the shape of each output tensor must be determinable.
        :param src_op: The  framework op as defined in the model
        :param output_tensor_info: A list of output tensor infos
        :param model: The framework model
        :return: An output tensor, which is a tensor info with a valid dimension field.
        """
        output_dims = [tensor_info['dimensions'] for tensor_info in output_tensor_info]
        if any(not dim for dim in output_dims):
            output_dims = self.infer_output_shapes(src_op, model=model,
                                                   perform_shape_inference=True)
        for i, tensor_info in enumerate(output_tensor_info):
            tensor_info.dimensions = output_dims[i]
        return output_tensor_info

    @abstractmethod
    def set_static_tensor_to_param(self, tensors):
        """
        Sets a static tensor to a param. This method is called by the base class,
        meaning instances of this class are expected to have static tensors become params. This
        method takes a single tensor, and changes it to a param object. Note that a static tensor
        must have a data field defined.
        :param tensors: The tensor to be made a param.
        """

    def set_axis_orders(self, tensors, tensor_layouts: Dict[str, str]):
        """
        Returns the corresponding IR axis order from the tensor layouts defined by each tensor
        passed as argument.
        :param tensor_layouts: The list of possible tensor layouts
        :param tensors: The list of TensorInfo objects
        """
        # first check all tensors have the same tensor layout,
        # otherwise implicit permute may cause issues,
        # user should be aware
        layouts = [tensor.layout for tensor in tensors]
        if not check_all_equal(layouts):
            log_warning(" Distinct tensor layouts for the same tensor type may introduce implicit "
                        "permutes for each tensor into the IR during conversion. "
                        "The following tensors were found to be distinct: {}"
                        , zip([str(tensor.name) for tensor in tensors], layouts))

        for tensor in tensors:
            # need to preset this here, because we may need an implicit permute
            if tensor.layout == tensor_layouts['NCHW']:
                self.axis_orders[tensor.name] = 'NCS'
            elif tensor.layout == tensor_layouts['NHWC']:
                self.axis_orders[tensor.name] = 'NSC'
            elif tensor.layout in tensor_layouts.values():
                # user has selected one of the runtime specific layouts, setting non-trivial
                # to avoid axis-tracking
                self.axis_orders[tensor.name] = 'NONTRIVIAL'
            else:
                # let IR axis tracking determine it
                self.axis_orders[tensor.name] = 'NOT_YET_DEFINED'

    def validate(self, *args, **kwargs):
        """
        The intention of this method is to call validation on the input tensors and params
        provided in the src_op, in comparison with the input_tensor_info and param_info
        defined in a config spec, or included initialized with an instance of an Op.
        It is optional to implement this method. The default behavior of this method is to
        return nothing.

        :param args: optional arguments
        :param kwargs: optional keyword arguments
        :return: User defined behavior, or nothing if method is not overwritten.
        """


class CustomOpCollection(collections.MutableMapping):
    """
    Organizes a custom op based on its type into a mapping, whereby the key is the op_type and the
    values are a FIFO queue of all instances of that op_type that have been seen in an op_collection
    instance.
    """

    def __init__(self, **kwargs):
        super(CustomOpCollection, self).__init__()
        self.op_count = 0
        if kwargs:
            for name, value in kwargs:
                setattr(self, name, value)

    def __getitem__(self, name: str):
        if name not in self.__dict__:
            raise KeyError("Op: {} has not been registered with a Collection".format(name))
        return getattr(self, name)

    def __setitem__(self, op_type: str, op_queue: List[CustomOp]):
        """
        Sets an entry in a custom op collection, where the key is expected to be a Valid CustomOp
        type and the value is an op_queue, which contains all custom_ops of a certain type
        in the order that they appear in the model when its nodes are traversed using <graph>.node.

        e.x MyOpCollection[op_type] = [ArgmaxOp_1, ArgMaxOp_2.......]

        :param op_type: The type of the op
        :type op_queue: list of custom ops
        :return:
        """
        if not isinstance(op_queue, list):
            raise TypeError("Op queue argument must be a list. Got: {}".format(type(op_queue)))
        if not all([op.op_type == op_type for op in op_queue]):
            raise RuntimeError("Expected all provided custom Ops to be of the same type: {}, "
                               "instead got: {}",
                               op_type, [op.op_type != op_type for op in op_queue])

        for op in op_queue:
            if not isinstance(op, CustomOp):
                raise TypeError("Argument is not a valid Custom Op")
            elif op_type not in self.__dict__:
                setattr(self, op_type, [op])
                self.op_count += 1
            else:
                self.__dict__[op_type].append(op)

    def __delitem__(self, key):
        if hasattr(self, key):
            self.__dict__.pop(key)

    def __iter__(self):
        return iter(self)

    def __len__(self):
        return self.op_count

    def get_first_of(self, op_type: str):
        """
        Gets the first element of its internal queue for a given op_type, raises an IndexError if
        there are no ops left in the queue.

        :param op_type: The type of the op to be extracted
        :return:
        """
        op_queue = getattr(self, op_type)
        op = op_queue[0]
        self.__dict__[op_type].pop(0)
        return op

    def get_op_by_name(self, op_type: str, name: str):
        op_queue = getattr(self, op_type)
        ops = [op for op in op_queue if op.name == name]
        if not ops:
            raise LookupError("Custom op with name: {} was not found "
                              "in the Custom Op Collection".format(name))
        return ops[0]
