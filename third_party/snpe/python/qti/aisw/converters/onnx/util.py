# ==============================================================================
#
#  Copyright (c) 2018-2022 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from functools import reduce
from operator import mul
import numpy as np

try:
    import onnx
    from onnx import defs, TensorProto
    from onnx.numpy_helper import to_array as extract_onnx_tensor
except:
    onnx = None # converter will throw before we try anything in here

from qti.aisw.converters.common.utils import code_to_message
from qti.aisw.converters.common.utils.converter_utils import *
from qti.aisw.converters.common.converter_ir.op_adapter import IRPaddingStrategies
from qti.aisw.converters.common.utils.translation_utils import broadcastable


code_to_enum = {'i': onnx.AttributeProto.INT,
                'f': onnx.AttributeProto.FLOAT,
                's': onnx.AttributeProto.STRING,
                't': onnx.AttributeProto.TENSOR,
                'g': onnx.AttributeProto.GRAPH,
                'li': onnx.AttributeProto.INTS,
                'lf': onnx.AttributeProto.FLOATS,
                'ls': onnx.AttributeProto.STRINGS,
                'lt': onnx.AttributeProto.TENSORS,
                'lg': onnx.AttributeProto.GRAPHS}

onnx_to_np_dtype = {
    # int types
    TensorProto.INT8: np.dtype('int8'),
    TensorProto.INT16: np.dtype('int16'),
    TensorProto.INT32: np.dtype('int32'),
    TensorProto.INT64: np.dtype('int64'),
    TensorProto.UINT8: np.dtype('uint8'),
    TensorProto.UINT16: np.dtype('uint16'),
    TensorProto.UINT32: np.dtype('uint32'),
    TensorProto.UINT64: np.dtype('uint64'),

    # float types
    TensorProto.FLOAT16: np.dtype('float16'),
    TensorProto.FLOAT: np.dtype('float32'),
    TensorProto.BOOL: np.dtype('bool_')
}

KNOWN_ATTRIBUTE_DEFAULTS = dict(dilations=[1, 1],
                                strides=[1, 1],
                                pads=[0, 0, 0, 0],
                                output_shape=[],
                                axes=[],
                                consumed_inputs=[],
                                kernel_shape=[])


def is_broadcast(onnx_op, graph=None):
    attrs = extract_attributes(onnx_op, [('axis', 'i', 0), ('broadcast', 'i', 0)])

    if graph is not None:
        # newer version of onnx(e.g version 7 of Mul or Add) do not have axis and broadcast attributes
        # hence another way to check would be to make sure all inputs to op are the same shape
        input_names = list(map(str, onnx_op.input))
        input_buffers_shape = []
        for name in input_names:
            if graph.has_buffer(name):
                input_buffers_shape.append(list(graph.get_buffer(name).shape))
            else:
                input_buffers_shape.append(list(graph.weights.fetch(name).shape))
        if any(shape != input_buffers_shape[0] for shape in input_buffers_shape):
            return True

    return attrs['axis'] != 0 or attrs['broadcast'] == 1


def assert_no_broadcast(onnx_op):
    log_assert(not is_broadcast(onnx_op),
               code_to_message.get_error_message("ERROR_BROADCAST_NOT_SUPPORTED")(onnx_op.name))


class NamedDict(dict):
    def __getattr__(self, key):
        return self[key]


def extract_initializer_tensor(initializer):
    return extract_onnx_tensor(initializer)


def extract_attributes(onnx_op, attr_infos=None, schema=None, validate=False):
    """Ensure the existence and extract well typed attributes from an onnx
    NodeProto.
    :param attr_infos: a list of attributes to extract in the form [(attr_name, attr_type, attr_value)]
    :param schema:   an op_schema object for the onnx_op
    :param validate:  an optional validator function that is registered with the schema
                     of the form:  validator(src_op, attr_name, attr_value)

    Each entry in attr_info should be either a 2- or 3-tuple.
    * The first element should be the string name of an attribute.
    * The second element should by a type code for the attribute corresponding to:
      - i for int attributes
      - f for float attributes
      - s for string attributes
      - t for tensor attributes (returned as a numpy array)
      - g for graph attributes
      - lx, where x is one of the preceding attribute type identifiers, for list valued attributes
    * The third element, if present, specifies a default value should the attribute not be present.
      If no default is specified, this function will thrown an error.

    The return object will have a named property for each attribute info."""
    onnx_attrs = {}
    if not attr_infos and schema:
        attr_infos = schema.attributes()

    for attr in onnx_op.attribute:
        onnx_attrs[attr.name] = attr
        if schema and not validate:
            if not schema.check_supported_attributes(str(attr.name)):
                log_warning(code_to_message.get_warning_message("WARNING_UNSUPPORTED_ATTRIBUTE")
                            (attr.name, onnx_op.op_type, onnx_op.input[0]))

    ret = NamedDict()
    for attr_info in attr_infos:
        name = attr_info[0]
        if not name in onnx_attrs:
            if len(attr_info) == 3:
                ret[name] = attr_info[2]
                continue
            else:
                try:
                    ret[name] = KNOWN_ATTRIBUTE_DEFAULTS[name]
                    continue
                except KeyError:
                    raise ValueError(code_to_message.get_error_message("ERROR_ATTRIBUTE_MISSING")(onnx_op.name, name))
        attr = onnx_attrs[name]
        code = attr_info[1]
        requested_type = code_to_enum[code]
        if attr.type != requested_type:
            msg = code_to_message.get_error_message("ERROR_ATTRIBUTE_WRONG_TYPE")(onnx_op.name,
                                                                                  name,
                                                                                  onnx.AttributeProto.AttributeType.Name(requested_type),
                                                                                  onnx.AttributeProto.AttributeType.Name(attr.type))
            raise TypeError(msg)
        value = extract_onnx_type(code, attr)

        if validate and schema:
            schema.validate_data_constraints(onnx_op)
            schema.get_validate_method("validate_attribute_values")(onnx_op, name, value)
        ret[name] = value

    return ret


def extract_onnx_type(code, attr):
    ret = ''
    if code == 'i':
        ret = int(attr.i)
    elif code == 'f':
        ret = float(attr.f)
    elif code == 's':
        ret = str((attr.s).decode('utf-8'))
    elif code == 'g':
        ret = attr.g
    elif code == 't':
        ret = extract_onnx_tensor(attr.t)
    elif code == 'li':
        ret = list(map(int, attr.ints))
    elif code == 'lf':
        ret = list(map(float, attr.floats))
    elif code == 'ls':
        ret = list(map(str, attr.strings))
    elif code == 'lg':
        ret = list(attr.graphs)
    elif code == 'lt':
        ret = list(map(extract_onnx_tensor, attr.tensors))
    return ret


def extract_padding_mode(auto_pad, node_name, ceil_mode=0, right_handed=False):
    if right_handed == True:
        return IRPaddingStrategies.PADDING_SIZE_EXPLICIT_RIGHTHANDED
    elif auto_pad == 'VALID':
        return IRPaddingStrategies.PADDING_SIZE_IMPLICIT_VALID
    elif auto_pad == 'SAME_UPPER':
        return IRPaddingStrategies.PADDING_SIZE_IMPLICIT_SAME_END
    elif auto_pad == 'SAME_LOWER':
        return IRPaddingStrategies.PADDING_SIZE_IMPLICIT_SAME_BEGIN
    elif ceil_mode != 0:
        return IRPaddingStrategies.PADDING_SIZE_EXPLICIT
    elif auto_pad == 'NOTSET':
        return IRPaddingStrategies.PADDING_SIZE_EXPLICIT_FLOOR
    else:
        raise ValueError(code_to_message.get_error_message("ERROR_PADDING_TYPE_UNSUPPORTED")(node_name, auto_pad))


def broadcast_to(data, new_shape):
    """
    Broadcasts data into a new shape if possible
    :param new_shape: shape to be broadcasted into
    :param data: data to be broadcasted
    :return: broadcasted data if possible or original data if not
    """
    if data.shape != new_shape and broadcastable(data.shape, new_shape):
        return numpy.broadcast_to(data, new_shape).astype(numpy.float32)
    return data


def product(nums):
    if len(nums) == 0:
        return 1
    else:
        return reduce(mul, nums)


def get_quant_info(zp):
    log_assert(isinstance(zp, np.ndarray),
               "Zero point is not a numpy array")
    if   zp.dtype == np.uint8:
        return False, 8, -int(zp[0])
    elif zp.dtype == np.int8:
        return True, 8, -2**(8-1)
    elif zp.dtype == np.uint32:
        return False, 32, -int(zp[0])
    elif zp.dtype == np.int32:
        return True, 32, -2**(32-1)
    else:
      raise ValueError("Unsupported zero point type: ",zp.dtype)


def get_encoding(name, scale, zp):
    is_symmetric, bw, new_zp = get_quant_info(zp)
    return { "name": name,
             "bw": bw,
             "min": ((np.iinfo(zp.dtype).min - zp) * scale)[0],
             "max": ((np.iinfo(zp.dtype).max - zp) * scale)[0],
             "scale": scale[0],
             "offset": new_zp,
             "is_symmetric": is_symmetric }


def downcast_dtype_64bit_to_32bit(tensor_name, tensor_dtype):
    numpy_dtype_downcast = {
        np.dtype('int64'): np.int32,
        np.dtype('uint64'): np.uint32,
        np.dtype('float64'): np.float32,
    }
    if tensor_dtype in numpy_dtype_downcast:
        prev_dtype = tensor_dtype
        tensor_dtype = numpy_dtype_downcast[tensor_dtype]
        log_debug1(code_to_message.get_debugging_message("DEBUG_DOWNCAST_TENSOR")
                   (prev_dtype, np.dtype(tensor_dtype), tensor_name))

    return tensor_dtype


class WeightData(object):
    def __init__(self, weights, was_scalar=False):
        """
        :param weights: weights from the network
        :param was_scalar: boolean to determine if original weight was initialized as scalar.
                           Since QuIR expects that all inputs are tensors, this information will be
                           helpful for any op specific usecases such as determining output_shape
        """
        self.weights = weights
        # Track if the weights have been retrieved for use in another layer
        # Weights can be provided in one of two ways: initializers or constant ops
        # Constant ops being used as weight providers are setup with the weights from
        # the start and thus don't need to retrieve weights from the weight provider
        # again. SNPE layers like Conv/Matmul/GEMM/etc store weights internally and
        # will attempt to retrieve the weights. The consumed field will track which
        # Constant ops are being used as weight providers so they can be pruned from
        # the network at the end
        self.consumed = False
        self.was_scalar = was_scalar


# ------------------------------------------------------------------------------
#   WeightProvider
# ------------------------------------------------------------------------------
class WeightProvider(object):
    def __init__(self, model):
        self.weight_map = {}
        for tensor in model.graph.initializer:
            onnx_tensor = extract_onnx_tensor(tensor)
            was_scalar = False
            if not tensor.dims:
                # tensor of dim 1 is empty array in onnx resulting in broken numpy array,
                # since unable to modify tensor.dims, reshaping to have numpy array proper config
                onnx_tensor = onnx_tensor.reshape(1)
                was_scalar = True
            self.weight_map[str(tensor.name)] = WeightData(onnx_tensor, was_scalar)

    def was_scalar(self, key):
        return self.weight_map[key].was_scalar

    def consumed(self, key):
        if not key in self.weight_map:
            return False
        return self.weight_map[key].consumed

    def fetch(self, *keys, **kwargs):
        ret = []
        # Prunable indicates whether the weights have been consumed in such a way as to
        # allow pruning of the node (eg Const ops that contain weights are consumed by
        # Conv/FC/etc and thus can be pruned from the network. Const ops that are inputs
        # to a node cannot
        consumed = kwargs.get('prunable', True)
        for key in keys:
            key = str(key)
            log_debug(code_to_message.get_debugging_message("DEBUG_RETRIEVE_WEIGHTS")(key))
            if key not in self.weight_map:
                raise KeyError(code_to_message.get_error_message("ERROR_WEIGHTS_MISSING_KEY")(key))
            self.weight_map[key].consumed = consumed
            if kwargs.get('dtype') is None:
                tensor_dtype = downcast_dtype_64bit_to_32bit(key, self.weight_map[key].weights.dtype)
            else:
                # Assumes downcasting of provided dtype, if required, will be handled by fetch caller
                tensor_dtype = kwargs.get('dtype')
            # Explicitly copy the data so if later ops modify it, the original data remains intact
            # may need to be changed to user data types
            ret.append(numpy.require(self.weight_map[key].weights.copy(), dtype=tensor_dtype))
        if len(ret) == 1:
            return ret[0]
        else:
            return ret

    def has(self, key):
        return key in self.weight_map

    def type(self, key):
        if key not in self.weight_map:
            raise KeyError(code_to_message.get_error_message("ERROR_WEIGHTS_MISSING_KEY")(key))
        return self.weight_map[key].weights.dtype

    def has_all(self, keys):
        return all(self.has(key) for key in keys)

    def insert(self, key, weights, was_scalar=False):
        log_debug("Inserting weights for {}, was_scalar:{}", key, was_scalar)
        self.weight_map[key] = WeightData(weights, was_scalar)


def get_type_dims_info(source, node_name):
    """
    :param source: structure to query for node_name's info
    :param node_name: the name of the node to query info for
    :return: (bool, elem_type, dims)
    """
    for info in source:
        if info.name == node_name:
            dims = [int(dim.dim_value) for dim in info.type.tensor_type.shape.dim]
            return True, info.type.tensor_type.elem_type, dims
    return False, None, None
