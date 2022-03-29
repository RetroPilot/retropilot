# ==============================================================================
#
#  Copyright (c) 2019-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
import sys
import numpy

if sys.version_info[0] > 2:
    string_types = [str, bytes]
else:
    string_types = [unicode, str]


# ------------------------------------------------------------------------------
#   Udo Misc Helper Methods
# ------------------------------------------------------------------------------
def title_case(string):
    if not isoneofinstance(string, string_types):
        raise TypeError("Cannot change non string object to camel case")
    else:
        lower_case = string.lower()
        return lower_case.title()


def is_static_tensor(tensor):
    return tensor.static


def isoneofinstance(object, instances):
    if iter(instances):
        for instance in instances:
            if isinstance(object, instance):
                return True
        return False


def reverse_dict(orig_dict):
    return {v: k for k, v in orig_dict.items()}


def set_data_types_to_internal(data_types):
    if any(data_type == "ALL_VALID" for data_type in data_types):
        return list(SnpeUdoConstants.SNPE_UDO_DATATYPES.keys())
    else:
        return data_types


def get_per_core_data_types(per_core_data_types):
    core_types = list(per_core_data_types.keys())
    internal_data_types = set_data_types_to_internal(list(per_core_data_types.values()))
    assert all(core_type in SnpeUdoConstants.SNPE_UDO_CORETYPES for core_type in core_types), \
        "ERROR: Unsupported UDO Coretype"
    assert all(data_type in SnpeUdoConstants.SNPE_UDO_DATATYPES for data_type in internal_data_types), \
        "ERROR: Unsupported UDO Datatype"
    core_types = get_internal_core_types(core_types)
    data_types = [SnpeUdoConstants.SNPE_UDO_DATATYPES[data_type] for data_type in internal_data_types]
    return dict(zip(core_types, data_types))


def check_all_equal(iterable):
    return not iterable or iterable.count(iterable[0]) == len(iterable)


# ------------------------------------------------------------------------------
#   Udo Module Helper Functions and Classes
# ------------------------------------------------------------------------------
def udo_property(name):
    attr_name = '_' + name

    @property
    def prop(self):
        return getattr(self, attr_name, list())

    @prop.setter
    def prop(self, value):
        raise ValueError('Cannot set this field')

    return prop


class SnpeUdoConstants(object):
    SNPE_UDO_CORETYPES = {'CPU': 'SNPE_UDO_CORETYPE_CPU',
                          'GPU': 'SNPE_UDO_CORETYPE_GPU',
                          'DSP': 'SNPE_UDO_CORETYPE_DSP'}

    SNPE_UDO_TENSOR_LAYOUT = {'NHWC': 'SNPE_UDO_LAYOUT_NHWC',
                              'NCHW': 'SNPE_UDO_LAYOUT_NCHW',
                              'NDCHW': 'SNPE_UDO_LAYOUT_NDCHW',
                              'GPU_OPTIMAL1': 'SNPE_UDO_LAYOUT_GPU_OPTIMAL1',
                              'GPU_OPTIMAL2': 'SNPE_UDO_LAYOUT_GPU_OPTIMAL2',
                              'DSP_OPTIMAL1': 'SNPE_UDO_LAYOUT_DSP_OPTIMAL1',
                              'DSP_OPTIMAL2': 'SNPE_UDO_LAYOUT_DSP_OPTIMAL2',
                              'NONTRIVIAL': 'SNPE_UDO_LAYOUT_LAST'}

    SNPE_UDO_DATATYPES = {'FLOAT_16': 'SNPE_UDO_DATATYPE_FLOAT_16',
                          'FLOAT_32': 'SNPE_UDO_DATATYPE_FLOAT_32',
                          'FIXED_4': 'SNPE_UDO_DATATYPE_FIXED_4',
                          'FIXED_8': 'SNPE_UDO_DATATYPE_FIXED_8',
                          'FIXED_16': 'SNPE_UDO_DATATYPE_FIXED_16',
                          'UINT_8': 'SNPE_UDO_DATATYPE_UINT_8',
                          'UINT_16': 'SNPE_UDO_DATATYPE_UINT_16',
                          'UINT_32': 'SNPE_UDO_DATATYPE_UINT_32',
                          'INT_32': 'SNPE_UDO_DATATYPE_INT_32',
                          'STRING': 'SNPE_UDO_DATATYPE_UINT_8'}

    SNPE_CALCULATION_TYPES = {
        'CPU': 'SNPE_UDO_DATATYPE_FLOAT_16 | SNPE_UDO_DATATYPE_FLOAT_32',
        'GPU': 'SNPE_UDO_DATATYPE_FLOAT_16 | SNPE_UDO_DATATYPE_FLOAT_32',
        'DSP': 'SNPE_UDO_DATATYPE_INT_8 | SNPE_UDO_DATATYPE_INT_16'}

    SNPE_UDO_QUANT_TYPES = {'TF': 'SNPE_UDO_QUANTIZATION_TF',
                            'SKIP': 'SNPE_UDO_QUANTIZATION_NONE',
                            'QMN': 'SNPE_UDO_QUANTIZATION_QMN'}

    snpe_udo_coretypes = reverse_dict(SNPE_UDO_CORETYPES)


dtype_to_snpe_udo = {
    # int types
    numpy.dtype('int8'): "SNPE_UDO_DATATYPE_INT_8",
    numpy.dtype('int16'): "SNPE_UDO_DATATYPE_INT_16",
    numpy.dtype('int32'): "SNPE_UDO_DATATYPE_INT_32",
    numpy.dtype('int64'): "SNPE_UDO_DATATYPE_INT_64",
    numpy.dtype('uint8'): "SNPE_UDO_DATATYPE_UINT_8",
    numpy.dtype('uint16'): "SNPE_UDO_DATATYPE_UINT_16",
    numpy.dtype('uint32'): "SNPE_UDO_DATATYPE_UINT_32",
    numpy.dtype('uint64'): "SNPE_UDO_DATATYPE_UINT_64",

    # float types
    numpy.dtype('float16'): "SNPE_UDO_DATATYPE_FLOAT_16",
    numpy.dtype('float32'): "SNPE_UDO_DATATYPE_FLOAT_32",

    # bool type
    numpy.dtype('bool'): "SNPE_UDO_DATATYPE_BOOL_8",

    # inbuilt types
    int: "SNPE_UDO_DATATYPE_INT_32",
    float: "SNPE_UDO_DATATYPE_FLOAT_32",
    str: "SNPE_UDO_DATATYPE_UINT_8",
    bool: "SNPE_UDO_DATATYPE_BOOL_8"
}

def is_quant_type(snpe_type):
    return 'SNPE_UDO_DATATYPE_UINT' in snpe_type

def get_np_type_from_backend_type(snpe_type):
    if not snpe_type in SnpeUdoConstants.SNPE_UDO_DATATYPES.values():
        raise TypeError("Unknown SNPE UDO datatype conversion requested: {}".format(snpe_type))
    return reverse_dict(dtype_to_snpe_udo)[snpe_type]


def get_snpe_type(data):
    dtype = type(data) if not isinstance(data, numpy.ndarray) else data.dtype
    if isinstance(data, (tuple, list)):
        dtypes = [type(data_elem) for data_elem in data]
        if check_all_equal(dtypes):
            dtype = dtypes[0]
        else:
            # extremely unlikely, but we'll check anyway
            raise TypeError("Data value is an iterator with inconsistent types: {}".format(dtypes))
    return dtype_to_snpe_udo[dtype]


def convert_to_backend_type_from_numpy(dtype):
    if dtype not in dtype_to_snpe_udo:
        dtype = numpy.random.randn(1).astype(dtype).dtype
    return dtype_to_snpe_udo[dtype]


def get_internal_dtype(data, op_attr):
    try:
        candidate_type = get_snpe_type(data)
    except KeyError:
        src_type = type(data) if not isinstance(data, numpy.ndarray) else data.dtype
        raise KeyError("The provided data_type: {} is not a valid snpe_type".format(src_type))

    if op_attr.allowed_data_types and candidate_type not in op_attr.allowed_data_types:
        src_type = type(data) if not isinstance(data, numpy.ndarray) else data.dtype
        raise TypeError(
            "The provided datatype: {} is not a valid datatype defined for: {}. Expected one of {}"
            .format(src_type, op_attr.name, op_attr.allowed_data_types))
    return candidate_type


def get_internal_tensor_layout(layout, tensor_name):
    if layout in SnpeUdoConstants.SNPE_UDO_TENSOR_LAYOUT:
        return SnpeUdoConstants.SNPE_UDO_TENSOR_LAYOUT[layout]
    raise TypeError("Invalid layout: {} does not have "
                    "a mapping for tensor_name: {}".format(layout, tensor_name))


def get_internal_data_type(data_type, tensor_name):
    if data_type in SnpeUdoConstants.SNPE_UDO_DATATYPES:
        return SnpeUdoConstants.SNPE_UDO_DATATYPES[data_type]
    raise TypeError("Invalid data_type: {} does not have "
                    "a mapping for tensor_name: {}".format(data_type, tensor_name))


def get_internal_core_types(core_types: list, name=""):
    internal_core_types = []
    if not isinstance(core_types, list):
        raise TypeError("Error in {} spec: "
                        "Expected {} for core_type instead got {} "
                        .format(name, list, type(core_types)))
    for core_type in core_types:
        if core_type in SnpeUdoConstants.SNPE_UDO_CORETYPES:
            internal_core_types.append(SnpeUdoConstants.SNPE_UDO_CORETYPES[core_type])
        else:
            raise TypeError("Invalid core_type: {} does not have "
                            "a mapping for: {}".format(core_type, name))
    return internal_core_types
