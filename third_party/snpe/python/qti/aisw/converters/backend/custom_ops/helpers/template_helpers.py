# ==============================================================================
#
#  Copyright (c) 2020-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from qti.aisw.converters.backend.custom_ops.helpers.udo_module_helpers import *


# ------------------------------------------------------------------------------
#   SNPE HTP Package Generator Template Helpers
# ------------------------------------------------------------------------------
def get_unique_datatypes(operator):
    all_datatypes = list()
    unique_datatypes = list()
    for output in operator.output:
        all_datatypes.extend(output.data_type)
    for input_ in operator.input:
        all_datatypes.extend(input_.data_type)
    for data_type in all_datatypes:
        if all_datatypes.count(data_type) == 1:
            unique_datatypes.append(data_type)
    return unique_datatypes


def _template_builder(operator):
    tensor_types = ["typename TensorType"]
    num_of_unique_datatypes = len(get_unique_datatypes(operator))
    if num_of_unique_datatypes >= 1:
        tensor_types.extend(["typename TensorType{}".format(str(idx)) for idx in range(1, num_of_unique_datatypes + 1)])
    return "template<{}>".format(','.join(tensor_types))


def get_tensor_type_mapping(operator, data_type, cur_idx: int, prefix):
    """
     This maps an operator and its data-types into a variable signature. The idea is that for each unique datatype
     in the operator input and output datatypes, there will be a corresponding TensorType template element.

     See usage in format output

    """
    if any(data_type in get_unique_datatypes(operator) for data_type in data_type):
        cur_idx = cur_idx + 1
        tensor_str = "TensorType{} &{},".format(str(cur_idx), prefix)
    else:
        if cur_idx == 0:
            tensor_str = "TensorType& {},".format(prefix)
        else:
            tensor_str = "TensorType{}& {},".format(str(cur_idx), prefix)
    return tensor_str


def get_hexnn_tensor_sig(operator,
                         func_tab_size,
                         cur_idx,
                         *,
                         output_only=False,
                         input_only=False):
    """
    Formats the input and output names that appear in the wrapper.cpp file. It creates variable signatures for
    all inputs and outputs belonging to an operator.

    :param input_only: Boolean control to return just the output strings if True
    :param output_only: Boolean control to return just the input strings if True
    :param operator: The name of the operator
    :param func_tab_size: A tab size for the function signature to ensure the variable signature is aligned
    :param cur_idx: The current index for the tensor type
    :return: output tensor signatures, input tensor signatures or both
    """
    out_strings = ""
    in_strings = ""

    def get_tensor_string(data_type,
                          prefix: str,
                          qualifier: str = "",
                          container_type: str = "",
                          container_value_type_qualifier=""):
        tensor_mapping = str(get_tensor_type_mapping(operator,
                                                     data_type, cur_idx,
                                                     prefix)).rstrip()
        if container_type:
            split_tensor_mapping = tensor_mapping.split("&")
            tensor_mapping = "{}<{} {}*>".format(container_type, container_value_type_qualifier,
                                                 split_tensor_mapping[0]) + "&" + split_tensor_mapping[1]

        return " " * func_tab_size + qualifier + " " + tensor_mapping + '\n'

    for idx, output in enumerate(operator.output):
        out_name = "out" if "out" in output.name else output.name
        prefix = "{}_{}".format(out_name, idx) if out_name == "out" else out_name
        out_strings = out_strings + get_tensor_string(output.data_type, prefix)
    if output_only:
        return out_strings
    for idx, input_ in enumerate(operator.input):
        in_name = "in" if "in" in input_.name else input_.name
        prefix = "{}_{}".format(in_name, idx) if in_name == "in" else in_name
        in_strings = in_strings + get_tensor_string(input_.data_type, prefix, qualifier="const")
    if input_only:
        return in_strings
# remove trailing commas
    if not (operator.tensor_param or operator.scalar_param):
        in_strings = in_strings.rstrip('\n,') + ')'
    return str(out_strings + in_strings)


def get_hexnn_param_sig(operator, func_tab_size):
    """
    Produces a parameter argument signature based on its datatype
    :param operator: The operator instance
    :param func_tab_size: The function signature tab size for far (note the signatures for inputs
    and outputs are computed previously)
    :return: Th string signature for all the params for the operator instance
    """
    tensor_param = ''
    if operator.tensor_param:
         for param in operator.tensor_param:
            dtype = "Tensor"
            tensor_param += " " * func_tab_size + "const {}& {},".format(dtype, param.name) + '\n'
    scalar_param = ''
    if operator.scalar_param:
        for param in operator.scalar_param:
            dtype = "Tensor"
            scalar_param += " " * func_tab_size + "const {}& {},".format(dtype, param.name) + '\n'

    param_string = scalar_param + tensor_param
    return param_string.rstrip('\n,') + ')'
