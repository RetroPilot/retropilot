# -*- mode: python -*-
# =============================================================================
#
#  Copyright (c) 2019-2022 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import copy
import numpy as np
from math import isclose
from .converter_utils import *
from . import code_to_message


def pads_symmetric(pads):
    num_dims = len(pads)//2
    for i in range(num_dims):
        if pads[i] != pads[i+num_dims]:
            return False
    return True


# pads: format expected is [x1_begin, x2_begin, ..., xn_begin, x1_end, x2_end, ..., xn_end]
def pads_righthanded(pads):
    num_dims = len(pads)//2
    for i in range(num_dims):
        if pads[i] != 0:
            return False
    # don't call all zeros right-handed
    return not all(x == 0 for x in pads)


def expand_shape(shapes_list):
    """
    Expands the smaller rank shapes to match the largest one
    :type input_shapes: list[list[int]]
    :rtype: list[list[int]]
    """
    max_rank = max(len(input_shape) for input_shape in shapes_list)
    # create a copy of input_shapes and then expand to avoid changing original input_shapes
    expanded_shapes = copy.deepcopy(shapes_list)
    for expanded_shape in expanded_shapes:
        while len(expanded_shape) < max_rank:
            expanded_shape.insert(0, 1)
    return expanded_shapes


def __broadcastable(shapes_: list) -> bool:
    """ Wrapper over broadcastable function to check if given shapes are broadcastable """

    for i in range(len(shapes_) - 1):
        if any(not broadcastable(shapes_[i], shapes_[j]) for j in range(i + 1, len(shapes_))):
            return False
    return True


def are_shapes_broadcastable(shapes_list: list, axis_order, align_channels=True) -> bool:
    """
        Helper function to check if input shapes are broadcastable by either permuting the shapes to IR or in src axis format.
        (e.g. input_shapes: [[64], [1,255,255,64]], result: True
              input_shapes: [[[1,64, 255, 255], [1,64,1,1]], result: True
        :param input_shapes: list of shapes to broadcast
        :param axis_order: the AxisOrder object of the src fw
        :param align_channels: This flag is only used when at least one input is 4D and has a Channel dimension.
                               If align_channels is True then permute the shapes to IR to align the channel
                               dimensions before broadcasting. Otherwise the original input shapes are checked if
                               broadcastable
        :return: boolean if broadcast is possible True otherwise False
    """

    if len(shapes_list) == 1:
        return True

    shapes_to_check = shapes_list[:]
    if align_channels:
        # check if original input shapes permuted to ir is broadcastable
        shapes_to_check = [axis_order.permute_shape_to_ir(shape) for shape in shapes_list]
    return __broadcastable(shapes_to_check)


def get_broadcasted_shape(shapes_list: list, axis_order, align_channels=True) -> list:
    """
    Helper function to compute and return broadcasted output shape by either permuting shapes to IR
    or by expanding shapes in src axis format if shapes differ in rank
    (e.g. input_shapes: [[64], [1,255,255,64]], result: [1,255,255,64]
          input_shapes: [[255], [1,64,255,255]], result: [1,64,255,255])
    :param input_shapes: list of shapes to broadcast
    :param axis_order: the AxisOrder object of the src fw
    :param align_channels: This flag is only used when at least one input is 4D and has a Channel dimension.
                           If align_channels is True then permute the shapes to IR to align the channel dimensions
                           before broadcasting. Otherwise the input shapes are expanded to match ranks before
                           checking for broadcasting
    :return: the broadcasted output shape
    :raises: ValueError if any shape in the input_shapes list are not broadcastable with eachother
    """

    def _get_output_shape(shapes_):
        # Uses numpy function to calculate shape, numpy throws shape mismatch errors
        inputs = [np.zeros(shape, dtype=np.byte) for shape in shapes_]
        try:
            output_shape_ = list(np.broadcast(*inputs).shape)
        except ValueError:
            raise ValueError("Shape mismatch, {} cannot be broadcast to a single shape".format(shapes_))

        return output_shape_

    if len(shapes_list) == 1:
        return shapes_list[0]

    shapes_to_check = shapes_list[:]
    if align_channels:
        shapes_to_check = [axis_order.permute_shape_to_ir(shape) for shape in shapes_list]
    else:
        shapes_to_check = expand_shape(shapes_to_check)

    if not __broadcastable(shapes_to_check):
        ValueError("Shape mismatch, {} cannot be broadcast to a single shape".format(shapes_list))

    broadcasted_shape = _get_output_shape(shapes_to_check)

    if align_channels:
        return axis_order.permute_shape_from_ir(broadcasted_shape)
    else:
        return broadcasted_shape


# -----------------------------
# Util for common chain replace
# -----------------------------
def chain_matched_eltwise_ops(graph, matched_node_list, op_class):
    """
    Replaces matched elementwise op nodes that have > 2 inputs with a chain of binary elementwise nodes
    :param graph: The IROpGraph object
    :param matched_node_list: The list of nodes that are product or sum elementwise ops with > 2 inputs
    :param op_class: A reference to the op_adapter class for the elementwise operation we're replacing
    """
    for nodes_tuple in matched_node_list:
        old_node = nodes_tuple[0]
        old_node_name = old_node.op.name
        old_node_inputs = old_node.input_names
        current_idx = graph.list_nodes().index(old_node)

        log_debug(code_to_message.get_debugging_message("DEBUG_ELEMENTWISEBINARY_CHAIN")
                  (old_node_name, old_node.op.type, old_node_inputs))

        # For an elementwise node with n inputs, n-2 new nodes must be added to the graph because there are n-1
        # operations and we are reusing the original node. The first node in the chain takes the first two inputs to
        # the original node. Subsequent nodes take their first input from the previous node in the chain and take their
        # second input from the next input of the original node. Inserted nodes are named according to their index in
        # the chain. Graph structures are maintained such that the graph is consistent with these modifications.
        new_nodes_names = []
        for i in range(len(old_node_inputs) - 2):
            if i == 0:
                new_node_inputs = [old_node_inputs[i], old_node_inputs[i+1]]
            else:
                new_node_inputs = [new_nodes_names[-1], old_node_inputs[i+1]]
            # Create new operation and add to the graph at the proper index
            new_nodes_names.append("%s_chain_%d" % (old_node_name, i))
            new_op = op_class(new_nodes_names[-1])
            graph.add(new_op, new_node_inputs, new_nodes_names[-1], idx=current_idx)
            current_idx += 1
        # Update the input names of the original node to be the last node in the chain
        old_node.input_names = [new_nodes_names[-1], old_node_inputs[-1]]

        # Cleanup the graph structures
        for name in set(old_node_inputs[:-1]):
            graph.remove_node_as_consumer(old_node, name)


# -------------------------------------------------------
# General
# -------------------------------------------------------
def expand_to_rank(shape, rank):
    """
    :type shape: list[int]
    :type rank: int
    :rtype: list[int]
    """
    result = shape[:]
    while len(result) < rank:
        result.insert(0, 1)
    return result


def to_list(val):
    if not val:
        return []
    if type(val) != list:
        return [val]
    return val


def broadcastable(shape1, shape2):
    """
    Checks if two shapes are can be broadcast into one another in the numpy sense.
    :param shape1: Shape of the data1
    :param shape2: Shape of the data2
    :return: boolean if broadcast is possible otherwise false
    """

    # loop backwards on both shapes and validate each index for broadcasting.
    # Eg: for [4,11,1,9] with [8,9], we only need to validate 8 and 9.
    for shape_idx1, shape_idx2 in zip(shape1[::-1], shape2[::-1]):
        if shape_idx1 != 1 and shape_idx2 != 1 and shape_idx1 != shape_idx2:
            return False
    return True


def compare_values(val1, val2, rtol=1.e-5, atol=1.e-6):
    """
    :param val1: type: (str, float, int, ndarray, list, set, dict)
    :param val2: type: (str, float, int, ndarray, list, set, dict)
    :param rtol: type: float The relative tolerance parameter to use if vals are numeric.
    :param atol: type: float The absolute tolerance parameter to use if vals are numeric.
    :return:
    """
    if type(val1) != type(val2):
        return False
    if type(val1) == list and type(val2) == list or \
            (type(val1) == set and type(val2) == set):
        if len(val1) != len(val2):
            return False
        return all([compare_values(i, j) for i, j in zip(val1, val2)])
    elif type(val1) == dict and type(val2) == dict:
        return all(val1_key in val2 and compare_values(val1_val, val2[val1_key])
                   for val1_key, val1_val in val1.items())
    elif type(val1 != val2) is np.ndarray:
        # Check if any value in arrays are different. Need shape check first since numpy.allclose
        # broadcasts if shapes are not equal
        return val1.shape == val2.shape and np.allclose(val1, val2, rtol=rtol, atol=atol)
    else:
        if type(val1) == float and type(val2) == float:
            # do tolerance comparison for floats
            return isclose(val1, val2, rel_tol=rtol, abs_tol=atol)
        return val1 == val2


def quantize_params(float_data, enc):
    saturate_float_data = np.clip(float_data, enc['min'], enc['max'])
    offset = -enc['offset'] if enc['offset'] > 0 else enc['offset']
    quantized_data = np.rint(saturate_float_data / enc['scale'] - offset)
    return quantized_data


def dequantize_params(quantized_data, enc):
    offset = -enc['offset'] if enc['offset'] > 0 else enc['offset']
    dequantized_data = enc['scale'] * (quantized_data + offset)
    return dequantized_data


def get_si_notation(n, total):
    if total > 0:
        percent = 100*float(n)/total
    else:
        percent = 0
    if n < 1000:
        return "%d (%.3g%%)" % (n, percent)
    elif n < 1000*1000:
        return '%dk (%.3g%%)' % (n/1000, percent)
    else:
        return '%dM (%.3g%%)' % (n/(1000*1000), percent)
