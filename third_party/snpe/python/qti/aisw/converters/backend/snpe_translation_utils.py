# =============================================================================
#
#  Copyright (c) 2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from qti.aisw.converters.common.converter_ir.op_adapter import IRPaddingStrategies
from qti.aisw.converters.common.utils.converter_utils import log_assert, log_debug3
from qti.aisw.converters.common.utils import code_to_message
from qti.aisw.converters.common.utils.translation_utils import pads_symmetric, pads_righthanded


def validate_snpe_padding(node):
    supported_strategies = [IRPaddingStrategies.PADDING_SIZE_IMPLICIT_VALID,
                            IRPaddingStrategies.PADDING_SIZE_IMPLICIT_SAME_END,
                            IRPaddingStrategies.PADDING_SIZE_EXPLICIT,
                            IRPaddingStrategies.PADDING_SIZE_EXPLICIT_FLOOR,
                            IRPaddingStrategies.PADDING_SIZE_EXPLICIT_RIGHTHANDED,
                            ]
    padding_size_strategy = node.op.padding_size_strategy
    log_assert(padding_size_strategy in supported_strategies,
               "Unsupported SNPE Padding Strategy {}".format(padding_size_strategy))

    # For explicit strategy SNPE only allows symmetric or right handed
    pads = [node.op.pady_before, node.op.padx_before, node.op.pady_after, node.op.padx_after]
    if padding_size_strategy == IRPaddingStrategies.PADDING_SIZE_EXPLICIT or \
       padding_size_strategy == IRPaddingStrategies.PADDING_SIZE_EXPLICIT_FLOOR:
        log_assert(pads_symmetric(pads), code_to_message.get_error_message("ERROR_ASYMMETRIC_PADS_VALUES"))
    elif padding_size_strategy == IRPaddingStrategies.PADDING_SIZE_EXPLICIT_RIGHTHANDED:
        log_assert(pads_righthanded(pads), code_to_message.get_error_message("ERROR_ASYMMETRIC_PADS_VALUES"))


def adjust_padding_strategy(translation, node, input_w, input_h, kernel_w, kernel_h):
    explicit_padding_strategies = [
        IRPaddingStrategies.PADDING_SIZE_EXPLICIT,
        IRPaddingStrategies.PADDING_SIZE_EXPLICIT_FLOOR
    ]
    padding_strategy = node.op.padding_size_strategy

    # pads needs to be [x_begin, y_begin, x_end, y_end] for this util function
    pads_list = [node.op.padx_before, node.op.pady_before, node.op.padx_after, node.op.pady_after]

    if padding_strategy in explicit_padding_strategies and not pads_symmetric(pads_list):
        pad_same_begin = translation.calc_same_padding(node, input_w, input_h,
                                                       kernel_w=kernel_w, kernel_h=kernel_h, same_begin=True)
        pad_same_end = translation.calc_same_padding(node, input_w, input_h,
                                                     kernel_w=kernel_w, kernel_h=kernel_h, same_begin=False)
        if pads_righthanded(pads_list) and padding_strategy == IRPaddingStrategies.PADDING_SIZE_EXPLICIT_FLOOR:
            # Override padding strategy for Righthanded Padding
            padding_strategy = IRPaddingStrategies.PADDING_SIZE_EXPLICIT_RIGHTHANDED
            log_debug3("Overriding padding strategy to Righthanded for Node {}".format(node.op.name))
        elif pads_list == pad_same_begin:
            padding_strategy = IRPaddingStrategies.PADDING_SIZE_IMPLICIT_SAME_BEGIN
            log_debug3("Overriding padding strategy to SAME_BEGIN for Node {}".format(node.op.name))
        elif pads_list == pad_same_end:
            padding_strategy = IRPaddingStrategies.PADDING_SIZE_IMPLICIT_SAME_END
            log_debug3("Overriding padding strategy to SAME_END for Node {}".format(node.op.name))
        else:
            raise ValueError("Unsupported Asymmetric padding values {} for node {}".format(pads_list, node.op.name))

    return padding_strategy


def get_pad_size(padding_size_strategy, padx_before, padx_after, pady_before, pady_after):
    """
    Adjust pad amounts based on padding strategy
    return: (pad_x, pad_y) after applying padding strategy
    """
    if padding_size_strategy == IRPaddingStrategies.PADDING_SIZE_EXPLICIT_RIGHTHANDED:
        pad_x = padx_after
        pad_y = pady_after
    else:
        pad_x = padx_before
        pad_y = pady_before

    return pad_x, pad_y
