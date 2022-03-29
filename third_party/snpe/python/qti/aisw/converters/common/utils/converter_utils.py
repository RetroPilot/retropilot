# -*- mode: python -*-
# =============================================================================
#
#  Copyright (c) 2018-2020 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import logging
import sys
import argparse
import numpy
from collections import OrderedDict
from enum import Enum

# -----------------------------------------------------------------------------------------------------
#   Common Functions
# -----------------------------------------------------------------------------------------------------


def sanitize_args(args, args_to_ignore=[]):
    sanitized_args = []
    if isinstance(args, argparse.Namespace):
        args_dict = vars(args)
    elif isinstance(args, dict):
        args_dict = args
    else:
        raise TypeError("args needs to be of type argparse.Namespace or Dict")

    for k, v in list(sorted(args_dict.items())):
        if k in args_to_ignore:
            continue
        sanitized_args.append('{}={}'.format(k, v))
    return "{} {}".format(sys.argv[0].split('/')[-1], ' '.join(sanitized_args))


def get_string_from_txtfile(filename):
    if not filename:
        return ""
    if filename.endswith('.txt'):
        try:
            with open(filename, 'r') as myfile:
                file_data = myfile.read()
            return file_data
        except Exception as e:
            logger.error("Unable to open file %s: %s" % (filename, e))
            sys.exit(-1)
    else:
        logger.error("File %s: must be a text file." % filename)
        sys.exit(-1)


# Returns the i-th bit of val
def get_bit(val, i):
    return val & (1 << i)


# Returns the indices of all the set bits in val
def get_bits(val):
    count = 0
    bits = []
    while (val):
        if val & 1:
            bits.append(count)
        count += 1
        val >>= 1
    return bits


def uniques(values):
    """
    :type values: list
    :rtype: list
    """
    dictionary = OrderedDict()
    for v in values:
        if v not in dictionary:
            dictionary[v] = v
    return list(dictionary.keys())


# -----------------------------------------------------------------------------------------------------
#   Caffee Common Functions
# -----------------------------------------------------------------------------------------------------

class NpUtils(object):
    def blob2arr(self, blob):
        if hasattr(blob, "shape"):
            return numpy.ndarray(buffer=blob.data, shape=blob.shape, dtype=numpy.float32)
        else:
            # Caffe-Segnet fork doesn't have shape field exposed on blob.
            return numpy.ndarray(buffer=blob.data, shape=blob.data.shape, dtype=numpy.float32)


# -----------------------------------------------------------------------------------------------------
#   Logging
# -----------------------------------------------------------------------------------------------------
# @deprecated
# TODO: remove once cleanup of converters is done to use method below instead
logger = logging.getLogger(__name__)


def setUpLogger(verbose):
    formatter = '%(asctime)s - %(lineno)d - %(levelname)s - %(message)s'
    lvl = logging.INFO
    if verbose:
        lvl = logging.DEBUG
    logger = logging.getLogger()
    logger.setLevel(lvl)
    formatter = logging.Formatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(lvl)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

# --- end of deprecated ---

LOGGER = None
HANDLER = None
LOG_LEVEL = logging.INFO

# Custom Logging
logging.VERBOSE = LOG_LEVEL_VERBOSE = 5
logging.DEBUG_3 = DEBUG_LEVEL_IR_TO_BACKEND = 11
logging.DEBUG_2 = DEBUG_LEVEL_IR_OPTIMIZATION = 12
logging.DEBUG_1 = DEBUG_LEVEL_CONVERTER_TO_IR = 13

# add the custom log-levels
logging.addLevelName(DEBUG_LEVEL_IR_TO_BACKEND, "DEBUG_3")
logging.addLevelName(DEBUG_LEVEL_IR_OPTIMIZATION, "DEBUG_2")
logging.addLevelName(DEBUG_LEVEL_CONVERTER_TO_IR, "DEBUG_1")
logging.addLevelName(LOG_LEVEL_VERBOSE, "VERBOSE")


def setup_logging(debug_lvl, name=None):
    global LOGGER
    global HANDLER
    global LOG_LEVEL

    if debug_lvl == -1:  # --debug is not set
        LOG_LEVEL = logging.INFO
    elif debug_lvl == 0:  # --debug is set with no specific level. i.e: print every debug message.
        LOG_LEVEL = logging.DEBUG
    elif debug_lvl == 1:
        LOG_LEVEL = logging.DEBUG_1
    elif debug_lvl == 2:
        LOG_LEVEL = logging.DEBUG_2
    elif debug_lvl == 3:
        LOG_LEVEL = logging.DEBUG_3
    elif debug_lvl == 4:
        LOG_LEVEL = logging.VERBOSE
    else:
        log_assert("Unknown debug level provided. Got {}", debug_lvl)

    if LOGGER is None:
        LOGGER = logging.getLogger(name)
    LOGGER.setLevel(LOG_LEVEL)

    if HANDLER is None:
        formatter = logging.Formatter('%(asctime)s - %(lineno)d - %(levelname)s - %(message)s')
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)
        HANDLER = handler
    HANDLER.setLevel(LOG_LEVEL)


def log_assert(cond, msg, *args):
    assert cond, msg.format(*args)


def log_debug(msg, *args):
    if LOGGER:
        LOGGER.debug(msg.format(*args))


def log_debug1(msg, *args):
    def debug1(msg, *args, **kwargs):
        if LOGGER and LOGGER.isEnabledFor(logging.DEBUG_1):
            LOGGER._log(logging.DEBUG_1, msg, args, kwargs)
    debug1(msg.format(*args))


def log_debug2(msg, *args):
    def debug2(msg, *args, **kwargs):
        if LOGGER and LOGGER.isEnabledFor(logging.DEBUG_2):
            LOGGER._log(logging.DEBUG_2, msg, args, kwargs)
    debug2(msg.format(*args))


def log_debug3(msg, *args):
    def debug3(msg, *args, **kwargs):
        if LOGGER and LOGGER.isEnabledFor(logging.DEBUG_3):
            LOGGER._log(logging.DEBUG_3, msg, args, kwargs)
    debug3(msg.format(*args))


def log_verbose(msg, *args):
    def verbose(msg, *args, **kwargs):
        if LOGGER and LOGGER.isEnabledFor(logging.VERBOSE):
            LOGGER._log(logging.VERBOSE, msg, args, kwargs)
    verbose(msg.format(*args))


def log_error(msg, *args):
    if LOGGER:
        LOGGER.error(msg.format(*args))


def log_info(msg, *args):
    if LOGGER:
        LOGGER.info(msg.format(*args))


def log_warning(msg, *args):
    if LOGGER:
        LOGGER.warning(msg.format(*args))


def get_log_level():
    global LOG_LEVEL
    return LOG_LEVEL


def is_log_level_debug():
    global LOG_LEVEL
    return LOG_LEVEL == logging.DEBUG


def log_debug_msg_as_status(msg, *args):
    log_debug(msg + "."*50, *args)


# -----------------------------------------------------------------------------------------------------
#   Translation Helpers
# -----------------------------------------------------------------------------------------------------
"""
Following functions sanitize op/layer type names and attach converter type for registering translations
"""


def get_op_info(type_name):
    """Return the op name and version, if specified"""
    op_data = str(type_name).split('-')
    if len(op_data) > 1:
        return [op_data[0], int(op_data[1])]
    op_data.append(0)
    return op_data


def op_type(type_name):
    """Return the actual onnx op name"""
    data = get_op_info(type_name)
    return data[0]


def converter_type(type_name, src_converter):
    """Convert an src converter type name string to a namespaced format"""
    return src_converter + '_' + (op_type(type_name)).lower()

