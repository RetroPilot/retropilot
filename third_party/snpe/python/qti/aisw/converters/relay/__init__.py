# ==============================================================================
#
#  Copyright (c) 2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

# TVM needs to be imported here first to introduce the module name "qti.tvm" replacement
# logic in python/tvm/__init__.py, so the following python files can directly use
# "import tvm" to import the TVM module
from qti import tvm