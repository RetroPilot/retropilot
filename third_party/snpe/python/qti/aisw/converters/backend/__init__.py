# ==============================================================================
#
#  Copyright (c) 2019-2020 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

# @deprecated
# to allow for backward compatibility adding this import also at top-level so that
# it is possible to do <from qti.aisw import modeltools>
# moving forward the signature to use will be <from qti.aisw.dlc_utils import modeltools>
try:
    from qti.aisw.converters.backend.ir_to_dlc import DLCBackend as NativeBackend
    from qti.aisw.dlc_utils import libDlModelToolsPy3 as modeltools
    from qti.aisw.dlc_utils import libDlContainerPy3 as dlcontainer
except ImportError as ie:
    raise ie
