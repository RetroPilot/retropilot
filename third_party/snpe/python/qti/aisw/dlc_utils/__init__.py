# ==============================================================================
#
#  Copyright (c) 2019-2020 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
import sys

try:
    from . import libDlModelToolsPy3 as modeltools
    from . import libDlContainerPy3 as dlcontainer
except ImportError as ie1:
    try:
        import libDlModelToolsPy3 as modeltools
        import libDlContainerPy3 as dlcontainer
    except ImportError:
        raise ie1
