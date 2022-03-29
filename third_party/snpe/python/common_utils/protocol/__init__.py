#
# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
import platform
if 'Microsoft'in platform.platform() or 'Windows' in platform.platform():
    from .tshell import Tshell as protocol
else:
    from .adb import Adb as protocol
