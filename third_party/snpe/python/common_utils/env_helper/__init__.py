#
# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
import platform
if 'Microsoft'in platform.platform() or 'Windows' in platform.platform():
    from .windows_env import WindowsEnvHelper as env_helper
else:
    from .android_env import AndroidEnvHelper as env_helper
