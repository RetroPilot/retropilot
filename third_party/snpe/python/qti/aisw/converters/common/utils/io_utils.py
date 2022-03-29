# =============================================================================
#
#  Copyright (c) 2017-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import os
from qti.aisw.converters.common.utils.converter_utils import *


def check_validity(resource, *, is_path=False, is_directory=False, must_exist=True, create_missing_directory=False,
                   extensions=[]):
    resource_path = os.path.abspath(resource)
    # Check to see if output_path is what needs to be validated
    if create_missing_directory:
        log_debug("Checking if output_path directory exists")
        # Split the path in head and tail pair
        output_path_head_tail = os.path.split(resource_path)
        # Now check if directory exists, if not then create the directory
        if not os.path.exists(output_path_head_tail[0]):
            try:
                log_debug("Creating output_path directory: " + str(output_path_head_tail[0]))
                os.makedirs(output_path_head_tail[0])
            except OSError as error:
                raise OSError(str(error) + '\n{} is not a valid directory path'.format(str(output_path_head_tail[0])))
    if is_path and os.path.isdir(resource_path):
        # For the case that resource path can be either dir or file
        is_directory = True
    if must_exist and not os.path.exists(resource_path):
        raise IOError('{} does not exist'.format(str(resource)))
    elif not is_directory:
        if must_exist and os.path.exists(resource_path) and not os.path.isfile(resource_path):
            raise IOError('{} is not a valid {} file'.format(str(resource), str(extensions)))
        if extensions and \
                not any([os.path.splitext(resource_path)[1] == str(extension) for extension in extensions]):
            raise IOError("{} is not a valid file extension: {}".format(resource, str(extensions)))
    else:
        if os.path.exists(resource_path) and not os.path.isdir(resource_path):
            raise IOError('{} is not a valid directory'.format(str(resource)))
        elif extensions:
            raise ValueError("Directories cannot have a file extension".format(str(resource)))