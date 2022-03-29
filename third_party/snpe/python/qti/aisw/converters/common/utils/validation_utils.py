# =============================================================================
#
#  Copyright (c) 2017-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import argparse
from . import io_utils

valid_processor_choices = ('snapdragon_801', 'snapdragon_820', 'snapdragon_835')
valid_runtime_choices = ('cpu', 'gpu', 'dsp')


class ValidateTargetArgs(argparse.Action):
    def __call__(self, parser, args, values, option_string=None):
        specified_runtime, specified_processor = values
        if specified_runtime not in valid_runtime_choices:
            raise ValueError('invalid runtime_target {s1!r}. Valid values are {s2}'.format(s1=specified_runtime,
                                                                                           s2=valid_runtime_choices)
                             )
        if specified_processor not in valid_processor_choices:
            raise ValueError('invalid processor_target {s1!r}. Valid values are {s2}'.format(s1=specified_processor,
                                                                                             s2=valid_processor_choices)
                             )
        setattr(args, self.dest, values)


def check_filename_encoding(filename):
    try:
        filename.encode('utf-8')
    except UnicodeEncodeError:
        raise ValueError("Converter expects string arguments to be UTF-8 encoded: %s" % filename)


# Validation for generic file, optional validation for file existing already
def validate_filename_arg(*, must_exist=False, create_missing_directory=False):
    class ValidateFilenameArg(argparse.Action):
        def __call__(self, parser, args, value, option_string=None):
            check_filename_encoding(value)
            io_utils.check_validity(value, create_missing_directory=create_missing_directory, must_exist=must_exist)
            setattr(args, self.dest, value)

    return ValidateFilenameArg


# Validation for the path of generic file or folder
def validate_pathname_arg(*, must_exist=False):
    class ValidatePathnameArg(argparse.Action):
        def __call__(self, parser, args, value, option_string=None):
            check_filename_encoding(value)
            io_utils.check_validity(value, is_path=True, must_exist=must_exist)
            setattr(args, self.dest, value)

    return ValidatePathnameArg


def check_xml():
    class ValidateXmlFileArgs(argparse.Action):
        def __call__(self, parser, args, values, option_string=None):
            for value in values:
                io_utils.check_validity(value, extensions=[".xml"])
            if hasattr(args, self.dest) and getattr(args, self.dest) is not None:
                old_values = getattr(args, self.dest)
                values.extend(old_values)
            setattr(args, self.dest, values)

    return ValidateXmlFileArgs

def check_json():
    class ValidateXmlFileArgs(argparse.Action):
        def __call__(self, parser, args, values, option_string=None):
            for value in values:
                io_utils.check_validity(value, extensions=[".json"])
            if hasattr(args, self.dest) and getattr(args, self.dest) is not None:
                old_values = getattr(args, self.dest)
                values.extend(old_values)
            setattr(args, self.dest, values)
    return ValidateXmlFileArgs

