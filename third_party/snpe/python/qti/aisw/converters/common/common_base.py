# ==============================================================================
#
#  Copyright (c) 2020-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

'''
This file contains things common to all blocks of the Converter Stack.
It will contain things common to the Frontend and Backend.
'''

from abc import ABC
from qti.aisw.converters.common.utils.converter_utils import *
from qti.aisw.converters.common.utils import validation_utils
from qti.aisw.converters.common.utils.argparser_util import ArgParserWrapper, CustomHelpFormatter


class ConverterBase(ABC):

    class ArgParser(ArgParserWrapper):
        def __init__(self, **kwargs):
            super(ConverterBase.ArgParser, self).__init__(formatter_class=CustomHelpFormatter, **kwargs)

            self.add_required_argument("--input_network", "-i", type=str,
                                       action=validation_utils.validate_pathname_arg(must_exist=True),
                                       help="Path to the source framework model.")

            self.add_optional_argument("--debug", type=int, nargs='?', default=-1,
                                       help="Run the converter in debug mode.")

    def __init__(self, args):
        self.input_model_path = args.input_network
        self.debug = args.debug
        if not self.debug:
            # If --debug provided without any argument, enable all the debug modes upto log_debug3
            self.debug = 3
        setup_logging(self.debug)
