# ==============================================================================
#
#  Copyright (c) 2020 - 2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
from qti.aisw.converters.backend.custom_ops import snpe_udo_config
from qti.aisw.converters.common.utils.argparser_util import *


class AutogenParser(ArgParserWrapper):
    def __init__(self):
        super(AutogenParser, self).__init__(description="This tool generates a "
                                                        "UDO (User Defined Operation) package using a "
                                                        "required user provided config file.")
        # add description
        self.add_required_argument("--config_path", '-p', help="The path to your config file that defines your UDO. "
                                                               "Please see <udo/examples> for an example")
        self.add_optional_argument("--debug", action="store_true", help="Returns debugging information from generating"
                                                                        " the package")
        self.add_optional_argument("--output_path", "-o",  help="Path where the package should be saved")
        self.add_optional_argument("--ignore_includes", "-c", action="store_false",
                                   help=argparse.SUPPRESS)
        self.add_optional_argument("-f", "--force-generation", action="store_true",
                                   help="This option will delete the entire existing package "
                                        "Note appropriate file permissions must be set to use this option.")
        self.add_optional_argument("--gen_cmakelists", action="store_true", help=argparse.SUPPRESS)


class AutoGenerator(snpe_udo_config.UdoGenerator):
    def __init__(self):
        snpe_udo_config.UdoGenerator.__init__(self)
        self.parser = AutogenParser()
