# ==============================================================================
#
#  Copyright (c) 2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from qti.aisw.converters.relay.relay_to_ir import RelayConverterFrontend
from qti.aisw.converters.relay.importers.pytorch_importer import PyTorchImporter


class PyTorchConverterFrontend(RelayConverterFrontend):
    class ArgParser(RelayConverterFrontend.ArgParser):
        def __init__(self, **kwargs):
            super(PyTorchConverterFrontend.ArgParser, self).__init__(conflict_handler='resolve',
                                                                    parents=[PyTorchImporter.ArgParser()],
                                                                    **kwargs)

    def __init__(self, args, **kwargs):
        super(PyTorchConverterFrontend, self).__init__(args,
                                                      importer=PyTorchImporter(args),
                                                      **kwargs)
