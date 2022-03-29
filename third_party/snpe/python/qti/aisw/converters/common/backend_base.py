# ==============================================================================
#
#  Copyright (c) 2020 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from abc import abstractmethod, ABC
from qti.aisw.converters.common.utils import validation_utils, converter_utils
from qti.aisw.converters.common.converter_ir.translation import Translation
from qti.aisw.converters.common.common_base import ConverterBase


class ConverterBackend(ConverterBase, ABC):
    class ArgParser(ConverterBase.ArgParser):
        def __init__(self, **kwargs):
            super(ConverterBackend.ArgParser, self).__init__(**kwargs)
            self.add_optional_argument('-o', '--output_path', type=str,
                                       action=validation_utils.validate_filename_arg(must_exist=False,
                                                                                     create_missing_directory=True),
                                       help='Path where the converted Output model should be saved.If not '
                                            'specified, the converter model will be written to a file with same '
                                            'name as the input model')
            self.add_optional_argument('--copyright_file', type=str,
                                       action=validation_utils.validate_filename_arg(must_exist=True),
                                       help='Path to copyright file. If provided, the content of the file will '
                                            'be added to the output model.')

    def __init__(self, args):
        super(ConverterBackend, self).__init__(args)
        self.output_model_path = args.output_path
        self.converter_command = converter_utils.sanitize_args(args,
                                                               args_to_ignore=['input_network', 'i', 'output_path',
                                                                               'o'])
        self.copyright_str = converter_utils.get_string_from_txtfile(args.copyright_file)

    @abstractmethod
    def save(self, ir_graph):
        return NotImplementedError("save() not implemented for {}".format(self.__class__.__name__))


class BackendTranslationBase(Translation, ABC):
    # translation method keys
    ADD_OP_TO_BACKEND = 'add_op_to_backend'

    def __init__(self):
        super(BackendTranslationBase, self).__init__()
        self.register_method(self.ADD_OP_TO_BACKEND, self.add_op_to_backend)

    @abstractmethod
    def add_op_to_backend(self, node, ir_graph, backend, **kwargs):
        return NotImplementedError("add_op_to_backend not implemented for {}".format(self.__class__.__name__))
