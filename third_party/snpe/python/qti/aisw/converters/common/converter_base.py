# ==============================================================================
#
#  Copyright (c) 2019-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================


import traceback
from abc import abstractmethod, ABC
import qti.aisw.converters.common.converter_ir.op_graph as op_graph
from qti.aisw.converters.common.converter_ir.axis_tracker import AxisOrder
from qti.aisw.converters.common.converter_ir.op_policies import ConversionNamePolicy
from qti.aisw.converters.common.utils.converter_utils import *
from qti.aisw.converters.common.common_base import ConverterBase


class ConverterFrontend(ConverterBase, ABC):

    class ArgParser(ConverterBase.ArgParser):
        def __init__(self, **kwargs):
            super(ConverterFrontend.ArgParser, self).__init__(**kwargs)
            self.add_optional_argument('--out_node', '--out_name', type=str, dest='out_names', action='append', default=[],
                                       help="Name of the graph\'s output Tensor Names. Multiple output names "
                                            "should be provided separately like: \n"
                                            "    --out_name out_1 --out_name out_2")
            self.add_optional_argument('--input_type', "-t", nargs=2, action='append',
                                       help='Type of data expected by each input op/layer. Type for each input '
                                            'is |default| if not specified. For example: "data" image.Note that '
                                            'the quotes should always be included in order to handle special '
                                            'characters, spaces,etc. For multiple inputs specify multiple '
                                            '--input_type on the command line.\n'
                                            'Eg: \n'
                                            '   --input_type "data1" image --input_type "data2" opaque \n'
                                            'These options get used by DSP runtime and following descriptions '
                                            'state how input will be handled for each option.\n'
                                            'Image: \n'
                                            'Input is float between 0-255 and the input\'s mean is 0.0f '
                                            'and the input\'s max is 255.0f. We will cast the float to uint8ts '
                                            'and pass the uint8ts to the DSP. \n'
                                            'Default: \n'
                                            'Pass the input as floats to the dsp directly and the DSP '
                                            'will quantize it.\n'
                                            'Opaque: \n'
                                            'Assumes input is float because the consumer layer(i.e next '
                                            'layer) requires it as float, therefore it won\'t be quantized.\n'
                                            'Choices supported:\n   ' + '\n   '.join(op_graph.InputType.
                                                                                     get_supported_types()),
                                       metavar=('INPUT_NAME', 'INPUT_TYPE'), default=[])
            self.add_optional_argument('--input_dtype', nargs=2, action='append',
                                       metavar=('INPUT_NAME', 'INPUT_DTYPE'), default=[],
                                       help="The names and datatype of the network input layers specified "
                                            "in the format [input_name datatype], "
                                            "for example: \n"
                                            "    'data' 'float32'\n"
                                            "Default is float32 if not specified\n"
                                            "Note that the quotes should always be included in order to handle"
                                            "special characters, spaces, etc. \n"
                                            "For multiple inputs specify multiple --input_dtype on the command "
                                            "line like: \n"
                                            "    --input_dtype 'data1' 'float32' --input_dtype 'data2' 'float32'")
            self.add_optional_argument('--input_encoding', "-e", nargs='+', action='append',
                                       help='Usage: '
                                            '    --input_encoding "INPUT_NAME" INPUT_ENCODING_IN [INPUT_ENCODING_OUT]\n'
                                            'Input encoding of the network inputs. Default is bgr. \n'
                                            'e.g.\n'
                                            '   --input_encoding "data" rgba \n'
                                            'Quotes must wrap the input node name to handle special characters, \n'
                                            'spaces, etc. To specify encodings for multiple inputs, invoke \n'
                                            '--input_encoding for each one. \n'
                                            'e.g.\n'
                                            '    --input_encoding "data1" rgba --input_encoding "data2" other\n'
                                            'Optionally, an output encoding may be specified for an input node by \n'
                                            'providing a second encoding. The default output encoding is bgr.\n'
                                            'e.g. \n'
                                            '    --input_encoding "data3" rgba rgb \n'
                                            'Input encoding types:\n '
                                            '    image color encodings: bgr,rgb, nv21, nv12, ... \n'
                                            '    time_series: for inputs of rnn models; \n'
                                            '    other: not available above or is unknown. \n'
                                            'Supported encodings:\n   ' + '\n   '.join(op_graph.InputEncodings.
                                                                                       get_supported_encodings()),
                                       metavar="\b", default=[])

            q_group = self.add_argument_group(title='Quantizer Options')
            q_group.add_argument('--quantization_overrides', type=str, default="",
                                 help='Use this option to specify a json file with parameters to use for '
                                      'quantization. These will override any quantization data carried from conversion '
                                      '(eg TF fake quantization) or calculated during the normal quantization process. '
                                      'Format defined as per AIMET specification.')
            q_group.add_argument('--keep_quant_nodes', default=False, action="store_true",
                                 help='Use this option to keep activation quantization nodes in the graph rather than '
                                      'stripping them.')

    def __init__(self, args,
                 naming_policy=ConversionNamePolicy(),
                 shape_inference_policy=None,
                 axis_order=AxisOrder(),
                 custom_op_factory=None):
        super(ConverterFrontend, self).__init__(args)
        self.output_names = args.out_names

        for input_encoding in args.input_encoding:
            if len(input_encoding) not in [2, 3]:
                raise ValueError('Received incorrect number of input encodings for input {}. Got {}, expected \n'
                                 'one input encoding and one (optional) output encoding per graph input in the \n'
                                 'following format: \n'
                                 '    --input_encoding "INPUT_NAME" INPUT_ENCODING_IN [INPUT_ENCODING_OUT] \n'
                                 .format(input_encoding[0], len(input_encoding) - 1))

        self.graph = op_graph.IROpGraph(naming_policy, shape_inference_policy,
                                        args.input_type, args.input_dtype, args.input_encoding, axis_order,
                                        args.quantization_overrides, args.keep_quant_nodes,
                                        output_nodes=self.output_names)

        self.custom_op_config_paths = args.custom_op_config_paths
        self.custom_op_factory = custom_op_factory

    @abstractmethod
    def convert(self):
        """
        Convert the input framework model to IROpGraph: to be overridden by each framework
        """
        pass

    # TODO: Move once converter base hierarchy is refactored
    def populate_custom_op_collection(self,
                                      model,
                                      converter_type='onnx',
                                      **kwargs):
        # Create a custom op collection based on configs provided by user
        if self.custom_op_config_paths is not None:
            for config_path in self.custom_op_config_paths:
                try:
                    self.custom_op_factory.parse_config(config_path,
                                                        model=model,
                                                        converter_type=converter_type,
                                                        **kwargs)
                except Exception as e:
                    if not is_log_level_debug():
                        traceback.print_exc()
                    log_error("Error populating custom ops from: {}\n {}".format(config_path,
                                                                                 str(e)))
                    sys.exit(-1)

                if not len(self.custom_op_factory.op_collection) and \
                        not self.custom_op_factory.default_op_collection:
                    raise LookupError("CUSTOM_OP_NOT_FOUND: "
                                      "None of the custom Ops present in the "
                                      "config were found in the provided model.")
