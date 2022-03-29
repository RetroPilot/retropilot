# ==============================================================================
#
#  Copyright (c) 2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from qti.aisw.converters.common.utils.argparser_util import ArgParserWrapper
from qti.aisw.converters.relay.passes.pattern_match.tflite_detection_postprocess import IdentifyTFLiteDetectionPostProcess
from .relay_importer import RelayImporter
import tvm
from tvm.relay.frontend import tflite as tflite_to_relay
from tvm import relay
import tvm.relay.op.op as _op

# TFLite.Model.Model has changed to TFLite.Model from 1.14 to 2.1
try:
    import tflite
except TypeError:
    import tflite.Model as tflite


class TFLiteImporter(RelayImporter):
    class ArgParser(ArgParserWrapper):
        def __init__(self, **kwargs):
            super(TFLiteImporter.ArgParser, self).__init__(conflict_handler='resolve', **kwargs)
            self.add_required_argument('-d', '--input_dim', nargs=2, action='append',
                                       metavar=('INPUT_NAME', 'INPUT_DIM'),
                                       help="The names and dimensions of the network input layers specified "
                                            "in the format [input_name comma-separated-dimensions], "
                                            "for example: \n"
                                            "    'data' 1,224,224,3\n"
                                            "Note that the quotes should always be included in order to handle"
                                            "special characters, spaces, etc. \n"
                                            "For multiple inputs specify multiple --input_dim on the command "
                                            "line like: \n"
                                            "    --input_dim 'data1' 1,224,224,3 --input_dim 'data2' 1,50,100,3")
            self.add_optional_argument('--input_dtype', nargs=2, action='append',
                                       metavar=('INPUT_NAME', 'INPUT_DTYPE'),
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

    def __init__(self, args):
        super(TFLiteImporter, self).__init__(args)

        self.shape_dict = {}
        for in_name, in_dims in args.input_dim:
            self.shape_dict[in_name] = [int(i) for i in in_dims.split(',')]

        if args.input_dtype:
            self.dtype_dict = {in_name: in_dtype for in_name, in_dtype in args.input_dtype}
        else:
            self.dtype_dict = {}
            for input_name in self.shape_dict:
                if input_name not in self.dtype_dict:
                    self.dtype_dict[input_name] = "float32"

    def convert_to_relay(self, input_model_path, **kwargs):
        if isinstance(input_model_path, str):
            tflite_model_buf = open(input_model_path, "rb").read()
        elif isinstance(input_model_path, bytes):
            tflite_model_buf = input_model_path
        else:
            raise TypeError("Unsupported type {} for {}".format(type(input_model_path), input_model_path))
        tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
        try:
            self.mod, self.params, self.output_names_dict =\
                    tflite_to_relay.from_tflite(tflite_model, self.shape_dict, self.dtype_dict)
        except ValueError:
            self.mod, self.params = \
                tflite_to_relay.from_tflite(tflite_model, self.shape_dict, self.dtype_dict)
            self.output_names_dict = {}

        self._post_process()

        return self.mod, self.params, self.output_names_dict

    def _post_process(self):
        """post-process Relay module, including necessary fixes and optimizations"""

        span_output_dict = {}
        def visit_module(expr: relay.expr):
            if hasattr(expr, 'span'):
                if expr.span in span_output_dict and expr.span != None:
                    self.output_names_dict.setdefault(hash(expr), (expr, span_output_dict[expr.span]))

        def _rewrite_output_names_dict():
            """rewrite the output_names_dict after pass"""
            nonlocal span_output_dict
            span_output_dict = {expr.span: name for _, (expr, name) in self.output_names_dict.items()}
            self.output_names_dict = {}
            relay.analysis.post_order_visit(self.mod["main"], visit_module)
            #self.output_names_dict = self.new_dict

        # register custom relay ops
        self._register_ops()

        # Prepare for Relay Passes
        # Currently only one pass is required, which compress
        # tflite_detection_postprocess expression back to one ir.
        seq = tvm.transform.Sequential([
            IdentifyTFLiteDetectionPostProcess()
        ])

        # need opt_level=3 to trigger ConvertLayout
        with tvm.transform.PassContext(opt_level=3):
            self.mod = seq(self.mod)
        _rewrite_output_names_dict()

    # TODO: revisit if we should put this functionality to another module or package.
    # Current put it here just because there are not so many OPs we need to register.
    @staticmethod
    def _register_ops():
        ##########################################################
        # op tflite_detection_postprocess
        ##########################################################

        def tflite_detection_postprocess_rel(arg_types, attrs):
            assert len(arg_types) == 3
            batch_num = arg_types[0].shape[0]
            nms_top_k = attrs['nms_top_k']

            # Return the shape of in sequence:
            # Scores, boxes valid_count adn cls_ids
            scores_type = relay.TensorType([batch_num, nms_top_k], 'float32')
            boxes_type = relay.TensorType([batch_num, nms_top_k, 4], 'float32')
            valid_type = relay.TensorType([batch_num], 'int32')
            cls_ids_type = relay.TensorType([batch_num, nms_top_k], 'float32')

            return relay.TupleType([scores_type, boxes_type, valid_type, cls_ids_type])

        tflite_detection_postprocess_op_name = "detection_postprocess"
        _op.register(tflite_detection_postprocess_op_name)
        _op.get(tflite_detection_postprocess_op_name).set_num_inputs(3)
        _op.get(tflite_detection_postprocess_op_name).add_argument("class_prob", "expr", "the input class probability tensor.")
        _op.get(tflite_detection_postprocess_op_name).add_argument("box_prob", "expr", "the input box probability tensor.")
        _op.get(tflite_detection_postprocess_op_name).add_argument("anchors", "var", "the input pre-defined yxhw anchors tensor.")
        _op.get(tflite_detection_postprocess_op_name).set_attrs_type_key("DictAttrs")
        # call customized relation functions
        _op.get(tflite_detection_postprocess_op_name).add_type_rel("tflite_detection_postprocess", tflite_detection_postprocess_rel)
        _op.get(tflite_detection_postprocess_op_name).set_support_level(1)
        _op.register_pattern(tflite_detection_postprocess_op_name, _op.OpPattern.OPAQUE)
        _op.register_stateful(tflite_detection_postprocess_op_name, False)

        ##########################################################
        # End of Op tflite_detection_postprocess
        ##########################################################
