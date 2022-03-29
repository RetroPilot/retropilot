# ==============================================================================
#
#  Copyright (c) 2019-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import importlib
import os
import sys
import traceback

from google.protobuf import text_format

from qti.aisw.converters.common.converter_ir.axis_tracker import AxisOrders
from qti.aisw.converters.common.converter_ir import op_adapter, translation
from qti.aisw.converters.common.utils import converter_utils, code_to_message
from qti.aisw.converters.common.converter_base import ConverterFrontend
from qti.aisw.converters.common.utils.converter_utils import *
from .caffe_base_translation import CaffeTranslations, CaffeTranslationBase
from .caffe_policies import CaffeNamePolicy, CaffeShapeInferencePolicy
from .weight_provider import RandomWeightProvider, BlobWeightProvider


# ------------------------------------------------------------------------------
#   The Converter Class
# ------------------------------------------------------------------------------
class CaffeConverterFrontend(ConverterFrontend):
    class ArgParser(ConverterFrontend.ArgParser):
        def __init__(self, **kwargs):
            super(CaffeConverterFrontend.ArgParser, self).__init__(**kwargs)
            # add command-line options custom to caffe converter
            self.add_optional_argument('-b', '--caffe_bin', type=str,
                                       help='Input caffe binary file containing the weight data')
            self.add_optional_argument('--udl', type=str, nargs=2, metavar=('UDL_MODULE', 'FACTORY_FUNCTION'),
                                       help='Option to add User Defined Layers. Provide Filename, Function name.'
                                            '1.Filename: Name of python module to load for registering custom '
                                            'udl(note: must be in PYTHONPATH). If file part of package list the '
                                            'package.filename as you would when doing a python import.'
                                            '2.Function name: Name of the udl factory function that return a '
                                            'dictionary of key layer type and value function callback.',
                                       default=[])
            self.add_optional_argument('--enable_preprocessing',
                                       action="store_const", const=True, default=False,
                                       help='If specified, converter will enable preprocessing specified by a data'
                                            'layer transform_param subtract_mean is supported.')

    def __init__(self, args, *, custom_op_factory=None):
        super(CaffeConverterFrontend, self).__init__(args,
                                                     naming_policy=CaffeNamePolicy(),
                                                     shape_inference_policy=CaffeShapeInferencePolicy(),
                                                     axis_order=AxisOrders.CAFFE,
                                                     custom_op_factory=custom_op_factory)
        self.translations = CaffeTranslations
        self.caffe_weights_path = args.caffe_bin
        self.udl_args = args.udl
        self.input_dim = list()
        self.network_dim = []
        self.udl_layer_dict = {}
        self.spec = None
        self.enable_preprocessing = args.enable_preprocessing

        # Caffe specific:  This control caffe's output with verbose option
        if not converter_utils.is_log_level_debug():
            # The levels are
            # 0 - debug
            # 1 - info (still a LOT of outputs)
            # 2 - warnings
            # 3 - errors
            os.environ['GLOG_minloglevel'] = '2'

    def convert(self):
        # import of Caffe has to come after the setting of GLOG_minloglevel for it to take effect
        try:
            import caffe
            import caffe.proto.caffe_pb2 as caffe_pb2
        except ImportError as e:
            raise Exception(code_to_message.get_error_message("ERROR_CAFFE_NOT_FOUND")(e.msg, str(sys.path)))

        # these need to be imported so they are evaluated and translations are registered.
        # importing them here since the modules import caffe as well.
        from . import input_translations, data_translations, math_translations, nn_translations, proposal_translation,\
            noop_translations, rnn_translations, udl_translation

        # Add udl module if provided
        if len(self.udl_args):
            log_info("Loading UDLs from module {} using factory function {}", self.udl_args[0], self.udl_args[1])
            udl_factory_func = getattr(importlib.import_module(self.udl_args[0]), self.udl_args[1])
            self.udl_layer_dict = udl_factory_func()
            # standardize the layer types so that type matching is easier below
            self.udl_layer_dict = {converter_type(k, "caffe"): v for k, v in self.udl_layer_dict.items()}

        caffe.set_mode_cpu()
        # get caffe spec
        try:
            self.spec = caffe_pb2.NetParameter()
            with open(self.input_model_path, 'rb') as text_file:
                text_format.Merge(text_file.read(), self.spec)
        except Exception as e:
            print(code_to_message.get_error_message('ERROR_CAFFE_CAFFE_PARSING_ERROR')(self.input_model_path,
                                                                                       str(e)))
            print(code_to_message.get_progress_message('INFO_CAFFE_CAFFE_INSTALLATION_ERROR')(caffe.__file__))
            sys.exit(1)

        # get weight provider
        caffe_net=None
        if self.caffe_weights_path is None:
            self.graph.weights = RandomWeightProvider(self.spec, self.graph)
        else:
            caffe_net = caffe.Net(self.input_model_path, self.caffe_weights_path, caffe.TEST)
            self.graph.weights = BlobWeightProvider(caffe_net.params)

        # assign input type as data when inputs are not written  as layers in prototxt
        input_type = converter_type("data", "caffe")

        # If there are additional inputs that are not written as layers in prototxt, create data layers for
        # these. Note that only the input {} input_shape {} and input_dim syntax is supported here.
        for index in range(len(self.spec.input)):
            self.translations.apply_method_to_op(input_type,
                                                 input_translations.CaffeInputTranslation.ADD_INPUT_OP_FROM_SPEC,
                                                 self.graph, self.spec, index)

        # extract parameters, infer shapes, etc.
        layers = self.spec.layer if len(self.spec.layer) != 0 else self.spec.layers

        # Populate the custom op using the provided model
        if caffe_net is None and self.custom_op_config_paths:
            raise RuntimeError('The caffe model binary is required to ingest a custom op. Please provide '
                               'caffe model via the \'-b\' converter option')
        elif self.custom_op_config_paths:
                from . import custom_op_translations
                self.populate_custom_op_collection(layers, converter_type='caffe', caffe_net=caffe_net)

        for i, layer in enumerate(layers):
            if self._is_in_train_phase(layer):
                # Skip train layers
                continue
            log_debug(code_to_message.get_debugging_message("DEBUG_CONVERTING_NODE")(i, layer.type))
            src_type = converter_type(layer.type, "caffe")

            # cases where input is a layer
            if src_type == converter_type("data", "caffe") or src_type == converter_type("input", "caffe"):
                self.translations.apply_method_to_op(src_type,
                                                     input_translations.CaffeInputTranslation.ADD_INPUT_OP,
                                                     layer, self.graph, spec=self.spec)
                if self.enable_preprocessing:
                    self.setup_preprocessing(src_type, layer, self.graph)
            else:
                try:
                    # first check if layer is a registered custom op in an op collection.
                    # If so, the layer is added and the loop continues.
                    if self.custom_op_factory and layer.type in self.custom_op_factory.op_collection:
                        src_type = converter_type('custom', "caffe")
                        self.translations.apply_method_to_op(src_type,
                                                             translation.ConversionTranslationBase.ADD_OP,
                                                             layer,
                                                             self.graph)
                        continue

                    # Check if layer is UDL. If so call factory function to extract blob info
                    # and then add to graph
                    # TODO: Remove UDL
                    if src_type in self.udl_layer_dict:
                        udl_obj = self.udl_layer_dict[src_type]
                        udl_layer = (layer, udl_obj)
                        self.translations.apply_method_to_op(op_adapter.UdlOp.TRANSLATION_KEY,
                                                             translation.ConversionTranslationBase.ADD_OP,
                                                             udl_layer,
                                                             self.graph)
                    else:
                        self.translations.apply_method_to_op(src_type,
                                                             translation.ConversionTranslationBase.ADD_OP,
                                                             layer,
                                                             self.graph)
                except Exception as e:
                    if converter_utils.is_log_level_debug():
                        traceback.print_exc()
                    log_error("Node %s: Type : %s %s" % (layer.name, layer.type, e))
                    sys.exit(-1)

        self.graph.eval_macs_params()
        return self.graph

    @staticmethod
    def _is_in_train_phase(layer):
        if layer.include:
            import caffe.proto.caffe_pb2 as caffe_pb2
            caffe_phases = {pair[0]: pair[1] for pair in list(caffe_pb2.Phase.items())}
            phases = [state.phase for state in layer.include if state.phase is not None]
            return caffe_phases['TRAIN'] in phases
        return False

    def setup_preprocessing(self, src_type, layer, graph):
        if layer.transform_param.mean_value:
            src_type = op_adapter.SubtractMeanOp.TRANSLATION_KEY
            self.translations.apply_method_to_op(src_type,
                                                 CaffeTranslationBase.ADD_INPUT_PREPROCESSING_OP,
                                                 layer,
                                                 self.graph)

