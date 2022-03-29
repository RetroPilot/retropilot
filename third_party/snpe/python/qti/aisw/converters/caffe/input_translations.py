# ==============================================================================
#
#  Copyright (c) 2019-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
from qti.aisw.converters.common.utils import code_to_message

from .caffe_base_translation import CaffeTranslationBase, CaffeTranslations
from qti.aisw.converters.common.converter_ir import op_adapter
from qti.aisw.converters.common.utils.converter_utils import *


class CaffeInputTranslation(CaffeTranslationBase):

    ADD_INPUT_OP_FROM_SPEC = "ADD_INPUT_FROM_SPEC"

    def __init__(self):
        CaffeTranslationBase.__init__(self)
        self.register_method(self.ADD_INPUT_OP_FROM_SPEC, self.add_input_op_from_spec)

    @staticmethod
    def add_input_op_from_spec(graph, spec, index):
        # Need to add a data layer
        input_name = str(spec.input[index])
        if len(spec.input_shape) > 0:
            input_dim = list(map(int, spec.input_shape[index].dim))
        elif len(spec.input_dim) > 0:
            input_dim = list(map(int, spec.input_dim))
        else:
            raise ValueError(code_to_message.get_error_message('ERROR_CAFFE_DATA_LAYER_ERR_NO_INPUT_DIM')
                             (str(input_name)))

        return graph.add_input(input_name, input_dim)

    def add_input_op(self, src_input, graph, **kwargs):
        src_type = converter_type(src_input.type, "caffe")
        if src_type == converter_type("data", "caffe"):
            spec = kwargs.get("spec")
            input_dim = list()
            if len(spec.input_shape) > 0:
                input_dim = list(map(int, spec.input_shape[0].dim))
            elif len(spec.input_dim) > 0:
                input_dim = list(map(int, spec.input_dim))
        elif src_type == converter_type("input", "caffe"):
            if not hasattr(src_input, "input_param"):
                raise ValueError(code_to_message.get_error_message('ERROR_CAFFE_NO_INPUT_PARAM_SPECIFIED')
                                 (str(src_input.name)))

            input_param = src_input.input_param
            input_dim = list(map(int, input_param.shape[0].dim))
        else:
            raise ValueError("Unsupported Caffe input layer type: {}".format(src_input.type))

        if len(input_dim) == 0:
            raise ValueError(code_to_message.get_error_message('ERROR_CAFFE_DATA_LAYER_ERR_NO_INPUT_DIM')
                             (str(src_input.name)))

        return graph.add_input(str(src_input.top[0]), input_dim)

    def add_op(self, src_op, graph, **kwargs):
        raise NotImplementedError("add_op() for {} not implemented. Call add_input_op() instead."
                                  .format(str(self.__class__.__name__)))

    def extract_input_names(self, src_op, graph):
        raise NotImplementedError("extract_input_names() for {} not implemented ".format(str(self.__class__.__name__)))

    def extract_output_names(self, src_op, graph):
        raise NotImplementedError("extract_output_names() for {} not implemented ".format(str(self.__class__.__name__)))


CaffeTranslations.register_translation(CaffeInputTranslation(),
                                       converter_type('input', 'caffe'),
                                       converter_type('data', 'caffe'),
                                       op_adapter.InputOp.TRANSLATION_KEY)


class CaffeSubtractMeanTranslation(CaffeTranslationBase):
    def __init__(self):
        CaffeTranslationBase.__init__(self)

    def extract_parameters(self, layer, graph):
        transform_param = layer.transform_param
        return op_adapter.SubtractMeanOp(layer.name + "_subtract_mean",
                                         list(transform_param.mean_value))


CaffeTranslations.register_translation(CaffeSubtractMeanTranslation(),
                                       converter_type('subtract_mean', 'caffe'),
                                       op_adapter.SubtractMeanOp.TRANSLATION_KEY)
