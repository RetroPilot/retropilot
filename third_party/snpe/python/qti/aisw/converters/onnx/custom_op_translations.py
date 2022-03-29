# ==============================================================================
#
#  Copyright (c) 2019-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from .onnx_translations import *
from qti.aisw.converters.backend.custom_ops.op_factory import OpFactory
from qti.aisw.converters.backend.custom_ops.core import get_internal_dtype
import numpy as np


# ------------------------------------------------------------------------------
#   Custom Op
# ------------------------------------------------------------------------------
class OnnxCustomOpTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.custom_op = None

    def extract_input_names(self, src_op, graph):
        return [str(input.name) for input in self.custom_op.inputs]

    def extract_output_names(self, src_op, graph):
        return [str(output.name) for output in self.custom_op.outputs]

    def extract_parameters(self, src_op, graph):
        custom_op = OpFactory.op_collection.get_first_of(src_op.op_type)
        package_name = OpFactory.get_package_name(custom_op.op_type)
        self.custom_op = custom_op

        for name, custom_param in custom_op.params.items():
            param = custom_param.param
            if param.data is None:
                if not param.static:
                    raise ValueError(
                        code_to_message.get_error_message("ERROR_CUSTOM_OP_PARAM_NO_DATA")
                        (name, custom_op.op_type))
                elif graph.weights.has(name):
                    param.data = np.asarray(graph.weights.weight_map[str(name)].weights)
                    param.data_type = get_internal_dtype(param.data, param)
                    param.dimensions = param.data.shape
                    param.rank = len(param.data.shape)
                    graph.weights.weight_map[str(name)].consumed = True
                else:
                    raise LookupError(code_to_message.get_error_message("ERROR_CANNOT"
                                                                        "_INGEST_STATIC_INPUT")
                                      (str(name)))

        inputs, outputs, scalar_params, tensor_params = custom_op.as_dict()
        return op_adapter.CustomOp(name=src_op.name,
                                   package_name=package_name,
                                   custom_type=src_op.op_type,
                                   axis_orders=custom_op.axis_orders,
                                   inputs=inputs,
                                   outputs=outputs,
                                   output_dims=custom_op.output_dims,
                                   tensor_params=tensor_params,
                                   scalar_params=scalar_params)


OnnxTranslations.register_translation(OnnxCustomOpTranslation(),
                                      converter_type('custom', 'onnx'),
                                      op_adapter.CustomOp.TRANSLATION_KEY)
