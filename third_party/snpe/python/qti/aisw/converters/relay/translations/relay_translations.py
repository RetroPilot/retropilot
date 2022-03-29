# ==============================================================================
#
#  Copyright (c) 2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from abc import ABCMeta

from qti.aisw.converters.common.converter_ir import translation
from qti.aisw.converters.common.converter_ir.op_graph import IROpGraph
from qti.aisw.converters.common.utils.converter_utils import log_debug1

from tvm import relay


class RelayTranslationBase(translation.ConversionTranslationBase, metaclass=ABCMeta):

    def __init__(self):
        super(RelayTranslationBase, self).__init__()
        self.extract_parameters = None

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        return {}

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        raise NotImplementedError("translate_op for {} not implemented ".format(str(self.__class__.__name__)))

    # Thin wrapper added so that Op Translation can override if needed
    def extract_input_names(self,
                            relay_expr: relay.expr.Call,
                            converter_context,
                            **kwargs):
        return converter_context.get_input_names(relay_expr)

    # Thin wrapper added so that Op Translation can override if needed
    def extract_output_names(self,
                             relay_expr: relay.expr.Call,
                             converter_context,
                             **kwargs):
        num_outputs = kwargs.get("num_outputs", 1)
        return converter_context.get_output_names(relay_expr, num_outputs)

    def add_op(self,
               relay_expr: relay.expr.Call,
               quir_graph: IROpGraph,
               **kwargs):
        converter_context = kwargs.get("converter_context")
        relay_params = kwargs.get("relay_params")

        attr_dict = self.extract_attributes(relay_expr, relay_params)

        input_names = self.extract_input_names(relay_expr,
                                               converter_context=converter_context)

        ir_op = self.translate_op(relay_expr,
                                  relay_params,
                                  converter_context,
                                  quir_graph,
                                  attr_dict,
                                  input_names)

        num_outputs = ir_op.num_outputs
        output_names = self.extract_output_names(relay_expr,
                                                 converter_context=converter_context,
                                                 num_outputs=num_outputs)

        log_debug1("Op {} Type {} inputs {}", ir_op.name, ir_op.type, input_names)
        log_debug1("Op {} Type {} outputs {}", ir_op.name, ir_op.type, output_names[:num_outputs])

        ir_node = converter_context.add_op_to_graph(relay_expr, ir_op, input_names, output_names[:num_outputs])
        return ir_node
