# =============================================================================
#
#  Copyright (c) 2016-2020 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
from qti.aisw.converters.common.utils import code_to_message
from qti.aisw.converters.common.converter_ir.op_adapter import ElementwiseSumOp
from qti.aisw.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from qti.aisw.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    GraphSequence
)
from qti.aisw.converters.tensorflow.util import ConverterError


class AddNLayerResolver(LayerResolver, object):
    class Descriptor(LayerDescriptor):
        def __init__(self, name, nodes):
            super(AddNLayerResolver.Descriptor, self).__init__('ElementWiseSumN', name, nodes)

    def __init__(self):
        self.sequence = GraphSequence([ConverterSequenceNode('root', ['AddN'])])
        self.sequence.set_outputs(['root'])

    def resolve_layer(self, graph_matcher, graph_helper):
        matches = graph_matcher.match_sequence(self.sequence)
        if len(matches) == 0:
            return []

        descriptors = []
        for match in matches:
            add_op = match['root']
            descriptors.append(AddNLayerResolver.Descriptor(str(add_op.name), match.consumed_nodes))
        return descriptors


class AddNLayerBuilder(LayerBuilder):
    def build_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type ir_graph: converters.common.converter_ir.op_graph.IROpGraph
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: ConcatLayerResolver.Descriptor
        :rtype: int
        """
        input_names = self.get_input_names(converter_context, descriptor, input_descriptors)
        if len(input_names) < 2:
            raise ConverterError(code_to_message.get_error_message('ERROR_TF_ADD_N_NUM_OF_INPUTS')(descriptor.layer_name))
        output_name = descriptor.output_names[0]

        return ir_graph.add(ElementwiseSumOp(descriptor.layer_name),
                            input_names, output_name)

