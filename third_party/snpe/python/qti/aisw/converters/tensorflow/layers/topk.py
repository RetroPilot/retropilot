# =============================================================================
#
#  Copyright (c) 2018-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from qti.aisw.converters.common.converter_ir.op_adapter import TopKOp
from qti.aisw.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from qti.aisw.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    NonConsumableConverterSequenceNode,
    GraphSequence
)


class TopKLayerResolver(LayerResolver, object):
    class Descriptor(LayerDescriptor):
        def __init__(self, name, nodes, k, is_sorted, output_names=None):
            super(TopKLayerResolver.Descriptor, self).__init__('TopK', name, nodes, output_names=output_names)
            self.k = k
            self.is_sorted = is_sorted

        def is_input_tensor(self, op, tensor):
            return tensor == op.inputs[0]

    def __init__(self):
        sequence = GraphSequence([
            ConverterSequenceNode('topk', ['TopKV2']),
            NonConsumableConverterSequenceNode('input', ['?']),
            NonConsumableConverterSequenceNode('topk/k', ['?'])
        ])
        sequence.set_inputs('topk', ['input','topk/k'])
        sequence.set_outputs(['topk'])
        self.sequences = [sequence]

    def resolve_layer(self, graph_matcher, graph_helper):
            descriptors = []
            for sequence in self.sequences:
                for match in graph_matcher.match_sequence(sequence):
                    topk_op = match['topk']
                    k_op = match['topk/k']
                    k = int(graph_helper.evaluate_tensor_output(k_op.outputs[0]))
                    is_sorted = bool(topk_op.get_attr('sorted'))
                    consumed_nodes = match.consumed_nodes
                    descriptors.append(
                        TopKLayerResolver.Descriptor(str(topk_op.name),
                                                     consumed_nodes,
                                                     k=k,
                                                     is_sorted=is_sorted,
                                                     output_names=[str(topk_op.outputs[0].name),
                                                                   str(topk_op.outputs[1].name)]))
            return descriptors


class TopKLayerBuilder(LayerBuilder):
    def build_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type ir_graph: converters.common.converter_ir.op_graph.IROpGraph
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: TopKLayerResolver.Descriptor
        :rtype: int
        """
        input_name = self.get_input_name(converter_context, descriptor, input_descriptors)
        return ir_graph.add(TopKOp(name=descriptor.layer_name,
                                   k=descriptor.k,
                                   is_sorted=descriptor.is_sorted),
                            input_names=input_name,
                            output_names=descriptor.output_names)
