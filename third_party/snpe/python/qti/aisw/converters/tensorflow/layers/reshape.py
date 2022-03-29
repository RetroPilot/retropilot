# =============================================================================
#
#  Copyright (c) 2015-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from qti.aisw.converters.common.converter_ir.op_adapter import ReshapeOp
from qti.aisw.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from qti.aisw.converters.tensorflow.layers.constant import ConstantLayerResolver
from qti.aisw.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    GraphSequence,
)


class ReshapeLayerResolver(LayerResolver, object):
    class Descriptor(LayerDescriptor):
        def __init__(self, name, nodes, reshape_op):
            super(ReshapeLayerResolver.Descriptor, self).__init__('Reshape', name, nodes)
            self.reshape_op = reshape_op

        def is_input_tensor(self, op, tensor):
            return tensor == op.inputs[0]

    def __init__(self):
        sequence_reshape = GraphSequence([ConverterSequenceNode('root', ['Reshape', 'Squeeze', 'ExpandDims'])])
        sequence_reshape.set_outputs(['root'])

        self.sequences = [sequence_reshape]

    def resolve_layer(self, graph_matcher, graph_helper):
        descriptors = []
        for sequence in self.sequences:
            matches = graph_matcher.match_sequence(sequence)
            for match in matches:
                reshape_op = match['root']
                consumed_nodes = match.consumed_nodes
                reshape_descriptor = ReshapeLayerResolver.Descriptor(str(reshape_op.name),
                                                                     consumed_nodes,
                                                                     reshape_op)
                descriptors.append(reshape_descriptor)

        return descriptors


class ReshapeLayerBuilder(LayerBuilder):
    def build_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type ir_graph: converters.common.converter_ir.op_graph.IROpGraph
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: ReshapeLayerResolver.Descriptor
        :rtype: int
        """
        input_name = self.get_input_name(converter_context, descriptor, input_descriptors[:1])
        output_shape = converter_context.graph_helper.get_op_output_shape(descriptor.reshape_op)
        return ir_graph.add(ReshapeOp(descriptor.output_names[0],
                                      output_shape),
                            input_names=input_name,
                            output_names=descriptor.output_names[0])

    def transform_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):
        non_ignored_inputs = [d for d in input_descriptors if not d.is_ignored]
        if len(non_ignored_inputs) == 0:
            # only set descriptor as ignored if there are no inputs to ignored op
            is_input_independent = [len(d.child_ops[0].inputs) == 0 for d in input_descriptors if d.child_ops]
            if is_input_independent and all(is_input_independent):
                descriptor.set_ignored(True)
