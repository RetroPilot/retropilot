# =============================================================================
#
#  Copyright (c) 2015-2020 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from qti.aisw.converters.common.converter_ir.op_adapter import LogSoftmaxOp
from qti.aisw.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from qti.aisw.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    GraphSequence,
    NonConsumableConverterSequenceNode
)


class LogSoftmaxLayerResolver(LayerResolver, object):
    class Descriptor(LayerDescriptor):
        def __init__(self, name, operations, **kwargs):
            super(LogSoftmaxLayerResolver.Descriptor, self).__init__('LogSoftMax', name, operations)
            self.axis = kwargs.get('axis')

    def __init__(self):
        sequence_two_dim_logsoftmax = GraphSequence([ConverterSequenceNode('root', ['LogSoftMax'])])
        sequence_two_dim_logsoftmax.set_outputs(['root'])

        sequence_multi_dim_logsoftmax = GraphSequence([
            ConverterSequenceNode('sub', ['Sub']),
            ConverterSequenceNode('Rank', ['Constant']),
            ConverterSequenceNode('y', ['Constant']),
            ConverterSequenceNode('pack', ['Pack']),
            ConverterSequenceNode('slice', ['Slice']),
            ConverterSequenceNode('Shape_1', ['Constant']),
            ConverterSequenceNode('size', ['Constant']),
            ConverterSequenceNode('concat', ['ConcatV2']),
            ConverterSequenceNode('values_0', ['Constant']),
            ConverterSequenceNode('axis', ['Constant']),
            ConverterSequenceNode('Reshape', ['Reshape']),
            ConverterSequenceNode('logsoftmax', ['LogSoftmax']),
            ConverterSequenceNode('root', ['Reshape']),
            ConverterSequenceNode('shape', ['Constant']),
            NonConsumableConverterSequenceNode('input', ['?'])
        ])
        sequence_multi_dim_logsoftmax.set_inputs('sub', ['Rank', 'y'])
        sequence_multi_dim_logsoftmax.set_inputs('pack', ['sub'])
        sequence_multi_dim_logsoftmax.set_inputs('slice', ['Shape_1','pack','size'])
        sequence_multi_dim_logsoftmax.set_inputs('concat', ['values_0','slice','axis'])
        sequence_multi_dim_logsoftmax.set_inputs('Reshape', ['input','concat'])
        sequence_multi_dim_logsoftmax.set_inputs('logsoftmax', ['Reshape'])
        sequence_multi_dim_logsoftmax.set_inputs('root', ['logsoftmax','shape'])
        sequence_multi_dim_logsoftmax.set_outputs(['root'])

        self.sequences = [sequence_two_dim_logsoftmax, sequence_multi_dim_logsoftmax]

    def resolve_layer(self, graph_matcher, graph_helper):
        potential_descriptors = []
        for sequence in self.sequences:
            matches = graph_matcher.match_sequence(sequence)
            for match in matches:
                logsoftmax_op = match['root']
                axis = int(logsoftmax_op.get_attr('axis'))
                consumed_nodes = match.consumed_nodes
                potential_descriptors.append(
                    LogSoftmaxLayerResolver.Descriptor('LogSoftmax',
                                                        str(logsoftmax_op.name),
                                                        consumed_nodes,
                                                        axis=axis))
        return potential_descriptors


class LogSoftmaxLayerBuilder(LayerBuilder):
    def build_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type ir_graph: converters.common.converter_ir.op_graph.IROpGraph
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: LogSoftmaxLayerResolver.Descriptor
        :rtype: int
        """
        input_name = self.get_input_name(converter_context, descriptor, input_descriptors)
        output_name = descriptor.output_names[0]
        return ir_graph.add(LogSoftmaxOp(name=descriptor.layer_name),
                            input_names=input_name,
                            axis=descriptor.axis,
                            output_names=output_name)