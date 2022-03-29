# =============================================================================
#
#  Copyright (c) 2018-2020 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from qti.aisw.converters.common.converter_ir.op_adapter import ArgMinOp
from qti.aisw.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from qti.aisw.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    NonConsumableConverterSequenceNode,
    GraphSequence
)
from qti.aisw.converters.tensorflow.util import ConverterError


class ArgMinLayerResolver(LayerResolver, object):

    class Descriptor(LayerDescriptor):
        def __init__(self, name, nodes, axis, output_names=None):
            super(ArgMinLayerResolver.Descriptor, self).__init__('ArgMin', name, nodes, output_names=output_names)
            self.axis = axis

    def __init__(self):
        self.sequence = GraphSequence([
            ConverterSequenceNode('root', ['ArgMin']),
            ConverterSequenceNode('axis', ['Const']),
            NonConsumableConverterSequenceNode('input', ['?']),
        ])
        self.sequence.set_inputs('root', ['input', 'axis'])
        self.sequence.set_outputs(['root'])

    def resolve_layer(self, graph_matcher, graph_helper):
        descriptors = []
        for match in graph_matcher.match_sequence(self.sequence):
            argmin_op = match['root']
            input_op = match['input']
            axis_op = match['axis']

            input_shape = graph_helper.get_op_output_shape(input_op)
            input_rank = len(input_shape)

            axis = int(graph_helper.evaluate_tensor_output(axis_op.outputs[0]))
            if axis < 0:
                axis += input_rank

            consumed_nodes = match.consumed_nodes
            argmin_descriptor = ArgMinLayerResolver.Descriptor(
                str(argmin_op.name), consumed_nodes, axis,
                output_names=[str(argmin_op.outputs[0].name)])
            descriptors.extend([argmin_descriptor])

        return descriptors


class ArgMinLayerBuilder(LayerBuilder):
    def build_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type ir_graph: converters.common.converter_ir.op_graph.IROpGraph
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: ArgMinLayerResolver.Descriptor
        :rtype: int
        """
        input_name = self.get_input_name(converter_context, descriptor, input_descriptors)
        output_name = descriptor.output_names[0]
        return ir_graph.add(ArgMinOp(descriptor.layer_name, axis=descriptor.axis),
                     input_name, output_name)
