# =============================================================================
#
#  Copyright (c) 2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from qti.aisw.converters.common.converter_ir.op_adapter import OneHotOp, ReshapeOp
from qti.aisw.converters.tensorflow.common import LayerDescriptor, LayerBuilder, LayerResolver
from qti.aisw.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    NonConsumableConverterSequenceNode,
    GraphSequence
)
from abc import ABCMeta
from qti.aisw.converters.tensorflow.layers.constant import ConstantLayerResolver
from qti.aisw.converters.tensorflow.util import GraphHelper


class OneHotLayerResolver(LayerResolver, object):
    __metaclass__ = ABCMeta

    class Descriptor(LayerDescriptor):
        def __init__(self, name, nodes, depth, on_value, off_value, axis, output_names=None):
            super(OneHotLayerResolver.Descriptor, self).__init__('OneHot', name, nodes, output_names=output_names)
            self.depth = depth
            self.on_value = on_value
            self.off_value = off_value
            self.axis = axis

        def is_input_tensor(self, op, tensor):
            return tensor == op.inputs[0]

    def __init__(self):
        sequence_one_hot = GraphSequence([
            ConverterSequenceNode('root', ['OneHot']),
            NonConsumableConverterSequenceNode('indices', ['?']),
            NonConsumableConverterSequenceNode('depth', ['?']),
            NonConsumableConverterSequenceNode('on_value', ['?']),
            NonConsumableConverterSequenceNode('off_value', ['?'])
        ])
        sequence_one_hot.set_inputs('root', ['indices', 'depth', 'on_value', 'off_value'])
        sequence_one_hot.set_outputs(['root'])

        self.sequences = [sequence_one_hot]

    def resolve_layer(self, graph_matcher, graph_helper):
        potential_descriptors = []
        for sequence in self.sequences:
            for match in graph_matcher.match_sequence(sequence):
                one_hot_op = match['root']
                consumed_nodes = match.consumed_nodes
                axis = one_hot_op.get_attr('axis')

                _, depth_tensor, on_value_tensor, off_value_tensor = GraphHelper.get_op_input_tensors(one_hot_op, ('?', '?', '?', '?'))

                depth = graph_helper.evaluate_tensor_output(depth_tensor)
                on_value = graph_helper.evaluate_tensor_output(on_value_tensor)
                off_value = graph_helper.evaluate_tensor_output(off_value_tensor)

                one_hot_descriptor = OneHotLayerResolver.Descriptor(
                    str(one_hot_op.name), consumed_nodes, depth, on_value, off_value, axis)
                potential_descriptors.append(one_hot_descriptor)

        return potential_descriptors


class OneHotLayerBuilder(LayerBuilder):

    def build_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type ir_graph: converters.common.converter_ir.op_graph.IROpGraph
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: OneHotLayerResolver.Descriptor
        :rtype: int
        """
        input_name = self.get_input_name(converter_context, descriptor, input_descriptors)
        return ir_graph.add(OneHotOp(descriptor.layer_name,
                                     depth=descriptor.depth,
                                     on_value=descriptor.on_value,
                                     off_value=descriptor.off_value,
                                     axis=descriptor.axis),
                            input_names=input_name,
                            output_names=descriptor.output_names)
