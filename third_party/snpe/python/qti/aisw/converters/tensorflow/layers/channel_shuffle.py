# =============================================================================
#
#  Copyright (c) 2018-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import sys

from qti.aisw.converters.common.utils import code_to_message
from qti.aisw.converters.common.converter_ir.op_adapter import ChannelShuffleOp
from qti.aisw.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from qti.aisw.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    NonConsumableConverterSequenceNode,
    GraphSequence
)
from qti.aisw.converters.tensorflow.util import ConverterError


class ChannelShuffleLayerResolver(LayerResolver, object):

    class Descriptor(LayerDescriptor):
        def __init__(self, name, nodes, groups, output_names=None):
            super(ChannelShuffleLayerResolver.Descriptor, self).__init__('ChannelShuffle', name, nodes, output_names=output_names)
            self.groups = groups

    def __init__(self):
        self.sequence = GraphSequence([
            ConverterSequenceNode('reshape_out', ['Reshape']),
            ConverterSequenceNode('transpose', ['Transpose']),
            ConverterSequenceNode('reshape_in', ['Reshape']),
            ConverterSequenceNode('shape_in', ['Const']),
            ConverterSequenceNode('order', ['Const']),
            ConverterSequenceNode('shape_out', ['Const']),
            NonConsumableConverterSequenceNode('input', ['?']),
        ])
        self.sequence.set_inputs('reshape_out', ['shape_out', 'transpose'])
        self.sequence.set_inputs('transpose', ['order', 'reshape_in'])
        self.sequence.set_inputs('reshape_in', ['shape_in', 'input'])
        self.sequence.set_outputs(['reshape_out'])

    def resolve_layer(self, graph_matcher, graph_helper):
        descriptors = []
        for match in graph_matcher.match_sequence(self.sequence):
            input_op = match['input']
            reshape_out_op = match['reshape_out']
            reshape_in_op = match['reshape_in']
            transpose_op = match['transpose']

            input_shape = graph_helper.get_op_output_shape(input_op)
            reshape_in_shape = graph_helper.get_op_output_shape(reshape_in_op)
            transpose_shape = graph_helper.get_op_output_shape(transpose_op)
            reshape_out_shape = graph_helper.get_op_output_shape(reshape_out_op)

            if len(reshape_in_shape) < 2:
                continue

            num_channels = input_shape[-1]
            num_groups = reshape_in_shape[-2]
            num_channels_prime = num_channels / num_groups

            if num_channels % num_groups != 0:
                continue

            is_channel_shuffle = True
            # first reshape must divide the channel dimension to [num_groups, num_channels_prime]
            is_channel_shuffle &= reshape_in_shape == input_shape[:-1] + [num_groups, num_channels_prime]
            # transpose must permute the last two dimensions only
            is_channel_shuffle &= transpose_shape == input_shape[:-1] + [num_channels_prime, num_groups]
            # output shape must be equal to the input shape
            is_channel_shuffle &= reshape_out_shape == input_shape

            if not is_channel_shuffle:
                continue

            consumed_nodes = match.consumed_nodes
            descriptors.append(ChannelShuffleLayerResolver.Descriptor(
                str(reshape_out_op.name), consumed_nodes, num_groups,
                output_names=[str(reshape_out_op.outputs[0].name)])
            )

        return descriptors


class ChannelShuffleLayerBuilder(LayerBuilder):
    def build_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type ir_graph: converters.common.converter_ir.op_graph.IROpGraph
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: ChannelShuffleLayerResolver.Descriptor
        :rtype: int
        """
        input_name = self.get_input_name(converter_context, descriptor, input_descriptors)
        output_name = descriptor.output_names[0]
        return ir_graph.add(ChannelShuffleOp(descriptor.layer_name,
                                             groups=descriptor.groups),
                            input_name,
                            output_name)

