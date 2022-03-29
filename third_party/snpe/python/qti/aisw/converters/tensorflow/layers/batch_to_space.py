# =============================================================================
#
#  Copyright (c) 2018-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from qti.aisw.converters.common.converter_ir.op_adapter import BatchToSpaceOp
from qti.aisw.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from qti.aisw.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    GraphSequence,
    NonConsumableConverterSequenceNode
)
from qti.aisw.converters.common.utils.converter_utils import *


class BatchToSpaceLayerResolver(LayerResolver, object):

    class Descriptor(LayerDescriptor):
        def __init__(self, layer_type, name, nodes, block_shape, crops):
            super(BatchToSpaceLayerResolver.Descriptor, self).__init__(layer_type, name, nodes)
            self.block_shape = block_shape
            self.crops = crops

        def is_input_tensor(self, op, tensor):
            return tensor == op.inputs[0]

    def __init__(self):
        self.sequence = GraphSequence([
            ConverterSequenceNode('root', ['BatchToSpaceND']),
            NonConsumableConverterSequenceNode('input', ['?']),
            NonConsumableConverterSequenceNode('block_shape', ['?']),
            NonConsumableConverterSequenceNode('crops', ['?']),
        ])

        self.sequence.set_inputs('root', ['input', 'block_shape', 'crops'])
        self.sequence.set_outputs(['root'])

    def resolve_layer(self, graph_matcher, graph_helper):
        matches = graph_matcher.match_sequence(self.sequence)
        potential_descriptors = []
        for match in matches:
            batch_to_space_op = match['root']

            block_shape_op = match['block_shape']
            crops_op = match['crops']

            block_shape_tensor, _, _ = graph_helper.get_static_data_info(batch_to_space_op, block_shape_op.outputs[0])
            crops_tensor, _, _ = graph_helper.get_static_data_info(batch_to_space_op, crops_op.outputs[0])
            consumed_nodes = match.consumed_nodes

            potential_descriptors.append(BatchToSpaceLayerResolver.Descriptor('BatchToSpace',
                                                                              str(batch_to_space_op.name),
                                                                              consumed_nodes,
                                                                              block_shape_tensor,
                                                                              crops_tensor))

        return potential_descriptors


class BatchToSpaceLayerBuilder(LayerBuilder):
    def build_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type ir_graph: converters.common.converter_ir.op_graph.IROpGraph
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: BatchToSpaceLayerResolver.Descriptor
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :rtype: int
        """
        input_name = self.get_input_name(converter_context, descriptor, input_descriptors)
        output_name = descriptor.output_names[0]
        return ir_graph.add(BatchToSpaceOp(name=descriptor.layer_name,
                                           block_shape=descriptor.block_shape,
                                           crops=descriptor.crops),
                            input_names=[input_name],
                            output_names=[output_name])
