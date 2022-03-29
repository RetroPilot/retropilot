# =============================================================================
#
#  Copyright (c) 2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import numpy as np

from qti.aisw.converters.common.converter_ir.op_adapter import CastOp
from qti.aisw.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from qti.aisw.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    NonConsumableConverterSequenceNode,
    GraphSequence
)


class CastLayerResolver(LayerResolver, object):
    TF_ATTRIBUTE_SRC_TENSOR_TYPE = "SrcT"
    TF_ATTRIBUTE_DST_TENSOR_TYPE = "DstT"

    class Descriptor(LayerDescriptor):
        def __init__(self, name, nodes, from_type, to_type):
            super(CastLayerResolver.Descriptor, self).__init__('Cast', name, nodes)
            self.from_type = from_type
            self.to_type = to_type

    def __init__(self):
        self.sequence = GraphSequence([
            ConverterSequenceNode('root', ['Cast']),
            NonConsumableConverterSequenceNode('any', ['?'])
        ])
        self.sequence.set_inputs('root', ['any'])
        self.sequence.set_outputs(['root'])

    def resolve_layer(self, graph_matcher, graph_helper):
        matches = graph_matcher.match_sequence(self.sequence)
        descriptors = []
        for match in matches:
            cast_op = match['root']
            from_type = cast_op.get_attr(self.TF_ATTRIBUTE_SRC_TENSOR_TYPE).name
            to_type = cast_op.get_attr(self.TF_ATTRIBUTE_DST_TENSOR_TYPE).name
            cast_desc = CastLayerResolver.Descriptor(str(cast_op.name), match.consumed_nodes, from_type, to_type)
            descriptors.append(cast_desc)

        return descriptors


class CastLayerBuilder(LayerBuilder):
    def build_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type ir_graph: converters.common.converter_ir.op_graph.IROpGraph
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: ReshapeLayerResolver.Descriptor
        :rtype: int
        """
        input_names = self.get_input_names(converter_context, descriptor, input_descriptors)
        return ir_graph.add(CastOp(descriptor.layer_name, from_type=descriptor.from_type, to_type=descriptor.to_type),
                            input_names[0],
                            descriptor.output_names[0])
