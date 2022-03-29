# =============================================================================
#
#  Copyright (c) 2016-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from qti.aisw.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from qti.aisw.converters.tensorflow.sequences.ignored import (
    ignored_sequence_1,
    dropout_sequence,
    dropout_cell_sequence,
    real_div_sequence,
    placeholder_with_default_sequence,
    pack_4_strided_slice_sequence,
    pack_const_mul_strided_slice_sequence,
    pack_3_strided_slice_sequence,
    pack_strided_slice_mul_sequence,
    shape_strided_slice_pack_sequence,
    shape_strided_slice_pack_sequence1,
    shape_strided_slice_pack_sequence2,
    unused_pattern_feeding_into_multi_class_nms_sequence,
    unused_pattern_feeding_into_multi_class_nms_sequence1,
    unused_pattern_feeding_into_multi_class_nms_sequence2,
)


class IgnoredLayersResolver(LayerResolver, object):
    class Descriptor(LayerDescriptor):
        def __init__(self, name, nodes):
            super(IgnoredLayersResolver.Descriptor, self).__init__('IgnoredLayer', name, nodes)
            # define pattern one to be ignored

    def __init__(self):
        self.sequences = [
            ignored_sequence_1,
            dropout_sequence,
            dropout_cell_sequence,
            real_div_sequence,
            placeholder_with_default_sequence,
            pack_4_strided_slice_sequence,
            pack_const_mul_strided_slice_sequence,
            pack_3_strided_slice_sequence,
            pack_strided_slice_mul_sequence,
            shape_strided_slice_pack_sequence,
            shape_strided_slice_pack_sequence1,
            shape_strided_slice_pack_sequence2,
            unused_pattern_feeding_into_multi_class_nms_sequence,
            unused_pattern_feeding_into_multi_class_nms_sequence1,
            unused_pattern_feeding_into_multi_class_nms_sequence2,
        ]

    def resolve_layer(self, graph_matcher, graph_helper):
        descriptors = []
        for pattern_output_nodes in self.sequences:
            matches = graph_matcher.match_sequence(pattern_output_nodes)
            if len(matches) == 0:
                continue

            for match in matches:
                consumed_nodes = match.consumed_nodes
                d = IgnoredLayersResolver.Descriptor(str(consumed_nodes[0].name), consumed_nodes)
                descriptors.append(d)

        return descriptors


class IgnoredLayersBuilder(LayerBuilder):

    def transform_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):
        descriptor.set_ignored(True)

    def build_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type ir_graph: converters.common.converter_ir.op_graph.IROpGraph
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: IgnoredLayersResolver.Descriptor
        :rtype: int
        """
        return None
