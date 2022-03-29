# =============================================================================
#
#  Copyright (c) 2015-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from qti.aisw.converters.common.converter_ir.op_adapter import CropOp
from qti.aisw.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from qti.aisw.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    NonConsumableConverterSequenceNode,
    GraphSequence
)


class CropLayerResolver(LayerResolver, object):

    class Descriptor(LayerDescriptor):
        def __init__(self, name, nodes, offset, counts, size, input_op, output_names=None):
            super(CropLayerResolver.Descriptor, self).__init__('Crop', name, nodes, output_names=output_names)
            self.offset = offset
            self.size = size
            self.counts = counts
            self.input_op = input_op

        def is_input_op(self, op):
            return op == self.input_op

    def __init__(self):
        self.sequence = GraphSequence([
            ConverterSequenceNode('root', ['Slice']),
            NonConsumableConverterSequenceNode('input', ['?']),
            NonConsumableConverterSequenceNode('offsets', ['?']),
            NonConsumableConverterSequenceNode('size', ['?']),
        ])
        self.sequence.set_inputs('root', ['input', 'offsets', 'size'])
        self.sequence.set_outputs(['root'])

    def resolve_layer(self, graph_matcher, graph_helper):
        matches = graph_matcher.match_sequence(self.sequence)
        descriptors = []
        for match in matches:
            slice_op = match['root']
            input_shape = graph_helper.get_op_output_shape(match['input'])
            offset = graph_helper.evaluate_tensor_output(match['offsets'].outputs[0])
            size_tensor = match['size'].outputs[0]
            size, size_shape, size_consumed_nodes = graph_helper.get_static_data_info(slice_op, size_tensor)
            counts = graph_helper.evaluate_tensor_output(match['size'].outputs[0]).copy()
            for index in range(0, len(size)):
                if size[index] == -1:
                    size[index] = input_shape[index] - offset[index]
                    counts[index] = 0
            consumed_nodes = match.consumed_nodes
            if size_consumed_nodes:
                size_consumed_nodes.extend(consumed_nodes)
                consumed_nodes = size_consumed_nodes
            input_op = slice_op

            crop_desc = CropLayerResolver.Descriptor(str(slice_op.name), consumed_nodes, offset, counts, size, input_op)
            descriptors.append(crop_desc)

        return descriptors


class CropLayerBuilder(LayerBuilder):
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
        return ir_graph.add(CropOp(descriptor.output_names[0],
                                   descriptor.offset.tolist(),
                                   descriptor.counts.tolist(),
                                   descriptor.size.tolist()),
                            input_names[0],
                            descriptor.output_names[0])
