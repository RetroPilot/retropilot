# =============================================================================
#
#  Copyright (c) 2018-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from qti.aisw.converters.tensorflow.layers.constant import ConstantLayerResolver
from qti.aisw.converters.common.converter_ir.op_adapter import ExtractGlimpseOp
from qti.aisw.converters.common.utils import code_to_message
from qti.aisw.converters.tensorflow.util import GraphHelper
from qti.aisw.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from qti.aisw.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    GraphSequence,
    NonConsumableConverterSequenceNode
)
from qti.aisw.converters.tensorflow.util import ConverterError


class ExtractGlimpseLayerResolver(LayerResolver, object):
    class Descriptor(LayerDescriptor):
        def __init__(self, name, operations, glimpse_width, glimpse_height, centered, normalized,
                     noise,output_names=None):
            super(ExtractGlimpseLayerResolver.Descriptor, self).__init__('ExtractGlimpse', name,
                                                                         operations,
                                                                         output_names=output_names)
            self.glimpse_width = glimpse_width
            self.glimpse_height = glimpse_height
            self.centered = centered
            self.normalized = normalized
            self.noise = noise

    def __init__(self):
        sequence_extract_glimpse = GraphSequence([
            NonConsumableConverterSequenceNode('input', ['?']),
            NonConsumableConverterSequenceNode('offsets', ['?']),
            ConverterSequenceNode('size', ['Const']),
            ConverterSequenceNode('extract_glimpse', ['ExtractGlimpse'])
        ])
        sequence_extract_glimpse.set_inputs('extract_glimpse', ['input', 'size', 'offsets'])
        sequence_extract_glimpse.set_outputs(['extract_glimpse'])

        self.sequences = [sequence_extract_glimpse]

    def resolve_layer(self, graph_matcher, graph_helper):
        potential_descriptors = []
        for sequence in self.sequences:
            matches = graph_matcher.match_sequence(sequence)
            for match in matches:
                extract_glimpse = match['extract_glimpse']
                _, _, offsets = GraphHelper.get_op_input_tensors(extract_glimpse, ('?', 'Const', '?'))
                offsets_value = graph_helper.evaluate_tensor_output(offsets)
                offsets_shape = graph_helper.get_op_output_shape(offsets.op)
                size = match['size']
                size_value = graph_helper.evaluate_tensor_output(size.outputs[0])
                if size_value.size != 2:
                    raise ConverterError(
                        code_to_message.get_error_message('ERROR_TF_RESOLVE_EXTRACT_GLIMPSE_SIZE'))

                output_op_nodes_names = [str(extract_glimpse.outputs[0].name)]
                consumed_nodes = match.consumed_nodes
                centered = bool(extract_glimpse.get_attr('centered'))
                normalized = bool(extract_glimpse.get_attr('normalized'))
                noise = bool(extract_glimpse.get_attr('uniform_noise'))
                if noise:
                    noise = ExtractGlimpseOp.NoiseType.UNIFORM
                else:
                    noise = ExtractGlimpseOp.NoiseType.GAUSSIAN

                extract_glimpse_descriptor = ExtractGlimpseLayerResolver.Descriptor(str(extract_glimpse.name), consumed_nodes,
                                                           size_value[1], size_value[0],
                                                           centered,normalized, noise,
                                                           output_names=output_op_nodes_names)
                potential_descriptors.append(extract_glimpse_descriptor)

                if offsets.op.type == 'Const':
                    constant_descriptor = ConstantLayerResolver.Descriptor(str(offsets.op),
                                                                           [offsets.op],
                                                                           offsets_value,
                                                                           offsets_shape,
                                                                           extract_glimpse_descriptor)
                    potential_descriptors.append(constant_descriptor)
        return potential_descriptors


class ExtractGlimpseLayerBuilder(LayerBuilder):
    def build_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type ir_graph: converters.common.converter_ir.op_graph.IROpGraph
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: ExtractGlimpseLayerResolver.Descriptor
        :rtype: int
        """
        input_names = self.get_input_names(converter_context, descriptor, input_descriptors)
        output_name = descriptor.output_names[0]
        return ir_graph.add(ExtractGlimpseOp(name=descriptor.layer_name,
                                             glimpse_width=descriptor.glimpse_width,
                                             glimpse_height=descriptor.glimpse_height,
                                             centered=descriptor.centered,
                                             normalized=descriptor.normalized,
                                             noise=descriptor.noise),
                            input_names=input_names,
                            output_names=output_name)
