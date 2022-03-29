# =============================================================================
#
#  Copyright (c) 2018-2020 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from qti.aisw.converters.common.converter_ir.op_adapter import ImageProjectiveTransformOp
from qti.aisw.converters.common.utils import code_to_message
from qti.aisw.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from qti.aisw.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    GraphSequence,
)
from qti.aisw.converters.tensorflow.util import ConverterError


class ImageProjectiveTransformLayerResolver(LayerResolver, object):
    class Descriptor(LayerDescriptor):
        def __init__(self, name, operations, interpolation_mode, output_names=None):
            super(ImageProjectiveTransformLayerResolver.Descriptor, self).__init__(
                'ImageProjectiveTransform', name, operations, output_names=output_names)
            self.interpolation_mode = interpolation_mode

    def __init__(self):
        self.sequence = GraphSequence([ConverterSequenceNode('root', ['ImageProjectiveTransform'])])
        self.sequence.set_outputs(['root'])

    def resolve_layer(self, graph_matcher, graph_helper):
        potential_descriptors = []
        matches = graph_matcher.match_sequence(self.sequence)
        for match in matches:
            image_proj_transform = match['root']

            output_op_nodes_names = [str(image_proj_transform.outputs[0].name)]
            consumed_nodes = match.consumed_nodes

            interpolation_mode = str(image_proj_transform.get_attr('interpolation').decode('utf-8'))
            if interpolation_mode not in ["BILINEAR", "NEAREST"]:
                raise ConverterError(
                    code_to_message.get_error_message('ERROR_TF_RESOLVE_IMAGE_TRANSFORM_INTERPOLATION'))

            potential_descriptors.append(
                ImageProjectiveTransformLayerResolver.Descriptor(str(image_proj_transform.name),
                                                                 consumed_nodes, interpolation_mode,
                                                                 output_names=output_op_nodes_names)
            )
        return potential_descriptors


class ImageProjectiveTransformLayerBuilder(LayerBuilder):
    def build_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type ir_graph: converters.common.converter_ir.op_graph.IROpGraph
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: ImageProjectiveTransformLayerResolver.Descriptor
        :rtype: int
        """
        input_names = self.get_input_names(converter_context, descriptor, input_descriptors)

        return ir_graph.add(ImageProjectiveTransformOp(name=descriptor.layer_name,
                                                       interpolation_mode=descriptor.interpolation_mode),
                            input_names=input_names,
                            output_names=descriptor.output_names)
