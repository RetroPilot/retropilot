# =============================================================================
#
#  Copyright (c) 2019-2020 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import numpy as np

from qti.aisw.converters.common.utils.translation_utils import compare_values
from qti.aisw.converters.tensorflow.layers.constant import ConstantLayerResolver
from qti.aisw.converters.common.converter_ir.op_adapter import CropAndResizeOp
from qti.aisw.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from qti.aisw.converters.tensorflow.util import GraphHelper, get_const_op_value
from qti.aisw.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    NonConsumableConverterSequenceNode,
    GraphSequence
)
from qti.aisw.converters.tensorflow.util import ConverterError
from qti.aisw.converters.common.utils import code_to_message


class CropAndResizeLayerResolver(LayerResolver, object):

    class Descriptor(LayerDescriptor):
        def __init__(self, name, nodes, num_boxes, crop_height, crop_width, interpolation_method,
                     extrapolation_value, output_names=None):
            super(CropAndResizeLayerResolver.Descriptor, self).__init__('CropAndResize', name,
                                                               nodes, output_names=output_names)
            self.num_boxes = num_boxes
            self.crop_height = crop_height
            self.crop_width = crop_width
            self.interpolation_method = interpolation_method
            self.extrapolation_value = extrapolation_value

        def is_input_tensor(self, op, tensor):
            # Ignores a static crop_size input which has already been consumed by the resolver
            if tensor.op.type == "Const" and compare_values(get_const_op_value(tensor.op), np.array([self.crop_width,
                                                                                                    self.crop_height])):
                return False
            return True

    def __init__(self):
        sequence_crop_and_resize = GraphSequence([
            NonConsumableConverterSequenceNode('input', ['?']),
            NonConsumableConverterSequenceNode('boxes', ['?']),
            NonConsumableConverterSequenceNode('box_ind', ['?']),
            NonConsumableConverterSequenceNode('crop_size', ['?']),
            ConverterSequenceNode('crop_and_resize', ['CropAndResize']),
        ])
        sequence_crop_and_resize.set_inputs('crop_and_resize', ['input', 'boxes', 'box_ind', 'crop_size'])
        sequence_crop_and_resize.set_outputs(['crop_and_resize'])

        self.sequences = [sequence_crop_and_resize]

    def resolve_layer(self, graph_matcher, graph_helper):
        potential_descriptors = []
        for sequence in self.sequences:
            matches = graph_matcher.match_sequence(sequence)
            for match in matches:
                crop_and_resize = match['crop_and_resize']

                try:
                   _, boxes, box_ind, crop_size = GraphHelper.get_op_input_tensors(crop_and_resize, ('?', '?', '?', 'Const'))
                except TensorNotFoundError:
                    raise ConverterError(
                        code_to_message.get_error_message('ERROR_TF_RESOLVE_CROP_AND_RESIZE_SIZE_NOT_CONST'))

                box_ind_value = graph_helper.evaluate_tensor_output(box_ind).astype('uint32')
                box_ind_shape = graph_helper.get_op_output_shape(box_ind.op)
                boxes_value = graph_helper.evaluate_tensor_output(boxes)
                boxes_shape = graph_helper.get_op_output_shape(boxes.op)

                if len(box_ind_shape) == 1:
                    box_ind_shape = box_ind_shape[-1]
                else:
                    raise ConverterError(code_to_message.get_error_message('ERROR_TF_RESOLVE_CROP_AND_RESIZE_NUM_BOXES'))

                crop_size_value = graph_helper.evaluate_tensor_output(crop_size)
                if crop_size_value.size != 2:
                    raise ConverterError(
                        code_to_message.get_error_message('ERROR_TF_RESOLVE_CROP_AND_RESIZE_SIZE')(crop_size_value.size))

                consumed_nodes = match.consumed_nodes

                interpolation_method = crop_and_resize.get_attr('method')
                supported_interpolation = ['BILINEAR', 'NEAREST']
                if interpolation_method.decode().upper() not in supported_interpolation:
                    raise ConverterError(code_to_message.get_error_message("ERROR_TF_CROPANDRESIZE_INTERPOLATION_UNKNOWN")
                                         (interpolation_method.decode()))

                extrapolation_value = float(crop_and_resize.get_attr('extrapolation_value'))

                crop_and_resize_descriptor = CropAndResizeLayerResolver.Descriptor(
                    str(crop_and_resize.name), consumed_nodes, box_ind_shape, crop_size_value[1],
                    crop_size_value[0], interpolation_method, extrapolation_value)
                potential_descriptors.append(crop_and_resize_descriptor)

                if box_ind.op.type == 'Const':

                    constant_descriptor = ConstantLayerResolver.Descriptor(str(box_ind.op),
                                                                           [box_ind.op],
                                                                           box_ind_value,
                                                                           box_ind_shape,
                                                                           crop_and_resize_descriptor,
                                                                           quantizable=False)
                    potential_descriptors.append(constant_descriptor)

                if boxes.op.type == 'Const':
                    constant_descriptor = ConstantLayerResolver.Descriptor(str(boxes.op),
                                                                           [boxes.op],
                                                                           boxes_value,
                                                                           boxes_shape,
                                                                           crop_and_resize_descriptor)
                    potential_descriptors.append(constant_descriptor)

        return potential_descriptors


class CropAndResizeLayerBuilder(LayerBuilder):
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
        return ir_graph.add(CropAndResizeOp(descriptor.layer_name,
                                            num_boxes=descriptor.num_boxes,
                                            crop_height=descriptor.crop_height,
                                            crop_width=descriptor.crop_width,
                                            interpolation_method=descriptor.interpolation_method,
                                            extrapolation_value=descriptor.extrapolation_value),
                            input_names=input_names,
                            output_names=descriptor.output_names)
