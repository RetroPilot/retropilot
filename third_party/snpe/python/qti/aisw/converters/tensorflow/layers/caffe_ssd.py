# =============================================================================
#
#  Copyright (c) 2019-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

"""
This resolver/builder for caffe style ssd is used to support strictly
the Keras implementation found here: https://github.com/pierluigiferrari/ssd_keras/blob/master/models/keras_ssd300.py
"""

import sys
import numpy as np
from qti.aisw.converters.common.converter_ir.op_adapter import DetectionOutputOp, ReshapeOp
from qti.aisw.converters.common.utils.code_to_message import get_error_message
from qti.aisw.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from qti.aisw.converters.tensorflow.util import ConverterError
from qti.aisw.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    NonConsumableConverterSequenceNode,
    GraphSequence
)
from qti.aisw.converters.tensorflow.layers.convolution import ConvolutionLayerResolver
from qti.aisw.converters.common.utils.converter_utils import log_assert


class CaffeSsdLayerResolver(LayerResolver, object):
    class Descriptor(LayerDescriptor):
        def __init__(self, name, nodes, ssd_op, priorboxes_data, score_threshold, iou_threshold, num_classes,
                     output_dims, keep_top_k, nms_top_k):
            super(CaffeSsdLayerResolver.Descriptor, self).__init__('DetectionOutput', name, nodes)
            self.ssd_op = ssd_op
            self.priorboxes_data = priorboxes_data
            self.score_threshold = score_threshold
            self.iou_threshold = iou_threshold
            self.num_classes = num_classes
            self.output_dims = output_dims
            self.keep_top_k = keep_top_k
            self.nms_top_k = nms_top_k

        @property
        def output_names(self):
            """
            :rtype: [str]
            """
            return [self.layer_name]

        def is_output_op(self, op):
            return op == self.ssd_op

    def is_final_resolution(self):
        return True

    @staticmethod
    def convert_centroids_to_corner(tensor):
        """
        Convert coordinates for axis-aligned 2D boxes between two coordinate formats.
            2) (cx, cy, w, h) - the 'centroids' format -> (xmin, ymin, xmax, ymax) - the 'corners' format
        Arguments:
            tensor (ndarray): A Numpy nD array containing the four consecutive coordinates
                to be converted somewhere in the last axis.
        Returns:
            A Numpy nD array, a copy of the input tensor with the converted coordinates
            in place of the original coordinates and the unaltered elements of the original
            tensor elsewhere.
        """

        # code used as-is from implementation
        ind = 0
        tensor1 = np.copy(tensor).astype(np.float)
        tensor1[..., ind] = tensor[..., ind] - tensor[..., ind + 2] / 2.0  # Set x_min
        tensor1[..., ind + 1] = tensor[..., ind + 1] - tensor[..., ind + 3] / 2.0  # Set y_min
        tensor1[..., ind + 2] = tensor[..., ind] + tensor[..., ind + 2] / 2.0  # Set x_max
        tensor1[..., ind + 3] = tensor[..., ind + 1] + tensor[..., ind + 3] / 2.0  # Set y_max

        return tensor1

    def resolve_layer(self, graph_matcher, graph_helper):
        descriptors = []
        caffe_ssd_sequence = GraphSequence([
            ConverterSequenceNode('anchors1/Tile', ['Tile']),
            ConverterSequenceNode('anchors2/Tile', ['Tile']),
            ConverterSequenceNode('anchors3/Tile', ['Tile']),
            ConverterSequenceNode('anchors4/Tile', ['Tile']),
            ConverterSequenceNode('anchors5/Tile', ['Tile']),
            ConverterSequenceNode('anchors6/Tile', ['Tile']),
            ConverterSequenceNode('anchors1_reshape/Reshape', ['Reshape']),
            ConverterSequenceNode('anchors2_reshape/Reshape', ['Reshape']),
            ConverterSequenceNode('anchors3_reshape/Reshape', ['Reshape']),
            ConverterSequenceNode('anchors4_reshape/Reshape', ['Reshape']),
            ConverterSequenceNode('anchors5_reshape/Reshape', ['Reshape']),
            ConverterSequenceNode('anchors6_reshape/Reshape', ['Reshape']),
            ConverterSequenceNode('classes_softmax/truediv', ['RealDiv']),
            ConverterSequenceNode('boxes_concat/concat', ['ConcatV2']),
            ConverterSequenceNode('anchors_concat/concat', ['ConcatV2']),
            ConverterSequenceNode('predictions/concat', ['ConcatV2']),
            ConverterSequenceNode('decoded_predictions/strided_slice_14', ['StridedSlice']),
            ConverterSequenceNode('decoded_predictions/concat', ['ConcatV2']),
            ConverterSequenceNode('decoded_predictions/loop_over_batch/Shape', ['Shape']),
            ConverterSequenceNode('decoded_predictions/loop_over_batch/strided_slice', ['StridedSlice']),
            ConverterSequenceNode('decoded_predictions/loop_over_batch/while/Exit_2', ['Exit']),
            ConverterSequenceNode('decoded_predictions/loop_over_batch/TensorArrayStack/TensorArraySizeV3',
                                  ['TensorArraySizeV3']),
            ConverterSequenceNode('decoded_predictions/loop_over_batch/TensorArrayStack/range', ['Range']),
            ConverterSequenceNode('decoded_predictions/loop_over_batch/TensorArray_1', ['TensorArrayV3']),
            ConverterSequenceNode('decoded_predictions/loop_over_batch/TensorArrayStack/TensorArrayGatherV3',
                                  ['TensorArrayGatherV3']),
            NonConsumableConverterSequenceNode('priorbox1', ['Const']),
            NonConsumableConverterSequenceNode('priorbox2', ['Const']),
            NonConsumableConverterSequenceNode('priorbox3', ['Const']),
            NonConsumableConverterSequenceNode('priorbox4', ['Const']),
            NonConsumableConverterSequenceNode('priorbox5', ['Const']),
            NonConsumableConverterSequenceNode('priorbox6', ['Const']),
            NonConsumableConverterSequenceNode('stub_1', ['?']),
            NonConsumableConverterSequenceNode('stub_2', ['?']),
            NonConsumableConverterSequenceNode('stub_3', ['?']),
            NonConsumableConverterSequenceNode('stub_4', ['?']),
            NonConsumableConverterSequenceNode('stub_5', ['?']),
            NonConsumableConverterSequenceNode('stub_6', ['?']),
            NonConsumableConverterSequenceNode('stub_7', ['?']),
            NonConsumableConverterSequenceNode('stub_8', ['?']),
            NonConsumableConverterSequenceNode('stub_9', ['?']),
            NonConsumableConverterSequenceNode('stub_10', ['?']),
            NonConsumableConverterSequenceNode('stub_11', ['?']),
            NonConsumableConverterSequenceNode('stub_12', ['?']),
            NonConsumableConverterSequenceNode('stub_13', ['?']),
            NonConsumableConverterSequenceNode('stub_14', ['?']),
            NonConsumableConverterSequenceNode('stub_15', ['?']),
            NonConsumableConverterSequenceNode('stub_16', ['?']),
            NonConsumableConverterSequenceNode('stub_17', ['?']),
            NonConsumableConverterSequenceNode('stub_18', ['?']),
            NonConsumableConverterSequenceNode('stub_19', ['?']),
            NonConsumableConverterSequenceNode('stub_20', ['?']),
            NonConsumableConverterSequenceNode('stub_21', ['?']),
            NonConsumableConverterSequenceNode('stub_22', ['?']),
            NonConsumableConverterSequenceNode('stub_23', ['?']),
            NonConsumableConverterSequenceNode('stub_24', ['?']),
            NonConsumableConverterSequenceNode('stub_25', ['?']),
            NonConsumableConverterSequenceNode('stub_26', ['?']),
            NonConsumableConverterSequenceNode('stub_27', ['?']),
            NonConsumableConverterSequenceNode('stub_28', ['?']),
            NonConsumableConverterSequenceNode('stub_29', ['?']),
        ])
        caffe_ssd_sequence.set_inputs('anchors1/Tile', ['priorbox1', 'stub_1'])
        caffe_ssd_sequence.set_inputs('anchors2/Tile', ['priorbox2', 'stub_2'])
        caffe_ssd_sequence.set_inputs('anchors3/Tile', ['priorbox3', 'stub_3'])
        caffe_ssd_sequence.set_inputs('anchors4/Tile', ['priorbox4', 'stub_4'])
        caffe_ssd_sequence.set_inputs('anchors5/Tile', ['priorbox5', 'stub_5'])
        caffe_ssd_sequence.set_inputs('anchors6/Tile', ['priorbox6', 'stub_6'])
        caffe_ssd_sequence.set_inputs('anchors1_reshape/Reshape', ['anchors1/Tile', 'stub_7'])
        caffe_ssd_sequence.set_inputs('anchors2_reshape/Reshape', ['anchors2/Tile', 'stub_8'])
        caffe_ssd_sequence.set_inputs('anchors3_reshape/Reshape', ['anchors3/Tile', 'stub_9'])
        caffe_ssd_sequence.set_inputs('anchors4_reshape/Reshape', ['anchors4/Tile', 'stub_10'])
        caffe_ssd_sequence.set_inputs('anchors5_reshape/Reshape', ['anchors5/Tile', 'stub_11'])
        caffe_ssd_sequence.set_inputs('anchors6_reshape/Reshape', ['anchors6/Tile', 'stub_12'])
        caffe_ssd_sequence.set_inputs('anchors_concat/concat', ['anchors1_reshape/Reshape', 'anchors2_reshape/Reshape',
                                                                'anchors3_reshape/Reshape', 'anchors4_reshape/Reshape',
                                                                'anchors5_reshape/Reshape', 'anchors6_reshape/Reshape',
                                                                'stub_13'])
        caffe_ssd_sequence.set_inputs('predictions/concat', ['stub_14', 'stub_15', 'anchors_concat/concat', 'stub_16'])
        caffe_ssd_sequence.set_inputs('decoded_predictions/strided_slice_14', ['predictions/concat', 'stub_17',
                                                                               'stub_18', 'stub_19'])
        caffe_ssd_sequence.set_inputs('decoded_predictions/concat', ['decoded_predictions/strided_slice_14', 'stub_20',
                                                                     'stub_21', 'stub_22', 'stub_23', 'stub_24'])
        caffe_ssd_sequence.set_inputs('decoded_predictions/loop_over_batch/Shape', ['decoded_predictions/concat'])
        caffe_ssd_sequence.set_inputs('decoded_predictions/loop_over_batch/strided_slice',
                                      ['decoded_predictions/loop_over_batch/Shape', 'stub_25', 'stub_26', 'stub_27'])
        caffe_ssd_sequence.set_inputs('decoded_predictions/loop_over_batch/TensorArray_1',
                                      ['decoded_predictions/loop_over_batch/strided_slice'])
        caffe_ssd_sequence.set_inputs('decoded_predictions/loop_over_batch/TensorArrayStack/TensorArraySizeV3',
                                      ['decoded_predictions/loop_over_batch/TensorArray_1',
                                       'decoded_predictions/loop_over_batch/while/Exit_2'])
        caffe_ssd_sequence.set_inputs('decoded_predictions/loop_over_batch/TensorArrayStack/range',
                                      ['stub_28',
                                       'decoded_predictions/loop_over_batch/TensorArrayStack/TensorArraySizeV3',
                                       'stub_29'])
        caffe_ssd_sequence.set_inputs('decoded_predictions/loop_over_batch/TensorArrayStack/TensorArrayGatherV3',
                                      ['decoded_predictions/loop_over_batch/TensorArray_1',
                                       'decoded_predictions/loop_over_batch/TensorArrayStack/range',
                                       'decoded_predictions/loop_over_batch/while/Exit_2'])
        caffe_ssd_sequence.set_outputs(["decoded_predictions/loop_over_batch/TensorArrayStack/TensorArrayGatherV3"])

        matches = graph_matcher.match_sequence(caffe_ssd_sequence)
        for match in matches:
            ssd_op = match['decoded_predictions/loop_over_batch/TensorArrayStack/TensorArrayGatherV3']
            ssd_scope = '/'.join(ssd_op.name.split('/')[:1])
            detection_out_ops_map = {n.identifier: n.original_node
                                     for n in graph_matcher.graph if n.identifier.startswith(ssd_scope)}
            detection_out_ops = set(detection_out_ops_map.values())
            detection_out_ops.update(match.consumed_nodes)

            # get priorboxes to be embedded as part of detection_out op
            priorboxes = []
            priorboxes_variances = []
            # In the Keras implementation each priorbox op is packed as (box_coord1,variance1, box_coord2,variance2...)
            # snpe expects list of ALL prior boxes followed by ALL their variances. So need to unpack then concat.
            for i in range(1, 7):
                p_boxes, p_var = np.split(graph_helper.evaluate_tensor_output(match['priorbox' + str(i)].outputs[0]), 2, -1)
                # keras implementation has the coordinates at cx, cy, w, h, snpe needs x_min, y_min, x_max, y_max
                p_boxes = self.convert_centroids_to_corner(p_boxes)
                priorboxes.extend(p_boxes.flatten().tolist())
                priorboxes_variances.extend(p_var.flatten().tolist())
            priorboxes_data = priorboxes + priorboxes_variances

            # remove all ops related to priorbox
            priorboxes_concat_op = match['anchors1_reshape/Reshape']
            # determine scope name used to get all ops related to priorboxes
            if "anchors" in priorboxes_concat_op.name:
                priorboxes_scope = "anchors"
            elif "priorboxes" in priorboxes_concat_op.name:
                priorboxes_scope = "priorboxes"
            else:
                raise ConverterError(get_error_message('ERROR_TF_UNABLE_TO_DETERMINE_SCOPE_FOR_OP')
                                     (priorboxes_concat_op.name, "anchors or priorboxes in name"))

            priorboxes_ops_map = {n.identifier: n.original_node
                                  for n in graph_matcher.graph if n.identifier.startswith(priorboxes_scope)}
            priorboxes_ops = set(priorboxes_ops_map.values())
            detection_out_ops.update(priorboxes_ops)  # update total consumed nodes

            # extract attributes for SSD detection boxes
            num_classes = graph_helper.get_op_output_shape(match["decoded_predictions/concat"])[2] - 4
            keep_top_k = self._resolve_keep_top_k(graph_matcher, graph_helper)
            nms_top_k = self._resolve_nms_top_k(graph_matcher, graph_helper)
            score_threshold = self._resolve_score_threshold(graph_matcher, graph_helper)
            iou_threshold = self._resolve_iou_threshold(detection_out_ops_map, ssd_scope)
            output_dims = [graph_helper.get_op_output_shape(
                                    match["decoded_predictions/loop_over_batch/TensorArrayStack/TensorArrayGatherV3"])]
            if output_dims[0][-1] == 6:
                output_dims[0][-1] = 7
            elif output_dims[0][-1] != 7:
                raise ConverterError(get_error_message('ERROR_TF_SSD_CAFFE_CAN_NOT_RESOLVE_OUTPUT_DIM'))

            descriptors.append(CaffeSsdLayerResolver.Descriptor(ssd_scope, list(detection_out_ops), ssd_op,
                                                                priorboxes_data, score_threshold, iou_threshold,
                                                                num_classes, output_dims, keep_top_k, nms_top_k))
        return descriptors

    @staticmethod
    def _resolve_keep_top_k(graph_matcher, graph_helper):
        keep_top_k = [n.original_node for n in graph_matcher.graph if n.identifier.startswith("top_k")]
        log_assert(len(keep_top_k) == 1, get_error_message('ERROR_TF_SSD_CAFFE_CAN_NOT_RESOLVE_KEEP_TOP_K'))
        return graph_helper.evaluate_tensor_output(keep_top_k[0].outputs[0]).item()

    @staticmethod
    def _resolve_nms_top_k(graph_matcher, graph_helper):
        keep_top_k = [n.original_node for n in graph_matcher.graph if n.identifier.startswith("nms_max_output_size")]
        log_assert(len(keep_top_k) == 1, get_error_message('ERROR_TF_SSD_CAFFE_CAN_NOT_RESOLVE_NMS_TOP_K'))
        return graph_helper.evaluate_tensor_output(keep_top_k[0].outputs[0]).item()

    @staticmethod
    def _resolve_score_threshold(graph_matcher, graph_helper):
        # Note: The Keras implementation doesnt use the score_threshold in tf.non_maximum_suppression, instead have
        #       their own filtering of scores using a confidence_threshold. Hence using that here.
        score_threshold = [n.original_node for n in graph_matcher.graph if n.identifier.startswith("confidence_thresh")]
        log_assert(len(score_threshold) == 1, get_error_message('ERROR_TF_SSD_CAFFE_CAN_NOT_RESOLVE_SCORE_THRESHOLD'))
        return graph_helper.evaluate_tensor_output(score_threshold[0].outputs[0]).item()

    @staticmethod
    def _resolve_iou_threshold(detection_out_ops_map, ssd_scope):
        # note spelling error on non_maximum_suppresion, second 's' is missing(from keras implementation)
        iou_op_name = '{}/loop_over_batch/while/loop_over_classes/while/cond/non_maximum_suppresion/iou_threshold' \
            .format(ssd_scope)
        iou_op = detection_out_ops_map.get(iou_op_name, None)
        if iou_op is not None and iou_op.type == 'Const':
            # evaluating the iou_op's output tensor with tf is erroring out possibly since the iou_threshold
            # is defined in a loop. So as a workaround if the iou_op is const, then it's value will be part of the
            # node definition for the tensor (which is the case for the keras implementation)
            return iou_op.node_def.attr["value"].tensor.float_val[0]

        raise ConverterError(get_error_message('ERROR_TF_SSD_CAFFE_CAN_NOT_RESOLVE_IOU_THRESHOLD'))


class CaffeSsdLayerBuilder(LayerBuilder):

    def build_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type ir_graph: converters.common.converter_ir.op_graph.IROpGraph
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: ConcatLayerResolver.Descriptor
        :rtype: int
        """

        # Since we removed all ops related to priorbox calculation our inputs will include the conv layers
        # used for priorbox so need to filter that
        input_descriptors = [d for d in input_descriptors if not isinstance(d, ConvolutionLayerResolver.Descriptor)]

        # we need to revert input (classes, boxes) -> (boxes, classes) (the former is implemented in reference keras,
        # while the latter is in snpe)
        ssd_boxes_input = ssd_classes_input = ""
        axis = 1
        for i, name in enumerate(self.get_input_names(converter_context, descriptor, input_descriptors)):
            # Snpe expects both the inputs flattened.
            input_tensor = converter_context.get_output_tensors_between(input_descriptors[i], descriptor)[0].op
            input_shape = converter_context.graph_helper.get_op_output_shape(input_tensor)
            output_dims = [reduce(int.__mul__, input_shape[axis:])]
            output_shape = input_shape[:axis] + output_dims
            output_name = name + '_flatten'
            ir_graph.add(ReshapeOp(name + '_flatten',
                                   output_shape),
                         input_names=[name],
                         output_names=[output_name])
            if "boxes" in name:
                ssd_boxes_input = output_name
            elif "classes" in name:
                ssd_classes_input = output_name
            else:
                raise ConverterError(get_error_message('ERROR_TF_CAFFE_SSD_UNKNOWN_INPUT')(name))

        ssd_detection_input_names = [ssd_boxes_input, ssd_classes_input]
        if ssd_boxes_input == "" or ssd_classes_input == "":
            raise ConverterError(get_error_message('ERROR_TF_CAFFE_SSD_REQUIRES_2_INPUTS'))

        # below hard coded defaults align with the keras implementation used in reference. Not able to grab the
        # values from model since they were not represented by a tf Op
        return ir_graph.add(DetectionOutputOp(name=descriptor.layer_name,
                                              output_dims=descriptor.output_dims,
                                              num_classes=descriptor.num_classes,
                                              share_location=True,
                                              background_label_id=0,
                                              nms_threshold=descriptor.iou_threshold,
                                              confidence_threshold=descriptor.score_threshold,
                                              nms_top_k=descriptor.nms_top_k,
                                              nms_eta=1.0,
                                              code_type=DetectionOutputOp.PriorBoxType.CENTER_SIZE,
                                              keep_top_k=descriptor.keep_top_k,
                                              variance_encoded_in_target=False,
                                              priorbox_data=descriptor.priorboxes_data),
                            input_names=ssd_detection_input_names,
                            output_names=descriptor.output_names)
