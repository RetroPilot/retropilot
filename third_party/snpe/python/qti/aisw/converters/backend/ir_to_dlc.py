# =============================================================================
#
#  Copyright (c) 2018-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import os
import sys
import numpy as np

try:
    from qti.aisw.dlc_utils import modeltools
except ImportError as ie1:
    print("Failed to find necessary python package")
    print(str(ie1))
    print("Please ensure that libDlModelToolsPy3.so is discoverable your PYTHONPATH")
    sys.exit(1)

from qti.aisw.converters.common.converter_ir import translation, op_adapter, op_graph
from qti.aisw.converters.common.converter_ir.axis_tracker import AxisTracker, AxisOrders
from qti.aisw.converters.common.backend_base import ConverterBackend, BackendTranslationBase
from qti.aisw.converters.common.utils import code_to_message
from qti.aisw.converters.common.utils.converter_utils import *
from qti.aisw.converters.common.utils import validation_utils
from .snpe_translation_utils import (
    validate_snpe_padding,
    adjust_padding_strategy,
    get_pad_size
)

# ------------------------------------------------------------------------------
#   Module Level enum/Functions
# ------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
# IR consts to dlc dictionary. This holds the translation between the string constants in IR graph
# to what is defined in modeltools.
# -------------------------------------------------------------------------------------------------
ir_consts_to_dlc = {
    # conv
    op_adapter.PadOp.Mode.ZERO: modeltools.PADDING_ZERO,
    op_adapter.PadOp.Mode.REFLECT: modeltools.PADDING_REFLECT,
    op_adapter.PadOp.Mode.CONSTANT: modeltools.PADDING_CONSTANT,
    op_adapter.PadOp.Mode.EDGE: modeltools.PADDING_EDGE,
    op_adapter.IRPaddingStrategies.PADDING_SIZE_EXPLICIT: modeltools.PADDING_SIZE_EXPLICIT,
    op_adapter.IRPaddingStrategies.PADDING_SIZE_IMPLICIT_VALID: modeltools.PADDING_SIZE_IMPLICIT_VALID,
    op_adapter.IRPaddingStrategies.PADDING_SIZE_IMPLICIT_SAME_END: modeltools.PADDING_SIZE_IMPLICIT_SAME,
    op_adapter.IRPaddingStrategies.PADDING_SIZE_EXPLICIT_FLOOR: modeltools.PADDING_SIZE_EXPLICIT_FLOOR,
    # get mapped to righthanded in DNN MODEL
    op_adapter.IRPaddingStrategies.PADDING_SIZE_EXPLICIT_RIGHTHANDED: modeltools.PADDING_SIZE_EXPLICIT_ASYMMETRIC,

    op_adapter.NeuronOp.Type.RELU: modeltools.NEURON_RELU,
    op_adapter.NeuronOp.Type.RELU_MIN_MAX: modeltools.NEURON_RELU_MIN_MAX,
    op_adapter.NeuronOp.Type.TANH: modeltools.NEURON_TANH,
    op_adapter.NeuronOp.Type.LOGISTIC: modeltools.NEURON_LOGISTIC,
    op_adapter.NeuronOp.Type.ELU: modeltools.NEURON_ELU,
    op_adapter.NeuronOp.Type.HSWISH: modeltools.NEURON_HSWISH,
    op_adapter.NeuronOp.Type.NONE: modeltools.NEURON_NONE,

    # pooling
    op_adapter.PoolOp.Type.MAX: modeltools.POOL_MAX,
    op_adapter.PoolOp.Type.AVG: modeltools.POOL_AVG,

    # scaling
    op_adapter.ResizeOp.Mode.BILINEAR: modeltools.RESIZE_BILINEAR,
    op_adapter.ResizeOp.Mode.NEAREST_NEIGHBOR: modeltools.RESIZE_NEAREST_NEIGHBOR,

    # ssd
    op_adapter.DetectionOutputOp.PriorBoxType.CORNER: modeltools.PRIORBOX_TYPE_CORNER,
    op_adapter.DetectionOutputOp.PriorBoxType.CENTER_SIZE: modeltools.PRIORBOX_TYPE_CENTER_SIZE,
    op_adapter.DetectionOutputOp.PriorBoxType.CORNER_SIZE: modeltools.PRIORBOX_TYPE_CORNER_SIZE,

    # embedding
    op_adapter.EmbeddingOp.PartitionStrategy.MOD: modeltools.EMBEDDING_PARTITION_STRATEGY_MOD,
    op_adapter.EmbeddingOp.PartitionStrategy.DIV: modeltools.EMBEDDING_PARTITION_STRATEGY_DIV,

    # channel shuffle
    op_adapter.ChannelShuffleOp.GROUPED: modeltools.CHANNEL_SHUFFLE_GROUPED,

    # layer affinity
    "LAYER_AFFINITY_CPU_FLOAT32": modeltools.LAYER_AFFINITY_CPU_FLOAT32,
    "LAYER_AFFINITY_GPU_FLOAT32_16_HYBRID": modeltools.LAYER_AFFINITY_GPU_FLOAT32_16_HYBRID,
    "LAYER_AFFINITY_DSP_FIXED8_TF": modeltools.LAYER_AFFINITY_DSP_FIXED8_TF,
    "LAYER_AFFINITY_GPU_FLOAT16": modeltools.LAYER_AFFINITY_GPU_FLOAT16
}


DlcTranslations = translation.TranslationBank()


class DLCBackend(ConverterBackend):
    class ArgParser(ConverterBackend.ArgParser):
        def __init__(self, **kwargs):
            super(DLCBackend.ArgParser, self).__init__(**kwargs)
            self.add_optional_argument('--model_version', type=str, default=None,
                                       help='User-defined ASCII string to identify the model, only first '
                                            '64 bytes will be stored')
            self.add_optional_argument('--validation_target', nargs=2,
                                       action=validation_utils.ValidateTargetArgs,
                                       help="A combination of processor and runtime target against which model "
                                            "will be validated. \n"
                                            "Choices for RUNTIME_TARGET: \n   {cpu, gpu, dsp}. \n"
                                            "Choices for PROCESSOR_TARGET: \n"
                                            "   {snapdragon_801, snapdragon_820, snapdragon_835}.\n"
                                            "If not specified, will validate model against "
                                            "{snapdragon_820, snapdragon_835} across all runtime targets.",
                                       metavar=('RUNTIME_TARGET', 'PROCESSOR_TARGET'),
                                       default=[], )
            self.add_optional_argument('--strict', dest="enable_strict_validation",
                                       action="store_true",
                                       default=False,
                                       help="If specified, will validate in strict mode whereby model will not "
                                            "be produced if it violates constraints of the specified validation "
                                            "target. If not specified, will validate model in permissive mode "
                                            "against the specified validation target.")
            self.add_optional_argument("--udo_config_paths", "-udo", nargs='+',
                                       dest="custom_op_config_paths",
                                       action=validation_utils.check_json(),
                                       help="Path to the UDO configs (space separated, if multiple)")

    def __init__(self, args):
        super(DLCBackend, self).__init__(args)
        self.model_version = args.model_version
        self.validation_target = args.validation_target
        self.enable_strict_validation = args.enable_strict_validation
        self.model = modeltools.Model()

    def save(self, graph):
        # get converter args for saving dlc
        if self.output_model_path is None:
            filename, _ = os.path.splitext(os.path.realpath(self.input_model_path))
            output_path = filename + ".dlc"
        else:
            output_path = self.output_model_path

        # add validation target
        if len(self.validation_target) == 0:
            log_debug3("no validation target specified. Using defaults.")
            self.model.add_validation_targets(self.model.get_validation_targets())
        else:
            log_debug3("validation target :" + str(tuple(self.validation_target)))
            self.model.add_validation_targets(tuple(self.validation_target))

        # set validation mode
        if self.enable_strict_validation:
            log_debug3("strict validation is enabled.")
            self.model.set_strict_validation(True)

        log_info(code_to_message.get_progress_message("INFO_DLC_SAVE_LOCATION")(output_path))
        DlcTranslations.apply_method_to_all_ops(BackendTranslationBase.ADD_OP_TO_BACKEND, graph, self)

        for buf in graph.list_buffers():
            self.model.set_buffer_axis_order(buf.name, buf.get_axis_annotations())
        if graph.quantization_params:
            self.model.add_quantization_params(graph.quantization_params)
        self.model.set_converter_command(self.converter_command)
        self.model.set_model_copyright(self.copyright_str)
        if self.model_version:
            self.model.set_model_version(self.model_version[:64])
        self.model.save(output_path)
        log_info(code_to_message.get_progress_message("INFO_CONVERSION_SUCCESS"))


# ------------------------------------------------------------------------------
#   Translations
# ------------------------------------------------------------------------------
def register(dlc_translation):
    # Allows more than one target to be specified per class
    if isinstance(dlc_translation.TARGET, tuple) or isinstance(dlc_translation.TARGET, list):
        DlcTranslations.register_translation(dlc_translation(), *dlc_translation.TARGET)
    else:
        DlcTranslations.register_translation(dlc_translation(), dlc_translation.TARGET)
    return dlc_translation


@register
class DlcInputTranslation(BackendTranslationBase):
    TARGET = op_adapter.InputOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        # update the node name, as it must match the output name. Output
        # name may have changed after ir_optimizations (e.x squashing)
        node.op.name = node.output_names[0]
        backend.model.add_data_layer(node.op.name,
                                     node.op.shape,
                                     node.op.input_encoding_in,
                                     node.op.input_encoding_out,
                                     node.op.input_type)


@register
class DlcArgMaxTranslation(BackendTranslationBase):
    TARGET = op_adapter.ArgMaxOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        backend.model.add_argmax_layer(node.op.name,
                                       node.input_names[0],
                                       node.output_names[0],
                                       node.op.axis,
                                       node.op.keep_dims)


@register
class DlcArgMinTranslation(BackendTranslationBase):
    TARGET = op_adapter.ArgMinOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        backend.model.add_argmin_layer(node.op.name,
                                       node.input_names[0],
                                       node.output_names[0],
                                       node.op.axis,
                                       node.op.keep_dims)


@register
class DlcBatchnormTranslation(BackendTranslationBase):
    TARGET = op_adapter.BatchnormOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        backend.model.add_batchnorm_layer(node.op.name,
                                          node.op.weights,
                                          node.op.bias,
                                          node.op.compute_statistics,
                                          node.op.use_mu_sigma,
                                          node.op.across_spatial,
                                          node.input_names[0],
                                          node.output_names[0],
                                          node.op.epsilon,
                                          node.op.normalize_variance)


@register
class DlcChannelShuffleTranslation(BackendTranslationBase):
    TARGET = op_adapter.ChannelShuffleOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        backend.model.add_channel_shuffle_layer(node.op.name,
                                                node.op.groups,
                                                ir_consts_to_dlc[node.op.shuffle_mode],
                                                node.input_names[0],
                                                node.output_names[0])


@register
class DlcConvolutionTranslation(BackendTranslationBase):
    TARGET = [op_adapter.ConvolutionOp.TRANSLATION_KEY, op_adapter.DepthwiseConvolutionOp.TRANSLATION_KEY]

    def calc_same_padding(self, node, input_w, input_h, kernel_w, kernel_h, same_begin=False):
        pad_x_begin, pad_x_end = op_adapter.ConvolutionOp.calc_same_padding_size(input_w,
                                                                                 kernel_w,
                                                                                 node.op.dilationx,
                                                                                 node.op.stridex,
                                                                                 same_begin=same_begin)
        pad_y_begin, pad_y_end = op_adapter.ConvolutionOp.calc_same_padding_size(input_h,
                                                                                 kernel_h,
                                                                                 node.op.dilationy,
                                                                                 node.op.stridey,
                                                                                 same_begin=same_begin)
        return [pad_x_begin, pad_y_begin, pad_x_end, pad_y_end]

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        input_shape = graph.get_input_shapes(node)[0]
        # Assuming input is in NHWC format
        input_h, input_w = input_shape[1:3]

        if not hasattr(node.op, 'weights') and not hasattr(node.op, 'bias'):
            kernel_h, kernel_w = graph.get_buffer(node.input_names[1]).shape[:2]
        else:
            kernel_h, kernel_w = node.op.weights.shape[:2]
        node.op.padding_size_strategy = adjust_padding_strategy(self, node, input_w, input_h, kernel_w, kernel_h)

        validate_snpe_padding(node)

        pad_x, pad_y = get_pad_size(node.op.padding_size_strategy,
                                            node.op.padx_before,
                                            node.op.padx_after,
                                            node.op.pady_before,
                                            node.op.pady_after)

        # Represents static convolution layer
        if hasattr(node.op, 'weights') and hasattr(node.op, 'bias'):
            backend.model.add_conv_layer(node.op.name,
                                         np.ascontiguousarray(node.op.weights),
                                         node.op.bias,
                                         pad_x,
                                         pad_y,
                                         ir_consts_to_dlc[node.op.padding_mode],
                                         int(ir_consts_to_dlc[node.op.padding_size_strategy]),
                                         node.op.stridex,
                                         node.op.stridey,
                                         node.op.dilationx,
                                         node.op.dilationy,
                                         node.input_names[0],
                                         node.output_names[0],
                                         node.op.groups)
        else:
            # Represents dynamic convolution layer
            backend.model.add_dynamic_conv_layer(node.op.name,
                                                 pad_x,
                                                 pad_y,
                                                 ir_consts_to_dlc[node.op.padding_mode],
                                                 int(ir_consts_to_dlc[node.op.padding_size_strategy]),
                                                 node.op.stridex,
                                                 node.op.stridey,
                                                 node.op.dilationx,
                                                 node.op.dilationy,
                                                 node.input_names,
                                                 node.output_names[0],
                                                 node.op.groups)


@register
class DlcConcatTranslation(BackendTranslationBase):
    TARGET = op_adapter.ConcatOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):

        if node.op.axis > 4:
            raise ValueError(code_to_message.get_error_message('ERROR_SNPE_TILE_AXIS_NOT_SUPPORTED')
                             (str(node.op.name), node.op.axis))

        backend.model.add_concatenation_layer(node.op.name,
                                              node.input_names,
                                              node.output_names[0],
                                              node.op.axis)


@register
class DlcConstantTranslation(BackendTranslationBase):
    TARGET = op_adapter.ConstantOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        node.op.name = node.output_names[0]
        shape = list(node.op.tensor.shape)
        if not shape:
            shape = [1]
        backend.model.add_const_layer(node.op.name,
                                      shape,
                                      node.op.tensor,
                                      node.op.quantizable)


@register
class DlcConvertTranslation(BackendTranslationBase):
    TARGET = op_adapter.ConvertOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        quant_params = graph.get_layer_quantization_param(node.op.name)
        output_encoding = quant_params['output_encodings'][0]
        backend.model.add_convert_layer(node.op.name,
                                        node.input_names[0],
                                        node.output_names[0],
                                        output_encoding)


@register
class DlcCropTranslation(BackendTranslationBase):
    TARGET = op_adapter.CropOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        backend.model.add_crop_layer(node.op.name,
                                     node.op.offsets,
                                     node.op.counts,
                                     node.op.output_shape,
                                     node.input_names[0],
                                     node.output_names[0])


@register
class DlcCropAndResizeTranslation(BackendTranslationBase):
    TARGET = op_adapter.CropAndResizeOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        backend.model.add_crop_and_resize_layer(node.op.name,
                                                input_names=node.input_names,
                                                output_name=node.output_names[0],
                                                crop_height=node.op.crop_height,
                                                crop_width=node.op.crop_width,
                                                interpolation_method=node.op.interpolation_method,
                                                extrapolation_value=node.op.extrapolation_value)


@register
class DlcCrossCorrelationTranslation(BackendTranslationBase):
    TARGET = op_adapter.CrossCorrelationOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        log_assert(len(node.input_names) == 2, "Layer %s: expected exactly two input blobs" % node.op.name)
        backend.model.add_cross_correlation_layer(node.op.name,
                                                  node.input_names[0],
                                                  node.input_names[1],
                                                  node.output_names[0])


@register
class DlcDeconvolutionTranslation(BackendTranslationBase):
    TARGET = op_adapter.DeconvolutionOp.TRANSLATION_KEY

    def calc_same_padding(self, node, input_w, input_h, kernel_h, kernel_w, same_begin=False):
        pad_x_begin, pad_x_end = op_adapter.DeconvolutionOp.calc_same_padding_size(input_w,
                                                                                   kernel_w,
                                                                                   node.op.stridex,
                                                                                   node.op.output_paddingx,
                                                                                   same_begin=same_begin)
        pad_y_begin, pad_y_end = op_adapter.DeconvolutionOp.calc_same_padding_size(input_h,
                                                                                   kernel_h,
                                                                                   node.op.stridey,
                                                                                   node.op.output_paddingy,
                                                                                   same_begin=same_begin)
        return [pad_x_begin, pad_y_begin, pad_x_end, pad_y_end]

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        if not hasattr(node.op, 'weights') or not hasattr(node.op, 'bias'):
            raise ValueError("Dynamic deconvolution is unsupported by this converter.")

        if node.op.stridex != node.op.stridey:
            raise ValueError(code_to_message.get_error_message("ERROR_DECONV_RECTANGULAR_STRIDE_UNSUPPORTED"))

        input_shape = graph.get_input_shapes(node)[0]
        # Assuming input is in NHWC format
        input_h, input_w = input_shape[1:3]

        kernel_h, kernel_w = node.op.weights.shape[:2]
        node.op.padding_size_strategy = adjust_padding_strategy(self, node, input_w, input_h, kernel_w, kernel_h)
        validate_snpe_padding(node)

        pad_x, pad_y = get_pad_size(node.op.padding_size_strategy,
                                            node.op.padx_before,
                                            node.op.padx_after,
                                            node.op.pady_before,
                                            node.op.pady_after)

        log_assert(node.op.stridex == node.op.stridey,
                   code_to_message.get_error_message("ERROR_DECONV_RECTANGULAR_STRIDE_UNSUPPORTED"))
        log_assert(node.op.padx_before == node.op.pady_before,
                   code_to_message.get_error_message('ERROR_SNPE_DECONV_NO_SUPPORT_RECT_PADDING'))

        backend.model.add_deconvolution_layer(node.op.name,
                                              np.ascontiguousarray(node.op.weights),
                                              node.op.bias,
                                              node.op.stridex,
                                              int(ir_consts_to_dlc[node.op.padding_size_strategy]),
                                              pad_x,
                                              pad_y,
                                              node.input_names[0],
                                              node.output_names[0],
                                              node.op.output_width,
                                              node.op.output_height,
                                              node.op.groups,
                                              node.op.output_paddingx,
                                              node.op.output_paddingy)


@register
class DlcDetectionOutputTranslation(BackendTranslationBase):
    TARGET = [op_adapter.DetectionOutputOp.TRANSLATION_KEY, op_adapter.NonMaxSuppresionOp.TRANSLATION_KEY]

    @staticmethod
    def get_detection_outputs(node, graph):
        """
        :return: output_names, output_dims
        """

        if node.op.type == op_adapter.DetectionOutputOp.TRANSLATION_KEY:
            # SNPE expects 1 output when Caffe style SSD is used to retain backward compatibility
            #      IR: scores[Batch, max_num_det], boxes[Batch, max_num_det, 4], classes[Batch, max_num_det],
            #          num_det[Batch],
            #      Caffe Style: 1 output of shape [Batch, 1, max_num_det, 7]

            # Add the single output with its IR buffer
            output_names = [node.op.name]
            input_dim = graph.get_buffer(node.input_names[0]).get_buf_dims()
            # 7: [image_batch, label, confidence, x_min, y_min, x_max, y_max]
            # 0 dimension indicates dynamic resizing of # of outputs
            output_dims = [[input_dim[0], 1, 0, 7]]
            detection_out_buf = op_graph.Buffer(output_names[0], output_dims[0], node,
                                                axis_format=AxisTracker.AxisFormat.NONTRIVIAL)
            graph.buffers[output_names[0]] = detection_out_buf

            # Cleanup unused buffers and update to new one
            for output_name in node.output_names:
                graph.delete_buffer(output_name)
            node.output_names = output_names

        else:
            # SNPE spec expects [scores, boxes, classes, valid_det] for outputs.
            # IR spec outputs [boxes, scores, classes, valid_det], so the first two output names should
            # be switched
            output_names = node.output_names
            boxes_name = output_names[0]
            output_names[0] = output_names[1]  # scores as out [0]
            output_names[1] = boxes_name  # boxes as out [1]
            output_dims = []
            for i, output_name in enumerate(output_names):
                output_buf = graph.get_buffer(output_name)
                output_dims.append(output_buf.shape)
        return output_names, output_dims

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        # TODO: remove handling of both nms and ssd here once new IR node
        if node.op.type == op_adapter.NonMaxSuppresionOp.TRANSLATION_KEY and \
                "scale_y" not in node.op.attrs:
            # Generic TF NMS case
            # delta scales indicate SSD case, where box decode is merged into NMS layer

            # remove num_valid_detections output which is not supported in SNPE
            # SNPE output: [boxes, scores, classes, output_features*]
            # IR output: [boxes, scores, classes, num_valid_detections, output_features*]
            num_valid_detections = node.output_names[3]
            num_det_buf = graph.get_buffer(num_valid_detections)
            if len(num_det_buf.consumers) != 0:
                raise RuntimeError("Unable to remove num_valid_detections output of NMS Op which is not supported "
                                   "as output by SNPE. Consumers of buffer found: {} for node {}."
                                   .format([node_.op.name for node_ in num_det_buf.consumers], node.op.name))
            graph.delete_buffer(num_valid_detections)
            node.output_names.remove(num_valid_detections)

            backend.model.add_multi_class_nms_layer(name=node.op.name,
                                                    input_names=node.input_names,
                                                    output_names=node.output_names,
                                                    scoreThreshold=node.op.score_threshold,
                                                    iouThreshold=node.op.iou_threshold,
                                                    maxDetectionPerClass=node.op.max_detections_per_class,
                                                    maxTotalDetections=node.op.max_total_detections)
            return

        # Handle SSD usecase
        # TODO: remove handling of both nms and ssd here once new IR node
        # assign params based on whether DetectionOutputOp or (decodeBox+nms)Op was used
        delta_scales = [node.op.scale_y, node.op.scale_x, node.op.scale_h, node.op.scale_w]
        share_location = True
        if node.op.type == op_adapter.DetectionOutputOp.TRANSLATION_KEY:
            score_threshold = node.op.confidence_threshold
            iou_threshold = node.op.nms_threshold
            detection_limit = node.op.nms_top_k
            keep_top_k = node.op.keep_top_k
            use_bg_in_nms = False
            if node.op.share_location is False:
                share_location = False
            log_assert(node.op.variance_encoded_in_target is False,
                       "DetectionOut only supports variance encoded in boxes. Op {} has variance encoded in target."
                       .format(node.op.name))
        else:
            score_threshold = node.op.score_threshold
            iou_threshold = node.op.iou_threshold
            detection_limit = node.op.max_detections_per_class
            keep_top_k = node.op.max_total_detections
            use_bg_in_nms = True

        bg_id = getattr(node.op, "background_label_id", 0)
        nms_eta = getattr(node.op, "nms_eta", 1.0)
        input_names = node.input_names
        output_names, output_dims = self.get_detection_outputs(node, graph)

        backend.model.add_detection_output_layer(node.op.name,
                                                 input_names,
                                                 output_names,
                                                 output_dims,
                                                 delta_scales,
                                                 score_threshold,
                                                 iou_threshold,
                                                 modeltools.NMS_TYPE_REGULAR,
                                                 bg_id,
                                                 use_bg_in_nms,
                                                 True,  # output_background
                                                 share_location,
                                                 nms_eta,
                                                 detection_limit,
                                                 keep_top_k)


@register
class DlcDropoutTranslation(BackendTranslationBase):
    TARGET = op_adapter.DropoutOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        backend.model.add_dropout_layer(node.op.name,
                                        node.op.keep,
                                        node.input_names[0],
                                        node.output_names[0])


@register
class DlcElementwiseAndTranslation(BackendTranslationBase):
    TARGET = op_adapter.ElementwiseAndOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        backend.model.add_elementwise_binary_logic_and_layer(node.op.name,
                                                             node.input_names,
                                                             node.output_names[0])


@register
class DlcElementwiseDivTranslation(BackendTranslationBase):
    TARGET = op_adapter.ElementwiseDivOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        backend.model.add_elementwise_binary_div_layer(node.op.name,
                                                       node.input_names,
                                                       node.output_names[0])


@register
class DlcElementwiseEqualTranslation(BackendTranslationBase):
    TARGET = op_adapter.ElementwiseEqualOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        backend.model.add_elementwise_binary_equal_layer(node.op.name,
                                                         node.input_names,
                                                         node.output_names[0])


@register
class DlcElementwiseGreaterTranslation(BackendTranslationBase):
    TARGET = op_adapter.ElementwiseGreaterOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        backend.model.add_elementwise_binary_greater_layer(node.op.name,
                                                           node.input_names,
                                                           node.output_names[0])


@register
class DlcElementwiseGreaterEqualTranslation(BackendTranslationBase):
    TARGET = op_adapter.ElementwiseGreaterEqualOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        backend.model.add_elementwise_binary_greater_equal_layer(node.op.name,
                                                                 node.input_names,
                                                                 node.output_names[0])


@register
class DlcElementwiseLessTranslation(BackendTranslationBase):
    TARGET = op_adapter.ElementwiseLessOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        backend.model.add_elementwise_binary_less_layer(node.op.name,
                                                        node.input_names,
                                                        node.output_names[0])


@register
class DlcElementwiseLessEqualTranslation(BackendTranslationBase):
    TARGET = op_adapter.ElementwiseLessEqualOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        backend.model.add_elementwise_binary_less_equal_layer(node.op.name,
                                                              node.input_names,
                                                              node.output_names[0])


@register
class DlcElementwiseMaxTranslation(BackendTranslationBase):
    TARGET = op_adapter.ElementwiseMaxOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        backend.model.add_elementwise_binary_max_layer(node.op.name,
                                                       node.input_names,
                                                       node.output_names[0])


@register
class DlcElementwiseMinTranslation(BackendTranslationBase):
    TARGET = op_adapter.ElementwiseMinOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        backend.model.add_elementwise_binary_min_layer(node.op.name,
                                                       node.input_names,
                                                       node.output_names[0])


@register
class DlcElementwiseNotEqualTranslation(BackendTranslationBase):
    TARGET = op_adapter.ElementwiseNotEqualOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        backend.model.add_elementwise_binary_not_equal_layer(node.op.name,
                                                             node.input_names,
                                                             node.output_names[0])


@register
class DlcElementwiseOrTranslation(BackendTranslationBase):
    TARGET = op_adapter.ElementwiseOrOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        backend.model.add_elementwise_binary_logic_or_layer(node.op.name,
                                                            node.input_names,
                                                            node.output_names[0])


@register
class DlcElementwiseProductTranslation(BackendTranslationBase):
    TARGET = op_adapter.ElementwiseProductOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        backend.model.add_elementwise_binary_product_layer(node.op.name,
                                                           node.input_names,
                                                           node.output_names[0])


@register
class DlcElementwiseSelectTranslation(BackendTranslationBase):
    TARGET = op_adapter.ElementwiseSelectOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        backend.model.add_elementwise_select_layer(node.op.name,
                                                   node.input_names,
                                                   node.output_names[0])


@register
class DlcElementwiseSubTranslation(BackendTranslationBase):
    TARGET = op_adapter.ElementwiseSubOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        backend.model.add_elementwise_binary_sub_layer(node.op.name,
                                                       node.input_names,
                                                       node.output_names[0])


@register
class DlcElementwiseSumTranslation(BackendTranslationBase):
    TARGET = op_adapter.ElementwiseSumOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        backend.model.add_elementwise_binary_sum_layer(node.op.name,
                                                       node.input_names,
                                                       node.output_names[0])


@register
class DlcElementwiseUnaryAbsTranslation(BackendTranslationBase):
    TARGET = op_adapter.ElementwiseUnaryAbsOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        backend.model.add_elementwise_unary_abs_layer(node.op.name,
                                                      node.input_names[0],
                                                      node.output_names[0])


@register
class DlcElementwiseUnaryCeilTranslation(BackendTranslationBase):
    TARGET = op_adapter.ElementwiseUnaryCeilOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        backend.model.add_elementwise_unary_ceil_layer(node.op.name,
                                                       node.input_names[0],
                                                       node.output_names[0])


@register
class DlcElementwiseUnaryExpTranslation(BackendTranslationBase):
    TARGET = op_adapter.ElementwiseUnaryExpOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        backend.model.add_elementwise_unary_exp_layer(node.op.name,
                                                      node.input_names[0],
                                                      node.output_names[0])


@register
class DlcElementwiseUnaryFloorTranslation(BackendTranslationBase):
    TARGET = op_adapter.ElementwiseUnaryFloorOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        backend.model.add_elementwise_unary_floor_layer(node.op.name,
                                                        node.input_names[0],
                                                        node.output_names[0])


@register
class DlcElementwiseUnaryLogTranslation(BackendTranslationBase):
    TARGET = op_adapter.ElementwiseUnaryLogOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        backend.model.add_elementwise_unary_log_layer(node.op.name,
                                                      node.input_names[0],
                                                      node.output_names[0])


@register
class DlcElementwiseUnaryNegTranslation(BackendTranslationBase):
    TARGET = op_adapter.ElementwiseUnaryNegOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        backend.model.add_elementwise_unary_neg_layer(node.op.name,
                                                      node.input_names[0],
                                                      node.output_names[0])


@register
class DlcElementwiseUnaryNotTranslation(BackendTranslationBase):
    TARGET = op_adapter.ElementwiseUnaryNotOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        backend.model.add_elementwise_unary_lnot_layer(node.op.name,
                                                       node.input_names[0],
                                                       node.output_names[0])


@register
class DlcElementwiseUnaryRoundTranslation(BackendTranslationBase):
    TARGET = op_adapter.ElementwiseUnaryRoundOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        backend.model.add_elementwise_unary_round_layer(node.op.name,
                                                        node.input_names[0],
                                                        node.output_names[0])


@register
class DlcElementwiseUnaryRsqrtTranslation(BackendTranslationBase):
    TARGET = op_adapter.ElementwiseUnaryRsqrtOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        backend.model.add_elementwise_unary_rsqrt_layer(node.op.name,
                                                        node.input_names[0],
                                                        node.output_names[0])


@register
class DlcElementwiseUnarySinTranslation(BackendTranslationBase):
    TARGET = op_adapter.ElementwiseUnarySinOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        backend.model.add_elementwise_unary_sin_layer(node.op.name,
                                                      node.input_names[0],
                                                      node.output_names[0])


@register
class DlcElementwiseUnarySqrtTranslation(BackendTranslationBase):
    TARGET = op_adapter.ElementwiseUnarySqrtOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        backend.model.add_elementwise_unary_sqrt_layer(node.op.name,
                                                       node.input_names[0],
                                                       node.output_names[0])


@register
class DlcEmbeddingTranslation(BackendTranslationBase):
    TARGET = op_adapter.EmbeddingOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        backend.model.add_embedding_layer(name=node.op.name,
                                          output_dim=node.op.output_dim,
                                          input_names=node.input_names,
                                          output_name=node.output_names[0],
                                          partition_strategy=ir_consts_to_dlc[node.op.embedding_strategy])


@register
class DlcExtractGlimpseTranslation(BackendTranslationBase):
    TARGET = op_adapter.ExtractGlimpseOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        backend.model.add_extract_glimpse_layer(node.op.name,
                                                input_names=node.input_names,
                                                output_name=node.output_names[0],
                                                glimpse_width=node.op.glimpse_width,
                                                glimpse_height=node.op.glimpse_height,
                                                centered=node.op.centered,
                                                normalized=node.op.normalized,
                                                uniform_noise=(node.op.noise == 'NOISE_UNIFORM'))


@register
class DlcFullyConnectedTranslation(BackendTranslationBase):
    TARGET = op_adapter.FullyConnectedOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        backend.model.add_fc_layer(node.op.name,
                                   [node.op.weights],
                                   node.op.bias,
                                   node.input_names,
                                   node.output_names[0])


@register
class DlcGatherTranslation(BackendTranslationBase):
    TARGET = op_adapter.GatherOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        backend.model.add_gather_layer(node.op.name,
                                       node.input_names[0],
                                       node.input_names[1],
                                       node.output_names[0],
                                       node.op.axis)


@register
class DlcGenerateProposalsOp(BackendTranslationBase):
    TARGET = op_adapter.GenerateProposalsOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        backend.model.add_generate_proposals_layer(node.op.name,
                                                   node.op.spatial_scale,
                                                   node.op.pre_nms_top_n,
                                                   node.op.post_nms_top_n,
                                                   node.op.nms_thresh,
                                                   node.op.min_size,
                                                   node.op.correct_transform_coords,
                                                   node.op.anchors,
                                                   node.op.im_info,
                                                   node.input_names[0],
                                                   node.input_names[1],
                                                   node.output_names[0],
                                                   node.ouput_names[1])


@register
class DlcGruTranslation(BackendTranslationBase):
    TARGET = op_adapter.GruOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        backend.model.add_gru_layer(node.op.name,
                                    node.op.state_gate,
                                    node.op.forget_gate,
                                    node.op.control_gate,
                                    ir_consts_to_dlc[node.op.activation],
                                    ir_consts_to_dlc[node.op.gate_activation],
                                    ir_consts_to_dlc[node.op.rec_gate_activation],
                                    node.op.backwards,
                                    node.input_names[0],
                                    node.output_names[0])


@register
class DlcImageProjectiveTransformTranslation(BackendTranslationBase):
    TARGET = op_adapter.ImageProjectiveTransformOp.TRANSLATION_KEY
    supported_modes = {'NEAREST': ir_consts_to_dlc[op_adapter.ResizeOp.Mode.NEAREST_NEIGHBOR],
                       'BILINEAR': ir_consts_to_dlc[op_adapter.ResizeOp.Mode.BILINEAR]}

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        if node.op.interpolation_mode not in self.supported_modes:
            raise KeyError("Unsupported interpolation mode {} provided. Supported modes {}".format(
                node.op.interpolation_mode, self.supported_modes
            ))

        backend.model.add_image_projective_transform_layer(name=node.op.name,
                                                           input_names=node.input_names,
                                                           output_name=node.output_names[0],
                                                           interpolation=self.supported_modes[
                                                               node.op.interpolation_mode])


@register
class DlcLayernormTranslation(BackendTranslationBase):
    TARGET = op_adapter.LayerNormOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        weights_buffer = graph.get_buffer(node.input_names[1])
        weights_node = weights_buffer.producer
        if weights_node.op.type == op_adapter.ConstantOp.TRANSLATION_KEY:
            node.op.weights = weights_node.op.tensor
        else:
            raise RuntimeError("Only Constant weights supported in SNPE currently")

        if len(node.input_names) > 2:
            bias_buffer = graph.get_buffer(node.input_names[2])
            bias_node = bias_buffer.producer
            if bias_node.op.type == op_adapter.ConstantOp.TRANSLATION_KEY:
                node.op.bias = bias_node.op.tensor
            else:
                raise RuntimeError("Only Constant bias supported in SNPE currently")
        else:
            node.op.bias = np.zeros(weights_buffer.shape[-1])

        backend.model.add_layernorm_layer(node.op.name,
                                          weights=node.op.weights,
                                          bias=node.op.bias,
                                          input_name=node.input_names[0],
                                          output_name=node.output_names[0],
                                          axes=node.op.axes,
                                          epsilon=node.op.epsilon)


@register
class DlcL2NormTranslation(BackendTranslationBase):
    TARGET = op_adapter.L2NormOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        axis = node.op.axis
        if isinstance(node.op.axis, np.ndarray):
            if len(node.op.axis) != 1:
                raise ValueError("Only support scalar axis for l2Norm got array results {} for op {}"
                                 .format(node.op.axis, node.op.name))
            axis = node.op.axis[0]
        backend.model.add_l2_norm_layer(node.op.name,
                                        node.input_names[0],
                                        node.output_names[0],
                                        int(axis),
                                        float(node.op.epsilon))


@register
class DlcLstmTranslation(BackendTranslationBase):
    TARGET = op_adapter.LstmOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        backend.model.add_lstm_layer(node.op.name,
                                     node.op.input_weights,
                                     node.op.gate_bias,
                                     node.op.hidden_state_weights,
                                     node.op.w_xc_static,
                                     node.op.backward,
                                     node.op.reset_state_at_time_step_0,
                                     node.input_names[0],
                                     node.op.sequence_continuation_name,
                                     node.op.x_static_name,
                                     node.op.c_0_input_name,
                                     node.op.h_0_input_name,
                                     node.output_names,
                                     node.op.cell_weights,
                                     node.op.cell_clip_threshold,
                                     node.op.proj_weights,
                                     node.op.proj_bias,
                                     node.op.output_clip_threshold,
                                     node.op.normalization_weights)



@register
class DlcMatMulTranslation(BackendTranslationBase):
    TARGET = op_adapter.MatMulOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        backend.model.add_matmul_layer(node.op.name,
                                       node.op.bias,
                                       node.op.transpose_a,
                                       node.op.transpose_b,
                                       node.input_names,
                                       node.output_names[0])


@register
class DlcMaxYTranslation(BackendTranslationBase):
    TARGET = op_adapter.MaxYOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        backend.model.add_max_y_layer(node.op.name,
                                      node.input_names[0],
                                      node.output_names[0])


@register
class DlcMomentTranslation(BackendTranslationBase):
    TARGET = op_adapter.MomentOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        backend.model.add_moments_layer(name=node.op.name,
                                        input_name=node.input_names[0],
                                        output_names=node.output_names,
                                        axes=node.op.axes,
                                        keep_dims=node.op.keep_dims)


@register
class DlcNeuronTranslation(BackendTranslationBase):
    TARGET = op_adapter.NeuronOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        backend.model.add_neuron_layer(node.op.name,
                                       ir_consts_to_dlc[node.op.neuron_type],
                                       node.input_names[0],
                                       node.output_names[0],
                                       node.op.a,
                                       node.op.b,
                                       node.op.min_clamp,
                                       node.op.max_clamp)

@register
class DlcOneHotTranslation(BackendTranslationBase):
    TARGET = op_adapter.OneHotOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        backend.model.add_one_hot_layer(node.op.name,
                                        node.op.depth,
                                        node.input_names[0],
                                        node.output_names[0],
                                        node.op.axis,
                                        node.op.on_value,
                                        node.op.off_value)

@register
class DlcPackTranslation(BackendTranslationBase):
    TARGET = op_adapter.PackOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        backend.model.add_pack_layer(name=node.op.name,
                                     input_names=node.input_names,
                                     output_name=node.output_names[0],
                                     axis=node.op.axis)


@register
class DlcPadTranslation(BackendTranslationBase):
    TARGET = op_adapter.PadOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        supported_modes = [op_adapter.PadOp.Mode.CONSTANT,
                           op_adapter.PadOp.Mode.REFLECT,
                           op_adapter.PadOp.Mode.EDGE]
        if node.op.mode not in supported_modes:
            raise ValueError(
                code_to_message.get_error_message("ERROR_PAD_UNSUPPORTED_MODE")(node.op.mode))

        backend.model.add_pad_layer(node.op.name,
                                    node.input_names[0],
                                    node.op.pads.tolist(),
                                    ir_consts_to_dlc[node.op.mode],
                                    node.op.constant_value,
                                    node.output_names[0])


@register
class DlcPermuteTranslation(BackendTranslationBase):
    TARGET = op_adapter.PermuteOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        backend.model.add_permute_layer(node.op.name,
                                        node.op.order,
                                        node.input_names[0],
                                        node.output_names[0])


@register
class DlcPixelShuffleTranslation(BackendTranslationBase):
    TARGET = op_adapter.PixelShuffleOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        backend.model.add_pixel_shuffle_layer(name=node.op.name,
                                              input_name=node.input_names[0],
                                              output_name=node.output_names[0],
                                              upscale_factor=node.op.upscale_factor)


@register
class DlcPoolTranslation(BackendTranslationBase):
    TARGET = op_adapter.PoolOp.TRANSLATION_KEY

    def calc_same_padding(self, node, input_w, input_h, kernel_w, kernel_h, same_begin=False):
        pad_x_begin, pad_x_end = op_adapter.PoolOp.calc_same_padding_size(input_w,
                                                                          kernel_w,
                                                                          node.op.dilation_x,
                                                                          node.op.stride_x,
                                                                          same_begin=same_begin)
        pad_y_begin, pad_y_end = op_adapter.PoolOp.calc_same_padding_size(input_h,
                                                                          kernel_h,
                                                                          node.op.dilation_y,
                                                                          node.op.stride_x,
                                                                          same_begin=same_begin)
        return [pad_x_begin, pad_y_begin, pad_x_end, pad_y_end]

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        input_shape = graph.get_input_shapes(node)[0]
        # Assuming input is in NHWC format
        input_h, input_w = input_shape[1:3]

        kernel_h, kernel_w = node.op.size_y, node.op.size_x
        node.op.padding_size_strategy = adjust_padding_strategy(self, node, input_w, input_h, kernel_w, kernel_h)

        validate_snpe_padding(node)

        pad_x, pad_y = get_pad_size(node.op.padding_size_strategy,
                                            node.op.padx_before,
                                            node.op.padx_after,
                                            node.op.pady_before,
                                            node.op.pady_after)

        backend.model.add_pooling_layer(node.op.name,
                                        ir_consts_to_dlc[node.op.pool_type],
                                        node.op.size_x,
                                        node.op.size_y,
                                        node.op.stride_x,
                                        node.op.stride_y,
                                        node.op.dilation_x,
                                        node.op.dilation_y,
                                        pad_x,
                                        pad_y,
                                        int(ir_consts_to_dlc[node.op.padding_size_strategy]),
                                        node.input_names[0],
                                        node.output_names[0],
                                        node.op.pool_region_include_padding)


@register
class DlcPowerTranslation(BackendTranslationBase):
    TARGET = op_adapter.ElementwisePowerOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        flat_power = np.ravel(node.op.power)
        if not np.all(flat_power == flat_power[0]):
            raise ValueError("Power attribute on {} node {} only supported as scalar, received: {}".
                             format(node.op.type, node.op.name, node.op.power))

        backend.model.add_power_layer(node.op.name,
                                      1.0,  # scale/shift attribute handled in Caffe front-end converter
                                      0.0,
                                      flat_power[0],
                                      node.input_names[0],
                                      node.output_names[0])


@register
class DlcPreluTranslation(BackendTranslationBase):
    TARGET = op_adapter.PreluOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        if node.op.channel_shared:
            raise ValueError(code_to_message.get_error_message('ERROR_PRELU_NON_CHANNEL_SHARED_SUPPORT_ONLY')
                             (str(node.op.name)))

        if len(node.op.coeff.shape) == 1 and node.op.coeff.shape[0] == 1:
            # coeff size should equal input depth
            input_buf = graph.get_buffer(node.input_names[0])
            node.op.coeff = np.ones(input_buf.shape[-1], dtype=np.float32) * node.op.coeff[0]

        if all([i==1 for i in node.op.coeff.shape[:-1]]):
            node.op.coeff = node.op.coeff.reshape(node.op.coeff.shape[-1])
        input_rank = graph.get_buffer(node.input_names[0]).rank()
        while len(node.op.coeff.shape) > input_rank - 1 and node.op.coeff.shape[0]==1:
            node.op.coeff = node.op.coeff.squeeze(0)

        backend.model.add_prelu_layer(node.op.name,
                                      node.op.coeff,
                                      node.input_names[0],
                                      node.output_names[0])


@register
class DlcProposalTranslation(BackendTranslationBase):
    TARGET = op_adapter.ProposalOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        backend.model.add_proposal_layer(node.op.name,
                                         node.op.feat_stride,
                                         node.op.scales,
                                         node.op.ratios,
                                         node.op.anchor_base_size,
                                         node.op.min_bbox_size,
                                         node.op.max_num_proposals,
                                         node.op.max_num_rois,
                                         node.op.iou_threshold_nms,
                                         node.input_names,
                                         node.output_names[0])


@register
class DlcReduceMaxTranslation(BackendTranslationBase):
    TARGET = op_adapter.ReduceMaxOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        backend.model.add_reduction_max_layer(node.op.name,
                                              node.input_names[0],
                                              node.output_names[0],
                                              node.op.axes,
                                              node.op.keep_dims)


@register
class DlcReduceMeanTranslation(BackendTranslationBase):
    TARGET = op_adapter.ReduceMeanOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        backend.model.add_reduction_mean_layer(node.op.name,
                                               node.input_names[0],
                                               node.output_names[0],
                                               node.op.axes,
                                               node.op.keep_dims)


@register
class DlcReduceMinTranslation(BackendTranslationBase):
    TARGET = op_adapter.ReduceMinOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        backend.model.add_reduction_min_layer(node.op.name,
                                              node.input_names[0],
                                              node.output_names[0],
                                              node.op.axes,
                                              node.op.keep_dims)


@register
class DlcReduceProdTranslation(BackendTranslationBase):
    TARGET = op_adapter.ReduceProdOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        backend.model.add_reduction_prod_layer(node.op.name,
                                               node.input_names[0],
                                               node.output_names[0],
                                               node.op.axes,
                                               node.op.keep_dims)


@register
class DlcReduceSumTranslation(BackendTranslationBase):
    TARGET = op_adapter.ReduceSumOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        backend.model.add_reduction_sum_layer(node.op.name,
                                              node.input_names[0],
                                              node.output_names[0],
                                              node.op.axes,
                                              node.op.keep_dims)


@register
class DlcReshapeTranslation(BackendTranslationBase):
    TARGET = op_adapter.ReshapeOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        backend.model.add_reshape_layer(node.op.name,
                                        node.op.output_shape,
                                        node.input_names[0],
                                        node.output_names[0])


@register
class DlcRNormTranslation(BackendTranslationBase):
    TARGET = op_adapter.RNormOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        if node.op.across_channels:
            add_method = backend.model.add_cmrn_layer
        else:
            add_method = backend.model.add_local_norm_layer

        add_method(node.op.name,
                   node.op.size,
                   node.op.alpha,
                   node.op.beta,
                   node.op.k,
                   node.input_names[0],
                   node.output_names[0])


@register
class DlcRoiAlignTranslation(BackendTranslationBase):
    TARGET = op_adapter.RoiAlignOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        backend.model.add_roialign_layer(node.op.name,
                                         node.op.spatial_scale,
                                         node.op.pooled_size_h,
                                         node.op.pooled_size_w,
                                         node.op.sampling_ratio,
                                         node.input_names[0],
                                         node.input_names[1],
                                         node.output_names[0],
                                         node.output_names[1] if len(node.output_names) > 1 else "",
                                         node.op.tiled_batch_h,
                                         node.op.tiled_batch_w,
                                         node.op.batch_pad_h,
                                         node.op.batch_pad_w,
                                         node.op.pad_value)


@register
class DlcRoiPoolingTranslation(BackendTranslationBase):
    TARGET = op_adapter.RoiPoolingOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        log_assert(node.op.output_shape[0] == 1,
                   code_to_message.get_error_message("ERROR_ROI_POOL_BATCH_UNSUPPORTED"))

        backend.model.add_roipooling_layer(node.op.name,
                                           node.op.pooled_size_w,
                                           node.op.pooled_size_h,
                                           node.op.spatial_scale,
                                           node.op.output_shape,
                                           node.input_names,
                                           node.output_names[0])


@register
class DlcResizeTranslation(BackendTranslationBase):
    TARGET = op_adapter.ResizeOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        supported_modes = {'nearest': ir_consts_to_dlc[op_adapter.ResizeOp.Mode.NEAREST_NEIGHBOR],
                           # for now mapping linear to bilinear since pytorch bilinear is
                           # changing to linear when model gets exported to onnx.
                           'linear': ir_consts_to_dlc[op_adapter.ResizeOp.Mode.BILINEAR],
                           'bilinear': ir_consts_to_dlc[op_adapter.ResizeOp.Mode.BILINEAR]}
        node.op.resize_mode = supported_modes[node.op.resize_mode]

        backend.model.add_scaling_layer(node.op.name,
                                        node.op.output_shape,
                                        node.op.pad_value,
                                        node.op.maintain_aspect_ratio,
                                        node.op.resize_mode,
                                        node.op.scale_height,
                                        node.op.scale_width,
                                        node.input_names[0],
                                        node.output_names[0],
                                        node.op.align_corners,
                                        node.op.half_pixel_centers)


@register
class DlcRnnTransformationTranslation(BackendTranslationBase):
    TARGET = op_adapter.RnnTransformationOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        backend.model.add_tx_layer(node.op.name,
                                   node.op.weights,
                                   node.op.bias,
                                   node.op.activation,
                                   node.input_names[0],
                                   node.output_names[0])


@register
class DlcScaleTranslation(BackendTranslationBase):
    TARGET = op_adapter.ScaleOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        backend.model.add_scale_layer(node.op.name,
                                      node.op.weights,
                                      node.op.bias,
                                      node.input_names,
                                      node.output_names[0],
                                      node.op.axis,
                                      node.op.num_axes,)


@register
class DlcSliceTranslation(BackendTranslationBase):
    TARGET = op_adapter.SliceOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        backend.model.add_slice_layer(node.op.name,
                                      node.input_names[0],
                                      node.op.axis,
                                      node.op.slice_points,
                                      node.output_names)


@register
class DlcStridedSliceTranslation(BackendTranslationBase):
    TARGET = op_adapter.StridedSliceOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        backend.model.add_strided_slice_layer(node.op.name,
                                              node.input_names[0],
                                              node.output_names[0],
                                              node.op.begin,
                                              node.op.end,
                                              node.op.strides,
                                              node.op.shrink_axis_mask,
                                              node.op.begin_mask,
                                              node.op.end_mask,
                                              node.op.new_axis_mask)


@register
class DlcSoftmaxTranslation(BackendTranslationBase):
    TARGET = op_adapter.SoftmaxOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        input_rank = graph.get_buffer(node.input_names[0]).rank()
        if node.op.axis != input_rank - 1:
            raise ValueError("Unsupported axis value on node {}, got {}, expected input rank minus 1, {}".
                             format(node.op.name, node.op.axis, input_rank - 1))
        backend.model.add_softmax_layer(node.op.name,
                                        node.input_names[0],
                                        node.output_names[0])


@register
class DlcSpaceToDepthTranslation(BackendTranslationBase):
    TARGET = op_adapter.SpaceToDepthOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        backend.model.add_space_to_depth_layer(name=node.op.name,
                                               input_name=node.input_names[0],
                                               output_name=node.output_names[0],
                                               downscale_factor=node.op.downscale_factor,
                                               data_format=node.op.data_format)


@register
class DlcSsdTranslation(BackendTranslationBase):
    TARGET = op_adapter.SsdOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        backend.model.add_box_decoder_layer(node.op.name,
                                            node.input_names,
                                            [node.output_names[0]],
                                            scale_y=node.op.scale_y,
                                            scale_x=node.op.scale_x,
                                            scale_h=node.op.scale_h,
                                            scale_w=node.op.scale_w)


@register
class DlcSubtractMeanTranslation(BackendTranslationBase):
    TARGET = op_adapter.SubtractMeanOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        backend.model.add_subtract_mean_layer(node.op.name,
                                              node.op.mean_values,
                                              node.input_names[0],
                                              node.output_names[0])


@register
class DlcTileTranslation(BackendTranslationBase):
    TARGET = op_adapter.TileOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        backend.model.add_tile_layer(name=node.op.name,
                                     multiples=node.op.multiples,
                                     input_name=node.input_names[0],
                                     output_name=node.output_names[0])


@register
class DlcUdlTranslation(BackendTranslationBase):
    TARGET = op_adapter.UdlOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        backend.model.add_user_defined_layer(node.op.name,
                                             node.op.layer_type,
                                             node.input_names,
                                             node.output_names,
                                             node.op.output_dims,
                                             node.op.blob)


@register
class DlcUdoTranslation(BackendTranslationBase):
    TARGET = op_adapter.CustomOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        # placed here to avoid import errors when UDO flag is turned off
        udo_ir_consts_to_dlc = {"SNPE_UDO_DATATYPE_FLOAT_16": modeltools.SNPE_UDO_DATATYPE_FLOAT_16,
                                "SNPE_UDO_DATATYPE_FLOAT_32": modeltools.SNPE_UDO_DATATYPE_FLOAT_32,
                                "SNPE_UDO_DATATYPE_FIXED_4": modeltools.SNPE_UDO_DATATYPE_FIXED_4,
                                "SNPE_UDO_DATATYPE_FIXED_8": modeltools.SNPE_UDO_DATATYPE_FIXED_8,
                                "SNPE_UDO_DATATYPE_FIXED_16": modeltools.SNPE_UDO_DATATYPE_FIXED_16,
                                "SNPE_UDO_DATATYPE_FIXED_32": modeltools.SNPE_UDO_DATATYPE_FIXED_32,
                                "SNPE_UDO_DATATYPE_UINT_8": modeltools.SNPE_UDO_DATATYPE_UINT_8,
                                "SNPE_UDO_DATATYPE_UINT_16": modeltools.SNPE_UDO_DATATYPE_UINT_16,
                                "SNPE_UDO_DATATYPE_UINT_32": modeltools.SNPE_UDO_DATATYPE_UINT_32,
                                "SNPE_UDO_DATATYPE_INT_32": modeltools.SNPE_UDO_DATATYPE_INT_32,
                                "SNPE_UDO_DATATYPE_LAST": modeltools.SNPE_UDO_DATATYPE_LAST,

                                "SNPE_UDO_LAYOUT_NHWC": modeltools.SNPE_UDO_LAYOUT_NHWC,
                                "SNPE_UDO_LAYOUT_NCHW": modeltools.SNPE_UDO_LAYOUT_NCHW,
                                "SNPE_UDO_LAYOUT_NDHWC": modeltools.SNPE_UDO_LAYOUT_NDHWC,
                                "SNPE_UDO_LAYOUT_GPU_OPTIMAL1": modeltools.SNPE_UDO_LAYOUT_GPU_OPTIMAL1,
                                "SNPE_UDO_LAYOUT_GPU_OPTIMAL2": modeltools.SNPE_UDO_LAYOUT_GPU_OPTIMAL2,
                                "SNPE_UDO_LAYOUT_DSP_OPTIMAL1": modeltools.SNPE_UDO_LAYOUT_DSP_OPTIMAL1,
                                "SNPE_UDO_LAYOUT_DSP_OPTIMAL2": modeltools.SNPE_UDO_LAYOUT_DSP_OPTIMAL2,
                                "SNPE_UDO_LAYOUT_LAST": modeltools.SNPE_UDO_LAYOUT_LAST,

                                "SNPE_UDO_PARAMTYPE_SCALAR": modeltools.SNPE_UDO_PARAMTYPE_SCALAR,
                                "SNPE_UDO_PARAMTYPE_STRING": modeltools.SNPE_UDO_PARAMTYPE_STRING,
                                "SNPE_UDO_PARAMTYPE_TENSOR": modeltools.SNPE_UDO_PARAMTYPE_TENSOR,
                                "SNPE_UDO_PARAMTYPE_LAST": modeltools.SNPE_UDO_PARAMTYPE_LAST}

        # the following are properties of a udo tensor param, where a tensor param may be an input,
        # output or conventional attribute like kernel_size etc.
        # Note each input, output and tensor param will have these properties defined.
        udo_tensor_param_attrs = ['layout', 'param_type', 'data_type']

        def resolve_to_external(name, attr_dict):
            # delete attributes not used by SNPE
            if hasattr(attr_dict[name], "per_core_data_types"):
                del attr_dict[name]["per_core_data_types"]
            if hasattr(attr_dict[name], "allowed_data_types"):
                del attr_dict[name]["allowed_data_types"]
            if hasattr(attr_dict[name], "default_value"):
                del attr_dict[name]["default_value"]
            # resolve other types to C++ enum values
            for udo_tensor_attr in udo_tensor_param_attrs:
                if udo_tensor_attr in attr_dict[name]:
                    attr_dict[name][udo_tensor_attr] = udo_ir_consts_to_dlc[
                        attr_dict[name][udo_tensor_attr]]

        updated_inputs = []
        updated_outputs = []
        updated_attrs = []

        # get list of updated input names as input names may have changed due to optimizations
        # change the input dict to an input list, and add the name, to preserve order in pybind
        for i, (name, value) in enumerate(list(node.op.inputs.items())):
            if name != node.input_names[i]:
                name = node.input_names[i]
            updated_inputs.append(value)
            updated_inputs[i]["name"] = name

        # get list of updated output names as output names may have changed due to optimizations
        # we change the output dict to an output list, and add the name, to preserve order in pybind
        for i, (name, value) in enumerate(list(node.op.outputs.items())):
            if name != node.output_names[i]:
                name = node.output_names[i]
            updated_outputs.append(value)
            updated_outputs[i]["name"] = name

        # changes udo tensor param attrs for inputs, outputs and tensor params
        # into modeltools constants
        # e.x changes kernel_size['tensor_layout'] from 'SNPE_UDO_LAYOUT_NHWC'
        # to modeltools.SNPE_UDO_LAYOUT_NHWC
        attrs = node.op.tensor_params
        attrs.update(node.op.scalar_params)
        for name in attrs:
            resolve_to_external(name, attrs)
        for name in node.op.inputs:
            resolve_to_external(name, node.op.inputs)
        for name in node.op.outputs:
            resolve_to_external(name, node.op.outputs)

        node.op.inputs = updated_inputs
        node.op.outputs = updated_outputs
        attrs_list = []

        # SNPE needs attributes to be formatted as a list
        for key, value in attrs.items():
            # each attribute must have a name key
            if "name" not in value.keys():
                value["name"] = key
            attrs_list.append(value)
        backend.model.add_udo_layer(node.op.name,
                            node.op.output_dims,
                            str(node.op.package_name),
                            str(node.op.custom_type),
                            node.op.inputs,
                            node.op.outputs,
                            attrs_list)


@register
class DlcUnpackTranslation(BackendTranslationBase):
    TARGET = op_adapter.UnpackOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        backend.model.add_unpack_layer(name=node.op.name,
                                       input_name=node.input_names[0],
                                       output_names=node.output_names,
                                       axis=node.op.axis,
                                       num=node.op.num)


@register
class DlcUpsampleIndexBaseTranslation(BackendTranslationBase):
    TARGET = op_adapter.UpsampleIndexBasedOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        pool_id = self.model.get_layer_id(node.input_names[1])
        backend.model.add_upsample_layer(node.op.name,
                                         node.op.pool_size,
                                         node.op.pool_stride,
                                         node.op.pad,
                                         node.op.output_height,
                                         node.op.output_width,
                                         node.input_names[0],
                                         node.output_names[0],
                                         pool_id)


@register
class DlcUpsampleSparseTranslation(BackendTranslationBase):
    TARGET = op_adapter.UpsampleSparseOp.TRANSLATION_KEY

    def add_op_to_backend(self, node, graph, backend, **kwargs):
        backend.model.add_upsample_layer(node.op.name,
                                         node.op.pool_size,
                                         node.op.pool_stride,
                                         node.op.pad,
                                         node.op.output_height,
                                         node.op.output_width,
                                         node.input_names[0],
                                         node.output_names[0])
