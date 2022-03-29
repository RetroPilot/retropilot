# ==============================================================================
#
#  Copyright (c) 2021-2022 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import numpy as np

from qti.aisw.converters.common.converter_ir.axis_tracker import AxisTracker
from qti.aisw.converters.common.converter_ir.op_graph import IROpGraph
from qti.aisw.converters.common.converter_ir.op_adapter import IRPaddingStrategies
from qti.aisw.converters.common.utils import translation_utils
from qti.aisw.converters.common.utils.converter_utils import (
    converter_type,
    log_assert,
    log_debug1,
    log_debug3,
)
from qti.aisw.converters.common.converter_ir.op_adapter import (
    BatchnormOp,
    ChannelShuffleOp,
    ConstantOp,
    ConvolutionOp,
    DeconvolutionOp,
    DepthwiseConvolutionOp,
    DetectionOutputOp,
    ElementwiseSumOp,
    FullyConnectedOp,
    MatMulOp,
    NeuronOp,
    NoopOp,
    PadOp,
    PixelShuffleOp,
    PoolOp,
    PreluOp,
    ResizeOp,
    SoftmaxOp
)

from qti.aisw.converters.relay.translations.relay_translations import RelayTranslationBase
from qti.aisw.converters.relay.translations import RelayTranslations

import tvm
from tvm import relay


# ------------------------------------------------------------------------------
#   Adaptive Average Pool2D
# ------------------------------------------------------------------------------
class RelayAdaptiveAvgPool2DTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayAdaptiveAvgPool2DTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}
        adaptive_avg_pool_attr = relay_expr.attrs
        attr_dict['layout'] = adaptive_avg_pool_attr.layout
        attr_dict["output_size"] = adaptive_avg_pool_attr.output_size if hasattr(adaptive_avg_pool_attr, 'output_size') else None

        log_debug3("\tlayout {}", attr_dict['layout'])
        log_debug3("\toutput_size {}", attr_dict['output_size'])

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, PoolOp.TRANSLATION_KEY)
        input_shape = converter_context.get_input_shapes(relay_expr)[0]
        data_layout = attr_dict['layout']
        output_size = attr_dict['output_size']

        if data_layout != "NHWC":
            raise ValueError("No support {} data layout".format(data_layout))

        if output_size is None:
            size_y = input_shape[1]
            size_x = input_shape[2]
            stride_y = 1
            stride_x = 1
        else:
            h = input_shape[1]
            w = input_shape[2]
            output_size_h = int(output_size[0])
            output_size_w = int(output_size[1]) if len(output_size) == 2 else int(output_size[0])
            stride_y = int(h / output_size_h)
            stride_x = int(w / output_size_w)
            size_y = h - (output_size_h - 1) * stride_y
            size_x = w - (output_size_w - 1) * stride_x

        log_debug3("\tstride_y {}", stride_y)
        log_debug3("\tstride_x {}", stride_x)
        log_debug3("\tsize_y {}", size_y)
        log_debug3("\tsize_x {}", size_x)


        ir_op = PoolOp(op_name,
                       pool_type=PoolOp.Type.AVG,
                       size_x=size_x,
                       size_y=size_y,
                       stride_x=stride_x,
                       stride_y=stride_y)
        return ir_op


RelayTranslations.register_translation(RelayAdaptiveAvgPool2DTranslation(),
                                       converter_type('adaptive_avg_pool2d', 'relay'))


# ------------------------------------------------------------------------------
#   BatchNorm
# ------------------------------------------------------------------------------
class RelayBatchNormTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayBatchNormTranslation, self).__init__()

    # Returns a dictionary of parameters
    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):

        attr_dict = {}
        batchnorm_attrs = relay_expr.attrs

        attr_dict["epsilon"] = batchnorm_attrs.epsilon if hasattr(batchnorm_attrs, 'epsilon') else 1e-5
        attr_dict["center"] = batchnorm_attrs.center if hasattr(batchnorm_attrs, 'center') else True
        attr_dict["scale"] = batchnorm_attrs.scale if hasattr(batchnorm_attrs, 'scale') else True
        attr_dict["axis"] = batchnorm_attrs.axis if hasattr(batchnorm_attrs, 'axis') else 1

        log_debug3("\tepsilon {}", attr_dict["epsilon"])
        log_debug3("\tcenter {}", attr_dict["center"])
        log_debug3("\tscale {}", attr_dict["scale"])
        log_debug3("\taxis {}", attr_dict["axis"])

        return  attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):

        op_name = converter_context.get_op_name(relay_expr, BatchnormOp.TRANSLATION_KEY)

        gamma = relay_params[input_names[1]].asnumpy()
        beta = relay_params[input_names[2]].asnumpy()
        moving_mean = relay_params[input_names[3]].asnumpy()
        moving_var = relay_params[input_names[4]].asnumpy()

        log_debug3("\tgamma shape {}", gamma.shape)
        log_debug3("\tbeta shape {}", beta.shape)
        log_debug3("\tmoving_mean shape {}", moving_mean.shape)
        log_debug3("\tmoving_var shape {}", moving_var.shape)

        if attr_dict["axis"] != 3:
            raise ValueError("In NHWC data layout, batchnorm channel is dimension 3, got {}".format(attr_dict["axis"]))

        center = attr_dict["center"]
        scale = attr_dict["scale"]
        epsilon = attr_dict["epsilon"]
        # weights = gamma/sqrt(var+epsilon)
        weights = gamma / np.sqrt(moving_var + epsilon)
        # bias = -mu/sqrt(var+epsilon)
        bias = -moving_mean / np.sqrt(moving_var + epsilon)
        if scale:
            # bias = -mu*gamma/sqrt(var+epsilon)
            bias *= gamma
        if center:
            # bias = -mu/sqrt(var+epsilon) + beta or bias = -mu*gamma/sqrt(var+epsilon) + beta
            bias += beta

        ir_op = BatchnormOp(op_name,
                            weights=weights,
                            bias=bias,
                            epsilon=epsilon,
                            gamma=gamma,
                            beta=beta)

        for name in input_names[1:]:
            input_names.remove(name)

        return ir_op


RelayTranslations.register_translation(RelayBatchNormTranslation(),
                                       converter_type('batch_norm', 'relay'))


# ------------------------------------------------------------------------------
#   BiasAdd
# ------------------------------------------------------------------------------
class RelayBiasaddTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayBiasaddTranslation, self).__init__()

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):

        op_name = converter_context.get_op_name(relay_expr, ElementwiseSumOp.TRANSLATION_KEY)

        bias = relay_params[input_names[1]]
        if isinstance(bias, tvm.runtime.ndarray.NDArray) or isinstance(bias, tvm.runtime.NDArray):
            bias = bias.asnumpy().astype(np.float32)

        log_debug3("\tbias shape {}", bias.shape)

        bias_output = op_name + "_const_bias"
        input_names[1] = bias_output
        quir_graph.add(ConstantOp(bias_output, bias), [], [bias_output])

        ir_op = ElementwiseSumOp(op_name)

        return ir_op


RelayTranslations.register_translation(RelayBiasaddTranslation(),
                                       converter_type('bias_add', 'relay'))


# ------------------------------------------------------------------------------
#   ChannelShuffle
# ------------------------------------------------------------------------------
class RelayChannelShuffleTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayChannelShuffleTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}
        attr_dict["groups"] = int(relay_expr.attrs.groups)
        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, ChannelShuffleOp.TRANSLATION_KEY)
        return ChannelShuffleOp(op_name, groups=attr_dict["groups"])

RelayTranslations.register_translation(RelayChannelShuffleTranslation(),
                                       converter_type("channel_shuffle", "relay"))


# ------------------------------------------------------------------------------
#   Conv Base
# ------------------------------------------------------------------------------
class RelayConvBaseTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayConvBaseTranslation, self).__init__()

    # Returns a dictionary of parameters
    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}

        conv_attrs = relay_expr.attrs

        attr_dict["kernel_layout"] = conv_attrs.kernel_layout
        log_debug3("\tkernel_layout {}", conv_attrs.kernel_layout)

        padding = [int(val) for val in conv_attrs.padding]
        log_debug3("\tpadding {}", padding)
        if len(padding) == 1:
            pad_left = pad_right = pad_top = pad_bottom = padding[0]
        elif len(padding) == 2:
            pad_top = pad_bottom = padding[0]
            pad_left = pad_right = padding[1]
        elif len(padding) == 4:
            pad_top = padding[0]
            pad_left = padding[1]
            pad_bottom = padding[2]
            pad_right = padding[3]
        else:
            raise ValueError("Unsupported Padding value {}".format(padding))
        attr_dict["pad_top"] = pad_top
        attr_dict["pad_bottom"] = pad_bottom
        attr_dict["pad_left"] = pad_left
        attr_dict["pad_right"] = pad_right

        attr_dict["padding_size_strategy"] = IRPaddingStrategies.PADDING_SIZE_EXPLICIT_FLOOR
        log_debug3("\tpadding strategy {}", attr_dict["padding_size_strategy"])

        strides = [int(val) for val in conv_attrs.strides]
        log_debug3("\tstrides {}", strides)
        stride_x = strides[0]
        stride_y = strides[1]
        attr_dict["stride_x"] = stride_x
        attr_dict["stride_y"] = stride_y

        dilation = [int(val) for val in conv_attrs.dilation]
        log_debug3("\tdilation {}", dilation)
        dilation_x = dilation[0]
        dilation_y = dilation[1]
        attr_dict["dilation_x"] = dilation_x
        attr_dict["dilation_y"] = dilation_y

        groups = int(conv_attrs.groups)
        log_debug3("\tgroups {}", groups)
        attr_dict["groups"] = groups

        return attr_dict


# ------------------------------------------------------------------------------
#   Conv
# ------------------------------------------------------------------------------
class RelayConvTranslation(RelayConvBaseTranslation):
    def __init__(self):
        super(RelayConvTranslation, self).__init__()

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):

        op_name = converter_context.get_op_name(relay_expr, ConvolutionOp.TRANSLATION_KEY)

        if input_names[1] not in relay_params:
            raise ValueError("Unsupported dynamic weights on tensor {}".format(input_names[1]))

        weights = relay_params[input_names[1]]
        if isinstance(weights, tvm.runtime.ndarray.NDArray) or isinstance(weights, tvm.runtime.NDArray):
            weights = weights.asnumpy()
        log_debug3("\tweights shape {}", weights.shape)

        kernel_layout = attr_dict["kernel_layout"]
        if kernel_layout == "HWIO":
            pass
        elif kernel_layout == "HWOI":
            log_debug3("\tHWOI kernel layout with shape {} detected, "
                       "Transposing the weights to make it 'HWIO'.".format(weights.shape))
            weights = np.transpose(weights, AxisTracker.AxisFormat.HWOI_TO_HWIO)
            weights = np.ascontiguousarray(weights)
            kernel_layout = "HWIO"
            log_debug3("\tTransposed weights to be of shape {}", weights.shape)
        else:
            raise ValueError("Unsupported kernel layout {}".format(kernel_layout))

        # Handle marking this Convolution as a DepthwiseConvolution
        num_input_channels = quir_graph.src_axis_order.extract_spatial_dims(
            quir_graph.get_buffer(input_names[0]).shape)[-1]
        num_output_channels = weights.shape[kernel_layout.find('O')]
        convolution_class = ConvolutionOp
        if attr_dict["groups"] == num_input_channels and num_input_channels == num_output_channels:
            convolution_class = DepthwiseConvolutionOp
            log_debug3("\tReshaping depthwise convolution weights of shape {}", weights.shape)
            weights = np.reshape(weights, (weights.shape[0], weights.shape[1], 1, -1))
            log_debug3("\tReshaped depthwise convolution weights to shape {}", weights.shape)

        weights_op = ConstantOp(input_names[1], tensor=weights)
        quir_graph.add(weights_op, [], [input_names[1]], axis_formats=[AxisTracker.AxisFormat.HWIO])

        if len(input_names) > 2:
            if input_names[2] not in relay_params:
                raise ValueError("Unsupported dynamic biases on tensor {}".format(input_names[2]))
            bias = relay_params[input_names[2]]
            if isinstance(bias, tvm.runtime.ndarray.NDArray) or isinstance(bias, tvm.runtime.NDArray):
                bias = bias.asnumpy().astype(np.float32)
            log_debug3("\tbias shape {}", bias.shape)
            bias_op = ConstantOp(input_names[2], tensor=bias)
            quir_graph.add(bias_op, [], [input_names[2]], axis_formats=[AxisTracker.AxisFormat.ANY])

        ir_op = convolution_class(op_name,
                                  padx_before=attr_dict["pad_left"],
                                  padx_after=attr_dict["pad_right"],
                                  pady_before=attr_dict["pad_top"],
                                  pady_after=attr_dict["pad_bottom"],
                                  stridex=attr_dict["stride_x"],
                                  stridey=attr_dict["stride_y"],
                                  dilationx=attr_dict["dilation_x"],
                                  dilationy=attr_dict["dilation_y"],
                                  groups=attr_dict["groups"],
                                  padding_size_strategy=attr_dict["padding_size_strategy"])

        return ir_op


RelayTranslations.register_translation(RelayConvTranslation(),
                                       converter_type('conv2d', 'relay'))


# ------------------------------------------------------------------------------
#   Conv2D_Transpose
# ------------------------------------------------------------------------------
class RelayConvTransposeTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayConvTransposeTranslation, self).__init__()

    # Returns a dictionary of parameters
    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}

        conv_attrs = relay_expr.attrs

        log_debug3("\tdata layout {}", conv_attrs.data_layout)
        if conv_attrs.data_layout != "NHWC":
            # QUIR expects data to be "NHWC"
            raise ValueError("Unsupported data layout {}".format(conv_attrs.data_layout))

        log_debug3("\tkernel layout {}", conv_attrs.kernel_layout)
        if conv_attrs.kernel_layout != "OIHW":
            raise ValueError("Unsupported kernel layout {}".format(conv_attrs.kernel_layout))
        attr_dict["kernel_layout"] = conv_attrs.kernel_layout

        log_debug3("\tout layout {}", conv_attrs.out_layout)
        if conv_attrs.out_layout != "":
            # This attribute is not supported, so only empty/default is accepted
            raise ValueError("Unsupported out layout {}".format(conv_attrs.out_layout))

        log_debug3("\tout dtype {}", conv_attrs.out_dtype)
        if conv_attrs.out_dtype != "float32":
            # Only float32 is currently supported
            raise ValueError("Unsupported out dtype {}".format(conv_attrs.out_dtype))

        padding = [int(val) for val in conv_attrs.padding]
        log_debug3("\tpadding {}", padding)
        if len(padding) == 1:
            pad_left = pad_right = pad_top = pad_bottom = padding[0]
        elif len(padding) == 2:
            pad_top = pad_bottom = padding[0]
            pad_left = pad_right = padding[1]
        elif len(padding) == 4:
            pad_top = padding[0]
            pad_left = padding[1]
            pad_bottom = padding[2]
            pad_right = padding[3]
        else:
            raise ValueError("Unsupported Padding value {}".format(padding))
        attr_dict["pad_top"] = pad_top
        attr_dict["pad_bottom"] = pad_bottom
        attr_dict["pad_left"] = pad_left
        attr_dict["pad_right"] = pad_right

        attr_dict["padding_size_strategy"] = IRPaddingStrategies.PADDING_SIZE_EXPLICIT_FLOOR
        log_debug3("\tpadding strategy {}", attr_dict["padding_size_strategy"])

        strides = [int(val) for val in conv_attrs.strides]
        log_debug3("\tstrides {}", strides)
        stride_x = strides[0]
        stride_y = strides[1]
        attr_dict["stride_x"] = stride_x
        attr_dict["stride_y"] = stride_y

        dilation = [int(val) for val in conv_attrs.dilation]
        log_debug3("\tdilation {}", dilation)
        dilation_x = dilation[0]
        dilation_y = dilation[1]
        attr_dict["dilation_x"] = dilation_x
        attr_dict["dilation_y"] = dilation_y

        groups = int(conv_attrs.groups)
        log_debug3("\tgroups {}", groups)
        attr_dict["groups"] = groups

        output_padding = conv_attrs.output_padding
        log_debug3("\toutput padding {}", conv_attrs.output_padding)
        attr_dict["output_padding_x"] = output_padding[0]
        attr_dict["output_padding_y"] = output_padding[1]


        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, DeconvolutionOp.TRANSLATION_KEY)

        kernel_layout = attr_dict["kernel_layout"]
        if kernel_layout == "OIHW":
            if input_names[1] not in relay_params:
                raise ValueError("Unsupported dynamic weights on tensor {}".format(input_names[1]))
            weights = relay_params[input_names[1]]
            if isinstance(weights, tvm.runtime.ndarray.NDArray) or isinstance(weights, tvm.runtime.NDArray):
                weights = weights.asnumpy()
            log_debug3("\tweights shape {}", weights.shape)
            weights = np.transpose(weights, AxisTracker.AxisFormat.OIHW_TO_HWOI)
            log_debug3("\ttransposed deconv weights to {}", weights.shape)
        else:
            raise ValueError("Unsupported kernel layout {}".format(kernel_layout))

        weights_op = ConstantOp(input_names[1], tensor=weights)
        quir_graph.add(weights_op, [], [input_names[1]], axis_formats=[AxisTracker.AxisFormat.HWIO])

        if len(input_names) > 2:
            if input_names[2] not in relay_params:
                raise ValueError("Unsupported dynamic biases on tensor {}".format(input_names[2]))
            bias = relay_params[input_names[2]]
            if isinstance(bias, tvm.runtime.ndarray.NDArray) or isinstance(bias, tvm.runtime.NDArray):
                bias = bias.asnumpy().astype(np.float32)
            log_debug3("\tbias shape {}", bias.shape)
            bias_op = ConstantOp(input_names[2], tensor=bias)
            quir_graph.add(bias_op, [], [input_names[2]], axis_formats=[AxisTracker.AxisFormat.ANY])

        ir_op = DeconvolutionOp(op_name,
                                padx_before=attr_dict["pad_left"],
                                padx_after=attr_dict["pad_right"],
                                pady_before=attr_dict["pad_top"],
                                pady_after=attr_dict["pad_bottom"],
                                stridex=attr_dict["stride_x"],
                                stridey=attr_dict["stride_y"],
                                dilationx=attr_dict["dilation_x"],
                                dilationy=attr_dict["dilation_y"],
                                output_paddingx=attr_dict["output_padding_x"],
                                output_paddingy=attr_dict["output_padding_y"],
                                groups=attr_dict["groups"],
                                padding_size_strategy=attr_dict["padding_size_strategy"])

        return ir_op


RelayTranslations.register_translation(RelayConvTransposeTranslation(),
                                       converter_type('conv2d_transpose', 'relay'))


# ------------------------------------------------------------------------------
#   Dense
# ------------------------------------------------------------------------------
class RelayDenseTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayDenseTranslation, self).__init__()

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):

        if input_names[1] in relay_params:
            op_name = converter_context.get_op_name(relay_expr, FullyConnectedOp.TRANSLATION_KEY)

            weights = relay_params[input_names[1]]
            if isinstance(weights, tvm.runtime.ndarray.NDArray) or isinstance(weights, tvm.runtime.NDArray):
                weights = weights.asnumpy()

            # Weights has shape [out_units, in_units]
            bias = np.zeros(weights.shape[-2], dtype=np.float32)

            log_debug3("\tweight shape {}", weights.shape)
            log_debug3("\tbias shape {}", bias.shape)

            ir_op = FullyConnectedOp(op_name, weights, bias)

            # Only data input is needed in IR graph. Weights and Bias inputs are ignored
            for name in input_names[1:]:
                input_names.remove(name)
        else:
            op_name = converter_context.get_op_name(relay_expr, MatMulOp.TRANSLATION_KEY)
            shape_b = quir_graph.get_buffer(input_names[1]).shape
            bias = np.zeros(shape_b[-2], dtype=np.float32)
            ir_op = MatMulOp(op_name, bias=bias, transpose_a=False, transpose_b=True)

        return ir_op


RelayTranslations.register_translation(RelayDenseTranslation(),
                                       converter_type('dense', 'relay'))


# ------------------------------------------------------------------------------
#   DepthToSpace
# ------------------------------------------------------------------------------
class RelayDepthToSpaceTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayDepthToSpaceTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}
        dts_attrs = relay_expr.attrs

        attr_dict["layout"] = dts_attrs.layout
        log_debug3("\tdata layout {}", dts_attrs.layout)
        if dts_attrs.layout != "NHWC":
            raise ValueError("Unsupported kernel layout {}".format(dts_attrs.layout))

        attr_dict["mode"] = dts_attrs.mode
        log_debug3("\tmode {}", dts_attrs.mode)

        attr_dict["upscale_factor"] = dts_attrs.block_size
        log_debug3("\tblock size {}", dts_attrs.block_size)

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, PixelShuffleOp.TRANSLATION_KEY)

        ir_op = PixelShuffleOp(op_name,
                               upscale_factor=attr_dict["upscale_factor"],
                               data_format=attr_dict["layout"])

        return ir_op


RelayTranslations.register_translation(RelayDepthToSpaceTranslation(),
                                       converter_type('depth_to_space', 'relay'))


# ------------------------------------------------------------------------------
#   Detecion PostPorcess
# ------------------------------------------------------------------------------
class RelayDetectionPostProcessTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayDetectionPostProcessTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = dict(relay_expr.attrs)
        attr_dict['variance_encoded_in_target'] = False if attr_dict['variance_encoded_in_target'] == 0 else True
        attr_dict['share_location'] =  False if attr_dict['share_location'] == 0 else True

        log_debug3("\tvariance_encoded_in_target {}", attr_dict['variance_encoded_in_target'])
        log_debug3("\tshare_location {}", attr_dict['share_location'])

        for k, v in attr_dict.items():
            if isinstance(v, tvm.tir.expr.IntImm):
                attr_dict[k] = v.value
            elif isinstance(v, tvm.tir.expr.FloatImm):
                attr_dict[k] = v.value
            elif isinstance(v, tvm.ir.container.Array):
                attr_dict[k] = list(v)
            elif isinstance(v, tvm.runtime.container.String):
                attr_dict[k] = str(v)
            log_debug3("\t{} {}", k, v)

        return attr_dict


    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, DetectionOutputOp.TRANSLATION_KEY)
        if input_names[2] not in relay_params:
            raise ValueError("Unsupported dynamic weights on tensor {}".format(input_names[2]))
        quir_graph.add(ConstantOp(input_names[2], relay_params[input_names[2]].asnumpy()), [], [input_names[2]])

        ir_op = DetectionOutputOp(op_name,
                                  output_dims=attr_dict['output_dims'],
                                  num_classes=attr_dict['num_classes'],
                                  share_location=attr_dict['share_location'],
                                  background_label_id=attr_dict['background_label_id'],
                                  nms_threshold=attr_dict['nms_threshold'],
                                  confidence_threshold=attr_dict['confidence_threshold'],
                                  nms_top_k=attr_dict['nms_top_k'],
                                  nms_eta=attr_dict['nms_eta'],
                                  code_type=attr_dict['code_type'],
                                  keep_top_k=attr_dict['keep_top_k'],
                                  variance_encoded_in_target=attr_dict['variance_encoded_in_target'],
                                  scale_h=attr_dict['scale_h'],
                                  scale_w=attr_dict['scale_w'],
                                  scale_y=attr_dict['scale_y'],
                                  scale_x=attr_dict['scale_x']
                                  )

        return ir_op


RelayTranslations.register_translation(RelayDetectionPostProcessTranslation(),
                                       converter_type('detection_postprocess', 'relay'))


# ------------------------------------------------------------------------------
#   Dropout
# ------------------------------------------------------------------------------
class RelayDropoutTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayDropoutTranslation, self).__init__()

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, NoopOp.TRANSLATION_KEY)
        return NoopOp(op_name)


RelayTranslations.register_translation(RelayDropoutTranslation(),
                                       converter_type('dropout', 'relay'))


# ------------------------------------------------------------------------------
#   Global Average Pool2D
# ------------------------------------------------------------------------------
class RelayGlobalAvgPool2DTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayGlobalAvgPool2DTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}
        attr_dict['layout'] = relay_expr.attrs.layout

        log_debug3("\tlayout {}", attr_dict['layout'])

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, PoolOp.TRANSLATION_KEY)
        input_shape = converter_context.get_input_shapes(relay_expr)[0]
        layout = attr_dict['layout']

        if layout == "NHWC":
            size_x = input_shape[2]
            size_y = input_shape[1]
        else:
            raise ValueError("No support {} data layout".format(layout))

        log_debug3("\tsize_x {}", size_x)
        log_debug3("\tsize_y {}", size_y)

        ir_op = PoolOp(op_name,
                       pool_type=PoolOp.Type.AVG,
                       size_x=size_x,
                       size_y=size_y)
        return ir_op


RelayTranslations.register_translation(RelayGlobalAvgPool2DTranslation(),
                                       converter_type('global_avg_pool2d', 'relay'))


# ------------------------------------------------------------------------------
#   LeakyRelu
# ------------------------------------------------------------------------------
class RelayLeakyReluTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayLeakyReluTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}

        leaky_relu_attrs = relay_expr.attrs
        attr_dict["alpha"] = leaky_relu_attrs.alpha
        log_debug3("\talpha {}", attr_dict["alpha"])

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):

        op_name = converter_context.get_op_name(relay_expr, PreluOp.TRANSLATION_KEY)

        alpha = attr_dict["alpha"]
        coeff = alpha * np.ones(quir_graph.get_buffer(input_names[0]).shape[-1], dtype=np.float32)

        ir_op = PreluOp(op_name,
                        coeff=coeff)
        return ir_op


RelayTranslations.register_translation(RelayLeakyReluTranslation(),
                                       converter_type('leaky_relu', 'relay'))


# ------------------------------------------------------------------------------
#   Logistic
# ------------------------------------------------------------------------------
class RelayLogisticTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayLogisticTranslation, self).__init__()

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, NeuronOp.TRANSLATION_KEY)

        ir_op = NeuronOp(op_name,
                         NeuronOp.Type.LOGISTIC,
                         a=1.0)
        return ir_op


RelayTranslations.register_translation(RelayLogisticTranslation(),
                                       converter_type('sigmoid', 'relay'))


# ------------------------------------------------------------------------------
#   PadOp
# ------------------------------------------------------------------------------
class RelayPadTranslation(RelayTranslationBase):
    class RelayPadMode:
        CONSTANT = 'constant'
        REFLECT = 'reflect'
        EDGE = 'edge'
    def __init__(self):
        super(RelayPadTranslation, self).__init__()
        self.supported_modes = {self.RelayPadMode.CONSTANT : PadOp.Mode.CONSTANT,
                                self.RelayPadMode.REFLECT : PadOp.Mode.REFLECT,
                                self.RelayPadMode.EDGE : PadOp.Mode.EDGE}

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}
        pad_pairs = list()
        for pad in relay_expr.attrs.pad_width:
            pad_pairs.append([int(i) for i in pad])

        attr_dict["pad_pairs"] = pad_pairs
        attr_dict["pad_mode"] = relay_expr.attrs.pad_mode

        # pad value from float, or tvm.relay.Expr, optional, default=0
        # if not in relay_expr.attrs, it will be default value or tvm.relay.Expr
        if hasattr(relay_expr.attrs, 'pad_value'):
            attr_dict["pad_value"] = relay_expr.attrs.pad_value
        else:
            attr_dict["pad_value"] = None

        log_debug3("\tpad_pairs {}", pad_pairs)
        log_debug3("\tpad_mode {}", attr_dict["pad_mode"])

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, PadOp.TRANSLATION_KEY)

        pad_pairs = attr_dict["pad_pairs"]
        pad_pairs = np.asarray(pad_pairs, dtype=np.dtype('int32'))
        mode = attr_dict["pad_mode"]
        pad_value = attr_dict["pad_value"]

        if pad_value is None:
            # pad constant value from inputs[1] expr.Constant
            # if no found constant from param, set to zero by default
            pad_value_op_name = input_names[1]
            if pad_value_op_name in relay_params:
                expr_const_pad_value = relay_params[pad_value_op_name]
                pad_value = float(expr_const_pad_value.asnumpy())
            else:
                log_debug2("\tNo Padding value, use default as zero")

                pad_value = 0

        log_debug3("\tpad_value {}", pad_value)

        ir_op = PadOp(op_name,
                        pads=pad_pairs,
                        constant_value=pad_value,
                        mode=self.supported_modes[mode])

        # Only data input is needed in IR graph. Pad value input is ignored
        for name in input_names[1:]:
            input_names.remove(name)

        return ir_op


RelayTranslations.register_translation(RelayPadTranslation(),
                                       converter_type('pad', 'relay'))


# ------------------------------------------------------------------------------
#   Pooling Base
# ------------------------------------------------------------------------------
class RelayPoolingBaseTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayPoolingBaseTranslation, self).__init__()

    # Returns a dictionary of parameters
    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}
        pool_attrs = relay_expr.attrs

        if pool_attrs.layout != "NHWC":
            raise ValueError("Unsupported layout {}".format(pool_attrs.layout))

        pool_size = pool_attrs.pool_size
        log_debug3("\tpool_size {}", pool_size)
        if isinstance(pool_size, int):
            attr_dict["size_x"] = attr_dict["size_y"] = int(pool_size)
        else:
            attr_dict["size_x"] = int(pool_size[0])
            attr_dict["size_y"] = int(pool_size[1])

        padding = pool_attrs.padding
        log_debug3("\tpadding {}", padding)
        if len(padding) == 2:
            pad_top = pad_bottom = int(padding[0])
            pad_left = pad_right = int(padding[1])
        elif len(padding) == 4:
            pad_top = int(padding[0])
            pad_left = int(padding[1])
            pad_bottom = int(padding[2])
            pad_right = int(padding[3])
        else:
            raise ValueError("Unsupported Padding value {}".format(padding))
        attr_dict["pady_before"] = pad_top
        attr_dict["pady_after"] = pad_bottom
        attr_dict["padx_before"] = pad_left
        attr_dict["padx_after"] = pad_right

        strides = [int(val) for val in pool_attrs.strides]
        log_debug3("\tstrides {}", strides)
        stride_x = strides[0]
        stride_y = strides[1]
        attr_dict["stride_x"] = int(stride_x)
        attr_dict["stride_y"] = int(stride_y)

        ceil_mode = getattr(pool_attrs, "ceil_mode", False)
        if ceil_mode:
            attr_dict["padding_size_strategy"] = IRPaddingStrategies.PADDING_SIZE_EXPLICIT
        else:
            attr_dict["padding_size_strategy"] = IRPaddingStrategies.PADDING_SIZE_EXPLICIT_FLOOR
        log_debug3("\tpadding strategy {}", attr_dict["padding_size_strategy"])

        attr_dict["pool_region_include_padding"] = getattr(pool_attrs, "count_include_pad", False)
        log_debug3("\tpool_region_include_padding {}", attr_dict["pool_region_include_padding"])

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, PoolOp.TRANSLATION_KEY)

        ir_op = PoolOp(op_name,
                       pool_type=attr_dict["pool_type"],
                       size_x=attr_dict["size_x"],
                       size_y=attr_dict["size_y"],
                       stride_x=attr_dict["stride_x"],
                       stride_y=attr_dict["stride_y"],
                       padx_before=attr_dict["padx_before"],
                       padx_after=attr_dict["padx_after"],
                       pady_before=attr_dict["pady_before"],
                       pady_after=attr_dict["pady_after"],
                       padding_size_strategy=attr_dict["padding_size_strategy"],
                       pool_region_include_padding=attr_dict["pool_region_include_padding"])

        return ir_op


# ------------------------------------------------------------------------------
#   AvgPooling2D
# ------------------------------------------------------------------------------
class RelayAvgPoolTranslation(RelayPoolingBaseTranslation):
    def __init__(self):
        super(RelayAvgPoolTranslation, self).__init__()

    # Returns a dictionary of parameters
    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = super().extract_attributes(relay_expr, relay_params)
        attr_dict["pool_type"] = PoolOp.Type.AVG
        return attr_dict


RelayTranslations.register_translation(RelayAvgPoolTranslation(),
                                       converter_type('avg_pool2d', 'relay'))


# ------------------------------------------------------------------------------
#   MaxPooling2D
# ------------------------------------------------------------------------------
class RelayMaxPoolTranslation(RelayPoolingBaseTranslation):
    def __init__(self):
        super(RelayMaxPoolTranslation, self).__init__()

    # Returns a dictionary of parameters
    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = super().extract_attributes(relay_expr, relay_params)
        attr_dict["pool_type"] = PoolOp.Type.MAX
        return attr_dict


RelayTranslations.register_translation(RelayMaxPoolTranslation(),
                                       converter_type('max_pool2d', 'relay'))


# ------------------------------------------------------------------------------
#   Prelu
# ------------------------------------------------------------------------------
class RelayPreluTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayPreluTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}
        prelu_attrs = relay_expr.attrs
        attr_dict["axis"] = prelu_attrs.axis
        log_debug3("\taxis {}", attr_dict["axis"])

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):

        op_name = converter_context.get_op_name(relay_expr, PreluOp.TRANSLATION_KEY)

        channel_axis = attr_dict["axis"]
        slope_input_name = input_names[1]

        input_shape = quir_graph.get_buffer(input_names[0]).shape

        log_assert(channel_axis == len(input_shape)-1,
                   "Expect the channel axis is the last dimension, but got "
                   "channel_axis={} for data_tensor_rank={}",
                   channel_axis, len(input_shape))

        log_assert(slope_input_name in relay_params,
                   "Only support PRelu with constant slope(second input). "
                   "But {} is not in relay_params.",
                   slope_input_name)

        slope = relay_params[slope_input_name]
        if isinstance(slope, (tvm.runtime.ndarray.NDArray, tvm.runtime.NDArray)):
            slope = slope.asnumpy().astype(np.float32)

        # Remove the slope input, which is considered as an attribute in IR.
        input_names.pop(1)

        return PreluOp(op_name, coeff=slope)


RelayTranslations.register_translation(RelayPreluTranslation(),
                                       converter_type('prelu', 'relay'))


# ------------------------------------------------------------------------------
#   Relu
# ------------------------------------------------------------------------------
class RelayReluTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayReluTranslation, self).__init__()

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, NeuronOp.TRANSLATION_KEY)

        ir_op = NeuronOp(op_name, NeuronOp.Type.RELU)
        return ir_op


RelayTranslations.register_translation(RelayReluTranslation(),
                                       converter_type('relu', 'relay'))


# ------------------------------------------------------------------------------
#   Softmax
# ------------------------------------------------------------------------------
class RelaySoftmaxTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelaySoftmaxTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}
        attr_dict["axis"] = int(relay_expr.attrs.axis)

        log_debug3("\taxis {}", attr_dict["axis"])

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, SoftmaxOp.TRANSLATION_KEY)

        ir_op = SoftmaxOp(op_name, axis=attr_dict["axis"])
        return ir_op


RelayTranslations.register_translation(RelaySoftmaxTranslation(),
                                       converter_type('softmax', 'relay'))


# ------------------------------------------------------------------------------
#   Upsampling
# ------------------------------------------------------------------------------
class RelayUpsamplingTranslation(RelayTranslationBase):

    # scaling method names in relay
    class ScaleModes:
        BICUBIC = "bicubic"
        BILINEAR = "bilinear"
        NEAREST_NEIGHBOR = "nearest_neighbor"

    # name mapping from relay to quir
    RELAY_CONSTS_TO_IR = {
        ScaleModes.BILINEAR: "bilinear",
        ScaleModes.NEAREST_NEIGHBOR: "nearest"
    }

    def __init__(self):
        super(RelayUpsamplingTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}

        upsampling_attrs = relay_expr.attrs
        attr_dict["scale_h"] = getattr(upsampling_attrs, "scale_h")
        attr_dict["scale_w"] = getattr(upsampling_attrs, "scale_w")
        log_debug3("\tscale_h {}", attr_dict["scale_h"])
        log_debug3("\tscale_w {}", attr_dict["scale_w"])

        attr_dict["layout"] = getattr(upsampling_attrs, "layout")
        log_debug3("\tlayout {}", attr_dict["layout"])

        scale_mode = getattr(upsampling_attrs, "method", self.ScaleModes.NEAREST_NEIGHBOR)
        if scale_mode == self.ScaleModes.BICUBIC:
            raise ValueError("Unsupported scale method {}".format(scale_mode))

        attr_dict["resize_mode"] = self.RELAY_CONSTS_TO_IR[scale_mode]
        log_debug3("\tresize mode {}", attr_dict["resize_mode"])

        attr_dict["align_corners"] = getattr(upsampling_attrs, "align_corners", False)
        log_debug3("\talign_corners {}", attr_dict["align_corners"])

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, ResizeOp.TRANSLATION_KEY)

        input_shape = converter_context.get_input_shapes(relay_expr)[0]

        if attr_dict["layout"] == "NHWC":
            output_shape = [input_shape[0],
                            round(input_shape[1] * attr_dict["scale_h"]),
                            round(input_shape[2] * attr_dict["scale_w"]),
                            input_shape[3]]
        else:
            raise ValueError("Unsupported data layout {}".format(attr_dict["layout"]))

        ir_op = ResizeOp(op_name,
                         output_shape,
                         resize_mode=attr_dict["resize_mode"],
                         align_corners=attr_dict["align_corners"])
        return ir_op


RelayTranslations.register_translation(RelayUpsamplingTranslation(),
                                       converter_type('upsampling', 'relay'))

