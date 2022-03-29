# ==============================================================================
#
#  Copyright (c) 2019-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import collections
import numpy

import caffe
import caffe.proto.caffe_pb2 as caffe_pb2

from .caffe_base_translation import CaffeTranslationBase, CaffeTranslations
from qti.aisw.converters.common.converter_ir import op_adapter
from qti.aisw.converters.common.converter_ir.axis_tracker import AxisTracker
from qti.aisw.converters.common.converter_ir.op_adapter import IRPaddingStrategies
from qti.aisw.converters.common.utils import code_to_message
from qti.aisw.converters.common.utils.converter_utils import *


# helpers
def get_conv_params(conv_param):
    parms_type = collections.namedtuple("ConvParams", ["pad_x", "pad_y", "stridex", "stridey"])
    pad_x, pad_y = 0, 0
    if conv_param.pad_h or conv_param.pad_w:
        pad_x = conv_param.pad_w
        pad_y = conv_param.pad_h
    elif isinstance(conv_param.pad, int):
        # Segnet version of caffe.proto has defined  pad optional (not repeated).
        # It implies that it is scalar rather vector
        pad_x = conv_param.pad
        pad_y = conv_param.pad
    else:
        if len(conv_param.pad) > 0:
            pad_x = conv_param.pad[0]
            pad_y = conv_param.pad[0]
        if len(conv_param.pad) > 1:
            pad_x = conv_param.pad[1]

    stride_x, stride_y = 1, 1
    if conv_param.stride_h or conv_param.stride_w:
        stride_x = conv_param.stride_w
        stride_y = conv_param.stride_h
    elif isinstance(conv_param.stride, int):
        # Segnet version of caffe.proto has defined  stride optional (not repeated).
        # It implies that it is scalar rather vector
        stride_x = conv_param.stride
        stride_y = conv_param.stride
    else:
        if len(conv_param.stride) > 0:
            stride_x = conv_param.stride[0]
            stride_y = conv_param.stride[0]
        if len(conv_param.stride) > 1:
            stride_x = conv_param.stride[1]

    return parms_type(pad_x, pad_y, stride_x, stride_y)


# -----------------------------------------------------------------
# Converter translations
# -----------------------------------------------------------------
class CaffeBatchNormalizationTranslation(CaffeTranslationBase):
    def __init__(self):
        CaffeTranslationBase.__init__(self)

    def extract_parameters(self, layer, graph):
        # from the batch_norm layer we get weights W1 and bias B1:
        # y  = W1.x + B1
        input_name = graph.naming_policy.get_input_names(layer, layer.bottom)[0]
        prev_bias = None
        if hasattr(graph.get_buffer(input_name).producer.op, 'bias'):
            prev_bias = len(graph.get_buffer(input_name).producer.op.bias)
        weights, bias = graph.weights.get_batch_norm_weights(layer, prev_bias)

        if any(numpy.isnan(weights)) or any(numpy.isinf(weights)):
            log_warning("Got NaN/Inf weights in {} layer, initializing with 1's".format(layer.name))
            weights.fill(1)
        if any(numpy.isnan(bias)) or any(numpy.isinf(bias)):
            bias.fill(0)
            log_warning("Got NaN/Inf bias in {} layer, initializing with 0's".format(layer.name))

        # If use_global_stats is False (Caffe training mode) treat this as instance normalization
        compute_statistics = False
        epsilon = layer.batch_norm_param.eps
        if layer.batch_norm_param.HasField("use_global_stats") and not layer.batch_norm_param.use_global_stats:
            # Reset weights and biases to 1s and 0s
            weights.fill(1)
            bias.fill(0)
            compute_statistics = True

        return op_adapter.BatchnormOp(layer.name,
                                      weights=weights,
                                      bias=bias,
                                      compute_statistics=compute_statistics,
                                      use_mu_sigma=True,
                                      across_spatial=True,
                                      epsilon=epsilon)


CaffeTranslations.register_translation(CaffeBatchNormalizationTranslation(),
                                       converter_type('batchnorm', 'caffe'),
                                       op_adapter.BatchnormOp.TRANSLATION_KEY)


# This is placeholder in-case needed in the future(no use-case identified so far),
# hence not registered as part of CaffeTranslations atm and no sdk-tests exercising this layer
class CaffeBNTranslation(CaffeTranslationBase):
    def __init__(self):
        CaffeTranslationBase.__init__(self)

    def extract_parameters(self, layer, graph):
        weights, bias = graph.weights.get_bn_weights(layer)

        return op_adapter.BatchnormOp(layer.name,
                                      weights=weights,
                                      bias=bias,
                                      compute_statistics=False,
                                      use_mu_sigma=False,  # unused
                                      across_spatial=False)  # unused


# CaffeTranslations.register_translation(CaffeBNTranslation(),
#                                        converter_type('bn', 'caffe'))


class CaffeConvTranslation(CaffeTranslationBase):

    _WEIGHT = "_weight"
    _BIAS = "_bias"

    def __init__(self):
        CaffeTranslationBase.__init__(self)
        self.weight_node_name = ""
        self.bias_node_name = ""

    def extract_parameters(self, layer, graph):
        conv_param = layer.convolution_param
        self.weight_node_name = layer.name + self._WEIGHT
        self.bias_node_name = layer.name + self._BIAS

        # Extract and add static weights/biases in Caffe to IR graph as ConstantOp inputs
        bias_term = getattr(conv_param, "bias_term", True)
        c_weights, c_bias = graph.weights.get_conv_weights(layer, bias_term)

        weights_op = op_adapter.ConstantOp(self.weight_node_name, tensor=c_weights)
        log_debug(code_to_message.get_debugging_message('DEBUG_EXTRACT_WEIGHTS')(layer.name, c_weights.shape))
        graph.add(weights_op, [], [self.weight_node_name], axis_formats=[AxisTracker.AxisFormat.OIHW])

        bias_op = op_adapter.ConstantOp(self.bias_node_name, tensor=c_bias)
        log_debug(code_to_message.get_debugging_message('DEBUG_EXTRACT_BIAS')(layer.name, c_bias.shape))
        graph.add(bias_op, [], [self.bias_node_name], axis_formats=[AxisTracker.AxisFormat.ANY])

        # Extract convolution parameters
        groups = getattr(conv_param, "group", 1)
        pad_x, pad_y, stride_x, stride_y = get_conv_params(conv_param)

        dilation_x, dilation_y = 1, 1
        if len(conv_param.dilation) > 0:
            dilation_x = conv_param.dilation[0]
            dilation_y = conv_param.dilation[0]
        if len(conv_param.dilation) > 1:
            dilation_x = conv_param.dilation[1]

        # Determine whether this is a regular ConvolutionOp or DepthwiseConvolutionOp
        input_name = self.extract_input_names(layer, graph)[0]
        num_input_channels = graph.src_axis_order.extract_spatial_dims(graph.get_buffer(input_name).shape)[-1]
        weights_shape = graph.get_buffer(self.weight_node_name).shape
        num_output_channels = graph.src_axis_order.extract_conv_weights_dims(weights_shape)[-1]

        convolution_class = op_adapter.ConvolutionOp
        # Criteria is that groups == num_input_channels (from input) == num_output_channels
        if groups == num_input_channels and num_input_channels == num_output_channels:
            convolution_class = op_adapter.DepthwiseConvolutionOp

        return convolution_class(layer.name,
                                 padx_before=pad_x,
                                 padx_after=pad_x,
                                 pady_before=pad_y,
                                 pady_after=pad_y,
                                 stridex=stride_x,
                                 stridey=stride_y,
                                 dilationx=dilation_x,
                                 dilationy=dilation_y,
                                 groups=groups,
                                 padding_mode=op_adapter.PadOp.Mode.ZERO,
                                 padding_size_strategy=IRPaddingStrategies.PADDING_SIZE_EXPLICIT_FLOOR)

    def extract_input_names(self, layer, graph):
        # Extend source input names with constant nodes added for weights/bias
        src_input_names = list(map(str, layer.bottom))
        src_input_names.extend([self.weight_node_name, self.bias_node_name])
        return src_input_names


CaffeTranslations.register_translation(CaffeConvTranslation(),
                                       converter_type('convolution', 'caffe'),
                                       converter_type(caffe.proto.caffe_pb2.V1LayerParameter.CONVOLUTION, 'caffe'),
                                       op_adapter.ConvolutionOp.TRANSLATION_KEY)


class CaffeCrossCorrelationTranslation(CaffeTranslationBase):
    def __init__(self):
        CaffeTranslationBase.__init__(self)

    def extract_parameters(self, layer, graph):
        return op_adapter.CrossCorrelationOp(name=layer.name)


CaffeTranslations.register_translation(CaffeCrossCorrelationTranslation(),
                                       converter_type('cudnncrosscorrelation', 'caffe'),
                                       op_adapter.CrossCorrelationOp.TRANSLATION_KEY)


class CaffeDeConvTranslation(CaffeConvTranslation):

    def __init__(self):
        CaffeTranslationBase.__init__(self)

    def extract_parameters(self, layer, graph):
        conv_param = layer.convolution_param
        self.weight_node_name = layer.name + self._WEIGHT
        self.bias_node_name = layer.name + self._BIAS

        # Extract and add static weights/biases in Caffe to IR graph as ConstantOp inputs
        bias_term = getattr(conv_param, "bias_term", True)
        c_weights, c_bias = graph.weights.get_deconv_weights(layer, bias_term)
        weights_op = op_adapter.ConstantOp(self.weight_node_name, tensor=c_weights)
        log_debug(code_to_message.get_debugging_message('DEBUG_EXTRACT_WEIGHTS')(layer.name, c_weights.shape))
        graph.add(weights_op, [], [self.weight_node_name], axis_formats=[AxisTracker.AxisFormat.IOHW])

        bias_op = op_adapter.ConstantOp(self.bias_node_name, tensor=c_bias)
        log_debug(code_to_message.get_debugging_message('DEBUG_EXTRACT_BIAS')(layer.name, c_bias.shape))
        graph.add(bias_op, [], [self.bias_node_name], axis_formats=[AxisTracker.AxisFormat.ANY])

        # Extract deconvolution parameters
        pad_x, pad_y, stride_x, stride_y = get_conv_params(conv_param)
        groups = conv_param.group if hasattr(conv_param, "group") else 1

        return op_adapter.DeconvolutionOp(name=layer.name,
                                          stridex=stride_x,
                                          stridey=stride_y,
                                          padx_before=pad_x,
                                          padx_after=pad_x,
                                          pady_before=pad_y,
                                          pady_after=pad_y,
                                          padding_size_strategy=IRPaddingStrategies.PADDING_SIZE_EXPLICIT,
                                          output_height=0,
                                          output_width=0,
                                          groups=groups)


CaffeTranslations.register_translation(CaffeDeConvTranslation(),
                                       converter_type('deconvolution', 'caffe'),
                                       converter_type(caffe.proto.caffe_pb2.V1LayerParameter.DECONVOLUTION, 'caffe'),
                                       op_adapter.DeconvolutionOp.TRANSLATION_KEY)


class CaffeFullyConnectedTranslation(CaffeTranslationBase):
    def __init__(self):
        CaffeTranslationBase.__init__(self)

    def extract_parameters(self, layer, graph):
        # Compute parameters for fc layer
        c_input_names = graph.naming_policy.get_input_names(layer, layer.bottom)
        if len(c_input_names) != 1:
          raise ValueError("FullyConnected expects only 1 input, got {}".format(len(c_input_names)))
        input_depth = graph.get_buffer(c_input_names[0]).get_buf_dims()[1]
        fc_param = layer.inner_product_param
        bias_term = getattr(fc_param, "bias_term", True)
        c_weights, c_bias = graph.weights.get_fc_weights(layer, input_depth, bias_term)

        return op_adapter.FullyConnectedOp(name=layer.name,
                                           weights=c_weights,
                                           bias=c_bias)


CaffeTranslations.register_translation(CaffeFullyConnectedTranslation(),
                                       converter_type('innerproduct', 'caffe'),
                                       converter_type(caffe.proto.caffe_pb2.V1LayerParameter.INNER_PRODUCT, 'caffe'),
                                       op_adapter.FullyConnectedOp.TRANSLATION_KEY)


class CaffeMVNTranslation(CaffeTranslationBase):
    def __init__(self):
        CaffeTranslationBase.__init__(self)

    def extract_parameters(self, layer, graph):
        prev_layer = graph.naming_policy.get_input_names(layer, layer.bottom)[0]
        layer_depth = graph.get_buffer(prev_layer).get_buf_dims()[1]

        # Generate unit weights and zero bias. Makes it compatible with Batchnorm
        weights = numpy.full(layer_depth,1.0,dtype=numpy.float32)
        bias = numpy.full(layer_depth,0.0,dtype=numpy.float32)

        return op_adapter.BatchnormOp(layer.name,
                                      weights=weights,
                                      bias=bias,
                                      compute_statistics=True,
                                      use_mu_sigma=True,
                                      across_spatial=not layer.mvn_param.across_channels,
                                      epsilon=layer.mvn_param.eps,
                                      normalize_variance=layer.mvn_param.normalize_variance)


CaffeTranslations.register_translation(CaffeMVNTranslation(),
                                       converter_type('mvn', 'caffe'))


class CaffeNormalizeTranslation(CaffeTranslationBase):
    def __init__(self):
        CaffeTranslationBase.__init__(self)

    def extract_parameters(self, layer, graph):
        weights = graph.weights.get_normalize_weights(layer).flatten(order='C')

        # from the normalize layer we get weights W:
        # if channel_shared is true, there is only a single weight which we will
        # replicate across the input channels.
        if layer.norm_param.channel_shared:
            input_name = graph.naming_policy.get_input_names(layer, layer.bottom)[0]
            input_depth = graph.get_buffer(input_name).get_buf_dims()[1]
            weights = weights[0] * numpy.ones([input_depth], dtype=numpy.float32)
        # this layer does not support bias values. construct an array of zeros.
        bias = numpy.zeros(shape=[len(weights)], dtype=numpy.float32)

        return op_adapter.BatchnormOp(layer.name,
                                      weights=weights,
                                      bias=bias,
                                      compute_statistics=True,
                                      use_mu_sigma=False,  # compute RMS
                                      across_spatial=layer.norm_param.across_spatial)


CaffeTranslations.register_translation(CaffeNormalizeTranslation(),
                                       converter_type('normalize', 'caffe'))


class CaffePoolTranslation(CaffeTranslationBase):
    def __init__(self):
        CaffeTranslationBase.__init__(self)

    def extract_parameters(self, layer, graph):
        pool_param = layer.pooling_param

        c_pool_type = op_adapter.PoolOp.Type.MAX
        if pool_param.pool:
            c_pool_type = op_adapter.PoolOp.Type.AVG

        size_x = pool_param.kernel_size
        size_y = size_x
        if pool_param.kernel_h or pool_param.kernel_w:
            size_x = pool_param.kernel_w
            size_y = pool_param.kernel_h

        stride_x = pool_param.stride
        stride_y = stride_x
        if pool_param.stride_h or pool_param.stride_w:
            stride_x = pool_param.stride_w
            stride_y = pool_param.stride_h

        pad_x = pool_param.pad
        pad_y = pad_x
        if pool_param.pad_h or pool_param.pad_w:
            pad_x = pool_param.pad_w
            pad_y = pool_param.pad_h

        include_padding = True
        input_name = graph.naming_policy.get_input_names(layer, layer.bottom)[0]
        input_dim = graph.get_buffer(input_name).get_buf_dims()
        if pool_param.global_pooling:
            size_y = input_dim[2]
            size_x = input_dim[3]
            stride_x, stride_y = 1, 1
            pad_x, pad_y = 0, 0
            include_padding = False

        # if there is a second top, this will be upsampled later.
        if len(layer.top) > 1:
            if size_x != size_y or stride_x != stride_y or pad_x != pad_y:
                raise ValueError(
                    code_to_message.get_error_message('ERROR_CAFFE_INDEX_BASED_UPSAMPLING_DOES_NOT_SUPPORT_RECT_POOL')
                    (str(layer.name)))  # TODO: should this go inside ir-to-dlc?

        return op_adapter.PoolOp(layer.name,
                                 pool_type=c_pool_type,
                                 size_x=size_x,
                                 size_y=size_y,
                                 stride_x=stride_x,
                                 stride_y=stride_y,
                                 padx_before=pad_x,
                                 padx_after=pad_x,
                                 pady_before=pad_y,
                                 pady_after=pad_y,
                                 padding_size_strategy=IRPaddingStrategies.PADDING_SIZE_EXPLICIT,
                                 pool_region_include_padding=include_padding)


CaffeTranslations.register_translation(CaffePoolTranslation(),
                                       converter_type('pooling', 'caffe'),
                                       converter_type(caffe.proto.caffe_pb2.V1LayerParameter.POOLING, 'caffe'),
                                       op_adapter.PoolOp.TRANSLATION_KEY)


class CaffeRoiPoolTranslation(CaffeTranslationBase):
    def __init__(self):
        CaffeTranslationBase.__init__(self)

    def extract_parameters(self, layer, graph):
        roi_pool_param = layer.roi_pooling_param

        pooled_size_h = roi_pool_param.pooled_h
        pooled_size_w = roi_pool_param.pooled_w
        spatial_scale = roi_pool_param.spatial_scale

        # The output depth is equal to the input feature map depth. We are assuming that input[0] is the feature map.
        input_name = graph.naming_policy.get_input_names(layer, layer.bottom)[0]
        input_dims = graph.get_buffer(input_name).get_buf_dims()
        output_dim = [input_dims[0], input_dims[1], pooled_size_h, pooled_size_w]
        log_debug(code_to_message.get_debugging_message("DEBUG_INFERRED_SHAPE")(layer.name, output_dim))

        return op_adapter.RoiPoolingOp(layer.name,
                                       output_shape=output_dim,
                                       pooled_size_h=pooled_size_h,
                                       pooled_size_w=pooled_size_w,
                                       spatial_scale=spatial_scale)

    def infer_output_shapes(self, op, input_shapes):
        return [op.output_shape]


CaffeTranslations.register_translation(CaffeRoiPoolTranslation(),
                                       converter_type('roipooling', 'caffe'),
                                       op_adapter.RoiPoolingOp.TRANSLATION_KEY)


class CaffeScaleTranslation(CaffeTranslationBase):
    def __init__(self):
        CaffeTranslationBase.__init__(self)

    def add_op(self, src_op, graph):
        ops = self.extract_parameters(src_op, graph)
        for op in ops:
            if op.type == op_adapter.ScaleOp.TRANSLATION_KEY:
                input_names = self.extract_input_names(src_op, graph)

            else:
                input_names = graph.list_nodes()[-1].output_names[0]  # make input previous op's output
            output_names = self.extract_output_names(src_op, graph)
            graph.add(op, input_names, output_names)

    def extract_parameters(self, layer, graph):
        scale_param = layer.scale_param
        bias_term = getattr(scale_param, "bias_term", False)
        input_name = graph.naming_policy.get_input_names(layer, layer.bottom)[0]
        input_dim = graph.get_buffer(input_name).get_buf_dims()
        input_depth = input_dim[1]

        s_weights, s_bias = graph.weights.get_scale_weights(layer, bias_term, input_depth)
        axis = scale_param.axis
        if axis < 0:
            axis = len(input_dim) + axis
        num_axes = scale_param.num_axes

        return [op_adapter.ScaleOp(layer.name,
                                   weights=s_weights,
                                   bias=s_bias,
                                   axis=axis,
                                   num_axes=num_axes)]


CaffeTranslations.register_translation(CaffeScaleTranslation(),
                                       converter_type('scale', 'caffe'),
                                       op_adapter.ScaleOp.TRANSLATION_KEY)


# This is placeholder in-case needed in the future(no use-case identified so far),
# hence not registered as part of CaffeTranslations atm and no sdk-tests exercising this layer
# Note: Caffe upsample is different from say onnx upsample. Upsample in onnx is resize(which is what the new op is
#       now).
class CaffeUpsampleTranslation(CaffeTranslationBase):
    def __init__(self):
        CaffeTranslationBase.__init__(self)

    def extract_parameters(self, layer, graph):
        upsample_param = layer.upsample_param

        if upsample_param.upsample_mode == upsample_param.DENSE:
            # TODO: if this Op gets enabled this check might need to go to ir_to_dlc
            raise ValueError(code_to_message.get_error_message('ERROR_CAFFE_NO_SUPPORT_DENSE_UPSAMPLING')
                             (str(layer.name)))
        # If sparse mode is not enabled, extract params from pooling layer
        elif upsample_param.upsample_mode != upsample_param.SPARSE:
            pool_name = graph.naming_policy.get_input_names(layer, layer.bottom)[1]
            pool_dims = graph.get_buffer(pool_name).get_buf_dims()
            pool_op = graph.get_buffer(pool_name).producer.op

            return op_adapter.UpsampleIndexBasedOp(layer.name,
                                                   pool_size=pool_op.size_x,
                                                   pool_stride=pool_op.stridex,
                                                   pad=pool_op.padx,
                                                   output_height=pool_dims[2],
                                                   output_width=pool_dims[3])
        # Otherwise, use scale, upsample_h and upsample_w fields from upsample
        # param
        else:
            input_name = graph.naming_policy.get_input_names(layer, layer.bottom)[1]
            input_dim = graph.get_buffer(input_name).get_buf_dims()
            if upsample_param.upsample_h != 0:
                output_height = upsample_param.upsample_h
            else:
                output_height = input_dim[0]*upsample_param.scale

            if upsample_param.upsample_w != 0:
                output_width = upsample_param.upsample_w
            else:
                output_width = input_dim[1]*upsample_param.scale

            return op_adapter.UpsampleSparseOp(layer.name,
                                               pool_size=upsample_param.scale,
                                               pool_stride=upsample_param.scale,
                                               pad=0,
                                               output_height=output_height,
                                               output_width=output_width)


# CaffeTranslations.register_translation(CaffeUpsampleTranslation(),
#                                        converter_type('upsample', 'caffe'))

class CaffeArgMaxTranslation(CaffeTranslationBase):
    def __init__(self):
        CaffeTranslationBase.__init__(self)

    def extract_parameters(self, layer, graph):
        argmax_param = layer.argmax_param
        input_name = graph.naming_policy.get_input_names(layer, layer.bottom)[0]
        input_dim = graph.get_buffer(input_name).get_buf_dims()

        axis = argmax_param.axis
        if axis < 0:
            axis = len(input_dim) + axis

        return op_adapter.ArgMaxOp(layer.name, axis=axis, keep_dims=True)


CaffeTranslations.register_translation(CaffeArgMaxTranslation(),
                                       converter_type('argmax', 'caffe'),
                                       op_adapter.ArgMaxOp.TRANSLATION_KEY)
