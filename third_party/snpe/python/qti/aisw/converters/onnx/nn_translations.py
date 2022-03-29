# ==============================================================================
#
#  Copyright (c) 2018-2022 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from .onnx_translations import *
from qti.aisw.converters.common.converter_ir.axis_tracker import AxisTracker
from qti.aisw.converters.common.converter_ir.op_adapter import IRPaddingStrategies
from qti.aisw.converters.common.utils import translation_utils
from qti.aisw.converters.common.converter_ir.op_graph import QuantParams

# ------------------------------------------------------------------------------
#   AveragePool, MaxPool
# ------------------------------------------------------------------------------
class OnnxPoolTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('AveragePool', [1, 7, 10, 11])
        self.register_op_schema('MaxPool', [1, 8, 10, 11])

    def extract_parameters(self, src_op, graph):
        params = extract_attributes(src_op, attr_infos=
        [('ceil_mode', 'i', 0),  # new attribute since MaxPool-10 and AveragePool-10
         ('storage_order', 'i', 0),  # new attribute since MaxPool-8
         ('dilations', 'li', [1, 1]),  # new attribute since MaxPool-10
         ('count_include_pad', 'i', 0),  # new attribute since AveragePool-7
         ('strides', 'li', [1, 1]),
         # stride default to 1 if not present since MaxPool-11 and AveragePool-11
         ('kernel_shape', 'li'),
         ('pads', 'li'),
         ('auto_pad', 's', 'NOTSET')],
                                    schema=self.op_schema(op_type=src_op.op_type),
                                    validate=True)
        padding_size_strategy = extract_padding_mode(params.auto_pad, src_op.name, params.ceil_mode,
                                                     translation_utils.pads_righthanded(
                                                         params.pads))
        if str(src_op.op_type) == 'AveragePool':
            pool_type = op_adapter.PoolOp.Type.AVG
        else:
            pool_type = op_adapter.PoolOp.Type.MAX
            log_assert(len(src_op.output) == 1, code_to_message.get_error_message(
                "ERROR_MAXPOOL_OPTIONAL_INDICES_OUTPUT"))

        num_input_pads = len(params.pads) // 2  # number of pads per spatial axis
        log_assert(num_input_pads == 2,
                   code_to_message.get_error_message("ERROR_NUMBER_OF_PADS_UNSUPPORTED")
                   (src_op.name, num_input_pads))

        # Note: For pads assumes 2D input where dimensions are NCHW and HW are the only spatial dims
        return op_adapter.PoolOp(src_op.name,
                                 pool_type=pool_type,
                                 size_y=params.kernel_shape[0],
                                 size_x=params.kernel_shape[1],
                                 stride_y=params.strides[0],
                                 stride_x=params.strides[1],
                                 dilation_y=params.dilations[0],
                                 dilation_x=params.dilations[1],
                                 pady_before=params.pads[0],
                                 pady_after=params.pads[num_input_pads],
                                 padx_before=params.pads[1],
                                 padx_after=params.pads[1 + num_input_pads],
                                 padding_size_strategy=padding_size_strategy,
                                 pool_region_include_padding=params.count_include_pad)


OnnxTranslations.register_translation(OnnxPoolTranslation(),
                                      converter_type('AveragePool', 'onnx'),
                                      converter_type('MaxPool', 'onnx'),
                                      op_adapter.PoolOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   BatchNormalization
# ------------------------------------------------------------------------------
class OnnxBatchNormalizationTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('BatchNormalization', [1, 6, 7, 9]) \
            .register_method(self.validate_attribute_values)

    def extract_parameters(self, src_op, graph):
        params = extract_attributes(src_op, schema=self.op_schema(), validate=True)

        input_names = list(src_op.input)
        gamma, beta, mu, var = graph.weights.fetch(*input_names[1:])

        gamma_quant_enc = graph.get_overridden_encoding(input_names[1])
        beta_quant_enc = graph.get_overridden_encoding(input_names[2])

        if gamma_quant_enc:
            quantized_gamma = translation_utils.quantize_params(gamma, gamma_quant_enc[0])
            gamma = translation_utils.dequantize_params(quantized_gamma, gamma_quant_enc[0])
            # remove gamma encodings since already applied
            graph.remove_overridden_encoding(input_names[1])

        if beta_quant_enc:
            quantized_beta = translation_utils.quantize_params(beta, beta_quant_enc[0])
            beta = translation_utils.dequantize_params(quantized_beta, beta_quant_enc[0])
            # remove beta encodings since already applied
            graph.remove_overridden_encoding(input_names[2])

        # y = gamma*( (x-mu)/sqrt(var+epsilon) ) + beta
        # weights = gamma/sqrt(var+epsilon)
        weights = gamma / numpy.sqrt(var + params.epsilon)
        # bias = -mu*gamma/sqrt(var+epsilon) + beta = -mu*weights + beta
        bias = -mu * weights + beta
        spatial = bool(params.spatial if 'spatial' in params else 1)

        return op_adapter.BatchnormOp(src_op.name,
                                      weights,
                                      bias,
                                      across_spatial=spatial,
                                      epsilon=params.epsilon,
                                      gamma=gamma,
                                      beta=beta)

    def extract_input_names(self, src_op, graph):
        return [src_op.input[0]]

    @staticmethod
    def validate_attribute_values(self, attr_name, attr_value):
        # is_test is only supported in test mode, which is_test = 1
        if attr_name == 'is_test':
            log_assert(attr_value, code_to_message.get_error_message('ERROR_BATCHNORM_TEST_ONLY'))


OnnxTranslations.register_translation(OnnxBatchNormalizationTranslation(),
                                      converter_type('BatchNormalization', 'onnx'),
                                      op_adapter.BatchnormOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Conv
# ------------------------------------------------------------------------------
class OnnxConvTranslation(OnnxTranslationBase):

    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Conv', [1, 11])

    def extract_parameters(self, src_op, graph):
        input_names = self.extract_input_names(src_op, graph)
        params = extract_attributes(src_op, schema=self.op_schema(), validate=True)

        # Extract weights and biases and add them as ConstantOp inputs to the ConvolutionOp
        weights_constant_op = self.fetch_constant_op(input_names[1], graph, prunable=False, fail_if_dynamic=False)
        if weights_constant_op and not graph.has_buffer(input_names[1]):
            if params.kernel_shape:
                log_assert(tuple(params.kernel_shape) == weights_constant_op.tensor.shape[2:],
                           code_to_message.get_error_message("ERROR_KERNEL_SHAPE_DIFFERS_FROM_WEIGHTS"))
            log_debug(code_to_message.get_debugging_message('DEBUG_EXTRACT_WEIGHTS')(src_op.name,
                                                                                     weights_constant_op.tensor.shape))
            graph.add(weights_constant_op, [], [input_names[1]], axis_formats=[AxisTracker.AxisFormat.OIHW])
        elif graph.has_buffer(input_names[1]):
            # Properly set the axis format if the buffer already exists in the graph
            graph.get_buffer(input_names[1]).axis_format = AxisTracker.AxisFormat.OIHW

        bias_op_name = None
        if len(input_names) > 2:
            bias_op_name = input_names[2]
            bias_constant_op = self.fetch_constant_op(input_names[2], graph, prunable=False, fail_if_dynamic=False)
            if bias_constant_op and not graph.has_buffer(input_names[2]):
                log_debug(code_to_message.get_debugging_message('DEBUG_EXTRACT_BIAS')(src_op.name,
                                                                                      bias_constant_op.tensor.shape))
                graph.add(bias_constant_op, [], [input_names[2]], axis_formats=[AxisTracker.AxisFormat.ANY])

        # Extract the remaining attributes and calculate the padding size
        padding_mode = extract_padding_mode(params.auto_pad, src_op.name)
        num_input_pads = len(params.pads) // 2  # number of pads per spatial axis
        log_assert(num_input_pads == 2,
                   code_to_message.get_error_message("ERROR_NUMBER_OF_PADS_UNSUPPORTED")
                   (src_op.name, num_input_pads))

        # set the input padding size
        input_buf = graph.get_buffer(str(src_op.input[0]))
        input_shape = input_buf.shape
        weights_shape = graph.get_buffer(input_names[1]).shape
        params.pads = op_adapter.ConvolutionOp.calc_conv_padding_size(input_shape[2:],
                                                                      weights_shape[2:],
                                                                      params.dilations,
                                                                      params.strides,
                                                                      padding_mode,
                                                                      params.pads)
        # Note: For pads assumes 2D input where dimensions are NCHW and HW are the only spatial dims

        # Handle marking this Convolution as a DepthwiseConvolution
        num_input_channels = graph.src_axis_order.extract_spatial_dims(graph.get_buffer(input_names[0]).shape)[-1]
        num_output_channels = graph.src_axis_order.extract_conv_weights_dims(weights_shape)[-1]
        convolution_class = op_adapter.ConvolutionOp
        if params.group == num_input_channels and num_input_channels == num_output_channels:
            convolution_class = op_adapter.DepthwiseConvolutionOp

        return convolution_class(src_op.name,
                                 bias_op_name=bias_op_name,
                                 pady_before=params.pads[0],
                                 pady_after=params.pads[num_input_pads],
                                 padx_before=params.pads[1],
                                 padx_after=params.pads[1 + num_input_pads],
                                 padding_size_strategy=padding_mode,
                                 stridex=params.strides[1],
                                 stridey=params.strides[0],
                                 dilationx=params.dilations[1],
                                 dilationy=params.dilations[0],
                                 groups=params.group)


OnnxTranslations.register_translation(OnnxConvTranslation(),
                                      converter_type('Conv', 'onnx'),
                                      op_adapter.ConvolutionOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   ConvTranspose
# ------------------------------------------------------------------------------
class OnnxConvTransposeTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        schema_dict = self.register_op_schema('ConvTranspose', [1, 11])
        schema_dict.register_method(self.validate_attribute_values)
        schema_dict.replace_default_values(output_shape=[0, 0], output_padding=[0, 0],
                                           kernel_shape=[])

    def extract_parameters(self, src_op, graph):
        input_names = self.extract_input_names(src_op, graph)
        params = extract_attributes(src_op, schema=self.op_schema(), validate=True)

        # Extract weights and biases and add them as ConstantOp inputs to the DeconvolutionOp
        weights_constant_op = self.fetch_constant_op(input_names[1], graph, prunable=False, fail_if_dynamic=False)
        if weights_constant_op and not graph.has_buffer(input_names[1]):
            if params.kernel_shape:
                log_assert(tuple(params.kernel_shape) == weights_constant_op.tensor.shape[2:],
                           code_to_message.get_error_message("ERROR_KERNEL_SHAPE_DIFFERS_FROM_WEIGHTS"))
            log_debug(code_to_message.get_debugging_message('DEBUG_EXTRACT_WEIGHTS')(src_op.name,
                                                                                     weights_constant_op.tensor.shape))
            graph.add(weights_constant_op, [], [input_names[1]], axis_formats=[AxisTracker.AxisFormat.IOHW])
        elif graph.has_buffer(input_names[1]):
            # Properly set the axis format if the buffer already exists in the graph
            graph.get_buffer(input_names[1]).axis_format = AxisTracker.AxisFormat.IOHW

        bias_op_name = None
        if len(input_names) > 2:
            bias_op_name = input_names[2]
            bias_constant_op = self.fetch_constant_op(input_names[2], graph, prunable=False, fail_if_dynamic=False)
            if bias_constant_op and not graph.has_buffer(input_names[2]):
                log_debug(code_to_message.get_debugging_message('DEBUG_EXTRACT_BIAS')(src_op.name,
                                                                                      bias_constant_op.tensor.shape))
                graph.add(bias_constant_op, [], [input_names[2]], axis_formats=[AxisTracker.AxisFormat.ANY])

        # Extract the remaining attributes and calculate the padding size
        padding_mode = extract_padding_mode(params.auto_pad, src_op.name)
        num_input_pads = len(params.pads) // 2  # number of pads per spatial axis
        log_assert(num_input_pads == 2,
                   code_to_message.get_error_message("ERROR_NUMBER_OF_PADS_UNSUPPORTED")
                   (src_op.name, num_input_pads))

        # Extract and verify the output padding values
        output_padding = params.output_padding
        if any(output_padding):
            log_assert(output_padding < params.strides,
                       code_to_message.get_error_message(
                           "ERROR_DECONV_OUTPUT_PADDING_NOT_LESS_THAN_STRIDE")
                       (params.strides, output_padding))

        weights_shape = graph.get_buffer(input_names[1]).shape
        params.pads = op_adapter.DeconvolutionOp.calc_deconv_padding_size(weights_shape[2:],
                                                                          params.dilations,
                                                                          params.strides,
                                                                          padding_mode,
                                                                          params.pads,
                                                                          output_padding)

        # Note: For pads assumes 2D input where dimensions are NCHW and HW are the only spatial dims
        return op_adapter.DeconvolutionOp(src_op.name,
                                          bias_op_name=bias_op_name,
                                          stridex=params.strides[1],
                                          stridey=params.strides[0],
                                          pady_before=params.pads[0],
                                          pady_after=params.pads[num_input_pads],
                                          padx_before=params.pads[1],
                                          padx_after=params.pads[1 + num_input_pads],
                                          output_paddingx=params.output_padding[1],
                                          output_paddingy=params.output_padding[0],
                                          padding_size_strategy=padding_mode,
                                          output_height=params.output_shape[0],
                                          output_width=params.output_shape[1],
                                          groups=params.group)

    @staticmethod
    def validate_attribute_values(src_op, attr_name, attr_value):
        if attr_name == 'output_padding':
            log_assert(len(attr_value) <= 2,
                       code_to_message.get_error_message(
                           "ERROR_DECONV_OUTPUT_PADDING_LENGTH_UNSUPPORTED")
                       (len(attr_value)))


OnnxTranslations.register_translation(OnnxConvTransposeTranslation(),
                                      converter_type('ConvTranspose', 'onnx'),
                                      op_adapter.DeconvolutionOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   FC
# ------------------------------------------------------------------------------
class OnnxFCTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)

    def add_op(self, src_op, graph):
        node = super().add_op(src_op, graph)

        # Update weight/bias quantization info if it exists
        src_input_names = list(map(str, src_op.input))
        if graph.has_buffer(src_input_names[1]):
            graph.merge_quantization_params(graph.get_buffer(src_input_names[1]).producer.op.name,
                                            op.name, src_input_names[1], 'weights',
                                            encoding_type=QuantParams.PARAM_ENCODINGS)

        if len(src_input_names) > 2 and graph.has_buffer(src_input_names[2]):
                graph.merge_quantization_params(graph.get_buffer(src_input_names[2]).producer.op.name,
                                                op.name, src_input_names[2], 'bias',
                                                encoding_type=QuantParams.PARAM_ENCODINGS)

        return node

    def extract_parameters(self, src_op, graph):
        # Note: Schema is not used here since this op is not part of the Onnx spec.
        params = extract_attributes(src_op, attr_infos=
        [('axis', 'i', 1),
         ('axis_w', 'i', 1)])
        log_assert(params.axis == 1, code_to_message.get_error_message("ERROR_FC_AXIS_UNSUPPORTED"))
        log_assert(params.axis_w == 1,
                   code_to_message.get_error_message("ERROR_FC_AXIS_W_UNSUPPORTED"))

        input_names = graph.naming_policy.get_input_names(src_op, src_op.input)
        if len(input_names) == 2:
            weights = graph.weights.fetch(input_names[1])
            bias = np.zeros(weights.shape[1])
            bias_op_name = None
        else:
            weights, bias = graph.weights.fetch(*input_names[1:3])
            bias_op_name = input_names[2]
        return op_adapter.FullyConnectedOp(src_op.name, weights, bias, bias_op_name=bias_op_name)

    def extract_input_names(self, src_op, graph):
        return [str(src_op.input[0])]


OnnxTranslations.register_translation(OnnxFCTranslation(),
                                      converter_type('FC', 'onnx'),
                                      op_adapter.FullyConnectedOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   GlobalAveragePool, GlobalMaxPool
# ------------------------------------------------------------------------------
class OnnxGlobalPoolTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('GlobalAveragePool', [1])
        self.register_op_schema('GlobalMaxPool', [1])

    def extract_parameters(self, src_op, graph):
        input_buf = graph.get_buffer(str(src_op.input[0]))

        if str(src_op.op_type) == 'GlobalAveragePool':
            pool_type = op_adapter.PoolOp.Type.AVG
        else:
            pool_type = op_adapter.PoolOp.Type.MAX

        return op_adapter.PoolOp(src_op.name,
                                 pool_type=pool_type,
                                 size_x=input_buf.shape[3],
                                 size_y=input_buf.shape[2],
                                 stride_x=input_buf.shape[3],
                                 stride_y=input_buf.shape[2])


OnnxTranslations.register_translation(OnnxGlobalPoolTranslation(),
                                      converter_type('GlobalAveragePool', 'onnx'),
                                      converter_type('GlobalMaxPool', 'onnx'))


# ------------------------------------------------------------------------
#   InstanceNormalization
# ------------------------------------------------------------------------------
class OnnxInstanceNormalizationTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('InstanceNormalization', [1, 6])

    def extract_parameters(self, src_op, graph):
        params = extract_attributes(src_op, schema=self.op_schema(), validate=True)
        input_names = list(map(str, src_op.input))
        weights, bias = graph.weights.fetch(*input_names[1:])
        return op_adapter.BatchnormOp(src_op.name,
                                      weights,
                                      bias,
                                      epsilon=params.epsilon,
                                      compute_statistics=True,
                                      use_mu_sigma=True,
                                      across_spatial=True)

    def extract_input_names(self, src_op, graph):
        return [src_op.input[0]]
        # rest is handled by OnnxBatchNormalizationTranslation


OnnxTranslations.register_translation(OnnxInstanceNormalizationTranslation(),
                                      converter_type('InstanceNormalization', 'onnx'),
                                      'instancenorm')


# ------------------------------------------------------------------------------
#   MaxRoiPool
# ------------------------------------------------------------------------------
class OnnxMaxRoiPoolTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('MaxRoiPool', [1])

    def add_op(self, src_op, graph):
        ops = self.extract_parameters(src_op, graph)
        input_names = self.extract_input_names(src_op, graph)
        output_names = self.extract_output_names(src_op, graph)
        if len(ops) > 1:
            graph.add(ops[0], [], input_names[1])
        last_node = graph.add(ops[-1], input_names, output_names)

        # add src op info for roi pool operation
        self.add_src_op_info(last_node.op.name, src_op, graph)

        return last_node

    def extract_parameters(self, src_op, graph):
        params = extract_attributes(src_op, schema=self.op_schema(), validate=True)
        input_names = list(map(str, src_op.input))
        input_buf = graph.get_buffer(input_names[0])
        ops = []
        roi_name = input_names[1]

        if not graph.has_buffer(roi_name):
            roi_values = graph.weights.fetch(roi_name, prunable=False)
            roi_tensor = roi_values.astype(numpy.float32)
            roi_shape = roi_tensor.shape
            ops.append(op_adapter.ConstantOp(roi_name, roi_tensor))
        else:
            roi_shape = graph.get_buffer(roi_name).shape

        output_shape = [roi_shape[0],
                        input_buf.shape[1],
                        params.pooled_shape[0],
                        params.pooled_shape[1]]

        ops.append(op_adapter.RoiPoolingOp(src_op.name,
                                           output_shape,
                                           pooled_size_h=params.pooled_shape[0],
                                           pooled_size_w=params.pooled_shape[1],
                                           spatial_scale=params.spatial_scale))
        return ops

    def infer_output_shapes(self, op, input_shapes):
        return [op.output_shape]


OnnxTranslations.register_translation(OnnxMaxRoiPoolTranslation(),
                                      converter_type('MaxRoiPool', 'onnx'),
                                      op_adapter.RoiPoolingOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Prelu, LeakyRelu
# ------------------------------------------------------------------------------
# Also handles LeakyRelu as a bonus.
class OnnxPreluTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('PRelu', [1, 6, 7, 9])
        self.register_op_schema('LeakyRelu', [1, 6])

    def extract_parameters(self, src_op, graph):
        input_names = list(map(str, src_op.input))
        input_buf = graph.get_buffer(input_names[0])

        if str(src_op.op_type) == 'LeakyRelu':
            params = extract_attributes(src_op, schema=self.op_schema(op_type=src_op.op_type),
                                        validate=True)
            slope = np.array([params.alpha], dtype=numpy.float32)
        else:
            slope = graph.weights.fetch(input_names[1])
            if len(slope.shape) == 1:
                slope = np.ones(input_buf.shape[1], dtype=numpy.float32) * slope[0]
            else:
                rank_diff = len(slope.shape) - len(input_buf.shape)
                if rank_diff < 0:
                    # Prepending 1's to slope shape and then broadcasting to match input rank
                    slope_shape = [1] * abs(rank_diff) + list(slope.shape)
                    slope = numpy.broadcast_to(slope, slope_shape)
                slope = numpy.require(slope, dtype=numpy.float32)

        return op_adapter.PreluOp(src_op.name, coeff=slope)

    def extract_input_names(self, src_op, graph):
        return [src_op.input[0]]


OnnxTranslations.register_translation(OnnxPreluTranslation(),
                                      converter_type('Prelu', 'onnx'),
                                      converter_type('LeakyRelu', 'onnx'),
                                      op_adapter.PreluOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Lrn
# ------------------------------------------------------------------------------
class OnnxLrnTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('LRN', [1])

    def extract_parameters(self, src_op, graph):
        params = extract_attributes(src_op, schema=self.op_schema())
        return op_adapter.RNormOp(src_op.name,
                                  size=params.size,
                                  alpha=params.alpha / params.size,
                                  beta=params.beta,
                                  k=params.bias,
                                  across_channels=True)


OnnxTranslations.register_translation(OnnxLrnTranslation(),
                                      converter_type('LRN', 'onnx'),
                                      op_adapter.RNormOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   LpPool
# ------------------------------------------------------------------------------
class OnnxLpPoolTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('LpPool', [1, 2, 11])

    def extract_parameters(self, src_op, graph):
        params = extract_attributes(src_op, schema=self.op_schema(), validate=True)
        padding_size_strategy = extract_padding_mode(params.auto_pad, src_op.name)
        pool_type = op_adapter.PoolOp.Type.L2

        if translation_utils.pads_righthanded(params.pads):
            padding_size_strategy = IRPaddingStrategies.PADDING_SIZE_EXPLICIT_RIGHTHANDED
        num_input_pads = len(params.pads) // 2  # number of pads per spatial axis
        log_assert(num_input_pads == 2,
                   code_to_message.get_error_message("ERROR_NUMBER_OF_PADS_UNSUPPORTED")
                   (src_op.name, num_input_pads))
        # Note: For pads assumes 2D input where dimensions are NCHW and HW are the only spatial dims
        return op_adapter.L2PoolOp(src_op.name,
                                   pool_type=pool_type,
                                   size_y=params.kernel_shape[0],
                                   size_x=params.kernel_shape[1],
                                   stride_y=params.strides[0],
                                   stride_x=params.strides[1],
                                   pady_before=params.pads[0],
                                   pady_after=params.pads[num_input_pads],
                                   padx_before=params.pads[1],
                                   padx_after=params.pads[1 + num_input_pads],
                                   padding_size_strategy=padding_size_strategy,
                                   p=params.p)


OnnxTranslations.register_translation(OnnxLpPoolTranslation(),
                                      converter_type('LpPool', 'onnx'),
                                      op_adapter.L2PoolOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   NonMaxSuppression
# ------------------------------------------------------------------------------
class OnnxNonMaxSuppressionTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('NonMaxSuppression', [10, 11])

    def extract_parameters(self, src_op, graph):
        params = extract_attributes(src_op, schema=self.op_schema(), validate=True)
        if params.center_point_box:
            raise ValueError("NonMaxSuppression: Only support minmax representation (y1,x1,y2,x2)")

        # Require static parameters
        input_names = list(map(str, src_op.input))
        if (len(input_names) > 2 and
            any([graph.has_buffer(buf) and not self.fetch_constant_op(buf,graph) for buf in input_names[2:]])):
            raise ValueError(
                ('NonMaxSuppression: Only supports static parameters for '
                'max_output_boxes_per_class, iou_threshold, score_threshold')
            )
        max_output_boxes_per_class, iou_threshold, score_threshold = self.fetch_const_input(src_op, graph)
        return op_adapter.FakeNonMaxSuppressionOp(
            name=src_op.name,
            max_total_detections=max_output_boxes_per_class,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold
        )

    def fetch_const_input(self, src_op, graph):
        # per onnx spec
        max_output_boxes_per_class = 0
        iou_threshold = 0
        score_threshold = 0

        # Handle optional parameters
        input_names = list(map(str, src_op.input))
        if len(input_names) > 2 and input_names[2]:
            max_output_boxes_per_class = self.fetch_constant_op(input_names[2], graph).tensor
            max_output_boxes_per_class = int(max_output_boxes_per_class.item(0))
        if len(input_names) > 3 and input_names[3]:
            iou_threshold = self.fetch_constant_op(input_names[3], graph).tensor
            iou_threshold = iou_threshold.item(0)
        if len(input_names) > 4 and input_names[4]:
            score_threshold = self.fetch_constant_op(input_names[4], graph).tensor
            score_threshold = score_threshold.item(0)

        return max_output_boxes_per_class, iou_threshold, score_threshold

    def extract_input_names(self, src_op, graph):
        actual_input_name = []
        for inp in map(str, src_op.input):
            if not (inp and graph.weights.has(inp)):
                actual_input_name.append(inp)
        return actual_input_name


OnnxTranslations.register_translation(OnnxNonMaxSuppressionTranslation(),
                                      converter_type('NonMaxSuppression', 'onnx'),
                                      op_adapter.FakeNonMaxSuppressionOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   ROI Align
# ------------------------------------------------------------------------------
class OnnxRoiAlignTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('RoiAlign', [10])

    def add_op(self, src_op, graph):
        ops = self.extract_parameters(src_op, graph)
        input_names = self.extract_input_names(src_op, graph)
        output_names = self.extract_output_names(src_op, graph)

        if len(ops) > 1:
            batch_indices_node = graph.add(ops[0], [], input_names[2])
            # add src_op info for added constant op
            graph.add_src_op_info(batch_indices_node.op.name, None,
                                  batch_indices_node.output_names[0])

        last_node = graph.add(ops[-1], input_names, output_names)
        # add src op info for roi align operation
        self.add_src_op_info(last_node.op.name, src_op, graph)

        return last_node

    def extract_parameters(self, src_op, graph):
        params = extract_attributes(src_op, schema=self.op_schema())
        ops = []
        indices_name = str(src_op.input[2])
        # If the input is stored as weights we need to create a const node
        if not graph.has_buffer(indices_name):
            indices = graph.weights.fetch(indices_name, prunable=False)
            indices.tensor = indices.tensor.astype(numpy.int32)
            ops.append(op_adapter.ConstantOp(indices_name, indices, quantizable=False))
        else:
            indices_op = graph.get_buffer(indices_name).producer.op

            if op_adapter.ConstantOp.TRANSLATION_KEY is indices_op.type:
                indices_op.quantizable = False
                indices_op.tensor = indices_op.tensor.astype(numpy.int32)

        ops.append(op_adapter.RoiAlignOp(src_op.name,
                                         pooled_size_h=params.output_height,
                                         pooled_size_w=params.output_width,
                                         spatial_scale=params.spatial_scale,
                                         sampling_ratio=params.sampling_ratio,
                                         mode=params.mode))
        return ops


OnnxTranslations.register_translation(OnnxRoiAlignTranslation(),
                                      converter_type('RoiAlign', 'onnx'),
                                      op_adapter.RoiAlignOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   SpaceToDepth
# ------------------------------------------------------------------------------
class OnnxSpaceToDepthTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('SpaceToDepth', [1, 13])

    def extract_parameters(self, src_op, graph):
        params = extract_attributes(src_op, schema=self.op_schema(), validate=True)
        return op_adapter.SpaceToDepthOp(
            name=src_op.name,
            downscale_factor=params.blocksize,
        )

    def extract_input_names(self, src_op, graph):
        return [src_op.input[0]]


OnnxTranslations.register_translation(OnnxSpaceToDepthTranslation(),
                                      converter_type('SpaceToDepth', 'onnx'),
                                      op_adapter.SpaceToDepthOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   DepthToSpace
# ------------------------------------------------------------------------------
class OnnxDepthToSpaceTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('DepthToSpace', [1, 11, 13])

    def extract_parameters(self, src_op, graph):
        SUPPORTED_DEPTHTOSPACE_MODES = {'DCR': op_adapter.PixelShuffleOp.Mode.DCR,
                                        'CRD': op_adapter.PixelShuffleOp.Mode.CRD}
        params = extract_attributes(src_op, schema=self.op_schema(), validate=True)
        if not 'mode' in params:
            mode = 'DCR'
        elif params.mode not in SUPPORTED_DEPTHTOSPACE_MODES:
            raise ValueError("Unsupported depthtospace mode {}".format(params.mode))
        else:
            mode = params.mode
        return op_adapter.PixelShuffleOp(name=src_op.name,
                                         upscale_factor=params.blocksize,
                                         mode=SUPPORTED_DEPTHTOSPACE_MODES[mode])

    def extract_input_names(self, src_op, graph):
        return [src_op.input[0]]


OnnxTranslations.register_translation(OnnxDepthToSpaceTranslation(),
                                      converter_type('DepthToSpace', 'onnx'),
                                      op_adapter.PixelShuffleOp.TRANSLATION_KEY)
