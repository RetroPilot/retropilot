# ==============================================================================
#
#  Copyright (c) 2018-2022 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from .onnx_translations import *
from .util import *


# ------------------------------------------------------------------------------
#   Cast
# ------------------------------------------------------------------------------
class OnnxCastTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Cast', [1, 6, 9, 13])

    def extract_parameters(self, src_op, graph):
        log_warning(code_to_message.get_warning_message("WARNING_CAST_TYPE")(str(src_op.name)))
        params = extract_attributes(src_op,
                                    attr_infos=[('to', 'i', 0)],
                                    schema=self.op_schema(op_type=src_op.op_type),
                                    validate=True)
        cast_dtype = onnx_to_np_dtype.get(params.to).name
        from_type = graph.tensor_to_np_dtype.get(str(src_op.input[0]))
        if from_type:
            from_type = from_type.name
        # Raise error when cast type is not in list of supported types
        if cast_dtype is None:
            raise ValueError(code_to_message.get_error_message('ERROR_CAST_TYPE_UNSUPPORTED')
                             (str(src_op.name), cast_dtype.name))
        if not graph.has_buffer(str(src_op.input[0])):
            const_input = graph.weights.fetch(str(src_op.input[0]))
            graph.add(op_adapter.ConstantOp(str(src_op.input[0]), const_input), [], str(src_op.input[0]))
            # make constant input unconsumed to avoid prune error
            graph.weights.weight_map[str(src_op.input[0])].consumed = False
            return op_adapter.CastOp(str(src_op.output[0]), to_type=cast_dtype)
        if not from_type:
            return op_adapter.CastOp(str(src_op.name), to_type=cast_dtype)
        return op_adapter.CastOp(str(src_op.name), from_type=from_type, to_type=cast_dtype)


OnnxTranslations.register_translation(OnnxCastTranslation(),
                                      converter_type('Cast', 'onnx'))


# ------------------------------------------------------------------------------
#   ChannelShuffle
# ------------------------------------------------------------------------------
class OnnxChannelShuffleTranslation(OnnxTranslationBase):
    def extract_parameters(self, src_op, graph):
        # Note: Schema is not used here since this is not a valid Onnx Op
        params = extract_attributes(src_op,
                                    ('groups', 'i'))
        return op_adapter.ChannelShuffleOp(src_op.name, groups=params.groups)


OnnxTranslations.register_translation(OnnxChannelShuffleTranslation(),
                                      converter_type('Channel_Shuffle', 'onnx'),
                                      op_adapter.ChannelShuffleOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Clip
# ------------------------------------------------------------------------------
class OnnxClipTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Clip', [1, 6, 11, 12])

    def extract_parameters(self, src_op, graph):
        params = extract_attributes(src_op, schema=self.op_schema())

        min_name = str(src_op.input[1]) if len(src_op.input) > 1 else ''
        min_op = self.fetch_constant_op(min_name, graph)
        if min_op is None:
            min_val = params.min if 'min' in params else numpy.finfo(numpy.float32).min
        else:
            min_val = min_op.tensor.item(0)

        max_name = str(src_op.input[2]) if len(src_op.input) > 2 else ''
        max_op = self.fetch_constant_op(max_name, graph)
        if max_op is None:
            max_val = params.max if 'max' in params else numpy.finfo(numpy.float32).max
        else:
            max_val = max_op.tensor.item(0)

        return op_adapter.NeuronOp(src_op.name,
                                   op_adapter.NeuronOp.Type.RELU_MIN_MAX,
                                   min_clamp=min_val,
                                   max_clamp=max_val)

    def extract_input_names(self, src_op, graph):
        return [list(src_op.input)[0]]


OnnxTranslations.register_translation(OnnxClipTranslation(), converter_type('Clip', 'onnx'))


# ------------------------------------------------------------------------------
#   Concat
# ------------------------------------------------------------------------------
class OnnxConcatTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Concat', [1, 4, 11])

    def add_op(self, src_op, graph):
        op = self.extract_parameters(src_op, graph)
        input_names = self.extract_input_names(src_op, graph)
        output_names = self.extract_output_names(src_op, graph)

        if op.type == op_adapter.ConcatOp.TRANSLATION_KEY:
            for input_name in input_names:
                if not graph.has_buffer(input_name) and graph.weights.has(input_name):
                    const_op = self.fetch_constant_op(input_name, graph, prunable=False)
                    const_node = graph.add(const_op, [], input_name)
                    graph.add_src_op_info(input_name, None, const_node.output_names[0])

        if op.type == op_adapter.ConstantOp.TRANSLATION_KEY:
            self.add_src_op_info(op.name, src_op, graph)
            return graph.add(op, [], output_names)

        self.add_src_op_info(op.name, src_op, graph)
        return graph.add(op, input_names, output_names)

    def extract_parameters(self, src_op, graph):
        params = extract_attributes(src_op, schema=self.op_schema())

        # static concatenation used for reshaping shape tensors
        if graph.weights.has_all(src_op.input):
            data = [graph.weights.fetch(input_name) for input_name in src_op.input]
            concat_data = numpy.concatenate(data, params.axis)
            graph.weights.insert(str(src_op.output[0]), concat_data)
            return op_adapter.StaticOp(src_op.name)

        # handle single input concats
        if len(src_op.input) == 1:
            if graph.weights.has_all(src_op.input):
                graph.weights.insert(str(src_op.output[0]), graph.weights.fetch(src_op.input[0]))
                return op_adapter.StaticOp(src_op.name)
            return op_adapter.NoopOp(src_op.name)

        # handle all constant input to concat
        input_names = list(map(str, src_op.input))
        const_input_ops = []
        for input_name in input_names:
            const_input_op = self.fetch_constant_op(input_name, graph, prunable=False, fail_if_dynamic=False)
            if const_input_op is not None:
                const_input_ops.append(const_input_op)
        if len(const_input_ops) == len(input_names):
            log_debug1(code_to_message.get_debugging_message("DEBUG_STATIC_OP")(src_op.name))
            data = []
            for const_input_op in const_input_ops:
                data.append(const_input_op.tensor)
            concat_data = numpy.concatenate(data, params.axis)
            graph.weights.insert(str(src_op.output[0]), concat_data)
            return op_adapter.ConstantOp(str(src_op.output[0]), concat_data)

        return op_adapter.ConcatOp(src_op.name, params.axis)

    def extract_input_names(self, src_op, graph):
        # If this was translated to a static op don't return input names
        if graph.weights.has_all(src_op.input):
            return []
        else:
            return list(map(str, src_op.input))

    def extract_output_names(self, src_op, graph):
        # If this was translated to a static op don't return output names
        if graph.weights.has_all(src_op.input):
            return []
        else:
            return [str(src_op.output[0])]


OnnxTranslations.register_translation(OnnxConcatTranslation(),
                                      converter_type('Concat', 'onnx'),
                                      op_adapter.ConcatOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Constant
# ------------------------------------------------------------------------------
class OnnxConstantTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Constant', [1, 9])

    def extract_parameters(self, src_op, graph):
        params = extract_attributes(src_op, schema=self.op_schema())
        # ONNX return numpy "array scalar" for ONNX scalar.
        # the problem is, "array scalar" has shape attribute as an empty tuple.
        # which may break backends.
        # So we reshape "array scalar" to exactly an array with shape (1, )
        was_scalar = False
        if not params.value.shape:
            params.value = params.value.reshape(1)
            was_scalar = True

        graph.weights.insert(src_op.output[0], params.value, was_scalar)
        # Constant op is a special case... the output name is the real name
        return op_adapter.ConstantOp(src_op.output[0], params.value)

    def infer_output_shapes(self, op, input_shapes):
        return [list(op.tensor.shape)]


OnnxTranslations.register_translation(OnnxConstantTranslation(),
                                      converter_type('Constant', 'onnx'),
                                      op_adapter.ConstantOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   ConstantOfShape
# ------------------------------------------------------------------------------
class OnnxConstantOfShapeTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('ConstantOfShape', [9])

    def extract_parameters(self, src_op, graph):
        params = extract_attributes(src_op, schema=self.op_schema())
        input_names = list(map(str, src_op.input))
        was_scalar = False

        # Only support when input is static
        const_op = self.fetch_constant_op(input_names[0], graph, prunable=False, fail_if_not_found=True)

        log_debug1(code_to_message.get_debugging_message("DEBUG_STATIC_OP")(src_op.name))
        shape = const_op.tensor.astype(numpy.int32)
        tensor_dtype = downcast_dtype_64bit_to_32bit(src_op.name, params.value.dtype)
        data = numpy.full(shape, params.value[0], dtype=tensor_dtype)
        if not data.shape:
            data = data.reshape(1)
            was_scalar = True
        graph.weights.insert(src_op.output[0], data, was_scalar)
        return op_adapter.ConstantOp(src_op.output[0], data)

    def extract_input_names(self, src_op, graph):
        return []


OnnxTranslations.register_translation(OnnxConstantOfShapeTranslation(),
                                      converter_type('ConstantOfShape', 'onnx'))

# ------------------------------------------------------------------------------
#   DequantizeLinear
# ------------------------------------------------------------------------------
class OnnxDequantizeLinearTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('DequantizeLinear', [10, 13])

    def add_op(self, src_op, graph):
        op, enc = self.extract_parameters(src_op, graph)
        input_names = self.extract_input_names(src_op, graph)
        output_names = self.extract_output_names(src_op, graph)
        node = graph.add(op, input_names, output_names)
        if op.type == op_adapter.DequantizeOp.TRANSLATION_KEY:
            graph.add_quantization_params(node.op.name,
                                          output_encodings=enc)
        else:
            graph.add_quantization_params(node.op.name,
                                          param_encodings=enc)

        self.add_src_op_info(node.op.name, src_op, graph)
        return node

    def extract_parameters(self, src_op, graph):
        params = extract_attributes(src_op, schema=self.op_schema())

        # Three inputs data, scale(s), and zero point(s)
        inputs = src_op.input
        outputs = src_op.output

        log_assert(len(inputs) >= 2,
                   code_to_message.get_error_message("ERROR_QUANTIZE_INVALID_INPUTS")(len(inputs)))

        # Retrieve the scales
        log_assert(graph.weights.has(inputs[1]),
                   code_to_message.get_error_message("ERROR_STATIC_QUANTIZE_PARAM")("scale", src_op.name, src_op.op_type))
        scale = np.array(graph.weights.fetch(inputs[1])).astype(np.float32)

        # Check if zero point provided, otherwise use default of 0
        zp = np.array([0]).astype(np.uint8)
        if len(inputs) > 2:
            log_assert(graph.weights.has(inputs[2]),
                       code_to_message.get_error_message("ERROR_STATIC_QUANTIZE_PARAM")("zero point", src_op.name, src_op.op_type))
            zp = graph.weights.fetch(inputs[2], dtype=graph.weights.type(inputs[2]))

        log_assert(len(scale) == 1 and len(zp) == 1,
                   "Per-channel quantization currently unsupported, len of scale and zero point must be 1")

        # TODO Finish support of per-channel quant for get_encoding
        if 'axis' in params:
            axis = params.axis
            if axis < 0:
                axis += len(input.shape)

        output_name = str(outputs[0])
        enc = get_encoding(output_name, scale, zp)

        if graph.weights.has(inputs[0]):
            # It's quantized parameters, quantize and store
            w = graph.weights.fetch(inputs[0])
            w = (w - zp) * scale
            graph.weights.insert(output_name, w)
            return op_adapter.ConstantOp(output_name, w), enc

        stripped_enc = {k:enc[k] for k in enc if k != 'name'}
        return op_adapter.DequantizeOp(src_op.name, **stripped_enc), enc

    def extract_input_names(self, src_op, graph):
        # If this was translated to a static op don't return output names
        if graph.weights.consumed(str(src_op.input[0])):
            return []
        else:
            return [str(src_op.input[0])]

    def infer_output_shapes(self, op, input_shapes):
        print('dequant input_shape:',input_shapes, ' for output shape: ',[input_shapes[0]])
        return [input_shapes[0]]

    def extract_output_names(self, src_op, graph):
        return [str(src_op.output[0])]


OnnxTranslations.register_translation(OnnxDequantizeLinearTranslation(),
                                      converter_type('DequantizeLinear', 'onnx'),
                                      op_adapter.DequantizeOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Expand
# ------------------------------------------------------------------------------
class OnnxExpandTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Expand', [8, 13])

    def extract_parameters(self, src_op, graph):
        src_input_names = list(map(str, src_op.input))

        shape_constant_op = self.fetch_constant_op(src_input_names[1], graph, dtype=np.int32, fail_if_not_found=True)
        log_debug1(code_to_message.get_debugging_message("DEBUG_STATIC_OP")(src_input_names[1]))

        output_shape = [int(dim_size) for dim_size in shape_constant_op.tensor]

        return op_adapter.ExpandOp(src_op.name,
                                   output_shape)

    def extract_input_names(self, src_op, graph):
        return [str(src_op.input[0])]


OnnxTranslations.register_translation(OnnxExpandTranslation(), converter_type('Expand', 'onnx'))


# ------------------------------------------------------------------------------
#   Initializer
# ------------------------------------------------------------------------------
class OnnxInitializerTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)

    def extract_parameters(self, initializer, graph):
        params = extract_initializer_tensor(initializer)

        # ONNX return numpy "array scalar" for ONNX scalar.
        # the problem is, "array scalar" has shape attribute as an empty tuple.
        # which may break backends.
        # So we reshape "array scalar" to exactly an array with shape (1, )
        if not params.shape:
            params = params.reshape(1)

        # Constant op is a special case... the output name is the real name
        return op_adapter.ConstantOp(initializer.name, params)

    def extract_input_names(self, src_op, graph):
        return []

    def extract_output_names(self, src_op, graph):
        return [src_op.name]

    def infer_output_shapes(self, op, input_shapes):
        return [list(op.tensor.shape)]


OnnxTranslations.register_translation(OnnxInitializerTranslation(),
                                      converter_type('Initializer', 'onnx'))


# ------------------------------------------------------------------------------
#   Flatten
# ------------------------------------------------------------------------------
class OnnxFlattenTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Flatten', [1, 9, 11])

    def extract_parameters(self, src_op, graph):
        params = extract_attributes(src_op, schema=self.op_schema())
        axis = params.axis

        # SNPE uses weights at construction time, not dynamically. Ensure they
        # are preprocessed statically.
        input_name = str(src_op.input[0])
        if graph.weights.has(input_name):
            # static flatten of weight parameters
            output_name = str(src_op.output[0])
            w = graph.weights.fetch(input_name)
            pre_axes = w.shape[:axis]
            post_axes = w.shape[axis:]
            output_shape = [product(pre_axes), product(post_axes)]
            w = numpy.reshape(w, output_shape)
            graph.weights.insert(output_name, w)
            log_info(code_to_message.get_progress_message("INFO_STATIC_RESHAPE")(input_name, output_name, output_shape))
            return op_adapter.StaticOp(src_op.name)

        input_buf = graph.get_buffer(str(src_op.input[0]))
        input_shape = input_buf.shape

        pre_axes = input_shape[:axis]
        post_axes = input_shape[axis:]
        output_shape = [product(pre_axes), product(post_axes)]

        # Otherwise this is a dynamic flatten so add the flatten/reshape op
        return op_adapter.ReshapeOp(src_op.name, output_shape)

    def extract_input_names(self, src_op, graph):
        return [name for name in list(map(str, src_op.input)) if not graph.weights.consumed(name)]

    def extract_output_names(self, src_op, graph):
        # If this was translated to a static op don't return output names
        if graph.weights.consumed(str(src_op.input[0])):
            return []
        else:
            return [str(src_op.output[0])]


OnnxTranslations.register_translation(OnnxFlattenTranslation(), converter_type('Flatten', 'onnx'))


# ------------------------------------------------------------------------------
#   Gather
# ------------------------------------------------------------------------------
class OnnxGatherTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Gather', [1, 11, 13])

    def add_op(self, src_op, graph, **kwargs):
        input_op, translated_ops = self.extract_parameters(src_op, graph)
        input_names = self.extract_input_names(src_op, graph)
        output_names = self.extract_output_names(src_op, graph)
        # input op should only be 1 (either data or indices) or None.
        # if input_op = None => either both data and indices are constant or both are dynamic
        if input_op:
            node = graph.add(input_op, [], input_op.name)
            graph.add_src_op_info(node.op.name, None, node.output_names[0])

        if translated_ops[0].type == op_adapter.ConstantOp.TRANSLATION_KEY:
            # when gather op is represented as a constant op
            last_node = graph.add(translated_ops[0], [], output_names)
            self.add_src_op_info(last_node.op.name, src_op, graph)
        else:
            # when gather is represented as gather or gather + reshape
            if len(translated_ops) == 2:
                gather_output_names = [output_names[0] + '_pre_reshape']
            else:
                gather_output_names = [output_names[0]]

            last_node = graph.add(translated_ops[0], input_names, gather_output_names)
            graph.add_src_op_info(last_node.op.name, None, gather_output_names[0])

            if len(translated_ops) == 2:
                last_node = graph.add(translated_ops[1], gather_output_names, output_names)
                graph.add_src_op_info(last_node.op.name, None, last_node.output_names[0])

        return last_node

    def extract_parameters(self, src_op, graph):
        params = extract_attributes(src_op, schema=self.op_schema())
        input_op = None
        translated_ops = []
        input_data_name = str(src_op.input[0])
        indices_name = str(src_op.input[1])
        axis = params.axis

        input_names = list(map(str, src_op.input))
        const_input_ops = []
        const_input_op = self.fetch_constant_op(input_data_name, graph, dtype=None, fail_if_dynamic=False)
        if const_input_op is not None:
            const_input_ops.append(const_input_op)
        const_input_op = self.fetch_constant_op(indices_name, graph, dtype=numpy.int32, prunable=False,
                                                fail_if_dynamic=False)
        if const_input_op is not None:
            const_input_ops.append(const_input_op)

        # If both input and indices are static then interpret gather and return const op
        if len(const_input_ops) == len(input_names):
            log_debug1(code_to_message.get_debugging_message("DEBUG_STATIC_OP")(src_op.name))
            input_data = const_input_ops[0].tensor
            indices = const_input_ops[1].tensor
            was_scalar = graph.weights.was_scalar(indices_name)
            gather_data = numpy.take(input_data, indices, axis=axis)
            graph.weights.insert(str(src_op.output[0]), gather_data, was_scalar=was_scalar)
            translated_ops.append(op_adapter.ConstantOp(src_op.output[0], gather_data))
            return input_op, translated_ops

        # If only input is stored as weights then create a corresponding const op
        if not graph.has_buffer(input_data_name) and graph.weights.has(input_data_name):
            input_data = graph.weights.fetch(input_data_name, prunable=False)
            input_op = op_adapter.ConstantOp(input_data_name, input_data)

        # If only indices is stored as weights then create a corresponding const op
        if not graph.has_buffer(indices_name) and graph.weights.has(indices_name):
            indices = graph.weights.fetch(indices_name, prunable=False).astype(numpy.int32)
            input_op = op_adapter.ConstantOp(indices_name, indices, quantizable=False)
        else:
            indices_op = graph.get_buffer(indices_name).producer.op
            if op_adapter.ConstantOp.TRANSLATION_KEY is indices_op.type:
                indices_op.quantizable = False
                indices_op.tensor = indices_op.tensor.astype(numpy.int32)

        translated_ops.append(op_adapter.GatherOp(src_op.name, axis=axis))

        if graph.weights.has(indices_name) and graph.weights.was_scalar(indices_name):
            input_buf_shape = graph.get_buffer(src_op.input[0]).shape
            output_shape = input_buf_shape[:axis] + input_buf_shape[axis+1:]
            reshape_op_name = src_op.name
            if src_op.name:
                reshape_op_name = 'Reshape_post_' + src_op.name
            translated_ops.append(op_adapter.ReshapeOp(reshape_op_name,
                                                       output_shape=output_shape))

        return input_op, translated_ops


OnnxTranslations.register_translation(OnnxGatherTranslation(),
                                      converter_type('Gather', 'onnx'),
                                      op_adapter.GatherOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   OneHot
# ------------------------------------------------------------------------------
class OnnxOneHotTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('OneHot', [9, 11])

    def add_op(self, src_op, graph):
        ops = self.extract_parameters(src_op, graph)
        input_names = self.extract_input_names(src_op, graph)
        output_names = self.extract_output_names(src_op, graph)

        if len(ops) == 2:
            onehot_output_name = [output_names[0] + '_pre_reshape']
        else:
            onehot_output_name = [output_names[0]]

        last_node = graph.add(ops[0], input_names, onehot_output_name)
        graph.add_src_op_info(last_node.op.name, input_names[0], onehot_output_name[0])

        if len(ops) == 2:
            last_node = graph.add(ops[1], onehot_output_name, output_names)
            graph.add_src_op_info(last_node.op.name, onehot_output_name[0], last_node.output_names[0])

        return last_node

    def extract_parameters(self, src_op, graph):
        params = extract_attributes(src_op, schema=self.op_schema())
        input_names = list(map(str, src_op.input))
        ops = []

        depth_const_op = self.fetch_constant_op(input_names[1], graph)
        depth = depth_const_op.tensor[0]
        if depth < 0:
            raise ValueError(code_to_message.get_error_message("ERROR_ONEHOT_NEG_DEPTH")(depth))

        values_const_op = self.fetch_constant_op(input_names[2], graph)
        values = values_const_op.tensor

        ops.append(op_adapter.OneHotOp(src_op.name, depth=depth, on_value=values[1], off_value=values[0], axis=params.axis))

        # if indices input was a scalar then reshape one_hot output
        if graph.weights.has(input_names[0]) and graph.weights.was_scalar(input_names[0]):
            output_shape = [depth]
            reshape_op_name = src_op.name
            if src_op.name:
                reshape_op_name = 'Reshape_post_' + src_op.name
            ops.append(op_adapter.ReshapeOp(reshape_op_name,
                                            output_shape=output_shape))

        return ops

    def extract_input_names(self, src_op, graph):
        # Filter depth and values from the input
        return [str(src_op.input[0])]


OnnxTranslations.register_translation(OnnxOneHotTranslation(),
                                      converter_type('OneHot', 'onnx'))


# ------------------------------------------------------------------------------
#   Pad
# ------------------------------------------------------------------------------
class OnnxPadTranslation(OnnxTranslationBase):
    class OnnxPadMode:
        CONSTANT = 'constant'
        REFLECT = 'reflect'
        EDGE =  'edge'
    supported_modes = {OnnxPadMode.CONSTANT : op_adapter.PadOp.Mode.CONSTANT,
                       OnnxPadMode.REFLECT : op_adapter.PadOp.Mode.REFLECT,
                       OnnxPadMode.EDGE : op_adapter.PadOp.Mode.EDGE}

    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Pad', [1, 2, 11, 13])\
            .register_method(self.validate_attribute_values)

    def extract_parameters(self, src_op, graph):
        params = extract_attributes(src_op, schema=self.op_schema(), validate=True)
        pads_name = str(src_op.input[1]) if len(src_op.input) > 1 else ''
        const_name = str(src_op.input[2]) if len(src_op.input) > 2 else ''
        pads = None

        if pads_name:
            pads_op = self.fetch_constant_op(pads_name, graph, dtype=numpy.int32)
            if pads_op is not None:
                pads = pads_op.tensor
        elif 'pads' in params:
            pads = params.pads
        elif 'paddings' in params:
            pads = params.paddings

        if pads is None:
            raise ValueError("Failed to retrieve pads value on {} source op {}".format(src_op.op_type,
                                                                                       src_op.name))

        # Pads/paddings need to be translated from r1_begin, r2_begin...r1_end, r2_end, ...
        # to pairs (r1_begin, r1_end), (r2_begin, r2_end)...
        input_buf = graph.get_buffer(str(src_op.input[0]))
        rank = len(input_buf.shape)
        log_assert(rank == len(pads) / 2,
                   "Rank of input tensor: {} must equal (# pads/2): {}"
                   .format(rank, int(len(pads) / 2)))

        pad_pairs = []
        for index in range(rank):
            pad_pairs.append([pads[index], pads[index + rank]])
        pad_pairs = np.asarray(pad_pairs, dtype=np.dtype('int32'))

        constant_value = 0
        if const_name:
            const_op = self.fetch_constant_op(const_name, graph, dtype=numpy.int32)
            if const_op is not None:
                constant_value = const_op.tensor[0]
        elif 'value' in params:
            constant_value = params.value

        return op_adapter.PadOp(src_op.name,
                                mode=self.supported_modes[params.mode],
                                pads=pad_pairs,
                                constant_value=constant_value)

    def extract_input_names(self, src_op, graph):
        # Filter if there are any parameters like 'pads' in inputs
        # For example, 'pads' are already handled in extract_parameters
        return [str(src_op.input[0])]

    @staticmethod
    def validate_attribute_values(src_op, attr_name, attr_value):
        if attr_name == 'mode':
            src_op_mode = attr_value
            if src_op_mode not in OnnxPadTranslation.supported_modes:
                raise ValueError(code_to_message.get_error_message("ERROR_PAD_UNSUPPORTED_MODE")(src_op_mode))


OnnxTranslations.register_translation(OnnxPadTranslation(),
                                      converter_type('Pad', 'onnx'),
                                      op_adapter.PadOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   QuantizeLinear
# ------------------------------------------------------------------------------
class OnnxQuantizeLinearTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('QuantizeLinear', [10, 13])

    def add_op(self, src_op, graph):
        op, enc = self.extract_parameters(src_op, graph)
        input_names = self.extract_input_names(src_op, graph)
        output_names = self.extract_output_names(src_op, graph)
        node = graph.add(op, input_names, output_names)
        if op.type == op_adapter.QuantizeOp.TRANSLATION_KEY:
            graph.add_quantization_params(node.op.name,
                                          output_encodings=enc)
        else:
            graph.add_quantization_params(node.op.name,
                                          param_encodings=enc)

        self.add_src_op_info(node.op.name, src_op, graph)
        return node

    def extract_parameters(self, src_op, graph):
        params = extract_attributes(src_op, schema=self.op_schema())

        # Three inputs data, scale(s), and zero point(s)
        inputs = src_op.input
        outputs = src_op.output

        log_assert(len(inputs) >= 2,
                   code_to_message.get_error_message("ERROR_QUANTIZE_INVALID_INPUTS")(len(inputs)))

        # Retrieve the scales
        log_assert(graph.weights.has(inputs[1]),
                   code_to_message.get_error_message("ERROR_STATIC_QUANTIZE_PARAM")("scale", src_op.name, src_op.op_type))
        scale = graph.weights.fetch(inputs[1], prunable=False)

        # Check if zero point provided, otherwise use default of 0
        zp = np.array([0]).astype(np.uint8)
        if len(inputs) > 2:
            log_assert(graph.weights.has(inputs[2]),
                       code_to_message.get_error_message("ERROR_STATIC_QUANTIZE_PARAM")("zero point", src_op.name, src_op.op_type))
            zp = graph.weights.fetch(inputs[2], dtype=graph.weights.type(inputs[2]), prunable=False)

        # TODO Finish support of per-channel quant for get_encoding
        if 'axis' in params:
            axis = params.axis
            if axis < 0:
                axis += len(input.shape)

        output_name = str(outputs[0])
        enc = get_encoding(output_name, scale, zp)

        if graph.weights.has(inputs[0]):
            # It's quantized parameters, quantize and store
            w = graph.weights.fetch(inputs[0])
            w = numpy.clip((np.rint(w/scale) + zp), np.iinfo(zp.dtype).min, np.iinfo(zp.dtype).max)
            graph.weights.insert(output_name, w)
            return op_adapter.ConstantOp(output_name, w), enc

        stripped_enc = {k:enc[k] for k in enc if k != 'name'}
        return op_adapter.QuantizeOp(src_op.name, **stripped_enc), enc

    def extract_input_names(self, src_op, graph):
        # If this was translated to a static op don't return names
        if graph.weights.consumed(str(src_op.input[0])):
            return []
        else:
            return [str(src_op.input[0])]

    def infer_output_shapes(self, op, input_shapes):
        return [input_shapes[0]]

    def extract_output_names(self, src_op, graph):
        return [str(src_op.output[0])]


OnnxTranslations.register_translation(OnnxQuantizeLinearTranslation(),
                                      converter_type('QuantizeLinear', 'onnx'),
                                      op_adapter.QuantizeOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Range
# ------------------------------------------------------------------------------
class OnnxRangeTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Range', [11])

    def extract_parameters(self, src_op, graph):
        input_names = list(map(str, src_op.input))
        const_inputs = []

        # Only support when all inputs are static
        for input_name in input_names:
            const_op = self.fetch_constant_op(input_name, graph, prunable=False)
            if const_op is not None:
                const_inputs.append(const_op.tensor)

        log_assert(len(const_inputs) == 3,
                   code_to_message.get_error_message("ERROR_RANGE_INVALID_INPUTS")(len(const_inputs)))

        start = const_inputs[0].item(0)
        limit = const_inputs[1].item(0)
        delta = const_inputs[2].item(0)

        range_output = numpy.arange(start, limit, delta)
        graph.weights.insert(str(src_op.output[0]), range_output)
        return op_adapter.ConstantOp(src_op.output[0], range_output)

    def extract_input_names(self, src_op, graph):
        return []


OnnxTranslations.register_translation(OnnxRangeTranslation(),
                                      converter_type('Range', 'onnx'))


# ------------------------------------------------------------------------------
#   Reshape
# ------------------------------------------------------------------------------
class OnnxReshapeTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Reshape', [1, 5])

    def extract_parameters(self, src_op, graph):
        # There are two main versions of ONNX Reshape
        #    1. The old reshape, where shape is provided as an attribute
        #    2. The new reshape, where the shape is provided as a second input
        #
        # SNPE and the converter support two versions of Reshape:
        #    1. Dynamic reshaping with a statically provided output shape
        #    2. Static reshaping, performed at conversion time
        #
        # SNPE can't support the 2nd ONNX Reshape expclicitly, however we can
        # calculate the shape ahead of time and statically set in in the SNPE layer.
        # This will prevent the network from being resizable. In addition, if a
        # 'Shape' layer provided the shape it will have been saved as static,
        # eg weight data, in the converter and all ops operating on that data will
        # become static ops and will be pruned during the final conversion.
        shape = []
        if len(src_op.input) > 1:
            shape_input = str(src_op.input[1])
            # only support constant for second input, if dynamic fetch will fail.
            shape = self.fetch_constant_op(shape_input, graph, fail_if_not_found=True,
                                           dtype=numpy.int32).tensor.tolist()
        else:
            params = extract_attributes(src_op, schema=self.op_schema())
            if 'shape' in params:
                shape = params.shape

        log_assert(len(shape) != 0, "Unable to retrieve reshape shape")

        input_name = str(src_op.input[0])
        const_input_op = self.fetch_constant_op(input_name, graph, fail_if_dynamic=False)
        if const_input_op is not None:
            # static reshape of weight parameters
            output_name = str(src_op.output[0])
            log_info(code_to_message.get_progress_message("INFO_STATIC_RESHAPE")(input_name, output_name, shape))

            w = const_input_op.tensor
            w = numpy.reshape(w, shape)
            graph.weights.insert(output_name, w)
            return op_adapter.StaticOp(src_op.name)
        else:
            # dynamic reshape of activations
            input_buf = graph.get_buffer(input_name)
            input_shape = input_buf.shape

            remainder_size = product(input_shape)
            remainder_index = -1
            output_shape = []
            for i, s in enumerate(shape):
                if s == -1:
                    remainder_index = i
                    output_shape.append(0)
                elif s == 0:
                    remainder_size /= input_shape[i]
                    output_shape.append(input_shape[i])
                else:
                    remainder_size /= s
                    output_shape.append(s)
            if remainder_index >= 0:
                output_shape[remainder_index] = int(remainder_size)

            return op_adapter.ReshapeOp(src_op.name, output_shape)

    def extract_input_names(self, src_op, graph):
        input_name = str(src_op.input[0])
        if graph.weights.consumed(input_name):
            return []
        else:
            return [input_name]

    def extract_output_names(self, src_op, graph):
        # If this was translated to a static op don't return output names
        if graph.weights.consumed(str(src_op.input[0])):
            return []
        else:
            return [str(src_op.output[0])]


OnnxTranslations.register_translation(OnnxReshapeTranslation(),
                                      converter_type('Reshape', 'onnx'),
                                      op_adapter.ReshapeOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Resize
# ------------------------------------------------------------------------------
class OnnxResizeTranslation(OnnxTranslationBase):
    SUPPORTED_RESIZE_MODES = ['nearest', 'linear', 'bilinear']
    SUPPORTED_COORD_TRANSFORM_MODES = ['asymmetric', 'align_corners', 'half_pixel', 'tf_half_pixel_for_nn',
                                       'pytorch_half_pixel']

    def __init__(self):
        OnnxTranslationBase.__init__(self)
        schema_dict = self.register_op_schema('Resize', [10, 11])
        schema_dict.replace_default_values(mode='nearest')
        schema_dict.register_method(self.validate_attribute_values)

    def extract_parameters(self, src_op, graph):
        resize_schema = self.op_schema()
        params = extract_attributes(src_op, attr_infos=[('mode', 's', 'nearest'),
                                                        ('coordinate_transformation_mode', 's', 'asymmetric')],
                                    schema=resize_schema, validate=True)
        transformation_mode = params.coordinate_transformation_mode
        input_buf = graph.get_buffer(str(src_op.input[0]))
        if input_buf.rank() != 4:
            raise ValueError(code_to_message.get_error_message("ERROR_RESIZE_INPUT_DIMS")(input_buf.shape))

        align_corners = False
        half_pixel_centers = False
        # Determine coordinate transform mode include in Resize version >10
        if transformation_mode == "align_corners":
            align_corners = True
        elif (params.mode == "linear" and transformation_mode == "half_pixel") or \
                (params.mode == "nearest" and transformation_mode == "tf_half_pixel_for_nn"):
            half_pixel_centers = True
        elif (not params.mode == "linear" and transformation_mode == "half_pixel") or \
                (not params.mode == "nearest" and transformation_mode == "tf_half_pixel_for_nn"):
            raise ValueError(
                code_to_message.get_error_message("ERROR_RESIZE_INVALID_COORDINATE_TRANSFORMATION_MODE_MIX")
                (params.mode, transformation_mode))

        sizes = None
        input_shape = input_buf.shape
        input_height = input_shape[2]
        input_width = input_shape[3]
        if len(src_op.input) > 1:
            scales = None
            scales_input = str(src_op.input[2]) if len(src_op.input) > 2 else str(src_op.input[1])
            if graph.weights.has(scales_input):
                scales = graph.weights.fetch(scales_input).astype(numpy.float32).tolist()
            elif graph.has_buffer(scales_input):
                scales = graph.get_buffer(scales_input).shape
            if not scales and len(src_op.input) > 2:
                size_name = str(src_op.input[-1])
                if graph.weights.has(size_name):
                    # per onnx spec, size is int64. But int32 may be enough.
                    sizes = graph.weights.fetch(size_name).astype(numpy.int32).tolist()
                else:
                    sizes = graph.get_buffer(size_name).shape
                # Opset 11 has 4th parameter as output sizes,
                # here we are calculating scales from output sizes
                # per onnx spec, scales is float.
                scales = list(map(float, sizes))
                scales[-1] = (scales[-1]-1) / (input_width-1) if align_corners else (scales[-1] / input_width)
                scales[-2] = (scales[-2]-1) / (input_height-1) if align_corners else (scales[-2] / input_height)
        else:
            # deprecated. Added for Upsample version 7 and below
            scales = extract_attributes(src_op, attr_infos=[('scales', 'lf')], schema=resize_schema, validate=True).scales

        scale_height = scales[2]
        scale_width = scales[3]

        # Generate output shape using output_dims. Note: doing round() first since casting to int gets the floor
        # which was causing output dim error in models.
        output_height = sizes[2] if sizes else int(round(input_height * scale_height))
        output_width = sizes[3] if sizes else int(round(input_width * scale_width))
        output_shape = [input_shape[0], input_shape[1], output_height, output_width]

        # per onnx spec pytorch_half_pixel is same as half_pixel when length resized for dimension > 1.
        if transformation_mode == "pytorch_half_pixel":
            if output_height > 1 and output_width > 1:
                half_pixel_centers = True
            else:
                raise ValueError(
                    code_to_message.get_error_message("ERROR_RESIZE_PYTORCH_HALF_PIXEL_UNSUPPORTED_VALUE")
                    (transformation_mode, output_height, output_width))

        return op_adapter.ResizeOp(src_op.name,
                                   output_shape,
                                   resize_mode=params.mode,
                                   scale_height=scale_height,
                                   scale_width=scale_width,
                                   align_corners=align_corners,
                                   half_pixel_centers=half_pixel_centers)

    @classmethod
    def validate_attribute_values(cls, src_op, attr_name, attr_value):
        if attr_name == 'mode':
            src_op_mode = attr_value
            if src_op_mode not in cls.SUPPORTED_RESIZE_MODES:
                raise ValueError(code_to_message.get_error_message("ERROR_RESIZE_UNSUPPORTED_MODE")
                                 (src_op_mode,  cls.SUPPORTED_RESIZE_MODES))
        elif attr_name == 'scales':
            scales = attr_value
            if scales[0] != 1 or scales[1] != 1:
                log_warning(code_to_message.get_warning_message("WARNING_RESIZE"))
        elif attr_name == 'coordinate_transformation_mode':
            src_op_mode = attr_value
            if src_op_mode not in cls.SUPPORTED_COORD_TRANSFORM_MODES:
                raise ValueError(
                    code_to_message.get_error_message("ERROR_RESIZE_UNSUPPORTED_COORDINATE_TRANSFORMATION_MODE")
                    (src_op_mode, cls.SUPPORTED_COORD_TRANSFORM_MODES))

    def extract_input_names(self, src_op, graph):
        if len(src_op.input) > 2:
            return [str(src_op.input[0])]
        else:
            return [name for name in list(map(str, src_op.input)) if not graph.weights.consumed(name)]

    def infer_output_shapes(self, op, input_shapes):
        log_debug(code_to_message.get_debugging_message("DEBUG_INFERRED_SHAPE")(op.name, op.output_shape))
        return [op.output_shape]


OnnxTranslations.register_translation(OnnxResizeTranslation(),
                                      converter_type('Resize', 'onnx'),
                                      op_adapter.ResizeOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   ScatterND
# ------------------------------------------------------------------------------
class OnnxScatterNDTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('ScatterND', [11, 13])
        self.reduction_types = {"none": op_adapter.ScatterNDOp.ReductionTypes.REDUCTION_NONE,
                                "add": op_adapter.ScatterNDOp.ReductionTypes.REDUCTION_ADD,
                                "mul": op_adapter.ScatterNDOp.ReductionTypes.REDUCTION_MUL}
        self.is_static = False

    def add_op(self, src_op, graph, **kwargs):
        input_ops, translated_ops = self.extract_parameters(src_op, graph)
        input_names = self.extract_input_names(src_op, graph)
        output_names = self.extract_output_names(src_op, graph)
        # input op should only be 1 or 2 or None.
        # if input_op = None => all inputs are constant or all are dynamic
        # When input ops are None, scatter ND is a constant op
        if input_ops:
            for input_op in input_ops:
                node = graph.add(input_op, [], input_op.name)
                graph.add_src_op_info(node.op.name, None, node.output_names[0])

        if translated_ops[0].type == op_adapter.ConstantOp.TRANSLATION_KEY:
            # when scatter_nd op is represented as a constant op i.e input ops is None
            last_node = graph.add(translated_ops[0], [], output_names)
            self.add_src_op_info(last_node.op.name, src_op, graph)
        else:
            # when scatter nd op has one or more dynamic inputs
            last_node = graph.add(translated_ops[0], input_names, output_names)
            self.add_src_op_info(last_node.op.name, src_op, graph)

        return last_node

    def _perform_static_scatter_nd(self, input_data: np.ndarray, indices: np.ndarray,
                                   updates: np.ndarray, reduction: str = "none"):
        if reduction not in self.reduction_types:
            raise TypeError("Cannot perform static scatter nd. Expected reduction type"
                            " to be one of: {}, instead got: {}".format(list(self.reduction_types.keys()),
                                                                        reduction))

        # Perform only reduction = none since that is supported in 11,13
        # No need to reject other reduction values since the attribute only exists in 16
        # TODO: Check for other reduction types once new version is added

        static_scatter_data = np.copy(input_data)
        update_idx = indices.shape[:-1]
        for idx in np.ndindex(update_idx):
            static_scatter_data[indices[idx]] = updates

        return static_scatter_data

    def extract_parameters(self, src_op, graph):
        # Note there are no attributes to extract for versions 11, 13

        input_ops = []
        translated_ops = []
        input_data_name = str(src_op.input[0])
        indices_name = str(src_op.input[1])
        updates_name = str(src_op.input[2])

        input_names = list(map(str, src_op.input))
        const_input_ops = []

        # Create prunable const ops for all inputs if set
        const_input_op = self.fetch_constant_op(input_data_name, graph, dtype=None, fail_if_dynamic=False)
        if const_input_op is not None:
            const_input_ops.append(const_input_op)

        const_indices_op = self.fetch_constant_op(indices_name, graph, dtype=numpy.int32,
                                                  quantizable=False,
                                                  fail_if_dynamic=False)
        if const_indices_op is not None:
            const_input_ops.append(const_indices_op)

        const_updates_op = self.fetch_constant_op(updates_name, graph, dtype=numpy.int32,
                                                  quantizable=False,
                                                  fail_if_dynamic=False)
        if const_updates_op is not None:
            const_input_ops.append(const_updates_op)

        # If all inputs are static, then perform static scatter and return
        if len(const_input_ops) == len(input_names):
            log_debug1(code_to_message.get_debugging_message("DEBUG_STATIC_OP")(src_op.name))
            input_data = const_input_ops[0].tensor
            indices = const_input_ops[1].tensor
            updates = const_input_ops[2].tensor
            scatter_data = self._perform_static_scatter_nd(input_data, indices, updates)
            graph.weights.insert(str(src_op.output[0]), scatter_data)
            translated_ops.append(op_adapter.ConstantOp(src_op.output[0], scatter_data))
            self.is_static = True
            return input_ops, translated_ops

        # If input is stored as weights then create a corresponding const op
        input_data, indices = None, None
        if not graph.has_buffer(input_data_name) and graph.weights.has(input_data_name):
            input_data = graph.weights.fetch(input_data_name, prunable=False)
            input_ops.append(op_adapter.ConstantOp(input_data_name, input_data))

        # If indices is stored as weights then create a corresponding const op
        if not graph.has_buffer(indices_name) and graph.weights.has(indices_name):
            indices = graph.weights.fetch(indices_name, prunable=False).astype(numpy.int32)
            indices_op = op_adapter.ConstantOp(indices_name, indices, quantizable=False)
            input_ops.append(indices_op)
        else:
            indices_op = graph.get_buffer(indices_name).producer.op
            if op_adapter.ConstantOp.TRANSLATION_KEY is indices_op.type:
                indices_op.quantizable = False
                indices = indices_op.tensor = indices_op.tensor.astype(numpy.int32)

        if indices is not None:
            if np.any(indices < 0):
                if input_data is None:
                    raise ValueError("Cannot resolve constant negative indices for ScatterND indices: "
                                     "{} if input data is not static".format(indices_name))
                else:
                    with np.nditer(indices, op_flags=['readwrite']) as it:
                        for index in it:
                            if index < 0:
                                index += len(input_data.shape)

            # check to ensure unique indices if reduction is none
            # TODO: Change when reduction param is available from newer Onnx versions
            u_idx, counts = np.unique(indices, return_counts=True)
            if np.any(u_idx[counts > 1]):
                raise ValueError("Duplicate scatter indices are not supported when reduction is set to None")

        # If updates is stored as weights then create a corresponding const op
        if not graph.has_buffer(updates_name) and graph.weights.has(updates_name):
            updates = graph.weights.fetch(updates_name, prunable=False)
            input_ops.append(op_adapter.ConstantOp(updates_name, updates))

        translated_ops.append(op_adapter.ScatterNDOp(src_op.name))

        return input_ops, translated_ops

    def extract_input_names(self, src_op, graph):
        if self.is_static:
            return []
        else:
            return super().extract_input_names(src_op, graph)


OnnxTranslations.register_translation(OnnxScatterNDTranslation(),
                                      converter_type('ScatterND', 'onnx'),
                                      op_adapter.ScatterNDOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Shape
# ------------------------------------------------------------------------------
class OnnxShapeTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Shape', [1])

    def extract_parameters(self, src_op, graph):
        log_debug1(code_to_message.get_debugging_message("DEBUG_STATIC_OP")(src_op.name))
        input_name = str(src_op.input[0])

        constant_op = self.fetch_constant_op(input_name, graph, dtype=np.int32, fail_if_not_found=True,
                                             fail_if_dynamic=False)
        if constant_op:
            shape = constant_op.tensor.shape
        elif graph.has_buffer(input_name):
            shape = graph.get_buffer(input_name).shape

        output_name = str(src_op.output[0])
        graph.weights.insert(output_name, numpy.asarray(shape, dtype=numpy.int32))
        shape_param = numpy.asarray(shape, dtype=numpy.int32)
        return op_adapter.ConstantOp(output_name, shape_param)

    def extract_input_names(self, src_op, graph):
        return []

    def extract_output_names(self, src_op, graph):
        return [str(src_op.output[0])]


OnnxTranslations.register_translation(OnnxShapeTranslation(),
                                      converter_type('Shape', 'onnx'))


# ------------------------------------------------------------------------------
#   Slice
# ------------------------------------------------------------------------------
class OnnxSliceTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Slice', [1, 10, 11])

    def extract_parameters(self, src_op, graph):
        input_names = [str(x) for x in src_op.input]
        params = extract_attributes(src_op, schema=self.op_schema())
        const_inputs_params = self._fetch_inputs_as_params(src_op, graph, params)
        params.update(const_inputs_params)

        log_assert(len(params.starts) == len(params.axes),
                   "Node {}: expected same number of starts as axes",
                   src_op.name)
        log_assert(len(params.ends) == len(params.axes),
                   "Node {}: expected same number of ends as axes",
                   src_op.name)
        log_assert(all(params.steps),
                   "Node {}: expected all steps != 0",
                   src_op.name)

        # Static slicing used for shape tensors
        if graph.weights.has(input_names[0]):
            data = graph.weights.fetch(input_names[0])
            for i in range(len(params.axes)):
                start, end = self.get_indices(params.starts[i],
                                              params.ends[i],
                                              params.steps[i],
                                              data.shape[params.axes[i]])
                data = data.take(indices=list(range(start, end, params.steps[i])), axis=params.axes[i])
            output_name = str(src_op.output[0])
            graph.weights.insert(output_name, data)
            return op_adapter.StaticOp(src_op.name)

        input_buf = graph.get_buffer(input_names[0])
        rank = input_buf.rank()
        begin = [0] * rank
        end = [0] * rank
        strides = [0] * rank

        for index, axis in enumerate(params.axes):
            begin[axis], end[axis] = self.get_indices(params.starts[index],
                                                      params.ends[index],
                                                      params.steps[index],
                                                      input_buf.shape[axis])
            strides[axis] = params.steps[index]
            # add check to find if there is empty data case or out-of-range indices
            log_assert(begin[axis] < end[axis] if strides[axis] > 0 else begin[axis] > end[axis],
                       "Node {}: invalid stride for begin {} and end {} at axis {}",
                       src_op.name, begin[axis], end[axis], axis)
            log_assert(0 <= begin[axis] < input_buf.shape[axis],
                       "Node {}: begin:{} at axis {} is out-of-range",
                       src_op.name, begin[axis])
            log_assert(-1 <= end[axis] <= input_buf.shape[axis],
                       "Node {}: end:{} at axis {} is out-of-range",
                       src_op.name, end[axis], axis)

        for i, stride in enumerate(strides):
            if not stride:
                begin[i], end[i] = 0, input_buf.shape[i]
                strides[i] = 1

        return op_adapter.StridedSliceOp(name=src_op.name,
                                         begin=begin,
                                         end=end,
                                         strides=strides)

    def _fetch_inputs_as_params(self, src_op, graph, params):
        # opset 10,11 need handle 5 inputs, fetch constant input and add it to params
        # NOTE: Runtime does not allow dynamic input for starts, ends, axes and steps
        # input indices: data: 0, starts: 1, ends: 2, axes: 3(optional), steps: 4(optional)
        input_names = [str(x) for x in src_op.input]
        rank = 0
        if graph.has_buffer(input_names[0]):
            input_buf = graph.get_buffer(input_names[0])
            rank = input_buf.rank()
        elif graph.weights.has(input_names[0]):
            rank = len(graph.weights.fetch(input_names[0], prunable=False).shape)
        keys = ['data', 'starts', 'ends', 'axes', 'steps']
        if len(src_op.input) >= 3:
            for key, name in zip(keys[1:], input_names[1:]):
                # ONNX may use empty string as a placeholder
                # So add an and-condition to further check it.
                if name and graph.weights.has(name):
                    # handle INT_MAX and INT_MIN case in ONNX spec, require fetch int64 directly
                    # case: INT64_MAX -> cast to float and cast to int64 -> INT64_MIN
                    # case: INT64_MAX -> cast to int32 -> -1
                    params[key] = graph.weights.fetch(name, dtype=numpy.int64).tolist()
                    if key == 'axes':
                        for axis in params['axes']:
                            log_assert(-rank <= axis <= rank-1,
                            "expected axis range from {} to {}, but got {}",
                            -rank, rank-1, axis)
                elif graph.has_buffer(name):
                    raise ValueError(code_to_message.get_error_message('ERROR_SLICE_DYNAMIC_INPUTS')(name))

        if 'axes' not in params or len(params.axes) == 0:
            params['axes'] = list(range(len(params.starts)))
        if 'steps' not in params or len(params.steps) == 0:
            params['steps'] = list([1] * len(params.starts))

        return params

    def extract_input_names(self, src_op, graph):
        # If this was translated to a static op don't return input names
        if graph.weights.has(str(src_op.input[0])):
            return []
        else:
            # Handle constant and initializer cases, do not add them to input_names to avoid prune error.
            actual_input_names = []
            for input_name in map(str, src_op.input):
                if input_name in graph.buffers and not graph.weights.has(input_name):
                    actual_input_names.append(input_name)
            return actual_input_names

    def extract_output_names(self, src_op, graph):
        # If this was translated to a static op don't return output names
        if graph.weights.has(str(src_op.input[0])):
            return []
        else:
            return list(map(str, src_op.output))

    @staticmethod
    def get_indices(start, end, step, dim):
        # Negative values mean wrap around, like in python
        if start < 0:
            start = int(start % dim)
        # higher than the size, however, means stop at the end - 1.
        start = min(start, dim-1)

        if step < 0:
            end = max(end, -(dim+1))
        else:
            end = min(end, dim)

        if end < 0:
            end = end + dim

        return start, end


OnnxTranslations.register_translation(OnnxSliceTranslation(),
                                      converter_type('Slice', 'onnx'),
                                      op_adapter.StridedSliceOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Split
# ------------------------------------------------------------------------------
class OnnxSplitTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Split', [1, 2, 11])\
            .replace_default_values(split=[])

    def extract_parameters(self, src_op, graph):
        params = extract_attributes(src_op, schema=self.op_schema())
        slice_points = []
        if len(params.split) > 0:
            split_index = params.split[0]
            for size in params.split[1:]:
                slice_points.append(int(split_index))
                split_index += size

        if not len(slice_points) and len(src_op.output) == 1:
            return op_adapter.NoopOp(src_op.name)

        return op_adapter.SliceOp(src_op.name,
                                  axis=params.axis,
                                  slice_points=slice_points)


OnnxTranslations.register_translation(OnnxSplitTranslation(),
                                      converter_type('Split', 'onnx'),
                                      op_adapter.SliceOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Squeeze
# ------------------------------------------------------------------------------
class OnnxSqueezeTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Squeeze', [1, 11, 13])

    def extract_parameters(self, src_op, graph):
        input_name = str(src_op.input[0])
        params = extract_attributes(src_op, schema=self.op_schema())

        axes = []
        if len(src_op.input) > 1:
            axes_input = str(src_op.input[1])
            # only support constant for second input, if dynamic fetch will fail.
            axes = self.fetch_constant_op(axes_input, graph, dtype=numpy.int32).tensor.tolist()
        elif 'axes' in params:
            axes = params.axes

        const_input_op = self.fetch_constant_op(input_name, graph, fail_if_dynamic=False)
        if const_input_op is not None:
            # static squeeze of weight parameters
            output_name = str(src_op.output[0])
            w = graph.weights.fetch(input_name)
            if not len(axes):
                axes = [i for i, s in enumerate(w.shape) if s == 1]
            output_shape = [s for i, s in enumerate(w.shape) if i not in axes]

            log_info(code_to_message.get_progress_message("INFO_STATIC_RESHAPE")(input_name, output_name, output_shape))
            w = numpy.reshape(w, output_shape)
            graph.weights.insert(output_name, w)
            return op_adapter.StaticOp(src_op.name)

        # input is not a static parameter
        input_buf = graph.get_buffer(input_name)
        input_shape = input_buf.shape[:]

        if not len(axes):
            axes = [i for i, s in enumerate(input_shape) if s == 1]

        if not all(x < len(input_shape) for x in axes):
            raise ValueError(code_to_message.get_error_message("ERROR_SQUEEZE_DIM_GREATER_THAN_RANK")(axes,
                                                                                                      len(input_shape)))
        if not all((input_shape[x] == 1) for x in axes):
            raise ValueError(code_to_message.get_error_message("ERROR_SQUEEZE_DIMS_EQUAL_ONE")(axes,
                                                                                               input_shape))

        output_shape = [s for i, s in enumerate(input_shape) if i not in axes]

        return op_adapter.ReshapeOp(src_op.name, output_shape)

    def extract_input_names(self, src_op, graph):
        return [name for name in list(map(str, src_op.input)) if not graph.weights.consumed(name)]

    def extract_output_names(self, src_op, graph):
        # If this was translated to a static op don't return output names
        if graph.weights.consumed(str(src_op.input[0])):
            return []
        else:
            return [str(src_op.output[0])]


OnnxTranslations.register_translation(OnnxSqueezeTranslation(), converter_type('Squeeze', 'onnx'))


# ------------------------------------------------------------------------------
#   Tile
# ------------------------------------------------------------------------------
class OnnxTileTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Tile', [1, 6])

    def extract_parameters(self, src_op, graph):
        input_names = list(map(str, src_op.input))
        input_rank = len(graph.get_buffer(src_op.input[0]).shape)

        if len(input_names) == 3:
            # Represents Tile-1
            tiles = graph.weights.fetch(src_op.input[1])
            axis = graph.weights.fetch(src_op.input[2])
            repeats = [1] * input_rank
            repeats[axis] = tiles
        elif len(input_names) == 2:
            # Represents Tile-6
            repeats = graph.weights.fetch(src_op.input[1]).astype(numpy.uint32).tolist()
        else:
            raise ValueError("Only versions {} of {} node {} are supported".format(self.get_supported_version(),
                                                                                   src_op.op_type, src_op.name))

        return op_adapter.TileOp(src_op.name, multiples=repeats)

    def extract_input_names(self, src_op, graph):
        return [src_op.input[0]]


OnnxTranslations.register_translation(OnnxTileTranslation(),
                                      converter_type('Tile', 'onnx'),
                                      op_adapter.TileOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Transpose
# ------------------------------------------------------------------------------
class OnnxTransposeTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Transpose', [1, 13])

    def extract_parameters(self, src_op, graph):
        params = extract_attributes(src_op, schema=self.op_schema())
        input_name = str(src_op.input[0])
        const_op = self.fetch_constant_op(input_name, graph, fail_if_dynamic=False, fail_if_not_found=True)
        if const_op is not None:
            # static permute of weight parameters
            output_name = str(src_op.output[0])
            w = const_op.tensor
            log_debug1('static input: {} to: {}'.format(input_name, w.shape))
            log_debug1('transpose shape to : {}'.format(params.perm))
            w = numpy.transpose(w, params.perm)
            graph.weights.insert(output_name, w)
            log_info(code_to_message.get_progress_message("INFO_STATIC_RESHAPE")(input_name, output_name, w.shape))

            return op_adapter.StaticOp(src_op.name)

        log_debug1('input: {} to: {}'.format(input_name,graph.get_buffer(input_name).shape))
        log_debug1('transpose shape to : {}'.format(params.perm))
        return op_adapter.PermuteOp(src_op.name, params.perm)

    def extract_input_names(self, src_op, graph):
        return [name for name in list(map(str, src_op.input)) if not graph.weights.consumed(name)]

    def extract_output_names(self, src_op, graph):
        # If this was translated to a static op don't return output names
        # return [name for name in list(map(str, src_op.input)) if not graph.weights.consumed(name)]
        if graph.weights.consumed(str(src_op.input[0])):
            return []
        else:
            return [str(src_op.output[0])]


OnnxTranslations.register_translation(OnnxTransposeTranslation(),
                                      converter_type('Transpose', 'onnx'),
                                      op_adapter.PermuteOp.TRANSLATION_KEY)


# -----------------------------------------------------------------------------
#   Unsqueeze
# ------------------------------------------------------------------------------
class OnnxUnsqueezeTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Unsqueeze', [1, 11, 13])

    def extract_parameters(self, src_op, graph):
        params = extract_attributes(src_op, schema=self.op_schema(), validate=True)
        axes = []
        if len(src_op.input) > 1:
            axes_input = str(src_op.input[1])
            # only support constant for second input, if dynamic fetch will fail.
            axes = self.fetch_constant_op(axes_input, graph, dtype=numpy.int32).tensor.tolist()
        elif 'axes' in params:
            axes = params.axes

        if len(set(axes)) != len(axes):
            raise ValueError(code_to_message.get_error_message("ERROR_UNSQUEEZE_DUPLICATE_DIMS")(axes))

        input_name = str(src_op.input[0])

        const_input_op = self.fetch_constant_op(input_name, graph, fail_if_dynamic=False)
        if const_input_op is not None:
            log_debug1(code_to_message.get_debugging_message("DEBUG_STATIC_OP")(src_op.name))
            w = const_input_op.tensor
            shape = [] if graph.weights.was_scalar(input_name) else w.shape
            output_shape = self._get_unsqueezed_shape(shape, axes)
            w = numpy.reshape(w, output_shape)
            output_name = str(src_op.output[0])
            graph.weights.insert(output_name, w)
            return op_adapter.StaticOp(src_op.name)

        # input is not a static parameter
        input_buf = graph.get_buffer(input_name)
        input_shape = input_buf.shape[:]

        new_rank = len(input_shape) + len(axes)
        if not all(x < new_rank for x in axes):
            raise ValueError(code_to_message.get_error_message("ERROR_UNSQUEEZE_DIMS_GREATER_THAN_RANK")(axes,
                                                                                                         new_rank))
        output_shape = self._get_unsqueezed_shape(input_shape, axes)

        # Otherwise this is a dynamic unsqueeze so add the unsqueeze/reshape op
        return op_adapter.ReshapeOp(src_op.name, output_shape)

    def extract_input_names(self, src_op, graph):
        return [name for name in list(map(str, src_op.input)) if not graph.weights.consumed(name)]

    def extract_output_names(self, src_op, graph):
        # If this was translated to a static op don't return output names
        if graph.weights.consumed(str(src_op.input[0])):
            return []
        else:
            return [str(src_op.output[0])]

    @staticmethod
    def _get_unsqueezed_shape(org_shape, axes):
        output_shape = list(org_shape)
        for i in sorted(axes):
            # support negative axes since Unsqueeze-11
            if i < 0:
                i += len(output_shape)+1
            output_shape.insert(i, 1)
        return output_shape


OnnxTranslations.register_translation(OnnxUnsqueezeTranslation(), converter_type('Unsqueeze', 'onnx'))


# ------------------------------------------------------------------------------
#   Upsample
# ------------------------------------------------------------------------------
class OnnxUpsampleTranslation(OnnxResizeTranslation):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Upsample', [1, 7, 9])\
            .register_method(self.validate_attribute_values)


OnnxTranslations.register_translation(OnnxUpsampleTranslation(),
                                      converter_type('Upsample', 'onnx'),
                                      op_adapter.Upsample.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Where
# ------------------------------------------------------------------------------
class OnnxWhereTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Where', [9])
        self.input_names = None

    def extract_parameters(self, src_op, graph):
        self.input_names = list(map(str, src_op.input))
        if graph.weights.has(self.input_names[0]):
            condition_op = self.fetch_constant_op(self.input_names[0], graph, prunable=False)
            condition_tensor = condition_op.tensor.flatten()
            # Check Noop cases: Either all True yielding a pass-through of input1 or all False
            # yielding a pass-through of input2
            if all(condition for condition in condition_tensor):
                self.input_names = [self.input_names[1]]
                return op_adapter.NoopOp(src_op.name)
            elif all(not condition for condition in condition_tensor):
                self.input_names = [self.input_names[2]]
                return op_adapter.NoopOp(src_op.name)

        return op_adapter.ElementwiseSelectOp(name=src_op.name)

    def extract_input_names(self, src_op, graph):
        return self.input_names


OnnxTranslations.register_translation(OnnxWhereTranslation(),
                                      converter_type('Where', 'onnx'))
