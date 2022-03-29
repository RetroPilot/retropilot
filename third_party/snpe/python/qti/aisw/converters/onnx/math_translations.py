# ==============================================================================
#
#  Copyright (c) 2018-2022 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
from .onnx_translations import *

import distutils
from distutils import version


# ------------------------------------------------------------------------------
#   Abs
# ------------------------------------------------------------------------------
class OnnxAbsTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Abs', [1, 6])

    def extract_parameters(self, src_op, graph):
        return op_adapter.ElementwiseUnaryAbsOp(str(src_op.name))


OnnxTranslations.register_translation(OnnxAbsTranslation(),
                                      converter_type('Abs', 'onnx'),
                                      op_adapter.ElementwiseUnaryAbsOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Add
# ------------------------------------------------------------------------------
class OnnxAddTranslation(ElementwiseBinaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Add', [1, 6, 7])

    def extract_parameters(self, src_op, graph):
        op = op_adapter.ElementwiseSumOp(str(src_op.name))
        input_names = self.extract_input_names(src_op, graph)
        const_input_data = []
        for input_name in input_names:
            const_input_op = self.fetch_constant_op(input_name, graph, prunable=False, fail_if_dynamic=False)
            if const_input_op is not None:
                const_input_data.append(const_input_op.tensor)
        if len(const_input_data) == len(input_names):
            log_debug1(code_to_message.get_debugging_message("DEBUG_STATIC_OP")(src_op.name))
            data = numpy.add(*const_input_data)
            graph.weights.insert(str(src_op.output[0]), data)
            return op_adapter.ConstantOp(str(src_op.output[0]), data)
        else:
            return op


OnnxTranslations.register_translation(OnnxAddTranslation(),
                                      converter_type('Add', 'onnx'),
                                      op_adapter.ElementwiseSumOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   And
# ------------------------------------------------------------------------------
class OnnxAndTranslation(ElementwiseBinaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('And', [1, 7])

    def extract_parameters(self, src_op, graph):
        return op_adapter.ElementwiseAndOp(str(src_op.name))


OnnxTranslations.register_translation(OnnxAndTranslation(),
                                      converter_type('And', 'onnx'),
                                      op_adapter.ElementwiseAndOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   ArgMax
# ------------------------------------------------------------------------------
class OnnxArgMaxTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('ArgMax', [1, 11])

    def extract_parameters(self, src_op, graph):
        # these parameters belong to ArgMax
        params = extract_attributes(src_op, schema=self.op_schema(), validate=True)

        return op_adapter.ArgMaxOp(str(src_op.name),
                                   axis=params.axis,
                                   keep_dims=params.keepdims)


OnnxTranslations.register_translation(OnnxArgMaxTranslation(),
                                      converter_type('ArgMax', 'onnx'),
                                      op_adapter.ArgMaxOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   ArgMin
# ------------------------------------------------------------------------------
class OnnxArgMinTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('ArgMin', [1, 11])

    def extract_parameters(self, src_op, graph):
        # these parameters belong to ArgMin
        params = extract_attributes(src_op, schema=self.op_schema(), validate=True)

        return op_adapter.ArgMinOp(str(src_op.name),
                                   axis=params.axis,
                                   keep_dims=params.keepdims)


OnnxTranslations.register_translation(OnnxArgMinTranslation(),
                                      converter_type('ArgMin', 'onnx'),
                                      op_adapter.ArgMinOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Ceil
# ------------------------------------------------------------------------------
class OnnxCeilTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Ceil', [1, 6, 13])

    def extract_parameters(self, src_op, graph):
        return op_adapter.ElementwiseUnaryCeilOp(str(src_op.name))


OnnxTranslations.register_translation(OnnxCeilTranslation(),
                                      converter_type('Ceil', 'onnx'),
                                      op_adapter.ElementwiseUnaryCeilOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Div
# ------------------------------------------------------------------------------
class OnnxDivTranslation(ElementwiseBinaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Div', [1, 6, 7])

    def extract_parameters(self, src_op, graph):
        return op_adapter.ElementwiseDivOp(str(src_op.name))


OnnxTranslations.register_translation(OnnxDivTranslation(),
                                      converter_type('Div', 'onnx'),
                                      op_adapter.ElementwiseDivOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Elu
# ------------------------------------------------------------------------------
class OnnxEluTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Elu', [1, 6])

    def extract_parameters(self, src_op, graph):
        # these parameters belong to Elu
        params = extract_attributes(src_op, schema=self.op_schema())
        return op_adapter.NeuronOp(str(src_op.name),
                                   op_adapter.NeuronOp.extract_activation(src_op.op_type),
                                   a=params.alpha)


OnnxTranslations.register_translation(OnnxEluTranslation(),
                                      converter_type('Elu', 'onnx'))


# ------------------------------------------------------------------------------
#   Equal
# ------------------------------------------------------------------------------
class OnnxEqualTranslation(ElementwiseBinaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Equal', [1, 7, 11])

    def extract_parameters(self, src_op, graph):
        return op_adapter.ElementwiseEqualOp(str(src_op.name))


OnnxTranslations.register_translation(OnnxEqualTranslation(),
                                      converter_type('Equal', 'onnx'),
                                      op_adapter.ElementwiseEqualOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Erf
# ------------------------------------------------------------------------------
class OnnxErfTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Erf', [9, 13])

    def extract_parameters(self, src_op, graph):
        return op_adapter.ErfOp(str(src_op.name))


OnnxTranslations.register_translation(OnnxErfTranslation(),
                                      converter_type('Erf', 'onnx'),
                                      op_adapter.ErfOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Exp
# ------------------------------------------------------------------------------
class OnnxExpTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Exp', [1, 6])

    def extract_parameters(self, src_op, graph):
        return op_adapter.ElementwiseUnaryExpOp(str(src_op.name))


OnnxTranslations.register_translation(OnnxExpTranslation(),
                                      converter_type('Exp', 'onnx'),
                                      op_adapter.ElementwiseUnaryExpOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Floor
# ------------------------------------------------------------------------------
class OnnxFloorTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Floor', [1, 6])

    def extract_parameters(self, src_op, graph):
        return op_adapter.ElementwiseUnaryFloorOp(str(src_op.name))


OnnxTranslations.register_translation(OnnxFloorTranslation(),
                                      converter_type('Floor', 'onnx'),
                                      op_adapter.ElementwiseUnaryFloorOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   GEMM
# ------------------------------------------------------------------------------
class OnnxGemmTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Gemm', [1, 6, 7, 9, 11])
        self.params = None

    def extract_parameters(self, src_op, graph):
        log_warning(code_to_message.get_warning_message("WARNING_GEMM"))
        self.params = extract_attributes(src_op, schema=self.op_schema(), validate=True)
        input_names = list(map(str, src_op.input))
        bias = None
        # In newer opset versions, bias is made an optional parameter
        # in the Gemm operator. Default value of bias in this case is 0
        weights = graph.weights.fetch(input_names[1])
        if len(src_op.input) == 3:
            bias = graph.weights.fetch(input_names[2])
        weights = weights * self.params.alpha

        # Transpose weights if transB is given
        if self.params.transB:
            weights = numpy.ascontiguousarray(numpy.transpose(weights, (1, 0)))

        if bias is None:
            bias = numpy.zeros((weights.shape[1],))

        bias = bias * self.params.beta

        # Transpose input if transA is given
        if self.params.transA:
            permute_op = op_adapter.PermuteOp(input_names[0] + '_permute', order=[1, 0])
            graph.add(permute_op, [input_names[0]], [input_names[0] + '_permute'])
            graph.add_src_op_info(permute_op.name, [input_names[0]], [input_names[0] + '_permute'])

        return op_adapter.FullyConnectedOp(str(src_op.name),
                                           weights,
                                           bias,
                                           bias_op_name=input_names[2] if len(src_op.input) == 3 else None)

    def extract_input_names(self, src_op, graph):
        if self.params.transA:
            return [str(src_op.input[0]) + '_permute']
        return [str(src_op.input[0])]


OnnxTranslations.register_translation(OnnxGemmTranslation(), converter_type('Gemm', 'onnx'))


# ------------------------------------------------------------------------------
#   Greater
# ------------------------------------------------------------------------------
class OnnxGreaterTranslation(ElementwiseBinaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Greater', [1, 7, 9])

    def extract_parameters(self, src_op, graph):
        return op_adapter.ElementwiseGreaterOp(str(src_op.name))


OnnxTranslations.register_translation(OnnxGreaterTranslation(),
                                      converter_type('Greater', 'onnx'),
                                      op_adapter.ElementwiseGreaterOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   GreaterOrEqual
# ------------------------------------------------------------------------------
class OnnxGreaterOrEqualTranslation(ElementwiseBinaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('GreaterOrEqual', [12])

    def extract_parameters(self, src_op, graph):
        return op_adapter.ElementwiseGreaterEqualOp(str(src_op.name))


# GreaterOrEqual is announced in ONNX 1.7.0, add if statement to avoid warning
if distutils.version.LooseVersion(onnx.__version__) >= distutils.version.LooseVersion("1.7.0"):
    OnnxTranslations.register_translation(OnnxGreaterOrEqualTranslation(),
                                          converter_type('GreaterOrEqual', 'onnx'),
                                          op_adapter.ElementwiseGreaterEqualOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Identity
# ------------------------------------------------------------------------------
class OnnxIdentityTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Identity', [1])

    def extract_parameters(self, src_op, graph):
        # if the input buffer is not in the graph, that means
        # it is a const input. We replace all const inputs with a
        # const op. Otherwise the identity op is a no-op that
        # gets squashed later.
        if not graph.has_buffer(src_op.input[0]):
            const_input = graph.weights.fetch(str(src_op.input[0]))
            graph.weights.insert(str(src_op.output[0]), const_input)
            return op_adapter.ConstantOp(src_op.output[0], const_input)

        return op_adapter.NoopOp(str(src_op.name))

    def extract_input_names(self, src_op, graph):
        # if the input buffer is not in the graph, that means
        # it is a const input. We replace all const inputs with a
        # const op which do not need an input name.
        if not graph.has_buffer(src_op.input[0]):
            return []
        return str(src_op.input[0])


OnnxTranslations.register_translation(OnnxIdentityTranslation(),
                                      converter_type('Identity', 'onnx'))


# ------------------------------------------------------------------------------
#   Less
# ------------------------------------------------------------------------------
class OnnxLessTranslation(ElementwiseBinaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Less', [1, 7, 9])

    def extract_parameters(self, src_op, graph):
        return op_adapter.ElementwiseLessOp(str(src_op.name))


OnnxTranslations.register_translation(OnnxLessTranslation(),
                                      converter_type('Less', 'onnx'),
                                      op_adapter.ElementwiseLessOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   LessOrEqual
# ------------------------------------------------------------------------------
class OnnxLessOrEqualTranslation(ElementwiseBinaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('LessOrEqual', [12])

    def extract_parameters(self, src_op, graph):
        return op_adapter.ElementwiseLessEqualOp(str(src_op.name))


# LessOrEqual is announced in ONNX 1.7.0, add if statement to avoid warning
if distutils.version.LooseVersion(onnx.__version__) >= distutils.version.LooseVersion("1.7.0"):
    OnnxTranslations.register_translation(OnnxLessOrEqualTranslation(),
                                          converter_type('LessOrEqual', 'onnx'),
                                          op_adapter.ElementwiseLessEqualOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   LpNormalization
# ------------------------------------------------------------------------------
class OnnxLpNormalizationTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('LpNormalization', [1])

    def extract_parameters(self, src_op, graph):
        params = extract_attributes(src_op, schema=self.op_schema())

        if params.p != 2:
            raise ValueError("Only the L2-Norm is supported. "
                             "Found order of {}".format(params.p))

        # we use the default value of epsilon here
        return op_adapter.L2NormOp(src_op.name,
                                   axis=params.axis)


OnnxTranslations.register_translation(OnnxLpNormalizationTranslation(),
                                      converter_type('LpNormalization', 'onnx'),
                                      op_adapter.L2NormOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Log
# ------------------------------------------------------------------------------
class OnnxLogTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Log', [1, 6])

    def extract_parameters(self, src_op, graph):
        return op_adapter.ElementwiseUnaryLogOp(str(src_op.name))


OnnxTranslations.register_translation(OnnxLogTranslation(),
                                      converter_type('Log', 'onnx'),
                                      op_adapter.ElementwiseUnaryLogOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Matmul
# ------------------------------------------------------------------------------
class OnnxMatMulTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('MatMul', [1, 9, 13])
        self.input_names = []

    def add_op(self, src_op, graph):
        ops = self.extract_parameters(src_op, graph)
        input_names = self.extract_input_names(src_op, graph)
        output_names = self.extract_output_names(src_op, graph)
        if len(ops) > 1:
            weights_node = graph.add(ops[0], [], ops[0].name)
            # add src_op info for added constant op
            graph.add_src_op_info(weights_node.op.name, None, weights_node.output_names[0])

        last_node = graph.add(ops[-1], input_names, output_names)
        self.add_src_op_info(last_node.op.name, src_op, graph)

        return last_node

    def extract_parameters(self, src_op, graph):
        self.input_names = list(map(str, src_op.input))
        ops = []

        # Case 1: given AxB, B is a set of static weights
        if graph.weights.has(self.input_names[0]) or graph.weights.has(self.input_names[1]):
            weight_input_name, act_input_name = (self.input_names[0], self.input_names[1]) \
                if graph.weights.has(self.input_names[0]) else (self.input_names[1], self.input_names[0])

            weights = graph.weights.fetch(weight_input_name, prunable=False)
            bias = numpy.zeros(weights.shape[-1], dtype=numpy.float32)
            act_buf = graph.get_buffer(act_input_name)

            # TODO: remove to only translate to matmul once full support
            if len(weights.shape) == 2 and act_buf.rank() == 4:
                batch, depth, height, width = act_buf.shape
                if weights.shape[0] == depth * height * width:
                    ops.append(op_adapter.FullyConnectedOp(str(src_op.name), weights, bias))
                    self.input_names.remove(weight_input_name)
                    return ops
            # this is to support rank 1 matmul
            if len(weights.shape) == 2 and act_buf.rank() == 1:
                if weights.shape[0] == act_buf.shape[0]:
                    ops.append(op_adapter.FullyConnectedOp(str(src_op.name), weights, bias))
                    self.input_names.remove(weight_input_name)
                    return ops
            if not graph.has_buffer(weight_input_name):
                ops.append(op_adapter.ConstantOp(weight_input_name, weights))

        # Case 2: given AxB, B is in inputs
        else:
            shape_b = graph.get_buffer(self.input_names[1]).shape
            bias = numpy.zeros(shape_b[-1], dtype=numpy.float32)
        # Since ONNX Matmul does not support matrix transpose,
        # both transpose_a and transpose_b are set False
        ops.append(op_adapter.MatMulOp(name=str(src_op.name),
                                       bias=bias,
                                       transpose_a=False,
                                       transpose_b=False))
        return ops

    def extract_input_names(self, src_op, graph):
        return self.input_names


OnnxTranslations.register_translation(OnnxMatMulTranslation(), converter_type('MatMul', 'onnx'))


# ------------------------------------------------------------------------------
#   Max
# ------------------------------------------------------------------------------
class OnnxMaxTranslation(ElementwiseBinaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Max', [1, 6, 8, 12])

    def extract_parameters(self, src_op, graph):
        return op_adapter.ElementwiseMaxOp(str(src_op.name))


OnnxTranslations.register_translation(OnnxMaxTranslation(),
                                      converter_type('Max', 'onnx'),
                                      op_adapter.ElementwiseMaxOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Min
# ------------------------------------------------------------------------------
class OnnxMinTranslation(ElementwiseBinaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Min', [1, 6, 8])

    def extract_parameters(self, src_op, graph):
        return op_adapter.ElementwiseMinOp(str(src_op.name))


OnnxTranslations.register_translation(OnnxMinTranslation(),
                                      converter_type('Min', 'onnx'),
                                      op_adapter.ElementwiseMinOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Mul
# ------------------------------------------------------------------------------
class OnnxMulTranslation(ElementwiseBinaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Mul', [1, 6, 7])

    def extract_parameters(self, src_op, graph):
        return op_adapter.ElementwiseProductOp(str(src_op.name))


OnnxTranslations.register_translation(OnnxMulTranslation(),
                                      converter_type('Mul', 'onnx'),
                                      op_adapter.ElementwiseProductOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Neg
# ------------------------------------------------------------------------------
class OnnxNegTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Neg', [1, 6])

    def extract_parameters(self, src_op, graph):
        return op_adapter.ElementwiseUnaryNegOp(str(src_op.name))


OnnxTranslations.register_translation(OnnxNegTranslation(),
                                      converter_type('Neg', 'onnx'),
                                      op_adapter.ElementwiseUnaryNegOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Not
# ------------------------------------------------------------------------------
class OnnxNotTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Not', [1])

    def extract_parameters(self, src_op, graph):
        return op_adapter.ElementwiseUnaryNotOp(str(src_op.name))


OnnxTranslations.register_translation(OnnxNotTranslation(),
                                      converter_type('Not', 'onnx'),
                                      op_adapter.ElementwiseUnaryNotOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Or
# ------------------------------------------------------------------------------
class OnnxOrTranslation(ElementwiseBinaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Or', [1, 7])

    def extract_parameters(self, src_op, graph):
        return op_adapter.ElementwiseOrOp(str(src_op.name))


OnnxTranslations.register_translation(OnnxOrTranslation(),
                                      converter_type('Or', 'onnx'),
                                      op_adapter.ElementwiseOrOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Pow
# ------------------------------------------------------------------------------
class OnnxPowTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Pow', [1, 7, 12])

    def extract_parameters(self, src_op, graph):

        power_input_name = src_op.input[1]
        power_op = self.fetch_constant_op(power_input_name, graph, prunable=False, fail_if_dynamic=False)
        if power_op and not graph.has_buffer(power_input_name):
            graph.add(power_op, [], power_input_name)

        return op_adapter.ElementwisePowerOp(str(src_op.name))


OnnxTranslations.register_translation(OnnxPowTranslation(),
                                      converter_type('Pow', 'onnx'),
                                      op_adapter.ElementwisePowerOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   ReduceBase
# ------------------------------------------------------------------------------
class OnnxReduceBaseTranslation(OnnxTranslationBase):
    def __init__(self, ir_op_class):
        OnnxTranslationBase.__init__(self)
        self.ir_op_class = ir_op_class

    def extract_parameters(self, src_op, graph):
        input_buf = graph.get_buffer(str(src_op.input[0]))
        schema = self.op_schema()
        schema.replace_default_values(axes=range(input_buf.rank()))
        params = extract_attributes(src_op, schema=schema)

        return self.ir_op_class(str(src_op.name),
                                axes=params.axes,
                                keep_dims=params.keepdims)

    def extract_input_names(self, src_op, graph):
        return [str(src_op.input[0])]


# ------------------------------------------------------------------------------
#   ReduceMax
# ------------------------------------------------------------------------------
class OnnxReduceMaxTranslation(OnnxReduceBaseTranslation):
    def __init__(self):
        OnnxReduceBaseTranslation.__init__(self, op_adapter.ReduceMaxOp)
        self.register_op_schema('ReduceMax', [1, 11, 12, 13])


OnnxTranslations.register_translation(OnnxReduceMaxTranslation(),
                                      converter_type('ReduceMax', 'onnx'),
                                      op_adapter.ReduceMaxOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   ReduceMean
# ------------------------------------------------------------------------------
class OnnxReduceMeanTranslation(OnnxReduceBaseTranslation):
    def __init__(self):
        OnnxReduceBaseTranslation.__init__(self, op_adapter.ReduceMeanOp)
        self.register_op_schema('ReduceMean', [1, 11, 13])


OnnxTranslations.register_translation(OnnxReduceMeanTranslation(),
                                      converter_type('ReduceMean', 'onnx'),
                                      op_adapter.ReduceMeanOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   ReduceMin
# ------------------------------------------------------------------------------
class OnnxReduceMinTranslation(OnnxReduceBaseTranslation):
    def __init__(self):
        OnnxReduceBaseTranslation.__init__(self, op_adapter.ReduceMinOp)
        self.register_op_schema('ReduceMin', [1, 11, 12, 13])


OnnxTranslations.register_translation(OnnxReduceMinTranslation(),
                                      converter_type('ReduceMin', 'onnx'),
                                      op_adapter.ReduceMinOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   ReduceProd
# ------------------------------------------------------------------------------
class OnnxReduceProdTranslation(OnnxReduceBaseTranslation):
    def __init__(self):
        OnnxReduceBaseTranslation.__init__(self, op_adapter.ReduceProdOp)
        self.register_op_schema('ReduceProd', [1, 11, 13])


OnnxTranslations.register_translation(OnnxReduceProdTranslation(),
                                      converter_type('ReduceProd', 'onnx'),
                                      op_adapter.ReduceProdOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   ReduceSum
# ------------------------------------------------------------------------------
class OnnxReduceSumTranslation(OnnxReduceBaseTranslation):
    def __init__(self):
        OnnxReduceBaseTranslation.__init__(self, op_adapter.ReduceSumOp)
        self.register_op_schema('ReduceSum', [1, 11, 13])


OnnxTranslations.register_translation(OnnxReduceSumTranslation(),
                                      converter_type('ReduceSum', 'onnx'),
                                      op_adapter.ReduceSumOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Relu
# ------------------------------------------------------------------------------
class OnnxReluTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Relu', [1, 6])

    def extract_parameters(self, src_op, graph):
        return op_adapter.NeuronOp(str(src_op.name),
                                   op_adapter.NeuronOp.extract_activation(src_op.op_type))


OnnxTranslations.register_translation(OnnxReluTranslation(),
                                      converter_type('Relu', 'onnx'),
                                      op_adapter.NeuronOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Round
# ------------------------------------------------------------------------------
class OnnxRoundTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Round', [11])

    def extract_parameters(self, src_op, graph):
        return op_adapter.ElementwiseUnaryRoundOp(str(src_op.name))


OnnxTranslations.register_translation(OnnxRoundTranslation(),
                                      converter_type('Round', 'onnx'),
                                      op_adapter.ElementwiseUnaryRoundOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Sigmoid
# ------------------------------------------------------------------------------
class OnnxSigmoidTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Sigmoid', [1, 6])

    def extract_parameters(self, src_op, graph):
        return op_adapter.NeuronOp(str(src_op.name),
                                   op_adapter.NeuronOp.extract_activation(src_op.op_type), a=1.0)


OnnxTranslations.register_translation(OnnxSigmoidTranslation(), converter_type('Sigmoid', 'onnx'))


# ------------------------------------------------------------------------------
#   Sin
# ------------------------------------------------------------------------------
class OnnxSinTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Sin', [7])

    def extract_parameters(self, src_op, graph):
        return op_adapter.ElementwiseUnarySinOp(str(src_op.name))


OnnxTranslations.register_translation(OnnxSinTranslation(),
                                      converter_type('Sin', 'onnx'),
                                      op_adapter.ElementwiseUnarySinOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Softmax
# ------------------------------------------------------------------------------
class OnnxSoftmaxTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Softmax', [1, 11])

    def extract_parameters(self, src_op, graph):
        params = extract_attributes(src_op, schema=self.op_schema())
        axis = getattr(params, "axis", 1)
        return op_adapter.SoftmaxOp(str(src_op.name),
                                    axis=axis)


OnnxTranslations.register_translation(OnnxSoftmaxTranslation(),
                                      converter_type('Softmax', 'onnx'),
                                      op_adapter.SoftmaxOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Sub
# ------------------------------------------------------------------------------
class OnnxSubTranslation(ElementwiseBinaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Sub', [1, 6, 7])

    def extract_parameters(self, src_op, graph):
        return op_adapter.ElementwiseSubOp(str(src_op.name))


OnnxTranslations.register_translation(OnnxSubTranslation(),
                                      converter_type('Sub', 'onnx'),
                                      op_adapter.ElementwiseSubOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Sum
# ------------------------------------------------------------------------------
class OnnxSumTranslation(ElementwiseBinaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Sum', [1, 6, 8])

    def extract_parameters(self, src_op, graph):
        return op_adapter.ElementwiseSumOp(str(src_op.name))


OnnxTranslations.register_translation(OnnxSumTranslation(), converter_type('Sum', 'onnx'))


# ------------------------------------------------------------------------------
#   Sqrt
# ------------------------------------------------------------------------------
class OnnxSqrtTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Sqrt', [1, 6])

    def extract_parameters(self, src_op, graph):
        return op_adapter.ElementwiseUnarySqrtOp(str(src_op.name))


OnnxTranslations.register_translation(OnnxSqrtTranslation(),
                                      converter_type('Sqrt', 'onnx'),
                                      op_adapter.ElementwiseUnarySqrtOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Tanh
# ------------------------------------------------------------------------------
class OnnxTanhTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Tanh', [1, 6])

    def extract_parameters(self, src_op, graph):
        return op_adapter.NeuronOp(str(src_op.name),
                                   op_adapter.NeuronOp.extract_activation(src_op.op_type),
                                   a=1.0,
                                   b=1.0)


OnnxTranslations.register_translation(OnnxTanhTranslation(),
                                      converter_type('Tanh', 'onnx'))


# ------------------------------------------------------------------------------
#   ScaledTanh
# ------------------------------------------------------------------------------
class OnnxScaledTanhTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('ScaledTanh', [1, 6])

    def extract_parameters(self, src_op, graph):
        # these parameters belong to ScaledTanh
        params = extract_attributes(src_op, schema=self.op_schema())
        return op_adapter.NeuronOp(str(src_op.name),
                                   op_adapter.NeuronOp.extract_activation(src_op.op_type),
                                   a=params.alpha,
                                   b=params.beta)


# scaledtanh is removed in ONNX release v1.5.0, add if statement to avoid warning
if distutils.version.LooseVersion(onnx.__version__) < distutils.version.LooseVersion("1.5.0"):
    OnnxTranslations.register_translation(OnnxScaledTanhTranslation(),
                                          converter_type('ScaledTanh', 'onnx'))


# ------------------------------------------------------------------------------
#   TopK
# ------------------------------------------------------------------------------
class OnnxTopKTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('TopK', [1, 10, 11])

    def extract_parameters(self, src_op, graph):
        params = extract_attributes(src_op, schema=self.op_schema())
        input_names = list(src_op.input)
        input_buf = graph.get_buffer(str(src_op.input[0]))
        input_rank = input_buf.rank()
        input_dims = input_buf.get_buf_dims()

        # extract K as input in versions 10, 11 and as parameter in version 1
        if len(input_names) == 2:
            const_op = self.fetch_constant_op(input_names[1], graph)
            log_assert(const_op is not None,
                       "Input tensor {} of node {} could not be extracted.".format(input_names[1], src_op.name))
            k = const_op.tensor.astype(numpy.int64).item(0)
        else:
            k = params.k

        largest = params.largest if 'largest' in params else 1
        sorted = params.sorted if 'sorted' in params else 1
        axis = params.axis

        if axis < 0:
            axis += input_rank

        log_assert(input_rank >= 1,
                   code_to_message.get_error_message("ERROR_TOPK_INPUT_TENSOR_RANK")(input_rank))

        if k < 0 or input_dims[axis] < k:
            raise ValueError(
                code_to_message.get_error_message("ERROR_TOPK_K_INVALID")(k, input_dims[axis]))

        return op_adapter.TopKOp(src_op.name,
                                 k=k,
                                 axis=axis,
                                 largest=largest,
                                 sorted=sorted)

    def extract_input_names(self, src_op, graph):
        return [src_op.input[0]]


OnnxTranslations.register_translation(OnnxTopKTranslation(),
                                      converter_type('TopK', 'onnx'),
                                      op_adapter.TopKOp.TRANSLATION_KEY)
