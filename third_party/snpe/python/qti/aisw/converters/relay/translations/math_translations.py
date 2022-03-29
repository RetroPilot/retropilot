# ==============================================================================
#
#  Copyright (c) 2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from qti.aisw.converters.common.utils.converter_utils import log_debug2, log_debug3, converter_type
from qti.aisw.converters.common.converter_ir.op_graph import IROpGraph
from qti.aisw.converters.common.converter_ir.op_adapter import (
    ArgMaxOp,
    ConstantOp,
    ElementwiseDivOp,
    ElementwiseMaxOp,
    ElementwiseMinOp,
    ElementwiseProductOp,
    ElementwiseSubOp,
    ElementwiseSumOp,
    ElementwiseUnaryAbsOp,
    ElementwiseUnaryExpOp,
    ElementwiseUnaryFloorOp,
    ElementwiseUnaryLogOp,
    ElementwiseUnarySqrtOp,
    NeuronOp,
    ReduceMeanOp
)

from qti.aisw.converters.relay.translations.relay_translations import RelayTranslationBase
from qti.aisw.converters.relay.translations import RelayTranslations

import tvm
from tvm import relay

import numpy as np


# ------------------------------------------------------------------------------
#   ArgMax
# ------------------------------------------------------------------------------
class RelayArgMaxTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayArgMaxTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}
        axis = relay_expr.attrs.axis
        keepdims = relay_expr.attrs.keepdims
        exclude = relay_expr.attrs.exclude

        if isinstance(axis, tvm.ir.container.Array):
            axis = [int(i) for i in axis]
        elif isinstance(axis, tvm.tir.expr.IntImm):
            axis = int(axis)
        else:
            TypeError("Argmax axis is of Unsupported datatype {}".format(type(axis)))

        if keepdims:
            keepdims = True
        else:
            keepdims = False

        attr_dict["axis"] = axis
        attr_dict["keepdims"] = keepdims
        attr_dict["exclude"] = exclude

        log_debug3("\taxis {} {}", type(axis), axis)
        log_debug3("\tkeepdims {} {}", type(keepdims), keepdims)
        log_debug3("\texclude {} {}", type(exclude), exclude)

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, ArgMaxOp.TRANSLATION_KEY)
        input_shapes = converter_context.get_input_shapes(relay_expr)

        axis = attr_dict["axis"]
        keepdims = attr_dict["keepdims"]
        exclude = attr_dict["exclude"]

        if exclude:
            axis = [i for i in range(len(input_shapes[0])) if i not in axis]

        if isinstance(axis, list):
            if len(axis) > 1:
                raise ValueError("Argmax axis only supported as scalar, got list {}".format(axis))
            axis = axis[0]

        ir_op = ArgMaxOp(op_name,
                         axis=axis,
                         keep_dims=keepdims)

        return ir_op


RelayTranslations.register_translation(RelayArgMaxTranslation(),
                                       converter_type('argmax', 'relay'))


# ------------------------------------------------------------------------------
#   ElementwiseBinaryBase
# ------------------------------------------------------------------------------
class RelayElementwiseBinaryBaseTranslation(RelayTranslationBase):
    def __init__(self, quir_op_type):
        super(RelayElementwiseBinaryBaseTranslation, self).__init__()
        self.quir_op_type = quir_op_type

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):

        op_name = converter_context.get_op_name(relay_expr, self.quir_op_type.TRANSLATION_KEY)


        input_0_name = input_names[0]
        input_1_name = input_names[1]

        if not quir_graph.has_buffer(input_0_name):
            # Op has 1st input as Constant tensor
            input_0_tensor = relay_params[input_0_name]
            log_debug3("\tconst tensor type {} shape {}", type(input_0_tensor), input_0_tensor.shape)
            if isinstance(input_0_tensor, tvm.runtime.ndarray.NDArray) or \
                    isinstance(input_0_tensor, tvm.runtime.NDArray):
                input_0_tensor = input_0_tensor.asnumpy()

            if not input_0_tensor.shape:
                input_0_tensor = np.reshape(input_0_tensor.data, (1,))
            log_debug3("\tconst tensor after type {} shape {}", type(input_0_tensor), input_0_tensor.shape)

            constant_output_name = op_name + "_const_0"
            input_names[0] = constant_output_name
            log_debug2("Adding Constant Op name:{} with output:{}", constant_output_name, constant_output_name)
            quir_graph.add(ConstantOp(constant_output_name, input_0_tensor), [], [constant_output_name])

        if not quir_graph.has_buffer(input_1_name):
            # Op has 2nd input as Constant tensor
            input_1_tensor = relay_params[input_names[1]]
            log_debug3("\tconst tensor type {} shape {}", type(input_1_tensor), input_1_tensor.shape)
            if isinstance(input_1_tensor, tvm.runtime.ndarray.NDArray) or\
                    isinstance(input_1_tensor, tvm.runtime.NDArray):
                input_1_tensor = input_1_tensor.asnumpy()

            if not input_1_tensor.shape:
                input_1_tensor = np.reshape(input_1_tensor.data, (1,))
            log_debug3("\tconst tensor after type {} shape {}", type(input_1_tensor), input_1_tensor.shape)

            constant_output_name = op_name + "_const_1"
            input_names[1] = constant_output_name
            log_debug2("Adding Constant Op name:{} with output:{}", constant_output_name, constant_output_name)
            quir_graph.add(ConstantOp(constant_output_name, input_1_tensor), [], [constant_output_name])

        ir_op = self.quir_op_type(op_name)

        return ir_op


# ------------------------------------------------------------------------------
#   ElementwiseDiv
# ------------------------------------------------------------------------------
class RelayElementwiseDivTranslation(RelayElementwiseBinaryBaseTranslation):
    def __init__(self):
        super(RelayElementwiseDivTranslation, self).__init__(ElementwiseDivOp)

RelayTranslations.register_translation(RelayElementwiseDivTranslation(),
                                       converter_type('divide', 'relay'))


# ------------------------------------------------------------------------------
#   ElementwiseSum
# ------------------------------------------------------------------------------
class RelayElementwiseSumTranslation(RelayElementwiseBinaryBaseTranslation):
    def __init__(self):
        super(RelayElementwiseSumTranslation, self).__init__(ElementwiseSumOp)

RelayTranslations.register_translation(RelayElementwiseSumTranslation(),
                                       converter_type('add', 'relay'))


# ------------------------------------------------------------------------------
#   ElementwiseSub
# ------------------------------------------------------------------------------
class RelayElementwiseSubTranslation(RelayElementwiseBinaryBaseTranslation):
    def __init__(self):
        super(RelayElementwiseSubTranslation, self).__init__(ElementwiseSubOp)

RelayTranslations.register_translation(RelayElementwiseSubTranslation(),
                                       converter_type('subtract', 'relay'))


# ------------------------------------------------------------------------------
#   ElementwiseMax
# ------------------------------------------------------------------------------
class RelayElementwiseMaxTranslation(RelayElementwiseBinaryBaseTranslation):
    def __init__(self):
        super(RelayElementwiseMaxTranslation, self).__init__(ElementwiseMaxOp)

RelayTranslations.register_translation(RelayElementwiseMaxTranslation(),
                                       converter_type('maximum', 'relay'))


# ------------------------------------------------------------------------------
#   ElementwiseMin
# ------------------------------------------------------------------------------
class RelayElementwiseMinTranslation(RelayElementwiseBinaryBaseTranslation):
    def __init__(self):
        super(RelayElementwiseMinTranslation, self).__init__(ElementwiseMinOp)

RelayTranslations.register_translation(RelayElementwiseMinTranslation(),
                                       converter_type('minimum', 'relay'))


# ------------------------------------------------------------------------------
#   ElementwiseProd
# ------------------------------------------------------------------------------
class RelayElementwiseProdTranslation(RelayElementwiseBinaryBaseTranslation):
    def __init__(self):
        super(RelayElementwiseProdTranslation, self).__init__(ElementwiseProductOp)

RelayTranslations.register_translation(RelayElementwiseProdTranslation(),
                                       converter_type('multiply', 'relay'))


# ------------------------------------------------------------------------------
#   Tanh
# ------------------------------------------------------------------------------
class RelayTanhTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayTanhTranslation, self).__init__()

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, NeuronOp.TRANSLATION_KEY)

        ir_op = NeuronOp(op_name,
                         NeuronOp.Type.TANH,
                         a=1.0,
                         b=1.0)
        return ir_op


RelayTranslations.register_translation(RelayTanhTranslation(),
                                       converter_type('tanh', 'relay'))


# ------------------------------------------------------------------------------
#   ElementwiseUnaryBase
# ------------------------------------------------------------------------------
class RelayElementwiseUnaryBaseTranslation(RelayTranslationBase):
    def __init__(self, quir_op_type):
        super(RelayElementwiseUnaryBaseTranslation, self).__init__()
        self.quir_op_type = quir_op_type

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):

        op_name = converter_context.get_op_name(relay_expr, self.quir_op_type.TRANSLATION_KEY)

        ir_op = self.quir_op_type(op_name)

        return ir_op


# ------------------------------------------------------------------------------
#   ElementwiseUnaryAbs
# ------------------------------------------------------------------------------
class RelayElementwiseUnaryAbsTranslation(RelayElementwiseUnaryBaseTranslation):
    def __init__(self):
        super(RelayElementwiseUnaryAbsTranslation, self).__init__(ElementwiseUnaryAbsOp)

RelayTranslations.register_translation(RelayElementwiseUnaryAbsTranslation(),
                                       converter_type('abs', 'relay'))


# ------------------------------------------------------------------------------
#   ElementwiseUnaryExp
# ------------------------------------------------------------------------------
class RelayElementwiseUnaryExpTranslation(RelayElementwiseUnaryBaseTranslation):
    def __init__(self):
        super(RelayElementwiseUnaryExpTranslation, self).__init__(ElementwiseUnaryExpOp)

RelayTranslations.register_translation(RelayElementwiseUnaryExpTranslation(),
                                       converter_type('exp', 'relay'))


# ------------------------------------------------------------------------------
#   ElementwiseUnaryFloor
# ------------------------------------------------------------------------------
class RelayElementwiseUnaryFloorTranslation(RelayElementwiseUnaryBaseTranslation):
    def __init__(self):
        super(RelayElementwiseUnaryFloorTranslation, self).__init__(ElementwiseUnaryFloorOp)

RelayTranslations.register_translation(RelayElementwiseUnaryFloorTranslation(),
                                       converter_type('floor', 'relay'))


# ------------------------------------------------------------------------------
#   ElementwiseUnaryLog
# ------------------------------------------------------------------------------
class RelayElementwiseUnaryLogTranslation(RelayElementwiseUnaryBaseTranslation):
    def __init__(self):
        super(RelayElementwiseUnaryLogTranslation, self).__init__(ElementwiseUnaryLogOp)

RelayTranslations.register_translation(RelayElementwiseUnaryLogTranslation(),
                                       converter_type('log', 'relay'))


# ------------------------------------------------------------------------------
#   ElementwiseUnarySqrt
# ------------------------------------------------------------------------------
class RelayElementwiseUnarySqrtTranslation(RelayElementwiseUnaryBaseTranslation):
    def __init__(self):
        super(RelayElementwiseUnarySqrtTranslation, self).__init__(ElementwiseUnarySqrtOp)

RelayTranslations.register_translation(RelayElementwiseUnarySqrtTranslation(),
                                       converter_type('sqrt', 'relay'))


# ------------------------------------------------------------------------------
#   MeanOp
# ------------------------------------------------------------------------------
class RelayMeanTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayMeanTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}
        attr_dict['keep_dims'] = relay_expr.attrs.keepdims
        axis_attrs = relay_expr.attrs.axis
        if isinstance(axis_attrs, tvm.ir.container.Array):
            attr_dict['axis'] = [int(i) for i in axis_attrs]
        elif isinstance(axis_attrs, tvm.tir.IntImm):
            attr_dict['axis'] = [int(axis_attrs)]
        else:
            attr_dict['axis'] = list()
        attr_dict['exclude'] = relay_expr.attrs.exclude

        log_debug3("\taxis {}", attr_dict['axis'])
        log_debug3("\tkeep_dims {}", attr_dict['keep_dims'])
        log_debug3("\texclude {}", attr_dict['exclude'])

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, ReduceMeanOp.TRANSLATION_KEY)
        axis = attr_dict['axis']
        exclude = attr_dict['exclude']
        input_shape = converter_context.get_input_shapes(relay_expr)[0]
        input_dim = len(input_shape)
        if len(axis) == 0:
            axis = [i for i in range(input_dim)]

        if exclude:
            axis = [i for i in range(input_dim) if i not in axis]

        keep_dims = attr_dict['keep_dims']
        ir_op = ReduceMeanOp(op_name,
                             axes=axis,
                             keep_dims=keep_dims)

        return ir_op

RelayTranslations.register_translation(RelayMeanTranslation(),
                                       converter_type('mean', 'relay'))
