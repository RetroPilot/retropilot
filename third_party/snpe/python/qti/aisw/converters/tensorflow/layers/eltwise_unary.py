# =============================================================================
#
#  Copyright (c) 2018-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import numpy as np
from qti.aisw.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from qti.aisw.converters.common.converter_ir.op_adapter import (
    ConstantOp,
    ElementwisePowerOp,
    ElementwiseUnaryAbsOp,
    ElementwiseUnaryCeilOp,
    ElementwiseUnaryExpOp,
    ElementwiseUnaryFloorOp,
    ElementwiseUnaryLogOp,
    ElementwiseUnaryNegOp,
    ElementwiseUnaryNotOp,
    ElementwiseUnaryRoundOp,
    ElementwiseUnaryRsqrtOp,
    ElementwiseUnarySinOp,
    ElementwiseUnarySqrtOp
)
from abc import ABCMeta
from qti.aisw.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    NonConsumableConverterSequenceNode,
    GraphSequence
)


class EltWiseUnaryLayerResolver(LayerResolver, object):
    __metaclass__ = ABCMeta

    def __init__(self, layer_type, op_type, descriptor_class):
        super(EltWiseUnaryLayerResolver, self).__init__()
        self._layer_type = layer_type
        self._op_type = op_type
        self._descriptor_class = descriptor_class

        self.sequence = GraphSequence([
            ConverterSequenceNode('root', [self._op_type]),
            NonConsumableConverterSequenceNode('input1', ['?']),
        ])
        self.sequence.set_inputs('root', ['input1'])
        self.sequence.set_outputs(['root'])

    def resolve_layer(self, graph_matcher, graph_helper):
        descriptors = []
        non_const_input_sequences = [self.sequence]
        for sequence in non_const_input_sequences:
            for match in graph_matcher.match_sequence(sequence):
                eltwise_op = match['root']
                descriptor = self._descriptor_class(self._layer_type, str(eltwise_op.name), match.consumed_nodes)
                descriptors.append(descriptor)
        return descriptors


class EltWiseUnaryLayerBuilder(LayerBuilder):
    def build_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type ir_graph: converters.common.converter_ir.op_graph.IROpGraph
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: EltWiseUnaryAbsLayerResolver.Descriptor
        :rtype: int
        """
        input_name = self.get_input_name(converter_context, descriptor, input_descriptors)
        output_name = descriptor.output_names[0]
        return ir_graph.add(self._op_class(descriptor.layer_name),
                            input_name,
                            output_name)


class EltWiseUnaryAbsLayerResolver(EltWiseUnaryLayerResolver):
    class Descriptor(LayerDescriptor):
        pass

    def __init__(self):
        super(EltWiseUnaryAbsLayerResolver, self).__init__('ElementWiseUnaryAbs', 'Abs',
                                                           EltWiseUnaryAbsLayerResolver.Descriptor)


class EltWiseUnaryAbsLayerBuilder(EltWiseUnaryLayerBuilder):
    def __init__(self):
        super(EltWiseUnaryAbsLayerBuilder, self).__init__()
        self._op_class = ElementwiseUnaryAbsOp


class EltWiseUnaryCeilLayerResolver(EltWiseUnaryLayerResolver):
    class Descriptor(LayerDescriptor):
        pass

    def __init__(self):
        super(EltWiseUnaryCeilLayerResolver, self).__init__('ElementWiseUnaryCeil', 'Ceil',
                                                            EltWiseUnaryCeilLayerResolver.Descriptor)


class EltWiseUnaryCeilLayerBuilder(EltWiseUnaryLayerBuilder):
    def __init__(self):
        super(EltWiseUnaryCeilLayerBuilder, self).__init__()
        self._op_class = ElementwiseUnaryCeilOp


class EltWiseUnaryExpLayerResolver(EltWiseUnaryLayerResolver):
    class Descriptor(LayerDescriptor):
        pass

    def __init__(self):
        super(EltWiseUnaryExpLayerResolver, self).__init__('ElementWiseUnaryExp', 'Exp',
                                                           EltWiseUnaryExpLayerResolver.Descriptor)


class EltWiseUnaryExpLayerBuilder(EltWiseUnaryLayerBuilder):
    def __init__(self):
        super(EltWiseUnaryExpLayerBuilder, self).__init__()
        self._op_class = ElementwiseUnaryExpOp


class EltWiseUnaryFloorLayerResolver(EltWiseUnaryLayerResolver):
    class Descriptor(LayerDescriptor):
        pass

    def __init__(self):
        super(EltWiseUnaryFloorLayerResolver, self).__init__('ElementWiseUnaryFloor', 'Floor',
                                                             EltWiseUnaryFloorLayerResolver.Descriptor)


class EltWiseUnaryFloorLayerBuilder(EltWiseUnaryLayerBuilder):
    def __init__(self):
        super(EltWiseUnaryFloorLayerBuilder, self).__init__()
        self._op_class = ElementwiseUnaryFloorOp


class EltWiseUnaryLogLayerResolver(EltWiseUnaryLayerResolver):
    class Descriptor(LayerDescriptor):
        pass

    def __init__(self):
        super(EltWiseUnaryLogLayerResolver, self).__init__('ElementWiseUnaryLog', 'Log',
                                                           EltWiseUnaryLogLayerResolver.Descriptor)


class EltWiseUnaryLogLayerBuilder(EltWiseUnaryLayerBuilder):
    def __init__(self):
        super(EltWiseUnaryLogLayerBuilder, self).__init__()
        self._op_class = ElementwiseUnaryLogOp


class EltWiseUnaryLogicalNotLayerResolver(EltWiseUnaryLayerResolver):
    class Descriptor(LayerDescriptor):
        pass

    def __init__(self):
        super(EltWiseUnaryLogicalNotLayerResolver, self).__init__('ElementWiseUnaryLogicalNot', 'LogicalNot',
                                                                  EltWiseUnaryLogicalNotLayerResolver.Descriptor)


class EltWiseUnaryLogicalNotLayerBuilder(EltWiseUnaryLayerBuilder):
    def __init__(self):
        super(EltWiseUnaryLogicalNotLayerBuilder, self).__init__()
        self._op_class = ElementwiseUnaryNotOp


class EltWiseUnaryNegLayerResolver(EltWiseUnaryLayerResolver):
    class Descriptor(LayerDescriptor):
        pass

    def __init__(self):
        super(EltWiseUnaryNegLayerResolver, self).__init__('ElementWiseUnaryNeg', 'Neg',
                                                           EltWiseUnaryNegLayerResolver.Descriptor)


class EltWiseUnaryNegLayerBuilder(EltWiseUnaryLayerBuilder):
    def __init__(self):
        super(EltWiseUnaryNegLayerBuilder, self).__init__()
        self._op_class = ElementwiseUnaryNegOp


class EltWiseUnaryRoundLayerResolver(EltWiseUnaryLayerResolver):
    class Descriptor(LayerDescriptor):
        pass

    def __init__(self):
        super(EltWiseUnaryRoundLayerResolver, self).__init__('ElementWiseUnaryRound', 'Round',
                                                             EltWiseUnaryRoundLayerResolver.Descriptor)


class EltWiseUnaryRoundLayerBuilder(EltWiseUnaryLayerBuilder):
    def __init__(self):
        super(EltWiseUnaryRoundLayerBuilder, self).__init__()
        self._op_class = ElementwiseUnaryRoundOp


class EltWiseUnaryRsqrtLayerResolver(EltWiseUnaryLayerResolver):
    class Descriptor(LayerDescriptor):
        pass

    def __init__(self):
        super(EltWiseUnaryRsqrtLayerResolver, self).__init__('ElementWiseUnaryRsqrt', 'Rsqrt',
                                                             EltWiseUnaryRsqrtLayerResolver.Descriptor)


class EltWiseUnaryRsqrtLayerBuilder(EltWiseUnaryLayerBuilder):
    def __init__(self):
        super(EltWiseUnaryRsqrtLayerBuilder, self).__init__()
        self._op_class = ElementwiseUnaryRsqrtOp


class EltWiseUnarySinLayerResolver(EltWiseUnaryLayerResolver):
    class Descriptor(LayerDescriptor):
        pass

    def __init__(self):
        super(EltWiseUnarySinLayerResolver, self).__init__('ElementWiseUnarySin', 'Sin',
                                                           EltWiseUnarySinLayerResolver.Descriptor)


class EltWiseUnarySinLayerBuilder(EltWiseUnaryLayerBuilder):
    def __init__(self):
        super(EltWiseUnarySinLayerBuilder, self).__init__()
        self._op_class = ElementwiseUnarySinOp


class EltWiseUnarySqrtLayerResolver(EltWiseUnaryLayerResolver):
    class Descriptor(LayerDescriptor):
        pass

    def __init__(self):
        super(EltWiseUnarySqrtLayerResolver, self).__init__('ElementWiseUnarySqrt', 'Sqrt',
                                                            EltWiseUnarySqrtLayerResolver.Descriptor)


class EltWiseUnarySqrtLayerBuilder(EltWiseUnaryLayerBuilder):
    def __init__(self):
        super(EltWiseUnarySqrtLayerBuilder, self).__init__()
        self._op_class = ElementwiseUnarySqrtOp


class EltWiseUnarySquareLayerResolver(EltWiseUnaryLayerResolver):
    class Descriptor(LayerDescriptor):
        pass

    def __init__(self):
        super(EltWiseUnarySquareLayerResolver, self).__init__('ElementWiseUnarySquare', 'Square',
                                                              EltWiseUnarySquareLayerResolver.Descriptor)


class EltWiseUnarySquareLayerBuilder(LayerBuilder):
    def build_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type ir_graph: converters.common.converter_ir.op_graph.IROpGraph
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: EltWiseUnarySquareLayerResolver.Descriptor
        :rtype: int
        """
        input_name = self.get_input_name(converter_context, descriptor, input_descriptors)
        output_name = descriptor.output_names[0]

        pow_op_name = descriptor.layer_name + "_pow"
        pow_op = ConstantOp(pow_op_name, tensor=np.asarray([2], dtype=np.float32))
        ir_graph.add(pow_op, [], pow_op_name)

        return ir_graph.add(ElementwisePowerOp(descriptor.layer_name),
                            [input_name, pow_op_name],
                            output_name)
