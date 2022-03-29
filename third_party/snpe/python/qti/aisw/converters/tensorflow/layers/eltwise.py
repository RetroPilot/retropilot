# =============================================================================
#
#  Copyright (c) 2015-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import numpy as np
from qti.aisw.converters.common.converter_ir.op_adapter import (
    ElementwiseAndOp,
    ElementwiseDivOp,
    ElementwiseEqualOp,
    ElementwiseFloorDivOp,
    ElementwiseGreaterOp,
    ElementwiseGreaterEqualOp,
    ElementwiseLessOp,
    ElementwiseLessEqualOp,
    ElementwiseMaxOp,
    ElementwiseMinOp,
    ElementwiseNotEqualOp,
    ElementwiseOrOp,
    ElementwisePowerOp,
    ElementwiseProductOp,
    ElementwiseSelectOp,
    ElementwiseSubOp,
    ElementwiseSumOp,
)
from qti.aisw.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from qti.aisw.converters.tensorflow.layers.constant import ConstantLayerResolver
from abc import ABC, abstractmethod
from qti.aisw.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    NonConsumableConverterSequenceNode,
    GraphSequence
)


class EltWiseBiasaddLayerResolver(LayerResolver):
    class Descriptor(LayerDescriptor):
        def __init__(self, name, nodes, bias_name, bias):
            super(EltWiseBiasaddLayerResolver.Descriptor, self).__init__('BiasAdd', name, nodes)
            self.bias_tensor = bias
            self.bias_name = bias_name

    def __init__(self):
        super(EltWiseBiasaddLayerResolver, self).__init__()

        self.sequence = GraphSequence([
            ConverterSequenceNode('root', ['BiasAdd']),
            NonConsumableConverterSequenceNode('bias', ['?']),
            NonConsumableConverterSequenceNode('input', ['?']),
        ])
        self.sequence.set_inputs('root', ['input', 'bias'])
        self.sequence.set_outputs(['root'])

    def resolve_layer(self, graph_matcher, graph_helper):
        descriptors = []
        for match in graph_matcher.match_sequence(self.sequence):
            eltwise_op = match['root']
            bias_op = match['bias']
            bias = graph_helper.evaluate_tensor_output(bias_op.outputs[0])
            descriptor = EltWiseBiasaddLayerResolver.Descriptor(str(eltwise_op.name),
                                                                match.consumed_nodes,
                                                                str(bias_op.name),
                                                                bias)
            descriptors.append(descriptor)

        return descriptors


class EltWiseBiasaddLayerBuilder(LayerBuilder):
    def build_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type ir_graph: converters.common.converter_ir.op_graph.IROpGraph
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: EltWiseBiasaddResolver.Descriptor
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :rtype: int
        """
        input_names = self.get_input_names(converter_context, descriptor, input_descriptors)
        output_name = descriptor.output_names[0]

        return ir_graph.add(ElementwiseSumOp(descriptor.layer_name),
                            input_names,
                            output_name)


class EltWiseLayerResolver(LayerResolver, ABC, object):

    class Descriptor(LayerDescriptor):
        def __init__(self, op_type, name, nodes, output_names=None, unary=False):
            super(EltWiseLayerResolver.Descriptor, self).__init__(op_type, name, nodes, output_names=output_names)
            self.unary = unary

    def __init__(self, layer_type, op_type, descriptor_class):
        super(EltWiseLayerResolver, self).__init__()
        self._layer_type = layer_type
        self._op_type = op_type
        self._descriptor_class = descriptor_class

        self.sequence = GraphSequence([
            ConverterSequenceNode('root', self._op_type),
        ])
        self.sequence.set_outputs(['root'])

    def resolve_layer(self, graph_matcher, graph_helper):
        descriptors = []

        for match in graph_matcher.match_sequence(self.sequence):
            eltwise_op = match['root']
            descriptor = self._descriptor_class(self._layer_type, str(eltwise_op.name), match.consumed_nodes,
                                                unary=(eltwise_op.inputs[0] == eltwise_op.inputs[1]))
            descriptors.append(descriptor)

        return descriptors


class EltWiseLayerBuilder(LayerBuilder, ABC):
    def build_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type ir_graph: converters.common.converter_ir.op_graph.IROpGraph
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: converters.tensorflow.common.LayerDescriptor
        :rtype: int
        """
        input_names = self.get_input_names(converter_context, descriptor, input_descriptors)
        output_name = descriptor.output_names[0]

        # In case there is one input used twice, duplicate the input name
        if descriptor.unary:
            input_names = input_names * 2
        elif len(input_names) != 2:
            raise ValueError("Op {} has only one input {}".format(descriptor.layer_name, input_names))

        return ir_graph.add(self._op_class(descriptor.layer_name),
                            input_names,
                            output_name)


class EltWiseAndLayerResolver(EltWiseLayerResolver):
    class Descriptor(EltWiseLayerResolver.Descriptor):
        pass

    def __init__(self):
        super(EltWiseAndLayerResolver, self).__init__('ElementWiseAnd', ['LogicalAnd'],
                                                      EltWiseAndLayerResolver.Descriptor)


class EltWiseAndLayerBuilder(EltWiseLayerBuilder):
    def __init__(self):
        super(EltWiseAndLayerBuilder, self).__init__()
        self._op_class = ElementwiseAndOp


class EltWiseEqualLayerResolver(EltWiseLayerResolver):
    class Descriptor(EltWiseLayerResolver.Descriptor):
        pass

    def __init__(self):
        super(EltWiseEqualLayerResolver, self).__init__('ElementWiseEqual', ['Equal'],
                                                        EltWiseEqualLayerResolver.Descriptor)


class EltWiseEqualLayerBuilder(EltWiseLayerBuilder):
    def __init__(self):
        super(EltWiseEqualLayerBuilder, self).__init__()
        self._op_class = ElementwiseEqualOp


class EltWiseFloorDivLayerBuilder(EltWiseLayerBuilder):
    def __init__(self):
        super(EltWiseFloorDivLayerBuilder, self).__init__()
        self._op_class = ElementwiseFloorDivOp


class EltWiseFloorDivLayerResolver(EltWiseLayerResolver):
    class Descriptor(EltWiseLayerResolver.Descriptor):
        pass

    def __init__(self):
        super(EltWiseFloorDivLayerResolver, self).__init__('ElementWiseFloorDiv', ['FloorDiv'],
                                                           EltWiseFloorDivLayerResolver.Descriptor)


class EltWiseGreaterLayerResolver(EltWiseLayerResolver):
    class Descriptor(EltWiseLayerResolver.Descriptor):
        pass

    def __init__(self):
        super(EltWiseGreaterLayerResolver, self).__init__('ElementWiseGreater', ['Greater'],
                                                          EltWiseGreaterLayerResolver.Descriptor)


class EltWiseGreaterLayerBuilder(EltWiseLayerBuilder):
    def __init__(self):
        super(EltWiseGreaterLayerBuilder, self).__init__()
        self._op_class = ElementwiseGreaterOp


class EltWiseGreaterEqualLayerResolver(EltWiseLayerResolver):
    class Descriptor(EltWiseLayerResolver.Descriptor):
        pass

    def __init__(self):
        super(EltWiseGreaterEqualLayerResolver, self).__init__('ElementWiseGreaterEqual', ['GreaterEqual'],
                                                               EltWiseGreaterEqualLayerResolver.Descriptor)


class EltWiseGreaterEqualLayerBuilder(EltWiseLayerBuilder):
    def __init__(self):
        super(EltWiseGreaterEqualLayerBuilder, self).__init__()
        self._op_class = ElementwiseGreaterEqualOp


class EltWiseLessLayerResolver(EltWiseLayerResolver):
    class Descriptor(EltWiseLayerResolver.Descriptor):
        pass

    def __init__(self):
        super(EltWiseLessLayerResolver, self).__init__('ElementWiseLess', ['Less'], EltWiseLessLayerResolver.Descriptor)


class EltWiseLessLayerBuilder(EltWiseLayerBuilder):
    def __init__(self):
        super(EltWiseLessLayerBuilder, self).__init__()
        self._op_class = ElementwiseLessOp


class EltWiseLessEqualLayerResolver(EltWiseLayerResolver):
    class Descriptor(EltWiseLayerResolver.Descriptor):
        pass

    def __init__(self):
        super(EltWiseLessEqualLayerResolver, self).__init__('ElementWiseLessEqual', ['LessEqual'],
                                                            EltWiseLessEqualLayerResolver.Descriptor)


class EltWiseLessEqualLayerBuilder(EltWiseLayerBuilder):
    def __init__(self):
        super(EltWiseLessEqualLayerBuilder, self).__init__()
        self._op_class = ElementwiseLessEqualOp


class EltWiseNotEqualLayerResolver(EltWiseLayerResolver):
    class Descriptor(EltWiseLayerResolver.Descriptor):
        pass

    def __init__(self):
        super(EltWiseNotEqualLayerResolver, self).__init__('ElementWiseNotEqual', ['NotEqual'],
                                                           EltWiseNotEqualLayerResolver.Descriptor)


class EltWiseNotEqualLayerBuilder(EltWiseLayerBuilder):
    def __init__(self):
        super(EltWiseNotEqualLayerBuilder, self).__init__()
        self._op_class = ElementwiseNotEqualOp


class EltWiseOrLayerResolver(EltWiseLayerResolver):
    class Descriptor(EltWiseLayerResolver.Descriptor):
        pass

    def __init__(self):
        super(EltWiseOrLayerResolver, self).__init__('ElementWiseOr', ['LogicalOr'],
                                                     EltWiseOrLayerResolver.Descriptor)


class EltWiseOrLayerBuilder(EltWiseLayerBuilder):
    def __init__(self):
        super(EltWiseOrLayerBuilder, self).__init__()
        self._op_class = ElementwiseOrOp


class EltWisePowLayerResolver(EltWiseLayerResolver):
    class Descriptor(EltWiseLayerResolver.Descriptor):
        pass

    def __init__(self):
        super(EltWisePowLayerResolver, self).__init__('ElementWisePow', ['Pow'],
                                                      EltWisePowLayerResolver.Descriptor)


class EltWisePowLayerBuilder(EltWiseLayerBuilder):
    def __init__(self):
        super(EltWisePowLayerBuilder, self).__init__()
        self._op_class = ElementwisePowerOp


class EltWiseSelectLayerResolver(EltWiseLayerResolver):
    class Descriptor(EltWiseLayerResolver.Descriptor):
        pass

    def __init__(self):
        super(EltWiseSelectLayerResolver, self).__init__('ElementWiseSelect', ['Select'],
                                                         EltWiseSelectLayerResolver.Descriptor)

    def resolve_layer(self, graph_matcher, graph_helper):
        descriptors = []

        for match in graph_matcher.match_sequence(self.sequence):
            eltwise_op = match['root']
            descriptor = self._descriptor_class(self._layer_type, str(eltwise_op.name), match.consumed_nodes,
                                                unary=False)
            descriptors.append(descriptor)

        return descriptors


class EltWiseSelectLayerBuilder(LayerBuilder):
    def build_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type ir_graph: converters.common.converter_ir.op_graph.IROpGraph
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: EltWiseSelectLayerResolver.Descriptor
        :rtype: int
        """
        input_names = self.get_input_names(converter_context, descriptor, input_descriptors)
        output_name = descriptor.output_names[0]

        # tf.where can have either 1 input or 3 inputs, for 1 input the Op used is Where
        # Select is used when 3 inputs are provided
        if len(input_descriptors) != 3:
            raise ValueError("Op {} has invalid input length {}:{}".format(descriptor.layer_name, len(input_names),
                                                                        input_names))

        return ir_graph.add(ElementwiseSelectOp(descriptor.layer_name),
                            input_names,
                            output_name)

class EltWiseSumLayerResolver(EltWiseLayerResolver):
    class Descriptor(EltWiseLayerResolver.Descriptor):
        pass

    def __init__(self):
        super(EltWiseSumLayerResolver, self).__init__('ElementWiseSum', ['Add', 'AddV2'],
                                                      EltWiseSumLayerResolver.Descriptor)


class EltWiseSumLayerBuilder(EltWiseLayerBuilder):
    def __init__(self):
        super(EltWiseSumLayerBuilder, self).__init__()
        self._op_class = ElementwiseSumOp


class EltWiseSubLayerResolver(EltWiseLayerResolver):
    class Descriptor(EltWiseLayerResolver.Descriptor):
        pass

    def __init__(self):
        super(EltWiseSubLayerResolver, self).__init__('ElementWiseSub', ['Sub'], EltWiseSubLayerResolver.Descriptor)


class EltWiseSubLayerBuilder(EltWiseLayerBuilder):
    def __init__(self):
        super(EltWiseSubLayerBuilder, self).__init__()
        self._op_class = ElementwiseSubOp


class EltWiseMulLayerResolver(EltWiseLayerResolver):
    class Descriptor(EltWiseLayerResolver.Descriptor):
        pass

    def __init__(self):
        super(EltWiseMulLayerResolver, self).__init__('ElementWiseMul', ['Mul'], EltWiseMulLayerResolver.Descriptor)


class EltWiseMulLayerBuilder(EltWiseLayerBuilder):
    def __init__(self):
        super(EltWiseMulLayerBuilder, self).__init__()
        self._op_class = ElementwiseProductOp


class EltWiseMaxLayerResolver(EltWiseLayerResolver):
    class Descriptor(EltWiseLayerResolver.Descriptor):
        pass

    def __init__(self):
        super(EltWiseMaxLayerResolver, self).__init__('ElementWiseMax', ['Maximum'], EltWiseMaxLayerResolver.Descriptor)


class EltWiseMaxLayerBuilder(EltWiseLayerBuilder):
    def __init__(self):
        super(EltWiseMaxLayerBuilder, self).__init__()
        self._op_class = ElementwiseMaxOp


class EltWiseMinLayerResolver(EltWiseLayerResolver):
    class Descriptor(EltWiseLayerResolver.Descriptor):
        pass

    def __init__(self):
        super(EltWiseMinLayerResolver, self).__init__('ElementWiseMin', ['Minimum'], EltWiseMinLayerResolver.Descriptor)


class EltWiseMinLayerBuilder(EltWiseLayerBuilder):
    def __init__(self):
        super(EltWiseMinLayerBuilder, self).__init__()
        self._op_class = ElementwiseMinOp


class EltWiseDivLayerResolver(EltWiseLayerResolver):
    class Descriptor(EltWiseLayerResolver.Descriptor):
        pass

    def __init__(self):
        super(EltWiseDivLayerResolver, self).__init__('ElementWiseDiv', ['RealDiv'], EltWiseDivLayerResolver.Descriptor)


class EltWiseDivLayerBuilder(EltWiseLayerBuilder):
    def __init__(self):
        super(EltWiseDivLayerBuilder, self).__init__()
        self._op_class = ElementwiseDivOp

    def transform_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):

        constant_input_descriptor = [d for d in input_descriptors if isinstance(d, ConstantLayerResolver.Descriptor)]
        if len(constant_input_descriptor) == 1 and np.all(constant_input_descriptor[0].value == 1):
            descriptor.set_ignored(True)
            constant_input_descriptor[0].set_ignored(True)
