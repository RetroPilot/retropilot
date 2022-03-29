# =============================================================================
#
#  Copyright (c) 2019-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import numpy as np
from qti.aisw.converters.common.utils import code_to_message
from qti.aisw.converters.common.converter_ir.op_adapter import L2NormOp
from qti.aisw.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from qti.aisw.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    GraphSequence,
    NonConsumableConverterSequenceNode
)
from qti.aisw.converters.tensorflow.util import ConverterError


class L2NormLayerResolver(LayerResolver, object):
    class Descriptor(LayerDescriptor):
        def __init__(self, name, operations, **kwargs):
            super(L2NormLayerResolver.Descriptor, self).__init__('L2Norm', name, operations)
            self.shape = kwargs.get('shape')
            self.epsilon = kwargs.get('epsilon')
            self.axis = kwargs.get('axis')

    def __init__(self):
        # Graph topology of tf.nn.l2_normalize
        self.sequence = GraphSequence([
            NonConsumableConverterSequenceNode('input', ['?']),
            ConverterSequenceNode('a', ['Square']),
            ConverterSequenceNode('axis', ['Const', 'Identity']),
            ConverterSequenceNode('b', ['Sum']),
            ConverterSequenceNode('epsilon', ['Const', 'Identity']),
            ConverterSequenceNode('c', ['Maximum']),
            ConverterSequenceNode('d', ['Rsqrt']),
            ConverterSequenceNode('e', ['Mul'])
        ])
        self.sequence.set_inputs('a', ['input'])
        self.sequence.set_inputs('b', ['a', 'axis'])
        self.sequence.set_inputs('c', ['b', 'epsilon'])
        self.sequence.set_inputs('d', ['c'])
        self.sequence.set_inputs('e', ['d', 'input'])
        self.sequence.set_outputs(['e'])

    # For now, elementwise resolver cannot work with epsilon node.
    # Will meet error "ElementWise resolver must implement broadcast method.".
    def is_final_resolution(self):
        return True

    def resolve_layer(self, graph_matcher, graph_helper):
        matches = graph_matcher.match_sequence(self.sequence)
        potential_descriptors = []
        for match in matches:
            l2n_op = match['a']
            epsilon_op = match['epsilon']
            axis_op = match['axis']
            input_op = match['input']

            shape = graph_helper.get_op_output_shape(input_op)
            if epsilon_op.type not in ['Identity', 'Const']:
                raise ConverterError(code_to_message.get_error_message('ERROR_TF_L2NORM_RESOLVE_EPSILON'))
            if axis_op.type not in ['Identity', 'Const']:
                raise ConverterError(code_to_message.get_error_message('ERROR_TF_L2NORM_RESOLVE_AXIS'))
            epsilon = graph_helper.evaluate_tensor_output(epsilon_op.outputs[0])
            axis = graph_helper.evaluate_tensor_output(axis_op.outputs[0])
            consumed_nodes = match.consumed_nodes
            potential_descriptors.append(L2NormLayerResolver.Descriptor(str(l2n_op.name),
                                                                        consumed_nodes,
                                                                        shape=shape,
                                                                        epsilon=epsilon,
                                                                        axis=axis))
        return potential_descriptors


class L2NormLayerBuilder(LayerBuilder):
    def build_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type ir_graph: converters.common.converter_ir.op_graph.IROpGraph
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: InstanceNormRMSLayerResolver.Descriptor
        :rtype: int
        """
        input_name = self.get_input_name(converter_context, descriptor, input_descriptors)
        if isinstance(descriptor.axis, np.ndarray):
            return ir_graph.add(L2NormOp(descriptor.layer_name,
                                         axis=descriptor.axis,
                                         epsilon=descriptor.epsilon),
                                input_names=input_name,
                                output_names=descriptor.output_names[0])
        elif isinstance(descriptor.axis, np.int32):
            return ir_graph.add(L2NormOp(descriptor.layer_name,
                                         axis=descriptor.axis,
                                         epsilon=descriptor.epsilon),
                                input_names=input_name,
                                output_names=descriptor.output_names[0])
        else:
            raise ConverterError(code_to_message.get_error_message('ERROR_TF_L2NORM_AXIS_TYPE'))
