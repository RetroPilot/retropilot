# =============================================================================
#
#  Copyright (c) 2015-2020 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import numpy as np

from qti.aisw.converters.common.utils import code_to_message
from qti.aisw.converters.common.converter_ir.op_adapter import PreluOp
from qti.aisw.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from qti.aisw.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    GraphSequence,
    NonConsumableConverterSequenceNode
)
from qti.aisw.converters.tensorflow.util import ConverterError


class PReLuLayerResolver(LayerResolver, object):
    class Descriptor(LayerDescriptor):
        def __init__(self, name, operations, coefficients, output_names):
            super(PReLuLayerResolver.Descriptor, self).__init__('PReLU', name, operations,
                                                                output_names=output_names)
            self.coefficients = coefficients

    def __init__(self):
        sequence_prelu = GraphSequence([
            ConverterSequenceNode('a', ['Relu']),
            ConverterSequenceNode('b', ['Abs']),
            ConverterSequenceNode('c', ['Sub']),
            ConverterSequenceNode('d', ['Mul']),
            ConverterSequenceNode('e', ['Mul']),
            ConverterSequenceNode('f', ['Add']),  # output
            ConverterSequenceNode('unknown', ['?']),
            NonConsumableConverterSequenceNode('alphas', ['?']),
            NonConsumableConverterSequenceNode('inputs', ['?'])
        ])
        sequence_prelu.set_inputs('a', ['inputs'])
        sequence_prelu.set_inputs('b', ['inputs'])
        sequence_prelu.set_inputs('c', ['inputs', 'b'])
        sequence_prelu.set_inputs('d', ['alphas', 'c'])
        sequence_prelu.set_inputs('e', ['d', 'unknown'])
        sequence_prelu.set_inputs('f', ['a', 'e'])
        sequence_prelu.set_outputs(['f'])

        sequence_prelu_negative_alpha = GraphSequence([
            ConverterSequenceNode('a', ['Relu']),
            ConverterSequenceNode('b', ['Neg']),
            ConverterSequenceNode('c', ['Neg']),
            ConverterSequenceNode('d', ['Relu']),
            ConverterSequenceNode('e', ['Mul']),
            ConverterSequenceNode('f', ['Add', 'AddV2']),  # output
            NonConsumableConverterSequenceNode('alphas', ['?']),
            NonConsumableConverterSequenceNode('inputs', ['?'])
        ])
        sequence_prelu_negative_alpha.set_inputs('a', ['inputs'])
        sequence_prelu_negative_alpha.set_inputs('b', ['inputs'])
        sequence_prelu_negative_alpha.set_inputs('c', ['alphas'])
        sequence_prelu_negative_alpha.set_inputs('d', ['b'])
        sequence_prelu_negative_alpha.set_inputs('e', ['d', 'c'])
        sequence_prelu_negative_alpha.set_inputs('f', ['a', 'e'])
        sequence_prelu_negative_alpha.set_outputs(['f'])

        sequence_prelu_negative_relu = GraphSequence([
            ConverterSequenceNode('relu_pos', ['Relu']),
            ConverterSequenceNode('neg_1', ['Neg']),
            ConverterSequenceNode('neg_2', ['Neg']),
            ConverterSequenceNode('relu_neg', ['Relu']),
            ConverterSequenceNode('mul', ['Mul']),
            ConverterSequenceNode('f', ['Add']),  # output
            NonConsumableConverterSequenceNode('alphas', ['?']),
            NonConsumableConverterSequenceNode('inputs', ['?'])
        ])
        sequence_prelu_negative_relu.set_inputs('relu_pos', ['inputs'])
        sequence_prelu_negative_relu.set_inputs('neg_1', ['inputs'])
        sequence_prelu_negative_relu.set_inputs('relu_neg', ['neg_1'])
        sequence_prelu_negative_relu.set_inputs('neg_2', ['relu_neg'])
        sequence_prelu_negative_relu.set_inputs('mul', ['neg_2', 'alphas'])
        sequence_prelu_negative_relu.set_inputs('f', ['relu_pos', 'mul'])
        sequence_prelu_negative_relu.set_outputs(['f'])

        sequence_prelu_maximum = GraphSequence([
            ConverterSequenceNode('maximum', ['Maximum']),
            ConverterSequenceNode('max_const', ['Const']),
            ConverterSequenceNode('abs', ['Abs']),
            ConverterSequenceNode('sub', ['Sub']),
            ConverterSequenceNode('mul_1', ['Mul']),
            ConverterSequenceNode('mul_2', ['Mul']),
            ConverterSequenceNode('f', ['Add']),  # output
            ConverterSequenceNode('unknown', ['?']),
            NonConsumableConverterSequenceNode('alphas', ['?']),
            NonConsumableConverterSequenceNode('inputs', ['?'])
        ])

        sequence_prelu_maximum.set_inputs('maximum', ['inputs', 'max_const'])
        sequence_prelu_maximum.set_inputs('abs', ['inputs'])
        sequence_prelu_maximum.set_inputs('sub', ['inputs', 'abs'])
        sequence_prelu_maximum.set_inputs('mul_1', ['alphas', 'sub'])
        sequence_prelu_maximum.set_inputs('mul_2', ['mul_1', 'unknown'])
        sequence_prelu_maximum.set_inputs('f', ['maximum', 'mul_2'])
        sequence_prelu_maximum.set_outputs(['f'])

        self.sequences = [sequence_prelu, sequence_prelu_negative_alpha, sequence_prelu_negative_relu, sequence_prelu_maximum]

    def resolve_layer(self, graph_matcher, graph_helper):
        potential_descriptors = []
        for sequence in self.sequences:
            matches = graph_matcher.match_sequence(sequence)
            for match in matches:
                coefficients = match['alphas']
                add_op = match['f']
                if not graph_helper.check_tensor_const_origin(coefficients.outputs[0])[0]:
                    raise ConverterError(code_to_message.get_error_message('ERROR_TF_RESOLVE_PRELU_COEFF'))

                output_op_nodes_names = [str(match[node.identifier].outputs[0].name) for node in sequence.output_nodes]
                consumed_nodes = match.consumed_nodes

                coefficients = graph_helper.evaluate_tensor_output(coefficients.outputs[0])

                add_op_inputs_ = graph_helper.get_op_input_tensors(add_op, ('?', '?'))
                add_op_input_shape = graph_helper.get_tensor_output_shape(add_op_inputs_[0])

                rank_diff = len(coefficients.shape) - len(add_op_input_shape)
                if rank_diff < 0:
                    # Prepending 1's to coefficients shape and then broadcasting to match input rank
                    coefficients_shape = [1] * abs(rank_diff) + list(coefficients.shape)
                    coefficients = np.broadcast_to(coefficients, coefficients_shape)

                potential_descriptors.append(
                    PReLuLayerResolver.Descriptor(str(add_op.name), consumed_nodes,
                                                  coefficients,
                                                  output_names=output_op_nodes_names))
        return potential_descriptors


class LeakyReLuLayerResolver(PReLuLayerResolver, object):
    class Descriptor(LayerDescriptor):
        def __init__(self, name, operations, coefficients):
            super(LeakyReLuLayerResolver.Descriptor, self).__init__('LeakyRelu', name,
                                                                    operations)

            self.coefficients = coefficients

    def __init__(self):
        # Graph pattern of tf.nn.leaky_relu prior to LeakyRelu op
        sequence_old_leaky_relu = GraphSequence([
            NonConsumableConverterSequenceNode('input', ['?']),
            ConverterSequenceNode('negative_slope', ['Const']),
            ConverterSequenceNode('mul', ['Mul']),
            ConverterSequenceNode('root', ['Maximum']),
        ])

        sequence_old_leaky_relu.set_inputs('mul', ['input', 'negative_slope'])
        sequence_old_leaky_relu.set_inputs('root', ['input', 'mul'])
        sequence_old_leaky_relu.set_outputs(['root'])

        sequence_old_2_leaky_relu = GraphSequence([
            NonConsumableConverterSequenceNode('input', ['?']),
            # 2 negative slope value case where alpha for leakyRelu will be their difference
            ConverterSequenceNode('negative_slope', ['?']),
            ConverterSequenceNode('negative_slope2', ['?']),
            ConverterSequenceNode('abs', ['Abs']),
            ConverterSequenceNode('mul1', ['Mul']),
            ConverterSequenceNode('mul2', ['Mul']),
            ConverterSequenceNode('root', ['Add'])  # output
        ])
        sequence_old_2_leaky_relu.set_inputs('abs', ['input'])
        sequence_old_2_leaky_relu.set_inputs('mul1', ['abs', 'negative_slope'])
        sequence_old_2_leaky_relu.set_inputs('mul2', ['input', 'negative_slope2'])
        sequence_old_2_leaky_relu.set_inputs('root', ['mul1', 'mul2'])
        sequence_old_2_leaky_relu.set_outputs(['root'])

        sequence_leaky_relu = GraphSequence([ConverterSequenceNode('root', ['LeakyReLU'])])
        sequence_leaky_relu.set_outputs(['root'])

        self.sequences = [sequence_old_leaky_relu, sequence_old_2_leaky_relu, sequence_leaky_relu]

    def resolve_layer(self, graph_matcher, graph_helper):
        potential_descriptors = []
        for sequence in self.sequences:
            matches = graph_matcher.match_sequence(sequence)
            for match in matches:
                leaky_relu_op = match['root']
                consumed_nodes = match.consumed_nodes

                if 'negative_slope' in match:
                    negative_slope_op = match['negative_slope']
                    negative_slope = graph_helper.evaluate_tensor_output(negative_slope_op.outputs[0])
                    if 'negative_slope2' in match:
                        negative_slope_op_2 = match['negative_slope2']
                        negative_slope_op_2 = graph_helper.evaluate_tensor_output(negative_slope_op_2.outputs[0])
                        negative_slope = negative_slope_op_2 - negative_slope
                else:
                    negative_slope = leaky_relu_op.node_def.attr['alpha'].f

                coefficients = negative_slope * np.ones(graph_helper.get_op_output_shape(leaky_relu_op)[-1], dtype=np.float32)
                potential_descriptors.append(
                    LeakyReLuLayerResolver.Descriptor(str(leaky_relu_op.name), consumed_nodes,
                                                      coefficients))
        return potential_descriptors


class PReLuLayerBuilder(LayerBuilder):
    def build_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type ir_graph: converters.common.converter_ir.op_graph.IROpGraph
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: ReluLayerResolver.Descriptor
        :rtype: int
        """
        if len(input_descriptors) > 1:
            non_constant_input_descriptors = []
            for input_descriptor in input_descriptors:
                if input_descriptor.layer_type != 'Constant':
                    non_constant_input_descriptors.append(input_descriptor)

            if len(non_constant_input_descriptors) == 1:
                input_name = self.get_input_name(converter_context, descriptor, non_constant_input_descriptors)
            else:
                raise ConverterError(code_to_message.get_error_message('ERROR_TF_LAYER_INPUT_COUNT_ERROR')
                                     (descriptor.layer_type, 1, len(input_descriptors)))

        else:
            input_name = self.get_input_name(converter_context, descriptor, input_descriptors)

        output_name = descriptor.output_names[0]
        return ir_graph.add(PreluOp(name=descriptor.layer_name,
                                    coeff=descriptor.coefficients),
                            input_names=input_name,
                            output_names=output_name)
