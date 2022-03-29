# =============================================================================
#
#  Copyright (c) 2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import numpy as np
from collections import OrderedDict

from qti.aisw.converters.common.utils import code_to_message
from qti.aisw.converters.common.converter_ir.op_adapter import LayerNormOp, ConstantOp
from qti.aisw.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from qti.aisw.converters.tensorflow.util import ConverterError
from qti.aisw.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    GraphSequence,
    NonConsumableConverterSequenceNode
)


class LayerNormLayerResolver(LayerResolver, object):
    class Descriptor(LayerDescriptor):
        def __init__(self, name, operations, axes, epsilon, input_tensors):
            super(LayerNormLayerResolver.Descriptor, self).__init__('LayerNorm', name, operations)
            self.axes = axes
            self.epsilon = epsilon
            self.input_tensors = input_tensors

        def is_input_tensor(self, op, tensor):
            return tensor == op.inputs[0]

    def __init__(self):
        self.sequence = GraphSequence([
            NonConsumableConverterSequenceNode('input', ['?']),
            ConverterSequenceNode('mean', ['Mean']),
            ConverterSequenceNode('StopGradient', ['StopGradient']),
            ConverterSequenceNode('SquaredDifference', ['SquaredDifference']),
            ConverterSequenceNode('variance', ['Mean']),
            ConverterSequenceNode('epsilon', ['?']),
            ConverterSequenceNode('add_0', ['Add', 'AddV2']),
            ConverterSequenceNode('Rsqrt', ['Rsqrt']),
            ConverterSequenceNode('gamma', ['?']),
            ConverterSequenceNode('mul_0', ['Mul']),
            ConverterSequenceNode('mul_1', ['Mul']),
            ConverterSequenceNode('mul_2', ['Mul']),
            ConverterSequenceNode('beta', ['?']),
            ConverterSequenceNode('sub', ['Sub']),
            ConverterSequenceNode('add_1', ['Add', 'AddV2']),
            NonConsumableConverterSequenceNode('mean_reduction_indices', ['?']),
            NonConsumableConverterSequenceNode('variance_reduction_indices', ['?']),
        ])
        self.sequence.set_inputs('mean', ['input', 'mean_reduction_indices'])
        self.sequence.set_inputs('StopGradient', ['mean'])
        self.sequence.set_inputs('SquaredDifference', ['input','StopGradient'])
        self.sequence.set_inputs('variance', ['SquaredDifference','variance_reduction_indices'])
        self.sequence.set_inputs('add_0', ['variance','epsilon'])
        self.sequence.set_inputs('Rsqrt', ['add_0'])
        self.sequence.set_inputs('mul_0', ['Rsqrt','gamma'])
        self.sequence.set_inputs('mul_1', ['input','mul_0'])
        self.sequence.set_inputs('mul_2', ['mean','mul_0'])
        self.sequence.set_inputs('sub', ['beta','mul_2'])
        self.sequence.set_inputs('add_1', ['mul_1','sub'])
        self.sequence.set_outputs(['add_1'])

        # No StopGradient present in sequence
        self.sequence_2 = GraphSequence([
            NonConsumableConverterSequenceNode('input', ['?']),
            ConverterSequenceNode('mean_reduction_indices', ['Const']),
            ConverterSequenceNode('mean', ['Mean']),
            ConverterSequenceNode('SquaredDifference', ['SquaredDifference']),
            ConverterSequenceNode('variance_reduction_indices', ['Const']),
            ConverterSequenceNode('variance', ['Mean']),
            ConverterSequenceNode('epsilon', ['Const']),
            ConverterSequenceNode('add_0', ['Add', 'AddV2']),
            ConverterSequenceNode('Rsqrt', ['Rsqrt']),
            ConverterSequenceNode('gamma', ['Const']),
            ConverterSequenceNode('mul_0', ['Mul']),
            ConverterSequenceNode('mul_1', ['Mul']),
            ConverterSequenceNode('mul_2', ['Mul']),
            ConverterSequenceNode('beta', ['Const']),
            ConverterSequenceNode('sub', ['Sub']),
            ConverterSequenceNode('add_1', ['Add', 'AddV2'])
        ])
        self.sequence_2.set_inputs('mean', ['input', 'mean_reduction_indices'])
        self.sequence_2.set_inputs('SquaredDifference', ['input', 'mean'])
        self.sequence_2.set_inputs('variance', ['SquaredDifference', 'variance_reduction_indices'])
        self.sequence_2.set_inputs('add_0', ['variance', 'epsilon'])
        self.sequence_2.set_inputs('Rsqrt', ['add_0'])
        self.sequence_2.set_inputs('mul_0', ['Rsqrt', 'gamma'])
        self.sequence_2.set_inputs('mul_1', ['input', 'mul_0'])
        self.sequence_2.set_inputs('mul_2', ['mean', 'mul_0'])
        self.sequence_2.set_inputs('sub', ['beta', 'mul_2'])
        self.sequence_2.set_inputs('add_1', ['mul_1', 'sub'])
        self.sequence_2.set_outputs(['add_1'])

        # SquaredDifference has been replaced with Sub -> Pow in this sequence
        self.sequence_3 = GraphSequence([
            NonConsumableConverterSequenceNode('input', ['?']),
            ConverterSequenceNode('mean_reduction_indices', ['Const']),
            ConverterSequenceNode('mean', ['Mean']),
            ConverterSequenceNode('difference', ['Sub']),
            ConverterSequenceNode('exponent', ['Const']),
            ConverterSequenceNode('square', ['Pow']),
            ConverterSequenceNode('variance_reduction_indices', ['Const']),
            ConverterSequenceNode('variance', ['Mean']),
            ConverterSequenceNode('epsilon', ['Const']),
            ConverterSequenceNode('add_0', ['Add', 'AddV2']),
            ConverterSequenceNode('Rsqrt', ['Rsqrt']),
            ConverterSequenceNode('gamma', ['Const']),
            ConverterSequenceNode('mul_0', ['Mul']),
            ConverterSequenceNode('mul_1', ['Mul']),
            ConverterSequenceNode('mul_2', ['Mul']),
            ConverterSequenceNode('beta', ['Const']),
            ConverterSequenceNode('sub', ['Sub']),
            ConverterSequenceNode('add_1', ['Add', 'AddV2'])
        ])
        self.sequence_3.set_inputs('mean', ['input', 'mean_reduction_indices'])
        self.sequence_3.set_inputs('difference', ['input', 'mean'])
        self.sequence_3.set_inputs('square', ['difference', 'exponent'])
        self.sequence_3.set_inputs('variance', ['square', 'variance_reduction_indices'])
        self.sequence_3.set_inputs('add_0', ['variance', 'epsilon'])
        self.sequence_3.set_inputs('Rsqrt', ['add_0'])
        self.sequence_3.set_inputs('mul_0', ['Rsqrt', 'gamma'])
        self.sequence_3.set_inputs('mul_1', ['input', 'mul_0'])
        self.sequence_3.set_inputs('mul_2', ['mean', 'mul_0'])
        self.sequence_3.set_inputs('sub', ['beta', 'mul_2'])
        self.sequence_3.set_inputs('add_1', ['mul_1', 'sub'])
        self.sequence_3.set_outputs(['add_1'])

        self.sequences = [self.sequence, self.sequence_2, self.sequence_3]

    def is_final_resolution(self):
        return True

    def resolve_layer(self, graph_matcher, graph_helper):
        potential_descriptors = []
        for sequence in self.sequences:
            matches = graph_matcher.match_sequence(sequence)
            for match in matches:
                input_op = match['mean']
                shape = graph_helper.get_op_output_shape(input_op)
                rank = len(shape)

                beta_op = match['beta']
                if beta_op.type not in ['Identity', 'Const', 'Switch']:
                    raise ConverterError(code_to_message.get_error_message('ERROR_TF_BATCHNORM_RESOLVE_BETA'))
                beta = graph_helper.evaluate_tensor_output(beta_op.outputs[0])

                gamma_op = match['gamma']
                if gamma_op.type not in ['Identity', 'Const', 'Switch']:
                    raise ConverterError(code_to_message.get_error_message('ERROR_TF_LAYERNORM_RESOLVE_GAMMA'))
                gamma = graph_helper.evaluate_tensor_output(gamma_op.outputs[0])

                epsilon_op = match['epsilon']
                if epsilon_op.type not in ['Identity', 'Const', 'Switch']:
                    raise ConverterError(code_to_message.get_error_message('ERROR_TF_BATCHNORM_RESOLVE_EPSILON'))
                epsilon = graph_helper.evaluate_tensor_output(epsilon_op.outputs[0])

                axes_op = match['mean_reduction_indices']
                if axes_op.type not in ['Identity', 'Const', 'Switch']:
                    raise ConverterError(code_to_message.get_error_message('ERROR_TF_LAYERNORM_RESOLVE_GAMMA'))
                axes = graph_helper.evaluate_tensor_output(axes_op.outputs[0])
                axes = [axes] if np.isscalar(axes) else axes.tolist()
                for i in range(len(axes)):
                    axes[i] = int(axes[i])
                    if axes[i] < 0:
                        axes[i] += rank

                if rank == 4 and axes == [1,2]:
                    # Let InstanceNorm handle this case
                    continue

                consumed_nodes = match.consumed_nodes
                input_tensors = OrderedDict()
                input_tensors["gamma"] = gamma
                input_tensors["beta"] = beta

                potential_descriptors.append(LayerNormLayerResolver.Descriptor(str(input_op.name),
                                                                               consumed_nodes,
                                                                               axes=axes,
                                                                               epsilon=epsilon,
                                                                               input_tensors=input_tensors))
        return potential_descriptors


class LayerNormLayerBuilder(LayerBuilder):
    def build_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type ir_graph: converters.common.converter_ir.op_graph.IROpGraph
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: LayerNormLayerResolver.Descriptor
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

        input_names = [input_name]

        # Add all the constant inputs
        for k,v in descriptor.input_tensors.items():
            if not isinstance(v, np.ndarray):
                v = np.atleast_1d(v)

            input_names.append(str(descriptor.layer_name)+"_"+k)
            ir_graph.add(ConstantOp(input_names[-1],
                                    v,
                                    quantizable=True),
                         [],
                         input_names[-1])

        # Add the layer norm op
        return ir_graph.add(LayerNormOp(descriptor.layer_name,
                                        axes=descriptor.axes,
                                        epsilon=descriptor.epsilon),
                            input_names=input_names,
                            output_names=descriptor.output_names[0])
