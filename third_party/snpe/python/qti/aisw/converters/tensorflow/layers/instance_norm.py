# =============================================================================
#
#  Copyright (c) 2018-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import numpy as np

from qti.aisw.converters.common.converter_ir.op_adapter import BatchnormOp
from qti.aisw.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from qti.aisw.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    GraphSequence,
    NonConsumableConverterSequenceNode
)


class InstanceNormLayerResolver(LayerResolver, object):
    class Descriptor(LayerDescriptor):
        def __init__(self, name, operations, input_op, shape, gamma, beta, epsilon):
            super(InstanceNormLayerResolver.Descriptor, self).__init__('InstanceNorm', name, operations)
            self.input_op = input_op
            self.shape = shape
            # SNPE runtime algo is y = x * WEIGHT / rms + BIAS
            # While L2 Normalization is y = x / rms
            # That requires WEIGHT = 1.0 and BIAS = 0.0 to mimic L2 Norm in SNPE
            # Shape of weights/biases should be same as the last dimension of input.
            self.weights = gamma
            self.biases = beta
            self.epsilon = epsilon

        def is_input_op(self, op):
            return len(op.inputs) and op.inputs[0].op == self.input_op

        def is_input_tensor(self, op, tensor):
            return tensor == op.inputs[0]

    def __init__(self):
        self.sequence1 = GraphSequence([
            NonConsumableConverterSequenceNode('input', ['?']),
            ConverterSequenceNode('mean', ['Mean']),
            ConverterSequenceNode('StopGradient', ['StopGradient']),
            ConverterSequenceNode('SquaredDifference', ['SquaredDifference']),
            ConverterSequenceNode('variance', ['Mean']),
            ConverterSequenceNode('epsilon', ['?']),
            ConverterSequenceNode('add', ['Add','AddV2']),
            ConverterSequenceNode('Rsqrt', ['Rsqrt']),
            ConverterSequenceNode('gamma', ['?']),
            ConverterSequenceNode('mul', ['Mul']),
            ConverterSequenceNode('mul_1', ['Mul']),
            ConverterSequenceNode('mul_2', ['Mul']),
            ConverterSequenceNode('beta', ['?']),
            ConverterSequenceNode('sub', ['Sub']),
            ConverterSequenceNode('add_1', ['Add','AddV2']),
            NonConsumableConverterSequenceNode('mean_reduction_indices', ['?']),
            NonConsumableConverterSequenceNode('variance_reduction_indices', ['?']),
        ])
        self.sequence1.set_inputs('mean', ['input', 'mean_reduction_indices'])
        self.sequence1.set_inputs('StopGradient', ['mean'])
        self.sequence1.set_inputs('SquaredDifference', ['input', 'StopGradient'])
        self.sequence1.set_inputs('variance', ['SquaredDifference', 'variance_reduction_indices'])
        self.sequence1.set_inputs('add', ['variance', 'epsilon'])
        self.sequence1.set_inputs('Rsqrt', ['add'])
        self.sequence1.set_inputs('mul', ['Rsqrt', 'gamma'])
        self.sequence1.set_inputs('mul_1', ['input', 'mul'])
        self.sequence1.set_inputs('mul_2', ['mean', 'mul'])
        self.sequence1.set_inputs('sub', ['beta', 'mul_2'])
        self.sequence1.set_inputs('add_1', ['mul_1', 'sub'])
        self.sequence1.set_outputs(['add_1'])

        self.sequence2 = GraphSequence([
            NonConsumableConverterSequenceNode('input', ['?']),
            ConverterSequenceNode('mean', ['?']),
            ConverterSequenceNode('StopGradient', ['StopGradient']),
            ConverterSequenceNode('SquaredDifference', ['SquaredDifference']),
            ConverterSequenceNode('variance', ['Mean']),
            ConverterSequenceNode('epsilon', ['?']),
            ConverterSequenceNode('add', ['Add']),
            ConverterSequenceNode('sqrt', ['Sqrt']),
            ConverterSequenceNode('sub', ['Sub']),
            ConverterSequenceNode('real_div', ['RealDiv']),
            NonConsumableConverterSequenceNode('mean_reduction_indices', ['?']),
            NonConsumableConverterSequenceNode('variance_reduction_indices', ['?']),
        ])
        self.sequence2.set_inputs('mean', ['input', 'mean_reduction_indices'])
        self.sequence2.set_inputs('StopGradient', ['mean'])
        self.sequence2.set_inputs('SquaredDifference', ['input', 'StopGradient'])
        self.sequence2.set_inputs('variance', ['SquaredDifference', 'variance_reduction_indices'])
        self.sequence2.set_inputs('add', ['variance', 'epsilon'])
        self.sequence2.set_inputs('sqrt', ['add'])
        self.sequence2.set_inputs('sub', ['input', 'mean'])
        self.sequence2.set_inputs('real_div', ['sub', 'sqrt'])
        self.sequence2.set_outputs(['real_div'])

        self.sequences = [self.sequence1, self.sequence2]

    def is_final_resolution(self):
        return True

    def resolve_layer(self, graph_matcher, graph_helper):
        potential_descriptors = []
        for sequence in self.sequences:
            matches = graph_matcher.match_sequence(sequence)
            for match in matches:
                bn_op = match['SquaredDifference']
                input_op = match['input']
                shape = graph_helper.get_op_output_shape(input_op)
                rank = len(shape)

                mean_reduction_indices_op = match['mean_reduction_indices']
                axes = graph_helper.evaluate_tensor_output(mean_reduction_indices_op.outputs[0])
                axes = [axes] if np.isscalar(axes) else axes.tolist()
                for i in range(len(axes)):
                    axes[i] = int(axes[i])
                    if axes[i] < 0:
                        axes[i] += rank

                if rank != 4 or axes != [1,2]:
                    # This is not InstanceNorm, LayerNorm will handle it
                    continue

                epsilon_op = match['epsilon']
                gamma = np.ones(shape[-1], dtype=np.float32)
                beta = np.zeros(shape[-1], dtype=np.float32)
                if 'gamma' in match:
                    try:
                        gamma = np.broadcast_to(graph_helper.evaluate_tensor_output(match['gamma'].outputs[0]), shape[-1])
                    except:
                        gamma = np.ones(shape[-1], dtype=np.float32)
                if 'beta' in match:
                    try:
                        beta = np.broadcast_to(graph_helper.evaluate_tensor_output(match['beta'].outputs[0]), shape[-1])
                    except:
                        beta = np.zeros(shape[-1], dtype=np.float32)

                consumed_nodes = match.consumed_nodes
                potential_descriptors.append(InstanceNormLayerResolver.Descriptor(str(bn_op.name),
                                                                                  consumed_nodes,
                                                                                  input_op=input_op,
                                                                                  shape=shape,
                                                                                  gamma=gamma,
                                                                                  beta=beta,
                                                                                  epsilon=epsilon_op.get_attr('value').float_val[0]))
        return potential_descriptors


class InstanceNormLayerBuilder(LayerBuilder):
    def build_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type ir_graph: converters.common.converter_ir.op_graph.IROpGraph
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: InstanceNormLayerResolver.Descriptor
        :rtype: int
        """
        input_name = self.get_input_name(converter_context, descriptor, input_descriptors)

        return ir_graph.add(BatchnormOp(descriptor.layer_name,
                                        descriptor.weights,
                                        descriptor.biases,
                                        compute_statistics=True,
                                        use_mu_sigma=True,
                                        across_spatial=True,
                                        epsilon=descriptor.epsilon),
                            input_names=input_name,
                            output_names=descriptor.output_names[0])
