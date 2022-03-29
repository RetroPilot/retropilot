# =============================================================================
#
#  Copyright (c) 2015-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import numpy as np

from qti.aisw.converters.common.utils import code_to_message
from qti.aisw.converters.common.converter_ir.op_adapter import BatchnormOp
from qti.aisw.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from qti.aisw.converters.tensorflow.util import GraphHelper
from qti.aisw.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    GraphSequence,
    NonConsumableConverterSequenceNode
)
from qti.aisw.converters.tensorflow.util import ConverterError


class BatchNormLayerResolver(LayerResolver, object):
    class Descriptor(LayerDescriptor):
        def __init__(self, name, operations, bn_mul_op, bn_folded=False, pre_calculated=False, *args, **kwargs):
            super(BatchNormLayerResolver.Descriptor, self).__init__('BatchNormalization', name, operations,
                                                                    output_names=kwargs.get('output_names', None))
            self.bn_mul_op = bn_mul_op
            self.bn_folded = bn_folded
            self.pre_calculated = pre_calculated
            if self.pre_calculated:
                self.weights = kwargs.get('weights')
                self.biases = kwargs.get('biases')
                self.scale = np.array([])
                self.beta = np.array([])
            else:
                mean = kwargs.get('mean')
                variance = kwargs.get('variance')
                epsilon = kwargs.get('epsilon')
                self.scale = kwargs.get('scale')
                self.beta = kwargs.get('beta')
                if len(variance) == 0:
                    variance = np.ones(self.scale.shape, dtype=np.float32)
                if len(mean) == 0:
                    mean = np.zeros(self.scale.shape, dtype=np.float32)
                stddev = 1 / np.sqrt(variance + epsilon)
                scaled_stddev = stddev * self.scale
                scaled_variance = variance * scaled_stddev
                scaled_mean = mean * scaled_stddev
                self.weights = scaled_stddev
                self.biases = (-1 * scaled_mean) + self.beta

        def is_input_op(self, op):
            return op == self.bn_mul_op

        def is_input_tensor(self, op, tensor):
            if "Const" in [tensor.op.type, GraphHelper.get_none_identity_input(tensor)[0].op.type]:
                return False

            # mark quant node as input since its needed for retrieving encodings, the fakequant translator
            # will handle the proper ignore
            if GraphHelper.get_none_identity_input(tensor)[0].op.type == "FakeQuantWithMinMaxVars":
                return True

            # This is hit when topology is resolved second time after transform and fakequant is marked ignored
            # Hence, now strip any identity/FakeQuant layer to determine if const inputs for weights/biases.
            if "Const" == GraphHelper.get_stripped_input(tensor, ["FakeQuantWithMinMaxVars", "Identity"]).op.type:
                return False

            return True

    def resolve_layer(self, graph_matcher, graph_helper):
        raise ConverterError(code_to_message.get_error_message('ERROR_TF_GENERAL_ABSTRACT_CLASS_MUST_BE_INHERITED'))


class BatchNormWithEltwiseLayerResolver(BatchNormLayerResolver):

    @staticmethod
    def _match_sequence(graph_matcher, graph_helper, sequence):
        descriptors = []
        matches = []
        # run graph matching for both unfolded/folded bn sequences
        # non-folded
        sequence.set_inputs('bn_output', ['mul_weights', 'sub'])
        sequence.set_outputs(['bn_output'])
        matches.extend(graph_matcher.match_sequence(sequence))

        # folded followed by FakeQuant
        sequence.clear_outputs()
        sequence.set_inputs('bn_output_folded_quant', ["mul_weights", "min", "max"])
        sequence.set_outputs(['bn_output_folded_quant', 'sub'])
        matches.extend(graph_matcher.match_sequence(sequence))

        # folded followed by Convolution
        sequence.clear_outputs()
        sequence.set_inputs('bn_output_folded_conv', ["mul_weights", "kernel"])
        sequence.set_outputs(['bn_output_folded_conv', 'sub'])
        matches.extend(graph_matcher.match_sequence(sequence))
        for match in matches:
            variance_op = match['variance']
            epsilon_op = match['epsilon']

            if variance_op.type not in ['Identity', 'Const', 'Switch']:
                raise ConverterError(code_to_message.get_error_message('ERROR_TF_BATCHNORM_RESOLVE_VARIANCE'))
            variance = graph_helper.evaluate_tensor_output(variance_op.outputs[0])
            if epsilon_op.type not in ['Identity', 'Const', 'Switch']:
                raise ConverterError(code_to_message.get_error_message('ERROR_TF_BATCHNORM_RESOLVE_EPSILON'))
            epsilon = graph_helper.evaluate_tensor_output(epsilon_op.outputs[0])

            mean_op = match['mean']
            if mean_op.type not in ['Identity', 'Const', 'Switch']:
                raise ConverterError(code_to_message.get_error_message('ERROR_TF_BATCHNORM_RESOLVE_MEAN'))
            mean = graph_helper.evaluate_tensor_output(mean_op.outputs[0])

            if 'scale' in match:
                scale_op = match['scale']
                if scale_op.type not in ['Identity', 'Const', 'Fill', 'Switch']:
                    raise ConverterError(code_to_message.get_error_message('ERROR_TF_BATCHNORM_RESOLVE_SCALE'))
                scale = graph_helper.evaluate_tensor_output(scale_op.outputs[0])
            else:
                scale = np.ones(shape=mean.shape, dtype=np.float32)

            beta_op = match['beta']
            if beta_op.type not in ['Identity', 'Const', 'Switch']:
                raise ConverterError(code_to_message.get_error_message('ERROR_TF_BATCHNORM_RESOLVE_BETA'))
            beta = graph_helper.evaluate_tensor_output(beta_op.outputs[0])

            # determine if this layer is folded
            bn_folded = False
            bn_mul_op = match['mul_weights']
            consumers_of_bn_mul = bn_mul_op.outputs[0].consumers()
            if len(consumers_of_bn_mul) == 1 and \
                    (consumers_of_bn_mul[0].type == "Add" or consumers_of_bn_mul[0].type == "AddV2"):
                # this is the expected sequence for non-folded bn node so use the output of this layer.
                output_op_nodes_names = [str(consumers_of_bn_mul[0].outputs[0].name)]
            else:
                bn_folded = True
                output_op_nodes_names = [str(bn_mul_op.outputs[0].name)]
            descriptors.append(
                BatchNormLayerResolver.Descriptor(str(bn_mul_op.name),
                                                  match.consumed_nodes,
                                                  bn_mul_op=bn_mul_op,
                                                  bn_folded=bn_folded,
                                                  mean=mean,
                                                  variance=variance,
                                                  epsilon=epsilon,
                                                  scale=scale,
                                                  beta=beta,
                                                  output_names=output_op_nodes_names))
        return descriptors

    def resolve_layer(self, graph_matcher, graph_helper):
        potential_descriptors = []
        unscaled_bn_sequence = GraphSequence([
            ConverterSequenceNode('add', ['Add', 'AddV2']),
            ConverterSequenceNode('rsqrt', ['Rsqrt']),
            ConverterSequenceNode('mul_weights', ['Mul']),
            ConverterSequenceNode('mul_mean', ['Mul']),
            ConverterSequenceNode('sub', ['Sub']),
            ConverterSequenceNode('bn_output', ['Add', 'AddV2']),
            NonConsumableConverterSequenceNode('bn_output_folded_quant', ['FakeQuantWithMinMaxVars']),
            NonConsumableConverterSequenceNode('bn_output_folded_conv', ['Conv2D', 'DepthwiseConv2dNative']),
            NonConsumableConverterSequenceNode('weights', ['?']),
            ConverterSequenceNode('mean', ['?']),
            ConverterSequenceNode('beta', ['?']),
            ConverterSequenceNode('variance', ['?']),
            ConverterSequenceNode('epsilon', ['?']),
            NonConsumableConverterSequenceNode('min', ['?']),
            NonConsumableConverterSequenceNode('max', ['?']),
            NonConsumableConverterSequenceNode('kernel', ['?'])
        ])
        unscaled_bn_sequence.set_inputs('add', ['variance', 'epsilon'])
        unscaled_bn_sequence.set_inputs('rsqrt', ['add'])
        unscaled_bn_sequence.set_inputs('mul_weights', ['rsqrt', 'weights'])
        unscaled_bn_sequence.set_inputs('mul_mean', ['rsqrt', 'mean'])
        unscaled_bn_sequence.set_inputs('sub', ['mul_mean', 'beta'])
        potential_descriptors.extend(self._match_sequence(graph_matcher, graph_helper, unscaled_bn_sequence))

        scaled_bn_sequence = GraphSequence([
            ConverterSequenceNode('add', ['Add', 'AddV2']),
            ConverterSequenceNode('rsqrt', ['Rsqrt']),
            ConverterSequenceNode('mul_scale', ['Mul']),
            ConverterSequenceNode('mul_mean', ['Mul']),
            ConverterSequenceNode('mul_weights', ['Mul']),
            ConverterSequenceNode('sub', ['Sub']),
            ConverterSequenceNode('scale', ['?']),
            ConverterSequenceNode('bn_output', ['Add', 'AddV2']),
            NonConsumableConverterSequenceNode('bn_output_folded_quant', ['FakeQuantWithMinMaxVars']),
            NonConsumableConverterSequenceNode('bn_output_folded_conv', ['Conv2D', 'DepthwiseConv2dNative']),
            NonConsumableConverterSequenceNode('weights', ['?']),
            ConverterSequenceNode('mean', ['?']),
            ConverterSequenceNode('beta', ['?']),
            ConverterSequenceNode('variance', ['?']),
            ConverterSequenceNode('epsilon', ['?']),
            NonConsumableConverterSequenceNode('min', ['?']),
            NonConsumableConverterSequenceNode('max', ['?']),
            NonConsumableConverterSequenceNode('kernel', ['?'])
        ])
        scaled_bn_sequence.set_inputs('add', ['variance', 'epsilon'])
        scaled_bn_sequence.set_inputs('rsqrt', ['add'])
        scaled_bn_sequence.set_inputs('mul_scale', ['rsqrt', 'scale'])
        scaled_bn_sequence.set_inputs('mul_weights', ['mul_scale', 'weights'])
        scaled_bn_sequence.set_inputs('mul_mean', ['mul_scale', 'mean'])
        scaled_bn_sequence.set_inputs('sub', ['mul_mean', 'beta'])
        potential_descriptors.extend(self._match_sequence(graph_matcher, graph_helper, scaled_bn_sequence))

        scaled_reshape_bn_sequence = GraphSequence([
            ConverterSequenceNode('add', ['Add', 'AddV2']),
            ConverterSequenceNode('rsqrt', ['Rsqrt']),
            ConverterSequenceNode('mul_scale', ['Mul']),
            ConverterSequenceNode('mul_mean', ['Mul']),
            ConverterSequenceNode('mul_weights', ['Mul']),
            ConverterSequenceNode('sub', ['Sub']),
            ConverterSequenceNode('reshape', ['Reshape']),
            ConverterSequenceNode('mean', ['?']),
            ConverterSequenceNode('beta', ['?']),
            ConverterSequenceNode('variance', ['?']),
            ConverterSequenceNode('epsilon', ['?']),
            ConverterSequenceNode('scale', ['?']),
            ConverterSequenceNode('bn_output', ['Add', 'AddV2']),
            NonConsumableConverterSequenceNode('bn_output_folded_quant', ['FakeQuantWithMinMaxVars']),
            NonConsumableConverterSequenceNode('bn_output_folded_conv', ['Conv2D', 'DepthwiseConv2dNative']),
            NonConsumableConverterSequenceNode('weights', ['?']),
            NonConsumableConverterSequenceNode('min', ['?']),
            NonConsumableConverterSequenceNode('max', ['?']),
            NonConsumableConverterSequenceNode('kernel', ['?']),
            ConverterSequenceNode('shape', ['?']),
        ])
        scaled_reshape_bn_sequence.set_inputs('add', ['variance', 'epsilon'])
        scaled_reshape_bn_sequence.set_inputs('rsqrt', ['add'])
        scaled_reshape_bn_sequence.set_inputs('mul_scale', ['rsqrt', 'scale'])
        scaled_reshape_bn_sequence.set_inputs('reshape', ['mul_scale', 'shape'])
        scaled_reshape_bn_sequence.set_inputs('mul_weights', ['weights', 'reshape'])
        scaled_reshape_bn_sequence.set_inputs('mul_mean', ['mul_scale', 'mean'])
        scaled_reshape_bn_sequence.set_inputs('sub', ['mul_mean', 'beta'])
        potential_descriptors.extend(self._match_sequence(graph_matcher, graph_helper, scaled_reshape_bn_sequence))

        return potential_descriptors


class GenericBatchNormLayerResolver(BatchNormLayerResolver):
    class Descriptor(BatchNormLayerResolver.Descriptor):
        pass

    def __init__(self):
        self.sequence = GraphSequence([
            NonConsumableConverterSequenceNode('inputs', ['?']),
            ConverterSequenceNode('mul', ['Mul']),
            ConverterSequenceNode('add', ['Add', 'AddV2']),
            NonConsumableConverterSequenceNode('weights', ['?']),
            NonConsumableConverterSequenceNode('biases', ['?'])
        ])
        self.sequence.set_inputs('mul', ['inputs', 'weights'])
        self.sequence.set_inputs('add', ['mul', 'biases'])
        self.sequence.set_outputs(['add'])

        self.sequence_1 = GraphSequence([
            NonConsumableConverterSequenceNode('inputs', ['?']),
            ConverterSequenceNode('mul', ['Mul']),
            ConverterSequenceNode('add', ['Add', 'AddV2']),
            NonConsumableConverterSequenceNode('weights', ['?']),
            NonConsumableConverterSequenceNode('biases', ['?']),
            NonConsumableConverterSequenceNode('mul_fake_quant', ['FakeQuantWithMinMaxVars']),
            NonConsumableConverterSequenceNode('mul_min', ['?']),
            NonConsumableConverterSequenceNode('mul_max', ['?'])
        ])
        self.sequence_1.set_inputs('mul', ['inputs', 'weights'])
        self.sequence_1.set_inputs('mul_fake_quant', ['mul', 'mul_min', 'mul_max'])
        self.sequence_1.set_inputs('add', ['mul_fake_quant', 'biases'])
        self.sequence_1.set_outputs(['add'])

        self.sequences = [self.sequence, self.sequence_1]

    def resolve_layer(self, graph_matcher, graph_helper):
        potential_descriptors = []
        for sequence in self.sequences:
            matches = graph_matcher.match_sequence(sequence)
            for match in matches:
                inputs_op = match['inputs']
                biases_op = match['biases']
                weights_op = match['weights']
                bn_op = match['mul']

                inputs_shape = graph_helper.get_op_output_shape(inputs_op)
                if not inputs_shape:
                    continue

                # only support constant weights and biases for batchnorm
                if not graph_helper.check_op_const_origin(biases_op)[0] or \
                        not graph_helper.check_op_const_origin(weights_op)[0]:
                    continue

                # squeeze weights/biases to 1D as expected for batchnorm, if not able to be squeezed as 1D, skip
                biases_tensor = np.atleast_1d(graph_helper.evaluate_tensor_output(biases_op.outputs[0]).squeeze())
                weights_tensor = np.atleast_1d(graph_helper.evaluate_tensor_output(weights_op.outputs[0]).squeeze())
                if len(weights_tensor.shape) != 1 or len(biases_tensor.shape) != 1:
                    continue

                channel_dims = inputs_shape[-1:]
                # broadcast weights to match channel dims
                if weights_tensor.shape[0] == 1:
                    weights_tensor = GenericBatchNormLayerResolver._broadcast_tensor(weights_tensor, channel_dims)
                # broadcast bias to match channel dims
                if biases_tensor.shape[0] == 1:
                    biases_tensor = GenericBatchNormLayerResolver._broadcast_tensor(biases_tensor, channel_dims)

                consumed_nodes = match.consumed_nodes
                output_op_nodes_names = [str(match[node.identifier].outputs[0].name) for node in self.sequence.output_nodes]

                potential_descriptors.append(
                    GenericBatchNormLayerResolver.Descriptor(str(bn_op.name),
                                                             consumed_nodes,
                                                             bn_mul_op=bn_op,
                                                             pre_calculated=True,
                                                             weights=weights_tensor,
                                                             biases=biases_tensor,
                                                             output_names=output_op_nodes_names))
        return potential_descriptors

    @classmethod
    def _broadcast_tensor(cls, tensor, shape):
        broadcasted_tensor = np.zeros(shape, dtype=np.float32)
        broadcasted_tensor = broadcasted_tensor + tensor
        return broadcasted_tensor


class BatchNormWithGlobalNormLayerResolver(BatchNormLayerResolver):
    class Descriptor(BatchNormLayerResolver.Descriptor):
        pass

    def __init__(self):
        self.sequence = GraphSequence([ConverterSequenceNode('root', ['BatchNormWithGlobalNormalization'])])
        self.sequence.set_outputs(['root'])

    def resolve_layer(self, graph_matcher, graph_helper):
        matches = graph_matcher.match_sequence(self.sequence)
        if len(matches) == 0:
            return []
        potential_descriptors = []
        for match in matches:
            bn_op = match['root']
            parameter_tensors = self._const_inputs(graph_helper, bn_op)
            if len(parameter_tensors) < 4:
                raise ConverterError(code_to_message.get_error_message('ERROR_TF_BATCHNORM_GLOBALNORMALIZATION_INPUT'))
            epsilon = bn_op.get_attr('variance_epsilon')
            mean = parameter_tensors[0]
            variance = parameter_tensors[1]
            beta = parameter_tensors[2]
            scale = parameter_tensors[3]
            consumed_nodes = match.consumed_nodes
            potential_descriptors.append(
                BatchNormWithGlobalNormLayerResolver.Descriptor(str(bn_op.name),
                                                                consumed_nodes,
                                                                bn_mul_op=bn_op,
                                                                mean=mean,
                                                                variance=variance,
                                                                epsilon=epsilon,
                                                                scale=scale,
                                                                beta=beta))
        return potential_descriptors

    @classmethod
    def _const_inputs(cls, graph_helper, bn_op):
        return [graph_helper.evaluate_tensor_output(tensor) for tensor in bn_op.inputs if tensor.op.type == 'Const']


class FusedBatchNormNormLayerResolver(BatchNormLayerResolver):
    class Descriptor(BatchNormLayerResolver.Descriptor):
        pass

    def __init__(self):
        self.sequence = GraphSequence([ConverterSequenceNode('root', ['FusedBatchNorm', 'FusedBatchNormV3'])])
        self.sequence.set_outputs(['root'])

    def resolve_layer(self, graph_matcher, graph_helper):
        matches = graph_matcher.match_sequence(self.sequence)
        potential_descriptors = []
        for match in matches:
            bn_op = match['root']
            parameter_tensors = self._get_parameter_tensors(graph_helper, bn_op)
            if len(parameter_tensors) < 4:
                raise ConverterError(code_to_message.get_error_message('ERROR_TF_BATCHNORM_GLOBALNORMALIZATION_INPUT'))
            epsilon = bn_op.get_attr('epsilon')
            # we want the last 4 inputs, as sometimes non-parameter input can be of type of Identity(eg: seen in
            # mobilenet fpn ssd)
            scale = parameter_tensors[-4]
            beta = parameter_tensors[-3]
            mean = parameter_tensors[-2]
            variance = parameter_tensors[-1]
            consumed_nodes = match.consumed_nodes

            potential_descriptors.append(
                FusedBatchNormNormLayerResolver.Descriptor(str(bn_op.name),
                                                           consumed_nodes,
                                                           bn_mul_op=bn_op,
                                                           mean=mean,
                                                           variance=variance,
                                                           epsilon=epsilon,
                                                           scale=scale,
                                                           beta=beta))
        return potential_descriptors

    @classmethod
    def _get_parameter_tensors(cls, graph_helper, bn_op):
        parameter_tensors = [t for t in bn_op.inputs if t.op.type in ['Const', 'Identity']]
        tensors_outputs = graph_helper.evaluate_tensors_output(parameter_tensors)
        return [tensors_outputs[t] for t in parameter_tensors]


class BatchNormLayerBuilder(LayerBuilder):
    def build_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type ir_graph: converters.common.converter_ir.op_graph.IROpGraph
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: BatchNormLayerResolver.Descriptor
        :rtype: int
        """
        input_name = self.get_input_name(converter_context, descriptor, input_descriptors)

        return ir_graph.add(BatchnormOp(descriptor.layer_name,
                                        descriptor.weights,
                                        descriptor.biases,
                                        gamma=descriptor.scale,
                                        beta=descriptor.beta),
                            input_name,
                            descriptor.output_names[0])
