# =============================================================================
#
#  Copyright (c) 2015-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import numpy as np

from qti.aisw.converters.common.utils import code_to_message
from qti.aisw.converters.tensorflow.layers.reshape import ReshapeLayerResolver
from qti.aisw.converters.common.converter_ir.op_adapter import PermuteOp
from qti.aisw.converters.common.converter_ir.op_adapter import FullyConnectedOp
from qti.aisw.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from qti.aisw.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    GraphSequence,
    NonConsumableConverterSequenceNode
)
from qti.aisw.converters.tensorflow.util import ConverterError


class FullyConnectedLayerResolver(LayerResolver, object):
    class Descriptor(LayerDescriptor):
        def __init__(self, name, nodes, matmul_op, bias_op, weights, biases,
                     transpose_a=False, transpose_b=False,
                     output_names=None):
            super(FullyConnectedLayerResolver.Descriptor, self).__init__('FullyConnected', name, nodes, output_names=output_names)
            self.matmul_op = matmul_op
            self.bias_op = bias_op
            self.weights = weights
            self.biases = biases
            self.transpose_a = transpose_a
            self.transpose_b = transpose_b

        def is_input_op(self, op):
            # Reshape can get merged input/output batch is same, so its input will be considered input to fc
            if op == self.matmul_op:
                return True
            elif self.matmul_op.inputs[0].op.type == "Reshape":
                return op == self.matmul_op.inputs[0].op
            return False

        def is_input_tensor(self, op, tensor):
            return tensor == op.inputs[0]

    def __init__(self):

        sequence =  GraphSequence([
            ConverterSequenceNode('matmul_op', ['MatMul']),
            ConverterSequenceNode('bias_op', ['BiasAdd', 'Add']),  # output
            NonConsumableConverterSequenceNode('biases', ['Identity', 'Const']),
            NonConsumableConverterSequenceNode('weights', ['?']),
            NonConsumableConverterSequenceNode('inputs', ['?'])
        ])
        sequence.set_inputs('matmul_op', ['inputs', 'weights'])
        sequence.set_inputs('bias_op', ['matmul_op', 'biases'])
        sequence.set_outputs(['bias_op'])

        sequence_without_bias = GraphSequence([
            ConverterSequenceNode('matmul_op', ['MatMul']),
            NonConsumableConverterSequenceNode('weights', ['?']),
            NonConsumableConverterSequenceNode('inputs', ['?'])
        ])
        sequence_without_bias.set_inputs('matmul_op', ['inputs', 'weights'])
        sequence_without_bias.set_outputs(['matmul_op'])

        sequence_with_reshape =  GraphSequence([
            ConverterSequenceNode('matmul_op', ['MatMul']),
            ConverterSequenceNode('reshape_op', ['Reshape']),
            ConverterSequenceNode('bias_op', ['BiasAdd', 'Add']),  # output
            NonConsumableConverterSequenceNode('biases', ['Identity', 'Const']),
            NonConsumableConverterSequenceNode('weights', ['?']),
            NonConsumableConverterSequenceNode('shape', ['?']),
            NonConsumableConverterSequenceNode('inputs', ['?'])
        ])
        sequence_with_reshape.set_inputs('matmul_op', ['inputs', 'weights'])
        sequence_with_reshape.set_inputs('reshape_op', ['matmul_op', 'shape'])
        sequence_with_reshape.set_inputs('bias_op', ['reshape_op', 'biases'])
        sequence_with_reshape.set_outputs(['bias_op'])

        self.sequences = [sequence_with_reshape,sequence_without_bias,sequence]

    def resolve_layer(self, graph_matcher, graph_helper):
        descriptors = []
        for sequence in self.sequences:
            matches = graph_matcher.match_sequence(sequence)
            for match in matches:
                matmul_op = match['matmul_op']
                weights_op = match['weights']
                biases_op = None
                bias_add_op = None
                reshape_op = None
                if weights_op.type not in ['Identity', 'Const', 'Split', 'FakeQuantWithMinMaxVars'] and \
                        not graph_helper.check_op_const_origin(weights_op)[0]:
                    # If weights is not Constant, we cannot convert Matmul to FullyConnected.
                    # Matmul Layer will be used instead
                    continue
                weights = graph_helper.evaluate_tensor_output(weights_op.outputs[0])
                _, _, weight_consumed_nodes = graph_helper.get_static_data_info(matmul_op, weights_op.outputs[0])

                try:
                    bias_add_op = match['bias_op']
                    biases_op = match['biases']
                    reshape_op = match['reshape_op']
                except KeyError:
                    pass

                if biases_op is not None and bias_add_op is not None:
                    if biases_op.type not in ['Identity', 'Const']:
                        # do we still need this check ?
                        raise ConverterError(
                            code_to_message.get_error_message('ERROR_TF_MATMUL_RESOLVE_BIAS')(bias_add_op.name))
                    biases = graph_helper.evaluate_tensor_output(biases_op.outputs[0])
                else:
                    biases = np.zeros(weights.shape[-1], dtype=np.float32)

                consumed_nodes = match.consumed_nodes
                consumed_nodes.extend(weight_consumed_nodes)

                output_op_nodes_names = [str(match[node.identifier].outputs[0].name) for node in sequence.output_nodes]

                # We need a output reshape after FC because our FC runtime will output the shape of rank <= 2
                # Make bias_add_op consumed by Reshape for output name and topological relation
                if reshape_op is not None:
                    reshape_consumed_nodes = [reshape_op, bias_add_op]
                    reshape_desc = ReshapeLayerResolver.Descriptor(str(reshape_op.name), reshape_consumed_nodes, reshape_op)
                    descriptors.append(reshape_desc)
                    # Remove reshape_consumed_nodes and change output names for FC Descriptor
                    # Note: remove consumed node from list individually to preserve order
                    consumed_nodes.remove(reshape_op)
                    if bias_add_op is not None:
                        consumed_nodes.remove(bias_add_op)
                    output_op_nodes_names = [str(reshape_op.inputs[0].name)]

                # Not feeding the reshape_shape here, because we have a reshape op above to deal with it
                descriptors.append(
                    FullyConnectedLayerResolver.Descriptor(str(matmul_op.name), consumed_nodes,
                                                           matmul_op, bias_add_op, weights, biases,
                                                           transpose_a=matmul_op.get_attr('transpose_a'),
                                                           transpose_b=matmul_op.get_attr('transpose_b'),
                                                           output_names=output_op_nodes_names))

        return descriptors


class FullyConnectedLayerBuilder(LayerBuilder):
    def build_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type ir_graph: converters.common.converter_ir.op_graph.IROpGraph
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: FullyConnectedLayerResolver.Descriptor
        :rtype: int
        """
        if descriptor.transpose_b:
            weight_tensor = descriptor.weights.copy()
        else:
            weight_tensor = np.transpose(descriptor.weights, (1, 0)).copy()

        input_names = self.get_input_names(converter_context, descriptor, input_descriptors)
        output_name = descriptor.output_names[0]
        if descriptor.transpose_a:
            input_name = input_names[0]
            transpose_output_name = input_name + '_transpose'
            temp = converter_context.get_input_layer_output_shape_for(descriptor.matmul_op)
            order = list(range(0, len(temp)))
            order = order[::-1]
            ir_graph.add(PermuteOp(input_name + '_transpose', order=order),
                         input_names=[input_name],
                         output_names=[transpose_output_name])
            input_names[0] = transpose_output_name

        bias_op = descriptor.bias_op
        return ir_graph.add(FullyConnectedOp(name=descriptor.layer_name,
                                             weights=weight_tensor,
                                             bias=descriptor.biases,
                                             bias_op_name=bias_op.name if bias_op else None,
                                             transpose_a=descriptor.transpose_a,
                                             transpose_b=descriptor.transpose_b),
                            input_names,
                            output_name)
