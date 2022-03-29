# =============================================================================
#
#  Copyright (c) 2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import numpy as np

from qti.aisw.converters.common.utils import code_to_message
from qti.aisw.converters.common.converter_ir.op_adapter import GeluOp
from qti.aisw.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from qti.aisw.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    GraphSequence,
    NonConsumableConverterSequenceNode
)
from qti.aisw.converters.tensorflow.util import ConverterError


class GeLuLayerResolver(LayerResolver, object):
    class Descriptor(LayerDescriptor):
        def __init__(self, name, operations, output_names):
            super(GeLuLayerResolver.Descriptor, self).__init__('GeLU', name, operations,
                                                                output_names=output_names)

    def __init__(self):
        sequence_gelu = GraphSequence([
            ConverterSequenceNode('out', ['Gelu']),
            NonConsumableConverterSequenceNode('inputs', ['?'])
        ])
        sequence_gelu.set_inputs('out', ['inputs'])
        sequence_gelu.set_outputs(['out'])

        sequence_low_level_gelu = GraphSequence([
            ConverterSequenceNode('a', ['Sqrt']),
            ConverterSequenceNode('b', ['RealDiv']),
            ConverterSequenceNode('c', ['Erf']),
            ConverterSequenceNode('d', ['Add','AddV2']),
            ConverterSequenceNode('e', ['Mul']),
            ConverterSequenceNode('out', ['Mul']),  # output
            NonConsumableConverterSequenceNode('sqrt_const', ['?']),
            NonConsumableConverterSequenceNode('inputs', ['?']),
            NonConsumableConverterSequenceNode('add1', ['?']),
            NonConsumableConverterSequenceNode('mul_const', ['?'])
        ])
        sequence_low_level_gelu.set_inputs('a', ['sqrt_const'])
        sequence_low_level_gelu.set_inputs('b', ['a','inputs'])
        sequence_low_level_gelu.set_inputs('c', ['b'])
        sequence_low_level_gelu.set_inputs('d', ['c','add1'])
        sequence_low_level_gelu.set_inputs('e', ['d', 'mul_const'])
        sequence_low_level_gelu.set_inputs('out', ['e','inputs'])
        sequence_low_level_gelu.set_outputs(['out'])

        sequence_const_low_level_gelu = GraphSequence([
            ConverterSequenceNode('b', ['RealDiv']),
            ConverterSequenceNode('c', ['Erf']),
            ConverterSequenceNode('d', ['Add','AddV2']),
            ConverterSequenceNode('e', ['Mul']),
            ConverterSequenceNode('out', ['Mul']),  # output
            NonConsumableConverterSequenceNode('sqrt_const', ['?']),
            NonConsumableConverterSequenceNode('inputs', ['?']),
            NonConsumableConverterSequenceNode('add1', ['?']),
            NonConsumableConverterSequenceNode('mul_const', ['?'])
        ])
        sequence_const_low_level_gelu.set_inputs('b', ['inputs', 'sqrt_const'])
        sequence_const_low_level_gelu.set_inputs('c', ['b'])
        sequence_const_low_level_gelu.set_inputs('d', ['c','add1'])
        sequence_const_low_level_gelu.set_inputs('e', ['inputs', 'mul_const'])
        sequence_const_low_level_gelu.set_inputs('out', ['e','d'])
        sequence_const_low_level_gelu.set_outputs(['out'])

        self.sequences = [sequence_gelu, sequence_low_level_gelu, sequence_const_low_level_gelu]

    def resolve_layer(self, graph_matcher, graph_helper):
        potential_descriptors = []
        for sequence in self.sequences:
            matches = graph_matcher.match_sequence(sequence)
            for match in matches:
                out_op = match['out']

                output_op_nodes_names = [str(match[node.identifier].outputs[0].name) for node in sequence.output_nodes]
                consumed_nodes = match.consumed_nodes

                potential_descriptors.append(
                    GeLuLayerResolver.Descriptor(str(out_op.name), consumed_nodes,
                                                 output_names=output_op_nodes_names))
        return potential_descriptors

class GeLuLayerBuilder(LayerBuilder):
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
        return ir_graph.add(GeluOp(name=descriptor.layer_name),
                            input_names=input_name,
                            output_names=output_name)
