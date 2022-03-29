# =============================================================================
#
#  Copyright (c) 2018-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import numpy as np

from qti.aisw.converters.common.utils import code_to_message
from qti.aisw.converters.common.converter_ir.op_adapter import PadOp
from qti.aisw.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from qti.aisw.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    NonConsumableConverterSequenceNode,
    GraphSequence
)
from qti.aisw.converters.tensorflow.util import ConverterError


class PadLayerResolver(LayerResolver, object):
    TF_ATTRIBUTE_MODE = 'mode'

    class TfPadMode:
        CONSTANT = 'CONSTANT'
        REFLECT = 'REFLECT'
        SYMMETRIC = 'SYMMETRIC'
    class Descriptor(LayerDescriptor):
        def __init__(self, name, nodes, paddings, mode, constant_values, output_names=None):
            super(PadLayerResolver.Descriptor, self).__init__('Pad', name, nodes, output_names=output_names)
            self.paddings = paddings
            self.mode = mode
            self.constant_values = constant_values

    def __init__(self):
        self.sequence_with_zero_padding = GraphSequence([
            ConverterSequenceNode('root', ['Pad', 'PadV2']),
            NonConsumableConverterSequenceNode('paddings', ['?']),
            NonConsumableConverterSequenceNode('input', ['?']),
        ])
        self.sequence_with_zero_padding.set_inputs('root', ['input', 'paddings'])
        self.sequence_with_zero_padding.set_outputs(['root'])

        self.sequence_with_const_padding = GraphSequence([
            ConverterSequenceNode('root', ['Pad', 'PadV2']),
            NonConsumableConverterSequenceNode('paddings', ['Const']),
            NonConsumableConverterSequenceNode('const_values', ['Const']),
            NonConsumableConverterSequenceNode('input', ['?']),
        ])
        self.sequence_with_const_padding.set_inputs('root', ['input', 'paddings', 'const_values'])
        self.sequence_with_const_padding.set_outputs(['root'])

        self.sequence_with_reflect_padding = GraphSequence([
            ConverterSequenceNode('mirror_pad', ['MirrorPad']),
            NonConsumableConverterSequenceNode('paddings', ['Const']),
            NonConsumableConverterSequenceNode('input', ['?']),
        ])
        self.sequence_with_reflect_padding.set_inputs('mirror_pad', ['input', 'paddings'])
        self.sequence_with_reflect_padding.set_outputs(['mirror_pad'])

        self.sequences = [self.sequence_with_zero_padding, self.sequence_with_const_padding,
                          self.sequence_with_reflect_padding]

    def resolve_layer(self, graph_matcher, graph_helper):
        descriptors = []
        for sequence in self.sequences:
            for match in graph_matcher.match_sequence(sequence):
                pad_op = None
                mode_values = PadOp.Mode.CONSTANT
                if 'root' in match:
                    pad_op = match['root']
                elif 'mirror_pad' in match:
                    pad_op = match['mirror_pad']
                    mode = pad_op.get_attr(self.TF_ATTRIBUTE_MODE)
                    if mode.decode() == self.TfPadMode.REFLECT:
                        mode_values = PadOp.Mode.REFLECT
                    elif mode.decode() == self.TfPadMode.SYMMETRIC:
                        mode_values = PadOp.Mode.SYMMETRIC
                    else:
                        raise ConverterError(code_to_message.get_error_message("ERROR_TF_PAD_MODE_UNKNOWN")
                                             (mode.decode()))

                input_op = match['input']
                paddings_op = match['paddings']

                paddings_tensor, paddings_shape, paddings_nodes = graph_helper.get_static_data_info(
                                                                      pad_op, paddings_op.outputs[0])

                input_rank = len(graph_helper.get_op_output_shape(input_op))
                if [input_rank, 2] != paddings_shape:
                    raise ConverterError(code_to_message.get_error_message(
                        'ERROR_TF_PAD_INVALID_PADDINGS')(str([input_rank, 2]), str(paddings_shape)))

                if 'const_values' in match:
                    const_values_op = match['const_values']
                    const_values = graph_helper.evaluate_tensor_output(const_values_op.outputs[0])
                else:
                    const_values = 0.0

                if not np.isscalar(const_values):
                    raise ConverterError(code_to_message.get_error_message('ERROR_TF_PAD_CONSTANT_NOT_SCALAR'))

                consumed_nodes = match.consumed_nodes
                consumed_nodes.extend(paddings_nodes)
                pad_descriptor = PadLayerResolver.Descriptor(
                    str(pad_op.name), consumed_nodes, paddings_tensor,
                    mode_values, const_values,
                    output_names=[str(pad_op.outputs[0].name)])
                descriptors.extend([pad_descriptor])

        return descriptors


class PadLayerBuilder(LayerBuilder):
    def build_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :param graph_helper:
        :type ir_graph: converters.common.converter_ir.op_graph.IROpGraph
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: PadLayerResolver.Descriptor
        :rtype: int
        """
        input_name = self.get_input_name(converter_context, descriptor, input_descriptors[:1])
        output_name = descriptor.output_names[0]
        return ir_graph.add(PadOp(descriptor.layer_name,
                                  pads=np.array(descriptor.paddings, dtype=np.dtype('int32')),
                                  mode=descriptor.mode,
                                  constant_value=float(descriptor.constant_values)),
                            input_name,
                            output_name)
