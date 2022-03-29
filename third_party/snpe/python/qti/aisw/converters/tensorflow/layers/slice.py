# =============================================================================
#
#  Copyright (c) 2015-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from qti.aisw.converters.common.utils import code_to_message
from qti.aisw.converters.common.converter_ir.op_adapter import SliceOp
from qti.aisw.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from qti.aisw.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    GraphSequence
)
from qti.aisw.converters.tensorflow.util import ConverterError
from qti.aisw.converters.tensorflow.util import GraphHelper
from qti.aisw.converters.tensorflow.util import TensorNotFoundError


class SliceLayerResolver(LayerResolver, object):
    class Descriptor(LayerDescriptor):
        def __init__(self, name, nodes, axis, split_sizes):
            super(SliceLayerResolver.Descriptor, self).__init__('Slice', name, nodes)
            self.axis = axis
            self.split_sizes = split_sizes

        def is_input_tensor(self, op, tensor):
            # resolver supports the attribute inputs only when Const, hence the non-const is the actual input
            if "Const" in [tensor.op.type, GraphHelper.get_none_identity_input(tensor)[0].op.type]:
                return False
            return True

        @property
        def output_names(self):
            return [str(t.name) for t in self.child_ops[-1].outputs]

    def __init__(self):
        self.sequence = GraphSequence([ConverterSequenceNode('root', ['Split', 'SplitV'])])
        self.sequence.set_outputs(['root'])

    def resolve_layer(self, graph_matcher, graph_helper):
        matches = graph_matcher.match_sequence(self.sequence)
        if len(matches) == 0:
            return []
        potential_descriptors = []
        for match in matches:
            split_op = match['root']
            split_axis, split_sizes = self.get_split_axis_and_sizes(graph_helper, split_op)
            consumed_nodes = match.consumed_nodes
            potential_descriptors.append(
                SliceLayerResolver.Descriptor(str(split_op.name), consumed_nodes,
                                              split_axis,
                                              split_sizes))
        return potential_descriptors

    @classmethod
    def get_split_axis_and_sizes(cls, graph_helper, split_op):
        try:
            _, split_sizes, split_axis = GraphHelper.get_op_input_tensors(split_op, ('?', 'Const', 'Const'))
            split_sizes = list(graph_helper.evaluate_tensor_output(split_sizes))
        except TensorNotFoundError:
            split_axis, _ = GraphHelper.get_op_input_tensors(split_op, ('Const', '?'))
            split_sizes = []

        split_axis = int(graph_helper.evaluate_tensor_output(split_axis))
        return split_axis, split_sizes


class SliceLayerBuilder(LayerBuilder):
    def build_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type ir_graph: converters.common.converter_ir.op_graph.IROpGraph
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: SliceLayerResolver.Descriptor
        :rtype: int
        """
        input_shape = converter_context.get_input_layer_output_shape_for(descriptor.child_ops[0])
        input_name = self.get_input_name(converter_context, descriptor, input_descriptors)
        split_points = self.get_split_positions(input_shape, descriptor.split_sizes, descriptor.axis)

        return ir_graph.add(SliceOp(name=descriptor.layer_name,
                                    axis=descriptor.axis,
                                    slice_points=split_points),
                            input_names=input_name,
                            output_names=descriptor.output_names)

    @classmethod
    def get_split_positions(cls, input_shape, split_sizes, split_axis):
        split_points = []
        if len(split_sizes) > 0:
            if sum(split_sizes) != input_shape[split_axis]:
                raise ConverterError(code_to_message.get_error_message('ERROR_TF_SLICE_SIZE_MISMATCH'))
            split_index = split_sizes[0]
            for size in split_sizes[1:]:
                split_points.append(int(split_index))
                split_index += size
        return split_points
