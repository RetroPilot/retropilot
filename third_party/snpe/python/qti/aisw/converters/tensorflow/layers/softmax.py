# =============================================================================
#
#  Copyright (c) 2015-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from qti.aisw.converters.common.converter_ir.op_adapter import SoftmaxOp
from qti.aisw.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from qti.aisw.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    GraphSequence,
    NonConsumableConverterSequenceNode
)


class SoftmaxLayerResolver(LayerResolver, object):

    class Descriptor(LayerDescriptor):
        def __init__(self, name, nodes, softmax_input_op, axis, output_names=None):
            super(SoftmaxLayerResolver.Descriptor, self).__init__('SoftMax', name, nodes, output_names=output_names)
            self.softmax_input_op = softmax_input_op
            self.axis = axis

        def is_input_op(self, op):
            return len(op.inputs) and op.inputs[0].op == self.softmax_input_op

        def is_input_tensor(self, op, tensor):
            return tensor == op.inputs[0]

    def __init__(self):
        sequence_two_dim_softmax = GraphSequence([ConverterSequenceNode('root', ['SoftMax'])])
        sequence_two_dim_softmax.set_outputs(['root'])

        sequence_multi_dim_softmax = GraphSequence([
            ConverterSequenceNode('max', ['Max']),
            ConverterSequenceNode('max_reduction_indicies', ['Const']),
            ConverterSequenceNode('sub', ['Sub']),
            ConverterSequenceNode('exp', ['Exp']),
            ConverterSequenceNode('sum', ['Sum']),
            ConverterSequenceNode('sum_reduction_indicies', ['Const']),
            ConverterSequenceNode('root', ['RealDiv']),
            NonConsumableConverterSequenceNode('input', ['?'])
        ])
        sequence_multi_dim_softmax.set_inputs('max', ['input', 'max_reduction_indicies'])
        sequence_multi_dim_softmax.set_inputs('sub', ['input', 'max'])
        sequence_multi_dim_softmax.set_inputs('exp', ['sub'])
        sequence_multi_dim_softmax.set_inputs('sum', ['exp', 'sum_reduction_indicies'])
        sequence_multi_dim_softmax.set_inputs('root', ['exp', 'sum'])
        sequence_multi_dim_softmax.set_outputs(['root'])

        sequence_with_default_axis = GraphSequence([
            ConverterSequenceNode('sub', ['Sub']),
            ConverterSequenceNode('slice/begin', ['Pack']),
            ConverterSequenceNode('slice', ['Slice']),
            ConverterSequenceNode('concat', ['ConcatV2']),
            ConverterSequenceNode('first_reshape', ['Reshape']),
            ConverterSequenceNode('root', ['Softmax']),
            ConverterSequenceNode('last_reshape', ['Reshape']),
            NonConsumableConverterSequenceNode('sub/rank', ['?']),
            NonConsumableConverterSequenceNode('sub/y', ['?']),
            NonConsumableConverterSequenceNode('slice/shape', ['?']),
            NonConsumableConverterSequenceNode('slice/size', ['?']),
            NonConsumableConverterSequenceNode('concat/values', ['?']),
            NonConsumableConverterSequenceNode('concat/axis', ['?']),
            NonConsumableConverterSequenceNode('input', ['?']),
            NonConsumableConverterSequenceNode('shape', ['?']),
        ])

        sequence_with_default_axis.set_inputs('sub', ['sub/rank', 'sub/y'])
        sequence_with_default_axis.set_inputs('slice', ['slice/shape', 'slice/begin', 'slice/size'])
        sequence_with_default_axis.set_inputs('slice/begin', ['sub'])
        sequence_with_default_axis.set_inputs('first_reshape', ['input', 'concat'])
        sequence_with_default_axis.set_inputs('concat', ['concat/values', 'slice', 'concat/axis'])
        sequence_with_default_axis.set_inputs('last_reshape', ['root', 'shape'])
        sequence_with_default_axis.set_inputs('root', ['first_reshape'])
        sequence_with_default_axis.set_outputs(['last_reshape'])

        sequence_with_nondefault_axis = GraphSequence([
            ConverterSequenceNode('sub_1', ['Sub']),
            ConverterSequenceNode('sub_2', ['Sub']),
            ConverterSequenceNode('slice/begin', ['Pack']),
            ConverterSequenceNode('shape_1', ['Shape']),
            ConverterSequenceNode('range_1', ['Range']),
            ConverterSequenceNode('concat_3/values_1', ['Pack']),
            ConverterSequenceNode('range_4', ['Range']),
            ConverterSequenceNode('slice', ['Slice']),
            ConverterSequenceNode('concat_3', ['ConcatV2']),
            ConverterSequenceNode('concat_1', ['ConcatV2']),
            ConverterSequenceNode('sub_3', ['Sub']),
            ConverterSequenceNode('transpose_1', ['Transpose']),
            ConverterSequenceNode('first_reshape', ['Reshape']),
            ConverterSequenceNode('range_3', ['Range']),
            ConverterSequenceNode('concat_2/values_1', ['Pack']),
            ConverterSequenceNode('range_2', ['Range']),
            ConverterSequenceNode('shape_2', ['Shape']),
            ConverterSequenceNode('root', ['Softmax']),
            ConverterSequenceNode('concat_2', ['ConcatV2']),
            ConverterSequenceNode('last_reshape', ['Reshape']),
            ConverterSequenceNode('transpose_2', ['Transpose']),
            NonConsumableConverterSequenceNode('sub_1/rank', ['?']),
            NonConsumableConverterSequenceNode('sub_1/y', ['?']),
            NonConsumableConverterSequenceNode('rank', ['?']),
            NonConsumableConverterSequenceNode('sub_3/y', ['?']),
            NonConsumableConverterSequenceNode('range_2/start', ['?']),
            NonConsumableConverterSequenceNode('range_2/delta', ['?']),
            NonConsumableConverterSequenceNode('range_1/start', ['?']),
            NonConsumableConverterSequenceNode('range_1/limit', ['?']),
            NonConsumableConverterSequenceNode('range_1/delta', ['?']),
            NonConsumableConverterSequenceNode('slice/size', ['?']),
            NonConsumableConverterSequenceNode('concat_3/values_3', ['?']),
            NonConsumableConverterSequenceNode('concat_3/axis', ['?']),
            NonConsumableConverterSequenceNode('concat_1/values_0', ['?']),
            NonConsumableConverterSequenceNode('concat_1/axis', ['?']),
            NonConsumableConverterSequenceNode('sub_2/y', ['?']),
            NonConsumableConverterSequenceNode('input', ['?']),
            NonConsumableConverterSequenceNode('range_4/start', ['?']),
            NonConsumableConverterSequenceNode('range_4/delta', ['?']),
            NonConsumableConverterSequenceNode('range_3/start', ['?']),
            NonConsumableConverterSequenceNode('range_3/limit', ['?']),
            NonConsumableConverterSequenceNode('range_3/delta', ['?']),
            NonConsumableConverterSequenceNode('concat_2/values_3', ['?']),
            NonConsumableConverterSequenceNode('concat_2/axis', ['?']),
        ])

        sequence_with_nondefault_axis.set_inputs('shape_1', ['transpose_1'])
        sequence_with_nondefault_axis.set_inputs('shape_2', ['transpose_1'])
        sequence_with_nondefault_axis.set_inputs('transpose_2', ['last_reshape', 'concat_2'])
        sequence_with_nondefault_axis.set_inputs('slice', ['shape_1', 'slice/begin', 'slice/size'])
        sequence_with_nondefault_axis.set_inputs('concat_3/values_1', ['sub_3'])
        sequence_with_nondefault_axis.set_inputs('concat_2/values_1', ['sub_2'])
        sequence_with_nondefault_axis.set_inputs('last_reshape', ['root', 'shape_2'])
        sequence_with_nondefault_axis.set_inputs('range_4', ['range_4/start', 'sub_2', 'range_4/delta'])
        sequence_with_nondefault_axis.set_inputs('range_3', ['range_3/start', 'range_3/limit', 'range_3/delta'])
        sequence_with_nondefault_axis.set_inputs('root', ['first_reshape'])
        sequence_with_nondefault_axis.set_inputs('sub_1', ['sub_1/rank', 'sub_1/y'])
        sequence_with_nondefault_axis.set_inputs('range_1', ['range_1/start', 'range_1/limit', 'range_1/delta'])
        sequence_with_nondefault_axis.set_inputs('transpose_1', ['input', 'concat_3'])
        sequence_with_nondefault_axis.set_inputs('concat_3', ['range_1', 'concat_3/values_1', 'range_2',
                                                              'concat_3/values_3', 'concat_3/axis'])
        sequence_with_nondefault_axis.set_inputs('slice/begin', ['sub_1'])
        sequence_with_nondefault_axis.set_inputs('sub_2', ['rank', 'sub_2/y'])
        sequence_with_nondefault_axis.set_inputs('sub_3', ['rank', 'sub_3/y'])
        sequence_with_nondefault_axis.set_inputs('range_2', ['range_2/start', 'sub_3', 'range_2/delta'])
        sequence_with_nondefault_axis.set_inputs('first_reshape', ['transpose_1', 'concat_1'])
        sequence_with_nondefault_axis.set_inputs('concat_1', ['concat_1/values_0', 'slice', 'concat_1/axis'])
        sequence_with_nondefault_axis.set_inputs('concat_2', ['range_3', 'concat_2/values_1', 'range_4',
                                                              'concat_2/values_3', 'concat_2/axis'])
        sequence_with_nondefault_axis.set_outputs(['transpose_2'])

        self.sequences = [sequence_two_dim_softmax, sequence_multi_dim_softmax, sequence_with_default_axis,
                          sequence_with_nondefault_axis]

    def resolve_layer(self, graph_matcher, graph_helper):
        potential_descriptors = []
        for sequence in self.sequences:
            matches = graph_matcher.match_sequence(sequence)
            for match in matches:
                softmax_op = match['root']

                # Existence of range_1 implies a match where non-default axis was specified based on analysis of those
                # cases. range_1 can be used to derive this non-default value since TF uses a sequence of Sub/Range
                # nodes to derive the axis attribute and uses that result to reshape the softmax input into a 2D tensor.
                # The axis attribute is then also used to reshape the 2D output back to the original shape.
                if 'range_1' in match:
                    axis_src_op = match['range_1']
                    axis_src_inputs = graph_helper.get_op_input_tensors(axis_src_op, ['Const', 'Const', 'Const'])
                    axis = graph_helper.get_static_data_info(axis_src_inputs[1])[0]
                else:
                    # TF default axis value of -1 (representing input_rank - 1)
                    axis = -1

                consumed_nodes = match.consumed_nodes
                softmax_input_op = match['input'] if 'input' in match else softmax_op.inputs[0].op
                potential_descriptors.append(
                    SoftmaxLayerResolver.Descriptor(str(softmax_op.name), consumed_nodes, softmax_input_op, axis))
        return potential_descriptors


class SoftmaxLayerBuilder(LayerBuilder):
    def build_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type ir_graph: converters.common.converter_ir.op_graph.IROpGraph
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: SoftmaxLayerResolver.Descriptor
        :rtype: int
        """
        input_name = self.get_input_name(converter_context, descriptor, input_descriptors)
        output_name = descriptor.output_names[0]
        return ir_graph.add(SoftmaxOp(name=descriptor.layer_name, axis=descriptor.axis),
                            input_names=input_name, output_names=output_name)
