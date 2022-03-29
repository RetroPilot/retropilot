# =============================================================================
#
#  Copyright (c) 2015-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from qti.aisw.converters.common.utils import code_to_message
from qti.aisw.converters.common.converter_ir.op_adapter import ConcatOp
from qti.aisw.converters.common.utils.translation_utils import compare_values
from qti.aisw.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from qti.aisw.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    GraphSequence
)
from qti.aisw.converters.tensorflow.layers.constant import ConstantLayerResolver
from qti.aisw.converters.tensorflow.layers.ignored_patterns import IgnoredLayersResolver
from qti.aisw.converters.tensorflow.util import ConverterError, get_const_op_value
from qti.aisw.converters.tensorflow.util import GraphHelper


class ConcatLayerResolver(LayerResolver, object):
    class Descriptor(LayerDescriptor):
        def __init__(self, name, nodes, axis, output_names=None):
            super(ConcatLayerResolver.Descriptor, self).__init__('Concatenation', name, nodes,
                                                                 output_names=output_names)
            self.axis = axis

        def is_input_tensor(self, op, tensor):
            # Ignores a static axis input which has already been consumed by the resolver
            if tensor.op.type == "Const" and compare_values(get_const_op_value(tensor.op), self.axis):
                return False
            return True

    def __init__(self):
        self.sequence = GraphSequence([ConverterSequenceNode('root', ['Concat', 'ConcatV2'])])
        self.sequence.set_outputs(['root'])

    def resolve_layer(self, graph_matcher, graph_helper):
        matches = graph_matcher.match_sequence(self.sequence)
        if len(matches) == 0:
            return []
        descriptors = []
        for match in matches:
            concat_op = match['root']
            consumed_nodes = match.consumed_nodes
            concat_descriptor = ConcatLayerResolver.Descriptor(str(concat_op.name), consumed_nodes,
                                                               None, [concat_op.outputs[0].name])

            non_const_inputs = [tensor for tensor in concat_op.inputs if tensor.op.type != 'Const']
            const_ops = [tensor.op for tensor in concat_op.inputs if tensor.op.type == 'Const']
            axis_tensor = None
            if len(non_const_inputs) < 2 or len(const_ops) > 1:
                for i in range(0, len(const_ops) - 1):
                    const_value = graph_helper.evaluate_tensor_output(const_ops[i].outputs[0])
                    const_shape = graph_helper.get_op_output_shape(const_ops[i].outputs[0])
                    descriptors.append(ConstantLayerResolver.Descriptor(str(const_ops[i].name),
                                                                        [const_ops[i]],
                                                                        const_value,
                                                                        const_shape,
                                                                        concat_descriptor))
                # Make the assumption that the axis is always the last constant
                axis_tensor = const_ops[-1]

            if not axis_tensor:
                axis_tensor = GraphHelper.filter_single_op_by_type([t.op for t in concat_op.inputs], 'Const')
            axis = int(graph_helper.evaluate_tensor_output(axis_tensor.outputs[0]))

            concat_descriptor.axis = axis
            descriptors.append(concat_descriptor)

        return descriptors


class ConcatLayerBuilder(LayerBuilder):

    def transform_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):
        # Optimization to avoid going to 5-Dimensional Concat only if batch input is 1.
        # Check the following to see if the optimization can be made
        # 1. Input op must be ExpandDims
        # 2. Axis of ExpandDims must match Axis of Concat
        # 3. Input data tensor to ExpandDims must have batch = 1
        # 4. Output of Concat must go to a reshape
        # 5. Reshape must be merging the batch and 5-th Dimension together
        get_tensor = converter_context.graph_helper.get_op_input_tensors
        evaluate_output = converter_context.graph_helper.evaluate_tensor_output
        get_shape = converter_context.graph_helper.get_op_output_shape
        if all(len(x.child_ops) != 0 and x.child_ops[-1].type == 'ExpandDims' and
               evaluate_output(get_tensor(x.child_ops[-1], ('?', 'Const'))[1]) == descriptor.axis and # Check ExpandDims axis == Concat Axis
               get_shape(get_tensor(x.child_ops[-1], ('?', 'Const'))[0])[0] == 1 # Check input batch == 1
               for x in input_descriptors) and \
           len(output_descriptors) == 1 and output_descriptors[0].child_ops[-1].type == 'Reshape' and \
           len(get_shape(output_descriptors[0].child_ops[-1])) == 4:
            for input_descriptor in input_descriptors:
                input_descriptor.set_ignored(True)

            output_descriptors[0].set_ignored(True)
            descriptor.axis = 0
            return

        if len(input_descriptors) == 1 and isinstance(input_descriptors[0], IgnoredLayersResolver.Descriptor):
            descriptor.set_ignored(True)
            return

        concat_outputs = [d for d in output_descriptors if isinstance(d, ConcatLayerResolver.Descriptor)]
        if concat_outputs == output_descriptors and len(concat_outputs) == 1:
            concat_on_concat_output = concat_outputs[0]
            if descriptor.axis == concat_on_concat_output.axis:
                converter_context.merge_descriptors(descriptor, concat_on_concat_output)

    def build_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type ir_graph: converters.common.converter_ir.op_graph.IROpGraph
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: ConcatLayerResolver.Descriptor
        :rtype: int
        """
        # Check for single input concat or a single input descriptor with less than two outputs
        if len(input_descriptors) < 2:
            if (len(input_descriptors) == 1 and
                    len(converter_context.get_output_tensors_between(input_descriptors[0], descriptor)) < 2):
                raise ConverterError(code_to_message.get_error_message('ERROR_TF_CONCAT_INPUT'))

        axis = descriptor.axis
        if axis < 0:
            max_shape = 0
            for input_d in input_descriptors:
                input_tensors = converter_context.get_output_tensors_between(input_d, descriptor)
                for t in input_tensors:
                    shape = converter_context.graph_helper.get_op_output_shape(t.op)
                    if len(shape) > max_shape:
                        max_shape = len(shape)
            axis += max_shape

        input_names = self.get_input_names(converter_context, descriptor, input_descriptors)
        return ir_graph.add(ConcatOp(descriptor.layer_name, descriptor.axis),
                            input_names,
                            descriptor.output_names[0])
