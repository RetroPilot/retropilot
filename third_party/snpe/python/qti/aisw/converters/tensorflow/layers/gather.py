# =============================================================================
#
#  Copyright (c) 2015-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from qti.aisw.converters.common.utils.code_to_message import get_error_message
from qti.aisw.converters.common.converter_ir.op_adapter import GatherOp, ReshapeOp
from qti.aisw.converters.tensorflow.common import LayerDescriptor, LayerBuilder, LayerResolver
from qti.aisw.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    NonConsumableConverterSequenceNode,
    GraphSequence
)
from abc import ABCMeta
from abc import abstractmethod
from qti.aisw.converters.tensorflow.layers.constant import ConstantLayerResolver
from qti.aisw.converters.tensorflow.util import GraphHelper
from qti.aisw.converters.tensorflow.util import TensorNotFoundError, ConverterError


class GatherLayerResolver(LayerResolver, object):
    __metaclass__ = ABCMeta

    class Descriptor(LayerDescriptor):
        def __init__(self, name, nodes, input_names, axis, output_names=None):
            super(GatherLayerResolver.Descriptor, self).__init__('Gather', name, nodes, output_names=output_names)
            self.input_names = input_names
            self.axis = axis

        def is_input_tensor(self, op, tensor):
            return tensor == op.inputs[0] or tensor == op.inputs[1]

    def __init__(self):
        sequence_1 = GraphSequence([
            ConverterSequenceNode('root', ['GatherV2']),
            NonConsumableConverterSequenceNode('params', ['?']),
            NonConsumableConverterSequenceNode('axis', ['?']),
            NonConsumableConverterSequenceNode('indices', ['?'])
        ])
        sequence_1.set_inputs('root', ['params', 'indices', 'axis'])
        sequence_1.set_outputs(['root'])

        # Filter seqs 2
        sequence_2 = GraphSequence([
            ConverterSequenceNode('root', ['Gather']),
            NonConsumableConverterSequenceNode('params', ['?']),
            NonConsumableConverterSequenceNode('indices', ['?'])
        ])
        sequence_2.set_inputs('root', ['params', 'indices'])
        sequence_2.set_outputs(['root'])

        self.sequences = [sequence_1, sequence_2]

    def resolve_layer(self, graph_matcher, graph_helper):
        descriptors = []
        for sequence in self.sequences:
            for match in graph_matcher.match_sequence(sequence):
                gather_op = match['root']
                consumed_nodes = match.consumed_nodes
                indices, params, axis = self.get_tensors(graph_helper, gather_op)
                params, const_params_consumed_ops = graph_helper.get_none_identity_input(params)
                indices, const_indices_consumed_ops = graph_helper.get_none_identity_input(indices)
                input_names = []

                input_names.extend([GraphHelper.indexed_tensor_name(params.op.name),
                                    GraphHelper.indexed_tensor_name(indices.op.name)])
                descriptor = GatherLayerResolver.Descriptor(str(gather_op.name), consumed_nodes,
                                                            input_names, axis, [gather_op.outputs[0].name])

                descriptors.append(descriptor)

                if indices.op.type == 'Const':
                    const_indices_shape = GraphHelper.get_tensor_output_shape(indices)
                    const_indices_val = graph_helper.evaluate_tensor_output(indices).astype('int32')
                    const_indices_descriptor = ConstantLayerResolver.Descriptor(str(indices.op.name),
                                                                                const_indices_consumed_ops,
                                                                                const_indices_val, const_indices_shape,
                                                                                descriptor)
                    descriptors.append(const_indices_descriptor)

                if params.op.type == 'Const':
                    const_shape = GraphHelper.get_tensor_output_shape(params)
                    const_val = graph_helper.evaluate_tensor_output(params)
                    const_descriptor = ConstantLayerResolver.Descriptor(str(params.op.name),
                                                                        const_params_consumed_ops,
                                                                        const_val, const_shape,
                                                                        descriptor)
                    descriptors.append(const_descriptor)
        return descriptors

    @classmethod
    def get_tensors(cls, graph_helper, gather_op):
        try:
            indices, params, axis = GraphHelper.get_op_input_tensors(gather_op, ('?', '?', 'Const'))
        except TensorNotFoundError:
            indices, params = GraphHelper.get_op_input_tensors(gather_op, ('?', '?'))
            axis = 0
        indices_shape = graph_helper.get_op_output_shape(indices.op)
        params_shape = graph_helper.get_op_output_shape(params.op)
        if not isinstance(axis, int):
            if len(GraphHelper.get_tensor_output_shape(axis)) == 0:
                pass
            elif len(indices_shape) == 0 and indices.op.type == 'Const':
                indices, axis = axis, indices
            elif len(params_shape) == 0 and params.op.type == 'Const':
                params, axis = axis, params
            axis = graph_helper.evaluate_tensor_output(axis)

        if (indices.dtype.name == 'float32' or indices.dtype.name == 'float64') \
                and (params.dtype.name == 'int32' or params.dtype.name == 'int64'):
            indices, params = params, indices
        return indices, params, axis


class GatherLayerBuilder(LayerBuilder):

    def transform_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):
        const_indices = [d for d in input_descriptors if (d.layer_name+":0") == descriptor.input_names[1] and isinstance(d, ConstantLayerResolver.Descriptor)]
        if len(const_indices) != 0:
            for d in const_indices:
                d.set_quantizable(False)

    def build_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type ir_graph: converters.common.converter_ir.op_graph.IROpGraph
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: EmbeddingLayerResolver.Descriptor
        :rtype: int
        """
        output_name = descriptor.output_names[0]
        input_buf_shape = ir_graph.get_buffer(descriptor.input_names[0]).shape
        # Find indices descriptor in input descriptors if it is constant
        const_indices_desc = [
            d for d in input_descriptors
            if (d.layer_name in descriptor.input_names[1] and isinstance(d, ConstantLayerResolver.Descriptor))
        ]

        gather_output_name = output_name
        if const_indices_desc and const_indices_desc[0].was_scalar:
            gather_output_name = output_name+'_pre_reshape'
        ir_graph.add(GatherOp(descriptor.layer_name,
                              axis=descriptor.axis),
                     descriptor.input_names,
                     gather_output_name)
        if const_indices_desc and const_indices_desc[0].was_scalar:
            output_shape = input_buf_shape[:descriptor.axis] + input_buf_shape[descriptor.axis+1:]
            ir_graph.add(ReshapeOp(descriptor.layer_name+'_reshape',
                                   output_shape=output_shape),
                         gather_output_name,
                         output_name)
