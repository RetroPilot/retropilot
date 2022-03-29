# =============================================================================
#
#  Copyright (c) 2015-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import math
from abc import ABCMeta

from qti.aisw.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from qti.aisw.converters.common.converter_ir.op_adapter import PoolOp, IRPaddingStrategies
from qti.aisw.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    GraphSequence
)


class PoolingLayerResolver(LayerResolver, object):
    __metaclass__ = ABCMeta

    class Descriptor(LayerDescriptor):
        def __init__(self, layer_type, name, operations, pooling_type, strides, padding_size_strategy,
                     kernel_dims):
            super(PoolingLayerResolver.Descriptor, self).__init__(layer_type, name, operations)
            self.pooling_type = pooling_type
            self.strides = strides
            self.padding_size_strategy = padding_size_strategy
            self.kernel_dims = kernel_dims

        def is_input_tensor(self, op, tensor):
            return tensor == op.inputs[0]

    def __init__(self, layer_type, descriptor_type, pooling_type, op_type):
        super(PoolingLayerResolver, self).__init__()
        self._layer_type = layer_type
        self._descriptor_type = descriptor_type
        self._pooling_type = pooling_type
        self._op_type = op_type

        self.sequence = GraphSequence([ConverterSequenceNode('root', [self._op_type])])
        self.sequence.set_outputs(['root'])

    def resolve_layer(self, graph_matcher, graph_helper):
        matches = graph_matcher.match_sequence(self.sequence)
        if len(matches) == 0:
            return []
        potential_descriptors = []
        for match in matches:
            pooling_op = match['root']
            kernel_dims = pooling_op.get_attr('ksize')
            strides = pooling_op.get_attr('strides')
            padding_size_strategy = pooling_op.get_attr('padding')
            consumed_nodes = match.consumed_nodes
            potential_descriptors.append(
                self._descriptor_type(self._layer_type, str(pooling_op.name), consumed_nodes,
                                      self._pooling_type,
                                      strides, padding_size_strategy, kernel_dims))
        return potential_descriptors


class PoolingLayerBuilder(LayerBuilder):
    def build_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type ir_graph: converters.common.converter_ir.op_graph.IROpGraph
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: PoolingLayerResolver.Descriptor
        :rtype: int
        """
        input_dims = converter_context.get_input_layer_output_shape_for(descriptor.child_ops[0])

        pads, ir_padding_strategy = self.calculate_padding(descriptor.padding_size_strategy, input_dims[1:3],
                                                           descriptor.strides[1:3], descriptor.kernel_dims[1:3])

        input_name = self.get_input_name(converter_context, descriptor, input_descriptors)
        output_name = descriptor.output_names[0]
        return ir_graph.add(PoolOp(name=descriptor.layer_name,
                                   pool_type=descriptor.pooling_type,
                                   size_x=descriptor.kernel_dims[2],
                                   size_y=descriptor.kernel_dims[1],
                                   stride_x=descriptor.strides[2],
                                   stride_y=descriptor.strides[1],
                                   pady_before=pads[0][0],
                                   pady_after=pads[0][1],
                                   padx_before=pads[1][0],
                                   padx_after=pads[1][1],
                                   padding_size_strategy=ir_padding_strategy,
                                   pool_region_include_padding=False),
                            input_names=[input_name],
                            output_names=[output_name])

    @classmethod
    def calculate_padding(cls, padding_size_strategy, input_size, strides, pool_dims):

        if padding_size_strategy.decode() == 'SAME':
            output_height = math.ceil(float(input_size[0]) / float(strides[0]))
            output_width = math.ceil(float(input_size[1]) / float(strides[1]))
            pad_y = ((output_height - 1) * strides[0] + pool_dims[0] - input_size[0])
            pad_x = ((output_width - 1) * strides[1] + pool_dims[1] - input_size[1])
            # We divide by two and truncate if odd padding given the runtime will
            # take care of Asymmetry
            pad_y = int(pad_y / 2)
            pad_x = int(pad_x / 2)
            pads = [[pad_y, pad_y], [pad_x, pad_x]]
            ir_padding_strategy = IRPaddingStrategies.PADDING_SIZE_IMPLICIT_SAME_END
        elif padding_size_strategy.decode() == 'VALID':
            pads = [[0, 0], [0, 0]]
            ir_padding_strategy = IRPaddingStrategies.PADDING_SIZE_IMPLICIT_VALID
        else:
            raise ValueError("Unsupported TF padding strategy {}".format(padding_size_strategy.decode()))

        return pads, ir_padding_strategy


class AvgPoolingLayerResolver(PoolingLayerResolver):
    class Descriptor(PoolingLayerResolver.Descriptor):
        pass

    def __init__(self):
        super(AvgPoolingLayerResolver, self).__init__('AvgPooling', AvgPoolingLayerResolver.Descriptor,
                                                      PoolOp.Type.AVG, 'AvgPool')


class MaxPoolingLayerResolver(PoolingLayerResolver):
    class Descriptor(PoolingLayerResolver.Descriptor):
        pass

    def __init__(self):
        super(MaxPoolingLayerResolver, self).__init__('MaxPooling', MaxPoolingLayerResolver.Descriptor,
                                                      PoolOp.Type.MAX, 'MaxPool')
