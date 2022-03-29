# =============================================================================
#
#  Copyright (c) 2018-2020 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import numpy as np
from qti.aisw.converters.common.converter_ir import op_adapter
from qti.aisw.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from qti.aisw.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    NonConsumableConverterSequenceNode,
    GraphSequence
)


class ReductionLayerResolver(LayerResolver):

    class Descriptor(LayerDescriptor):
        def __init__(self, layer_type, name, nodes, axes, keep_dims, output_names=None):
            super(ReductionLayerResolver.Descriptor, self).__init__(layer_type, name, nodes, output_names=output_names)
            self.axes = axes
            self.keep_dims = keep_dims

    def __init__(self, layer_type, op_type, descriptor_class):
        super(ReductionLayerResolver, self).__init__()
        self._layer_type = layer_type
        self._op_type = op_type
        self._descriptor_class = descriptor_class

        self.sequence = GraphSequence([
            ConverterSequenceNode('root', [self._op_type]),
            ConverterSequenceNode('reduction_indices', ['?']),
            NonConsumableConverterSequenceNode('input', ['?']),
        ])
        self.sequence.set_inputs('root', ['input', 'reduction_indices'])
        self.sequence.set_outputs(['root'])

    def resolve_layer(self, graph_matcher, graph_helper):
        descriptors = []
        for match in graph_matcher.match_sequence(self.sequence):
            reduction_op = match['root']
            reduction_indices_op = match['reduction_indices']

            if graph_helper.check_tensor_const_origin(reduction_indices_op.outputs[0])[0]:
                axes = graph_helper.evaluate_tensor_output(reduction_indices_op.outputs[0])
            else:
                raise ValueError("Unsupported dynamic reduction indices on reduction op {}", format(reduction_op.name))

            axes = [int(axes)] if np.isscalar(axes) else axes.tolist()

            keep_dims = bool(reduction_op.get_attr('keep_dims'))

            reduction_descriptor = self._descriptor_class(self._layer_type, str(reduction_op.name),
                                                          match.consumed_nodes, axes, keep_dims,
                                                          output_names=[str(reduction_op.outputs[0].name)])
            descriptors.extend([reduction_descriptor])

        return descriptors


class ReductionLayerBuilder(LayerBuilder):
    def __init__(self, ir_op_class):
        self.ir_op_class = ir_op_class

    def build_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type ir_graph: converters.common.converter_ir.op_graph.IROpGraph
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: ReductionLayerResolver.Descriptor
        :rtype: int
        """
        input_name = self.get_input_name(converter_context, descriptor, input_descriptors)
        output_name = descriptor.output_names[0]
        return ir_graph.add(self.ir_op_class(name=descriptor.layer_name,
                                             axes=descriptor.axes,
                                             keep_dims=descriptor.keep_dims),
                            input_names=[input_name],
                            output_names=[output_name])


class ReductionMeanLayerResolver(ReductionLayerResolver):
    class Descriptor(ReductionLayerResolver.Descriptor):
        pass

    def __init__(self):
        super(ReductionMeanLayerResolver, self).__init__('ReduceMean', 'Mean', ReductionMeanLayerResolver.Descriptor)


class ReductionMeanLayerBuilder(ReductionLayerBuilder):
    def __init__(self):
        super(ReductionMeanLayerBuilder, self).__init__(op_adapter.ReduceMeanOp)


class ReductionProdLayerResolver(ReductionLayerResolver):
    class Descriptor(ReductionLayerResolver.Descriptor):
        pass

    def __init__(self):
        super(ReductionProdLayerResolver, self).__init__('ReduceProd', 'Prod', ReductionProdLayerResolver.Descriptor)


class ReductionProdLayerBuilder(ReductionLayerBuilder):
    def __init__(self):
        super(ReductionProdLayerBuilder, self).__init__(op_adapter.ReduceProdOp)


class ReductionSumLayerResolver(ReductionLayerResolver):
    class Descriptor(ReductionLayerResolver.Descriptor):
        pass

    def __init__(self):
        super(ReductionSumLayerResolver, self).__init__('ReduceSum', 'Sum', ReductionSumLayerResolver.Descriptor)


class ReductionSumLayerBuilder(ReductionLayerBuilder):
    def __init__(self):
        super(ReductionSumLayerBuilder, self).__init__(op_adapter.ReduceSumOp)


class ReductionMinLayerResolver(ReductionLayerResolver):
    class Descriptor(ReductionLayerResolver.Descriptor):
        pass

    def __init__(self):
        super(ReductionMinLayerResolver, self).__init__('ReduceMin', 'Min', ReductionMinLayerResolver.Descriptor)


class ReductionMinLayerBuilder(ReductionLayerBuilder):
    def __init__(self):
        super(ReductionMinLayerBuilder, self).__init__(op_adapter.ReduceMinOp)


class ReductionMaxLayerResolver(ReductionLayerResolver):
    class Descriptor(ReductionLayerResolver.Descriptor):
        pass

    def __init__(self):
        super(ReductionMaxLayerResolver, self).__init__('ReduceMax', 'Max', ReductionMaxLayerResolver.Descriptor)


class ReductionMaxLayerBuilder(ReductionLayerBuilder):
    def __init__(self):
        super(ReductionMaxLayerBuilder, self).__init__(op_adapter.ReduceMaxOp)
