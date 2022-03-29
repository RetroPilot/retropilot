# =============================================================================
#
#  Copyright (c) 2015-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import numpy as np

from qti.aisw.converters.common.converter_ir.op_adapter import ConstantOp
from qti.aisw.converters.common.utils.converter_utils import log_debug1, log_info, log_verbose
from qti.aisw.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from qti.aisw.converters.tensorflow.layers.ignored_patterns import IgnoredLayersResolver
from qti.aisw.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    GraphSequence
)


class ConstantLayerResolver(LayerResolver, object):
    class Descriptor(LayerDescriptor):
        def __init__(self, name, nodes, value, shape, consumer, quantizable=True):
            super(ConstantLayerResolver.Descriptor, self).__init__('Constant', name, nodes)
            self.value = value
            self.was_scalar = False
            self.shape = shape
            if not shape:
                self.was_scalar = True
                self.shape = [1]

            self.consumer = consumer
            self.quantizable = quantizable

        def __repr__(self):
            str = super().__repr__()
            str += "consumer: {}\n".format(self.consumer.layer_name)
            return str

        def is_input_tensor(self, op, tensor):
            return False

        def set_quantizable(self, bool):
            self.quantizable = bool

    @staticmethod
    def _get_const_descriptors(graph_helper, op):
        desc = []
        for output_tensor in op.outputs:
            try:
                const_value = graph_helper.evaluate_tensor_output(output_tensor)
                const_shape = graph_helper.get_op_output_shape(output_tensor)
                quantizable = True if const_value.dtype == np.dtype('float32') else False
                desc.append(ConstantLayerResolver.Descriptor(str(output_tensor.name),
                                                             [op],
                                                             const_value,
                                                             const_shape,
                                                             None,
                                                             quantizable=quantizable))
                log_verbose("Adding constant desc for {}".format(str(output_tensor.name)))
            except:
                # some const tensor eval result in reevaluation, which tensorflow throws an internal error
                # e.g.: seen in cases where loops are present in model
                continue
        return desc

    @staticmethod
    def _get_remaining_ops(graph):
        ops = set()
        for node in graph:
            ops.add(node.original_node)
        return ops

    def resolve_layer(self, graph_matcher, graph_helper):
        def _add_const_desc(op_):
            const_nodes.add(op_)
            if op_ in remaining_ops:
                descriptors.extend(self._get_const_descriptors(graph_helper, op_))
                remaining_ops.remove(op_)

        log_info("Resolving static sub-graphs in network...")
        remaining_ops = self._get_remaining_ops(graph_matcher.graph)
        graph_input_ops = graph_helper.get_graph_input_ops()
        graph_output_tensors = graph_helper.get_graph_output_tensors()
        descriptors = []
        visited = set()
        const_nodes = set()
        for output_tensor in graph_output_tensors:
            queue = [output_tensor.op]
            while len(queue) and len(remaining_ops):
                op = queue.pop(0)
                visited.add(op)
                # Stop traversal either if op itself is input placeholder or user requested op to be input
                if op.type == "Placeholder" or op in graph_input_ops:
                    continue
                unresolved_inputs = [input_.op for input_ in op.inputs if input_.op not in visited]
                if op in remaining_ops:
                    # if op is const or all of op's inputs are visited and evaluated as const, evaluate this op
                    # as const as well
                    if op.type in ["Const", "Shape"] or all([input_.op in const_nodes for input_ in op.inputs]):
                        _add_const_desc(op)
                    # if all of op's inputs already visited but were not found to have const origin, no need to
                    # evaluate op for const. Since we are traversing up, once an op is hit its input origin would have
                    # been determined before visiting op again. (i.e acyclic graph support only here)
                    elif len(unresolved_inputs) == 0:
                        continue
                    else:
                        # if we have input ops that are not in the remaining ops need to evaluate to source for current
                        # op since it will lead to broken path. We dont want to evaluate path for all ops if not in
                        # remaining_ops since some resolvers consume quite a huge chuck of ops(e.g. ssd) that will slow
                        # static resolution if all evaluated for const origin
                        if any(input_op not in remaining_ops for input_op in unresolved_inputs):
                            is_const_origin, consumed_ops = graph_helper.check_op_const_origin(op)
                            if is_const_origin:
                                _add_const_desc(op)
                                visited.update(consumed_ops)
                                for consumed_op in consumed_ops:
                                    _add_const_desc(consumed_op)
                            # add the unresolved inputs to the beginning of the queue so that inputs get resolved for
                            # subsequent ops in the queue that depend on them.
                            else:
                                queue[:0] = unresolved_inputs
                        else:
                            queue[:0] = [*unresolved_inputs, op]
                else:
                    queue[:0] = unresolved_inputs

            if output_tensor.op in const_nodes:
                log_info("Output {} was evaluated as ConstOp during static sub-graphs resolution on network",
                         output_tensor.op.name)

        log_info("Resolving static sub-graphs in network, complete.")
        return descriptors


class ConstantLayerBuilder(LayerBuilder):

    def transform_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):
        if len(output_descriptors) == 1 and descriptor.consumer is not None and not descriptor.consumer == output_descriptors[0]:
            log_verbose("Set desc to ignored with name: {}".format(descriptor.layer_name))
            descriptor.set_ignored(True)

        # resolve actual consumer(if any) for the const layer since ignored layers will be removed after transformation
        ignored = [d for d in output_descriptors if isinstance(d, IgnoredLayersResolver.Descriptor)]
        if len(ignored):
            const_consumers = converter_context.topology_resolver.get_output_layers_for(descriptor)
            for desc in ignored:
                const_consumers.remove(desc)
            while len(ignored):
                outputs = converter_context.topology_resolver.get_output_layers_for(ignored.pop())
                ignored.extend([d for d in outputs if isinstance(d, IgnoredLayersResolver.Descriptor)])
                const_consumers.extend([d for d in outputs if not isinstance(d, IgnoredLayersResolver.Descriptor)])

            if not len(const_consumers):
                descriptor.set_ignored(True)

    def build_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type ir_graph: converters.common.converter_ir.op_graph.IROpGraph
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: ConstantLayerResolver.Descriptor
        :rtype: int
        """

        # ConstantOp has no inputs
        input_names = []

        if not isinstance(descriptor.value, np.ndarray):
            array = np.zeros(descriptor.shape, dtype=np.float32)
            array[...] = descriptor.value
            descriptor.value = array

        return ir_graph.add(ConstantOp(descriptor.output_names[0],
                                       descriptor.value,
                                       quantizable=descriptor.quantizable),
                            input_names,
                            descriptor.output_names[0])
