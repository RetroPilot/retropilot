# =============================================================================
#
#  Copyright (c) 2018-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
from qti.aisw.converters.common.utils.translation_utils import compare_values
from qti.aisw.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from qti.aisw.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    NonConsumableConverterSequenceNode,
    GraphSequence
)
from qti.aisw.converters.tensorflow.layers.batchnorm import BatchNormLayerResolver
from qti.aisw.converters.tensorflow.util import get_const_op_value


class FakeQuantPerChannelResolver(LayerResolver, object):
    class Descriptor(LayerDescriptor):
        def __init__(self, layer_type, name, nodes, is_act_quant, input_tensor_name, min, max, bw):
            super(FakeQuantPerChannelResolver.Descriptor, self).__init__(layer_type, name, nodes)
            self.is_act_quant = is_act_quant
            self.input_tensor_name = input_tensor_name
            self.min = min
            self.max = max
            self.bw = bw
            self.axis = 3


class FakeQuantLayerResolver(LayerResolver, object):
    class Descriptor(LayerDescriptor):
        def __init__(self, layer_type, name, nodes, is_act_quant, input_tensor_name, min, max, bw):
            super(FakeQuantLayerResolver.Descriptor, self).__init__(layer_type, name, nodes)

            self.is_act_quant = is_act_quant
            self.input_tensor_name = input_tensor_name
            self.min = min
            self.max = max
            self.bw = bw

        @property
        def output_names(self):
            return [str(self.child_ops[0].outputs[0].name)]

        def is_input_tensor(self, op, tensor):
            if tensor.op.type == "Const" and \
                    any([compare_values(get_const_op_value(tensor.op), t) for t in [float(self.min), float(self.max)]]):
                return False
            return True

        def is_output_op(self, op):
            return op in self.child_ops

        def get_output_names_for(self, input_tensors):
            return self.output_names

    def __init__(self):
        sequence1 = GraphSequence([
            ConverterSequenceNode('root', ['FakeQuantWithMinMaxVars','FakeQuantWithMinMaxVarsPerChannel']),
            ConverterSequenceNode('min', ['?']),
            ConverterSequenceNode('max', ['?']),
            NonConsumableConverterSequenceNode('input', ['?'])
        ])
        sequence1.set_inputs('root', ['input', 'min', 'max'])
        sequence1.set_outputs(['root'])

        # with bw set
        sequence2 = GraphSequence([
            ConverterSequenceNode('root', ['FakeQuantWithMinMaxVars','FakeQuantWithMinMaxVarsPerChannel']),
            ConverterSequenceNode('min', ['?']),
            ConverterSequenceNode('max', ['?']),
            ConverterSequenceNode('num_bits', ['?']),
            NonConsumableConverterSequenceNode('input', ['?'])
        ])
        sequence2.set_inputs('root', ['input', 'min', 'max', 'num_bits'])
        sequence2.set_outputs(['root'])

        self.sequences = [sequence1, sequence2]

    def resolve_layer(self, graph_matcher, graph_helper):
        potential_descriptors = []
        for sequence in self.sequences:
            matches = graph_matcher.match_sequence(sequence)
            for match in matches:
                fake_quant_op = match['root']
                min_op = match['min']
                max_op = match['max']
                input_op = match['input']

                # It's not activation-fake-quant node if input type is const or it originates from a const op
                is_act_quant = False if input_op.type in ['Const'] or \
                                        graph_helper.check_op_const_origin(input_op)[0] else True

                min = self._get_float(graph_helper, min_op)
                max = self._get_float(graph_helper, max_op)
                bw = self._get_float(graph_helper, matches["num_bits"]) if "num_bits" in matches else fake_quant_op.get_attr("num_bits")

                consumed_nodes = match.consumed_nodes

                if fake_quant_op.type =='FakeQuantWithMinMaxVarsPerChannel':
                    potential_descriptors.append(
                        FakeQuantPerChannelResolver.Descriptor('FakeQuantPerChannel', str(fake_quant_op.name), consumed_nodes,
                                                        is_act_quant, fake_quant_op.inputs[0].name, min, max, bw))
                else:
                    potential_descriptors.append(
                        FakeQuantLayerResolver.Descriptor('FakeQuant', str(fake_quant_op.name), consumed_nodes,
                                                        is_act_quant, fake_quant_op.inputs[0].name, min, max, bw))

        return potential_descriptors

    def _get_float(self, graph_helper, op):
        tensor = graph_helper.get_tensor_by_name(op.name)
        return graph_helper.evaluate_tensor_output(tensor)


class FakeQuantLayerBuilder(LayerBuilder):
    def build_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type ir_graph: converters.common.converter_ir.op_graph.IROpGraph
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: FakeQuantLayerResolver.Descriptor
        :rtype: int
        """
        return None

    def transform_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):

        descriptor.set_ignored(True)
        for i in range(1, len(input_descriptors)):
            # ignore inputs for min, max params
            input_descriptors[i].set_ignored(True)

        # TODO: remove adding the quantization params here when tf_to_ir uses op_graph_optimizations
        if descriptor.is_act_quant:
            # save quantization encodings for previous layer. ie quantization is done on the outputs of the previous
            # layer. node_x -> fakequant_node -> node_y
            if input_descriptors[0].layer_type == "RELU":
                fix_min=0.0
            else:
                fix_min=descriptor.min
            ir_graph.add_quantization_params(input_descriptors[0].layer_name,
                                             # here the descriptor.input_tensor_name is the output tensor name
                                             # for the input_descriptor
                                             output_encodings={"name": descriptor.input_tensor_name,
                                                               "bw": descriptor.bw,
                                                               "min": fix_min,
                                                               "max": descriptor.max})
        else:
            for output_descriptor in output_descriptors:
                # save quantization encodings for next layer. ie quantization is done on the const inputs of the next
                # layer. weights_node -> fakequant_node -> node_x
                if isinstance(descriptor, FakeQuantPerChannelResolver.Descriptor):
                    ir_graph.add_quantization_params(output_descriptor.layer_name,
                                                 # currently only weights are quantized in TF so using that as name
                                                 param_encodings={"name": "weights",
                                                                  "bw": descriptor.bw,
                                                                  "min": descriptor.min,
                                                                  "max": descriptor.max,
                                                                  "axis": descriptor.axis})
                else:
                    ir_graph.add_quantization_params(output_descriptor.layer_name,
                                                    # currently only weights are quantized in TF so using that as name
                                                    param_encodings={"name": "weights",
                                                                    "bw": descriptor.bw,
                                                                    "min": descriptor.min,
                                                                    "max": descriptor.max})

        # Only fuse activation-quant layer or if input is a folded batchnorm
        if len(output_descriptors) != 0:
            if descriptor.is_act_quant or \
                    len([d for d in input_descriptors if isinstance(d, BatchNormLayerResolver.Descriptor)
                                                         and d.bn_folded]) == 1:
                for output_descriptor in output_descriptors:
                    converter_context.replace_layer_input_with(output_descriptor, descriptor, input_descriptors)
