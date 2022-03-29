# =============================================================================
#
#  Copyright (c) 2015-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import numpy

from qti.aisw.converters.common.converter_ir.axis_tracker import AxisTracker
from qti.aisw.converters.common.converter_ir.op_adapter import ConstantOp, DeconvolutionOp
from qti.aisw.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from qti.aisw.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    GraphSequence,
    NonConsumableConverterSequenceNode
)
from qti.aisw.converters.tensorflow.layers.convolution import ConvolutionLayerBuilder


class DeconvolutionLayerResolver(LayerResolver, object):
    class Descriptor(LayerDescriptor):

        def __init__(self, name, nodes, deconv_op, bias_op, strides, padding_size_strategy, input_tensor,
                     weights_tensor=None, bias_tensor=None):
            super(DeconvolutionLayerResolver.Descriptor, self).__init__('Deconvolution', name, nodes)
            self.deconv_op = deconv_op
            self.bias_op = bias_op
            self.strides = strides
            self.padding_size_strategy = padding_size_strategy
            self.input_ops = [deconv_op] if bias_op is None else [deconv_op, bias_op]
            self.input_tensor = input_tensor
            self.weights_tensor = weights_tensor
            self.bias_tensor = bias_tensor

        def is_input_op(self, op):
            return op in self.input_ops

        def is_input_tensor(self, op, tensor):
            if (op == self.deconv_op and tensor == self.deconv_op.inputs[0]) or (tensor == self.deconv_op.outputs[0]):
                return False
            return True

        @property
        def output_names(self):
            if self.bias_op:
                output_name = str(self.bias_op.outputs[0].name)
            else:
                output_name = str(self.deconv_op.outputs[0].name)
            return [output_name]

    def __init__(self):
        super(DeconvolutionLayerResolver, self).__init__()
        self.graph_sequence = GraphSequence([
            NonConsumableConverterSequenceNode('input_sizes', ['?']),
            NonConsumableConverterSequenceNode('weights_source', ['?']),
            NonConsumableConverterSequenceNode('input', ['?']),
            ConverterSequenceNode('root', ['Conv2DBackpropInput']),
            NonConsumableConverterSequenceNode('bias_source', ['?']),
            ConverterSequenceNode('bias', ['Add', 'BiasAdd'])
        ])
        self.graph_sequence.set_inputs('root', ['input_sizes', 'weights_source', 'input'])

    def resolve_layer(self, graph_matcher, graph_helper):
        matches = []

        # Basic deconvolution sequence
        self.graph_sequence.set_outputs(['root'])
        matches.extend(graph_matcher.match_sequence(self.graph_sequence))

        # Basic deconvolution sequence with bias
        self.graph_sequence.clear_outputs()
        self.graph_sequence.set_inputs('bias', ['root', 'bias_source'])
        self.graph_sequence.set_outputs(['bias'])
        matches.extend(graph_matcher.match_sequence(self.graph_sequence))

        descriptors = []
        for match in matches:
            deconv_op = match['root']
            input_tensor = match['input'].outputs[0]
            consumed_nodes = list(match.consumed_nodes)

            weights_source_op = match['weights_source']
            weights_tensor = None
            if graph_helper.check_tensor_const_origin(weights_source_op.outputs[0])[0]:
                weights_tensor = graph_helper.evaluate_tensor_output(weights_source_op.outputs[0])
            else:
                raise ValueError("Dynamic weights on {} node of type {} are unsupported.".format(
                    deconv_op.name, deconv_op.type))

            bias_op, bias_tensor = None, None
            if 'bias' in match:
                bias_op = match['bias']
                bias_source_op = match['bias_source']
                if graph_helper.check_tensor_const_origin(bias_source_op.outputs[0])[0]:
                    bias_tensor = graph_helper.evaluate_tensor_output(bias_source_op.outputs[0])
                else:
                    continue

            # Extract attributes
            strides = deconv_op.get_attr('strides')
            padding_size_strategy = deconv_op.get_attr('padding')

            descriptors.append(
                DeconvolutionLayerResolver.Descriptor(str(deconv_op.name),
                                                      consumed_nodes,
                                                      deconv_op,
                                                      bias_op,
                                                      strides,
                                                      padding_size_strategy,
                                                      input_tensor,
                                                      weights_tensor=weights_tensor,
                                                      bias_tensor=bias_tensor))
        return descriptors


class DeconvolutionLayerBuilder(LayerBuilder, object):
    def build_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type ir_graph: converters.common.converter_ir.op_graph.IROpGraph
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: DeconvolutionLayerResolver.Descriptor
        :rtype: int
        """
        input_dims = converter_context.graph_helper.get_op_output_shape(descriptor.deconv_op.inputs[2])
        filter_dims = converter_context.graph_helper.get_op_output_shape(descriptor.deconv_op.inputs[1])

        if descriptor.bias_op:
            output_dims = converter_context.graph_helper.get_op_output_shape(descriptor.bias_op)
        else:
            output_dims = converter_context.graph_helper.get_op_output_shape(descriptor.deconv_op)

        pads, ir_padding_strategy = ConvolutionLayerBuilder.calculate_padding_size(input_size=output_dims[-3:-1],
                                                                                   output_size=input_dims[-3:-1],
                                                                                   strides=descriptor.strides[1:3],
                                                                                   padding_size_strategy=descriptor.padding_size_strategy,
                                                                                   filter_dims=filter_dims,
                                                                                   dilation=[1, 1])

        # IR expects activation, filter, bias for ordering of inputs
        input_names = [self.get_input_names(converter_context, descriptor, input_descriptors)[1]]

        if descriptor.weights_tensor is not None:
            input_names.append(descriptor.layer_name + "_weight")
            weights_tensor = numpy.transpose(descriptor.weights_tensor, AxisTracker.AxisFormat.HWOI_TO_HWIO).copy()
            weights_const_op = ConstantOp(name=input_names[-1], tensor=weights_tensor)
            ir_graph.add(weights_const_op, [], input_names[-1], axis_formats=[AxisTracker.AxisFormat.HWIO])
            weights_buffer = ir_graph.get_buffer(input_names[-1])
            weights_buffer.shape = list(weights_tensor.shape)
        else:
            raise ValueError("Dynamic weights on {} node {} are unsupported.".format(
                descriptor.layer_name, descriptor.layer_type))

        # Handle biases, depending on source operation and already extracted tensor
        if descriptor.bias_tensor is not None:
            input_names.append(descriptor.layer_name + "_bias")
            bias_const_op = ConstantOp(name=input_names[-1], tensor=descriptor.bias_tensor)
            ir_graph.add(bias_const_op, [], input_names[-1], axis_formats=[AxisTracker.AxisFormat.ANY])
        elif descriptor.bias_op is not None and descriptor.bias_tensor is None:
            raise ValueError("Dynamic bias on {} node of type {} are unsupported.".format(
                descriptor.layer_name, descriptor.layer_type))

        return ir_graph.add(DeconvolutionOp(name=descriptor.layer_name,
                                            bias_op_name=descriptor.bias_op.name if descriptor.bias_op else None,
                                            stridex=descriptor.strides[1],
                                            stridey=descriptor.strides[2],
                                            padding_size_strategy=ir_padding_strategy,
                                            pady_before=pads[0][0],
                                            pady_after=pads[0][1],
                                            padx_before=pads[1][0],
                                            padx_after=pads[1][1],
                                            output_width=output_dims[-2],
                                            output_height=output_dims[-3],
                                            groups=1),
                            input_names,
                            descriptor.output_names[0])
