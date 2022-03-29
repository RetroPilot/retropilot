# =============================================================================
#
#  Copyright (c) 2015-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import logging
import numpy as np

from qti.aisw.converters.common.converter_ir.axis_tracker import AxisTracker
from qti.aisw.converters.common.utils import code_to_message
from qti.aisw.converters.common.converter_ir.op_adapter import (
    ConstantOp,
    ConvolutionOp,
    DepthwiseConvolutionOp,
    ReshapeOp,
    ResizeOp,
    PadOp,
    IRPaddingStrategies
)
from qti.aisw.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from qti.aisw.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    GraphSequence,
    ConverterRepeatableSequenceTreeNode,
    NonConsumableConverterSequenceNode
)
from qti.aisw.converters.tensorflow.layers.ignored_patterns import IgnoredLayersResolver
from qti.aisw.converters.tensorflow.layers.crop import CropLayerResolver
from qti.aisw.converters.tensorflow.layers.pad import PadLayerResolver
from qti.aisw.converters.tensorflow.layers.resize import ResizeBilinearLayerResolver
from qti.aisw.converters.tensorflow.util import ConverterError, GraphHelper, OperationNotFoundError, TensorNotFoundError


class ConvolutionLayerResolver(LayerResolver, object):
    TF_ATTRIBUTE_DILATIONS = 'dilations'
    TF_ATTRIBUTE_STRIDES = 'strides'
    TF_ATTRIBUTE_PADDING = 'padding'
    TF_ATTRIBUTE_EXPLICIT_PADDING = 'explicit_paddings'

    def get_spatial_padding(self, conv_op):
        try:
            paddings = conv_op.get_attr(self.TF_ATTRIBUTE_EXPLICIT_PADDING)
        except ValueError:
            return [[0, 0], [0, 0]]

        spatial_padding = []
        for i in range(1, (len(paddings) - 1)):  # for NHWC, get HW
            for j in range(len(paddings[0])):
                spatial_padding.append([paddings[i][j]])

        return spatial_padding

    class Descriptor(LayerDescriptor):
        def __init__(self, name, nodes, conv_op, bias_op, output_op, strides, padding_size_strategy,
                     output_names=None, explicit_pads=list([[0, 0], [0, 0]]), resize_desc=None, pad_desc=None,
                     weights_tensor=None, bias_tensor=None):
            super(ConvolutionLayerResolver.Descriptor, self).__init__('Convolution', name, nodes,
                                                                      output_names=output_names)
            self.conv_op = conv_op
            self.bias_op = bias_op
            self.strides = strides
            self.padding_size_strategy = padding_size_strategy
            self.explicit_pads = explicit_pads  # Per tf docs: [[top, bottom], [left, right]]
            self.dilationX = 1
            self.dilationY = 1
            self.groups = len([op for op in nodes if op.type in ['Conv2D', 'FusedResizeAndPadConv2D']])
            self.output_op = output_op
            self.input_ops = [conv_op] if bias_op is None else [conv_op, bias_op]
            self.weights_tensor = weights_tensor
            self.bias_tensor = bias_tensor

            # Only FusedResizeAndPadConv2D op uses these descriptors to add resize/pad ops to the IR graph
            self.resize_desc = resize_desc
            self.pad_desc = pad_desc

        def is_input_op(self, op):
            return op in self.input_ops

        def is_input_tensor(self, op, tensor):
            if tensor == self.conv_op.outputs[0]:
                return False
            return True

    def __init__(self):
        # GraphSequence containing root and optional bias
        self.graph_sequence = GraphSequence([
            NonConsumableConverterSequenceNode('inputs', ['?']),
            NonConsumableConverterSequenceNode('weights_source', ['?']),
            ConverterSequenceNode('root', ['Conv2D']),
            NonConsumableConverterSequenceNode('bias_source', ['?']),
            ConverterSequenceNode('bias', ['Add', 'BiasAdd'])
        ])
        self.graph_sequence.set_inputs('root', ['inputs', 'weights_source'])

        self.fused_graph_sequence = GraphSequence([
            NonConsumableConverterSequenceNode('inputs', ['?']),
            NonConsumableConverterSequenceNode('size', ['?']),
            NonConsumableConverterSequenceNode('paddings', ['?']),
            NonConsumableConverterSequenceNode('weights_source', ['?']),
            ConverterSequenceNode('root', ['FusedResizeAndPadConv2D']),
            NonConsumableConverterSequenceNode('bias_source', ['?']),
            ConverterSequenceNode('bias', ['Add', 'BiasAdd'])
        ])
        self.fused_graph_sequence.set_inputs('root', ['inputs', 'size', 'paddings', 'weights_source'])

    def resolve_layer(self, graph_matcher, graph_helper):
        matches = []

        # Basic convolution sequence
        self.graph_sequence.set_outputs(['root'])
        matches.extend(graph_matcher.match_sequence(self.graph_sequence))

        # Basic convolution sequence with optional bias
        self.graph_sequence.clear_outputs()
        self.graph_sequence.set_inputs('bias', ['root', 'bias_source'])
        self.graph_sequence.set_outputs(['bias'])
        matches.extend(graph_matcher.match_sequence(self.graph_sequence))

        # Basic fused resize and pad sequence
        self.fused_graph_sequence.set_outputs(['root'])
        matches.extend(graph_matcher.match_sequence(self.fused_graph_sequence))

        # Basic fused resize and pad sequence with optional bias
        self.fused_graph_sequence.clear_outputs()
        self.fused_graph_sequence.set_inputs('bias', ['root', 'bias_source'])
        self.fused_graph_sequence.set_outputs(['bias'])
        matches.extend(graph_matcher.match_sequence(self.fused_graph_sequence))

        descriptors = []
        for match in matches:
            conv_op = match['root']
            bias_op = match['bias'] if 'bias' in match else None
            output_op = bias_op if bias_op else conv_op
            consumed_nodes = list(match.consumed_nodes)

            weights_source_op = match['weights_source']
            weights_tensor = self.get_weights_tensor(graph_helper, weights_source_op)

            bias_source_op = match['bias_source'] if 'bias_source' in match else None
            bias_tensor = self.get_bias_tensor(graph_helper, bias_source_op)

            # Represents cases where an elementwise add cannot be fused into a convolution node's bias
            if bias_tensor is not None and \
                    (len(bias_tensor.shape) != 1 or weights_tensor.shape[-1] != bias_tensor.shape[-1]):
                continue

            # Represents a dynamic bias, which is currently unsupported
            if bias_source_op is not None and bias_tensor is None:
                continue

            # Extract attributes
            strides = conv_op.get_attr(self.TF_ATTRIBUTE_STRIDES)
            padding_size_strategy = conv_op.get_attr(self.TF_ATTRIBUTE_PADDING)
            pads = self.get_spatial_padding(conv_op)

            # Attempt to extract dilations attribute, not present for fused resize and pad op
            try:
                dilations = conv_op.get_attr(self.TF_ATTRIBUTE_DILATIONS)
            except:
                dilations = [1, 1, 1, 1]

            # Extract the resize and pad descriptor, if the op is fused
            resize_desc, pad_desc = self.get_resize_pad_desc(graph_helper, conv_op)
            if resize_desc is not None or pad_desc is not None:
                consumed_nodes.extend([match['size'], match['paddings']])

            descriptor = ConvolutionLayerResolver.Descriptor(str(conv_op.name), consumed_nodes,
                                                             conv_op, bias_op, output_op,
                                                             strides, padding_size_strategy,
                                                             resize_desc=resize_desc,
                                                             pad_desc=pad_desc,
                                                             explicit_pads=pads,
                                                             weights_tensor=weights_tensor,
                                                             bias_tensor=bias_tensor,
                                                             output_names=[output_op.outputs[0].name])

            descriptor.dilationY = dilations[1]
            descriptor.dilationX = dilations[2]

            descriptors.append(descriptor)
        return descriptors

    def get_weights_tensor(self, graph_helper, weights_source_op):
        if graph_helper.check_tensor_const_origin(weights_source_op.outputs[0])[0]:
            return graph_helper.evaluate_tensor_output(weights_source_op.outputs[0])
        return None

    def get_bias_tensor(self, graph_helper, bias_source_op):
        if bias_source_op is not None:
            if graph_helper.check_tensor_const_origin(bias_source_op.outputs[0])[0]:
                return graph_helper.evaluate_tensor_output(bias_source_op.outputs[0])
        return None

    def get_resize_pad_desc(self, graph_helper, conv_op):
        resize_desc = None
        pad_desc = None

        try:
            conv_input, resize_size, pad, _ = GraphHelper.get_op_input_tensors(conv_op, ('?', '?', '?', '?'))
            input_tensor_shape = graph_helper.get_op_output_shape(conv_input)
            mul_const = graph_helper.evaluate_tensor_output(resize_size)
            if type(mul_const) is np.ndarray:
                mul_const = mul_const.squeeze().shape  # get the actual scale values for height and width
            if len(mul_const) < 2:
                mul_const = [0, 0]
            resize_desc = ResizeBilinearLayerResolver.Descriptor(str(resize_size.name),
                                                                 [resize_size.op],
                                                                 input_tensor_shape,
                                                                 resize_size,
                                                                 conv_op.get_attr('resize_align_corners'),
                                                                 mul_const,
                                                                 output_names=[str(resize_size.op.outputs[0].name)])
            mode = conv_op.get_attr('mode')
            if mode.decode() == "REFLECT":
                mode = PadOp.Mode.REFLECT
            elif mode.decode() == "SYMMETRIC":
                mode = PadOp.Mode.SYMMETRIC
            else:
                raise ConverterError(code_to_message.get_error_message("ERROR_TF_PAD_MODE_UNKNOWN")
                                     (mode.decode()))
            pad_desc = PadLayerResolver.Descriptor(str(pad.name),
                                                   [pad.op],
                                                   graph_helper.evaluate_tensor_output(pad),
                                                   mode,
                                                   0.0, # Only MirrorPad is matched, so constant val 0
                                                   output_names=[str(pad.op.outputs[0].name)])
        except TensorNotFoundError:
            pass
        return resize_desc, pad_desc


class DilatedConvolutionLayerResolver(ConvolutionLayerResolver, object):
    class Descriptor(ConvolutionLayerResolver.Descriptor):
        pass

    def __init__(self):
        super(DilatedConvolutionLayerResolver, self).__init__()

        # Basic dilated convolution sequence
        self.graph_sequence = GraphSequence([
            NonConsumableConverterSequenceNode('inputs', ['?']),
            NonConsumableConverterSequenceNode('dilation_sizes', ['?']),
            NonConsumableConverterSequenceNode('paddings', ['?']),
            ConverterSequenceNode('space_to_batch', ['SpaceToBatchND']),
            NonConsumableConverterSequenceNode('kernel', ['?']),
            ConverterSequenceNode('conv_op', ['Conv2D']),
            NonConsumableConverterSequenceNode('min', ['?']),
            NonConsumableConverterSequenceNode('max', ['?']),
            ConverterSequenceNode('fake_quant', ['FakeQuantWithMinMaxVars']),
            NonConsumableConverterSequenceNode('bias_source', ['?']),
            ConverterSequenceNode('bias', ['Add', 'BiasAdd']),
            NonConsumableConverterSequenceNode('block_shape_out', ['?']),
            NonConsumableConverterSequenceNode('crops', ['?']),
            ConverterSequenceNode('batch_to_space', ['BatchToSpaceND'])]
        )
        self.graph_sequence.set_inputs('space_to_batch', ['inputs', 'dilation_sizes', 'paddings'])
        self.graph_sequence.set_inputs('conv_op', ['space_to_batch', 'kernel'])
        self.graph_sequence.set_inputs('batch_to_space', ['conv_op', 'block_shape_out', 'crops'])

        self.__symmetry_pad = False

    def __if_symmetry_pad(self, paddings_tensor, crops_tensor):
        actual_padding_sizes = [[paddings_tensor[i][j] - crops_tensor[i][j] for j in range(len(paddings_tensor[0]))] for i in range(len(paddings_tensor))]

        for index in range(len(actual_padding_sizes)):
            assert len(actual_padding_sizes[index]) == 2
            if (actual_padding_sizes[index][1] - actual_padding_sizes[index][0]) > 0:
                return False
        self.__symmetry_pad = True
        return True

    def resolve_layer(self, graph_matcher, graph_helper):
        matches = []

        # Basic dilated convolution sequence
        self.graph_sequence.set_outputs(['batch_to_space'])
        matches.extend(graph_matcher.match_sequence(self.graph_sequence))

        # Basic dilated convolution sequence with optional bias
        self.graph_sequence.clear_outputs()
        self.graph_sequence.set_inputs('bias', ['batch_to_space', 'bias_source'])
        self.graph_sequence.set_outputs(['bias'])
        matches.extend(graph_matcher.match_sequence(self.graph_sequence))

        # Basic dilated convolution with fakequant and no optional bias
        self.graph_sequence.clear_inputs_for_nodes(['batch_to_space', 'bias'])
        self.graph_sequence.clear_outputs()
        self.graph_sequence.set_inputs('fake_quant', ["conv_op", "min", "max"])
        self.graph_sequence.set_inputs('batch_to_space', ['fake_quant', 'block_shape_out', 'crops'])
        self.graph_sequence.set_outputs(['batch_to_space'])
        matches.extend(graph_matcher.match_sequence(self.graph_sequence))

        # Basic dilated convolution with fakequant and an optional bias
        self.graph_sequence.clear_inputs_for_nodes(['batch_to_space', 'fake_quant'])
        self.graph_sequence.clear_outputs()
        self.graph_sequence.set_inputs('fake_quant', ["conv_op", "min", "max"])
        self.graph_sequence.set_inputs('batch_to_space', ['fake_quant', 'block_shape_out', 'crops'])
        self.graph_sequence.set_inputs('bias', ['batch_to_space', 'bias_source'])
        self.graph_sequence.set_outputs(['bias'])
        matches.extend(graph_matcher.match_sequence(self.graph_sequence))

        descriptors = []
        for match in matches:
            conv_op = match['conv_op']
            bias_op = match['bias'] if 'bias' in match else None
            output_op = conv_op
            strides = conv_op.get_attr(self.TF_ATTRIBUTE_STRIDES)
            padding_size_strategy = conv_op.get_attr(self.TF_ATTRIBUTE_PADDING)
            pads = self.get_spatial_padding(conv_op)
            consumed_nodes = match.consumed_nodes

            weights_source_op = match['kernel']
            weights_tensor = self.get_weights_tensor(graph_helper, weights_source_op)

            bias_source_op = match['bias_source'] if 'bias_source' in match else None
            bias_tensor = self.get_bias_tensor(graph_helper, bias_source_op)

            dilation_sizes = match['dilation_sizes']
            dilation_sizes = graph_helper.evaluate_tensor_output(dilation_sizes.outputs[0])
            if np.shape(dilation_sizes) != (2,):
                raise ConverterError(code_to_message.get_error_message('ERROR_TF_CONV_RESOLVE_DILATION')(conv_op.name))

            paddings_op = match['paddings']
            paddings_tensor = graph_helper.evaluate_tensor_output(paddings_op.outputs[0])
            if np.shape(paddings_tensor) != (2, 2):
                raise ConverterError(code_to_message.get_error_message('ERROR_TF_CONV_RESOLVE_PADDING')(np.shape(paddings_tensor), conv_op.name))

            space_to_batch_is_input = False
            space_to_batch_op = match['space_to_batch']
            batch_to_space_op = match['batch_to_space']

            crop_op = match['crops']
            crops_tensor = graph_helper.evaluate_tensor_output(crop_op.outputs[0])
            if np.shape(crops_tensor) != (2, 2):
                raise ConverterError(code_to_message.get_error_message('ERROR_TF_CONV_RESOLVE_CROP')(np.shape(paddings_tensor), conv_op.name))

            if paddings_tensor.any() and not np.array_equal(paddings_tensor, crops_tensor) \
                    and not self.__if_symmetry_pad(paddings_tensor, crops_tensor):
                # Reshape the padding tensor to be 4D, pad it with 1 before and after
                paddings_tensor = np.pad(paddings_tensor, ((1, 1), (0, 0)), 'constant')
                pad_descriptor = PadLayerResolver.Descriptor(
                    str(space_to_batch_op.name),
                    [space_to_batch_op, match['dilation_sizes'], match['paddings']],
                    paddings_tensor,
                    PadOp.Mode.CONSTANT,
                    0.0,
                    output_names=[str(space_to_batch_op.outputs[0].name)])
                descriptors.append(pad_descriptor)
            else:
                if self.__symmetry_pad:
                    padding_size_strategy = b'SYMMETRY'
                consumed_nodes.extend([space_to_batch_op, paddings_op, match['dilation_sizes']])
                space_to_batch_is_input = True

            crop_descriptor = None
            if crops_tensor.any() and not np.array_equal(paddings_tensor, crops_tensor) and not self.__symmetry_pad:
                crops_tensor = np.pad(crops_tensor, ((1, 1), (0, 0)), 'constant')
                offsets = crops_tensor[:, 0]
                size = np.array(graph_helper.get_op_output_shape(batch_to_space_op), dtype=np.int32)
                crop_descriptor = CropLayerResolver.Descriptor(
                    str(batch_to_space_op.name),
                    [batch_to_space_op, match['block_shape_out'], match['crops']],
                    offsets,
                    size, # Counts should be the same as the output shape
                    size,
                    batch_to_space_op,
                    output_names=[str(batch_to_space_op.outputs[0].name)])
                descriptors.append(crop_descriptor)
            else:
                consumed_nodes.extend([match['block_shape_out'], crop_op, batch_to_space_op])
                output_op = batch_to_space_op

            if bias_op is not None and crop_descriptor:
                bias_desc = IgnoredLayersResolver.Descriptor(str(bias_op.outputs[0].name), [bias_op])
                descriptors.append(bias_desc)
                consumed_nodes.remove(bias_op)

            d = ConvolutionLayerResolver.Descriptor(str(conv_op.name), consumed_nodes,
                                                    conv_op, bias_op, output_op,
                                                    strides, padding_size_strategy,
                                                    explicit_pads=pads,
                                                    weights_tensor=weights_tensor,
                                                    bias_tensor=bias_tensor)

            d.dilationY = int(dilation_sizes[0])
            d.dilationX = int(dilation_sizes[1])
            if space_to_batch_is_input:
                d.input_ops.insert(0, space_to_batch_op)
            descriptors.append(d)

        return descriptors


class DepthwiseConvolutionLayerResolver(ConvolutionLayerResolver, object):
    class Descriptor(ConvolutionLayerResolver.Descriptor):
        pass

    def __init__(self):
        super(DepthwiseConvolutionLayerResolver, self).__init__()

        self.graph_sequence = GraphSequence([
            NonConsumableConverterSequenceNode('input', ['?']),
            NonConsumableConverterSequenceNode('weights_source', ['?']),
            ConverterSequenceNode('root', ['DepthwiseConv2dNative']),
            NonConsumableConverterSequenceNode('bias_source', ['?']),
            ConverterSequenceNode('bias', ['BiasAdd', 'Add'])
        ])
        self.graph_sequence.set_inputs('root', ['input', 'weights_source'])

    def resolve_layer(self, graph_matcher, graph_helper):
        matches = []

        # Basic depthwise conv sequence
        self.graph_sequence.set_outputs(['root'])
        matches.extend(graph_matcher.match_sequence(self.graph_sequence))

        # Basic depthwise conv sequence with bias
        self.graph_sequence.clear_outputs()
        self.graph_sequence.set_inputs('bias', ['root', 'bias_source'])
        self.graph_sequence.set_outputs(['bias'])
        matches.extend(graph_matcher.match_sequence(self.graph_sequence))

        descriptors = []
        for match in matches:
            input_op = match['input']
            conv_op = match['root']
            bias_op = match['bias'] if 'bias' in match else None
            output_op = conv_op
            consumed_nodes = list(match.consumed_nodes)

            weights_source_op = match['weights_source']
            weights_tensor = self.get_weights_tensor(graph_helper, weights_source_op)

            bias_source_op = match['bias_source'] if 'bias_source' in match else None
            bias_tensor = self.get_bias_tensor(graph_helper, bias_source_op)

            # Extract attributes
            strides = conv_op.get_attr(self.TF_ATTRIBUTE_STRIDES)
            padding_size_strategy = conv_op.get_attr(self.TF_ATTRIBUTE_PADDING)
            pads = self.get_spatial_padding(conv_op)

            d = DepthwiseConvolutionLayerResolver.Descriptor(str(conv_op.name), consumed_nodes,
                                                             conv_op, bias_op, output_op,
                                                             strides, padding_size_strategy, explicit_pads=pads,
                                                             weights_tensor=weights_tensor, bias_tensor=bias_tensor)
            d.groups = graph_helper.get_op_output_shape(input_op.outputs[0])[-1]
            descriptors.append(d)

        return descriptors


class DilatedDepthwiseConvolutionLayerResolver(DepthwiseConvolutionLayerResolver, object):
    class Descriptor(DepthwiseConvolutionLayerResolver.Descriptor):
        pass

    def __init__(self):
        super(DilatedDepthwiseConvolutionLayerResolver, self).__init__()
        self.graph_sequence = GraphSequence([
            NonConsumableConverterSequenceNode('inputs', ['?']),
            NonConsumableConverterSequenceNode('dilation_sizes', ['?']),
            NonConsumableConverterSequenceNode('paddings', ['?']),
            ConverterSequenceNode('space_to_batch', ['SpaceToBatchND']),
            NonConsumableConverterSequenceNode('kernel', ['?']),
            ConverterSequenceNode('conv_op', ['DepthwiseConv2dNative']),
            NonConsumableConverterSequenceNode('min', ['?']),
            NonConsumableConverterSequenceNode('max', ['?']),
            ConverterSequenceNode('fake_quant', ['FakeQuantWithMinMaxVars']),
            NonConsumableConverterSequenceNode('bias', ['?']),
            ConverterSequenceNode('biasAdd', ['Add', 'BiasAdd']),
            NonConsumableConverterSequenceNode('block_shape_out', ['?']),
            NonConsumableConverterSequenceNode('crops', ['?']),
            ConverterSequenceNode('batch_to_space', ['BatchToSpaceND'])  # output
        ])
        self.graph_sequence.set_inputs('space_to_batch', ['inputs', 'dilation_sizes', 'paddings'])
        self.graph_sequence.set_inputs('conv_op', ['space_to_batch', 'kernel'])

        self.__symmetry_pad = False

    def __if_symmetry_pad(self, paddings_tensor, crops_tensor):
        actual_padding_sizes = [[paddings_tensor[i][j] - crops_tensor[i][j] for j in range(len(paddings_tensor[0]))] for i in range(len(paddings_tensor))]

        for index in range(len(actual_padding_sizes)):
            assert len(actual_padding_sizes[index]) == 2
            if abs(actual_padding_sizes[index][1] - actual_padding_sizes[index][0]) > 0:
                return False
        self.__symmetry_pad = True
        return True

    def resolve_layer(self, graph_matcher, graph_helper):
        matches = []
        # match sequence with/without quantization node
        # no fake-quant
        self.graph_sequence.set_inputs('batch_to_space', ['conv_op', 'block_shape_out', 'crops'])
        self.graph_sequence.set_outputs(['batch_to_space'])
        matches.extend(graph_matcher.match_sequence(self.graph_sequence))

        # with fake-quant
        self.graph_sequence.clear_inputs_for_nodes(['batch_to_space'])
        self.graph_sequence.clear_outputs()
        self.graph_sequence.set_inputs('fake_quant', ["conv_op", "min", "max"])
        self.graph_sequence.set_inputs('batch_to_space', ['fake_quant', 'block_shape_out', 'crops'])
        self.graph_sequence.set_outputs(['batch_to_space'])
        matches.extend(graph_matcher.match_sequence(self.graph_sequence))

        # SpaceToBatchND->Depthwise Conv->BiasAdd->BatchToSpaceND
        self.graph_sequence.clear_inputs_for_nodes(['batch_to_space'])
        self.graph_sequence.clear_outputs()
        self.graph_sequence.set_inputs('biasAdd', ['bias', 'conv_op'])
        self.graph_sequence.set_inputs('batch_to_space', ['biasAdd', 'block_shape_out', 'crops'])
        self.graph_sequence.set_outputs(['batch_to_space'])
        matches.extend(graph_matcher.match_sequence(self.graph_sequence))

        # SpaceToBatchND->Depthwise Conv->BatchToSpaceND->BiasAdd
        self.graph_sequence.clear_inputs_for_nodes(['biasAdd', 'batch_to_space'])
        self.graph_sequence.clear_outputs()
        self.graph_sequence.set_inputs('batch_to_space', ['conv_op', 'block_shape_out', 'crops'])
        self.graph_sequence.set_inputs('biasAdd', ['bias', 'batch_to_space'])
        self.graph_sequence.set_outputs(['biasAdd'])
        matches.extend(graph_matcher.match_sequence(self.graph_sequence))

        if len(matches) == 0:
            return []
        descriptors = []
        for match in matches:
            conv_op = match['conv_op']
            bias_op = match['biasAdd'] if 'biasAdd' in match else None
            output_op = conv_op

            strides = conv_op.get_attr(self.TF_ATTRIBUTE_STRIDES)
            padding = conv_op.get_attr(self.TF_ATTRIBUTE_PADDING)
            pads = self.get_spatial_padding(conv_op)

            consumed_nodes = match.consumed_nodes

            weights_source_op = match['kernel']
            weights_tensor = self.get_weights_tensor(graph_helper, weights_source_op)

            bias_source_op = match['bias'] if 'bias' in match else None
            bias_tensor = self.get_bias_tensor(graph_helper, bias_source_op)

            dilation_sizes = match['dilation_sizes']
            dilation_sizes = graph_helper.evaluate_tensor_output(dilation_sizes.outputs[0])
            if np.shape(dilation_sizes) != (2,):
                raise ConverterError(code_to_message.get_error_message('ERROR_TF_CONV_RESOLVE_DILATION')(conv_op.name))

            space_to_batch_is_input = False
            space_to_batch_op = match['space_to_batch']
            paddings_op = match['paddings']
            paddings_tensor = graph_helper.evaluate_tensor_output(paddings_op.outputs[0])

            batch_to_space_op = match['batch_to_space']
            crop_op = match['crops']
            crops_tensor = graph_helper.evaluate_tensor_output(crop_op.outputs[0])

            if paddings_tensor.any() and not np.array_equal(paddings_tensor, crops_tensor) \
                    and not self.__if_symmetry_pad(paddings_tensor, crops_tensor):
                # Reshape the padding tensor to be 4D, pad it with 1 before and after
                paddings_tensor = np.pad(paddings_tensor, ((1, 1), (0, 0)), 'constant')
                consumed_nodes.remove(match['space_to_batch'])
                pad_descriptor = PadLayerResolver.Descriptor(
                    str(space_to_batch_op.name),
                    [match['space_to_batch'], match['dilation_sizes'], match['paddings']],
                    paddings_tensor,
                    PadOp.Mode.CONSTANT,
                    0.0,
                    output_names=[str(space_to_batch_op.outputs[0].name)])
                descriptors.append(pad_descriptor)
            else:
                if self.__symmetry_pad:
                    padding = b'SYMMETRY'
                consumed_nodes.extend([space_to_batch_op, paddings_op, match['dilation_sizes']])
                space_to_batch_is_input = True

            crop_descriptor = None
            if crops_tensor.any() and not np.array_equal(paddings_tensor, crops_tensor) and not self.__symmetry_pad:
                crops_tensor = np.pad(crops_tensor, ((1, 1), (0, 0)), 'constant')
                offsets = crops_tensor[:, 0]
                size = np.array(graph_helper.get_op_output_shape(match['batch_to_space']), dtype=np.int32)
                consumed_nodes.remove(match['batch_to_space'])
                crop_descriptor = CropLayerResolver.Descriptor(
                    str(match['batch_to_space'].name),
                    [match['batch_to_space'], match['block_shape_out'], match['crops']],
                    offsets,
                    size, # Counts should be the same as the output shape
                    size,
                    match['batch_to_space'],
                    output_names=[str(match['batch_to_space'].outputs[0].name)])
                descriptors.append(crop_descriptor)
            else:
                consumed_nodes.extend([match['block_shape_out'], crop_op, batch_to_space_op])
                output_op = batch_to_space_op

            if bias_op is not None and crop_descriptor:
                bias_desc = IgnoredLayersResolver.Descriptor(str(bias_op.outputs[0].name), [bias_op])
                descriptors.append(bias_desc)
                consumed_nodes.remove(bias_op)

            d = DilatedDepthwiseConvolutionLayerResolver.Descriptor(
                str(conv_op.name), consumed_nodes, conv_op, bias_op, output_op, strides, padding,
                explicit_pads=pads, weights_tensor=weights_tensor, bias_tensor=bias_tensor)

            d.groups = graph_helper.get_op_output_shape(space_to_batch_op)[-1]
            d.dilationY = int(dilation_sizes[0])
            d.dilationX = int(dilation_sizes[1])
            if space_to_batch_is_input:
                d.input_ops.insert(0, space_to_batch_op)
            descriptors.append(d)

        return descriptors


class ConvolutionLayerBuilder(LayerBuilder):
    def build_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type ir_graph: converters.common.converter_ir.op_graph.IROpGraph
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: ConvolutionLayerResolver.Descriptor
        :rtype: int
        """
        input_names = self.get_input_names(converter_context, descriptor, input_descriptors)
        if not isinstance(descriptor, GroupedConvolutionLayerResolver.Descriptor):
            input_dims = converter_context.graph_helper.get_op_output_shape(descriptor.input_ops[0].inputs[0])
        else:
            # GroupedConvolution has split dimenstions as the first input - second input is activation
            input_dims = converter_context.graph_helper.get_op_output_shape(descriptor.input_ops[0].inputs[1])
        if descriptor.conv_op.type != "FusedResizeAndPadConv2D":
            filter_dims = converter_context.graph_helper.get_op_output_shape(descriptor.conv_op.inputs[1])
        else:
            # FusedResizeAndPadConv2D has activation, size, paddings, weights as inputs, respectively
            filter_dims = converter_context.graph_helper.get_op_output_shape(descriptor.conv_op.inputs[-1])
        output_dims = converter_context.graph_helper.get_op_output_shape(descriptor.output_op)

        # Adds weights constant input to graph, if the weights were statically resolved
        if descriptor.weights_tensor is not None:
            if len(input_names) > 1:
                input_names[1] = descriptor.layer_name + "_weight"
            else:
                input_names.append(descriptor.layer_name + "_weight")
            weights_const_op = ConstantOp(name=input_names[1], tensor=descriptor.weights_tensor)
            ir_graph.add(weights_const_op, [], input_names[1], axis_formats=[AxisTracker.AxisFormat.HWIO])
            weights_buffer = ir_graph.get_buffer(input_names[1])
            weights_buffer.shape = list(descriptor.weights_tensor.shape)

        # First build resize -> pad sequence if Conv is fused
        if descriptor.resize_desc is not None and descriptor.pad_desc is not None:
            resize_desc = descriptor.resize_desc
            pad_desc = descriptor.pad_desc
            resize_output_shape = [input_dims[0],
                                   *converter_context.graph_helper.evaluate_tensor_output(resize_desc.resize_op),
                                   input_dims[-1]]
            ir_graph.add(ResizeOp(resize_desc.output_names[0],
                                  resize_output_shape,
                                  pad_value=0.0,
                                  maintain_aspect_ratio=False,
                                  resize_mode=resize_desc.resize_mode,
                                  scale_height=resize_desc.mul_const[0],
                                  scale_width=resize_desc.mul_const[1],
                                  align_corners=resize_desc.align_corners),
                         input_names=input_descriptors[0].output_names[0],
                         output_names=resize_desc.output_names[0])

            ir_graph.add(PadOp(pad_desc.layer_name,
                               pads=np.asarray(pad_desc.paddings, dtype=np.dtype('int32')),
                               mode=pad_desc.mode,
                               constant_value=float(pad_desc.constant_values)),
                         input_names=resize_desc.output_names[0],
                         output_names=pad_desc.output_names[0])

            input_names[0] = descriptor.pad_desc.output_names[0]
            input_dims = resize_output_shape

        pads, ir_padding_strategy = ConvolutionLayerBuilder.calculate_padding_size(input_size=input_dims[-3:-1],
                                                                                   output_size=output_dims[-3:-1],
                                                                                   strides=descriptor.strides[1:3],
                                                                                   padding_size_strategy=descriptor.padding_size_strategy,
                                                                                   explicit_pads=descriptor.explicit_pads,
                                                                                   filter_dims=filter_dims,
                                                                                   dilation=[descriptor.dilationY,
                                                                                             descriptor.dilationX])

        if descriptor.bias_tensor is not None:
            if len(input_names) > 2:
                input_names[2] = descriptor.layer_name + "_bias"
            else:
                input_names.append(descriptor.layer_name + "_bias")
            bias_const_op = ConstantOp(name=input_names[2], tensor=descriptor.bias_tensor)
            ir_graph.add(bias_const_op, [], input_names[2], axis_formats=[AxisTracker.AxisFormat.ANY])
        elif descriptor.bias_op is not None:
            raise ValueError("Dynamic biases are unsupported for convolution.")

        return ir_graph.add(ConvolutionOp(name=descriptor.layer_name,
                                          bias_op_name=descriptor.bias_op.name if descriptor.bias_op else None,
                                          pady_before=pads[0][0],
                                          pady_after=pads[0][1],
                                          padx_before=pads[1][0],
                                          padx_after=pads[1][1],
                                          padding_mode=PadOp.Mode.ZERO,
                                          padding_size_strategy=ir_padding_strategy,
                                          stridex=int(descriptor.strides[2]),
                                          stridey=int(descriptor.strides[1]),
                                          dilationx=descriptor.dilationX,
                                          dilationy=descriptor.dilationY,
                                          groups=descriptor.groups),
                            input_names,
                            descriptor.output_names[0])

    @classmethod
    def calculate_padding_size(cls, input_size, output_size, strides, padding_size_strategy,
                               filter_dims, dilation, explicit_pads=list([[0, 0], [0, 0]])):

        if padding_size_strategy.decode() in ["SAME", "SYMMETRY"]:
            filter_h = filter_dims[0] + (filter_dims[0] - 1) * (dilation[0] - 1)
            filter_w = filter_dims[1] + (filter_dims[1] - 1) * (dilation[1] - 1)
            pad_y = max(((output_size[0] - 1) * strides[0] + filter_h - input_size[0]), 0)
            pad_x = max(((output_size[1] - 1) * strides[1] + filter_w - input_size[1]), 0)
            # We divide by two and truncate if odd padding given the runtime will
            # take care of Implicit Asymmetry
            pad_y = int(pad_y // 2)
            pad_x = int(pad_x // 2)
            pads = [[pad_y, pad_y], [pad_x, pad_x]]

            if padding_size_strategy.decode() == 'SAME':
                # i.e for odd padding, add the extra padding at the end
                ir_padding_strategy = IRPaddingStrategies.PADDING_SIZE_IMPLICIT_SAME_END
            else:
                ir_padding_strategy = IRPaddingStrategies.PADDING_SIZE_EXPLICIT_FLOOR
        elif padding_size_strategy.decode() == "EXPLICIT":
            pads = explicit_pads
            ir_padding_strategy = IRPaddingStrategies.PADDING_SIZE_EXPLICIT_FLOOR
        elif padding_size_strategy.decode() == 'VALID':
            pads = [[0, 0], [0, 0]]
            ir_padding_strategy = IRPaddingStrategies.PADDING_SIZE_IMPLICIT_VALID
        else:
            raise ValueError("Unsupported TF padding strategy {}".format(padding_size_strategy.decode()))

        return pads, ir_padding_strategy


class GroupedConvolutionLayerResolver(ConvolutionLayerResolver, object):
    class Descriptor(ConvolutionLayerResolver.Descriptor):
        def __init__(self, name, nodes, conv_op, bias_op, output_op, strides, padding_size_strategy, weights, biases,
                     output_names=None, explicit_pads=list([[0, 0], [0, 0]]), split_op=None):
            super(GroupedConvolutionLayerResolver.Descriptor, self).__init__(name, nodes, conv_op, bias_op, output_op,
                                                                             strides, padding_size_strategy,
                                                                             output_names=output_names,
                                                                             explicit_pads=explicit_pads,
                                                                             weights_tensor=weights,
                                                                             bias_tensor=biases)
            self.split_op = split_op

        def is_input_tensor(self, op, tensor):
            if self.split_op is not None and tensor != self.split_op.inputs[1]:
                return False
            return True

    def __init__(self):
        super(GroupedConvolutionLayerResolver, self).__init__()

        # grouped convolution with split
        tree_output_node = ConverterSequenceNode('conv_op', ['Conv2D'])
        self.sequence = GraphSequence([
            ConverterSequenceNode('split_inputs', ['Split']),
            ConverterSequenceNode('split_weights', ['Split']),
            ConverterRepeatableSequenceTreeNode('repeatable_graph', tree_output_node, tree_output_node),
            ConverterSequenceNode('concat_op', ['Concat']),
            ConverterSequenceNode('weights', ['Identity', 'Const']),
            NonConsumableConverterSequenceNode('inputs', ['?']),
            NonConsumableConverterSequenceNode('concat_dim', ['Const']),
            NonConsumableConverterSequenceNode('split_dim1', ['Const']),
            ConverterSequenceNode('split_dim2', ['Const'])
        ])
        self.sequence.set_inputs('split_inputs', ['split_dim1', 'inputs'])
        self.sequence.set_inputs('split_weights', ['split_dim2', 'weights'])
        self.sequence.set_inputs('repeatable_graph', ['split_inputs', 'split_weights'])
        self.sequence.set_inputs('concat_op', ['repeatable_graph', 'concat_dim'])
        self.sequence.set_outputs(['concat_op'])

        # grouped convolution with strided slice
        repeatable_sequence = GraphSequence([
            ConverterSequenceNode('ss', ['StridedSlice']),
            ConverterSequenceNode('ss_begin', ['Const']),
            ConverterSequenceNode('ss_end', ['Const']),
            ConverterSequenceNode('ss_strides', ['Const']),
            ConverterSequenceNode('conv', ['Conv2D']),
            ConverterSequenceNode('bias', ['BiasAdd']),
            ConverterSequenceNode('weights', ['Identity', 'Const']),
            ConverterSequenceNode('biases', ['Identity', 'Const'])
        ])
        repeatable_sequence.set_inputs('ss', ['ss_begin', 'ss_end', 'ss_strides'])
        repeatable_sequence.set_inputs('conv', ['ss', 'weights'])
        repeatable_sequence.set_inputs('bias', ['biases', 'conv'])
        repeatable_sequence.set_outputs(['bias'])

        self.sequence_with_strided_slice = GraphSequence([
            ConverterRepeatableSequenceTreeNode('repeatable_graph',
                                                tree_output_node=repeatable_sequence['bias'],
                                                tree_input_node=repeatable_sequence['ss']),
            ConverterSequenceNode('concat', ['Concat', 'ConcatV2']),
            ConverterSequenceNode('axis', ['Const']),
            NonConsumableConverterSequenceNode('input', ['?'])
        ])
        self.sequence_with_strided_slice.set_inputs('repeatable_graph', ['input'])
        self.sequence_with_strided_slice.set_inputs('concat', ['repeatable_graph', 'axis'])
        self.sequence_with_strided_slice.set_outputs(['concat'])

    def resolve_layer(self, graph_matcher, graph_helper):
        descriptors = []
        for match in graph_matcher.match_sequence(self.sequence):
            conv_op = match['conv_op_1']
            output_op = conv_op
            strides = conv_op.get_attr(self.TF_ATTRIBUTE_STRIDES)
            padding = conv_op.get_attr(self.TF_ATTRIBUTE_PADDING)
            pads = self.get_spatial_padding(conv_op)
            weights = match['weights']
            consumed_nodes = match.consumed_nodes
            output_op_nodes_names = [str(match[node.identifier].outputs[0].name) for node in
                                     self.sequence.output_nodes]
            concat_op = match['concat_op']
            concat_op_output_ops = graph_helper.get_op_outputs(concat_op)
            bias_op, biases = self.get_grouped_conv_bias(graph_helper, concat_op, concat_op_output_ops)
            if bias_op is not None and biases is not None:
                output_op_nodes_names = [str(bias_op.outputs[0].name)]
                consumed_nodes.append(bias_op)
            else:
                bias_op = None
                biases = np.zeros(weights.outputs[0].get_shape()[-1], dtype=np.float32)

            weights = graph_helper.evaluate_tensor_output(weights.outputs[0])
            split_op = match['split_inputs']
            descriptor = GroupedConvolutionLayerResolver.Descriptor(str(conv_op.name), consumed_nodes,
                                                                    conv_op, bias_op, output_op,
                                                                    strides, padding, weights, biases,
                                                                    explicit_pads=pads,
                                                                    output_names=output_op_nodes_names,
                                                                    split_op=split_op)
            descriptor.input_ops = [split_op]
            descriptors.append(descriptor)

        for match in graph_matcher.match_sequence(self.sequence_with_strided_slice):
            if not match.consumed_nodes:
                continue
            input_op = match['input']
            concat_op = match['concat']
            axis_op = match['axis']
            conv_ops = self._get_repeatable_op_by_id(match, 'conv')
            weight_ops = self._get_repeatable_op_by_id(match, 'weights')
            bias_ops = self._get_repeatable_op_by_id(match, 'biases')
            bias_add_ops = self._get_repeatable_op_by_id(match, 'bias')
            ss_ops = self._get_repeatable_op_by_id(match, 'ss')

            input_shape = graph_helper.get_op_output_shape(input_op)
            weight_shapes = [graph_helper.get_op_output_shape(weight_op) for weight_op in weight_ops]

            ss_strides = [graph_helper.evaluate_tensor_output(ss_strides_op.outputs[0]).tolist()
                          for ss_strides_op in self._get_repeatable_op_by_id(match, 'ss_strides')]
            ss_begins = [graph_helper.evaluate_tensor_output(ss_begin_op.outputs[0]).tolist()
                         for ss_begin_op in self._get_repeatable_op_by_id(match, 'ss_begin')]
            ss_ends = [graph_helper.evaluate_tensor_output(ss_end_op.outputs[0]).tolist()
                       for ss_end_op in self._get_repeatable_op_by_id(match, 'ss_end')]

            bias_add_shapes = [graph_helper.get_op_output_shape(bias_add_op) for bias_add_op in bias_add_ops]

            strides = [conv_op.get_attr(self.TF_ATTRIBUTE_STRIDES) for conv_op in conv_ops]
            paddings = [conv_op.get_attr(self.TF_ATTRIBUTE_PADDING) for conv_op in conv_ops]
            pads = [self.get_spatial_padding(conv_op) for conv_op in conv_ops]

            ss_shapes = [graph_helper.get_op_output_shape(ss_op.outputs[0])
                         for ss_op in ss_ops]

            num_groups = len(conv_ops)

            axis = graph_helper.evaluate_tensor_output(axis_op.outputs[0])

            is_grouped_convolution = True
            is_grouped_convolution &= self._elements_are_same(bias_add_shapes)
            is_grouped_convolution &= self._elements_are_same(weight_shapes)
            is_grouped_convolution &= self._elements_are_same(strides)
            is_grouped_convolution &= self._elements_are_same(paddings)
            is_grouped_convolution &= self._elements_are_same(ss_shapes)
            is_grouped_convolution &= self._elements_are_same(ss_strides)
            is_grouped_convolution &= not self._elements_are_same(ss_begins)
            is_grouped_convolution &= not self._elements_are_same(ss_ends)
            # stride slices must evenly divide the last dimension of input to number of groups
            is_grouped_convolution &= ss_shapes[0][-1] * num_groups == input_shape[-1]
            # strides must be all ones at all dimensions
            is_grouped_convolution &= ss_strides[0] == [1] * len(ss_strides[0])
            # concat must be on the last axis in grouped convolution
            is_grouped_convolution &= axis == -1 or axis == (len(bias_add_shapes[0]) - 1)

            if not is_grouped_convolution:
                logging.getLogger().warning(code_to_message.get_error_message('WARNING_TF_GROUP_CONV_RESOLVE'))
                continue

            weight_tensors = [graph_helper.evaluate_tensor_output(weight_op.outputs[0])
                              for weight_op in weight_ops]
            weights = np.concatenate(weight_tensors, axis=-1)

            bias_tensors = [graph_helper.evaluate_tensor_output(bias_op.outputs[0])
                            for bias_op in bias_ops]
            biases = np.concatenate(bias_tensors, axis=-1)

            descriptor = GroupedConvolutionLayerResolver.Descriptor(
                str(concat_op.name), match.consumed_nodes, conv_ops[0], None, conv_ops[0],
                strides[0], paddings[0], weights, biases, explicit_pads=pads[0],
                output_names=[str(concat_op.outputs[0].name)])
            descriptor.input_ops = ss_ops
            descriptor.output_op = concat_op
            descriptors.append(descriptor)

        return descriptors

    @classmethod
    def _get_repeatable_op_by_id(cls, match, name):
        ops = []
        indexed_id = name + '_{}'
        i = 1
        while indexed_id.format(i) in match:
            ops.append(match[indexed_id.format(i)])
            i += 1
        return ops

    @classmethod
    def _elements_are_same(cls, array):
        return all([element == array[0] for element in array])

    def get_grouped_conv_bias(self, graph_helper, input_op, conv_output_ops):
        bias_op, biases = None, None
        try:
            bias_op = graph_helper.filter_op_by_type(conv_output_ops, 'BiasAdd')
        except OperationNotFoundError:
            pass

        if bias_op is None:
            try:
                bias_op = graph_helper.filter_op_by_type(conv_output_ops, 'Add')
            except OperationNotFoundError:
                pass

        if bias_op is not None and graph_helper.check_tensor_const_origin(bias_op.inputs[1])[0]:
            biases = graph_helper.evaluate_tensor_output(bias_op.inputs[1])
        else:
            raise ValueError("Dynamic biases are not supported for grouped convolution.")

        return bias_op, biases


class DepthwiseConvolutionLayerBuilder(ConvolutionLayerBuilder):
    def build_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :param ir_graph: converters.common.converter_ir.op_graph.IROpGraph
        :param converter_context: converters.tensorflow.converter.ConverterContext
        :param descriptor: ConvolutionLayerResolver.Descriptor
        :param input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :param output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :return:
        """
        input_names = self.get_input_names(converter_context, descriptor, input_descriptors)
        input_dims = converter_context.graph_helper.get_op_output_shape(descriptor.input_ops[0].inputs[0])
        filter_dims = converter_context.graph_helper.get_op_output_shape(descriptor.conv_op.inputs[1])
        output_dims = converter_context.graph_helper.get_op_output_shape(descriptor.output_op)

        pads, ir_padding_strategy = ConvolutionLayerBuilder.calculate_padding_size(input_size=input_dims[-3:-1],
                                                                                   output_size=output_dims[-3:-1],
                                                                                   strides=descriptor.strides[1:3],
                                                                                   padding_size_strategy=descriptor.padding_size_strategy,
                                                                                   explicit_pads=descriptor.explicit_pads,
                                                                                   filter_dims=filter_dims,
                                                                                   dilation=[descriptor.dilationY,
                                                                                             descriptor.dilationX])

        # Weights axis format is HWIC, where C == num_output_channels / num_input_channels
        # Converter IR requires an axis format of HWIO, where I is constrained to be 1, so we reshape them here
        if descriptor.weights_tensor is not None:
            if len(input_names) > 1:
                input_names[1] = descriptor.layer_name + "_weight"
            else:
                input_names.append(descriptor.layer_name + "_weight")
            weights_tensor = descriptor.weights_tensor
            weights_tensor = np.reshape(weights_tensor, (weights_tensor.shape[0], weights_tensor.shape[1], 1, -1))
            weights_const_op = ConstantOp(name=input_names[1], tensor=weights_tensor)
            ir_graph.add(weights_const_op, [], input_names[1], axis_formats=[AxisTracker.AxisFormat.HWIO])
            weights_buffer = ir_graph.get_buffer(input_names[1])
            weights_buffer.shape = list(weights_tensor.shape)
        else:
            weights_reshape_name = descriptor.layer_name + "_weights_reshape"
            weights_tensor_size = np.prod(filter_dims)
            weights_spatial_size = filter_dims[0] * filter_dims[1]
            weights_output_shape = [filter_dims[0], filter_dims[1], 1, int(weights_tensor_size / weights_spatial_size)]
            depthwise_reshape_op = ReshapeOp(name=weights_reshape_name,
                                             output_shape=weights_output_shape)
            ir_graph.add(depthwise_reshape_op, input_names[1], weights_reshape_name)
            input_names[1] = weights_reshape_name

        # Add bias tensor depending on whether or not it exists
        if descriptor.bias_tensor is not None:
            if len(input_names) > 2:
                input_names[2] = descriptor.layer_name + "_bias"
            else:
                input_names.append(descriptor.layer_name + "_bias")
            bias_const_op = ConstantOp(name=input_names[2], tensor=descriptor.bias_tensor)
            ir_graph.add(bias_const_op, [], input_names[2], axis_formats=[AxisTracker.AxisFormat.ANY])
        elif descriptor.bias_op is not None:
            raise ValueError("Dynamic biases are unsupported for depthwise convolution.")

        return ir_graph.add(DepthwiseConvolutionOp(name=descriptor.layer_name,
                                                   bias_op_name=descriptor.bias_op.name if descriptor.bias_op else None,
                                                   pady_before=pads[0][0],
                                                   pady_after=pads[0][1],
                                                   padx_before=pads[1][0],
                                                   padx_after=pads[1][1],
                                                   padding_mode=PadOp.Mode.ZERO,
                                                   padding_size_strategy=ir_padding_strategy,
                                                   stridex=int(descriptor.strides[2]),
                                                   stridey=int(descriptor.strides[1]),
                                                   dilationx=descriptor.dilationX,
                                                   dilationy=descriptor.dilationY,
                                                   groups=descriptor.groups),
                            input_names,
                            descriptor.output_names[0])
