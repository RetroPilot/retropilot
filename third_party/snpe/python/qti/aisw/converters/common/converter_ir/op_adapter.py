# ==============================================================================
#
#  Copyright (c) 2018-2022 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import numpy as np
import math
import copy
from abc import ABC, abstractmethod
from enum import Enum
from math import ceil, floor
from typing import List

from qti.aisw.converters.common.utils import translation_utils
from qti.aisw.converters.common.utils import converter_utils
from qti.aisw.converters.common.utils import code_to_message
from qti.aisw.converters.common.converter_ir.axis_tracker import AxisOrder, AxisOrders, AxisTracker


class IRPaddingStrategies(Enum):
    """ Padding size strategies support in IR."""

    # No padding
    PADDING_SIZE_IMPLICIT_VALID = 0
    # Pad input so that output spatial size matches input. In case of odd total
    # pad value across a spatial dimension, the extra padding is place at the beginning.
    PADDING_SIZE_IMPLICIT_SAME_BEGIN = 1
    # Pad input so that output spatial size matches input. In case of odd total
    # pad value across a spatial dimension, the extra padding is place at the end.
    PADDING_SIZE_IMPLICIT_SAME_END = 2
    # padding values are applied only to the right-hand side of the input and floor operation
    # is used to calculate output dims.
    PADDING_SIZE_EXPLICIT_RIGHTHANDED = 3
    # padding values are explicitly specified by source framework and ceil operation is used
    # to calculate output dims
    PADDING_SIZE_EXPLICIT = 4
    # same as explicit, but floor operation is used to calculate output dims
    PADDING_SIZE_EXPLICIT_FLOOR = 5


class Op(ABC):
    @property
    def TRANSLATION_KEY(self):
        raise NotImplementedError

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.__dict__['attrs'] = {}
        return instance

    def __init__(self, name, type, num_outputs=1):
        self.name = name
        self.type = type
        self.num_outputs = num_outputs
        self.macs = 0
        self.params_count = 0  # i.e size of weights

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name

    def addattr(self, key, source, default, use_default_type=True):
        attr = source.get(key, default)
        # Use the default's type when value/type is not None
        if attr is None or type(default) is type(None):
            self.attrs[key] = attr
        else:
            if type(default) is np.ndarray or type(attr) is np.ndarray:
                self.attrs[key] = np.array(attr)
            elif type(default) is type:
                self.attrs[key] = attr
            elif use_default_type:
                self.attrs[key] = type(default)(attr)
            else:
                self.attrs[key] = attr

    def assertattr(self, key, source):
        if key in source:
            self.attrs[key] = source[key]
        else:
            raise KeyError("Op %s missing required argument %s" % (self.name, key))

    def hasattr(self, key):
        return key in self.list_params()

    def __getattr__(self, key):
        try:
            if key in self.__dict__['attrs']:
                return self.__dict__['attrs'][key]
            else:
                return self.__dict__[key]
        except KeyError:
            raise AttributeError("Op %s has no attribute %s" % (self.name, key))

    def __setattr__(self, key, value):
        if key in self.__dict__['attrs']:
            self.__dict__['attrs'][key] = value
        else:
            self.__dict__[key] = value

    @abstractmethod
    def infer_shape(self, input_shapes: list, input_axis_formats: list, num_outputs: int, axis_order: AxisOrder) -> list:
        raise NotImplementedError(
            "infer_shape for {} not implemented ".format(str(self.__class__.__name__)))

    def set_macs_params(self, input_shapes: list, output_shapes: list, axis_order):
        pass

    @staticmethod
    def get_general_macs_val(output_shapes, native_dim_size=3):
        """
        Calculates the macs(multiply and accumulates) value for given Op for the general case
        :param output_shapes: the inferred output shapes for Op
        :param native_dim_size: the dimension to start at for calculating macs value (note: negative of value is used
                                to index the output_dim)
        :return the macs value for Op
        """
        native_output_dims = output_shapes[0][-native_dim_size:]
        return np.prod(native_output_dims)

    def populate_axis_format(self, buf, axis_order, encodings):
        """
        This bypass is provided so that if some Op needs to override a format,
        it can override this function and do so
        :param buf: The opgraph.Buffer class object for assigning axis format
        :param axis_order: the src framework axis order
        :param encodings: the Encodings passed by user for inputs. Used to determine type of network
        """
        buf.populate_axis_format(axis_order, encodings)

    def list_params(self):
        """ This gets instance variables of this class as key/value"""

        instance_vars = dict(self.__dict__)
        # above will get the attrs as {'attrs': {name1:val1...} instead we want to expand that
        del instance_vars['attrs']
        op_attrs = self.attrs
        instance_vars.update(op_attrs)

        return instance_vars

    def is_equal(self, other_op):
        """
        Compares another op instance to current one based on type and attribute matching
        :param other_op: an op_adapter object
        :return: bool, msg. True if type and attr/params match, False otherwise. Plus message detailing what was
                            different
        """
        # instance equality check
        if not isinstance(other_op, self.__class__):
            return False, "{} is not an instance of current Op {}".format(other_op, self.__class__)

        # attr/param list equality check
        other_op_params = other_op.list_params()
        current_op_params = self.list_params()
        if not other_op_params.keys() == current_op_params.keys():
            return False, "Op adapter for {} not set with same attribute as current Op. Expected keys: {}. Got {}". \
                format(type(other_op.type), current_op_params.keys(), other_op_params.keys())
        # loop through attributes. Since we verified above both objects are same instance and have same attrs/params
        # we can use one to list all
        for attr_ in list(current_op_params.keys()):
            if not translation_utils.compare_values(other_op_params[attr_],
                                                    current_op_params[attr_]):
                return False, "Attribute match error for Op: {} Attribute: {}. Expected {}, Got {} ".format(
                    str(other_op.type), attr_, str(current_op_params[attr_]),
                    str(other_op_params[attr_]))

        return True, "Op {} is equal to current Op instance".format(other_op)

    def __eq__(self, other_op):
        return self.is_equal(other_op)[0]

    def update_param_quant_overrides(self, graph, node):
        return


class InputOp(Op):
    TRANSLATION_KEY = 'input'

    def __init__(self, name, shape, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.shape = shape
        self.assertattr('input_encoding_in', kargs)
        self.assertattr('input_encoding_out', kargs)
        self.assertattr('input_type', kargs)
        self.addattr('input_dtype', kargs, np.dtype("float32"))

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return [self.shape[:]]


class ArgMaxOp(Op):
    TRANSLATION_KEY = 'argmax'

    def __init__(self, name, axis, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        if not isinstance(axis, int):
            raise TypeError("Argmax axis is only supported as int, received {}".format(type(axis)))
        self.axis = axis
        self.addattr('keep_dims', kargs, False)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        output_shape = []
        for axis, shape in enumerate(input_shapes[0][:]):
            if axis == self.axis:
                if self.keep_dims:
                    output_shape.append(1)
            else:
                output_shape.append(shape)
        return [output_shape]


class ArgMinOp(Op):
    TRANSLATION_KEY = 'argmin'

    def __init__(self, name, axis, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.axis = axis
        self.addattr('keep_dims', kargs, False)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        output_shape = []
        for axis, shape in enumerate(input_shapes[0][:]):
            if axis == self.axis:
                if self.keep_dims:
                    output_shape.append(1)
                continue
            output_shape.append(shape)
        return [output_shape]


class BatchnormOp(Op):
    TRANSLATION_KEY = 'batchnorm'

    def __init__(self, name, weights, bias, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.weights = weights
        self.bias = bias
        self.addattr('compute_statistics', kargs, False)
        self.addattr('use_mu_sigma', kargs, False)
        self.addattr('across_spatial', kargs, False)
        self.addattr('epsilon', kargs, 1e-9)
        self.addattr('normalize_variance', kargs, True)
        self.addattr('gamma', kargs, np.array([]))
        self.addattr('beta', kargs, np.array([]))

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return [input_shapes[0]]

    def set_macs_params(self, input_shapes: list, output_shapes, axis_order):
        native_dim_size = 3
        if len(input_shapes[0]) in [1, 2]:
            native_dim_size = 1
        macs = self.get_general_macs_val(output_shapes, native_dim_size)
        # for instance cases below the multipliers account for mu and sigma computations
        if self.compute_statistics:
            if self.use_mu_sigma:
                self.macs = macs * 5
            else:
                self.macs = macs * 3
        else:
            self.macs = macs
        # TODO: remove once weights/biases are supported as inputs since it will be counted for in ConstantOp
        self.params_count = self.weights.shape[0] + self.bias.shape[0]


class BatchToSpaceOp(Op):
    TRANSLATION_KEY = 'batch_to_space'

    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr('block_shape', kargs)
        self.addattr('crops', kargs, [[0, 0], [0, 0]])

    def infer_shape(self, input_shapes: List[List[int]], input_axis_formats, num_outputs: int, axis_order) -> List[int]:
        input_batch, input_height, input_width, input_depth = axis_order.extract_spatial_dims(
            input_shapes[0])
        output_batch = input_batch / (self.block_shape[0] * self.block_shape[1])
        output_height = input_height * self.block_shape[0] - (self.crops[0][0] + self.crops[0][1])
        output_width = input_width * self.block_shape[1] - (self.crops[1][0] + self.crops[1][1])
        output_shape = axis_order.format_spatial_output_shape(batch_size=output_batch,
                                                              depth=input_depth,
                                                              height=output_height,
                                                              width=output_width)
        return [output_shape]


class CastOp(Op):
    TRANSLATION_KEY = 'cast'

    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr('to_type', kargs)
        # TODO: change default to previous input_tensor datatype once tensor datatype is tracked in IR
        # Defaulting to assumption of from_type == to_type to continue adhering with IR removal of all casts
        self.addattr('from_type', kargs, self.to_type)
        if not isinstance(self.to_type, str):
            raise TypeError("Cast to_type is expected to be a str, received {}".format(type(self.to_type)))
        if not isinstance(self.from_type, str):
            raise TypeError("Cast from_type is expected to be a str, received {}".format(type(self.from_type)))

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return input_shapes[:num_outputs]


class ChannelShuffleOp(Op):
    TRANSLATION_KEY = 'channel_shuffle'
    GROUPED = "CHANNEL_SHUFFLE_GROUPED"

    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr('groups', kargs)
        self.addattr('shuffle_mode', kargs, self.GROUPED)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return [input_shapes[0]]


class ColorTransformOp(Op):
    TRANSLATION_KEY = 'color_transform'

    def __init__(self, name, shape, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.shape = shape
        self.assertattr('input_encoding_in', kargs)
        self.assertattr('input_encoding_out', kargs)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return [self.shape[:]]


class ConcatOp(Op):
    TRANSLATION_KEY = 'concatenation'

    def __init__(self, name, axis):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.axis = axis

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        output_shape = input_shapes[0][:]

        # Axis parameter needs to be 0 to N-1
        # There is no better place to do this in common and no other place where we have access to input_shape
        if self.axis < 0:
            # Verify that the adjustment in axis will generate valid non-negative axis
            if self.axis + len(input_shapes[0]) < 0:
                raise ValueError("axis provided is {} but length of input_shape is {} for shape {}"
                                 .format(self.axis, len(input_shapes[0]), input_shapes[0]))
            self.axis = self.axis + len(input_shapes[0])

        axis = self.axis
        output_shape[axis] = sum(shape[axis] for shape in input_shapes)

        return [output_shape]


class ConstantOp(Op):
    TRANSLATION_KEY = 'constant'

    def __init__(self, name, tensor, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.tensor = tensor
        self.addattr('quantizable', kargs, True)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return [list(self.tensor.shape)]

    def update_param_quant_overrides(self, graph, node):
        if graph.user_quantization_overrides and self.name in graph.user_quantization_overrides['param_encodings']:
            graph.user_quantization_overrides['activation_encodings'][self.name] = \
                graph.user_quantization_overrides['param_encodings'][self.name]

    def set_macs_params(self, input_shapes: list, output_shapes: list, axis_order):
        self.params_count = np.prod(self.tensor.shape)


class ConvertOp(Op):
    TRANSLATION_KEY = 'convert'

    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.addattr('dynamic_input_data', kargs, False)
        self.addattr('dynamic_output_data', kargs, False)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return [input_shapes[0]]


class ConvolutionOp(Op):
    TRANSLATION_KEY = 'convolution'

    def __init__(self, name, bias_op_name=None, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.bias_op_name = bias_op_name # used for adjusting the bias encoding in ir_graph
        self.assertattr('padx_before', kargs)
        self.assertattr('padx_after', kargs)
        self.assertattr('pady_before', kargs)
        self.assertattr('pady_after', kargs)
        self.assertattr('stridex', kargs)
        self.assertattr('stridey', kargs)
        self.assertattr('dilationx', kargs)
        self.assertattr('dilationy', kargs)
        self.addattr('groups', kargs, 1)
        self.addattr('padding_mode', kargs, PadOp.Mode.ZERO)
        self.addattr('padding_size_strategy', kargs, IRPaddingStrategies.PADDING_SIZE_EXPLICIT_FLOOR)

    @staticmethod
    # Defined to be used for x & y separately
    def calc_same_padding_size(input_size, filter_size, dilation, stride, same_begin=False):
        # For same padding, output_size = input_size / stride
        output_size = ceil(input_size / stride)

        # o = [i + 2 * p - k - (k - 1) * (d - 1)] / s + 1
        kernel_extent = (dilation - 1) * (filter_size - 1)
        pad_total = (output_size - 1) * stride + kernel_extent + filter_size - input_size

        if same_begin:
            pad_begin = ceil(pad_total / 2)
            pad_end = pad_total // 2
        else:
            pad_begin = pad_total // 2
            pad_end = ceil(pad_total / 2)
        return [pad_begin, pad_end]

    @staticmethod
    def calc_conv_padding_size(input_hw, filter_size, dilations, strides, padding_mode, pads):
        """
        :param input_hw: input's height and width
        :param filter_size: type: filter's height and width e.g. 3x3 or 1x1
        :param dilations: dilations for height and width
        :param strides: strides for height and width
        :param padding_mode: type: IRPaddingStrategies
        :param pads: list[int], [pady_before, padx_before, pady_after, padx_after]
        :return: list[int]: actually input padding size for SAME padding mode
        """
        if padding_mode == IRPaddingStrategies.PADDING_SIZE_EXPLICIT_FLOOR:
            return pads
        elif padding_mode == IRPaddingStrategies.PADDING_SIZE_IMPLICIT_VALID:
            return [0, 0, 0, 0]
        elif (padding_mode == IRPaddingStrategies.PADDING_SIZE_IMPLICIT_SAME_BEGIN or
              padding_mode == IRPaddingStrategies.PADDING_SIZE_IMPLICIT_SAME_END):
            filter_h = filter_size[0] + ((filter_size[0] - 1) * (dilations[0] - 1))
            filter_w = filter_size[1] + ((filter_size[1] - 1) * (dilations[1] - 1))
            pad_y = max(((input_hw[0] - 1) * strides[0] + filter_h - input_hw[0]), 0)
            pad_x = max(((input_hw[1] - 1) * strides[1] + filter_w - input_hw[1]), 0)

        # From TF converter:
        # We divide by two and truncate if odd padding given the runtime will
        # take care of Implicit Asymmetry
        pady_before = pad_y // 2
        pady_after = pad_y // 2
        padx_before = pad_x // 2
        padx_after = pad_x // 2
        return [pady_before, padx_before, pady_after, padx_after]

    @staticmethod
    def calc_conv_output_dim(input_size, filter_size, pad_before, pad_after, stride, dilation,
                             padding_size_strategy):
        kernel_extent = filter_size + ((filter_size - 1) * (dilation - 1))
        full_size = float(pad_before + pad_after) + input_size - kernel_extent

        if padding_size_strategy == IRPaddingStrategies.PADDING_SIZE_IMPLICIT_VALID:
            output_dim = ceil(float(input_size - int(kernel_extent) + 1) / float(stride))
        elif padding_size_strategy == IRPaddingStrategies.PADDING_SIZE_IMPLICIT_SAME_BEGIN \
                or padding_size_strategy == IRPaddingStrategies.PADDING_SIZE_IMPLICIT_SAME_END:
            output_dim = ceil(float(input_size) / float(stride))
        elif padding_size_strategy == IRPaddingStrategies.PADDING_SIZE_EXPLICIT_FLOOR:
            output_dim = 1 + floor(full_size / float(stride))
        else:  # EXPLICIT or UNDEFINED
            output_dim = 1 + ceil(full_size / float(stride))

        return int(output_dim)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        batch_size, input_height, input_width, input_depth = axis_order.extract_spatial_dims(input_shapes[0])
        filter_size_height, filter_size_width, _, output_depth = axis_order.extract_conv_weights_dims(input_shapes[1])

        output_height = self.calc_conv_output_dim(input_height,
                                                  filter_size_height,
                                                  self.pady_before,
                                                  self.pady_after,
                                                  self.stridey,
                                                  self.dilationy,
                                                  self.padding_size_strategy)
        output_width = self.calc_conv_output_dim(input_width,
                                                 filter_size_width,
                                                 self.padx_before,
                                                 self.padx_after,
                                                 self.stridex,
                                                 self.dilationx,
                                                 self.padding_size_strategy)

        output_shape = axis_order.format_spatial_output_shape(batch_size=batch_size,
                                                              depth=output_depth,
                                                              height=output_height,
                                                              width=output_width)

        return [output_shape]

    def set_macs_params(self, input_shapes: list, output_shapes: list, axis_order):
        weights_shape = axis_order.extract_conv_weights_dims(input_shapes[1])
        # = filter size * number of filter positions
        self.macs = np.prod(weights_shape[0:3]) * self.get_general_macs_val(output_shapes)

    def update_param_quant_overrides(self, graph, node):
        # Handle cases where FakeQuant encodings have been added directly to the quantization_params
        if graph.quantization_params and self.name in graph.quantization_params:
            param_encodings = graph.quantization_params[self.name]['param_encodings']
            for encoding in param_encodings:
                if encoding['name'] == 'weights':
                    encoding_producer = graph.get_input_buffers(node)[1].producer
                elif len(node.input_names) == 3 and encoding['name'] == 'bias':
                    encoding_producer = graph.get_input_buffers(node)[2].producer
                else:
                    raise ValueError("Encoding for node {} is unhandled.".format(node.op.name))

                graph.add_quantization_params(encoding_producer.op.name,
                                              output_encodings={"name": encoding_producer.op.name,
                                                                "bw": encoding['bw'],
                                                                "min": encoding['min'],
                                                                "max": encoding['max']})


class CropOp(Op):
    TRANSLATION_KEY = 'crop'

    def __init__(self, name, offsets, counts, output_shape):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.offsets = offsets
        self.counts = counts
        self.output_shape = output_shape

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return [self.output_shape[:]]


class CropAndResizeOp(Op):
    TRANSLATION_KEY = 'crop_and_resize'

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr("num_boxes", kwargs)
        self.assertattr("crop_height", kwargs)
        self.assertattr("crop_width", kwargs)
        self.assertattr("interpolation_method", kwargs)
        self.assertattr("extrapolation_value", kwargs)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        _, _, _, depth = axis_order.extract_spatial_dims(input_shapes[0])
        output_shape = axis_order.format_spatial_output_shape(batch_size=self.num_boxes,
                                                              depth=depth,
                                                              height=self.crop_height,
                                                              width=self.crop_width)
        return [output_shape]


class CrossCorrelationOp(Op):
    TRANSLATION_KEY = 'cross_correlation'

    def __init__(self, name):
        Op.__init__(self, name, self.TRANSLATION_KEY)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return [input_shapes[0]]


class CustomOp(Op):
    TRANSLATION_KEY = 'custom'

    def __init__(self,
                 name,
                 package_name,
                 custom_type,
                 inputs,
                 outputs,
                 output_dims,
                 scalar_params,
                 tensor_params,
                 axis_orders):

        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.custom_type = custom_type
        self.output_dims = output_dims
        self.package_name = package_name
        self.axis_orders = axis_orders
        self.inputs = inputs
        self.outputs = outputs
        self.scalar_params = scalar_params
        self.tensor_params = tensor_params

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return self.output_dims

    def populate_axis_format(self, buf, axis_order, encodings):
        # if the axis order has been defined then we keep the format set by the CustomOp object.
        # Otherwise, the axis format will be set according to framework AxisOrder class using
        # the buffer rank when we call populate axis format.
        if self.axis_orders[buf.name] == 'NOT_YET_DEFINED':
            buf.populate_axis_format(axis_order, encodings)
            self.axis_orders[buf.name] = buf.axis_format
        else:
            buf.axis_format = self.axis_orders[buf.name]


class DeconvolutionOp(Op):
    TRANSLATION_KEY = 'deconvolution'

    def __init__(self, name, bias_op_name=None, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.bias_op_name = bias_op_name # used for adjusting the bias encoding in ir_graph
        self.addattr('stridex', kargs, 1)
        self.addattr('stridey', kargs, 1)
        self.addattr('padx_before', kargs, 0)
        self.addattr('padx_after', kargs, 0)
        self.addattr('pady_before', kargs, 0)
        self.addattr('pady_after', kargs, 0)
        self.addattr('padding_size_strategy', kargs, IRPaddingStrategies.PADDING_SIZE_EXPLICIT_FLOOR)
        self.addattr('output_paddingx', kargs, 0)
        self.addattr('output_paddingy', kargs, 0)
        self.addattr('output_height', kargs, 0)
        self.addattr('output_width', kargs, 0)
        self.addattr('groups', kargs, 1)

    @staticmethod
    # Defined to be used for x & y separately
    def calc_same_padding_size(input_size, filter_size, stride, output_padding, same_begin=False):
        # For same padding, output_size = input_size * stride
        output_size = input_size * stride

        # o = (i - 1) * s - 2 * p + k + output_padding
        pad_total = (input_size - 1) * stride + filter_size + output_padding - output_size

        if same_begin:
            pad_begin = ceil(pad_total / 2)
            pad_end = pad_total // 2
        else:
            pad_begin = pad_total // 2
            pad_end = ceil(pad_total / 2)
        return [pad_begin, pad_end]

    @staticmethod
    def calc_deconv_padding_size(filter_size, dilations, strides, padding_mode, pads,
                                 output_padding: list):
        """
        :param filter_size: type: filter's height and width e.g. 3x3 or 1x1
        :param dilations: dilations for height and width
        :param strides: strides for height and width
        :param padding_mode: type: IRPaddingStrategies
        :param pads: list[int], [pady_before, padx_before, pady_after, padx_after]
        :param output_padding: The output padding needed to adjust the pad values. Must be 1x2
        :return: list[int]: actual input padding size for SAME padding mode
        """

        total_padding = [0, 0]
        filter_h = (filter_size[0] - 1) * dilations[0] + 1
        filter_w = (filter_size[1] - 1) * dilations[1] + 1

        total_padding[0] = output_padding[0] - strides[0] + filter_h
        total_padding[1] = output_padding[1] - strides[1] + filter_w

        # In the implicit cases below, the output shape is not fixed so we
        # assume pad value which guarantees that the output shape is same upper or lower
        # which is "output_shape[i] = input_shape[i] * strides[i]"
        # TODO: Implicit same begin and same end are incorrectly switched in the IRPaddingStrategies definition.
        #  This will be replicated here but should be fixed when the other issue is addressed.
        if padding_mode == IRPaddingStrategies.PADDING_SIZE_IMPLICIT_SAME_END:
            pady_before = total_padding[0] - total_padding[0] // 2
            pady_after = total_padding[0] // 2
            padx_before = total_padding[1] - total_padding[1] // 2
            padx_after = total_padding[1] // 2
        elif padding_mode == IRPaddingStrategies.PADDING_SIZE_IMPLICIT_SAME_BEGIN:
            pady_after = total_padding[0] - total_padding[0] // 2
            pady_before = total_padding[0] // 2
            padx_after = total_padding[1] - total_padding[1] // 2
            padx_before = total_padding[1] // 2
        else:  # implicit valid or explicit
            return pads

        return [pady_before, padx_before, pady_after, padx_after]

    @staticmethod
    def calc_deconv_output_dim(input_size, filter_size, pad_before, pad_after, stride,
                               padding_size_strategy, output_padding=0):
        if padding_size_strategy == IRPaddingStrategies.PADDING_SIZE_IMPLICIT_VALID:
            # in this mode, padding of either output or input is not considered
            output_dim = input_size * stride + max(filter_size - stride, 0)
        elif padding_size_strategy == IRPaddingStrategies.PADDING_SIZE_IMPLICIT_SAME_BEGIN \
                or padding_size_strategy == IRPaddingStrategies.PADDING_SIZE_IMPLICIT_SAME_END:
            output_dim = input_size * stride  # output padding is implicitly considered
        else:  # EXPLICIT, EXPLICIT_FLOOR or UNDEFINED
            # output padding needs to be explicitly considered
            output_dim = stride * (input_size - 1) - (pad_before + pad_after) + filter_size + output_padding

        return int(output_dim)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        batch_size, input_height, input_width, _ = axis_order.extract_spatial_dims(input_shapes[0])
        filter_size_height, filter_size_width, _, output_depth = axis_order.extract_deconv_weights_dims(input_shapes[1])

        if self.output_height == 0:
            # calculate according to provided formula
            output_height = self.calc_deconv_output_dim(input_height,
                                                        filter_size_height,
                                                        self.pady_before,
                                                        self.pady_after,
                                                        self.stridey,
                                                        self.padding_size_strategy,
                                                        self.output_paddingy)

            output_width = self.calc_deconv_output_dim(input_width,
                                                       filter_size_width,
                                                       self.padx_before,
                                                       self.padx_after,
                                                       self.stridex,
                                                       self.padding_size_strategy,
                                                       self.output_paddingx)
        else:
            output_height = self.output_height
            output_width = self.output_width

        output_shape = axis_order.format_spatial_output_shape(batch_size=batch_size,
                                                              depth=output_depth * self.groups,
                                                              height=output_height,
                                                              width=output_width)

        return [output_shape]

    def update_param_quant_overrides(self, graph, node):
        # Handle cases where FakeQuant encodings have been added directly to the quantization_params
        if graph.quantization_params and self.name in graph.quantization_params:
            param_encodings = graph.quantization_params[self.name]['param_encodings']
            for encoding in param_encodings:
                if encoding['name'] == 'weights':
                    encoding_producer = graph.get_input_buffers(node)[1].producer
                elif len(node.input_names) == 3 and encoding['name'] == 'bias':
                    encoding_producer = graph.get_input_buffers(node)[2].producer
                else:
                    raise ValueError("Encoding for node {} is unhandled.".format(node.op.name))

                graph.add_quantization_params(encoding_producer.op.name,
                                              output_encodings={"name": encoding_producer.op.name,
                                                                "bw": encoding['bw'],
                                                                "min": encoding['min'],
                                                                "max": encoding['max']})

    def set_macs_params(self, input_shapes: list, output_shapes: list, axis_order):
        weights_shape = axis_order.extract_deconv_weights_dims(input_shapes[1])
        # = filter size * number of filter positions
        self.macs = int(np.prod(weights_shape[0:3]) * self.get_general_macs_val(input_shapes)/self.groups)


class DepthwiseConvolutionOp(ConvolutionOp):
    TRANSLATION_KEY = 'depthwise_convolution'


class DequantizeOp(Op):
    TRANSLATION_KEY = 'dequantize'

    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr('bw', kargs)
        self.addattr('min', kargs, 0.0)
        self.addattr('max', kargs, 0.0)
        self.addattr('scale', kargs, 0.0)
        self.addattr('offset', kargs, 0)
        self.addattr('is_symmetric', kargs, False)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return [input_shapes[0]]


class DetectionOutputOp(Op):
    TRANSLATION_KEY = 'detection_output'

    class PriorBoxType:
        CORNER = "PRIORBOX_TYPE_CORNER"
        CENTER_SIZE = "PRIORBOX_TYPE_CENTER_SIZE"
        CORNER_SIZE = "PRIORBOX_TYPE_CORNER_SIZE"

    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr('output_dims', kargs)
        self.assertattr('num_classes', kargs)
        self.assertattr('share_location', kargs)
        self.assertattr('background_label_id', kargs)
        self.assertattr('nms_threshold', kargs)
        self.assertattr('confidence_threshold', kargs)
        self.assertattr('nms_top_k', kargs)
        self.assertattr('nms_eta', kargs)
        self.assertattr('code_type', kargs)
        self.assertattr('keep_top_k', kargs)
        self.assertattr('variance_encoded_in_target', kargs)
        self.addattr('priorbox_data', kargs, None)  # gets filled out in optimization
        self.addattr('priorbox_center_size_data', kargs, None)  # gets filled out in optimization
        self.addattr('scale_h', kargs, 0)  # gets filled out in optimization
        self.addattr('scale_w', kargs, 0)  # gets filled out in optimization
        self.addattr('scale_y', kargs, 0)  # gets filled out in optimization
        self.addattr('scale_x', kargs, 0)  # gets filled out in optimization

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return self.output_dims[:]


class DropoutOp(Op):
    # TODO: revisit removal of OP per SNPE migration to QNN
    TRANSLATION_KEY = 'dropout'

    def __init__(self, name, keep):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.keep = keep

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return [input_shapes[0]]


class ElementwiseOp(Op):
    def __init__(self, name):
        Op.__init__(self, name, self.TRANSLATION_KEY)

    def infer_shape(self, input_shapes: list, input_axis_formats, num_outputs: int, axis_order) -> list:
        align_channels = False
        # this check is for elementwise binary op
        if len(input_shapes) == 2:
            image_input_idx_and_shapes = [(i, shape)
                                          for i, shape in enumerate(input_shapes)
                                          if input_axis_formats[i] in [AxisTracker.AxisFormat.NSC,
                                                                       AxisTracker.AxisFormat.NCS]]
            # If only 1 input is NCS/NSC and the other is 1D, then the other input (which is likely weights/bias)
            # can be broadcast in IR format along channels without the need to exapnd and match ranks
            if len(image_input_idx_and_shapes) == 1:
                image_data_shape = image_input_idx_and_shapes[0][1]

                non_image_input_idx = 1 - image_input_idx_and_shapes[0][0] # Changes 1 to 0 and 0 to 1
                non_image_data_shape = input_shapes[non_image_input_idx]

                converter_utils.log_debug(
                    "Op {}: Assuming {} is the image input shape, remaining input shape {}".format(self.name,
                                                                                                   image_data_shape,
                                                                                                   non_image_data_shape))
                if len(non_image_data_shape) == 1:
                    _, _, _, input_depth = axis_order.extract_spatial_dims(image_data_shape)
                    if non_image_data_shape[0] == input_depth:
                        align_channels = True
        return [translation_utils.get_broadcasted_shape(input_shapes, axis_order, align_channels=align_channels)]

    def set_macs_params(self, input_shapes: list, output_shapes: list, axis_order):
        self.macs = self.get_general_macs_val(output_shapes)


class ElementwiseAndOp(ElementwiseOp):
    TRANSLATION_KEY = 'elementwise_and'


class ElementwiseDivOp(ElementwiseOp):
    TRANSLATION_KEY = 'elementwise_div'


class ElementwiseEqualOp(ElementwiseOp):
    TRANSLATION_KEY = 'elementwise_equal'


class ElementwiseFloorDivOp(ElementwiseOp):
    TRANSLATION_KEY = 'elementwise_floor_div'


class ElementwiseGreaterOp(ElementwiseOp):
    TRANSLATION_KEY = 'elementwise_greater'


class ElementwiseGreaterEqualOp(ElementwiseOp):
    TRANSLATION_KEY = 'elementwise_greater_equal'


class ElementwiseLessOp(ElementwiseOp):
    TRANSLATION_KEY = 'elementwise_less'


class ElementwiseLessEqualOp(ElementwiseOp):
    TRANSLATION_KEY = 'elementwise_less_equal'


class ElementwiseMaxOp(ElementwiseOp):
    TRANSLATION_KEY = 'elementwise_max'


class ElementwiseMinOp(ElementwiseOp):
    TRANSLATION_KEY = 'elementwise_min'


class ElementwiseNotEqualOp(ElementwiseOp):
    TRANSLATION_KEY = 'elementwise_not_equal'


class ElementwiseOrOp(ElementwiseOp):
    TRANSLATION_KEY = 'elementwise_or'


class ElementwisePowerOp(ElementwiseOp):
    TRANSLATION_KEY = 'elementwise_power'


class ElementwiseProductOp(ElementwiseOp):
    TRANSLATION_KEY = 'elementwise_product'


class ElementwiseSelectOp(ElementwiseOp):
    TRANSLATION_KEY = 'elementwise_select'


class ElementwiseSubOp(ElementwiseOp):
    TRANSLATION_KEY = 'elementwise_sub'


class ElementwiseSumOp(ElementwiseOp):
    TRANSLATION_KEY = 'elementwise_sum'


class ElementwiseUnaryOp(Op):
    def __init__(self, name):
        Op.__init__(self, name, self.TRANSLATION_KEY)

    def infer_shape(self, input_shapes: list, input_axis_formats, num_outputs: int, axis_order) -> list:
        return [input_shapes[0]]


class ElementwiseUnaryAbsOp(ElementwiseUnaryOp):
    TRANSLATION_KEY = 'elementwise_unary_abs'


class ElementwiseUnaryCeilOp(ElementwiseUnaryOp):
    TRANSLATION_KEY = 'elementwise_unary_ceil'


class ElementwiseUnaryExpOp(ElementwiseUnaryOp):
    TRANSLATION_KEY = 'elementwise_unary_exp'


class ElementwiseUnaryFloorOp(ElementwiseUnaryOp):
    TRANSLATION_KEY = 'elementwise_unary_floor'


class ElementwiseUnaryLogOp(ElementwiseUnaryOp):
    TRANSLATION_KEY = 'elementwise_unary_log'


class ElementwiseUnaryNegOp(ElementwiseUnaryOp):
    TRANSLATION_KEY = 'elementwise_unary_neg'


class ElementwiseUnaryNotOp(ElementwiseUnaryOp):
    TRANSLATION_KEY = 'elementwise_unary_not'


class ElementwiseUnaryRoundOp(ElementwiseUnaryOp):
    TRANSLATION_KEY = 'elementwise_unary_round'


class ElementwiseUnaryRsqrtOp(ElementwiseUnaryOp):
    TRANSLATION_KEY = 'elementwise_unary_rsqrt'


class ElementwiseUnarySinOp(ElementwiseUnaryOp):
    TRANSLATION_KEY = 'elementwise_unary_sin'


class ElementwiseUnarySqrtOp(ElementwiseUnaryOp):
    TRANSLATION_KEY = 'elementwise_unary_sqrt'


class EmbeddingOp(Op):
    TRANSLATION_KEY = 'embedding'

    class PartitionStrategy:
        MOD = "EMBEDDING_PARTITION_STRATEGY_MOD"
        DIV = "EMBEDDING_PARTITION_STRATEGY_DIV"

    def __init__(self, name, output_dim, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.output_dim = output_dim
        self.addattr('embedding_strategy', kwargs, self.PartitionStrategy.MOD)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return [self.output_dim]


class ErfOp(Op):
    TRANSLATION_KEY = 'erf'

    def __init__(self, name):
        Op.__init__(self, name, self.TRANSLATION_KEY)

    def infer_shape(self, input_shapes: list, input_axis_formats, num_outputs: int, axis_order) -> list:
        return [input_shapes[0]]


class ExpandOp(Op):
    TRANSLATION_KEY = 'expand'

    def __init__(self, name, output_shape):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        if not isinstance(output_shape, list):
            output_shape = list(output_shape)
        self.output_shape = output_shape

    def infer_shape(self, input_shapes: list, input_axis_formats:list, num_outputs: int, axis_order) -> list:
        return [self.output_shape]


class ExtractGlimpseOp(Op):
    TRANSLATION_KEY = 'extract_glimpse'

    class NoiseType:
        UNIFORM = "NOISE_UNIFORM"
        GAUSSIAN = "NOISE_GAUSSIAN"
        ZERO = "NOISE_ZERO"

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr("glimpse_width", kwargs)
        self.assertattr("glimpse_height", kwargs)
        self.assertattr("centered", kwargs)
        self.assertattr("normalized", kwargs)
        self.assertattr("noise", kwargs)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        input_shape = input_shapes[0][:]
        if len(input_shape) != 4:
            raise ValueError("ExtractGlimpse accepts only 4-D tensor input")
        batch_size, _, _, depth = axis_order.extract_spatial_dims(input_shape)
        output_shape = axis_order.format_spatial_output_shape(batch_size=batch_size,
                                                              depth=depth,
                                                              height=self.glimpse_height,
                                                              width=self.glimpse_width)
        return [output_shape]


class FullyConnectedOp(Op):
    TRANSLATION_KEY = 'fully_connected'

    def __init__(self, name, weights, bias, bias_op_name=None, output_shape=None, transpose_a=False,
                 transpose_b=True):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.weights = weights
        self.bias = bias
        self.bias_op_name = bias_op_name  # used for adjusting the bias encoding in ir_graph
        self.output_shape = output_shape
        self.transpose_a = transpose_a
        self.transpose_b = transpose_b

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        if self.output_shape is None:
            batch = input_shapes[0][0]
            out_channels, in_channels = axis_order.extract_fc_weights_dims(list(self.weights.shape))
            self.output_shape = [batch, out_channels]
        return [self.output_shape]

    def set_macs_params(self, input_shapes: list, output_shapes: list, axis_order):
        self.macs = np.prod(self.weights.shape)  # weight shape accounts for input shape here
        self.params_count = np.prod(self.weights.shape) + self.bias.shape[0]


class GatherOp(Op):
    TRANSLATION_KEY = 'gather'

    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.addattr('axis', kargs, 0)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        output_shape = input_shapes[0][:self.axis] + list(input_shapes[1]) + input_shapes[0][
                                                                             self.axis + 1:]
        return [output_shape]


class GeluOp(Op):
    TRANSLATION_KEY = 'gelu'

    def __init__(self, name):
        Op.__init__(self, name, self.TRANSLATION_KEY)

    def infer_shape(self, input_shapes: list, input_axis_formats, num_outputs: int, axis_order) -> list:
        return [input_shapes[0]]


class GenerateProposalsOp(Op):
    TRANSLATION_KEY = 'generate_proposals'

    def __init__(self, name, anchors, im_info, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr('spatial_scale', kargs)
        self.assertattr('pre_nms_top_n', kargs)
        self.assertattr('post_nms_top_n', kargs)
        self.assertattr('nms_thresh', kargs)
        self.assertattr('min_size', kargs)
        self.addattr('correct_transform_coords', kargs, True)


class GruOp(Op):
    TRANSLATION_KEY = 'gru'

    def __init__(self, name, state_gate, forget_gate, control_gate, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.state_gate = state_gate
        self.forget_gate = forget_gate
        self.control_gate = control_gate
        self.addattr('activation', kargs, NeuronOp.Type.LOGISTIC)
        self.addattr('gate_activation', kargs, NeuronOp.Type.LOGISTIC)
        self.addattr('rec_gate_activation', kargs, NeuronOp.Type.TANH)
        self.addattr('backwards', kargs, False)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        def get_c_h_output_dims(axis_order, batch_size, output_depth):
            if axis_order == AxisOrders.ONNX:
                c_t_dims = [1, batch_size, output_depth]
                h_t_dims = [1, batch_size, output_depth]
            else:
                c_t_dims = [batch_size, output_depth]
                h_t_dims = [batch_size, output_depth]
            return [c_t_dims, h_t_dims]

        input_shape = input_shapes[0][:]
        time_steps = 1
        batch_size = 1
        output_dims = []
        if len(input_shape) == 3:
            batch_size, time_steps, _ = axis_order.extract_time_series_dims(input_shape)
        output_depth = self.control_gate['rec_weights'].shape[1]  # Num of hidden units
        output_dims.append(
            axis_order.format_time_series_output_shape(batch_size=batch_size,
                                                       time_steps=time_steps,
                                                       feature=output_depth)
        )

        if self.c_0_input_name and self.h_0_input_name:
            # Layer has exposed recurrent inputs, therefore we need to add c_T and h_T outputs
            c_dims, h_dims = get_c_h_output_dims(axis_order, batch_size, output_depth)
            output_dims.append(c_dims)
            output_dims.append(h_dims)

        return output_dims


class IdentityOp(Op):
    TRANSLATION_KEY = 'identity'

    def __init__(self, name):
        Op.__init__(self, name, self.TRANSLATION_KEY)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return input_shapes[:num_outputs]


class ImageProjectiveTransformOp(Op):
    TRANSLATION_KEY = 'image_projective_transform'

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.addattr("interpolation_mode", kwargs, "BILINEAR")
        self.addattr("output_shape", kwargs, None)
        if self.output_shape is not None and len(self.output_shape) != 2:
            raise ValueError("Output Shape specified in {0} needs to be 2-D in shape".format(name))

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        num_images, input_height, input_width, num_channels = input_shapes[0][:]
        if self.output_shape:
            return [[num_images, self.output_shape[0], self.output_shape[1], num_channels]]
        else:
            return [[num_images, input_height, input_width, num_channels]]


class L2NormOp(Op):
    TRANSLATION_KEY = 'l2_norm'

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.addattr('axis', kwargs, -1)
        self.addattr('epsilon', kwargs, 1e-12)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return [input_shapes[0]]

    def set_macs_params(self, input_shapes: list, output_shapes: list, axis_order):
        self.macs = self.get_general_macs_val(output_shapes) * 2  # 2 since summation is squared


class LayerNormOp(Op):
    TRANSLATION_KEY = 'layer_norm'
    EPSILON = 1e-9

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr('axes', kwargs)
        self.addattr('epsilon', kwargs, self.EPSILON)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return [input_shapes[0]]

    def set_macs_params(self, input_shapes: list, output_shapes: list, axis_order):
        self.macs = self.get_general_macs_val(output_shapes) * 5  # 5 to account for mu and sigma calculation


class LogSoftmaxOp(Op):
    TRANSLATION_KEY = 'logsoftmax'

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr('axis', kwargs)
        self.addattr('beta', kwargs, 1.0)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return input_shapes[:]


class LstmOp(Op):
    TRANSLATION_KEY = 'lstm'

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr('input_weights', kwargs)
        self.assertattr('gate_bias', kwargs)
        self.assertattr('hidden_state_weights', kwargs)
        self.assertattr('hidden_size', kwargs)
        self.addattr('cell_weights', kwargs, None)
        self.addattr('normalization_weights', kwargs, None)
        self.addattr('w_xc_static', kwargs, None)
        self.addattr('backward', kwargs, False)
        self.addattr('reset_state_at_time_step_0', kwargs, False)
        self.addattr('h_0_input_name', kwargs, '')
        self.addattr('c_0_input_name', kwargs, '')
        self.addattr('sequence_continuation_name', kwargs, '')
        self.addattr('x_static_name', kwargs, '')
        self.addattr('cell_clip_threshold', kwargs, 0.0)
        self.addattr('proj_weights', kwargs, None)
        self.addattr('proj_bias', kwargs, None)
        self.addattr('output_clip_threshold', kwargs, 0.0)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        def get_c_h_output_dims(batch_size, output_depth):
            c_t_dims = [batch_size, output_depth]
            h_t_dims = [batch_size, output_depth]
            return [c_t_dims, h_t_dims]

        input_shape = input_shapes[0][:]
        time_steps = 1
        batch_size = 1
        output_dims = []
        if len(input_shape) == 3:
            batch_size, time_steps, _ = axis_order.extract_time_series_dims(input_shape)
        output_depth = self.hidden_size
        output_dims.append(
            axis_order.format_time_series_output_shape(batch_size=batch_size,
                                                       time_steps=time_steps,
                                                       feature=output_depth)
        )

        if self.c_0_input_name and self.h_0_input_name:
            # Layer has exposed recurrent inputs, therefore we need to add c_T and h_T outputs
            c_dims, h_dims = get_c_h_output_dims(batch_size, output_depth)
            output_dims.append(c_dims)
            output_dims.append(h_dims)

        return output_dims

    def set_macs_params(self, input_shapes: list, output_shapes: list, axis_order):
        input_features = input_shapes[0][-1]
        output_features = output_shapes[0][-1]
        # need BTF order to get steps
        steps = axis_order.permute_shape_to_ir(output_shapes[0])[-2]
        self.macs = (4 * input_features * output_features * steps) + \
                    (4 * output_features * output_features * steps) + \
                    (3 * output_features * steps)
        self.params_count = np.prod(self.input_weights.shape) + \
                            np.prod(self.hidden_state_weights.shape) + \
                            np.prod(self.gate_bias.shape)
        if self.w_xc_static is not None:
            self.params_count += np.prod(self.w_xc_static.shape)


class MatMulOp(Op):
    TRANSLATION_KEY = 'matmul'

    def __init__(self, name, bias, transpose_a, transpose_b, bias_op_name=None):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.bias = bias
        self.transpose_a = transpose_a
        self.transpose_b = transpose_b
        self.bias_op_name = bias_op_name  # used for adjusting the bias encoding in ir_graph

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        if len(input_shapes[0]) == 1 or len(input_shapes[1]) == 1:
            raise ValueError(
                "Shape of matrix must be rank >=2 but rank is 1 for op {}".format(self.name))
        input1 = copy.deepcopy(input_shapes[0])
        input2 = copy.deepcopy(input_shapes[1])

        if self.transpose_a:
            input1[-2], input1[-1] = input1[-1], input1[-2]
        if self.transpose_b:
            input2[-2], input2[-1] = input2[-1], input2[-2]

        # use numpy to determine shape
        output_shape = list(np.matmul(np.random.rand(*input1), np.random.rand(*input2)).shape)
        return [output_shape]

    def set_macs_params(self, input_shapes: list, output_shapes: list, axis_order):
        self.params_count = self.bias.shape[0]
        # macs is m * n * k, where m and n are outer dims after transpose and k is the inner common dim
        k = input_shapes[0][-1]
        if self.transpose_a:
            k = input_shapes[0][-2]
        self.macs = np.prod(input_shapes[0][:-2]) * np.prod(input_shapes[1][:-2]) * k


class MaxYOp(Op):
    TRANSLATION_KEY = 'max_y'

    def __init__(self, name):
        Op.__init__(self, name, self.TRANSLATION_KEY)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        output_shape = input_shapes[0][:]
        if len(output_shape) < 3:
            raise ValueError("MaxY layer expects input shape of at lesat Rank 3")
        idx = len(output_shape) - 3
        output_shape[idx] = 1
        return [output_shape]


class MomentOp(Op):
    TRANSLATION_KEY = 'moment'

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr('axes', kwargs)
        self.addattr('keep_dims', kwargs, False)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        output_shape = []
        input_shape = input_shapes[0]
        for i in range(len(input_shape)):
            dim = input_shape[i]
            if i in self.axes:
                dim = 1
                if self.keep_dims:
                    output_shape.append(dim)
            else:
                output_shape.append(dim)
        return [output_shape, output_shape]


class NeuronOp(Op):
    TRANSLATION_KEY = 'neuron'

    class Type:
        RELU = "NEURON_RELU"
        RELU_MIN_MAX = "NEURON_RELU_MIN_MAX"
        TANH = "NEURON_TANH"
        LOGISTIC = "NEURON_LOGISTIC"
        ELU = "NEURON_ELU"
        HSWISH = "NEURON_HSWISH"
        NONE = "NEURON_NONE"

    def __init__(self, name, neuron_type, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.neuron_type = neuron_type
        self.addattr('a', kargs, 0.0)
        self.addattr('b', kargs, 0.0)
        self.addattr('min_clamp', kargs, 0.0)
        self.addattr('max_clamp', kargs, 0.0)

    @staticmethod
    def extract_activation(activation):
        acts = {'RELU': NeuronOp.Type.RELU,
                'TANH': NeuronOp.Type.TANH,
                'SIGMOID': NeuronOp.Type.LOGISTIC,
                'ELU': NeuronOp.Type.ELU}
        try:
            return acts[str(activation).upper()]
        except KeyError:
            raise ValueError(
                code_to_message.get_error_message("ERROR_ACTIVATION_FUNCTION_UNSUPPORTED")(
                    activation))

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return input_shapes[:]


class NonMaxSuppresionOp(Op):
    TRANSLATION_KEY = 'non_max_suppression'

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr("max_total_detections", kwargs)
        self.addattr("max_detections_per_class", kwargs, self.max_total_detections)
        self.assertattr("iou_threshold", kwargs)
        self.addattr("score_threshold", kwargs, 0.0)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        input_shape = input_shapes[0][:]
        out_dim_1 = [input_shape[0], self.max_total_detections, 4]
        out_dim_2 = [input_shape[0], self.max_total_detections]
        out_dim_3 = [input_shape[0], self.max_total_detections]
        out_dim_4 = [input_shape[0]]

        output_shape = [out_dim_1, out_dim_2, out_dim_3, out_dim_4]

        for i in range(0, len(input_shapes)):
            if i >= 2:
                shape = input_shapes[i][:]
                shape[0] = self.max_total_detections
                output_shape.append(shape)

        return output_shape


class FakeNonMaxSuppressionOp(Op):
    TRANSLATION_KEY = 'fake_non_max_suppression'

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr("max_total_detections", kwargs)
        self.addattr("max_detections_per_class", kwargs, self.max_total_detections)
        self.assertattr("iou_threshold", kwargs)
        self.addattr("score_threshold", kwargs, 0.0)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        output_shape= [[self.max_total_detections, 3]]
        return output_shape


class NoopOp(Op):
    TRANSLATION_KEY = 'noop'

    def __init__(self, name):
        Op.__init__(self, name, self.TRANSLATION_KEY)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return input_shapes[:num_outputs]


class OneHotOp(Op):
    TRANSLATION_KEY = 'one_hot'

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr('depth', kwargs)
        self.assertattr('axis', kwargs)
        self.addattr('on_value', kwargs, 1.0, use_default_type=False)
        self.addattr('off_value', kwargs, 0.0, use_default_type=False)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        axis = self.axis + len(input_shapes[0]) + 1 if self.axis < 0 else self.axis
        output_shape = [list(input_shapes[0][:axis]) + [self.depth] + list(input_shapes[0][axis:])]
        return output_shape


class PackOp(Op):
    TRANSLATION_KEY = 'pack'

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr('axis', kwargs)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        self.axis = self.axis + len(input_shapes[0]) + 1 if self.axis < 0 else self.axis
        output_shape = input_shapes[0][:]
        output_shape.insert(self.axis, len(input_shapes))
        return [output_shape]


class PadOp(Op):
    TRANSLATION_KEY = 'pad'

    class Mode:
        ZERO = "PADDING_ZERO"
        REFLECT = "PADDING_REFLECT"
        CONSTANT = "PADDING_CONSTANT"
        SYMMETRIC = "PADDING_SYMMETRIC"
        EDGE = "PADDING_EDGE"

    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr('pads', kargs)
        self.addattr('mode', kargs, self.Mode.CONSTANT)
        self.addattr('constant_value', kargs, 0.0)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        input_shape = input_shapes[0]
        output_shape = []

        for i in range(0, len(input_shape)):
            output_shape.append(input_shape[i] + self.pads[i][0] + self.pads[i][1])

        return [output_shape]


class PermuteOp(Op):
    TRANSLATION_KEY = 'permute'

    def __init__(self, name, order):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.order = order

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        output_shape = []
        input_shape = input_shapes[0][:]

        for axis in self.order:
            output_shape.append(input_shape[axis])

        return [output_shape]


class PixelShuffleOp(Op):
    TRANSLATION_KEY = 'pixel_shuffle'

    class Mode:
        DCR = "DEPTH_TO_SPACE_DCR"
        CRD = "DEPTH_TO_SPACE_CRD"

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr("upscale_factor", kwargs)
        self.addattr("data_format", kwargs, "NHWC")
        self.addattr("mode", kwargs, PixelShuffleOp.Mode.DCR)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        batch, height, width, depth = axis_order.extract_spatial_dims(input_shapes[0])

        output_shape = axis_order.format_spatial_output_shape(batch_size=batch,
                                                              depth=int(depth / (self.upscale_factor**2)),
                                                              height=height * self.upscale_factor,
                                                              width=width * self.upscale_factor)
        return [output_shape]


class PoolOp(Op):
    TRANSLATION_KEY = 'pool'

    class Type:
        MAX = "POOL_MAX"
        AVG = "POOL_AVG"
        L2 = "POOL_L2"

    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr('pool_type', kargs)
        self.assertattr('size_x', kargs)
        self.assertattr('size_y', kargs)
        self.addattr('stride_x', kargs, 1)
        self.addattr('stride_y', kargs, 1)
        self.addattr('dilation_x', kargs, 1)
        self.addattr('dilation_y', kargs, 1)
        self.addattr('padx_before', kargs, 0)
        self.addattr('padx_after', kargs, 0)
        self.addattr('pady_before', kargs, 0)
        self.addattr('pady_after', kargs, 0)
        self.addattr('padding_size_strategy', kargs, IRPaddingStrategies.PADDING_SIZE_EXPLICIT)
        self.addattr('pool_region_include_padding', kargs, True)

    @staticmethod
    # Defined to be used for x & y separately
    def calc_same_padding_size(input_size, filter_size, dilation, stride, same_begin=False):
        # For same padding, output_size = input_size / stride
        output_size = floor((input_size - 1) / stride) + 1

        ## o = [i + 2 * p - k - (k - 1) * (d - 1)] / s + 1
        kernel_extent = (dilation - 1) * (filter_size - 1)
        pad_total = (output_size - 1) * stride + kernel_extent + filter_size - input_size

        if same_begin:
            pad_begin = ceil(pad_total / 2)
            pad_end = pad_total // 2
        else:
            pad_begin = pad_total // 2
            pad_end = ceil(pad_total / 2)
        return [pad_begin, pad_end]

    @staticmethod
    def calc_pool_output_dim(input_size, pool_size, dilation, pad_before, pad_after, stride,
                             padding_size_strategy):
        padding = -(pad_before + pad_after)
        kernel_extent = pool_size + ((pool_size - 1) * (dilation - 1))
        full_size = float(input_size - padding - kernel_extent)

        if padding_size_strategy == IRPaddingStrategies.PADDING_SIZE_IMPLICIT_VALID:
            output_dim = ceil((1 + full_size) / stride)
        elif padding_size_strategy == IRPaddingStrategies.PADDING_SIZE_IMPLICIT_SAME_BEGIN \
                or padding_size_strategy == IRPaddingStrategies.PADDING_SIZE_IMPLICIT_SAME_END:
            output_dim = ceil((float(input_size) / stride))
        elif padding_size_strategy == IRPaddingStrategies.PADDING_SIZE_EXPLICIT_FLOOR:
            output_dim = 1 + floor(full_size / stride)
        elif padding_size_strategy == IRPaddingStrategies.PADDING_SIZE_EXPLICIT_RIGHTHANDED:
            full_size = float(input_size - padding - pool_size)
            output_dim = 1 + floor(full_size / stride)
        else:  # EXPLICIT or UNDEFINED
            output_dim = 1 + ceil(full_size / stride)

        if (output_dim - 1) * stride + padding >= input_size:
            # don't start a pool beyond the border of the image
            converter_utils.log_debug(
                code_to_message.get_debugging_message("DEBUG_OUTPUT_DIM_BEYOND_BORDER")(output_dim,
                                                                                        output_dim - 1))
            output_dim -= 1

        return int(output_dim)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        batch_size, input_height, input_width, depth = axis_order.extract_spatial_dims(
            input_shapes[0])
        output_height = self.calc_pool_output_dim(input_height,
                                                  self.size_y,
                                                  self.dilation_y,
                                                  self.pady_before,
                                                  self.pady_after,
                                                  self.stride_y,
                                                  self.padding_size_strategy)
        output_width = self.calc_pool_output_dim(input_width,
                                                 self.size_x,
                                                 self.dilation_x,
                                                 self.padx_before,
                                                 self.padx_after,
                                                 self.stride_x,
                                                 self.padding_size_strategy)

        output_shape = axis_order.format_spatial_output_shape(batch_size=batch_size,
                                                              depth=depth,
                                                              height=output_height,
                                                              width=output_width)
        converter_utils.log_debug(
            code_to_message.get_debugging_message("DEBUG_INFERRED_SHAPE")(self.name,
                                                                          output_shape))
        return [output_shape]

    def set_macs_params(self, input_shapes: list, output_shapes: list, axis_order):
        if self.pool_type == self.Type.AVG:
            self.macs = self.get_general_macs_val(output_shapes) * self.size_x * self.size_y
        elif self.pool_type == self.Type.L2:
            self.macs = self.get_general_macs_val(output_shapes)


class L2PoolOp(PoolOp):
    TRANSLATION_KEY = 'l2pool'

    def __init__(self, name, **kargs):
        PoolOp.__init__(self, name, **kargs)
        self.addattr('p', kargs, 2)


class PreluOp(Op):
    TRANSLATION_KEY = 'prelu'

    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr('coeff', kargs)
        self.addattr('channel_shared', kargs, False)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        if len(self.coeff.shape) != 1 and len(self.coeff.shape) != len(input_shapes[0]):
            raise ValueError("Prelu Op ({}) coefficient rank must equal either 1 or input rank {}. Got {} instead."
                             .format(self.name, len(input_shapes[0]), len(self.coeff.shape)))

        prelu_shapes = [input_shapes[0], list(self.coeff.shape)]
        if not translation_utils.are_shapes_broadcastable(prelu_shapes, axis_order, align_channels=not self.channel_shared):
            raise ValueError("Op: ({}), Shape mismatch, {} cannot be broadcast to a single shape".format(self.name,
                                                                                                         prelu_shapes))

        return [input_shapes[0]]

    def set_macs_params(self, input_shapes: list, output_shapes: list, axis_order):
        self.macs = self.get_general_macs_val(output_shapes)
        self.params_count = np.prod(self.coeff.shape)


class ProposalOp(Op):
    TRANSLATION_KEY = 'proposal'

    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr('feat_stride', kargs)
        self.assertattr('scales', kargs)
        self.assertattr('ratios', kargs)
        self.assertattr('anchor_base_size', kargs)
        self.assertattr('min_bbox_size', kargs)
        self.assertattr('max_num_proposals', kargs)
        self.assertattr('max_num_rois', kargs)
        self.assertattr('iou_threshold_nms', kargs)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        output_shape = [1, 1, self.max_num_rois, 5]
        return [output_shape]

    def populate_axis_format(self, buf, axis_order, encodings):
        buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL


class QuantizeOp(Op):
    TRANSLATION_KEY = 'quantize'

    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr('bw', kargs)
        self.addattr('min', kargs, 0.0)
        self.addattr('max', kargs, 0.0)
        self.addattr('scale', kargs, 0.0)
        self.addattr('offset', kargs, 0)
        self.addattr('is_symmetric', kargs, False)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return [input_shapes[0]]


class ReduceOp(Op):
    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr('axes', kargs)
        self.addattr('keep_dims', kargs, False)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        output_shape = []

        # Adjust any negative axes by mapping them to positive values
        input_rank = len(input_shapes[0])
        for i in range(len(self.axes)):
            if self.axes[i] < 0:
                self.axes[i] += input_rank

            if self.axes[i] < 0 or self.axes[i] >= input_rank:
                raise ValueError("Invalid axis value of {} for op {}".format(self.axes[i], self.name))

        # Check for duplicate axis after all negative ones are adjusted
        if len(self.axes) != len(set(self.axes)):
            raise ValueError("Duplicate axis are not permitted in attribute axes, got {}".format(self.axes))

        # Determine the output shape based on the op attributes
        for i, dim in enumerate(input_shapes[0]):
            if i in self.axes:
                # Reduction axes are only preserved if keep_dims is specified
                if self.keep_dims:
                    output_shape.append(1)
            else:
                output_shape.append(dim)

        # Handles scalar values where output_shape would be calculated as an empty list
        if len(output_shape) == 0:
            output_shape = [1]

        return [output_shape]


class ReduceMaxOp(ReduceOp):
    TRANSLATION_KEY = 'reduce_max'


class ReduceMeanOp(ReduceOp):
    TRANSLATION_KEY = 'reduce_mean'


class ReduceMinOp(ReduceOp):
    TRANSLATION_KEY = 'reduce_min'


class ReduceProdOp(ReduceOp):
    TRANSLATION_KEY = 'reduce_prod'


class ReduceSumOp(ReduceOp):
    TRANSLATION_KEY = 'reduce_sum'


class ReshapeOp(Op):
    TRANSLATION_KEY = 'reshape'

    def __init__(self, name, output_shape):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        if not isinstance(output_shape, list):
            output_shape = list(output_shape)
        self.output_shape = output_shape

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return [self.output_shape[:]]


class ResizeOp(Op):
    TRANSLATION_KEY = 'resize'

    class Mode:
        BILINEAR = "RESIZE_BILINEAR"
        NEAREST_NEIGHBOR = "RESIZE_NEAREST_NEIGHBOR"

    def __init__(self, name, output_shape, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.output_shape = output_shape
        self.addattr('pad_value', kargs, 0.0)
        self.addattr('maintain_aspect_ratio', kargs, False)
        self.addattr('resize_mode', kargs, "bilinear")
        self.addattr('scale_height', kargs, 0.0)
        self.addattr('scale_width', kargs, 0.0)
        self.addattr('align_corners', kargs, False)
        self.addattr('half_pixel_centers', kargs, False)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return [self.output_shape[:]]

    def set_macs_params(self, input_shapes: list, output_shapes: list, axis_order):
        if self.resize_mode in "bilinear":
            self.macs = self.get_general_macs_val(output_shapes)


class RNormOp(Op):
    TRANSLATION_KEY = 'rnorm'

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr('size', kwargs)
        self.assertattr('alpha', kwargs)
        self.assertattr('beta', kwargs)
        self.assertattr('k', kwargs)
        self.addattr('across_channels', kwargs, True)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        if not self.across_channels:
            input_rank = len(input_shapes[0])
            if input_rank != 4:
                raise ValueError("Input rank ({}}) must equal 4 for WITHIN LRN on node {}}".format(input_rank,
                                                                                                   self.name))

        return input_shapes[:]

    def set_macs_params(self, input_shapes: list, output_shapes: list, axis_order):
        if self.across_channels:
            self.macs = self.get_general_macs_val(output_shapes) * 3
        else:
            self.macs = self.get_general_macs_val(output_shapes) * (self.size ** 2)


class RoiAlignOp(Op):
    TRANSLATION_KEY = 'roi_align'

    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr('spatial_scale', kargs)
        self.assertattr('pooled_size_h', kargs)
        self.assertattr('pooled_size_w', kargs)
        self.assertattr('sampling_ratio', kargs)
        self.addattr('mode', kargs, 'avg')
        # implode batch parameters
        self.addattr('tiled_batch_h', kargs, -1)
        self.addattr('tiled_batch_w', kargs, -1)
        self.addattr('batch_pad_h', kargs, -1)
        self.addattr('batch_pad_w', kargs, -1)
        self.addattr('pad_value', kargs, 0.0)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        def calc_tiled_height(in_height):
            return self.tiled_batch_h * in_height + (self.tiled_batch_h - 1) * self.batch_pad_h

        def calc_tiled_width(in_width):
            return self.tiled_batch_w * in_width + (self.tiled_batch_w - 1) * self.batch_pad_w

        input_shape = input_shapes[0][:]
        _, _, _, depth = axis_order.extract_spatial_dims(input_shape)

        if self.tiled_batch_h > 0:
            output_shape = axis_order.format_spatial_output_shape(batch_size=1,
                                                                  height=calc_tiled_height(
                                                                      self.pooled_size_h),
                                                                  width=calc_tiled_width(
                                                                      self.pooled_size_w),
                                                                  depth=depth)
        else:
            output_shape = axis_order.format_spatial_output_shape(batch_size=1,
                                                                  height=self.pooled_size_h,
                                                                  width=self.pooled_size_w,
                                                                  depth=depth)
        return [output_shape]

    def set_macs_params(self, input_shapes: list, output_shapes: list, axis_order):
        self.macs = self.get_general_macs_val(output_shapes)


class RoiPoolingOp(Op):
    TRANSLATION_KEY = 'roi_pooling'

    def __init__(self, name, output_shape, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr('pooled_size_h', kargs)
        self.assertattr('pooled_size_w', kargs)
        self.assertattr('spatial_scale', kargs)
        self.output_shape = output_shape

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return [self.output_shape[:]]

    def set_macs_params(self, input_shapes: list, output_shapes: list, axis_order):
        self.macs = self.get_general_macs_val(output_shapes)


class RnnTransformationOp(Op):
    TRANSLATION_KEY = 'rnn_transformation'

    def __init__(self, name, weights, bias, activation):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.weights = weights
        self.bias = bias
        self.activation = activation

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        def get_c_h_output_dims(axis_order, batch_size, output_depth):
            if axis_order == AxisOrders.ONNX:
                c_t_dims = [1, batch_size, output_depth]
                h_t_dims = [1, batch_size, output_depth]
            else:
                c_t_dims = [batch_size, output_depth]
                h_t_dims = [batch_size, output_depth]
            return [c_t_dims, h_t_dims]

        input_shape = input_shapes[0][:]
        time_steps = 1
        batch_size = 1
        output_dims = []
        if len(input_shape) == 3:
            batch_size, time_steps, _ = axis_order.extract_time_series_dims(input_shape)
        output_depth = self.weights.shape[-2]  # Num of hidden units
        output_dims.append(
            axis_order.format_time_series_output_shape(batch_size=batch_size,
                                                       time_steps=time_steps,
                                                       feature=output_depth)
        )

        if self.c_0_input_name and self.h_0_input_name:
            # Layer has exposed recurrent inputs, therefore we need to add c_T and h_T outputs
            c_dims, h_dims = get_c_h_output_dims(axis_order, batch_size, output_depth)
            output_dims.append(c_dims)
            output_dims.append(h_dims)

        return output_dims

    def set_macs_params(self, input_shapes: list, output_shapes: list, axis_order):
        self.macs = self.get_general_macs_val(output_shapes) * 2


class ScaleOp(Op):
    TRANSLATION_KEY = 'scale'

    def __init__(self, name, weights, bias, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.weights = weights
        self.bias = bias
        self.assertattr('axis', kargs)
        self.assertattr('num_axes', kargs)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return [input_shapes[0]]

    def set_macs_params(self, input_shapes: list, output_shapes: list, axis_order):
        self.macs = self.get_general_macs_val(output_shapes)
        # TODO: remove once weights/biases are supported as inputs since it will be counted for in ConstantOp
        self.params_count = np.prod(self.bias.shape)
        if self.weights is not None:
            # weights are provided as params
            self.params_count += np.prod(self.weights.shape)


class ScatterNDOp(Op):
    TRANSLATION_KEY = 'scatter_nd'

    class ReductionTypes(Enum):
        REDUCTION_NONE = "none"
        REDUCTION_ADD = "add"
        REDUCTION_MUL = "mul"

    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.addattr('reduction', kargs, self.ReductionTypes.REDUCTION_NONE)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return [input_shapes[0]]

    def set_macs_params(self, input_shapes: list, output_shapes: list, axis_order):
        self.macs = self.get_general_macs_val(output_shapes)


class SliceOp(Op):
    TRANSLATION_KEY = 'slice'

    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr('axis', kargs)
        self.addattr('slice_points', kargs, [])

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        # slice_points not specified, so generate a slice_points equally sliced between outputs
        if not len(self.slice_points):
            if input_shapes[0][self.axis] % num_outputs:
                raise ValueError(
                    "SliceOp cannot split size {} along input axis {} evenly into {} outputs.".format(
                        input_shapes[0][self.axis], self.axis, num_outputs))
            slice_point_size = input_shapes[0][self.axis] / num_outputs
            for i in range(1, num_outputs):
                self.slice_points.append(int(i * slice_point_size))

        # include 0 and max size to create the extended list of points
        slice_points_extended = [0]
        slice_points_extended.extend(self.slice_points)
        slice_points_extended.append(input_shapes[0][self.axis])

        # simply subtract subsequent slice points to get axis dimension for each output shape
        output_shapes = [list(input_shapes[0][:]) for i in range(len(self.slice_points) + 1)]
        for i in range(len(output_shapes)):
            output_shapes[i][self.axis] = slice_points_extended[i + 1] - slice_points_extended[i]

        return output_shapes


class StridedSliceOp(Op):
    TRANSLATION_KEY = 'strided_slice'

    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr('begin', kargs)
        self.assertattr('end', kargs)
        self.assertattr('strides', kargs)
        self.addattr('begin_mask', kargs, 0)
        self.addattr('end_mask', kargs, 0)
        self.addattr('shrink_axis_mask', kargs, 0)
        self.addattr('new_axis_mask', kargs, 0)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        output_shape = []
        added_dims = 0

        if len(self.begin) < len(input_shapes[0]):
            raise ValueError("Unsupported length for begin in StridedSliceOp. Expected {}. Got {}."
                             .format(len(input_shapes[0]), len(self.begin)))
        if len(self.end) < len(input_shapes[0]):
            raise ValueError("Unsupported length for end in StridedSliceOp. Expected {}. Got {}."
                             .format(len(input_shapes[0]), len(self.end)))
        if len(self.strides) < len(input_shapes[0]):
            raise ValueError("Unsupported length for strides in StridedSliceOp. Expected {}. Got {}."
                             .format(len(input_shapes[0]), len(self.strides)))

        for i in range(len(self.begin)):
            while (self.new_axis_mask & (1 << (i+added_dims))):
                output_shape.append(int(1))
                added_dims += 1
            if not self.shrink_axis_mask & (1 << i+added_dims) and not self.new_axis_mask & (1 << i+added_dims):
                output_shape.append(int(math.ceil(float(self.end[i] - self.begin[i]) / self.strides[i])))
        return [output_shape]


class StaticOp(Op):
    TRANSLATION_KEY = 'static'

    def __init__(self, name):
        Op.__init__(self, name, self.TRANSLATION_KEY)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return []


class SoftmaxOp(Op):
    TRANSLATION_KEY = 'softmax'

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr("axis", kwargs)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        input_rank = len(input_shapes[0])

        if self.axis < 0:
            self.axis += input_rank

        if self.axis < 0 or self.axis >= input_rank:
            raise ValueError("Invalid axis parameter. Got: {}. For node: {}".format(self.axis, self.name))
        return input_shapes[:]


class SpaceToBatchOp(Op):
    TRANSLATION_KEY = 'space_to_batch'

    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr('block_shape', kargs)
        self.addattr('paddings', kargs, [[0, 0], [0, 0]])

    def infer_shape(self, input_shapes: List[List[int]], input_axis_formats, num_outputs: int, axis_order) -> List[int]:
        input_batch, input_height, input_width, input_depth = axis_order.extract_spatial_dims(
            input_shapes[0])
        output_batch = input_batch * self.block_shape[0] * self.block_shape[1]
        output_height = (input_height + self.paddings[0][0] + self.paddings[0][1]) / \
                        self.block_shape[0]
        output_width = (input_width + self.paddings[1][0] + self.paddings[1][1]) / self.block_shape[
            1]
        output_shape = axis_order.format_spatial_output_shape(batch_size=output_batch,
                                                              depth=input_depth,
                                                              height=output_height,
                                                              width=output_width)
        return [output_shape]


class SpaceToDepthOp(Op):
    TRANSLATION_KEY = 'space_to_depth'

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr("downscale_factor", kwargs)
        self.addattr("data_format", kwargs, "NHWC")

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        input_shape = input_shapes[0][:]

        batch, height, width, depth = axis_order.extract_spatial_dims(input_shape)

        output_shape = axis_order.format_spatial_output_shape(batch_size=batch,
                                                              depth=depth * (
                                                                          self.downscale_factor ** 2),
                                                              height=int(
                                                                  height / self.downscale_factor),
                                                              width=int(
                                                                  width / self.downscale_factor))

        return [output_shape]


class SsdOp(Op):
    TRANSLATION_KEY = 'ssd'

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr("scale_y", kwargs)
        self.assertattr("scale_x", kwargs)
        self.assertattr("scale_h", kwargs)
        self.assertattr("scale_w", kwargs)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return [input_shapes[0]]


class SubtractMeanOp(Op):
    TRANSLATION_KEY = 'subtract_mean'

    def __init__(self, name, mean_values):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.mean_values = mean_values

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return [input_shapes[0]]


class TileOp(Op):
    TRANSLATION_KEY = 'tile'

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr("multiples", kwargs)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        input_rank = len(input_shapes[0])
        multiples_len = len(self.multiples)

        if input_rank != multiples_len:
            raise ValueError(
                "Multiples length (%d) doesn't equal input rank (%d) on node %s" % (multiples_len,
                                                                                    input_rank,
                                                                                    self.name))
        return [[input_shapes[0][i] * self.multiples[i] for i in range(input_rank)]]


class TopKOp(Op):
    TRANSLATION_KEY = 'topk'

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr("k", kwargs)
        self.addattr("axis", kwargs, -1)
        self.addattr("largest", kwargs, 1)
        self.addattr("sorted", kwargs, 1)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        input_shape = input_shapes[0][:]
        axis = self.axis + len(input_shapes[0]) if self.axis < 0 else self.axis
        output_shape = [list(input_shape[:axis]) + [self.k] + list(input_shape[axis + 1:])]
        return output_shape * num_outputs


class UdlOp(Op):
    TRANSLATION_KEY = 'udl'

    def __init__(self, name, layer_type, blob, output_dims, expected_input_axis_orders,
                 expected_output_axis_orders):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.layer_type = layer_type
        self.blob = blob
        self.output_dims = output_dims
        self.expected_input_axis_orders = expected_input_axis_orders
        self.expected_output_axis_orders = expected_output_axis_orders

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return self.output_dims


class UnpackOp(Op):
    TRANSLATION_KEY = 'unpack'

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr('axis', kwargs)
        self.addattr('num', kwargs, None)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        self.axis = self.axis + len(input_shapes[0]) if self.axis < 0 else self.axis
        if self.num is None:
            # auto calculate
            self.num = input_shapes[0][self.axis]
        # output shape per tensor is shape of input minus dim for axis and num of outputs is equal
        # to the length of the target axis dim
        output_shape = [[*input_shapes[0][:self.axis], *input_shapes[0][self.axis + 1:]]] * self.num
        return output_shape


class Upsample(ResizeOp):
    TRANSLATION_KEY = "upsample"


class UpsampleIndexBasedOp(Op):
    TRANSLATION_KEY = 'upsample_index_based'

    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr('pool_size', kargs)
        self.addattr('pool_stride', kargs, 1)
        self.addattr('pad', kargs, 0)
        self.addattr('output_height', kargs, -1)
        self.addattr('output_width', kargs, -1)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return [input_shapes[0]]


class UpsampleSparseOp(Op):
    TRANSLATION_KEY = 'upsample_sparse'

    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr('pool_size', kargs)
        self.addattr('pool_stride', kargs, 1)
        self.addattr('pad', kargs, 0)
        self.addattr('output_height', kargs, -1)
        self.addattr('output_width', kargs, -1)

    def infer_shape(self, input_shapes, input_axis_formats, num_outputs, axis_order):
        return [input_shapes[0]]
