# ==============================================================================
#
#  Copyright (c) 2018-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from qti.aisw.converters.common.utils import code_to_message
from qti.aisw.converters.common.utils.converter_utils import *


class AxisTracker(object):
    """
    This class holds all enums, modules and subclasses needed to handle axis tracking between
    source framework to our desired format.
    """
    # Class (an another way of enum class)
    # TBD: Once minimum python version is upgraded for converter from 2.7 to 3.0
    #      replace with enum class
    class AxisAnnotations(object):
        """
        This class contains axis annotations required for axis tracking.
        """
        HEIGHT = 0
        WIDTH = 1
        CHANNEL = 2
        BATCH = 3
        TIME = 4
        FEATURE = 5
        ANY = 6
        # NONTRIVIAL indicates none of axis annotation is valid and not trivial to be derived
        # Layers such as reshape/flatten specify this axis annotation.
        NONTRIVIAL = 7
        # Weights annotations
        INPUT_CHANNELS = 8
        OUTPUT_CHANNELS = 9

    class AxisFormat(object):
        """
        Contains axis commonly used axis orders along with permute order to go to/from this well-defined formats
        """
        # Batch,Channel,Spatial. With one batch and two spatial dimensions,
        # equivalent to NCHW
        NCS = 'NCS'
        # Batch,Spatial,Channel. With one batch and two spatial dimensions,
        # equivalent to NHWC. This is the native data order for SNPE ops which
        # output feature maps.
        NSC = 'NSC'
        # Time,Batch,Feature.
        TBF = 'TBF'
        # Batch,Time,Feature. This is the native data order for SNPE RNN ops.
        BTF = 'BTF'
        # Batch,Feature.
        FEATURE = 'FEATURE'
        # used by Constant Op
        ANY = 'ANY'
        # Op specific data format.
        NONTRIVIAL = 'NONTRIVIAL'
        # Enum value used by buffers which have not yet undergone axis tracking.
        NOT_YET_DEFINED = 'NOT_YET_DEFINED'

        # well-known permute orders
        NCS_TO_NSC = [0, 2, 3, 1]
        NSC_TO_NCS = [0, 3, 1, 2]
        TBF_TO_BTF = BTF_TO_TBF = [1, 0, 2]

        # Weight axis formats
        HWIO = "HWIO"
        HWOI = "HWOI"
        IOHW = "IOHW"
        OIHW = "OIHW"

        # Weight permute orders
        IOHW_TO_HWIO = HWIO_TO_IOHW = OIHW_TO_HWOI = [2, 3, 0, 1]
        OIHW_TO_HWIO = [2, 3, 1, 0]
        HWIO_TO_OIHW = [3, 2, 0, 1]
        HWOI_TO_HWIO = [0, 1, 3, 2]

    @classmethod
    def get_axis_annotation_from_format(cls, axis_format):
        if axis_format == cls.AxisFormat.NCS:
            return [AxisTracker.AxisAnnotations.BATCH, AxisTracker.AxisAnnotations.CHANNEL,
                    AxisTracker.AxisAnnotations.HEIGHT, AxisTracker.AxisAnnotations.WIDTH]
        elif axis_format == cls.AxisFormat.NSC:
            return [AxisTracker.AxisAnnotations.BATCH, AxisTracker.AxisAnnotations.HEIGHT,
                    AxisTracker.AxisAnnotations.WIDTH, AxisTracker.AxisAnnotations.CHANNEL]
        elif axis_format == cls.AxisFormat.TBF:
            return [AxisTracker.AxisAnnotations.TIME, AxisTracker.AxisAnnotations.BATCH,
                    AxisTracker.AxisAnnotations.FEATURE]
        elif axis_format == cls.AxisFormat.BTF:
            return [AxisTracker.AxisAnnotations.BATCH, AxisTracker.AxisAnnotations.TIME,
                    AxisTracker.AxisAnnotations.FEATURE]
        elif axis_format == cls.AxisFormat.FEATURE:
            return [AxisTracker.AxisAnnotations.BATCH, AxisTracker.AxisAnnotations.FEATURE]
        elif axis_format == cls.AxisFormat.NONTRIVIAL:
            return [AxisTracker.AxisAnnotations.NONTRIVIAL]

        raise ValueError("Unknown axis format {}" % axis_format)

    @classmethod
    def get_axis_format_from_annotation(cls, axis_annotation):
        if axis_annotation == [cls.AxisAnnotations.BATCH, cls.AxisAnnotations.CHANNEL,
                               cls.AxisAnnotations.HEIGHT, cls.AxisAnnotations.WIDTH]:
            return cls.AxisFormat.NCS
        elif axis_annotation == [cls.AxisAnnotations.BATCH, cls.AxisAnnotations.HEIGHT,
                                 cls.AxisAnnotations.WIDTH, cls.AxisAnnotations.CHANNEL]:
            return cls.AxisFormat.NSC
        elif axis_annotation == [cls.AxisAnnotations.TIME, cls.AxisAnnotations.BATCH,
                                 cls.AxisAnnotations.FEATURE]:
            return cls.AxisFormat.TBF
        elif axis_annotation == [cls.AxisAnnotations.BATCH, cls.AxisAnnotations.TIME,
                                 cls.AxisAnnotations.FEATURE]:
            return cls.AxisFormat.BTF
        elif axis_annotation == [cls.AxisAnnotations.BATCH, cls.AxisAnnotations.FEATURE]:
            return cls.AxisFormat.FEATURE
        else:
            return cls.AxisFormat.NONTRIVIAL

    @classmethod
    def get_permute_order(cls, src_order, target_order, rank):
        if src_order == cls.AxisFormat.NCS:
            if target_order == cls.AxisFormat.NSC:
                if rank == 4:
                    return cls.AxisFormat.NCS_TO_NSC
                num_spatial = rank-2
                return [0] + [i+2 for i in range(num_spatial)] + [1]
        elif src_order == cls.AxisFormat.NSC:
            if target_order == cls.AxisFormat.NCS:
                if rank == 4:
                    return cls.AxisFormat.NSC_TO_NCS
                num_spatial = rank-2
                return [0, rank-1] + [i+1 for i in range(num_spatial)]
        elif src_order == cls.AxisFormat.TBF:
            if target_order == cls.AxisFormat.BTF:
                return cls.AxisFormat.TBF_TO_BTF
        elif src_order == cls.AxisFormat.BTF:
            if target_order == cls.AxisFormat.TBF:
                return cls.AxisFormat.BTF_TO_TBF
        else:
            raise ValueError("No permutation from %s to %s" % (src_order, target_order))

    @staticmethod
    def compute_permute_order(current_order, expected_order):
        log_debug("Current Axes=" + str(current_order) + " Expected Axes=" + str(expected_order))
        log_assert(set(current_order) == set(expected_order),
                   "Error: computing permute order for current and expected axes orders: values do not match;"
                   " Current order " + str(current_order) + " Expected order:" + str(expected_order) +
                   ". Make sure you are using correct Axis Annotations for orders.")
        permute_order = []
        for axis in expected_order:
            permute_order.append(current_order.index(axis))
        return permute_order

    @staticmethod
    def permute_shape(shape, order):
        return [shape[i] for i in order]

    @classmethod
    def enforce_input_type(cls, graph, input_name, op_name, target_format, permute_order):
        input_buf = graph.get_buffer(input_name)
        if input_buf.axis_format in [cls.AxisFormat.ANY, cls.AxisFormat.FEATURE, cls.AxisFormat.HWIO]:
            pass
        elif input_buf.axis_format == cls.AxisFormat.NONTRIVIAL:
            if input_buf.rank() == len(permute_order):
                graph.inject_implicit_permute(input_name, op_name, target_format, permute_order)
            else:
                log_debug2("inject_implicit_permute ignored for NONTRIVIAL axis format due to rank"
                           "({}) and permute_order({}) mismatch for input name: {}",
                           input_buf.rank(), len(permute_order), input_name)
        elif input_buf.axis_format != target_format:
            raise ValueError(code_to_message.get_error_message('ERROR_INPUT_DATA_ORDER_UNEXPECTED')
                             (input_name, target_format, input_buf.axis_format))

    @classmethod
    def image_to_spatial_first_order(cls, node, graph):
        """Axis transformation for layers which take in and emit only image-valued data"""
        cls.log_axes_to_spatial_first_order(node, graph)

        # (1) if any of our inputs are NONTRIVIAL, put a permute
        # of NCS -> NSC in front of them. This will be shared
        # with everyone who consumes that buffer, so don't specify consumers
        for name in node.input_names:
            # fetch input buffers one by one to avoid degenerate case where
            # an op uses the same input more than once and needs to permute it.
            cls.enforce_input_type(graph, name, node.op.name, cls.AxisFormat.NSC, cls.AxisFormat.NCS_TO_NSC)

        input_buffers = graph.get_input_buffers(node)
        input_orders = [buf.axis_format for buf in input_buffers]
        if cls.AxisFormat.NONTRIVIAL in input_orders:
            # Update output buffers to NONTRIVIAL when enforce_input_type fails to
            # inject implicit permute for NONTRIVIAL input buffer
            for buf in graph.get_output_buffers(node):
                buf.axis_format = cls.AxisFormat.NONTRIVIAL
        else:
            # (2) Update all of our output buffers to be in NSC order.Output buffer is not
            # explicitly checked, it is assumed to be in NCS order.
            for buf in graph.get_output_buffers(node):
                if buf.axis_format == cls.AxisFormat.NONTRIVIAL:
                    continue
                buf.shape = cls.permute_shape(buf.shape, cls.AxisFormat.NCS_TO_NSC)
                buf.axis_format = cls.AxisFormat.NSC
                node.op.output_shape = buf.shape

    @classmethod
    def feature_to_spatial_first_order(cls, node, graph):
        # Not much to do here, just mark the outputs
        for buf in graph.get_output_buffers(node):
            buf.axis_format = cls.AxisFormat.FEATURE

    @classmethod
    def time_series_to_spatial_first_order(cls, node, graph):
        for name in node.input_names:
            cls.enforce_input_type(graph, name, node.op.name, cls.AxisFormat.BTF, cls.AxisFormat.TBF_TO_BTF)

        for buf in graph.get_output_buffers(node):
            if buf.rank() == 3:
                buf.shape = cls.permute_shape(buf.shape, cls.AxisFormat.TBF_TO_BTF)
                buf.axis_format = cls.AxisFormat.BTF
            elif buf.rank() == 4:
                buf.axis_format = cls.AxisFormat.NSC

    @classmethod
    def eltwise_to_spatial_first_order(cls, node, graph):
        input_buffers = graph.get_input_buffers(node)
        input_orders = [buf.axis_format for buf in input_buffers]
        if cls.AxisFormat.NSC in input_orders:
            cls.image_to_spatial_first_order(node, graph)
        elif cls.AxisFormat.BTF in input_orders:
            cls.time_series_to_spatial_first_order(node, graph)
        elif cls.AxisFormat.FEATURE in input_orders:
            cls.feature_to_spatial_first_order(node, graph)
        else:
            # well hopefully someone knows
            for buf in graph.get_output_buffers(node):
                buf.axis_format = cls.AxisFormat.NONTRIVIAL

    @staticmethod
    def log_axes_to_spatial_first_order(node, graph):
        log_debug(code_to_message.get_debugging_message("DEBUG_AXES_TO_SPATIAL_FIRST_ORDER_ENTRY")
                  (node.op.name))
        for input_name in node.input_names:
            log_debug(
                      code_to_message.get_debugging_message("DEBUG_AXES_TO_SPATIAL_FIRST_ORDER_INPUT_SIZE")
                      (input_name, str(graph.get_buffer(input_name).shape)))


class AxisOrder(object):
    def __init__(self):
        # Default to SNPE order
        self.axis_formats = [
            AxisTracker.AxisFormat.ANY,
            AxisTracker.AxisFormat.FEATURE,
            AxisTracker.AxisFormat.BTF,
            AxisTracker.AxisFormat.NSC
        ]
        # permute_sequence - Contains the permute sequence from IR order to self order
        # permute_sequence_to_ir - Contains the permute sequence from self order to IR order
        self.permute_sequence = self.permute_sequence_to_ir = [
            [0],
            [0, 1],
            [0, 1, 2],
            [0, 1, 2, 3]
        ]

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        self_vars = dict(self.__dict__)
        other_vars = dict(other.__dict__)
        for var in list(self_vars.keys()):
            if self_vars[var] != other_vars[var]:
                return False
        return True

    def permute_shape_to_ir(self, shape: list) -> list:
        try:
            order = self.permute_sequence_to_ir[len(shape)-1]
            return AxisTracker.permute_shape(shape, order)
        except IndexError:
            raise ValueError("Unable to permute shape {} to NSC ordering".format(shape))

    def permute_shape_from_ir(self, shape: list) -> list:
        try:
            order = self.permute_sequence[len(shape)-1]
            return AxisTracker.permute_shape(shape, order)
        except IndexError:
            raise ValueError("Unable to permute shape {} to NCS ordering".format(shape))

    def get_axis_format(self, rank):
        if 5 > rank > 0:
            return self.axis_formats[rank-1]
        else:
            return AxisTracker.AxisFormat.NONTRIVIAL

    def get_permute_order(self, rank, src_axis_order, target_axis_order):
        if rank < 5:
            return self.permute_sequence[rank - 1]
        else:
            raise ValueError("No permute order defined for Src Order {0}, Target Order {1} and Rank {2}"
                             .format(src_axis_order, target_axis_order, rank))

    @classmethod
    def extract_time_series_dims(cls, shape):
        if len(shape) != 3:
            raise ValueError("Shape needs to be of length 3, passed {}".format(shape))
        return shape[:]

    @classmethod
    def format_time_series_output_shape(cls, batch_size, time_steps, feature):
        return [batch_size, time_steps, feature]

    @classmethod
    def extract_spatial_dims(cls, shape):
        if len(shape) != 4:
            raise ValueError("Shape needs to be of length 4, passed {}".format(shape))
        return shape[:]

    @classmethod
    def format_spatial_output_shape(cls, batch_size, height, width, depth):
        return [batch_size, height, width, depth]

    @classmethod
    def extract_conv_weights_dims(cls, weights_shape):
        if len(weights_shape) != 4:
            raise ValueError("Shape needs to be of length 4, passed {}".format(weights_shape))
        # weights are expected to be in shape [HWIO]
        return weights_shape[:]


    @classmethod
    def extract_deconv_weights_dims(cls, weights_shape):
        if len(weights_shape) != 4:
            raise ValueError("Shape needs to be of length 4, passed {}".format(weights_shape))
        # Note: This function will need to be updated once handling of HWOI_TO_HWIO transpose is not handled by the
        #       Tensorflow front-end. This is because deconv weights are HWOI in Tensorflow and expected output format
        #       from this function is HWIO.
        return weights_shape[:]

    @classmethod
    def extract_fc_weights_dims(cls, weights_shape):
        if len(weights_shape) != 2:
            raise ValueError("Shape needs to be of length 2, passed {}".format(weights_shape))
        # weights are expected to be in shape [out_channels, in_channels]
        return weights_shape[:]


class TfAxisOrder(AxisOrder):
    def __init__(self):
        # TF is same as SNPE order, Do Nothing
        super(TfAxisOrder, self).__init__()


class OnnxAxisOrder(AxisOrder):
    def __init__(self):
        self.axis_formats = [
            AxisTracker.AxisFormat.ANY,
            AxisTracker.AxisFormat.FEATURE,
            AxisTracker.AxisFormat.TBF,
            AxisTracker.AxisFormat.NCS
        ]
        # Contains the permute sequence from IR order to source framework order
        self.permute_sequence = [
            [0],
            [0, 1],
            [1, 0, 2],
            [0, 3, 1, 2]
        ]
        # Contains the permute sequence from source framework order to IR order
        self.permute_sequence_to_ir = [
            [0],
            [0, 1],
            [1, 0, 2],
            [0, 2, 3, 1]
        ]

    @classmethod
    def extract_time_series_dims(cls, shape):
        if len(shape) != 3:
            raise ValueError("Shape needs to be of length 3, passed {}".format(shape))
        time_steps, batch_size, feature = shape[:]
        return [batch_size, time_steps, feature]

    @classmethod
    def format_time_series_output_shape(cls, batch_size, time_steps, feature):
        return [time_steps, batch_size, feature]

    @classmethod
    def extract_spatial_dims(cls, shape):
        if len(shape) != 4:
            raise ValueError("Shape needs to be of length 4, passed {}".format(shape))
        batch_size, depth, height, width = shape[:]
        return [batch_size, height, width, depth]

    @classmethod
    def format_spatial_output_shape(cls, batch_size, height, width, depth):
        return [batch_size, depth, height, width]

    @classmethod
    def extract_conv_weights_dims(cls, weights_shape):
        if len(weights_shape) != 4:
            raise ValueError("Shape needs to be of length 4, passed {}".format(weights_shape))
        out_channels, in_channels, height, width = weights_shape[:]
        return [height, width, in_channels, out_channels]

    @classmethod
    def extract_deconv_weights_dims(cls, weights_shape):
        if len(weights_shape) != 4:
            raise ValueError("Shape needs to be of length 4, passed {}".format(weights_shape))
        in_channels, out_channels, height, width = weights_shape[:]
        return [height, width, in_channels, out_channels]

    @classmethod
    def extract_fc_weights_dims(cls, weights_shape):
        if len(weights_shape) != 2:
            raise ValueError("Shape needs to be of length 2, passed {}".format(weights_shape))
        in_channels, out_channels = weights_shape[:]
        return [out_channels, in_channels]


class CaffeAxisOrder(AxisOrder):
    def __init__(self):
        self.axis_formats = [
            AxisTracker.AxisFormat.ANY,
            AxisTracker.AxisFormat.FEATURE,
            AxisTracker.AxisFormat.TBF,
            AxisTracker.AxisFormat.NCS
        ]
        # Contains the permute sequence from IR order to source framework order
        self.permute_sequence = [
            [0],
            [0, 1],
            [1, 0, 2],
            [0, 3, 1, 2]
        ]
        # Contains the permute sequence from source framework order to IR order
        self.permute_sequence_to_ir = [
            [0],
            [0, 1],
            [1, 0, 2],
            [0, 2, 3, 1]
        ]

    @classmethod
    def extract_time_series_dims(cls, shape):
        if len(shape) != 3:
            raise ValueError("Shape needs to be of length 3, passed {}".format(shape))
        time_steps, batch_size, feature = shape[:]
        return [batch_size, time_steps, feature]

    @classmethod
    def format_time_series_output_shape(cls, batch_size, time_steps, feature):
        return [time_steps, batch_size, feature]

    @classmethod
    def extract_spatial_dims(cls, shape):
        if len(shape) != 4:
            raise ValueError("Shape needs to be of length 4, passed {}".format(shape))
        batch_size, depth, height, width = shape[:]
        return [batch_size, height, width, depth]

    @classmethod
    def format_spatial_output_shape(cls, batch_size, height, width, depth):
        return [batch_size, depth, height, width]

    @classmethod
    def extract_conv_weights_dims(cls, weights_shape):
        if len(weights_shape) != 4:
            raise ValueError("Shape needs to be of length 4, passed {}".format(weights_shape))
        out_channels, in_channels, height, width = weights_shape[:]
        return [height, width, in_channels, out_channels]

    @classmethod
    def extract_deconv_weights_dims(cls, weights_shape):
        if len(weights_shape) != 4:
            raise ValueError("Shape needs to be of length 4, passed {}".format(weights_shape))
        in_channels, out_channels, height, width = weights_shape[:]
        return [height, width, in_channels, out_channels]

    @classmethod
    def extract_fc_weights_dims(cls, weights_shape):
        if len(weights_shape) != 2:
            raise ValueError("Shape needs to be of length 2, passed {}".format(weights_shape))
        in_channels, out_channels = weights_shape[:]
        return [out_channels, in_channels]


class AxisOrders(object):
    TF = TfAxisOrder()
    ONNX = OnnxAxisOrder()
    CAFFE = CaffeAxisOrder()
