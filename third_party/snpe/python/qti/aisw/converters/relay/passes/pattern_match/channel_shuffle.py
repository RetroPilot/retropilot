# ==============================================================================
#
#  Copyright (c) 2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import tvm

from tvm.relay.dataflow_pattern import *
from tvm.relay.testing import run_infer_type

from qti.aisw.converters.common.utils.converter_utils import log_debug3


@tvm.ir.transform.module_pass(opt_level=3)
class IdentifyChannelShuffle:
    def __init__(self, data_layout):
        assert data_layout in ["NCHW", "NHWC"], "Unsupported data layout."
        self.data_layout = data_layout

    def transform_module(self, mod, ctx):
        data_layout = self.data_layout

        if data_layout == "NCHW":
            transpose_axes = [0, 2, 1, 3, 4]
        else:  # data_layout == "NHWC"
            transpose_axes = [0, 1, 2, 4, 3]

        class MatchAndRewrite(DFPatternCallback):
            def __init__(self):
                super(MatchAndRewrite, self).__init__(require_type=True)

                self._channel_shuffle_op = tvm.relay.op.op.get("channel_shuffle")

                # Match following pattern to channel_shffle op:
                # %27 = reshape(%26, newshape=[1, 2, 58, 28, 28]) /* ty=Tensor[(1, 2, 58, 28, 28), float32] */;
                # %28 = transpose(%27, axes=[0, 2, 1, 3, 4]) /* ty=Tensor[(1, 58, 2, 28, 28), float32] */;
                # %29 = reshape(%28, newshape=[1, -1, 28, 28]) /* ty=Tensor[(1, 116, 28, 28), float32] */;
                self._data = wildcard()
                self._reshape1 = is_op("reshape")(self._data)
                self._transpose = is_op("transpose")(self._reshape1).has_attr(
                    {"axes": transpose_axes}
                )
                self._reshape2 = is_op("reshape")(self._transpose)

                self.pattern = self._reshape2

            def callback(self, pre, post, node_map):
                def get_shape(key):
                    return run_infer_type(node_map[key][0]).checked_type.shape

                input_shape = get_shape(self._data)
                channel_expanded_shape = get_shape(self._reshape1)
                channel_shuffled_shape = get_shape(self._transpose)
                output_shape = get_shape(self._reshape2)

                def check_shape(shape_dim5, shape_dim4):
                    if not (len(shape_dim5) == 5 and len(shape_dim4) == 4):
                        return False

                    def reduce_channel(in_shape, channel):
                        """
                        Parameters
                        ----------
                        in_shape: shape in 5 dimentions
                            Shape to reduce to original layout

                        channel: int
                            Index of channel in current layout, eg:
                               "HCHW".index("C") == 1
                               "HWHC".index("C") == 3

                        Returns
                        -------
                        shape: shape in 4 dimentions
                            Reduced shape in original layout

                        """
                        # NCHW: [N, Group, C/Group, H, W] -> [N, Group * (C/Group), H, W] -> NCHW
                        # NHWC: [N, H, W, Group, C/Group] -> [N, H, W, Group * (C/Group)] -> NHWC
                        return (
                            in_shape[:channel]
                            + [in_shape[channel] * in_shape[channel + 1]]
                            + in_shape[channel + 2 :]
                        )

                    a = shape_dim4
                    b = reduce_channel(shape_dim5, data_layout.index("C"))
                    return (
                        a[0] == b[0] and a[1] == b[1] and a[2] == b[2] and a[3] == b[3]
                    )

                if not check_shape(channel_expanded_shape, input_shape):
                    log_debug3(
                        "Channel expanded shape {} mismatch with input shape {}.".format(
                            channel_expanded_shape, input_shape
                        )
                    )
                    return post

                if not check_shape(channel_shuffled_shape, output_shape):
                    log_debug3(
                        "Channel shuffled shape {} mismatch with output shape {}.".format(
                            channel_shuffled_shape, output_shape
                        )
                    )
                    return post

                groups = channel_expanded_shape[data_layout.index("C")]
                new_attrs = {
                    "data_layout": data_layout,
                    "groups": groups,
                }

                data = node_map[self._data][0]
                call_attrs = tvm.ir.make_node("DictAttrs", **new_attrs)
                call_channel_shuffle = tvm.relay.Call(
                    self._channel_shuffle_op, [data], call_attrs
                )
                return call_channel_shuffle

        new_expr = rewrite(MatchAndRewrite(), mod["main"])
        mod.update_func(mod.get_global_var("main"), new_expr)

        return mod
