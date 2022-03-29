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
class IdentifyUpsampling:
    def __init__(self, data_layout="NCHW"):
        assert data_layout == "NCHW", "Unsupported data layout {}".format(data_layout)
        self.data_layout = data_layout

    def transform_module(self, mod, ctx):
        data_layout = self.data_layout

        class MatchAndRewrite(DFPatternCallback):
            def __init__(self):
                super(MatchAndRewrite, self).__init__(require_type=True)

                # Match following pattern to nn.upsampling op:
                # %652 = reshape(%651, newshape=[1, 256, 13, 1, 13, 1]) /* ty=Tensor[(1, 256, 13, 1, 13, 1), float32] */;
                # %653 = repeat(%652, repeats=1, axis=0) /* ty=Tensor[(1, 256, 13, 1, 13, 1), float32] */;
                # %654 = repeat(%653, repeats=2, axis=3) /* ty=Tensor[(1, 256, 13, 2, 13, 1), float32] */;
                # %655 = repeat(%654, repeats=2, axis=5) /* ty=Tensor[(1, 256, 13, 2, 13, 2), float32] */;
                # %657 = reshape(%655, newshape=[1, 256, 26, 26]) /* ty=Tensor[(1, 256, 26, 26), float32] */;
                self._data = wildcard()
                self._reshape1 = is_op("reshape")(self._data)

                self._is_repeat = is_op("repeat")(wildcard()).has_attr({"repeats": 1}) |\
                                  is_op("repeat")(wildcard()).has_attr({"axis": 3}) |\
                                  is_op("repeat")(wildcard()).has_attr({"axis": 5})

                self._reshape2 = is_op("reshape")(wildcard())

                self.pattern = dominates(self._reshape1, self._is_repeat, self._reshape2)

            def callback(self, pre, post, node_map):
                def get_shape(key):
                    return run_infer_type(node_map[key][0]).checked_type.shape

                input_shape = get_shape(self._data)
                reshape1_shape = get_shape(self._reshape1)
                reshape2_shape = get_shape(self._reshape2)
                index_h = data_layout.index("H")
                index_w = data_layout.index("W")
                # Set constraint of destination shape of reshape op to be 6-dim
                if not (len(reshape1_shape) == 6 and len(reshape2_shape) == 4):
                    return post

                # Convert tvm.tir.expr.IntImm to int for division
                scale_h = int(reshape2_shape[index_h]) // int(input_shape[index_h])
                scale_w = int(reshape2_shape[index_w]) // int(input_shape[index_w])
                data = node_map[self._data][0]

                new_attrs = {
                    "scale_h": scale_h,
                    "scale_w": scale_w,
                    "layout": data_layout,
                    "method": "nearest_neighbor", # should always be nearest neighbor in this case
                    "align_corners": False,
                }

                call_upsampling = tvm.relay.op.nn.upsampling(data, **new_attrs)
                return call_upsampling

        new_expr = rewrite(MatchAndRewrite(), mod["main"])
        mod.update_func(mod.get_global_var("main"), new_expr)

        return mod
