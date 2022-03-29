# ==============================================================================
#
#  Copyright (c) 2021, 2022 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import tvm
import numpy as np

from tvm.relay.dataflow_pattern import *
from tvm import relay

@tvm.ir.transform.module_pass(opt_level=3)
class IdentifyTFLiteDetectionPostProcess:
    def transform_module(self, mod, ctx):
        class MatchAndRewrite(DFPatternCallback):
            def __init__(self):
                super(MatchAndRewrite, self).__init__(require_type=True)
                self.detection_postprocess = tvm.relay.op.op.get("detection_postprocess")
                # Match following patterns to Detection output post processing

                # box prob (loc_prob in tvm) preprocess from yxhw to xyhw
                #%532 = concatenate(%531, axis=1);
                #%533 = split(%532, indices_or_sections=4, axis=2);
                #%534 = %533.1;
                #%535 = %533.0;
                #%536 = %533.3;
                #%537 = %533.2;
                #%538 = (%534, %535, %536, %537);
                #%539 = concatenate(%538, axis=2);

                # Anchor preprocess from yxhw to ltrb
                #%540 = split(%v_param_365, indices_or_sections=4, axis=1);
                #%541 = %540.3;
                #%542 = %540.1;
                #%543 = multiply(%541, -0.5f);
                #%544 = %540.2;
                #%545 = %540.0;
                #%546 = multiply(%544, -0.5f);
                #%547 = multiply(%541, 0.5f);
                #%548 = multiply(%544, 0.5f);
                #%549 = add(%542, %543);
                #%550 = add(%545, %546);
                #%551 = add(%542, %547);
                #%552 = add(%545, %548);
                #%553 = (%549, %550, %551, %552);
                #%554 = concatenate(%553, axis=1);

                # NMS part
                #%555 = transpose(%430, axes=[0, 2, 1]);
                #%556 = reshape(%539, newshape=[1, 76824]);
                #%557 = expand_dims(%554, axis=0);
                #%558 = vision.multibox_transform_loc(%555, %556, %557, clip=False, threshold=-inff, variances=[1f, 1f, 1f, 1f]);
                #%559 = %558.0;
                #%560 = %558.1;
                #%561 = %558.1;
                #%562 = vision.non_max_suppression(%559, %560, %561, 100, 0.5f, meta[relay.attrs.NonMaximumSuppressionAttrs][0]);
                #%563 = vision.get_valid_counts(%562, 0f, meta[relay.attrs.GetValidCountsAttrs][0]);
                #%564 = %563.1;

                # Format processing
                #%565 = strided_slice(%564, begin=[0, 0, 0], end=[1, 100, 6], strides=[1], axes=None);
                #%566 = split(%565, indices_or_sections=6, axis=2);
                #%567 = %566.3;
                #%568 = %566.2;
                #%569 = %566.5;
                #%570 = %566.4;
                #%571 = (%567, %568, %569, %570);
                #%572 = %566.0;
                #%573 = %566.1;
                #%574 = concatenate(%571, axis=2);
                #%575 = reshape(%572, newshape=[1, -1]);
                #%576 = reshape(%573, newshape=[1, -1]);
                #%577 = %563.0;
                #%578 = (%574, %575, %576, %577);

                # originated from the tuple output condition of frontend/tflite.py
                #%579 = %578.2;
                #%580 = %578.0;
                #%581 = %578.3;
                #%582 = %578.1;
                #(%579, %580, %581, %582)

                self._anchors_yxhw = wildcard()
                self._anchors_split = is_op('split')(self._anchors_yxhw).has_attr({'indices_or_sections': 4, 'axis': 1})

                self._anchors_hight_pos = is_op('multiply')(is_tuple_get_item(self._anchors_split, 2), is_expr((relay.const(0.5))))
                self._anchors_hight_neg = is_op('multiply')(is_tuple_get_item(self._anchors_split, 2), is_expr((relay.const(-0.5))))
                self._anchors_width_pos = is_op('multiply')(is_tuple_get_item(self._anchors_split, 3), is_expr((relay.const(0.5))))
                self._anchors_width_neg = is_op('multiply')(is_tuple_get_item(self._anchors_split, 3), is_expr((relay.const(-0.5))))

                self._anchors_left = is_op('add')(is_tuple_get_item(self._anchors_split, 1), self._anchors_width_neg)
                self._anchors_right = is_op('add')(is_tuple_get_item(self._anchors_split, 1), self._anchors_width_pos)
                self._anchors_top = is_op('add')(is_tuple_get_item(self._anchors_split, 0), self._anchors_hight_neg)
                self._anchors_bottom = is_op('add')(is_tuple_get_item(self._anchors_split, 0), self._anchors_hight_pos)
                self._anchors_ltrb = is_tuple((self._anchors_left,
                                     self._anchors_top,
                                     self._anchors_right,
                                     self._anchors_bottom))
                self._anchors_concat = is_op('concatenate')(self._anchors_ltrb).has_attr({'axis': 1})


                self._box_prob_yxhw = is_op('concatenate')(wildcard()).has_attr({'axis': 1})
                self._box_prob_yxhw_split = is_op('split')(self._box_prob_yxhw).has_attr({'indices_or_sections': 4})
                self._box_prob_xywh_tuple = is_tuple((is_tuple_get_item(self._box_prob_yxhw_split, 1),
                                                      is_tuple_get_item(self._box_prob_yxhw_split, 0),
                                                      is_tuple_get_item(self._box_prob_yxhw_split, 3),
                                                      is_tuple_get_item(self._box_prob_yxhw_split, 2)))
                self._box_prob_xywh_concate = is_op('concatenate')(self._box_prob_xywh_tuple).has_attr({'axis': 2})
                self._box_prob_xywh_reshape = is_op("reshape")(self._box_prob_xywh_concate)
                self._class_prob = wildcard()
                self._class_prob_trnaspose = is_op("transpose")(self._class_prob).has_attr({"axes": [0, 2, 1]})
                self._anchors_concat_expand = is_op('expand_dims')(self._anchors_concat).has_attr({'axis': 0})

                self._vision_multibox_transform_loc = is_op("vision.multibox_transform_loc")(self._class_prob_trnaspose,
                                                                                             self._box_prob_xywh_reshape,
                                                                                             self._anchors_concat_expand)

                self._vision_non_max_suppression = is_op("vision.non_max_suppression")(is_tuple_get_item(self._vision_multibox_transform_loc, 0),
                                                                                       is_tuple_get_item(self._vision_multibox_transform_loc, 1),
                                                                                       is_tuple_get_item(self._vision_multibox_transform_loc, 1),
                                                                                       wildcard(),
                                                                                       wildcard())
                self._vision_get_valid_counts = is_op("vision.get_valid_counts")(self._vision_non_max_suppression, wildcard())
                self._get_valid_counts_tuple_get_item_0 = is_tuple_get_item(self._vision_get_valid_counts, 0)
                self._get_valid_counts_tuple_get_item_1 = is_tuple_get_item(self._vision_get_valid_counts, 1)

                self._strided_slice = is_op('strided_slice')(self._get_valid_counts_tuple_get_item_1).has_attr({'begin': [0, 0, 0]})
                self._strided_slice_split = is_op('split')(self._strided_slice).has_attr({'indices_or_sections': 6})
                self._boxes = is_tuple((is_tuple_get_item(self._strided_slice_split, 3),
                                        is_tuple_get_item(self._strided_slice_split, 2),
                                        is_tuple_get_item(self._strided_slice_split, 5),
                                        is_tuple_get_item(self._strided_slice_split, 4)))
                self._boxes_concat = is_op('concatenate')(self._boxes).has_attr({'axis': 2})
                self._class_ids = is_op('reshape')(is_tuple_get_item(self._strided_slice_split, 0))
                self._scores = is_op('reshape')(is_tuple_get_item(self._strided_slice_split, 1))

                self._output= is_tuple((self._boxes_concat, self._class_ids, self._scores, self._get_valid_counts_tuple_get_item_0))

                # Note that this reorder part is originated from the final process of frontend tflite.py
                # rather than the detection output
                self._output_reorder = is_tuple((is_tuple_get_item(self._output, None),
                                                 is_tuple_get_item(self._output, None),
                                                 is_tuple_get_item(self._output, None),
                                                 is_tuple_get_item(self._output, None)))

                self.pattern = self._output_reorder


            def callback(self, pre, post, node_map):
                new_attrs = {
                    'output_dims': [[node_map[self._class_prob][0].checked_type.shape[0].value, 0, 1, 7]],
                    'num_classes': node_map[self._class_prob][0].checked_type.shape[-1].value,
                    'share_location': True,
                    # considering some in some case 0 still prepresent an object, set -1 as background
                    'background_label_id': -1 ,
                    'nms_threshold': node_map[self._vision_non_max_suppression][0].args[4].data.numpy().tolist(),
                    'confidence_threshold': node_map[self._vision_multibox_transform_loc][0].attrs.threshold,
                    'nms_top_k': node_map[self._vision_non_max_suppression][0].args[3].data.numpy().tolist(),
                    'nms_eta': 1.0,
                    'code_type': "PRIORBOX_TYPE_CENTER_SIZE",
                    'keep_top_k': node_map[self._vision_non_max_suppression][0].args[3].data.numpy().tolist(),
                    'variance_encoded_in_target': False,
                    'scale_h': int(np.reciprocal(node_map[self._vision_multibox_transform_loc][0].attrs.variances[2].value)),
                    'scale_w': int(np.reciprocal(node_map[self._vision_multibox_transform_loc][0].attrs.variances[3].value)),
                    'scale_y': int(np.reciprocal(node_map[self._vision_multibox_transform_loc][0].attrs.variances[1].value)),
                    'scale_x': int(np.reciprocal(node_map[self._vision_multibox_transform_loc][0].attrs.variances[0].value))
                }

                class_prob = node_map[self._class_prob][0]
                box_prob = node_map[self._box_prob_yxhw][0]
                anchors = node_map[self._anchors_yxhw][0]
                call_attrs = tvm.ir.make_node("DictAttrs", **new_attrs)
                call_tflite_detection_postprocess = tvm.relay.Call(
                    self.detection_postprocess,
                    [class_prob, box_prob, anchors],
                    call_attrs,
                    span=node_map[self._class_prob_trnaspose][0].span
                )
                return call_tflite_detection_postprocess

        new_expr = rewrite(MatchAndRewrite(), mod["main"])
        mod.update_func(mod.get_global_var("main"), new_expr)

        return mod
