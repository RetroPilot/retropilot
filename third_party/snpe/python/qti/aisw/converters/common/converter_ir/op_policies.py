# ==============================================================================
#
#  Copyright (c) 2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================


class ConversionNamePolicy(object):
    def __init__(self):
        self.type_count = {}

    def get_op_name(self, op):
        count = self.type_count.get(op.type, 0)
        self.type_count[op.type] = count + 1
        if op.name:
            return str(op.name)
        else:
            return "%s_%d" % (op.type, count)

    def get_input_names(self, op, input_names):
        return list(map(str, input_names))

    def get_output_names(self, op, output_names):
        return list(map(str, output_names))

    def remove_output_name(self, output_name):
        return


class ConversionShapeInferencePolicy(object):

    def infer_shape(self, op, input_shapes):
        raise NotImplementedError("infer_shape for {} not implemented ".format(str(self.__class__.__name__)))
