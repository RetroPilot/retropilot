# ==============================================================================
#
#  Copyright (c) 2019-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from .caffe_base_translation import *
from qti.aisw.converters.backend.custom_ops.op_factory import OpFactory
from qti.aisw.converters.backend.custom_ops.core import \
    (
    get_internal_dtype,
    CustomTensorParam,
    get_np_type_from_backend_type,
    is_quant_type
)
import numpy as np
from qti.aisw.converters.common.utils import code_to_message, converter_utils
from qti.aisw.converters.common.converter_ir import op_adapter


# ------------------------------------------------------------------------------
#   Custom Op Translation
# ------------------------------------------------------------------------------
class CaffeCustomOpTranslation(CaffeTranslationBase):
    def __init__(self):
        CaffeTranslationBase.__init__(self)
        self.custom_op = None

    def extract_input_names(self, src_op, graph):
        return [str(input.name) for input in self.custom_op.inputs]

    def extract_output_names(self, src_op, graph):
        return [str(output.name) for output in self.custom_op.outputs]

    def extract_parameters(self, src_op, graph):
        custom_op = OpFactory.op_collection.get_first_of(src_op.type)
        package_name = OpFactory.get_package_name(custom_op.op_type)
        self.custom_op = custom_op
        static_idx = 0

        for name, custom_param in custom_op.params.items():
            param = custom_param.param
            if isinstance(param, CustomTensorParam) and ((isinstance(param.data, np.ndarray) and
                                                          param.data.size == 0) or param.data is None):
                # if the parameter does not have data, is not static,
                # and does not have a known default value
                # then we error out here. If it has a default value, then it has been set in
                # extract_attrs and is actually an empty value
                if hasattr(param, "static") and not param.static:
                    if param.default_value is None:
                        raise ValueError(code_to_message.
                                         get_error_message("ERROR_CUSTOM_OP_PARAM_NO_DATA")(name,
                                                                                            custom_op.op_type))
                    continue
                else:
                    try:
                        param.data = np.ascontiguousarray(graph.weights.weights_map[
                                                              src_op.name][static_idx].data)
                        # increment the index of the static parameters seen. Note this relies on
                        # the inputs/params being listed in order.
                        static_idx += 1

                    except IndexError:
                        # Occasionally, not all filler parameters have a corresponding value in
                        # the weight map. If a static filler parameter is not found in the blobs
                        # object, then the user must provide a default value. If no default value
                        # is provided then an error is raised.
                        if param.default_value is not None:
                            # if the data type is unsigned, then numpy treats the value as a string
                            # which can cause issues if it is saved as an array.
                            # We treat this as a pass through case and allow numpy to infer d-type
                            if not is_quant_type(param.data_type):
                                param.data = np.asarray(param.default_value). \
                                    astype(dtype=param.data_type)
                            else:
                                param.data = np.asarray(param.default_value)
                        else:
                            raise IndexError(code_to_message.get_error_message(
                                "ERROR_CANNOT_INGEST_CAFFE_STATIC_INPUT")
                                             (str(name)))

                    # set all other dependent fields
                    param.dimensions = list(custom_param.param.data.shape)
                    param.rank = len(custom_param.param.data.shape)

        inputs, outputs, scalar_params, tensor_params = custom_op.as_dict()
        return op_adapter.CustomOp(name=src_op.name,
                                   package_name=package_name,
                                   custom_type=custom_op.op_type,
                                   axis_orders=custom_op.axis_orders,
                                   inputs=inputs,
                                   outputs=outputs,
                                   output_dims=custom_op.output_dims,
                                   tensor_params=tensor_params,
                                   scalar_params=scalar_params)


CaffeTranslations.register_translation(CaffeCustomOpTranslation(),
                                       converter_utils.converter_type('custom', 'caffe'),
                                       op_adapter.CustomOp.TRANSLATION_KEY)
