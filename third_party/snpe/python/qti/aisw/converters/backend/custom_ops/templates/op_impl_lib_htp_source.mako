<%doc>
# ==============================================================================
#
#  Copyright (c) 2020-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================</%doc>

<%page expression_filter="n" expression_filter="trim" />
<%!
from qti.aisw.converters.backend.custom_ops.helpers.template_helpers import _template_builder, get_hexnn_tensor_sig, get_hexnn_param_sig %>
//==============================================================================
// Auto Generated Code for ${package_name}
//==============================================================================
#include "optimize.h"
#include "op_register_ext.h"

static constexpr auto operatorName = "${op_name}";

// op execute function declarations
${_template_builder(operator) | n}
<%self:_get_op_impl_input_output funcname='${operator.type_name.lower()}Impl'>
</%self:_get_op_impl_input_output>;

//op definitions
DEF_PACKAGE_OP((${operator.type_name.lower()}Impl<Tensor>), operatorName)

/* execute functions for ops */

${_template_builder(operator) | n}
<%self:_get_op_impl_input_output funcname='${operator.type_name.lower()}Impl'>
{
/*
* add code here
           * */
return GraphStatus::Success;
}
</%self:_get_op_impl_input_output>

<%def name="_get_op_impl_input_output(funcname)" buffered="True">
<%cur_idx = 0
func_tab_size = len("int(") +  len(funcname) + 1 %>
int ${funcname}(${get_hexnn_tensor_sig(operator, func_tab_size, cur_idx)}
%if operator.tensor_param or operator.scalar_param:
${get_hexnn_param_sig(operator, func_tab_size)|n}
%endif
${caller.body()}
</%def>
