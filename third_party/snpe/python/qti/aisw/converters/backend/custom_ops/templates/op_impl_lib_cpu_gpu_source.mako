<%doc>
# ==============================================================================
#
#  Copyright (c) 2020 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================</%doc>
<%namespace file="/helpers.mako" import="*" />
<%
formatted_op_name = _runtime_formatted_op_name(op_name)
tab = len(op_name) + len("OpDef::createOp(")
%>
//==============================================================================
// Auto Generated Code for ${package_name}
//==============================================================================
<%page expression_filter="n" />
#include "${op_name}ImplLib${runtime}.hpp"

std::unique_ptr<UdoUtil::UdoOperation>
${formatted_op_name}OpDef::createOp(void *perOpInfrastructure,
${" "*tab + "uint32_t numOfInputs,"}
${" "*tab + "SnpeUdo_TensorParam_t *inputs,"}
${" "*tab + "uint32_t numOfOutputs,"}
${" "*tab + "SnpeUdo_TensorParam_t *outputs,"}
${" "*tab + "uint32_t numOfStaticParams,"}
${" "*tab + "SnpeUdo_Param_t* params)"}
{
    /**
      * add code here
      * This method should
      * 1.) Validate Static Params
      * 2.) Create Op
      * 3.) Convert arguments and use appropriate constructor in ${op_name}.hpp
      */
    return nullptr;
}

SnpeUdo_ErrorType_t
${formatted_op_name+ str("Op::snpeUdoExecute(bool blocking, "
                                             "const uint32_t ID, "
                                             "SnpeUdo_ExternalNotify_t notifyFunc)")}
{
    %if str(runtime).lower() == 'gpu':
    // uses base class implementation by default, user can override here.
    return UdoUtil::UdoAdrenoOperation::snpeUdoExecute(blocking, ID, notifyFunc);
    %else:
    /**
      * add code here
      */
    return SNPE_UDO_NO_ERROR;
    %endif
}
