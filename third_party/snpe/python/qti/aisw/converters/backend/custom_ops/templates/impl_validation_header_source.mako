<%doc>
# ==============================================================================
#
#  Copyright (c) 2020-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================</%doc>
<% operator_names = [operator.type_name for operator in operators] %>
//==============================================================================
// Auto Generated Code for ${package_name}
//==============================================================================

#include "SnpeUdo/UdoBase.h"
#include "${package_name}${runtime}ImplValidationFunctions.hpp"

using namespace UdoUtil;

%for op_name in operator_names:
SnpeUdo_ErrorType_t
${op_name}${runtime}ValidationFunction::validateOperation(SnpeUdo_OpDefinition_t* def) {
    /**
     * add code here
     */
    return SNPE_UDO_NO_ERROR;
}

%endfor