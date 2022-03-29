<%doc> define all relevant variables</%doc>
<%doc>
# ==============================================================================
#
#  Copyright (c) 2020-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================</%doc>
<%
 package_name = package.name
 runtimes = package.supported_runtimes
 coretypes = package.core_types
 op_catalog = package.op_catalog_info
 operators = package.package_info.operators
 calculation_types = package.calculation_types
%>
<%namespace file="/helpers.mako" import="*" />
//==============================================================================
// Auto Generated Code for ${package_name}
//==============================================================================
#include <iostream>
#include "utils/UdoUtil.hpp"
%for runtime in runtimes:
#include "${package_name}${runtime.title()}ImplValidationFunctions.hpp"
%endfor

%for runtime in runtimes:
#ifndef UDO_LIB_NAME_${str(runtime).upper()}
#define UDO_LIB_NAME_${str(runtime).upper()} "libUdo${package_name}Impl${str(runtime).title()}.so"
#endif
%endfor

extern "C"
{

std::unique_ptr<UdoUtil::UdoVersion> regLibraryVersion;
std::unique_ptr<UdoUtil::UdoRegLibrary> regLibraryInfo;

SnpeUdo_ErrorType_t
SnpeUdo_initRegLibrary(void)
{
    regLibraryInfo.reset(new UdoUtil::UdoRegLibrary("${package_name}",
                                                   ${_to_bitmask(coretypes)}));

    regLibraryVersion.reset(new UdoUtil::UdoVersion);

    regLibraryVersion->setUdoVersion(1, 0, 0);

<%doc> Add library names : one for each supported coretype </%doc>
    /*
    ** User should fill in implementation library path here as needed.
    ** Note: The Implementation library path set here is relative, meaning each library to be used
    ** must be discoverable by the linker.
    */
%for idx, runtime in enumerate(runtimes):
    regLibraryInfo->addImplLib(UDO_LIB_NAME_${str(runtime).upper()}, ${_to_bitmask(coretypes[idx])}); //adding implementation libraries
%endfor

    %for operator in operators:
    //==============================================================================
    // Auto Generated Code for ${operator.type_name}
    //==============================================================================
    auto ${operator.type_name}Info = regLibraryInfo->addOperation("${operator.type_name}", ${_to_bitmask(operator.core_types)}, ${len(operator.input)}, ${len(operator.output)});

    %for i, coretype in enumerate(operator.core_types):
    ${operator.type_name}Info->addCoreInfo(${coretype}, ${calculation_types[i]}); //adding core info
    %endfor

    %for scalar_param in operator.scalar_param:
    ${operator.type_name}Info->addScalarParam("${scalar_param.name}", ${scalar_param.data_type}); // adding scalar param
    %endfor

    %for tensor_param in operator.tensor_param:
    ${operator.type_name}Info->addTensorParam("${tensor_param.name}", ${tensor_param.data_type}, ${tensor_param.layout}); // adding tensor param
    %endfor

    //inputs and outputs need to be added as tensor params
    %for i, input in enumerate(operator.input):

    ${operator.type_name}Info->addInputTensorInfo("${input.name}", ${_formatted_per_core_data_types(input.per_core_data_types)}, ${input.layout}, ${int(input.repeated)}, ${int(input.static)}); //adding tensor info
    %endfor

    //adding outputs
    %for output in operator.output:
    ${operator.type_name}Info->addOutputTensorInfo("${output.name}", ${_formatted_per_core_data_types(output.per_core_data_types)}, ${output.layout}, ${int(output.repeated)}); //adding tensor info
    %endfor

    // adding validation functions
    %for coretype in operator.core_types:
    UDO_VALIDATE_RETURN_STATUS(regLibraryInfo->registerValidationFunction("${operator.type_name}",
                                                ${str(coretype)},
                                                std::unique_ptr<${operator.type_name}${str(coretype).split('_')[-1].title()}ValidationFunction>
                                                    (new ${operator.type_name}${str(coretype).split('_')[-1].title()}ValidationFunction())))

    %endfor
    %endfor
    UDO_VALIDATE_RETURN_STATUS(regLibraryInfo->createRegInfoStruct())

    return SNPE_UDO_NO_ERROR;
}

SnpeUdo_ErrorType_t
SnpeUdo_getVersion(SnpeUdo_LibVersion_t** version) {

    UDO_VALIDATE_RETURN_STATUS(regLibraryVersion->getLibraryVersion(version))

    return SNPE_UDO_NO_ERROR;
}

SnpeUdo_ErrorType_t
SnpeUdo_getRegInfo(SnpeUdo_RegInfo_t** registrationInfo) {

    UDO_VALIDATE_RETURN_STATUS(regLibraryInfo->getLibraryRegInfo(registrationInfo))

    return SNPE_UDO_NO_ERROR;
}

SnpeUdo_ErrorType_t
SnpeUdo_terminateRegLibrary(void) {
    regLibraryInfo.reset();
    regLibraryVersion.reset();

    return SNPE_UDO_NO_ERROR;
}

SnpeUdo_ErrorType_t
SnpeUdo_validateOperation(SnpeUdo_OpDefinition_t* opDefinition) {
    UDO_VALIDATE_RETURN_STATUS(regLibraryInfo->snpeUdoValidateOperation(opDefinition))

    return SNPE_UDO_NO_ERROR;
}
}; //extern C
