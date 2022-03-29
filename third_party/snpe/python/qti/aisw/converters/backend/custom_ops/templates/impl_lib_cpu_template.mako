<%doc> define all relevant variables</%doc>
<%doc>
# ==============================================================================
#
#  Copyright (c) 2020 - 2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================</%doc>
<%operators = package.package_info.operators%>
<%namespace file="/helpers.mako" import="*" />
//==============================================================================
// Auto Generated Code for ${package_name}
//==============================================================================
#include "utils/UdoUtil.hpp"
% if core_type in package.core_types:
#include "SnpeUdo/UdoImpl.h"
%endif
%for operator in operators:
% if core_type in operator.core_types:
#include "${operator.type_name}ImplLib${runtime}.hpp"
%endif
%endfor


extern "C"
{
using namespace UdoUtil;
SnpeUdo_ErrorType_t SnpeUdo_initImplLibrary(void* globalInfrastructure)
{
    %if str(runtime).lower() == 'gpu':
    UDO_VALIDATE_MSG(globalInfrastructure == nullptr,
                    SNPE_UDO_INVALID_ARGUMENT,
                    "The provided global infrastructure is null")
    SnpeUdo_GpuInfrastructure_t* gpuGlobalInfrastructure = static_cast<SnpeUdo_GpuInfrastructure_t*>(globalInfrastructure);
    %endif

    UDO_VALIDATE_RETURN_STATUS(setImplementation( ${core_type}, "${package_name}"));

    UdoImplementationLib& ImplLib = getImplementation();

    ImplLib.setVersion(1,0,0);

    %for operator in operators:
    %if core_type in operator.core_types:
    %if str(runtime).lower() == 'gpu':
    UDO_VALIDATE_RETURN_STATUS(ImplLib.registerOpDefinition
                               ("${operator.type_name}",
                               std::unique_ptr<${_runtime_formatted_op_name(operator.type_name)}OpDef>(new ${_runtime_formatted_op_name(operator.type_name)}OpDef("${operator.type_name}",${len(operator.input)}, ${len(operator.output)}, gpuGlobalInfrastructure))))
    %else:
    UDO_VALIDATE_RETURN_STATUS(ImplLib.registerOpDefinition
                               ("${operator.type_name}",
                               std::unique_ptr<${_runtime_formatted_op_name(operator.type_name)}OpDef>(new ${_runtime_formatted_op_name(operator.type_name)}OpDef("${operator.type_name}",${len(operator.input)}, ${len(operator.output)}))))
    %endif
    %endif
    %endfor
    return SNPE_UDO_NO_ERROR;
}

SnpeUdo_ErrorType_t SnpeUdo_terminateImplLibrary(void)
{
    SnpeUdo_ErrorType_t status = SNPE_UDO_NO_ERROR;
    deleteImplementationInstance();
    return status;
}

SnpeUdo_ErrorType_t
SnpeUdo_getImpInfo(SnpeUdo_ImpInfo_t** info)
{
    return getImplementation().getImplementationInfo(info);
}

SnpeUdo_ErrorType_t
SnpeUdo_getVersion(SnpeUdo_LibVersion_t** version)
{
    return getImplementation().getVersion(version);
}

SnpeUdo_ErrorType_t
SnpeUdo_createOpFactory(SnpeUdo_CoreType_t udoCoreType,
                        void* perFactoryInfrastructure,
                        SnpeUdo_String_t operationType,
                        uint32_t numOfStaticParams,
                        SnpeUdo_Param_t *staticParams,
                        SnpeUdo_OpFactory_t* opFactory)
{


    return getImplementation().createOpFactory(operationType,
                                               perFactoryInfrastructure,
                                               numOfStaticParams,
                                               staticParams,
                                               opFactory);
}

SnpeUdo_ErrorType_t
SnpeUdo_createOperation(SnpeUdo_OpFactory_t opFactory,
                        void*, // Unused in CPU
                        uint32_t numOfInputs,
                        SnpeUdo_TensorParam_t* inputs,
                        uint32_t numOfOutputs,
                        SnpeUdo_TensorParam_t *outputs,
                        SnpeUdo_Operation_t* operation)
{
    auto result = (opFactory)->definition->createOp(opFactory->infrastructure,
                                                    numOfInputs,
                                                    inputs,
                                                    numOfOutputs,
                                                    outputs,
                                                    opFactory->numOfStaticParams,
                                                    opFactory->staticParams);

    UDO_VALIDATE_MSG(result == nullptr,
                    SNPE_UDO_WRONG_OPERATION,
                    "Could not create operation of type: "<<opFactory->definition->getOperationType())

    std::unique_ptr<_SnpeUdo_Operation_t> handle(new _SnpeUdo_Operation_t());
    handle->operation = std::move(result);
    *operation = handle.release();

    return SNPE_UDO_NO_ERROR;
}

SnpeUdo_ErrorType_t
SnpeUdo_executeOp(SnpeUdo_Operation_t operation,
                  bool blocking,
                  const uint32_t ID,
                  SnpeUdo_ExternalNotify_t notifyFunc)
{
    return operation->operation->snpeUdoExecute(blocking, ID, notifyFunc);
}

SnpeUdo_ErrorType_t
SnpeUdo_setOpIO(SnpeUdo_Operation_t operation,
                SnpeUdo_TensorParam_t *inputs,
                SnpeUdo_TensorParam_t *outputs)
{
    return operation->operation->snpeUdoSetIo(inputs, outputs);
}

SnpeUdo_ErrorType_t
SnpeUdo_profileOp(SnpeUdo_Operation_t operation, uint32_t *executionTime)
{
    return  operation->operation->snpeUdoProfile(executionTime);
}


SnpeUdo_ErrorType_t
SnpeUdo_releaseOp(SnpeUdo_Operation_t operation)
{
    auto status = SNPE_UDO_NO_ERROR;
    operation->operation.reset();

    return status;
}

SnpeUdo_ErrorType_t
SnpeUdo_releaseOpFactory(SnpeUdo_OpFactory_t opFactory)
{
    auto status = SNPE_UDO_NO_ERROR;
    delete opFactory; // manually allocated object in createOpFactory
    return status;
}

}; //extern C
