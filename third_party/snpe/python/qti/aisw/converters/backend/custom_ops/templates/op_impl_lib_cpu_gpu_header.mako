<%doc>
# ==============================================================================
#
#  Copyright (c) 2020 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================</%doc>
<%namespace file="/helpers.mako" import="*" />
//==============================================================================
// Auto Generated Code for ${package_name}
//==============================================================================
<%page expression_filter="trim" expression_filter="n" />
#pragma once
%if str(runtime).lower() != 'gpu':
#include "utils/UdoCpuOperation.hpp"
%else:
#include "utils/GPU/UdoAdrenoOperation.hpp"
%endif
#include "utils/IUdoOpDefinition.hpp"

class ${_runtime_formatted_op_name(op_name)}Op : public UdoUtil::${_runtime_formatted_operation_class()}
{
public:
    ${_get_op_function(op_name, runtime)}

    SnpeUdo_ErrorType_t
    snpeUdoExecute(bool blocking, uint32_t ID, SnpeUdo_ExternalNotify_t notifyFunc) override;
};

class ${_runtime_formatted_op_name(op_name)}OpDef : public UdoUtil::IUdoOpDefinition
{
public:
    ${_runtime_formatted_op_name(op_name)}OpDef() = delete;
    %if str(runtime).lower() == 'gpu':
    ${_runtime_formatted_op_name(op_name)}OpDef(const char *operationType, uint32_t numOfInputs, uint32_t numOfOutputs, SnpeUdo_GpuInfrastructure_t* gpuGlobalInfra)
    %else:
    ${_runtime_formatted_op_name(op_name)}OpDef(const char *operationType, uint32_t numOfInputs,uint32_t numOfOutputs)
    %endif
    :m_OperationType(operationType)
    ,m_NumOfInputs(numOfInputs)
    ,m_NumOfOutputs(numOfOutputs)
    %if str(runtime).lower() == 'gpu':
    ,m_GlobalInfra(gpuGlobalInfra)
    %endif
    {}

    std::unique_ptr<UdoUtil::UdoOperation>
    createOp(void *perOpInfrastucture,
             uint32_t numOfInputs,
             SnpeUdo_TensorParam_t *inputs,
             uint32_t numOfOutputs,
             SnpeUdo_TensorParam_t *outputs,
             uint32_t numOfStaticParams,
             SnpeUdo_Param_t* params) override;

    const char *getOperationType() const override { return m_OperationType; }

private:
    const char *m_OperationType;
    uint32_t m_NumOfInputs;
    uint32_t m_NumOfOutputs;
    %if str(runtime).lower() == 'gpu':
    SnpeUdo_GpuOpFactoryInfrastructure_t* m_GpuPerOpInfra;
    std::vector<size_t> m_GlobalKernelDim;
    std::vector<size_t> m_LocalKernelDim;
    SnpeUdo_GpuInfrastructure_t* m_GlobalInfra;
    %endif
};
