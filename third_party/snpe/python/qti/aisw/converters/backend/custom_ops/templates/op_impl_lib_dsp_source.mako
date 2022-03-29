<%doc>
# ==============================================================================
#
#  Copyright (c) 2020-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================</%doc>
<% op_name = operator.type_name %>
//==============================================================================
// Auto Generated Code for ${package_name}
//==============================================================================

#include <string.h>
#include "${str(op_name)}ImplLib${runtime.title()}.h"
#include "SnpeUdo/UdoImplDsp.h"

// operations info
char ${op_name}OpType [] = "${op_name}";
// static inputs are also considered as static parameters.
uint32_t ${op_name}StaticParamsNum = ${len(operator.tensor_param) + len(operator.scalar_param) + len([p for p in operator.input if p.static])};
// only count non-static inputs.
uint32_t ${op_name}InputsNum = ${len([i for i in operator.input if not i.static])};
uint32_t ${op_name}OutputsNum = ${len(operator.output)};
SnpeUdo_QuantizationType_t ${op_name}InputQuantizationTypes [] = {${','.join(input.quant_type for input in operator.input if not input.static)}};
SnpeUdo_QuantizationType_t ${op_name}OutputQuantizationTypes [] =  {${','.join(output.quant_type for output in operator.output)}};
SnpeUdo_HexNNTensorLayout_t* ${op_name}Layout = NULL;

UdoDspShared* new_${op_name}(SnpeUdo_HexNNv2GlobalInfra_t* infra)
{
    UdoDspShared *pOpObj = (*(infra->udoMalloc))(sizeof(UdoDspShared));
    if (pOpObj == NULL)
    {
        return NULL;
    }

    pOpObj->QueryOp = ${op_name}_QueryOperation;
    pOpObj->ValidateOp = ${op_name}_ValidateOperation;
    pOpObj->CreateOp = ${op_name}_CreateOpFactory;
    pOpObj->ExecuteOp = ${op_name}_ExecuteOp;

    return pOpObj;
}

SnpeUdo_ErrorType_t
${op_name}_QueryOperation (SnpeUdo_String_t operationType, uint32_t numOfStaticParams,
                        const SnpeUdo_Param_t* staticParams, uint32_t* numOfInputs,
                        SnpeUdo_QuantizationType_t** inputsQuantTypes,
                        SnpeUdo_HexNNTensorLayout_t** inputsLayouts, uint32_t* numOfOutputs,
                        SnpeUdo_QuantizationType_t** outputsQuantTypes,
                        SnpeUdo_HexNNTensorLayout_t** outputsLayouts)
{
    if(strcmp(operationType, ${op_name}OpType) == 0)
    {
        *numOfInputs = ${op_name}InputsNum;
        *inputsQuantTypes = ${op_name}InputQuantizationTypes;
        *inputsLayouts = ${op_name}Layout;
        *numOfOutputs = ${op_name}OutputsNum;
        *outputsQuantTypes = ${op_name}OutputQuantizationTypes;
        *outputsLayouts = ${op_name}Layout;
    } else
    {
        return SNPE_UDO_WRONG_OPERATION;
    }
    return SNPE_UDO_NO_ERROR;
}

SnpeUdo_ErrorType_t
${op_name}_ValidateOperation (SnpeUdo_String_t operationType, uint32_t numOfStaticParams,
                           const SnpeUdo_Param_t* staticParams)
{
    if(strcmp(operationType, ${op_name}OpType) == 0)
    {
        if (numOfStaticParams != ${op_name}StaticParamsNum)
        {
            return SNPE_UDO_UNSUPPORTED_FEATURE;
        }
    } else
    {
        return SNPE_UDO_WRONG_OPERATION;
    }
    return SNPE_UDO_NO_ERROR;
}

SnpeUdo_ErrorType_t
${op_name}_CreateOpFactory (SnpeUdo_HexNNv2GlobalInfra_t* infra, SnpeUdo_CoreType_t udoCoreType,
                         void* perFactoryInfrastructure,
                         SnpeUdo_String_t operationType, uint32_t numOfStaticParams,
                         SnpeUdo_Param_t* staticParams, SnpeUdo_OpFactory_t* opFactory)
{
    if(operationType == NULL || operationType == 0)
    {
        return SNPE_UDO_INVALID_ARGUMENT;
    }

    if(strcmp(operationType, ${op_name}OpType) == 0)
    {
        ${op_name}OpFactory* this_factory = (*(infra->udoMalloc))(sizeof(${op_name}OpFactory));
        int size = strlen(operationType) + 1; // +1 to hold the '\0' character
        this_factory->opType = (*(infra->udoMalloc))(size);
        strlcpy((this_factory->opType), operationType, size);
        *opFactory = (SnpeUdo_OpFactory_t) this_factory;
    }
    else
    {
        return SNPE_UDO_WRONG_OPERATION;
    }
    return SNPE_UDO_NO_ERROR;
}

SnpeUdo_ErrorType_t
${op_name}_ExecuteOp (SnpeUdo_HexNNv2GlobalInfra_t* infra, SnpeUdo_Operation_t operation,
                      bool blocking, const uint32_t ID, SnpeUdo_ExternalNotify_t notifyFunc)
{
    if(operation == NULL)
    {
        return SNPE_UDO_INVALID_ARGUMENT;
    }

    OpParams* m_Operation = (OpParams*) operation;
    char* m_OpType = ((${op_name}OpFactory*) (m_Operation->opFactory))->opType;
    if(m_OpType == NULL)
    {
        return SNPE_UDO_INVALID_ARGUMENT;
    }
    if(strcmp(m_OpType, ${op_name}OpType) == 0)
    {
       /**
        * add code here
        */
        return SNPE_UDO_NO_ERROR;
    }
    else
    {
        return SNPE_UDO_WRONG_OPERATION;
    }
}
