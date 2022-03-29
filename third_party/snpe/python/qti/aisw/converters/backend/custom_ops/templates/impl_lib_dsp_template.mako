<%doc>
# ==============================================================================
#
#  Copyright (c) 2020 -2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================</%doc>
<%operators = package.package_info.operators%>
//==============================================================================
// Auto Generated Code for ${package_name}
//==============================================================================

#include <string.h>
#include "SnpeUdo/UdoImplDsp.h"

%for operator in operators:
%if core_type in operator.core_types:
#include "${operator.type_name}ImplLib${runtime.title()}.h"
%endif
%endfor

#define NUM_OPS ${len(operators)}

// implementation library info
// Library version is set to default value of 1.0.0
SnpeUdo_HexNNv2GlobalInfra_t* infra = NULL;
SnpeUdo_LibVersion_t ver = {{1, 0, 0}, {API_VERSION_MAJOR, API_VERSION_MINOR, API_VERSION_TEENY}};
SnpeUdo_HexNNInfraType_t i_type= UDO_INFRA_HEXNN_V2;

// definition of UDO package name and operation names
char packageName [] = "${package_name}";
char opTypes [] = "${' '.join(operator.type_name for operator in operators)}";
SnpeUdo_ImpInfo_t implementationLibInfo = {SNPE_UDO_CORETYPE_DSP, packageName, opTypes, NUM_OPS};

int lib_initialized = 0;
SnpeUdo_OpTypesList *m_head = NULL;

/**
 * implementation of SnpeUdo_getVersion
 * @param version i.e. {{1, 0, 0}, {API_VERSION_MAJOR, API_VERSION_MINOR, API_VERSION_TEENY}}
 * @return error type - no error
 */
SnpeUdo_ErrorType_t SnpeUdo_getVersion (SnpeUdo_LibVersion_t** version) {
    *version = &ver;
    return SNPE_UDO_NO_ERROR;
}

/**
 * implementation of SnpeUdo_getImpInfo
 * @param implementationInfo i.e. {SNPE_UDO_CORETYPE_DSP, "${package_name}", "${' '.join(operator.type_name for operator in operators)}", 1}
 * @return error type - no error
 */
SnpeUdo_ErrorType_t
SnpeUdo_getImpInfo (SnpeUdo_ImpInfo_t** implementationInfo)
{
    *implementationInfo = &implementationLibInfo;
    return SNPE_UDO_NO_ERROR;
}


static void registerOpToOpList(const char *opType, UdoDspShared *obj)
{
    SnpeUdo_OpTypesList* newOpNode = (*(infra->udoMalloc)) (sizeof(SnpeUdo_OpTypesList));
    int size = strlen(opType) + 1;
    newOpNode->opType = (*(infra->udoMalloc))(size); // +1 to hold the '\0' character
    strlcpy(newOpNode->opType , opType, size);
    newOpNode->opFunctionPtr = obj;
    newOpNode->next = m_head;
    m_head  = newOpNode;
}

static UdoDspShared *getImplementOpPtr(char *opType)
{
    SnpeUdo_OpTypesList *current = m_head;
    while (current != NULL)
    {
        if (!strcmp(current->opType, opType))
            return current->opFunctionPtr;
        else
            current = current->next;
    }
    return NULL;
}

static void deleteOpList()
{
    SnpeUdo_OpTypesList* currentOp = m_head;
    SnpeUdo_OpTypesList* nextOp;

    while (currentOp != NULL)
    {
        nextOp = currentOp->next;
        (*(infra->udoFree))(currentOp->opType);
        (*(infra->udoFree))(currentOp->opFunctionPtr);
        (*(infra->udoFree))(currentOp);
        currentOp = nextOp;
    }

    m_head = NULL;
}

static SnpeUdo_ErrorType_t
checkVersion (void* globalInfrastructure)
{
    if (lib_initialized==0) {
        SnpeUdo_DspGlobalInfrastructure_t* h_infra = (SnpeUdo_DspGlobalInfrastructure_t*) globalInfrastructure;
        SnpeUdo_Version_t h_ver = h_infra->dspInfraVersion;
        SnpeUdo_Version_t api_ver = ver.apiVersion;
        if(h_ver.major==api_ver.major && h_infra->infraType==i_type)
        {
            infra = &(h_infra->hexNNv2Infra);
            // mutex lock/unlock global variables here for thread safety
            lib_initialized = 1;
            return SNPE_UDO_NO_ERROR;
        }
        return SNPE_UDO_INVALID_ARGUMENT;
    }
    return SNPE_UDO_NO_ERROR;
}

/**
 * implementation of SnpeUdo_initImplLibrary
 * @param globalInfrastructure
 * @return error type
 */
SnpeUdo_ErrorType_t
SnpeUdo_initImplLibrary (void* globalInfrastructure)
{
    SnpeUdo_ErrorType_t res = SNPE_UDO_NO_ERROR;
    res = checkVersion(globalInfrastructure);

    if (res != SNPE_UDO_NO_ERROR)
    {
        return res;
    }

%for operator in operators:
%if core_type in operator.core_types:

    UdoDspShared *${operator.type_name}Op  = new_${operator.type_name}(infra);
    if (${operator.type_name}Op == NULL)
    {
        return SNPE_UDO_INVALID_ARGUMENT;
    }
    registerOpToOpList("${operator.type_name}", ${operator.type_name}Op);

%endif
%endfor

    return SNPE_UDO_NO_ERROR;
}

/**
 * implementation of SnpeUdo_queryOperation
 * @param operationType
 * @param numOfInputs
 * @param inputsQuantTypes
 * @param numOfOutputs
 * @param outputsQuantTypes
 * @return
 */
SnpeUdo_ErrorType_t
SnpeUdo_queryOperation (SnpeUdo_String_t operationType, uint32_t numOfStaticParams,
                        const SnpeUdo_Param_t* staticParams, uint32_t* numOfInputs,
                        SnpeUdo_QuantizationType_t** inputsQuantTypes,
                        SnpeUdo_HexNNTensorLayout_t** inputsLayouts, uint32_t* numOfOutputs,
                        SnpeUdo_QuantizationType_t** outputsQuantTypes,
                        SnpeUdo_HexNNTensorLayout_t** outputsLayouts)
{
    char* get_op = strstr(opTypes, operationType);
    if (get_op != NULL)
    {
        UdoDspShared *op_ptr = getImplementOpPtr(operationType);
        if (op_ptr == NULL)
        {
            return SNPE_UDO_WRONG_OPERATION;
        }
        else
        {
            return op_ptr->QueryOp(operationType, numOfStaticParams,
                                   staticParams, numOfInputs, inputsQuantTypes,
                                   inputsLayouts, numOfOutputs, outputsQuantTypes,
                                   outputsLayouts);
        }
    }
    else
    {
        return SNPE_UDO_WRONG_OPERATION;
    }
}

SnpeUdo_ErrorType_t
SnpeUdo_validateOperation (SnpeUdo_String_t operationType, uint32_t numOfStaticParams,
                           const SnpeUdo_Param_t* staticParams)
{
    char* get_op = strstr(opTypes, operationType);
    if (get_op != NULL)
    {
        UdoDspShared *op_ptr = getImplementOpPtr(operationType);
        if (op_ptr == NULL)
        {
            return SNPE_UDO_WRONG_OPERATION;
        }
        else
        {
            return op_ptr->ValidateOp(operationType, numOfStaticParams, staticParams);
        }
    }
    else
    {
        return SNPE_UDO_WRONG_OPERATION;
    }
}

/**
 * implementation of SnpeUdo_createOpFactory
 * @param udoCoreType
 * @param perFactoryInfrastructure
 * @param operationType
 * @param numOfStaticParams
 * @param staticParams
 * @param opFactory
 * @return
 */
SnpeUdo_ErrorType_t
SnpeUdo_createOpFactory (SnpeUdo_CoreType_t udoCoreType, void* perFactoryInfrastructure,
                         SnpeUdo_String_t operationType, uint32_t numOfStaticParams,
                         SnpeUdo_Param_t* staticParams, SnpeUdo_OpFactory_t* opFactory)
{

    if(infra == NULL)
    {
        return SNPE_UDO_UNSUPPORTED_FEATURE;
    }
    if(operationType == NULL || operationType == 0)
    {
        return SNPE_UDO_INVALID_ARGUMENT;
    }

    char* get_op = strstr(opTypes, operationType);
    if (get_op != NULL)
    {
        UdoDspShared *op_ptr = getImplementOpPtr(operationType);
        if (op_ptr == NULL)
        {
            return SNPE_UDO_WRONG_OPERATION;
        }
        else
        {
            return op_ptr->CreateOp(infra, udoCoreType, perFactoryInfrastructure,
                                    operationType, numOfStaticParams, staticParams, opFactory);
        }
    }
    else
    {
        return SNPE_UDO_WRONG_OPERATION;
    }
}

/**
 * implementation of SnpeUdo_CreateOperation
 * @param OpFactory
 * @param perOpInfrastructure
 * @param numOfInputs
 * @param inputs
 * @param numOfOutputs
 * @param outputs
 * @param operation
 * @return
 */
SnpeUdo_ErrorType_t
SnpeUdo_createOperation (SnpeUdo_OpFactory_t OpFactory, void* perOpInfrastructure, uint32_t numOfInputs,
                                             SnpeUdo_TensorParam_t* inputs, uint32_t numOfOutputs,
                                             SnpeUdo_TensorParam_t* outputs, SnpeUdo_Operation_t* operation)
{

    if(OpFactory == NULL)
    {
        return SNPE_UDO_INVALID_ARGUMENT;
    }
    if((numOfInputs == 0 || inputs == NULL) && (numOfOutputs == 0 || outputs == NULL))
    {
        return SNPE_UDO_INVALID_ARGUMENT;
    }
    if(infra == NULL)
    {
        return SNPE_UDO_UNSUPPORTED_FEATURE;
    }

    OpParams* m_OpParams = (*(infra->udoMalloc)) (sizeof(OpParams));
    m_OpParams->opInfra = (SnpeUdo_HexNNv2OpInfra_t) perOpInfrastructure;
    m_OpParams->opFactory = OpFactory;
    m_OpParams->numInputParams = numOfInputs;
    // no inputs
    if(numOfInputs == 0 || inputs == NULL)
    {
        m_OpParams->numInputParams = 0;
        m_OpParams->InputParams = NULL;
    }
    else
    {
        m_OpParams->InputParams = inputs;
    }
    // no outputs
    m_OpParams->numOutputParams = numOfOutputs;
    if(numOfOutputs == 0 || outputs == NULL)
    {
        m_OpParams->numOutputParams = 0;
        m_OpParams->outputParams = NULL;
    }
    else
    {
        m_OpParams->outputParams = outputs;
    }
    *operation = (SnpeUdo_Operation_t) m_OpParams;
    return SNPE_UDO_NO_ERROR;
}

/**
 * implementation of SnpeUdo_executeOp
 * @param operation
 * @param blocking
 * @param ID
 * @param notifyFunc
 * @return
 */
SnpeUdo_ErrorType_t
SnpeUdo_executeOp (SnpeUdo_Operation_t operation, bool blocking, const uint32_t ID,
                                       SnpeUdo_ExternalNotify_t notifyFunc)
{
    if(operation == NULL)
    {
        return SNPE_UDO_INVALID_ARGUMENT;
    }
    OpParams* m_Operation = (OpParams*) operation;
    char* m_OpType = ((OpFactory*) (m_Operation->opFactory))->opType;
    if(m_OpType == NULL)
    {
        return SNPE_UDO_INVALID_ARGUMENT;
    }
    if(infra == NULL)
    {
        return SNPE_UDO_INVALID_ARGUMENT;
    }

    char* get_op = strstr(opTypes, m_OpType);
    if (get_op != NULL)
    {
        UdoDspShared *op_ptr = getImplementOpPtr(m_OpType);
        if (op_ptr == NULL)
        {
            return SNPE_UDO_WRONG_OPERATION;
        }
        else
        {
            return op_ptr->ExecuteOp(infra, operation, blocking, ID, notifyFunc);
        }
    }
    else
    {
        return SNPE_UDO_WRONG_OPERATION;
    }
}

/**
 * implementation of SnpeUdo_releaseOp
 * @param operation
 * @return error type
 */
SnpeUdo_ErrorType_t SnpeUdo_releaseOp (SnpeUdo_Operation_t operation)
{
    if(operation == NULL)
    {
        return SNPE_UDO_NO_ERROR;
    }
    if(infra == NULL)
    {
        return SNPE_UDO_UNSUPPORTED_FEATURE;
    }
    OpParams* m_Operation = (OpParams*) operation;
    (*(infra->udoFree)) (m_Operation);
    return SNPE_UDO_NO_ERROR;
}

/**
 * implementation of SnpeUdo_releaseOpFactory
 * @param opFactory
 * @return
 */
SnpeUdo_ErrorType_t SnpeUdo_releaseOpFactory (SnpeUdo_OpFactory_t opFactory)
{
    if (infra == NULL)
    {
        return SNPE_UDO_UNSUPPORTED_FEATURE;
    }
    if (opFactory == NULL)
    {
        return SNPE_UDO_NO_ERROR;
    }

    OpFactory* factory = (OpFactory*) opFactory;
    (*(infra->udoFree))((factory->opType));
    (*(infra->udoFree))(factory);
    return SNPE_UDO_NO_ERROR;
}

/**
 * implementation of SnpeUdo_terminateImplLibrary
 * @return
 */
SnpeUdo_ErrorType_t SnpeUdo_terminateImplLibrary()
{
    if(infra == NULL)
    {
        return SNPE_UDO_UNSUPPORTED_FEATURE;
    }

    deleteOpList();
    lib_initialized = 0;
    return SNPE_UDO_NO_ERROR;
}


