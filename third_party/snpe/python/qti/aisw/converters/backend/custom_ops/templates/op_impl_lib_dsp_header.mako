<%doc>
# ==============================================================================
#
#  Copyright (c) 2020 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================</%doc>
//==============================================================================
// Auto Generated Code for ${package_name}
//==============================================================================
/*
 * structs and enums used in UDO implementation libraries
 */

#include "SnpeUdo/UdoImplDsp.h"
#include "utils/UdoDspShared.h"

// structs of op factories

typedef struct ${op_name}_OpFactory_t {
    SnpeUdo_String_t opType;
} ${op_name}OpFactory;

UdoDspShared* new_${op_name}(SnpeUdo_HexNNv2GlobalInfra_t* infra);    //constructor

SnpeUdo_ErrorType_t ${op_name}_QueryOperation(SnpeUdo_String_t operationType,
                                              uint32_t numOfStaticParams,
                                              const SnpeUdo_Param_t* staticParams,
                                              uint32_t* numOfInputs,
                                              SnpeUdo_QuantizationType_t** inputsQuantTypes,
                                              SnpeUdo_HexNNTensorLayout_t** inputsLayouts,
                                              uint32_t* numOfOutputs,
                                              SnpeUdo_QuantizationType_t** outputsQuantTypes,
                                              SnpeUdo_HexNNTensorLayout_t** outputsLayouts);

SnpeUdo_ErrorType_t ${op_name}_ValidateOperation(SnpeUdo_String_t operationType,
                                                 uint32_t numOfStaticParams,
                                                 const SnpeUdo_Param_t* staticParams);

SnpeUdo_ErrorType_t ${op_name}_CreateOpFactory(SnpeUdo_HexNNv2GlobalInfra_t* infra,
                                               SnpeUdo_CoreType_t udoCoreType,
                                               void* perFactoryInfrastructure,
                                               SnpeUdo_String_t operationType,
                                               uint32_t numOfStaticParams,
                                               SnpeUdo_Param_t* staticParams,
                                               SnpeUdo_OpFactory_t* opFactory);

SnpeUdo_ErrorType_t ${op_name}_ExecuteOp(SnpeUdo_HexNNv2GlobalInfra_t* infra,
                                         SnpeUdo_Operation_t operation,
                                         bool blocking,
                                         const uint32_t ID,
                                         SnpeUdo_ExternalNotify_t notifyFunc);


// op specific execution struct for storing inputs, outputs and multithreading info
typedef struct ${op_name}OpInfo_t {
    /**
     * add code here
     */
} ${op_name}OpInfo;


