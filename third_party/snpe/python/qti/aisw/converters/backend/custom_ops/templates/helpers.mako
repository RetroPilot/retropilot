# ==============================================================================
#
#  Copyright (c) 2020 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
<%def name="_to_bitmask(qnn_types)" filter="trim" buffered="True">
        %if isinstance(qnn_types, list):
        <% formatted_string = ' | '.join([qnn_type for qnn_type in qnn_types]) %>
        ${formatted_string}
        %else:
        ${qnn_types}
        %endif
</%def>

<%def name="_get_op_function(op_name, runtime)" filter="trim" buffered="True">
      %if str(runtime).lower() == 'cpu':
      ${op_name}Op(SnpeUdo_TensorParam_t* inputs, uint32_t numOfInputs, SnpeUdo_TensorParam_t* outputs,
                   uint32_t numOfOutputs, SnpeUdo_CpuInfrastructure_t* infrastructure, uint32_t numOfStaticParams,
                   SnpeUdo_Param_t* params)
           : UdoCpuOperation(inputs, numOfInputs, outputs, numOfOutputs, infrastructure, numOfStaticParams,  params) {}
      %elif str(runtime).lower() == "gpu":
      ${op_name}AdrenoOp(SnpeUdo_GpuOpFactoryInfrastructure_t* infrastructure, cl_kernel kernel, cl_program program, std::vector<size_t>& globalDim, std::vector<size_t>& localDim)
            : UdoUtil::UdoAdrenoOperation(infrastructure, kernel, program, globalDim, localDim){}
      %endif
</%def>

<%def name="_runtime_formatted_op_name(op_name)" filter="trim" buffered="True">
     %if str(runtime).lower() == 'gpu':
     ${op_name + 'Adreno'}
     %else:
     ${op_name}
     %endif
</%def>

<%def name="_runtime_formatted_operation_class()" filter="trim" buffered="True">
     %if str(runtime).lower() == 'gpu':
     UdoAdrenoOperation
     %else:
     UdoCpuOperation
     %endif
</%def>

<%def name="_formatted_per_core_data_types(per_core_data_types)" filter="trim">
      <% string_per_core_data_types = str(tuple(per_core_data_types.items())) %>
      ${string_per_core_data_types.replace('(', '{').replace(')', '}').replace('\'', '')}
</%def>