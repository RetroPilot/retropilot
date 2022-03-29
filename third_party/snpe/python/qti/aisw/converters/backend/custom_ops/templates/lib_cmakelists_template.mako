<%doc>
# ==============================================================================
#
#  Copyright (c) 2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

"""This template is used to generate CMakeList.txt for Reg, ImplCpu libraries.

Args from python:
   package_name (str): package name.
   lib_name_suffix (str): suffix for the target library name, e.g. Reg, ImplCpu.
   util_src_files (list of str): A list of the util source file name.
   src_files (list of str): A list of the the source file names.
   runtimes_of_lib_def (list of str): A list of the runtime name to add defintion.
      This is for Reg lib only, and empty list for Impl lib.
"""</%doc>
#================================================================================
# Auto Generated Code for ${package_name}
#================================================================================

set( LIB_NAME Udo${package_name}${lib_name_suffix} )
set( UTIL_SOURCES
% for util_src in util_src_files:
   "../utils/${util_src}"
% endfor
   )
set( SOURCES
% for src in src_files:
   "${src}"
%endfor
   )
include_directories( ${'${ROOT_INCLUDES}'} )
add_library( ${'${LIB_NAME}'} SHARED ${'${UTIL_SOURCES}'} ${'${SOURCES}'} )
%for runtime in runtimes_of_lib_def:
<%
   flag_name = 'UDO_LIB_NAME_' + runtime.upper()
   lib_name = '${CMAKE_SHARED_LIBRARY_PREFIX}' + 'Udo' + package_name + 'Impl' + runtime.title() + '${CMAKE_SHARED_LIBRARY_SUFFIX}'
%>
if( NOT DEFINED ${flag_name} )
   set( ${flag_name} ${lib_name} )
endif()
target_compile_definitions(
   ${'${LIB_NAME}'}
   PRIVATE
   ${flag_name}="${'${' + flag_name + '}'}" )
%endfor

if( COMMAND install_shared_lib_to_platform_dir )
   install_shared_lib_to_platform_dir( ${'${LIB_NAME}'} )
endif()