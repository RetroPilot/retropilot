<%doc>
# ==============================================================================
#
#  Copyright (c) 2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

"""This template is used to generate the root CMakeList.txt for udo package.

Args from python:
   package_name (str): Package name.
   subdirs (list of str): A list of directory name to add_subdirectory in CMake.
"""</%doc>
#================================================================================
# Auto Generated Code for ${package_name}
#================================================================================
cmake_minimum_required( VERSION 3.14 )
project( ${package_name} )

set( ROOT_INCLUDES "${'${CMAKE_CURRENT_SOURCE_DIR}'}/include" )
if( DEFINED ENV{SNPE_ROOT} )
   list( APPEND ROOT_INCLUDES "$ENV{SNPE_ROOT}/include/zdl" )
elseif( DEFINED ENV{ZDL_ROOT} )
   file( GLOB ZDL_X86_64 "$ENV{ZDL_ROOT}/x86_64*" )
   list( APPEND ROOT_INCLUDES "${'${ZDL_X86_64}'}/include/zdl" )
else()
   message( FATAL_ERROR "SNPE_ROOT: Please set SNPE_ROOT or ZDL_ROOT to obtain Udo headers necessary to compile the package" )
endif()

% for subdir in subdirs:
if( EXISTS ${'${CMAKE_CURRENT_SOURCE_DIR}'}/${subdir}/CMakeLists.txt )
   add_subdirectory( ${subdir} )
endif()
% endfor
