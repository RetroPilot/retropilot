<%doc>
# ==============================================================================
#
#  Copyright (c) 2020 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================</%doc>
#================================================================================
# Auto Generated Code for ${package.name}
#================================================================================

# define relevant directories
SRC_DIR := ./

%if str(runtime).lower() != 'dsp':
# define library name and corresponding directory
%if str(runtime).lower() != 'cpu':
export RUNTIME := ${str(runtime).upper()}
export LIB_DIR := ../../../libs/$(TARGET)/$(RUNTIME)
%else:
export LIB_DIR := ../../../libs/$(TARGET)
%endif

library := $(LIB_DIR)/libUdo${package.name}Impl${runtime}.so

%if str(runtime).lower() == 'gpu':
# Note: add CL include path here to compile Gpu Library or set as env variable
# export CL_INCLUDE_PATH = <my_cl_include_path>
%endif

# define target architecture if not previously defined, default is x86
ifndef TARGET_AARCH_VARS
TARGET_AARCH_VARS:= -march=x86-64
endif

# specify package paths, should be able to override via command line?
UDO_PACKAGE_ROOT =${package.root}

include ../../../common.mk

%else:
# NOTE:
# - this Makefile is going to be used only to create DSP skels, so no need for android.min
%if str(dsp_arch_type).lower() == 'v68':
HEXAGON_TOOLS_VERSION = 8.4.06
ifdef HEXAGON_SDK4_ROOT
HEXAGON_SDK_ROOT = $(HEXAGON_SDK4_ROOT)
HEXAGON_TOOLS_ROOT = $(HEXAGON_SDK4_ROOT)/tools/HEXAGON_Tools/$(HEXAGON_TOOLS_VERSION)
endif

%endif
ifndef HEXAGON_SDK_ROOT
%if str(dsp_arch_type).lower() == 'v68':
$(error "HEXAGON_SDK_ROOT needs to be defined to compile a dsp library. Please set HEXAGON_SDK_ROOT to hexagon sdk installation.(Supported Version : 4.1.0)")
%else:
$(error "HEXAGON_SDK_ROOT needs to be defined to compile a dsp library. Please set HEXAGON_SDK_ROOT to hexagon sdk installation.(Supported Version : 3.5.2)")
%endif
endif

ifndef HEXAGON_TOOLS_ROOT
%if str(dsp_arch_type).lower() == 'v68':
$(error "HEXAGON_TOOLS_ROOT needs to be defined to compile a dsp library. Please set HEXAGON_TOOLS_ROOT to HEXAGON_SDK_ROOT/tools/HEXAGON_Tools/$(HEXAGON_TOOLS_VERSION)")
%else:
$(error "HEXAGON_TOOLS_ROOT needs to be defined to compile a dsp library. Please set HEXAGON_TOOLS_ROOT to HEXAGON_SDK_ROOT/tools/HEXAGON_Tools/8.3.07")
%endif
endif

%if str(dsp_arch_type).lower() == 'v68':
ifndef QNN_SDK_ROOT
$(error "QNN_SDK_ROOT needs to be defined to compile a dsp library. Please set QNN_SDK_ROOT to qnn sdk installation.")
endif
%endif

ifndef SDK_SETUP_ENV
$(error "SDK_SETUP_ENV needs to be defined to compile a dsp library. Please set SDK_SETUP_ENV=Done")
endif

%if str(dsp_arch_type).lower() == 'v68':
# define variant as V=hexagon_Release_dynamic_toolv84_v68 - it can be hardcoded too
ifndef V
V = hexagon_Release_dynamic_toolv84_v68
endif
%else:
# define variant as V=hexagon_Release_dynamic_toolv83_${dsp_arch_type} - it can be hardcoded too
ifndef V
V = hexagon_Release_dynamic_toolv83_${dsp_arch_type}
endif
%endif

V_TARGET = $(word 1,$(subst _, ,$(V)))
ifneq ($(V_TARGET),hexagon)
$(error Unsupported target '$(V_TARGET)' in variant '$(V)')
endif

# define package include paths and check API header path
# set package include paths, note package root will take precedence
# if includes are already present in package
UDO_PACKAGE_ROOT =${package.root}
PKG_NAME = ${package.name}

# must list all variants supported by this project
SUPPORTED_VS = $(default_VS)

%if str(dsp_arch_type).lower() == 'v68':
QNN_INCLUDE = $(QNN_SDK_ROOT)/include
QNN_HTP_INCLUDE = $(QNN_INCLUDE)/HTP

include $(HEXAGON_SDK_ROOT)/build/make.d/$(V_TARGET)_vs.min
include $(HEXAGON_SDK_ROOT)/build/defines.min

CXX_FLAGS += -std=c++17 -fvisibility=default -stdlib=libc++ -fexceptions -MMD -DTHIS_PKG_NAME=$(PKG_NAME)
CXX_FLAGS += -I$(QNN_INCLUDE) -I$(QNN_HTP_INCLUDE) -I$(QNN_HTP_INCLUDE)/core
CXX_FLAGS += -I$(HEXAGON_SDK_ROOT)/rtos/qurt/compute$(V_ARCH)/include/qurt
CXX_FLAGS += $(MHVX_DOUBLE_FLAG) -mhmx -DUSE_OS_QURT
CXX_FLAGS += -DQNN_API="__attribute__((visibility(\"default\")))"  -D__QAIC_HEADER_EXPORT="__attribute__((visibility(\"default\")))"

BUILD_DLLS = libUdo${package.name}Impl${runtime}

# sources for the DSP implementation library in src directory
SRC_DIR = ./
libUdo${package.name}Impl${runtime}.CXX_SRCS := $(wildcard $(SRC_DIR)/*.cpp)

%else:
# must list all the dependencies of this project
DEPENDENCIES = ATOMIC RPCMEM TEST_MAIN TEST_UTIL

# each dependency needs a directory definition
#  the form is <DEPENDENCY NAME>_DIR
#  for example:
#    DEPENDENCIES = FOO
#    FOO_DIR = $(HEXAGON_SDK_ROOT)/examples/common/foo
#

ATOMIC_DIR = $(HEXAGON_SDK_ROOT)/libs/common/atomic
RPCMEM_DIR = $(HEXAGON_SDK_ROOT)/libs/common/rpcmem
TEST_MAIN_DIR = $(HEXAGON_SDK_ROOT)/test/common/test_main
TEST_UTIL_DIR = $(HEXAGON_SDK_ROOT)/test/common/test_util

include $(HEXAGON_SDK_ROOT)/build/make.d/$(V_TARGET)_vs.min
include $(HEXAGON_SDK_ROOT)/build/defines.min

# set include paths as compiler flags
CC_FLAGS += -I $(UDO_PACKAGE_ROOT)/include

# if SNPE_ROOT is set and points to the SDK path, it will be used. Otherwise ZDL_ROOT will be assumed to point
# to a build directory
ifdef SNPE_ROOT
CC_FLAGS += -I $(SNPE_ROOT)/include/zdl
else ifdef ZDL_ROOT
CC_FLAGS += -I $(ZDL_ROOT)/x86_64-linux-clang/include/zdl
else
$(error SNPE_ROOT: Please set SNPE_ROOT or ZDL_ROOT to obtain Udo headers necessary to compile the package)
endif

# only build the shared object if dynamic option specified in the variant
ifeq (1,$(V_dynamic))
BUILD_DLLS = libUdo${package.name}Impl${runtime}
endif

# sources for the DSP implementation library in src directory
SRC_DIR = ./
libUdo${package.name}Impl${runtime}.C_SRCS := $(wildcard $(SRC_DIR)/*.c)

%endif

# copy final build products to the ship directory
BUILD_COPIES = $(DLLS) $(EXES) $(LIBS) $(UDO_PACKAGE_ROOT)/libs/dsp_${str(dsp_arch_type)}/

# always last
include $(RULES_MIN)

# define destination library directory, and copy files into lib/dsp
# this code will create it
SHIP_LIBS_DIR   := $(CURDIR)/$(V)
LIB_DIR         := $(UDO_PACKAGE_ROOT)/libs/dsp_${str(dsp_arch_type)}
OBJ_DIR         := $(UDO_PACKAGE_ROOT)/obj/local/dsp_${str(dsp_arch_type)}

.PHONY: dsp

dsp: tree
	mkdir -p ${"${OBJ_DIR}"};  ${"\\"}
	cp -Rf ${"${SHIP_LIBS_DIR}"}/. ${"${OBJ_DIR}"} ;${"\\"}
	rm -rf ${"${SHIP_LIBS_DIR}"};

%if str(dsp_arch_type).lower() == 'v68':
X86_LIBNATIVE_RELEASE_DIR = $(HEXAGON_SDK_ROOT)/tools/HEXAGON_Tools/8.4.06/Tools
X86_OBJ_DIR = $(UDO_PACKAGE_ROOT)/obj/local/x86-64_linux_clang/
X86_LIB_DIR = $(UDO_PACKAGE_ROOT)/libs/x86-64_linux_clang/

X86_CXX ?= clang++
X86_C__FLAGS = -D__HVXDBL__ -I$(X86_LIBNATIVE_RELEASE_DIR)/libnative/include -ffast-math -DUSE_OS_LINUX
X86_CXXFLAGS = -std=c++17 -I$(QNN_INCLUDE) -I$(QNN_HTP_INCLUDE) -I$(QNN_HTP_INCLUDE)/core -fPIC -Wall -Wreorder -Wno-missing-braces -Werror -Wno-format -Wno-unused-command-line-argument -fvisibility=default -stdlib=libc++
X86_CXXFLAGS += -DQNN_API="__attribute__((visibility(\"default\")))"  -D__QAIC_HEADER_EXPORT="__attribute__((visibility(\"default\")))"
X86_CXXFLAGS += $(X86_C__FLAGS) -fomit-frame-pointer -Wno-invalid-offsetof -DTHIS_PKG_NAME=$(PKG_NAME)
X86_LDFLAGS =  -Wl,--whole-archive -L$(X86_LIBNATIVE_RELEASE_DIR)/libnative/lib -lnative -Wl,--no-whole-archive -lpthread

OBJS = $(patsubst %.cpp,%.o,$($(BUILD_DLLS).CXX_SRCS))

X86_DIR:
	mkdir -p $(X86_OBJ_DIR)
	mkdir -p $(X86_LIB_DIR)

$(X86_OBJ_DIR)/%.o: %.cpp
	$(X86_CXX) $(X86_CXXFLAGS) -MMD -c $< -o $@

$(X86_OBJ_DIR)/$(BUILD_DLLS).so: $(patsubst %,$(X86_OBJ_DIR)/%,$(OBJS))
	$(X86_CXX) -fPIC -std=c++17 -g -shared -o $@ $^ $(X86_LDFLAGS)

X86_DLL: $(X86_OBJ_DIR)/$(BUILD_DLLS).so

dsp_x86:  X86_DIR X86_DLL
	mv $(X86_OBJ_DIR)/$(BUILD_DLLS).so $(X86_LIB_DIR)

%endif
%endif
