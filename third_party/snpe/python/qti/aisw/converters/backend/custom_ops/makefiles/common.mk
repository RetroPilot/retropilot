# ==============================================================================
#
#  Copyright (c) 2020 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

# define include paths
# set package include paths, note package root will take precedence
# if includes are already present in package
INCLUDES += -I $(UDO_PACKAGE_ROOT)/include

# if SNPE_ROOT is set and points to the SDK path, it will be used. Otherwise it will be assumed that ZDL_ROOT refers
# to the build directory
ifdef SNPE_ROOT
INCLUDES += -I $(SNPE_ROOT)/include/zdl
else ifdef ZDL_ROOT
INCLUDES += -I $(ZDL_ROOT)/x86_64-linux-clang/include/zdl
else
$(error SNPE_ROOT: Please set SNPE_ROOT or ZDL_ROOT to obtain Udo headers necessary to compile the package)
endif

# set compiler flags
CXXFLAGS += -std=c++11 -fPIC $(TARGET_AARCH_VARS) $(INCLUDES)

# set runtime specific compiler flags
ifdef CL_INCLUDE_PATH
CXXFLAGS += -I $(CL_INCLUDE_PATH)
endif

# define library sources
SOURCES := $(wildcard $(SRC_DIR)/*.cpp)

# define utility source directory
UTIL_SRC_DIR := ../utils

# define utility sources
UTIL_SOURCES := $(wildcard $(UTIL_SRC_DIR)/*.cpp)

# Make runtime specific adjustments to sources
ifeq ($(RUNTIME), GPU)
	UTIL_SOURCES += $(wildcard $(UTIL_SRC_DIR)/$(RUNTIME)/*.cpp)
endif

# define object directory
# Make runtime specific adjustments to directory structure
ifdef RUNTIME
OBJ_DIR := ../../../obj/local/$(TARGET)/$(RUNTIME)
else
OBJ_DIR :=../../../obj/local/$(TARGET)
endif

# define utility object directory
UTIL_OBJ_DIR = $(OBJ_DIR)

# setup object files in object directory
OBJECTS := $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(foreach x,$(SOURCES),$(notdir $(x))))

# setup utility object files
UTIL_OBJECTS := $(patsubst %.cpp,$(UTIL_OBJ_DIR)/%.o,$(foreach x,$(UTIL_SOURCES),$(notdir $(x))))

ifeq ($(RUNTIME), GPU)
	LINKFLAGS += -lOpenCL -L$(CL_LIBRARY_PATH)
endif

# Rule to make library
.PHONY: library
library: $(library)

# Implicit rule to compile and link object files with utils
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@

# implicit rule to compile util files
$(UTIL_OBJ_DIR)/%.o: $(UTIL_SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@

# implicit rule to compile runtime specific util files
$(UTIL_OBJ_DIR)/%.o: $(UTIL_SRC_DIR)/$(RUNTIME)/%.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@

# set up resources
directories := $(LIB_DIR) $(OBJ_DIR)

# Compile
$(library): $(OBJECTS) $(UTIL_OBJECTS) | $(directories)
	$(CXX) $(CXXFLAGS) $(LINKFLAGS) -shared $^ -o $@

# rule for object directory resource
$(OBJECTS): | $(OBJ_DIR)

# rule to create directories
$(directories):
	mkdir -p $@

.PHONY: clean
clean:
	rm -rf $(OBJ_DIR) $(LIB_DIR)
