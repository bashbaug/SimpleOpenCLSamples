# Copyright (c) 2021-2026 Ben Ashbaugh
#
# SPDX-License-Identifier: MIT

add_opencl_sample(
    NUMBER 00
    TARGET nbodyvk
    VERSION 300 # clCreateBufferWithProperties
    CATEGORY vulkan
    SOURCES main.cpp
    KERNELS nbodyvk.vert.spv nbodyvk.frag.spv
    INCLUDES ${Vulkan_INCLUDE_DIR}
    LIBS ${Vulkan_LIBRARY} glfw)
