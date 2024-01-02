# Copyright (c) 2021-2024 Ben Ashbaugh
#
# SPDX-License-Identifier: MIT

add_opencl_sample(
    NUMBER 00
    TARGET juliavk
    VERSION 300 # clCreateImageWithProperties
    CATEGORY vulkan
    SOURCES main.cpp
    KERNELS juliavk.vert.spv juliavk.frag.spv
    INCLUDES ${Vulkan_INCLUDE_DIR}
    LIBS ${Vulkan_LIBRARY} glfw)
