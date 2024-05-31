# Copyright (c) 2019-2024 Ben Ashbaugh
#
# SPDX-License-Identifier: MIT

add_opencl_sample(
    TEST
    NUMBER 99
    TARGET flashattention
    VERSION 200 # for clSetKernelExecInfo
    SOURCES main.cpp
    KERNELS flashattention_kernels.cl)
