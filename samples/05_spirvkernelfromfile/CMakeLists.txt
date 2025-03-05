# Copyright (c) 2019-2025 Ben Ashbaugh
#
# SPDX-License-Identifier: MIT

add_opencl_sample(
    TEST
    NUMBER 05
    TARGET spirvkernelfromfile
    VERSION 210
    SOURCES main.cpp
    KERNELS sample_kernel32.spv sample_kernel64.spv)
