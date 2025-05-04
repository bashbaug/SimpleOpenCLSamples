# Copyright (c) 2024-2025 Ben Ashbaugh
#
# SPDX-License-Identifier: MIT

add_opencl_sample(
    TEST
    NUMBER 20
    TARGET matrixexperiments-i8
    VERSION 200 # for clSetKernelExecInfo
    SOURCES main.cpp
    KERNELS matrix_helpers_i8.cl matrix_kernels_i8.cl)
