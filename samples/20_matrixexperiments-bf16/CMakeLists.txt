# Copyright (c) 2024-2025 Ben Ashbaugh
#
# SPDX-License-Identifier: MIT

add_opencl_sample(
    TEST
    NUMBER 20
    TARGET matrixexperiments-bf16
    VERSION 200 # for clSetKernelExecInfo
    SOURCES main.cpp
    KERNELS matrix_helpers_bf16.cl matrix_kernels_bf16.cl matrix_kernel_tiled_bf16.cl)
