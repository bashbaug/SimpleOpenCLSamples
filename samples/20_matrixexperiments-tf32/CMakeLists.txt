# Copyright (c) 2024-2025 Ben Ashbaugh
#
# SPDX-License-Identifier: MIT

add_opencl_sample(
    TEST
    NUMBER 20
    TARGET matrixexperiments-tf32
    VERSION 200 # for clSetKernelExecInfo
    SOURCES main.cpp
    KERNELS matrix_helpers_tf32.cl matrix_kernels_tf32.cl matrix_kernel_tiled_tf32.cl)
