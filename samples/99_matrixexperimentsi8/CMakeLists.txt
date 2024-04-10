# Copyright (c) 2019-2024 Ben Ashbaugh
#
# SPDX-License-Identifier: MIT

add_opencl_sample(
    TEST
    NUMBER 99
    TARGET matrixexperimentsi8
    VERSION 200 # for clSetKernelExecInfo
    SOURCES main.cpp
    KERNELS matrix_helpers_i8.cl matrix_kernels_i8.cl matrix_kernel_tiled_i8.cl)
