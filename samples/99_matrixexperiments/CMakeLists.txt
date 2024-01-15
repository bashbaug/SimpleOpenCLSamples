# Copyright (c) 2019-2024 Ben Ashbaugh
#
# SPDX-License-Identifier: MIT

add_opencl_sample(
    TEST
    NUMBER 99
    TARGET matrixexperiments
    VERSION 120
    SOURCES main.cpp
    KERNELS matrix_helpers.cl matrix_kernels.cl matrix_kernel_tiled.cl)
