# Copyright (c) 2019-2024 Ben Ashbaugh
#
# SPDX-License-Identifier: MIT

add_opencl_sample(
    # don't test by default due to long execution time
    NUMBER 99
    TARGET matrixexperiments
    VERSION 200 # for clSetKernelExecInfo
    SOURCES main.cpp
    KERNELS matrix_helpers.cl matrix_kernels.cl matrix_kernel_tiled.cl)
