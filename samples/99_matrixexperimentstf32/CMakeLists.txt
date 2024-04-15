# Copyright (c) 2019-2024 Ben Ashbaugh
#
# SPDX-License-Identifier: MIT

add_opencl_sample(
    # don't test by default due to long execution time
    NUMBER 99
    TARGET matrixexperimentstf32
    VERSION 120
    SOURCES main.cpp
    KERNELS matrix_helpers_tf32.cl matrix_kernels_tf32.cl matrix_kernel_tiled_tf32.cl)
