# Copyright (c) 2025 Ben Ashbaugh
#
# SPDX-License-Identifier: MIT

add_opencl_sample(
    TEST
    NUMBER 99
    TARGET blockreads
    VERSION 120
    SOURCES main.cpp
    KERNELS block_read_kernel.cl)
