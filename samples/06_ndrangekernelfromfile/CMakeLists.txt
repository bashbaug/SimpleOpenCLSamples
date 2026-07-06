# Copyright (c) 2019-2026 Ben Ashbaugh
#
# SPDX-License-Identifier: MIT

add_opencl_sample(
    TEST
    NUMBER 06
    TARGET ndrangekernelfromfile
    VERSION 120
    SOURCES main.cpp
    KERNELS ndrange_sample_kernel.cl)
