# Copyright (c) 2019-2024 Ben Ashbaugh
#
# SPDX-License-Identifier: MIT

add_opencl_sample(
    TEST
    NUMBER 05
    TARGET kernelfromfile
    VERSION 120
    SOURCES main.cpp
    KERNELS sample_kernel.cl)
