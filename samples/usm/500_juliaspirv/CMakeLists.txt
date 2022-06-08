# Copyright (c) 2022 Ben Ashbaugh
#
# SPDX-License-Identifier: MIT

add_opencl_sample(
    TEST
    NUMBER 04
    TARGET juliaspirv
    VERSION 300
    SOURCES main.cpp
    KERNELS julia.spv
    LIBS OpenCLExt)
