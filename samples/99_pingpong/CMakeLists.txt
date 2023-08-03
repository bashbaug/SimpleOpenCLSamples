# Copyright (c) 2023 Ben Ashbaugh
#
# SPDX-License-Identifier: MIT

add_opencl_sample(
    TEST
    NUMBER 99
    TARGET pingpong
    VERSION 300
    SOURCES main.cpp
    LIBS OpenCLExt)
