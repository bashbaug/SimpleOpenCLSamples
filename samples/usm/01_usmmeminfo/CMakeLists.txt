# Copyright (c) 2020-2025 Ben Ashbaugh
#
# SPDX-License-Identifier: MIT

add_opencl_sample(
    TEST
    NUMBER 01
    TARGET usmmeminfo
    VERSION 120
    CATEGORY usm
    SOURCES main.cpp
    LIBS OpenCLExt)
