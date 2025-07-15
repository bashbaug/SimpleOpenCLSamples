# Copyright (c) 2025 Ben Ashbaugh
#
# SPDX-License-Identifier: MIT

add_opencl_sample(
    TEST
    NUMBER 99
    TARGET dmemindirect
    VERSION 200
    CATEGORY usm
    SOURCES main.cpp
    LIBS OpenCLExt)
