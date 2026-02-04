# Copyright (c) 2020-2026 Ben Ashbaugh
#
# SPDX-License-Identifier: MIT

add_opencl_sample(
    TEST
    NUMBER 101
    TARGET dmemlinkedlist
    VERSION 200
    CATEGORY usm
    SOURCES main.cpp
    LIBS OpenCLExt)
