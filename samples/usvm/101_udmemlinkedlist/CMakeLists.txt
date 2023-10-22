# Copyright (c) 2023 Ben Ashbaugh
#
# SPDX-License-Identifier: MIT

add_opencl_sample(
    TEST
    NUMBER 101
    TARGET udmemlinkedlist
    VERSION 300
    CATEGORY usvm
    SOURCES main.cpp
    LIBS OpenCLExt)
