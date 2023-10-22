# Copyright (c) 2023 Ben Ashbaugh
#
# SPDX-License-Identifier: MIT

add_opencl_sample(
    TEST
    NUMBER 301
    TARGET usmemlinkedlist
    VERSION 200
    CATEGORY usvm
    SOURCES main.cpp
    LIBS OpenCLExt)
