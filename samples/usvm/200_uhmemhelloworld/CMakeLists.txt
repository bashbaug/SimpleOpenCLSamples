# Copyright (c) 2023 Ben Ashbaugh
#
# SPDX-License-Identifier: MIT

add_opencl_sample(
    TEST
    NUMBER 200
    TARGET uhmemhelloworld
    VERSION 300
    CATEGORY usvm
    SOURCES main.cpp
    LIBS OpenCLExt)
