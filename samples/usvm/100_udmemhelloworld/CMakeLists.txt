# Copyright (c) 2023 Ben Ashbaugh
#
# SPDX-License-Identifier: MIT

add_opencl_sample(
    TEST
    NUMBER 100
    TARGET udmemhelloworld
    VERSION 300
    CATEGORY usvm
    SOURCES main.cpp
    LIBS OpenCLExt)
