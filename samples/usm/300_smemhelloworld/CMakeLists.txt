# Copyright (c) 2020-2024 Ben Ashbaugh
#
# SPDX-License-Identifier: MIT

add_opencl_sample(
    TEST
    NUMBER 300
    TARGET smemhelloworld
    VERSION 120
    CATEGORY usm
    SOURCES main.cpp
    LIBS OpenCLExt)
