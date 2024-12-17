# Copyright (c) 2024 Ben Ashbaugh
#
# SPDX-License-Identifier: MIT

add_opencl_sample(
    TEST
    NUMBER 17
    TARGET concurrentdispatch
    VERSION 200 # for clSetKernelExecInfo
    SOURCES main.cpp)
