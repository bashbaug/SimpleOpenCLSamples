# Copyright (c) 2019-2024 Ben Ashbaugh
#
# SPDX-License-Identifier: MIT

add_opencl_sample(
    TEST
    NUMBER 11
    TARGET apitimes
    VERSION 200 # for clCreateCommandQueueWithProperties
    SOURCES main.cpp)
