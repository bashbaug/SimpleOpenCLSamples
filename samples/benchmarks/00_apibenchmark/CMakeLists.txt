# Copyright (c) 2022-2024 Ben Ashbaugh
#
# SPDX-License-Identifier: MIT

add_opencl_sample(
    TEST
    NUMBER 00
    TARGET apibenchmark
    VERSION 200
    CATEGORY benchmarks
    SOURCES main.cpp
    LIBS benchmark::benchmark OpenCLExt)
