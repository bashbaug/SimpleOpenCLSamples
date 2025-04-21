# Copyright (c) 2025 Ben Ashbaugh
#
# SPDX-License-Identifier: MIT

add_opencl_sample(
    TEST
    NUMBER 16
    TARGET relaxedallocations
    VERSION 200
    SOURCES main.cpp
    LIBS OpenCLExt)
