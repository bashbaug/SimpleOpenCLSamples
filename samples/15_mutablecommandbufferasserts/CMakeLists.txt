# Copyright (c) 2022-2025 Ben Ashbaugh
#
# SPDX-License-Identifier: MIT

add_opencl_sample(
    TEST
    NUMBER 12
    TARGET mutablecommandbufferasserts
    VERSION 120
    SOURCES main.cpp
    LIBS OpenCLExt)
