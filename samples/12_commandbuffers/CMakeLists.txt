# Copyright (c) 2022 Ben Ashbaugh
#
# SPDX-License-Identifier: MIT

add_opencl_sample(
    TEST
    NUMBER 12
    TARGET commandbuffers
    VERSION 120
    SOURCES main.cpp
    LIBS OpenCLExt)
