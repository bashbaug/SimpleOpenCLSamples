# Copyright (c) 2022 Ben Ashbaugh
#
# SPDX-License-Identifier: MIT

add_opencl_sample(
    TEST
    NUMBER 14
    TARGET ooqcommandbuffers
    VERSION 120
    SOURCES main.cpp
    LIBS OpenCLExt)
