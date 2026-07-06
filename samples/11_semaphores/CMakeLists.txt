# Copyright (c) 2019-2026 Ben Ashbaugh
#
# SPDX-License-Identifier: MIT

add_opencl_sample(
    TEST
    NUMBER 11
    TARGET semaphores
    VERSION 120
    SOURCES main.cpp
    LIBS OpenCLExt)
