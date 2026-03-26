# Copyright (c) 2025-2026 Ben Ashbaugh
#
# SPDX-License-Identifier: MIT

add_opencl_sample(
    TEST
    NUMBER 99
    TARGET globalvar
    VERSION 210
    SOURCES main.cpp
    KERNELS device_global64.spv)
