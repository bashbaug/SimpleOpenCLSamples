# Copyright (c) 2025 Ben Ashbaugh
#
# SPDX-License-Identifier: MIT

add_opencl_sample(
    TEST
    NUMBER 00
    TARGET spirvqueries
    VERSION 120
    SOURCES main.cpp
    INCLUDES ${SPIRV-Headers_SOURCE_DIR}/include)
