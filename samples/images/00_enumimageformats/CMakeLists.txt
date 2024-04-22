# Copyright (c) 2019-2024 Ben Ashbaugh
#
# SPDX-License-Identifier: MIT

add_opencl_sample(
    TEST
    NUMBER 00
    TARGET enumimageformats
    VERSION 300 # for OpenCL 2.0+ image formats
    CATEGORY images
    SOURCES main.cpp)
