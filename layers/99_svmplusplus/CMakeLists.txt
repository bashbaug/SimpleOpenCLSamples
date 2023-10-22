# Copyright (c) 2023 Ben Ashbaugh
#
# SPDX-License-Identifier: MIT

add_opencl_layer(
    NUMBER 99
    TARGET SvmPlusPlus
    VERSION 300
    SOURCES main.cpp emulate.cpp emulate.h
    LIBS OpenCLExt)
