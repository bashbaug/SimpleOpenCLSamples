# Copyright (c) 2022-2024 Ben Ashbaugh
#
# SPDX-License-Identifier: MIT

add_opencl_layer(
    NUMBER 10
    TARGET SemaEmu
    VERSION 300
    SOURCES main.cpp emulate.cpp emulate.h)
