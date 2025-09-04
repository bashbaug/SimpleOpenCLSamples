# Copyright (c) 2025 Ben Ashbaugh
#
# SPDX-License-Identifier: MIT

add_opencl_layer(
    NUMBER 21
    TARGET SGRotateEmu
    VERSION 300
    SOURCES main.cpp emulate.cpp emulate.h)
