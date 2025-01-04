# Copyright (c) 2022-2025 Ben Ashbaugh
#
# SPDX-License-Identifier: MIT

add_opencl_layer(
    NUMBER 10
    TARGET CmdBufEmu
    VERSION 300
    SOURCES main.cpp emulate.cpp emulate.h)
