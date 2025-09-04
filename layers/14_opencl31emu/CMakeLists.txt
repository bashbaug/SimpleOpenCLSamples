# Copyright (c) 2025-2026 Ben Ashbaugh
#
# SPDX-License-Identifier: MIT

add_opencl_layer(
    NUMBER 14
    TARGET OpenCL31Emu
    VERSION 310
    SOURCES main.cpp emulate.cpp emulate.h)
