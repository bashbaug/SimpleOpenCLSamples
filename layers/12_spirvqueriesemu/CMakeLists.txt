# Copyright (c) 2025-2026 Ben Ashbaugh
#
# SPDX-License-Identifier: MIT

add_opencl_layer(
    NUMBER 12
    TARGET SpirvQueriesEmu
    VERSION 300
    SOURCES main.cpp emulate.cpp emulate.h
    INCLUDES ${SPIRV-Headers_SOURCE_DIR}/include)
