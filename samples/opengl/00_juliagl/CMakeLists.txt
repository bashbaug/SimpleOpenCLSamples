# Copyright (c) 2021-2025 Ben Ashbaugh
#
# SPDX-License-Identifier: MIT

add_opencl_sample(
    NUMBER 00
    TARGET juliagl
    VERSION 120
    CATEGORY opengl
    SOURCES main.cpp
    INCLUDES ${OPENGL_INCLUDE_DIRS}
    LIBS ${OPENGL_LIBRARIES} glfw)
