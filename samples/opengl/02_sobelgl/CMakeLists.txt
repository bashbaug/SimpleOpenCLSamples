# Copyright (c) 2023-2026 Ben Ashbaugh
#
# SPDX-License-Identifier: MIT

add_opencl_sample(
    NUMBER 02
    TARGET sobelgl
    VERSION 120
    CATEGORY opengl
    SOURCES main.cpp
    INCLUDES ${OPENGL_INCLUDE_DIRS}
    LIBS ${OPENGL_LIBRARIES} glfw)
