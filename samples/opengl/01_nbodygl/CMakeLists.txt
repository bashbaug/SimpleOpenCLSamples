# Copyright (c) 2022-2024 Ben Ashbaugh
#
# SPDX-License-Identifier: MIT

add_opencl_sample(
    NUMBER 01
    TARGET nbodygl
    VERSION 120
    CATEGORY opengl
    SOURCES main.cpp
    INCLUDES ${OPENGL_INCLUDE_DIRS}
    LIBS ${OPENGL_LIBRARIES} glfw)
