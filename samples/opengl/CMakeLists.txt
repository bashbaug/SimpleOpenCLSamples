# Copyright (c) 2020-2026 Ben Ashbaugh
#
# SPDX-License-Identifier: MIT

if(NOT DEFINED OpenGL_GL_PREFERENCE)
    set(OpenGL_GL_PREFERENCE "LEGACY")
endif()

find_package(OpenGL)
find_package(glfw3 HINTS "../../external")

set(BUILD_OPENGL_SAMPLES TRUE)
if(NOT OPENGL_FOUND)
    message(STATUS "Skipping OpenGL Samples - OpenGL is not found.")
    set(BUILD_OPENGL_SAMPLES FALSE)
endif()
if(NOT glfw3_FOUND)
    message(STATUS "Skipping OpenGL Samples - GLFW is not found.")
    set(BUILD_OPENGL_SAMPLES FALSE)
endif()

if(BUILD_OPENGL_SAMPLES)
    add_subdirectory( 00_juliagl )
    add_subdirectory( 01_nbodygl )
    add_subdirectory( 02_sobelgl )
endif()
