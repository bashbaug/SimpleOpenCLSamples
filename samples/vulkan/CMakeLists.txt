# Copyright (c) 2020-2026 Ben Ashbaugh
#
# SPDX-License-Identifier: MIT

find_package(Vulkan)
find_package(glfw3 HINTS "../../external")

set(BUILD_VULKAN_SAMPLES TRUE)
if(NOT VULKAN_FOUND)
    message(STATUS "Skipping Vulkan Samples - Vulkan is not found.")
    set(BUILD_VULKAN_SAMPLES FALSE)
endif()
if(NOT glfw3_FOUND)
    message(STATUS "Skipping Vulkan Samples - GLFW is not found.")
    set(BUILD_VULKAN_SAMPLES FALSE)
endif()

if(BUILD_VULKAN_SAMPLES)
    add_subdirectory( 00_juliavk )
    add_subdirectory( 01_nbodyvk )
endif()
