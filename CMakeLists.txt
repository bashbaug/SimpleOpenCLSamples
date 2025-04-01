# Copyright (c) 2019-2025 Ben Ashbaugh
#
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_CXX_STANDARD 14)

if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, default to Release")
    set(CMAKE_BUILD_TYPE "Release" CACHE PATH "Build Type" FORCE)
endif()

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

project(SimpleOpenCLSamples VERSION 1.0)

option(SAMPLES_ENABLE_EXCEPTIONS "Enable Exceptions for OpenCL Errors")

set(OpenCL_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/OpenCL-Headers CACHE PATH "Path to OpenCL Headers")
find_package(OpenCL)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${OpenCL_INCLUDE_DIR})

add_subdirectory(external/OpenCL-Headers)
add_subdirectory(external/opencl-icd-loader)
set_target_properties(OpenCL PROPERTIES FOLDER "OpenCL-ICD-Loader")
set_target_properties(cllayerinfo PROPERTIES FOLDER "OpenCL-ICD-Loader")
set(OpenCL_LIBRARIES OpenCL)

if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/external/opencl-extension-loader)
    add_subdirectory(external/opencl-extension-loader)
else()
    message(STATUS "OpenCL Extension Loader is not found.")
endif()

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    enable_testing()
endif()

add_subdirectory(layers)
add_subdirectory(samples)
add_subdirectory(tutorials)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/install" CACHE PATH "Install Path" FORCE)
endif()
