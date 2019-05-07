# Copyright (c) 2019 Ben Ashbaugh
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

cmake_minimum_required(VERSION 2.8.11)

set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_CXX_STANDARD 14)

if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, default to Release")
    set(CMAKE_BUILD_TYPE "Release" CACHE PATH "Build Type" FORCE)
endif()

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

project(SimpleOpenCLSample)

set(OpenCL_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/OpenCL-Headers CACHE PATH "Path to OpenCL Headers")
find_package(OpenCL)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${OpenCL_INCLUDE_DIR})

# This will setup the OpenCL headers and OpenCL C++ bindings to require
# no more than OpenCL 2.1.  This means that these samples should be
# deployable to any system that has an OpenCL 2.1 or newer ICD loader,
# and that they should run correctly on any OpenCL 2.1 implementation.
# Changing these defines to a newer OpenCL version will enable the new
# OpenCL functionality, however it will also require a newer ICD loader
# and a newer OpenCL implementation.
add_definitions(-DCL_TARGET_OPENCL_VERSION=210)
add_definitions(-DCL_HPP_TARGET_OPENCL_VERSION=210)
add_definitions(-DCL_HPP_MINIMUM_OPENCL_VERSION=210)

add_subdirectory(external/opencl-icd-loader)

set_target_properties(icd_loader_test  PROPERTIES FOLDER "OpenCL-ICD-Loader")
set_target_properties(IcdLog           PROPERTIES FOLDER "OpenCL-ICD-Loader")
set_target_properties(OpenCL           PROPERTIES FOLDER "OpenCL-ICD-Loader")
set_target_properties(OpenCLDriverStub PROPERTIES FOLDER "OpenCL-ICD-Loader")

add_subdirectory(samples)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/install" CACHE PATH "Install Path" FORCE)
endif()
