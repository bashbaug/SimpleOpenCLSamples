# Copyright (c) 2019-2020 Ben Ashbaugh
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

cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_CXX_STANDARD 14)

if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, default to Release")
    set(CMAKE_BUILD_TYPE "Release" CACHE PATH "Build Type" FORCE)
endif()

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

project(SimpleOpenCLSamples VERSION 1.0)

set(OpenCL_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/OpenCL-Headers CACHE PATH "Path to OpenCL Headers")
find_package(OpenCL)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${OpenCL_INCLUDE_DIR})

option(USE_RUNTIME_LOADER "Enable cliloader Support and Build the Executable")
if (USE_RUNTIME_LOADER)
add_subdirectory(external/opencl-runtime-loader)
set(OpenCL_LIBRARIES OpenCLRuntimeLoader ${CMAKE_DL_LIBS})
else()
add_subdirectory(external/opencl-icd-loader)
set_target_properties(OpenCL PROPERTIES FOLDER "OpenCL-ICD-Loader")
set(OpenCL_LIBRARIES OpenCL)
endif()

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    enable_testing()
endif()

add_subdirectory(samples)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/install" CACHE PATH "Install Path" FORCE)
endif()
