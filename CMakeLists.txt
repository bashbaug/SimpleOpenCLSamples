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

# Setup OCL C headers
find_path(OPENCL_INCLUDE_DIR "CL/cl.h" "/usr/include" "/usr/local/include")

if (OPENCL_INCLUDE_DIR)
    message(STATUS "Found OCL C headers in " ${OPENCL_INCLUDE_DIR})
else()
    message(STATUS "OCL headers not found, initializing local clone...")
    execute_process(COMMAND git submodule update --init -- external/OpenCL-Headers
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    set(OpenCL_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/OpenCL-Headers CACHE PATH "Path to OpenCL Headers")
    include_directories(${OpenCL_INCLUDE_DIR})
    find_package(OpenCL)
endif()

# Setup OCL C++ headers
find_path(OPENCL_CPP_INCLUDE_DIR "CL/cl2.hpp" "/usr/include" "/usr/local/include")

if (OPENCL_CPP_INCLUDE_DIR)
    message(STATUS "Found OCL C++ headers in " ${OPENCL_CPP_INCLUDE_DIR})
else()
    message(STATUS "OCL C++ headers not found, using local headers...")
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
endif()

find_library(OPENCL_LOADER_PATH "OpenCL" "/usr/lib" "/usr/local/lib")

if (OPENCL_LOADER_PATH)
    message(STATUS "Found OCL ICD loader in " ${OPENCL_LOADER_PATH})
else()
    message(STATUS "OCL Loader not found, initializing local clone...")
    execute_process(COMMAND git submodule update --init -- external/opencl-icd-loader
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    add_subdirectory(external/opencl-icd-loader)
    set_target_properties(icd_loader_test  PROPERTIES FOLDER "OpenCL-ICD-Loader")
    set_target_properties(IcdLog           PROPERTIES FOLDER "OpenCL-ICD-Loader")
    set_target_properties(OpenCL           PROPERTIES FOLDER "OpenCL-ICD-Loader")
    set_target_properties(OpenCLDriverStub PROPERTIES FOLDER "OpenCL-ICD-Loader")
endif()

add_subdirectory(samples)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/install" CACHE PATH "Install Path" FORCE)
endif()
