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

add_subdirectory(external/opencl-icd-loader)
set_target_properties(icd_loader_test  PROPERTIES FOLDER "OpenCL-ICD-Loader")
set_target_properties(IcdLog           PROPERTIES FOLDER "OpenCL-ICD-Loader")
set_target_properties(OpenCL           PROPERTIES FOLDER "OpenCL-ICD-Loader")
set_target_properties(OpenCLDriverStub PROPERTIES FOLDER "OpenCL-ICD-Loader")

# This will setup the OpenCL headers and OpenCL C++ bindings to require
# no more than OpenCL 1.2.  This means that these samples should be
# deployable to any system that has an OpenCL 1.2 or newer ICD loader,
# and that they should run correctly on any OpenCL 1.2 implementation.
# Changing these defines to a newer OpenCL version will enable the new
# OpenCL functionality, however it will also require a newer ICD loader
# and a newer OpenCL implementation.
add_definitions(-DCL_TARGET_OPENCL_VERSION=120)
add_definitions(-DCL_HPP_TARGET_OPENCL_VERSION=120)
add_definitions(-DCL_HPP_MINIMUM_OPENCL_VERSION=120)

add_subdirectory(samples)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/install" CACHE PATH "Install Path" FORCE)
endif()
