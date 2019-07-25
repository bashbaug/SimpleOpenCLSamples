cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_CXX_STANDARD 11)

project(OpenCLStatic LANGUAGES CXX)

if(NOT ${OpenCL_INCLUDE_DIRS})
    find_package(OpenCL)
endif()
include_directories(
    ${OpenCL_INCLUDE_DIRS}
)

set( OPENCLSTATIC_SOURCE_FILES
    src/loader.cpp
)

source_group(Source FILES ${OPENCLSTATIC_SOURCE_FILES})

add_library(OpenCLStatic STATIC ${OPENCLSTATIC_SOURCE_FILES})
set_target_properties(OpenCLStatic PROPERTIES FOLDER "OpenCLStatic")
target_link_libraries(OpenCLStatic ${CMAKE_DL_LIBS})
target_compile_definitions(OpenCLStatic PRIVATE CL_TARGET_OPENCL_VERSION=220)

if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    target_compile_options(OpenCLStatic PRIVATE /EHsc)
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    target_compile_options(OpenCLStatic PRIVATE -Wall)
    set_target_properties(OpenCLStatic PROPERTIES LINK_FLAGS "-Wl,--version-script")
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
    # TODO
endif()
