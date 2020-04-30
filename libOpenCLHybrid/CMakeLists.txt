cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_CXX_STANDARD 11)

project(OpenCLHybrid LANGUAGES CXX)

if(NOT ${OpenCL_INCLUDE_DIRS})
    find_package(OpenCL)
endif()
include_directories(
    ${OpenCL_INCLUDE_DIRS}
)

set( OPENCLHYBRID_SOURCE_FILES
    src/loader.cpp
)

source_group(Source FILES ${OPENCLHYBRID_SOURCE_FILES})

add_library(OpenCLHybrid STATIC ${OPENCLHYBRID_SOURCE_FILES})
set_target_properties(OpenCLHybrid PROPERTIES FOLDER "OpenCLHybrid")
target_link_libraries(OpenCLHybrid ${CMAKE_DL_LIBS})
target_compile_definitions(OpenCLHybrid PRIVATE CL_TARGET_OPENCL_VERSION=220)

if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    target_compile_options(OpenCLHybrid PRIVATE /EHsc)
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    target_compile_options(OpenCLHybrid PRIVATE -Wall)
    set_target_properties(OpenCLHybrid PROPERTIES LINK_FLAGS "-Wl,--version-script")
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
    # TODO
endif()
