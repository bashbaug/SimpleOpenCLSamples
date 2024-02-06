# Copyright (c) 2019-2024 Ben Ashbaugh
#
# SPDX-License-Identifier: MIT

function(add_opencl_tutorial)
    set(options TEST)
    set(one_value_args TARGET VERSION CATEGORY)
    set(multi_value_args SOURCES KERNELS INCLUDES LIBS)
    cmake_parse_arguments(OPENCL_TUTORIAL
        "${options}" "${one_value_args}" "${multi_value_args}"
        ${ARGN}
    )

    if(NOT OPENCL_TUTORIAL_VERSION)
        message(STATUS "No OpenCL version specified for tutorial ${OPENCL_TUTORIAL_TARGET}, using OpenCL 3.0.")
        set(OPENCL_TUTORIAL_VERSION 300)
    endif()

    add_executable(${OPENCL_TUTORIAL_TARGET} ${OPENCL_TUTORIAL_SOURCES})

    target_include_directories(${OPENCL_TUTORIAL_TARGET} PRIVATE ${OpenCL_INCLUDE_DIR} ${OPENCL_TUTORIAL_INCLUDES})
    target_link_libraries(${OPENCL_TUTORIAL_TARGET} ${OpenCL_LIBRARIES} ${OPENCL_TUTORIAL_LIBS})

    target_compile_definitions(${OPENCL_TUTORIAL_TARGET} PRIVATE CL_TARGET_OPENCL_VERSION=${OPENCL_TUTORIAL_VERSION})
    target_compile_definitions(${OPENCL_TUTORIAL_TARGET} PRIVATE CL_HPP_TARGET_OPENCL_VERSION=${OPENCL_TUTORIAL_VERSION})
    target_compile_definitions(${OPENCL_TUTORIAL_TARGET} PRIVATE CL_HPP_MINIMUM_OPENCL_VERSION=${OPENCL_TUTORIAL_VERSION})
    if (WIN32)
        target_compile_definitions(${OPENCL_TUTORIAL_TARGET} PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX)
    endif()

    set_target_properties(${OPENCL_TUTORIAL_TARGET} PROPERTIES FOLDER "Tutorials/${OPENCL_TUTORIAL_CATEGORY}/${OPENCL_TUTORIAL_TARGET}")

    if(CMAKE_CONFIGURATION_TYPES)
        set(OPENCL_TUTORIAL_CONFIGS ${CMAKE_CONFIGURATION_TYPES})
    else()
        set(OPENCL_TUTORIAL_CONFIGS ${CMAKE_BUILD_TYPE})
    endif()
    foreach(CONFIG ${OPENCL_TUTORIAL_CONFIGS})
        install(TARGETS ${OPENCL_TUTORIAL_TARGET} CONFIGURATIONS ${CONFIG} DESTINATION ${CONFIG})
        install(FILES ${OPENCL_TUTORIAL_KERNELS} CONFIGURATIONS ${CONFIG} DESTINATION ${CONFIG})
    endforeach()
    if(OPENCL_TUTORIAL_TEST)
        add_test(NAME ${OPENCL_TUTORIAL_TARGET} COMMAND ${OPENCL_TUTORIAL_TARGET})
    endif()
endfunction()

add_subdirectory( interceptlayer )
