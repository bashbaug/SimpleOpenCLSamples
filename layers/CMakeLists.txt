# Copyright (c) 2022-2025 Ben Ashbaugh
#
# SPDX-License-Identifier: MIT

function(add_opencl_layer)
    set(options TEST)
    set(one_value_args NUMBER TARGET VERSION CATEGORY)
    set(multi_value_args SOURCES KERNELS INCLUDES LIBS)
    cmake_parse_arguments(OPENCL_LAYER
        "${options}" "${one_value_args}" "${multi_value_args}"
        ${ARGN}
    )

    if(NOT OPENCL_LAYER_VERSION)
        message(STATUS "No OpenCL version specified for sample ${OPENCL_LAYER_TARGET}, using OpenCL 3.0.")
        set(OPENCL_LAYER_VERSION 300)
    endif()
    if(NOT OPENCL_LAYER_NUMBER)
        message(STATUS "No sample number specified for sample ${OPENCL_LAYER_TARGET}, using 99.")
        set(OPENCL_LAYER_NUMBER 99)
    endif()

    add_library(${OPENCL_LAYER_TARGET} MODULE ${OPENCL_LAYER_SOURCES})

    target_include_directories(${OPENCL_LAYER_TARGET} PRIVATE ${OpenCL_INCLUDE_DIR} ${OPENCL_LAYER_INCLUDES})
    target_link_libraries(${OPENCL_LAYER_TARGET} ${OPENCL_LAYER_LIBS})

    target_compile_definitions(${OPENCL_LAYER_TARGET} PRIVATE CL_TARGET_OPENCL_VERSION=${OPENCL_LAYER_VERSION})
    target_compile_definitions(${OPENCL_LAYER_TARGET} PRIVATE CL_ENABLE_BETA_EXTENSIONS)
    target_compile_definitions(${OPENCL_LAYER_TARGET} PRIVATE CL_USE_DEPRECATED_OPENCL_1_0_APIS)
    target_compile_definitions(${OPENCL_LAYER_TARGET} PRIVATE CL_USE_DEPRECATED_OPENCL_1_1_APIS)
    target_compile_definitions(${OPENCL_LAYER_TARGET} PRIVATE CL_USE_DEPRECATED_OPENCL_1_2_APIS)
    target_compile_definitions(${OPENCL_LAYER_TARGET} PRIVATE CL_USE_DEPRECATED_OPENCL_2_0_APIS)
    target_compile_definitions(${OPENCL_LAYER_TARGET} PRIVATE CL_USE_DEPRECATED_OPENCL_2_1_APIS)
    target_compile_definitions(${OPENCL_LAYER_TARGET} PRIVATE CL_USE_DEPRECATED_OPENCL_2_2_APIS)

    if (WIN32)
        target_compile_definitions(${OPENCL_LAYER_TARGET} PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX)
    endif()

    set_target_properties(${OPENCL_LAYER_TARGET} PROPERTIES FOLDER "Layers/${OPENCL_LAYER_CATEGORY}/${OPENCL_LAYER_NUMBER}_${OPENCL_LAYER_TARGET}")

    if(CMAKE_CONFIGURATION_TYPES)
        set(OPENCL_LAYER_CONFIGS ${CMAKE_CONFIGURATION_TYPES})
    else()
        set(OPENCL_LAYER_CONFIGS ${CMAKE_BUILD_TYPE})
    endif()
    foreach(CONFIG ${OPENCL_LAYER_CONFIGS})
        install(TARGETS ${OPENCL_LAYER_TARGET} CONFIGURATIONS ${CONFIG} DESTINATION ${CONFIG})
        install(FILES ${OPENCL_LAYER_KERNELS} CONFIGURATIONS ${CONFIG} DESTINATION ${CONFIG})
    endforeach()
endfunction()

add_subdirectory( 00_example )

add_subdirectory( 10_cmdbufemu )
add_subdirectory( 11_semaemu )

set(BUILD_EXTENSION_LAYERS TRUE)
if(NOT TARGET OpenCLExt)
    message(STATUS "Skipping Extension Layers - OpenCL Extension Loader is not found.")
    set(BUILD_EXTENSION_LAYERS FALSE)
endif()

if(BUILD_EXTENSION_LAYERS)
    add_subdirectory( 99_svmplusplus )
endif()
