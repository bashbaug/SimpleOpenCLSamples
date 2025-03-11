# Copyright (c) 2019-2025 Ben Ashbaugh
#
# SPDX-License-Identifier: MIT

function(add_opencl_sample)
    set(options TEST)
    set(one_value_args NUMBER TARGET VERSION CATEGORY)
    set(multi_value_args SOURCES KERNELS INCLUDES LIBS)
    cmake_parse_arguments(OPENCL_SAMPLE
        "${options}" "${one_value_args}" "${multi_value_args}"
        ${ARGN}
    )

    if(NOT OPENCL_SAMPLE_VERSION)
        message(STATUS "No OpenCL version specified for sample ${OPENCL_SAMPLE_TARGET}, using OpenCL 3.0.")
        set(OPENCL_SAMPLE_VERSION 300)
    endif()
    if(NOT OPENCL_SAMPLE_NUMBER)
        message(STATUS "No sample number specified for sample ${OPENCL_SAMPLE_TARGET}, using 99.")
        set(OPENCL_SAMPLE_NUMBER 99)
    endif()

    add_executable(${OPENCL_SAMPLE_TARGET} ${OPENCL_SAMPLE_SOURCES})

    target_include_directories(${OPENCL_SAMPLE_TARGET} PRIVATE ${OpenCL_INCLUDE_DIR} ${OPENCL_SAMPLE_INCLUDES})
    target_link_libraries(${OPENCL_SAMPLE_TARGET} ${OpenCL_LIBRARIES} ${OPENCL_SAMPLE_LIBS})

    target_compile_definitions(${OPENCL_SAMPLE_TARGET} PRIVATE CL_TARGET_OPENCL_VERSION=${OPENCL_SAMPLE_VERSION})
    target_compile_definitions(${OPENCL_SAMPLE_TARGET} PRIVATE CL_ENABLE_BETA_EXTENSIONS)
    target_compile_definitions(${OPENCL_SAMPLE_TARGET} PRIVATE CL_HPP_TARGET_OPENCL_VERSION=${OPENCL_SAMPLE_VERSION})
    target_compile_definitions(${OPENCL_SAMPLE_TARGET} PRIVATE CL_HPP_MINIMUM_OPENCL_VERSION=${OPENCL_SAMPLE_VERSION})
    if (SAMPLES_ENABLE_EXCEPTIONS)
        target_compile_definitions(${OPENCL_SAMPLE_TARGET} PRIVATE CL_HPP_ENABLE_EXCEPTIONS)
    endif()
    if (WIN32)
        target_compile_definitions(${OPENCL_SAMPLE_TARGET} PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX)
    endif()

    set_target_properties(${OPENCL_SAMPLE_TARGET} PROPERTIES FOLDER "Samples/${OPENCL_SAMPLE_CATEGORY}/${OPENCL_SAMPLE_NUMBER}_${OPENCL_SAMPLE_TARGET}")

    if(CMAKE_CONFIGURATION_TYPES)
        set(OPENCL_SAMPLE_CONFIGS ${CMAKE_CONFIGURATION_TYPES})
    else()
        set(OPENCL_SAMPLE_CONFIGS ${CMAKE_BUILD_TYPE})
    endif()
    foreach(CONFIG ${OPENCL_SAMPLE_CONFIGS})
        install(TARGETS ${OPENCL_SAMPLE_TARGET} CONFIGURATIONS ${CONFIG} DESTINATION ${CONFIG})
        install(FILES ${OPENCL_SAMPLE_KERNELS} CONFIGURATIONS ${CONFIG} DESTINATION ${CONFIG})
    endforeach()
    if(OPENCL_SAMPLE_TEST)
        add_test(NAME ${OPENCL_SAMPLE_TARGET} COMMAND ${OPENCL_SAMPLE_TARGET})
    endif()
endfunction()

add_subdirectory( images )
add_subdirectory( opengl )
add_subdirectory( python )
add_subdirectory( vulkan )
add_subdirectory( svm )
add_subdirectory( usm )

add_subdirectory( 00_enumopencl )
add_subdirectory( 00_enumopenclpp )
add_subdirectory( 00_enumqueuefamilies )
add_subdirectory( 00_extendeddevicequeries )
add_subdirectory( 00_loaderinfo )
add_subdirectory( 00_newqueries )
add_subdirectory( 00_newqueriespp )
add_subdirectory( 01_copybuffer )
add_subdirectory( 02_copybufferkernel )
add_subdirectory( 03_mandelbrot )
add_subdirectory( 04_julia )
add_subdirectory( 04_sobel )
add_subdirectory( 05_kernelfromfile )
add_subdirectory( 05_spirvkernelfromfile )
add_subdirectory( 06_ndrangekernelfromfile )

add_subdirectory( 10_queueexperiments )
add_subdirectory( 16_floatatomics )

set(BUILD_EXTENSION_SAMPLES TRUE)
if(NOT TARGET OpenCLExt)
    message(STATUS "Skipping Extension Samples - OpenCL Extension Loader is not found.")
    set(BUILD_EXTENSION_SAMPLES FALSE)
endif()

if(BUILD_EXTENSION_SAMPLES)
    add_subdirectory( 11_semaphores )
    add_subdirectory( 12_commandbuffers )
    add_subdirectory( 12_commandbufferspp )
    add_subdirectory( 13_mutablecommandbuffers )
    add_subdirectory( 14_ooqcommandbuffers )
    add_subdirectory( 15_mutablecommandbufferasserts )
endif()

add_subdirectory( 99_svmfree )
