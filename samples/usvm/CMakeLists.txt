# Copyright (c) 2023 Ben Ashbaugh
#
# SPDX-License-Identifier: MIT

set(BUILD_USVM_SAMPLES TRUE)
if(NOT TARGET OpenCLExt)
    message(STATUS "Skipping USM Samples - OpenCL Extension Loader is not found.")
    set(BUILD_USVM_SAMPLES FALSE)
endif()

if(BUILD_USVM_SAMPLES)
    add_subdirectory( 00_usvmqueries )
    add_subdirectory( 01_usvmmeminfo )

    add_subdirectory( 100_udmemhelloworld )
    add_subdirectory( 101_udmemlinkedlist )

    add_subdirectory( 200_uhmemhelloworld )
    add_subdirectory( 201_uhmemlinkedlist )

    add_subdirectory( 300_usmemhelloworld )
    add_subdirectory( 301_usmemlinkedlist )
endif()
