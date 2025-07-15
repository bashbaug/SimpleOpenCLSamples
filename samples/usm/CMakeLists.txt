# Copyright (c) 2021-2025 Ben Ashbaugh
#
# SPDX-License-Identifier: MIT

set(BUILD_USM_SAMPLES TRUE)
if(NOT TARGET OpenCLExt)
    message(STATUS "Skipping USM Samples - OpenCL Extension Loader is not found.")
    set(BUILD_USM_SAMPLES FALSE)
endif()

if(BUILD_USM_SAMPLES)
    add_subdirectory( 00_usmqueries )
    add_subdirectory( 01_usmmeminfo )

    add_subdirectory( 100_dmemhelloworld )
    add_subdirectory( 101_dmemlinkedlist )

    add_subdirectory( 200_hmemhelloworld )
    add_subdirectory( 201_hmemlinkedlist )

    add_subdirectory( 300_smemhelloworld )
    add_subdirectory( 301_smemlinkedlist )
    add_subdirectory( 310_usmmigratemem )

    add_subdirectory( 400_sysmemhelloworld )

    add_subdirectory( 99_dmemindirect )
endif()
