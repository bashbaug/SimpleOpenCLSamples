# Copyright (c) 2022-2024 Ben Ashbaugh
#
# SPDX-License-Identifier: MIT

set(BUILD_BENCHMARK_SAMPLES TRUE)
if(NOT TARGET benchmark::benchmark)
    message(STATUS "Skipping Benchmark Samples - Google Benchmark is not found.")
    set(BUILD_BENCHMARK_SAMPLES FALSE)
endif()

if(BUILD_BENCHMARK_SAMPLES)
    add_subdirectory( 00_apibenchmark )
    add_subdirectory( 00_fileiobenchmark )
endif()
