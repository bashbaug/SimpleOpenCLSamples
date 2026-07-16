# Copyright (c) 2026 Ben Ashbaugh
#
# SPDX-License-Identifier: MIT

add_opencl_sample(
    NUMBER 21
    TARGET multithreading
    VERSION 120
    SOURCES main.cpp
    LIBS Threads::Threads)
