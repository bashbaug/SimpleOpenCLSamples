# Copyright (c) 2024 Ben Ashbaugh
#
# SPDX-License-Identifier: MIT

add_opencl_sample(
    TEST
    NUMBER 99
    TARGET attentionexperiments
    VERSION 200 # for clSetKernelExecInfo
    SOURCES main.cpp
    KERNELS attention_kernels.cl)
