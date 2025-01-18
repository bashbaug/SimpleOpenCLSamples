/*
// Copyright (c) 2025 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <CL/cl.h>
#include <CL/cl_ext.h>

extern const struct _cl_icd_dispatch* g_pNextDispatch;

#ifndef cl_khr_spirv_queries
#define cl_khr_spirv_queries 1
#define CL_KHR_SPIRV_QUERIES_EXTENSION_NAME "cl_khr_spirv_queries"
#define CL_DEVICE_SPIRV_EXTENDED_INSTRUCTION_SETS_KHR   0x12B9
#define CL_DEVICE_SPIRV_EXTENSIONS_KHR                  0x12BA
#define CL_DEVICE_SPIRV_CAPABILITIES_KHR                0x12BB
#endif

///////////////////////////////////////////////////////////////////////////////
// Override Functions

bool clGetDeviceInfo_override(
    cl_device_id device,
    cl_device_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret,
    cl_int* errcode_ret);

bool clGetPlatformInfo_override(
    cl_platform_id platform,
    cl_platform_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret,
    cl_int* errcode_ret);
