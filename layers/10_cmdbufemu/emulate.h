/*
// Copyright (c) 2022-2026 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <CL/cl.h>
#include <CL/cl_ext.h>

#include <map>

extern int  g_EmulateUnconditionally;
extern bool g_EnhancedErrorChecking;
extern bool g_KernelForProfiling;
extern bool g_SuggestedLocalWorkSize;

extern const struct _cl_icd_dispatch* g_pNextDispatch;

///////////////////////////////////////////////////////////////////////////////
// Override Functions

bool clGetDeviceInfo_override(
    cl_device_id device,
    cl_device_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret,
    cl_int* errcode_ret);

bool clGetEventInfo_override(
    cl_event event,
    cl_event_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret,
    cl_int* errcode_ret);

bool clGetEventProfilingInfo_override(
    cl_event event,
    cl_profiling_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret,
    cl_int* errcode_ret);

void* clGetExtensionFunctionAddressForPlatform_override(
    cl_platform_id platform,
    const char* func_name);

bool clGetPlatformInfo_override(
    cl_platform_id platform,
    cl_platform_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret,
    cl_int* errcode_ret);

///////////////////////////////////////////////////////////////////////////////
// Update Functions

void clReleaseEvent_update(
    cl_event event);
