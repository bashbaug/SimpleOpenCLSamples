/*
// Copyright (c) 2025 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <CL/cl.h>
#include <CL/cl_ext.h>

#include <map>

extern const struct _cl_icd_dispatch* g_pNextDispatch;

struct SLayerContext
{
    typedef std::map<cl_event, cl_event> CEventMap;
    CEventMap EventMap;
};

SLayerContext& getLayerContext(void);

///////////////////////////////////////////////////////////////////////////////
// Emulated Functions

cl_int CL_API_CALL clEnqueueHostTaskEXP_EMU(
    cl_command_queue queue,
    void(CL_CALLBACK* user_func)(void*),
    void* user_data,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event);

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

bool clGetPlatformInfo_override(
    cl_platform_id platform,
    cl_platform_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret,
    cl_int* errcode_ret);
