/*
// Copyright (c) 2025 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <CL/cl.h>
#include <CL/cl_ext.h>

#include <map>

struct SLayerContext
{
    typedef std::map<cl_event, cl_command_type> CEventMap;
    CEventMap EventMap;
};

SLayerContext& getLayerContext(void);

extern const struct _cl_icd_dispatch* g_pNextDispatch;

///////////////////////////////////////////////////////////////////////////////
// Override Functions

cl_int clBuildProgram_fallback(
    cl_program program,
    cl_uint num_devices,
    const cl_device_id* device_list,
    const char* options,
    void (CL_CALLBACK* pfn_notify)(cl_program program, void* user_data),
    void* user_data,
    cl_int errorCode);

cl_program clCreateProgramWithSource_override(
    cl_context context,
    cl_uint count,
    const char** strings,
    const size_t* lengths,
    cl_int* errcode_ret);

cl_int clGetDeviceInfo_fallback(
    cl_device_id device,
    cl_device_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret,
    cl_int errorCode);

cl_int clGetKernelSubGroupInfo_fallback(
    cl_kernel kernel,
    cl_device_id device,
    cl_kernel_sub_group_info param_name,
    size_t input_value_size,
    const void* input_value,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret,
    cl_int errorCode);