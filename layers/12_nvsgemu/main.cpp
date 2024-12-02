/*
// Copyright (c) 2022-2024 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#if defined _WIN32 || defined __CYGWIN__
#ifdef __GNUC__
#define CL_API_ENTRY __attribute__((dllexport))
#else
#define CL_API_ENTRY __declspec(dllexport)
#endif
#else
#if __GNUC__ >= 4
#define CL_API_ENTRY __attribute__((visibility("default")))
#else
#define CL_API_ENTRY
#endif
#endif

#include <CL/cl_layer.h>

#include <cstring>
#include <cstdio>

#include "layer_util.hpp"

#include "emulate.h"

const struct _cl_icd_dispatch* g_pNextDispatch = NULL;

static cl_int CL_API_CALL
clBuildProgram_layer(
    cl_program program,
    cl_uint num_devices,
    const cl_device_id* device_list,
    const char* options,
    void (CL_CALLBACK* pfn_notify)(cl_program program, void* user_data),
    void* user_data)
{
    cl_int errorCode = g_pNextDispatch->clBuildProgram(
        program,
        num_devices,
        device_list,
        options,
        pfn_notify,
        user_data);
    if (errorCode != CL_SUCCESS) {
        errorCode = clBuildProgram_fallback(
            program,
            num_devices,
            device_list,
            options,
            pfn_notify,
            user_data,
            errorCode);
    }

    return errorCode;
}

static cl_program CL_API_CALL
clCreateProgramWithSource_layer(
    cl_context context,
    cl_uint count,
    const char** strings,
    const size_t* lengths,
    cl_int* errcode_ret)
{
    return clCreateProgramWithSource_override(
        context,
        count,
        strings,
        lengths,
        errcode_ret);
}

static cl_int CL_API_CALL
clGetDeviceInfo_layer(
    cl_device_id    device,
    cl_device_info  param_name,
    size_t          param_value_size,
    void *          param_value,
    size_t *        param_value_size_ret)
{
    cl_int errorCode = g_pNextDispatch->clGetDeviceInfo(
        device,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
    if (errorCode == CL_SUCCESS &&
        param_name == CL_DEVICE_MAX_NUM_SUB_GROUPS &&
        param_value != nullptr &&
        param_value_size == sizeof(cl_uint) &&
        ((cl_uint*)param_value)[0] == 0) {
        errorCode = clGetDeviceInfo_fallback(
            device,
            param_name,
            param_value_size,
            param_value,
            param_value_size_ret,
            errorCode);
    }

    return errorCode;
}

static cl_int CL_API_CALL
clGetKernelSubGroupInfo_layer(
    cl_kernel kernel,
    cl_device_id device,
    cl_kernel_sub_group_info param_name,
    size_t input_value_size,
    const void* input_value,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    cl_int  errorCode = g_pNextDispatch->clGetKernelSubGroupInfo(
        kernel,
        device,
        param_name,
        input_value_size,
        input_value,
        param_value_size,
        param_value,
        param_value_size_ret);
    if (errorCode != CL_SUCCESS) {
        errorCode = clGetKernelSubGroupInfo_fallback(
            kernel,
            device,
            param_name,
            input_value_size,
            input_value,
            param_value_size,
            param_value,
            param_value_size_ret,
            errorCode);
    }

    return errorCode;
}

static struct _cl_icd_dispatch dispatch;
static void _init_dispatch()
{
    dispatch.clBuildProgram = clBuildProgram_layer;
    dispatch.clCreateProgramWithSource = clCreateProgramWithSource_layer;
    dispatch.clGetDeviceInfo = clGetDeviceInfo_layer;
    dispatch.clGetKernelSubGroupInfo = clGetKernelSubGroupInfo_layer;
}

CL_API_ENTRY cl_int CL_API_CALL clGetLayerInfo(
    cl_layer_info  param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    switch (param_name) {
    case CL_LAYER_API_VERSION:
        {
            auto ptr = (cl_layer_api_version*)param_value;
            auto value = cl_layer_api_version{CL_LAYER_API_VERSION_100};
            return writeParamToMemory(
                param_value_size,
                value,
                param_value_size_ret,
                ptr);
        }
        break;
#if defined(CL_LAYER_NAME)
    case CL_LAYER_NAME:
        {
            auto ptr = (char*)param_value;
            return writeStringToMemory(
                param_value_size,
                "Emulation Layer for NVIDIA Sub-Groups",
                param_value_size_ret,
                ptr);
        }
        break;
#endif
    default:
        return CL_INVALID_VALUE;
    }
    return CL_SUCCESS;
}

CL_API_ENTRY cl_int CL_API_CALL clInitLayer(
    cl_uint num_entries,
    const struct _cl_icd_dispatch* target_dispatch,
    cl_uint* num_entries_out,
    const struct _cl_icd_dispatch** layer_dispatch_ret)
{
    const size_t dispatchTableSize =
        sizeof(dispatch) / sizeof(dispatch.clGetPlatformIDs);

    if (target_dispatch == nullptr || 
        num_entries_out == nullptr ||
        layer_dispatch_ret == nullptr) {
        return CL_INVALID_VALUE;
    }

    if (num_entries < dispatchTableSize) {
        return CL_INVALID_VALUE;
    }

    _init_dispatch();

    g_pNextDispatch = target_dispatch;

    *layer_dispatch_ret = &dispatch;
    *num_entries_out = dispatchTableSize;

    return CL_SUCCESS;
}
