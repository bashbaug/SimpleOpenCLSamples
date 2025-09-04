/*
// Copyright (c) 2026 Ben Ashbaugh
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
clGetDeviceInfo_layer(
    cl_device_id    device,
    cl_device_info  param_name,
    size_t          param_value_size,
    void *          param_value,
    size_t *        param_value_size_ret)
{
    cl_int  errorCode = CL_SUCCESS;

    if (clGetDeviceInfo_override(
        device,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret,
        &errorCode) == false) {
        return g_pNextDispatch->clGetDeviceInfo(
            device,
            param_name,
            param_value_size,
            param_value,
            param_value_size_ret);
    }

    return errorCode;
}

static cl_int CL_API_CALL
clGetPlatformInfo_layer(
    cl_platform_id   platform,
    cl_platform_info param_name,
    size_t           param_value_size,
    void *           param_value,
    size_t *         param_value_size_ret)
{
    cl_int  errorCode = CL_SUCCESS;

    if (clGetPlatformInfo_override(
        platform,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret,
        &errorCode) == false) {
        return g_pNextDispatch->clGetPlatformInfo(
            platform,
            param_name,
            param_value_size,
            param_value,
            param_value_size_ret);
    }

    return errorCode;
}

static cl_int CL_API_CALL
clBuildProgram_layer(
    cl_program           program,
    cl_uint              num_devices,
    const cl_device_id * device_list,
    const char *         options,
    void (CL_CALLBACK *  pfn_notify)(cl_program program, void * user_data),
    void *               user_data)
{
    std::string newOptions = replaceBuildOptions(program, options);
    options = options && !newOptions.empty() ? newOptions.c_str() : options;
    return g_pNextDispatch->clBuildProgram(
        program,
        num_devices,
        device_list,
        options,
        pfn_notify,
        user_data);
}

static cl_int CL_API_CALL
clCompileProgram_layer(
    cl_program           program,
    cl_uint              num_devices,
    const cl_device_id * device_list,
    const char *         options,
    cl_uint              num_input_headers,
    const cl_program *   input_headers,
    const char **        header_include_names,
    void (CL_CALLBACK *  pfn_notify)(cl_program program, void * user_data),
    void *               user_data)
{
    std::string newOptions = replaceBuildOptions(program, options);
    options = options && !newOptions.empty() ? newOptions.c_str() : options;
    return g_pNextDispatch->clCompileProgram(
        program,
        num_devices,
        device_list,
        options,
        num_input_headers,
        input_headers,
        header_include_names,
        pfn_notify,
        user_data);
}

static struct _cl_icd_dispatch dispatch;
static void _init_dispatch()
{
    dispatch.clGetDeviceInfo = clGetDeviceInfo_layer;
    dispatch.clGetPlatformInfo = clGetPlatformInfo_layer;
    dispatch.clBuildProgram = clBuildProgram_layer;
    dispatch.clCompileProgram = clCompileProgram_layer;
#if defined(CL_VERSION_3_1)
    dispatch.clGetKernelSuggestedLocalWorkSize = clGetKernelSuggestedLocalWorkSize_EMU;
#endif
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
                "Emulation Layer for OpenCL 3.1",
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

CL_API_ENTRY cl_int CL_API_CALL clInitLayerWithProperties(
    cl_uint num_entries,
    const struct _cl_icd_dispatch* target_dispatch,
    cl_uint* num_entries_out,
    const struct _cl_icd_dispatch** layer_dispatch_ret,
    const cl_layer_properties* properties)
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

CL_API_ENTRY cl_int CL_API_CALL clInitLayer(
    cl_uint num_entries,
    const struct _cl_icd_dispatch* target_dispatch,
    cl_uint* num_entries_out,
    const struct _cl_icd_dispatch** layer_dispatch_ret)
{
    return clInitLayerWithProperties(
        num_entries,
        target_dispatch,
        num_entries_out,
        layer_dispatch_ret,
        nullptr);
}
