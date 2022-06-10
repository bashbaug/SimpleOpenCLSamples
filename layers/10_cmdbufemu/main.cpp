/*
// Copyright (c) 2022 Ben Ashbaugh
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

#define CHECK_RETURN_EXTENSION_FUNCTION( _funcname )                        \
    if (strcmp(func_name, #_funcname) == 0) {                               \
        return (void*)_funcname##_EMU;                                      \
    }

static void * CL_API_CALL
clGetExtensionFunctionAddressForPlatform_layer(
    cl_platform_id platform,
    const char *   func_name)
{
    void* ret = g_pNextDispatch->clGetExtensionFunctionAddressForPlatform(
        platform,
        func_name);

    if (ret) {
        return ret;
    }

    CHECK_RETURN_EXTENSION_FUNCTION( clCreateCommandBufferKHR );
    CHECK_RETURN_EXTENSION_FUNCTION( clFinalizeCommandBufferKHR );
    CHECK_RETURN_EXTENSION_FUNCTION( clRetainCommandBufferKHR );
    CHECK_RETURN_EXTENSION_FUNCTION( clReleaseCommandBufferKHR );
    CHECK_RETURN_EXTENSION_FUNCTION( clEnqueueCommandBufferKHR );
    CHECK_RETURN_EXTENSION_FUNCTION( clCommandBarrierWithWaitListKHR );
    CHECK_RETURN_EXTENSION_FUNCTION( clCommandCopyBufferKHR );
    CHECK_RETURN_EXTENSION_FUNCTION( clCommandCopyBufferRectKHR );
    CHECK_RETURN_EXTENSION_FUNCTION( clCommandCopyBufferToImageKHR );
    CHECK_RETURN_EXTENSION_FUNCTION( clCommandCopyImageKHR );
    CHECK_RETURN_EXTENSION_FUNCTION( clCommandCopyImageToBufferKHR );
    CHECK_RETURN_EXTENSION_FUNCTION( clCommandFillBufferKHR );
    CHECK_RETURN_EXTENSION_FUNCTION( clCommandFillImageKHR );
    CHECK_RETURN_EXTENSION_FUNCTION( clCommandNDRangeKernelKHR );
    CHECK_RETURN_EXTENSION_FUNCTION( clGetCommandBufferInfoKHR );

    return nullptr;
}

static cl_int CL_API_CALL
clGetPlatformInfo_layer(
    cl_platform_id   platform,
    cl_platform_info param_name,
    size_t           param_value_size,
    void *           param_value,
    size_t *         param_value_size_ret)
{
    return g_pNextDispatch->clGetPlatformInfo(
        platform,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
}

static struct _cl_icd_dispatch dispatch;
static void _init_dispatch()
{
    dispatch.clGetDeviceInfo = &clGetDeviceInfo_layer;
    dispatch.clGetExtensionFunctionAddressForPlatform = &clGetExtensionFunctionAddressForPlatform_layer;
    dispatch.clGetPlatformInfo = &clGetPlatformInfo_layer;
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

