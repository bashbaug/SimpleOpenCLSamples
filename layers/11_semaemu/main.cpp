/*
// Copyright (c) 2022-2025 Ben Ashbaugh
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

#ifndef CL_KHR_SEMAPHORE_EXTENSION_NAME
#define CL_KHR_SEMAPHORE_EXTENSION_NAME "cl_khr_semaphore"
#endif

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
clGetEventInfo_layer(
    cl_event         event,
    cl_event_info    param_name,
    size_t           param_value_size,
    void *           param_value,
    size_t *         param_value_size_ret)
{
    cl_int  errorCode = CL_SUCCESS;

    if (clGetEventInfo_override(
            event,
            param_name,
            param_value_size,
            param_value,
            param_value_size_ret,
            &errorCode) == false) {
        return g_pNextDispatch->clGetEventInfo(
            event,
            param_name,
            param_value_size,
            param_value,
            param_value_size_ret);
    }

    return errorCode;
}
static cl_int CL_API_CALL
clGetSemaphoreHandleForTypeKHR_EMU(
    cl_semaphore_khr sema_object,
    cl_device_id device,
    cl_external_semaphore_handle_type_khr handle_type,
    size_t handle_size,
    void* handle_ptr,
    size_t* handle_size_ret)
{
    return CL_INVALID_OPERATION;
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
    // For now, prefer the emulated functions, even if the extension is
    // supported natively.  Eventually this should become smarter.
    CHECK_RETURN_EXTENSION_FUNCTION( clCreateSemaphoreWithPropertiesKHR );
    CHECK_RETURN_EXTENSION_FUNCTION( clEnqueueWaitSemaphoresKHR );
    CHECK_RETURN_EXTENSION_FUNCTION( clEnqueueSignalSemaphoresKHR );
    CHECK_RETURN_EXTENSION_FUNCTION( clGetSemaphoreInfoKHR );
    CHECK_RETURN_EXTENSION_FUNCTION( clRetainSemaphoreKHR );
    CHECK_RETURN_EXTENSION_FUNCTION( clReleaseSemaphoreKHR );

    // This is currently required to run the semaphore conformance tests,
    // although it is part of cl_khr_external_semaphore, and not part of
    // cl_khr_semaphore.
    CHECK_RETURN_EXTENSION_FUNCTION( clGetSemaphoreHandleForTypeKHR );

    return g_pNextDispatch->clGetExtensionFunctionAddressForPlatform(
        platform,
        func_name);
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
clReleaseEvent_layer(
    cl_event         event)
{
    cl_uint refCount = 0;
    g_pNextDispatch->clGetEventInfo(
        event,
        CL_EVENT_REFERENCE_COUNT,
        sizeof(refCount),
        &refCount,
        nullptr);
    if (refCount == 1) {
        auto& context = getLayerContext();
        auto it = context.EventMap.find(event);
        if (it != context.EventMap.end()) {
            context.EventMap.erase(it);
        }
    }

    return g_pNextDispatch->clReleaseEvent(event);
}

static struct _cl_icd_dispatch dispatch;
static void _init_dispatch()
{
    dispatch.clGetDeviceInfo = clGetDeviceInfo_layer;
    dispatch.clGetEventInfo = clGetEventInfo_layer;
    dispatch.clGetExtensionFunctionAddressForPlatform = clGetExtensionFunctionAddressForPlatform_layer;
    dispatch.clGetPlatformInfo = clGetPlatformInfo_layer;
    dispatch.clReleaseEvent = clReleaseEvent_layer;
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
                "Emulation Layer for " CL_KHR_SEMAPHORE_EXTENSION_NAME,
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
