/*
// Copyright (c) 2023 Ben Ashbaugh
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

#define CHECK_RETURN_EXTENSION_FUNCTION( _funcname )                        \
    if (strcmp(func_name, #_funcname) == 0) {                               \
        return (void*)_funcname##_EMU;                                      \
    }

static void * CL_API_CALL
clGetExtensionFunctionAddressForPlatform_override(
    cl_platform_id platform,
    const char *   func_name)
{
    CHECK_RETURN_EXTENSION_FUNCTION( clSVMAllocWithPropertiesEXP );
    CHECK_RETURN_EXTENSION_FUNCTION( clSVMFreeWithPropertiesEXP );
    CHECK_RETURN_EXTENSION_FUNCTION( clGetSVMInfoEXP );
    CHECK_RETURN_EXTENSION_FUNCTION( clEnqueueSVMMemAdviseEXP );

    return g_pNextDispatch->clGetExtensionFunctionAddressForPlatform(
        platform,
        func_name);
}

static struct _cl_icd_dispatch dispatch;
static void _init_dispatch()
{
    dispatch.clGetDeviceInfo = clGetDeviceInfo_override;
    dispatch.clGetExtensionFunctionAddressForPlatform = clGetExtensionFunctionAddressForPlatform_override;
    dispatch.clGetPlatformInfo = clGetPlatformInfo_override;
    dispatch.clSetKernelArgSVMPointer = clSetKernelArgSVMPointer_override;
    dispatch.clSetKernelExecInfo = clSetKernelExecInfo_override;
    dispatch.clSVMFree = clSVMFree_override;
    dispatch.clEnqueueSVMMemcpy = clEnqueueSVMMemcpy_override;
    dispatch.clEnqueueSVMMemFill = clEnqueueSVMMemFill_override;
    dispatch.clEnqueueSVMMigrateMem = clEnqueueSVMMigrateMem_override;
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
                "Experimentation Layer for SVM and USM Unification",
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
