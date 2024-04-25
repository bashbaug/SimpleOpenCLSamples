/*
// Copyright (c) 2022 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

// This example layer was heavily inspired by the OpenCL-Layers-Tutorial:
// https://github.com/Kerilk/OpenCL-Layers-Tutorial

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

#include <stdio.h>

// Utility functions to properly check and return values from queries.
#include "layer_util.hpp"

// This is the dispatch table for this layer.
// It will contain functions that this layer hooks.
static struct _cl_icd_dispatch dispatch;

// This is the next dispatch table.
// The layer should use this dispatch table to make OpenCL calls.
static const struct _cl_icd_dispatch* pNextDispatch;

// This is the only function that this layer will hook.
// It simply prints the function arguments and return values.
static cl_int CL_API_CALL clGetPlatformIDs_layer(
    cl_uint num_entries,
    cl_platform_id* platforms,
    cl_uint* num_platforms)
{
    fprintf(stderr, "Example Layer: clGetPlatformIDs(num_entries: %d, platforms: %p, num_platforms: %p)\n",
        num_entries, platforms, num_platforms);

    cl_int res = pNextDispatch->clGetPlatformIDs(num_entries, platforms, num_platforms);

    fprintf(stderr, "Example Layer: clGetPlatformIDs result: %d, num_platforms: %d\n",
        res, num_platforms ? num_platforms[0] : 0);
    return res;
}

// This is a utility function to setup the dispatch table for this layer.
static void _init_dispatch()
{
    dispatch.clGetPlatformIDs = &clGetPlatformIDs_layer;
}

// This is boilerplate code that will be similar for all layers.

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
                "Example Layer",
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

    pNextDispatch = target_dispatch;

    *layer_dispatch_ret = &dispatch;
    *num_entries_out = dispatchTableSize;

    return CL_SUCCESS;
}

