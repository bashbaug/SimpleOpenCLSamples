/*
// Copyright (c) 2022-2024 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <CL/cl.h>
#include <CL/cl_layer.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <string>
#include <vector>

#include "layer_util.hpp"

#include "emulate.h"
#include "subgroups.cl.h"

const cl_uint g_NVDeviceVendorID = 0x10DE;

static inline bool isNV(cl_device_id device)
{
    cl_uint deviceVendorID = 0;
    g_pNextDispatch->clGetDeviceInfo(
        device,
        CL_DEVICE_VENDOR_ID,
        sizeof(deviceVendorID),
        &deviceVendorID,
        nullptr);
    return deviceVendorID == g_NVDeviceVendorID;
}

static inline bool isNV(cl_program program)
{
    cl_uint numDevices = 0;
    g_pNextDispatch->clGetProgramInfo(
        program,
        CL_PROGRAM_NUM_DEVICES,
        sizeof(numDevices),
        &numDevices,
        nullptr);

    std::vector<cl_device_id> devices(numDevices);
    g_pNextDispatch->clGetProgramInfo(
        program,
        CL_PROGRAM_DEVICES,
        numDevices * sizeof(cl_device_id),
        devices.data(),
        nullptr);

    return std::all_of(
        devices.begin(),
        devices.end(),
        [](cl_device_id device) { return isNV(device); });
}

static inline bool isNV(cl_context context)
{
    cl_uint numDevices = 0;
    g_pNextDispatch->clGetContextInfo(
        context,
        CL_CONTEXT_NUM_DEVICES,
        sizeof(numDevices),
        &numDevices,
        nullptr);

    std::vector<cl_device_id> devices(numDevices);
    g_pNextDispatch->clGetContextInfo(
        context,
        CL_CONTEXT_DEVICES,
        numDevices * sizeof(cl_device_id),
        devices.data(),
        nullptr);

    return std::all_of(
        devices.begin(),
        devices.end(),
        [](cl_device_id device) { return isNV(device); });
}

cl_int clBuildProgram_override(
    cl_program program,
    cl_uint num_devices,
    const cl_device_id* device_list,
    const char* options,
    void (CL_CALLBACK* pfn_notify)(cl_program program, void* user_data),
    void* user_data,
    cl_int errorCode)
{
    if (!isNV(program)) {
        return errorCode;
    }

    // TODO: add the -D define for emulated semaphores and retry build.

    return errorCode;
}

cl_program clCreateProgramWithSource_override(
    cl_context context,
    cl_uint count,
    const char** strings,
    const size_t* lengths,
    cl_int* errcode_ret)
{
    if (!isNV(context)) {
        return g_pNextDispatch->clCreateProgramWithSource(
            context,
            count,
            strings,
            lengths,
            errcode_ret);
    }

    if (count == 0 || strings == nullptr) {
        if (errcode_ret != nullptr) {
            *errcode_ret = CL_INVALID_VALUE;
        }
        return nullptr;
    }

    std::vector<const char*> newStrings;
    newStrings.reserve(count + 1);
    newStrings.insert(newStrings.end(), g_NVSubGroupString);
    newStrings.insert(newStrings.end(), strings, strings + count);

    std::vector<size_t> newLengths;
    if (lengths != nullptr) {
        newLengths.reserve(count + 1);
        newLengths.insert(newLengths.end(), 0);  // g_NVSubGroupString is nul-terminated
        newLengths.insert(newLengths.end(), lengths, lengths + count);
    }

    return g_pNextDispatch->clCreateProgramWithSource(
        context,
        count + 1,
        newStrings.data(),
        newLengths.size() ? newLengths.data() : nullptr,
        errcode_ret);
}


cl_int clGetDeviceInfo_override(
    cl_device_id device,
    cl_device_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret,
    cl_int errorCode)
{
    cl_uint deviceVendorID = 0;
    g_pNextDispatch->clGetDeviceInfo(
        device,
        CL_DEVICE_VENDOR_ID,
        sizeof(deviceVendorID),
        &deviceVendorID,
        nullptr);
    if (deviceVendorID != g_NVDeviceVendorID) {
        return errorCode;
    }

    switch(param_name) {
    case CL_DEVICE_MAX_NUM_SUB_GROUPS:
        {
            size_t  maxWorkGroupSize = 0;
            g_pNextDispatch->clGetDeviceInfo(
                device,
                CL_DEVICE_MAX_WORK_GROUP_SIZE,
                sizeof(maxWorkGroupSize),
                &maxWorkGroupSize,
                nullptr);

            size_t  warpSize = 0;
            g_pNextDispatch->clGetDeviceInfo(
                device,
                CL_DEVICE_WARP_SIZE_NV,
                sizeof(warpSize),
                &warpSize,
                nullptr);

            cl_uint maxNumSubGroups =
                static_cast<cl_uint>(maxWorkGroupSize / warpSize);
            return writeParamToMemory(
                param_value_size,
                maxNumSubGroups,
                param_value_size_ret,
                (cl_uint*)param_value);
        }
        break;
    default:
        break;
    }

    return errorCode;
}

cl_int clGetKernelSubGroupInfo_override(
    cl_kernel kernel,
    cl_device_id device,
    cl_kernel_sub_group_info param_name,
    size_t input_value_size,
    const void* input_value,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret,
    cl_int errorCode)
{
    cl_uint deviceVendorID = 0;
    g_pNextDispatch->clGetDeviceInfo(
        device,
        CL_DEVICE_VENDOR_ID,
        sizeof(deviceVendorID),
        &deviceVendorID,
        nullptr);
    if (deviceVendorID != g_NVDeviceVendorID) {
        return errorCode;
    }

    switch(param_name) {
    case CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE:
        if (input_value == nullptr || input_value_size % sizeof(size_t) != 0) {
            return CL_INVALID_VALUE;
        }
        {
            size_t  warpSize = 0;
            g_pNextDispatch->clGetDeviceInfo(
                device,
                CL_DEVICE_WARP_SIZE_NV,
                sizeof(warpSize),
                &warpSize,
                nullptr);

            return writeParamToMemory(
                param_value_size,
                warpSize,
                param_value_size_ret,
                (size_t*)param_value);
        }
        break;
    case CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE:
        if (input_value == nullptr || input_value_size % sizeof(size_t) != 0) {
            return CL_INVALID_VALUE;
        }
        {
            const size_t dim = input_value_size / sizeof(size_t);
            size_t  workGroupSize = 1;
            for (size_t i = 0; i < dim; ++i) {
                workGroupSize *= ((size_t*)input_value)[i];
            }

            size_t  warpSize = 0;
            g_pNextDispatch->clGetDeviceInfo(
                device,
                CL_DEVICE_WARP_SIZE_NV,
                sizeof(warpSize),
                &warpSize,
                nullptr);

            size_t numSubGroups = (workGroupSize + warpSize - 1) / warpSize;
            return writeParamToMemory(
                param_value_size,
                numSubGroups,
                param_value_size_ret,
                (size_t*)param_value);
        }
        break;
    case CL_KERNEL_LOCAL_SIZE_FOR_SUB_GROUP_COUNT:
        if (input_value == nullptr || input_value_size != sizeof(size_t)) {
            return CL_INVALID_VALUE;
        }
        {
        }
        break;
    case CL_KERNEL_MAX_NUM_SUB_GROUPS:
        {
            size_t  maxWorkGroupSize = 0;
            g_pNextDispatch->clGetKernelWorkGroupInfo(
                kernel,
                device,
                CL_KERNEL_WORK_GROUP_SIZE,
                sizeof(maxWorkGroupSize),
                &maxWorkGroupSize,
                nullptr);

            size_t  warpSize = 0;
            g_pNextDispatch->clGetDeviceInfo(
                device,
                CL_DEVICE_WARP_SIZE_NV,
                sizeof(warpSize),
                &warpSize,
                nullptr);

            size_t maxNumSubGroups = maxWorkGroupSize / warpSize;
            return writeParamToMemory(
                param_value_size,
                maxNumSubGroups,
                param_value_size_ret,
                (size_t*)param_value);
        }
        break;
    case CL_KERNEL_COMPILE_NUM_SUB_GROUPS:
        // Not sure how to implement this one...
        break;
    default:
        break;
    }
    
    return errorCode;
}