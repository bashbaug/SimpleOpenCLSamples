/*
// Copyright (c) 2022-2025 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <CL/cl_layer.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include "layer_util.hpp"

#include "emulate.h"

static constexpr cl_version version_cl_exp_host_task =
    CL_MAKE_VERSION(0, 1, 0);
#define CL_EXP_HOST_TASK_EXTENSION_NAME \
    "cl_exp_host_task"

SLayerContext& getLayerContext(void)
{
    static SLayerContext c;
    return c;
}

struct SHostTaskInfo
{
    void(CL_CALLBACK* user_func)(void*);
    void*   user_data;

    cl_event    signal_event;
};

static void CL_CALLBACK HostCallbackCaller(cl_event event, cl_int status, void* user_data)
{
    auto info = (SHostTaskInfo*)user_data;

    if (info->user_func) {
        info->user_func(info->user_data);
    }

    g_pNextDispatch->clSetUserEventStatus(info->signal_event, CL_COMPLETE);
    g_pNextDispatch->clReleaseEvent(info->signal_event);

    delete info;
}

cl_int CL_API_CALL clEnqueueHostTaskEXP_EMU(
    cl_command_queue queue,
    void(CL_CALLBACK* user_func)(void*),
    void* user_data,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    cl_int errorCode = CL_SUCCESS;

    cl_context context = nullptr;
    errorCode = g_pNextDispatch->clGetCommandQueueInfo(
        queue,
        CL_QUEUE_CONTEXT,
        sizeof(context),
        &context,
        nullptr);
    if (errorCode != CL_SUCCESS) {
        return errorCode;
    }

    cl_event signal = g_pNextDispatch->clCreateUserEvent(
        context,
        &errorCode);
    if (errorCode != CL_SUCCESS) {
        return errorCode;
    }

    auto info = new SHostTaskInfo;
    info->user_func = user_func;
    info->user_data = user_data;
    info->signal_event = signal;

    cl_event startEvent = nullptr;
    errorCode = g_pNextDispatch->clEnqueueBarrierWithWaitList(
        queue,
        num_events_in_wait_list,
        event_wait_list,
        &startEvent);

    if (errorCode == CL_SUCCESS) {
        errorCode = g_pNextDispatch->clSetEventCallback(
            startEvent,
            CL_COMPLETE,
            HostCallbackCaller,
            info);
    }

    if (errorCode == CL_SUCCESS) {
        errorCode = g_pNextDispatch->clEnqueueBarrierWithWaitList(
            queue,
            1,
            &signal,
            event);
    }

    if (event) {
        getLayerContext().EventMap[event[0]] = startEvent;
    } else {
        g_pNextDispatch->clReleaseEvent(startEvent);
    }

    // TODO: Clean up properly on errors!

    return errorCode;
}

bool clGetDeviceInfo_override(
    cl_device_id device,
    cl_device_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret,
    cl_int* errcode_ret)
{
    switch(param_name) {
    case CL_DEVICE_EXTENSIONS:
        {
            size_t  size = 0;
            g_pNextDispatch->clGetDeviceInfo(
                device,
                CL_DEVICE_EXTENSIONS,
                0,
                nullptr,
                &size );

            std::vector<char> deviceExtensions(size);
            g_pNextDispatch->clGetDeviceInfo(
                device,
                CL_DEVICE_EXTENSIONS,
                size,
                deviceExtensions.data(),
                nullptr );

            size = 0;
            g_pNextDispatch->clGetDeviceInfo(
                device,
                CL_DEVICE_VERSION,
                0,
                nullptr,
                &size );

            std::vector<char> deviceVersion(size);
            g_pNextDispatch->clGetDeviceInfo(
                device,
                CL_DEVICE_VERSION,
                size,
                deviceVersion.data(),
                nullptr );

            if( checkStringForExtension(
                    deviceExtensions.data(),
                    CL_EXP_HOST_TASK_EXTENSION_NAME ) == false )
            {
                std::string newExtensions;
                newExtensions += CL_EXP_HOST_TASK_EXTENSION_NAME;

                std::string oldExtensions(deviceExtensions.data());

                // If the old extension string ends with a space ensure the
                // new extension string does too.
                if( oldExtensions.back() == ' ' )
                {
                    newExtensions += ' ';
                }
                else
                {
                    oldExtensions += ' ';
                }

                oldExtensions += newExtensions;

                auto ptr = (char*)param_value;
                cl_int errorCode = writeStringToMemory(
                    param_value_size,
                    oldExtensions.c_str(),
                    param_value_size_ret,
                    ptr );

                if( errcode_ret )
                {
                    errcode_ret[0] = errorCode;
                }
                return true;
            }
        }
        break;
    case CL_DEVICE_EXTENSIONS_WITH_VERSION:
        {
            size_t  size = 0;
            g_pNextDispatch->clGetDeviceInfo(
                device,
                CL_DEVICE_EXTENSIONS_WITH_VERSION,
                0,
                nullptr,
                &size );

            size_t  numExtensions = size / sizeof(cl_name_version);
            std::vector<cl_name_version>    extensions(numExtensions);
            g_pNextDispatch->clGetDeviceInfo(
                device,
                CL_DEVICE_EXTENSIONS_WITH_VERSION,
                size,
                extensions.data(),
                nullptr );

            size = 0;
            g_pNextDispatch->clGetDeviceInfo(
                device,
                CL_DEVICE_VERSION,
                0,
                nullptr,
                &size );

            std::vector<char> deviceVersion(size);
            g_pNextDispatch->clGetDeviceInfo(
                device,
                CL_DEVICE_VERSION,
                size,
                deviceVersion.data(),
                nullptr );

            bool found = false;
            for( const auto& extension : extensions )
            {
                if( strcmp(extension.name, CL_EXP_HOST_TASK_EXTENSION_NAME) == 0 )
                {
                    found = true;
                    break;
                }
            }

            if( found == false )
            {
                extensions.emplace_back();
                cl_name_version& extension = extensions.back();

                memset(extension.name, 0, CL_NAME_VERSION_MAX_NAME_SIZE);
                strcpy(extension.name, CL_EXP_HOST_TASK_EXTENSION_NAME);

                extension.version = version_cl_exp_host_task;

                auto ptr = (cl_name_version*)param_value;
                cl_int errorCode = writeVectorToMemory(
                    param_value_size,
                    extensions,
                    param_value_size_ret,
                    ptr );

                if( errcode_ret )
                {
                    errcode_ret[0] = errorCode;
                }
                return true;
            }
        }
        break;
    default: break;
    }

    return false;
}

bool clGetEventInfo_override(
    cl_event event,
    cl_event_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret,
    cl_int* errcode_ret)
{
    switch(param_name) {
    case CL_EVENT_COMMAND_TYPE:
        {
            auto& context = getLayerContext();
            auto it = context.EventMap.find(event);
            if (it != context.EventMap.end()) {
                // !!! TODO: Need a new command type for host tasks?
                cl_command_type type = CL_COMMAND_NATIVE_KERNEL;
                auto ptr = (cl_command_type*)param_value;
                cl_int errorCode = writeParamToMemory(
                    param_value_size,
                    type,
                    param_value_size_ret,
                    ptr );
                if( errcode_ret )
                {
                    errcode_ret[0] = errorCode;
                }
                return true;
            }
        }
        break;
    default: break;
    }

    return false;
}

bool clGetEventProfilingInfo_override(
    cl_event event,
    cl_profiling_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret,
    cl_int* errcode_ret)
{
    switch(param_name) {
    case CL_PROFILING_COMMAND_QUEUED:
    case CL_PROFILING_COMMAND_SUBMIT:
    case CL_PROFILING_COMMAND_START:
        {
            auto& context = getLayerContext();
            auto it = context.EventMap.find(event);
            if (it != context.EventMap.end()) {
                cl_int errorCode = g_pNextDispatch->clGetEventProfilingInfo(
                    it->second,
                    param_name,
                    param_value_size,
                    param_value,
                    param_value_size_ret);
                if( errcode_ret )
                {
                    errcode_ret[0] = errorCode;
                }
                return true;
            }
        }
        break;
    default: break;
    }

    return false;
}

bool clGetPlatformInfo_override(
    cl_platform_id platform,
    cl_platform_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret,
    cl_int* errcode_ret)
{
    switch(param_name) {
    case CL_PLATFORM_EXTENSIONS:
        {
            size_t  size = 0;
            g_pNextDispatch->clGetPlatformInfo(
                platform,
                CL_PLATFORM_EXTENSIONS,
                0,
                nullptr,
                &size );

            std::vector<char> platformExtensions(size);
            g_pNextDispatch->clGetPlatformInfo(
                platform,
                CL_PLATFORM_EXTENSIONS,
                size,
                platformExtensions.data(),
                nullptr );

            size = 0;
            g_pNextDispatch->clGetPlatformInfo(
                platform,
                CL_PLATFORM_VERSION,
                0,
                nullptr,
                &size );

            std::vector<char> platformVersion(size);
            g_pNextDispatch->clGetPlatformInfo(
                platform,
                CL_PLATFORM_VERSION,
                size,
                platformVersion.data(),
                nullptr );

            if( checkStringForExtension(
                    platformExtensions.data(),
                    CL_EXP_HOST_TASK_EXTENSION_NAME ) == false )
            {
                std::string newExtensions;
                newExtensions += CL_EXP_HOST_TASK_EXTENSION_NAME;

                std::string oldExtensions(platformExtensions.data());

                // If the old extension string ends with a space ensure the
                // new extension string does too.
                if( oldExtensions.back() == ' ' )
                {
                    newExtensions += ' ';
                }
                else
                {
                    oldExtensions += ' ';
                }

                oldExtensions += newExtensions;

                auto ptr = (char*)param_value;
                cl_int errorCode = writeStringToMemory(
                    param_value_size,
                    oldExtensions.c_str(),
                    param_value_size_ret,
                    ptr );

                if( errcode_ret )
                {
                    errcode_ret[0] = errorCode;
                }
                return true;
            }
        }
        break;
    case CL_PLATFORM_EXTENSIONS_WITH_VERSION:
        {
            size_t  size = 0;
            g_pNextDispatch->clGetPlatformInfo(
                platform,
                CL_PLATFORM_EXTENSIONS_WITH_VERSION,
                0,
                nullptr,
                &size );

            size_t  numExtensions = size / sizeof(cl_name_version);
            std::vector<cl_name_version>    extensions(numExtensions);
            g_pNextDispatch->clGetPlatformInfo(
                platform,
                CL_PLATFORM_EXTENSIONS_WITH_VERSION,
                size,
                extensions.data(),
                nullptr );

            size = 0;
            g_pNextDispatch->clGetPlatformInfo(
                platform,
                CL_PLATFORM_VERSION,
                0,
                nullptr,
                &size );

            std::vector<char> platformVersion(size);
            g_pNextDispatch->clGetPlatformInfo(
                platform,
                CL_PLATFORM_VERSION,
                size,
                platformVersion.data(),
                nullptr );

            bool found = false;
            for( const auto& extension : extensions )
            {
                if( strcmp(extension.name, CL_EXP_HOST_TASK_EXTENSION_NAME) == 0 )
                {
                    found = true;
                    break;
                }
            }

            if( found == false )
            {
                extensions.emplace_back();
                cl_name_version& extension = extensions.back();

                memset(extension.name, 0, CL_NAME_VERSION_MAX_NAME_SIZE);
                strcpy(extension.name, CL_EXP_HOST_TASK_EXTENSION_NAME);

                extension.version = version_cl_exp_host_task;

                auto ptr = (cl_name_version*)param_value;
                cl_int errorCode = writeVectorToMemory(
                    param_value_size,
                    extensions,
                    param_value_size_ret,
                    ptr );

                if( errcode_ret )
                {
                    errcode_ret[0] = errorCode;
                }
                return true;
            }
        }
        break;
    default: break;
    }
    return false;
}
