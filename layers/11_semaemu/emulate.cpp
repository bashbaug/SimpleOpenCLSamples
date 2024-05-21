/*
// Copyright (c) 2022-2024 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <CL/cl_layer.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <string>
#include <vector>

#include "layer_util.hpp"

#include "emulate.h"

#ifndef CL_KHR_SEMAPHORE_EXTENSION_NAME
#define CL_KHR_SEMAPHORE_EXTENSION_NAME "cl_khr_semaphore"
#endif
#ifndef CL_SEMAPHORE_DEVICE_HANDLE_LIST_KHR
#define CL_SEMAPHORE_DEVICE_HANDLE_LIST_KHR 0x2053
#define CL_SEMAPHORE_DEVICE_HANDLE_LIST_END_KHR 0
#endif

static constexpr cl_version version_cl_khr_semaphore =
    CL_MAKE_VERSION(0, 9, 1);

SLayerContext& getLayerContext(void)
{
    static SLayerContext c;
    return c;
}

typedef struct _cl_semaphore_khr
{
    static _cl_semaphore_khr* create(
        cl_context context,
        const cl_semaphore_properties_khr* properties,
        cl_int* errcode_ret)
    {
        cl_semaphore_khr semaphore = NULL;
        cl_int errorCode = CL_SUCCESS;

        ptrdiff_t numProperties = 0;
        cl_semaphore_type_khr type = ~0;

        std::vector<cl_device_id> devices;

        if( properties )
        {
            const cl_semaphore_properties_khr* check = properties;
            bool found_CL_SEMAPHORE_TYPE_KHR = false;
            bool found_CL_SEMAPHORE_DEVICE_HANDLE_LIST_KHR = false;
            while( errorCode == CL_SUCCESS && check[0] != 0 )
            {
                cl_int  property = (cl_int)check[0];
                switch( property )
                {
                case CL_SEMAPHORE_TYPE_KHR:
                    if( found_CL_SEMAPHORE_TYPE_KHR )
                    {
                        errorCode = CL_INVALID_VALUE;
                    }
                    else
                    {
                        found_CL_SEMAPHORE_TYPE_KHR = true;
                        type = ((const cl_semaphore_type_khr*)(check + 1))[0];
                        check += 2;
                    }
                    break;
                case CL_SEMAPHORE_DEVICE_HANDLE_LIST_KHR:
                    if( found_CL_SEMAPHORE_DEVICE_HANDLE_LIST_KHR )
                    {
                        errorCode = CL_INVALID_VALUE;
                    }
                    else
                    {
                        found_CL_SEMAPHORE_DEVICE_HANDLE_LIST_KHR = true;
                        check++;
                        while(*check != CL_SEMAPHORE_DEVICE_HANDLE_LIST_END_KHR)
                        {
                            devices.push_back(((cl_device_id*)check)[0]);
                            check++;
                        }
                    }
                    break;
                default:
                    errorCode = CL_INVALID_VALUE;
                    break;
                }
            }
            numProperties = check - properties + 1;

            // validate device handles.
            if (!devices.empty()) {
              // for now - if CL_SEMAPHORE_DEVICE_HANDLE_LIST_KHR is specified
              // as part of sema_props, but it does not identify exactly one
              // valid device
              if (devices.size() > 1) {
                errorCode = CL_INVALID_DEVICE;
              } else {
                // if a device identified by CL_SEMAPHORE_DEVICE_HANDLE_LIST_KHR
                // is not one of the devices within context
                std::vector<cl_semaphore_type_khr> types;
                for (auto device : devices) {
                  if (device == nullptr ||
                      !isDeviceWithinContext(context, device)) {
                    errorCode = CL_INVALID_DEVICE;
                    break;
                  }
                }
              }
            }
        }
        switch( type )
        {
        case CL_SEMAPHORE_TYPE_BINARY_KHR:
            break;
        default:
            errorCode = CL_INVALID_VALUE;
        }
        if( errcode_ret )
        {
            errcode_ret[0] = errorCode;
        }
        if( errorCode == CL_SUCCESS )
        {
            semaphore = new _cl_semaphore_khr(context, type);
            semaphore->Properties.reserve(numProperties);
            semaphore->Properties.insert(
                semaphore->Properties.begin(),
                properties,
                properties + numProperties );

            semaphore->Devices=devices;
        }
        return semaphore;
    }

    static bool isValid( cl_semaphore_khr semaphore )
    {
        return semaphore && semaphore->Magic == cMagic;
    }

    const cl_uint Magic;
    const cl_context Context;
    const cl_semaphore_type_khr Type;
    std::vector<cl_semaphore_properties_khr> Properties;
    std::vector<cl_device_id>   Devices;

    std::atomic<cl_uint> RefCount;
    cl_event Event;

private:
    static constexpr cl_uint cMagic = 0x53454d41;   // "SEMA"

    _cl_semaphore_khr(cl_context context, cl_semaphore_type_khr type) :
        Magic(cMagic),
        Context(context),
        Type(type),
        RefCount(1),
        Event(NULL) {}
} cli_semaphore;

cl_semaphore_khr CL_API_CALL clCreateSemaphoreWithPropertiesKHR_EMU(
    cl_context context,
    const cl_semaphore_properties_khr *sema_props,
    cl_int *errcode_ret)
{
    return cli_semaphore::create(
        context,
        sema_props,
        errcode_ret);
}

cl_int CL_API_CALL clEnqueueWaitSemaphoresKHR_EMU(
    cl_command_queue command_queue,
    cl_uint num_semaphores,
    const cl_semaphore_khr *semaphores,
    const cl_semaphore_payload_khr *semaphore_payloads,
    cl_uint num_events_in_wait_list,
    const cl_event *event_wait_list,
    cl_event *event)
{
    if( num_semaphores == 0 )
    {
        return CL_INVALID_VALUE;
    }

    std::vector<cl_event> combinedWaitList;
    combinedWaitList.insert(
        combinedWaitList.end(),
        event_wait_list,
        event_wait_list + num_events_in_wait_list);

    for( cl_uint i = 0; i < num_semaphores; i++ )
    {
        if( !cli_semaphore::isValid(semaphores[i]) )
        {
            return CL_INVALID_SEMAPHORE_KHR;
        }
        if( semaphores[i]->Event == NULL )
        {
            // This is a semaphore that is not in a pending signal
            // or signaled state.  What should happen here?
            return CL_INVALID_OPERATION;
        }
        combinedWaitList.push_back(
            semaphores[i]->Event);
    }

    cl_int retVal = g_pNextDispatch->clEnqueueMarkerWithWaitList(
        command_queue,
        (cl_uint)combinedWaitList.size(),
        combinedWaitList.data(),
        event );

    for( cl_uint i = 0; i < num_semaphores; i++ )
    {
        g_pNextDispatch->clReleaseEvent(
            semaphores[i]->Event);
        semaphores[i]->Event = NULL;
    }

    if( event )
    {
        getLayerContext().EventMap[event[0]] = CL_COMMAND_SEMAPHORE_WAIT_KHR;
    }

    return retVal;
}

cl_int CL_API_CALL clEnqueueSignalSemaphoresKHR_EMU(
    cl_command_queue command_queue,
    cl_uint num_semaphores,
    const cl_semaphore_khr *semaphores,
    const cl_semaphore_payload_khr *sema_payload_list,
    cl_uint num_events_in_wait_list,
    const cl_event *event_wait_list,
    cl_event *event)
{
    if( num_semaphores == 0 )
    {
        return CL_INVALID_VALUE;
    }

    for( cl_uint i = 0; i < num_semaphores; i++ )
    {
        if( !cli_semaphore::isValid(semaphores[i]) )
        {
            return CL_INVALID_SEMAPHORE_KHR;
        }
        if( semaphores[i]->Event != NULL )
        {
            // This is a semaphore that is in a pending signal or signaled
            // state.  What should happen here?
            return CL_INVALID_OPERATION;
        }
    }

    cl_event    local_event = NULL;
    if( event == NULL )
    {
        event = &local_event;
    }

    cl_int retVal = g_pNextDispatch->clEnqueueMarkerWithWaitList(
        command_queue,
        num_events_in_wait_list,
        event_wait_list,
        event );

    for( cl_uint i = 0; i < num_semaphores; i++ )
    {
        semaphores[i]->Event = *event;
        g_pNextDispatch->clRetainEvent(
            semaphores[i]->Event );
    }

    if( local_event != NULL )
    {
        g_pNextDispatch->clReleaseEvent(
            local_event );
        local_event = NULL;
    }
    else
    {
        getLayerContext().EventMap[event[0]] = CL_COMMAND_SEMAPHORE_SIGNAL_KHR;
    }

    return retVal;
}

cl_int CL_API_CALL clGetSemaphoreInfoKHR_EMU(
    cl_semaphore_khr semaphore,
    cl_semaphore_info_khr param_name,
    size_t param_value_size,
    void *param_value,
    size_t *param_value_size_ret)
{
    if( !cli_semaphore::isValid(semaphore) )
    {
        return CL_INVALID_SEMAPHORE_KHR;
    }

    switch( param_name )
    {
    case CL_SEMAPHORE_CONTEXT_KHR:
        {
            auto ptr = (cl_context*)param_value;
            return writeParamToMemory(
                param_value_size,
                semaphore->Context,
                param_value_size_ret,
                ptr );
        }
    case CL_SEMAPHORE_REFERENCE_COUNT_KHR:
        {
            auto ptr = (cl_uint*)param_value;
            return writeParamToMemory(
                param_value_size,
                semaphore->RefCount.load(std::memory_order_relaxed),
                param_value_size_ret,
                ptr );
        }
    case CL_SEMAPHORE_PROPERTIES_KHR:
        {
            auto ptr = (cl_semaphore_properties_khr*)param_value;
            return writeVectorToMemory(
                param_value_size,
                semaphore->Properties,
                param_value_size_ret,
                ptr );
        }
        break;
    case CL_SEMAPHORE_TYPE_KHR:
        {
            auto ptr = (cl_semaphore_type_khr*)param_value;
            return writeParamToMemory(
                param_value_size,
                semaphore->Type,
                param_value_size_ret,
                ptr );
        }
        break;
    case CL_SEMAPHORE_PAYLOAD_KHR:
        {
            // For binary semaphores, the payload should be zero if the
            // semaphore is in the unsignaled state and one if it is in
            // the signaled state.
            cl_semaphore_payload_khr payload = 0;
            if( semaphore->Event != NULL )
            {
                cl_int  eventStatus = 0;
                g_pNextDispatch->clGetEventInfo(
                    semaphore->Event,
                    CL_EVENT_COMMAND_EXECUTION_STATUS,
                    sizeof( eventStatus ),
                    &eventStatus,
                    NULL );
                if( eventStatus == CL_COMPLETE )
                {
                    payload = 1;
                }
            }

            auto ptr = (cl_semaphore_payload_khr*)param_value;
            return writeParamToMemory(
                param_value_size,
                payload,
                param_value_size_ret,
                ptr );
        }
        break;
    case CL_SEMAPHORE_DEVICE_HANDLE_LIST_KHR:
        {
            auto ptr = (cl_device_id*)param_value;
            return writeVectorToMemory(
                param_value_size,
                semaphore->Devices,
                param_value_size_ret,
                ptr );
        }
        break;
    default:
        return CL_INVALID_VALUE;
    }

    return CL_INVALID_OPERATION;
}

cl_int CL_API_CALL clRetainSemaphoreKHR_EMU(
    cl_semaphore_khr semaphore)
{
    if( !cli_semaphore::isValid(semaphore) )
    {
        return CL_INVALID_SEMAPHORE_KHR;
    }

    semaphore->RefCount.fetch_add(1, std::memory_order_relaxed);
    return CL_SUCCESS;
}

cl_int CL_API_CALL clReleaseSemaphoreKHR_EMU(
    cl_semaphore_khr semaphore)
{
    if( !cli_semaphore::isValid(semaphore) )
    {
        return CL_INVALID_SEMAPHORE_KHR;
    }

    semaphore->RefCount.fetch_sub(1, std::memory_order_relaxed);
    if( semaphore->RefCount.load(std::memory_order_relaxed) == 0 )
    {
        delete semaphore;
    }
    return CL_SUCCESS;
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

            if( checkStringForExtension(
                    deviceExtensions.data(),
                    CL_KHR_SEMAPHORE_EXTENSION_NAME ) == false )
            {
                std::string newExtensions;
                newExtensions += CL_KHR_SEMAPHORE_EXTENSION_NAME;

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

            bool found = false;
            for( const auto& extension : extensions )
            {
                if( strcmp(extension.name, CL_KHR_SEMAPHORE_EXTENSION_NAME) == 0 )
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
                strcpy(extension.name, CL_KHR_SEMAPHORE_EXTENSION_NAME);

                extension.version = CL_MAKE_VERSION(0, 9, 0);

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
    case CL_DEVICE_SEMAPHORE_TYPES_KHR:
        {
            // If we decide to emulate multiple semaphore types we will need
            // to return an array, but for now we can return just the binary
            // semaphore type.
            auto ptr = (cl_semaphore_type_khr*)param_value;
            cl_semaphore_type_khr type = CL_SEMAPHORE_TYPE_BINARY_KHR;
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
                cl_command_type type = it->second;
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

            if( checkStringForExtension(
                    platformExtensions.data(),
                    CL_KHR_SEMAPHORE_EXTENSION_NAME ) == false )
            {
                std::string newExtensions;
                newExtensions += CL_KHR_SEMAPHORE_EXTENSION_NAME;

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

            bool found = false;
            for( const auto& extension : extensions )
            {
                if( strcmp(extension.name, CL_KHR_SEMAPHORE_EXTENSION_NAME) == 0 )
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
                strcpy(extension.name, CL_KHR_SEMAPHORE_EXTENSION_NAME);

                extension.version = CL_MAKE_VERSION(0, 9, 0);

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
    case CL_PLATFORM_SEMAPHORE_TYPES_KHR:
        {
            // If we decide to emulate multiple semaphore types we will need
            // to return an array, but for now we can return just the binary
            // semaphore type.
            auto ptr = (cl_semaphore_type_khr*)param_value;
            cl_semaphore_type_khr type = CL_SEMAPHORE_TYPE_BINARY_KHR;
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
        break;
    default: break;
    }
    return false;
}
