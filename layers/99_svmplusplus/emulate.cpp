/*
// Copyright (c) 2023 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <CL/cl_layer.h>

#include <memory>
#include <string>
#include <vector>

#include <cassert>

#include "layer_util.hpp"

#include "emulate.h"

static constexpr cl_version version_cl_exp_new_svm_extension =
    CL_MAKE_VERSION(0, 1, 0);

struct SLayerContext
{
};

SLayerContext& getLayerContext(void)
{
    static SLayerContext c;
    return c;
}

static cl_context getContext(
    cl_command_queue queue)
{
    cl_context context = nullptr;
    g_pNextDispatch->clGetCommandQueueInfo(
        queue,
        CL_QUEUE_CONTEXT,
        sizeof(context),
        &context,
        nullptr);
    return context;
}

static cl_context getContext(
    cl_kernel kernel)
{
    cl_context context = nullptr;
    g_pNextDispatch->clGetKernelInfo(
        kernel,
        CL_KERNEL_CONTEXT,
        sizeof(context),
        &context,
        nullptr);
    return context;
}

static cl_svm_mem_type_exp getSVMMemType(
    cl_context context,
    const void* ptr)
{
    cl_svm_mem_type_exp type = CL_SVM_MEM_TYPE_UNKNOWN_EXP;
    clGetMemAllocInfoINTEL(
        context,
        ptr,
        CL_MEM_ALLOC_TYPE_INTEL,
        sizeof(type),
        &type,
        nullptr);
    return type;
}

static bool isUSMPtr(
    cl_context context,
    const void* ptr)
{
    cl_svm_mem_type_exp type = getSVMMemType(context, ptr);
    return type == CL_SVM_MEM_TYPE_HOST_EXP ||
        type == CL_SVM_MEM_TYPE_DEVICE_EXP ||
        type == CL_SVM_MEM_TYPE_SHARED_EXP;
}

static cl_device_id getAssociatedDeviceFromProperties(
    const cl_svm_mem_properties_exp* props)
{
    if (props) {
        while(props[0] != 0) {
            cl_int property = (cl_int)props[0];
            switch(property) {
            case CL_SVM_MEM_ASSOCIATED_DEVICE_HANDLE_EXP: {
                    auto pdev = (const cl_device_id*)(props + 1);
                    return pdev[0];
                }
                break;
            default:
                // Not supporting other properties currently.
                assert(0 && "unknown SVM mem property");
                break;
            }
        }
    }
    return nullptr;
}

void* CL_API_CALL clSVMAllocWithPropertiesEXP_EMU(
    cl_context context,
    const cl_svm_mem_properties_exp* properties,
    cl_svm_mem_flags flags,
    size_t size,
    cl_uint alignment,
    cl_int* errcode_ret)
{
    cl_device_id device = getAssociatedDeviceFromProperties(properties);

    if (flags & CL_MEM_SVM_DEVICE_EXP) {
        return clDeviceMemAllocINTEL(
            context,
            device,
            nullptr,
            size,
            alignment,
            errcode_ret);
    }
    if (flags & CL_MEM_SVM_HOST_EXP) {
        return clHostMemAllocINTEL(
            context,
            nullptr,
            size,
            alignment,
            errcode_ret);
    }
    if (flags & CL_MEM_SVM_SHARED_EXP) {
        return clSharedMemAllocINTEL(
            context,
            device,
            nullptr,
            size,
            alignment,
            errcode_ret);
    }

    return g_pNextDispatch->clSVMAlloc(
        context,
        flags,
        size,
        alignment);
}

cl_int CL_API_CALL clSVMFreeWithPropertiesEXP_EMU(
    cl_context context,
    const cl_svm_free_properties_exp* properties,
    cl_svm_free_flags_exp flags,
    void* ptr)
{
    if (isUSMPtr(context, ptr)) {
        if (flags & CL_SVM_FREE_BLOCKING_EXP) {
            return clMemBlockingFreeINTEL(
                context,
                ptr);
        }
        return clMemFreeINTEL(
            context,
            ptr);
    }

    assert(!(flags & CL_SVM_FREE_BLOCKING_EXP) && "blocking SVM free is currently unsupported");
    g_pNextDispatch->clSVMFree(
        context,
        ptr);
    return CL_SUCCESS;
}

cl_int CL_API_CALL clGetSVMMemInfoEXP_EMU(
    cl_context context,
    const void* ptr,
    cl_svm_mem_info_exp param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    if (isUSMPtr(context, ptr)) {
        return clGetMemAllocInfoINTEL(
            context,
            ptr,
            param_name,
            param_value_size,
            param_value,
            param_value_size_ret);
    }

    assert(0 && "querying SVM pointers is currently unsupported");
    return CL_INVALID_OPERATION;
}

cl_int CL_API_CALL clGetDeviceInfo_override(
    cl_device_id device,
    cl_device_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    switch(param_name) {
    case CL_DEVICE_SVM_CAPABILITIES:
        {
            cl_device_unified_shared_memory_capabilities_intel deviceCaps = 0;
            g_pNextDispatch->clGetDeviceInfo(
                device,
                CL_DEVICE_DEVICE_MEM_CAPABILITIES_INTEL,
                sizeof(deviceCaps),
                &deviceCaps,
                nullptr );

            cl_device_unified_shared_memory_capabilities_intel hostCaps = 0;
            g_pNextDispatch->clGetDeviceInfo(
                device,
                CL_DEVICE_HOST_MEM_CAPABILITIES_INTEL,
                sizeof(hostCaps),
                &hostCaps,
                nullptr );

            // We can just check the single device shared capabilities:
            cl_device_unified_shared_memory_capabilities_intel sharedCaps = 0;
            g_pNextDispatch->clGetDeviceInfo(
                device,
                CL_DEVICE_SINGLE_DEVICE_SHARED_MEM_CAPABILITIES_INTEL,
                sizeof(sharedCaps),
                &sharedCaps,
                nullptr );

            cl_device_svm_capabilities svmCaps = 0;
            g_pNextDispatch->clGetDeviceInfo(
                device,
                CL_DEVICE_SVM_CAPABILITIES,
                sizeof(svmCaps),
                &svmCaps,
                nullptr );

            svmCaps |= (deviceCaps != 0) ? CL_DEVICE_SVM_DEVICE_ALLOC_EXP : 0;
            svmCaps |= (hostCaps   != 0) ? CL_DEVICE_SVM_HOST_ALLOC_EXP   : 0;
            svmCaps |= (sharedCaps != 0) ? CL_DEVICE_SVM_SHARED_ALLOC_EXP : 0;

            auto ptr = (cl_device_svm_capabilities*)param_value;
            return writeParamToMemory(
                param_value_size,
                svmCaps,
                param_value_size_ret,
                ptr );
        }
        break;
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
                    CL_EXP_UNIFIED_SVM_EXTENSION_NAME ) == false )
            {
                std::string newExtensions;

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
                return errorCode;
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
                if( strcmp(extension.name, CL_EXP_UNIFIED_SVM_EXTENSION_NAME) == 0 )
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
                strcpy(extension.name, CL_EXP_UNIFIED_SVM_EXTENSION_NAME);

                extension.version = version_cl_exp_new_svm_extension;

                auto ptr = (cl_name_version*)param_value;
                cl_int errorCode = writeVectorToMemory(
                    param_value_size,
                    extensions,
                    param_value_size_ret,
                    ptr );
                return errorCode;
            }
        }
        break;
    // USM aliases - pass through.
    case CL_DEVICE_HOST_MEM_CAPABILITIES_EXP:
    case CL_DEVICE_DEVICE_MEM_CAPABILITIES_EXP:
    case CL_DEVICE_SINGLE_DEVICE_SHARED_MEM_CAPABILITIES_EXP:
    case CL_DEVICE_CROSS_DEVICE_SHARED_MEM_CAPABILITIES_EXP:
    case CL_DEVICE_SHARED_SYSTEM_MEM_CAPABILITIES_EXP:
    default: break;
    }

    return g_pNextDispatch->clGetDeviceInfo(
        device,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
}

cl_int CL_API_CALL clGetPlatformInfo_override(
    cl_platform_id platform,
    cl_platform_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
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
                    CL_EXP_UNIFIED_SVM_EXTENSION_NAME ) == false )
            {
                std::string newExtensions;
                newExtensions += CL_EXP_UNIFIED_SVM_EXTENSION_NAME;

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
                return errorCode;
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
                if( strcmp(extension.name, CL_EXP_UNIFIED_SVM_EXTENSION_NAME) == 0 )
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
                strcpy(extension.name, CL_EXP_UNIFIED_SVM_EXTENSION_NAME);

                extension.version = version_cl_exp_new_svm_extension;

                auto ptr = (cl_name_version*)param_value;
                cl_int errorCode = writeVectorToMemory(
                    param_value_size,
                    extensions,
                    param_value_size_ret,
                    ptr );
                return errorCode;
            }
        }
        break;
    default: break;
    }

    return g_pNextDispatch->clGetPlatformInfo(
        platform,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
}

cl_int CL_API_CALL clSetKernelArgSVMPointer_override(
    cl_kernel kernel,
    cl_uint arg_index,
    const void* arg_value)
{
    cl_context context = getContext(kernel);

    if (isUSMPtr(context, arg_value)) {
        return clSetKernelArgMemPointerINTEL(
            kernel,
            arg_index,
            arg_value);
    }

    return g_pNextDispatch->clSetKernelArgSVMPointer(
        kernel,
        arg_index,
        arg_value);
}

void CL_API_CALL clSVMFree_override(
    cl_context context,
    void* ptr)
{
    if (isUSMPtr(context, ptr)) {
        clMemFreeINTEL(context, ptr);
    }

    g_pNextDispatch->clSVMFree(context, ptr);
}

cl_int CL_API_CALL clEnqueueSVMMemAdviseEXP_EMU(
    cl_command_queue command_queue,
    const void* ptr,
    size_t size,
    cl_svm_mem_advice_exp advice,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    // for now, just emit a marker
    return g_pNextDispatch->clEnqueueMarkerWithWaitList(
        command_queue,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

cl_int CL_API_CALL clEnqueueSVMMemcpy_override(
    cl_command_queue command_queue,
    cl_bool blocking_copy,
    void* dst_ptr,
    const void* src_ptr,
    size_t size,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    cl_context context = getContext(command_queue);

    if (isUSMPtr(context, dst_ptr) || isUSMPtr(context, src_ptr)) {
        return clEnqueueMemcpyINTEL(
            command_queue,
            blocking_copy,
            dst_ptr,
            src_ptr,
            size,
            num_events_in_wait_list,
            event_wait_list,
            event);
    }

    return g_pNextDispatch->clEnqueueSVMMemcpy(
        command_queue,
        blocking_copy,
        dst_ptr,
        src_ptr,
        size,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

cl_int CL_API_CALL clEnqueueSVMMemFill_override(
    cl_command_queue command_queue,
    void* svm_ptr,
    const void* pattern,
    size_t pattern_size,
    size_t size,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    cl_context context = getContext(command_queue);

    if (isUSMPtr(context, svm_ptr)) {
        return clEnqueueMemFillINTEL(
            command_queue,
            svm_ptr,
            pattern,
            pattern_size,
            size,
            num_events_in_wait_list,
            event_wait_list,
            event);
    }

    return g_pNextDispatch->clEnqueueSVMMemFill(
        command_queue,
        svm_ptr,
        pattern,
        pattern_size,
        size,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

cl_int CL_API_CALL clEnqueueSVMMigrateMem_override(
    cl_command_queue command_queue,
    cl_uint num_svm_pointers,
    const void** svm_pointers,
    const size_t* sizes,
    cl_mem_migration_flags flags,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    // for now, just emit a marker
    return g_pNextDispatch->clEnqueueMarkerWithWaitList(
        command_queue,
        num_events_in_wait_list,
        event_wait_list,
        event);
}
