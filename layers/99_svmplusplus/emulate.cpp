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
    CL_MAKE_VERSION(0, 1, 4);

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

static bool isUSMPtr(
    cl_context context,
    const void* ptr)
{
    cl_unified_shared_memory_type_intel type = CL_MEM_TYPE_UNKNOWN_INTEL;
    clGetMemAllocInfoINTEL(
        context,
        ptr,
        CL_MEM_ALLOC_TYPE_INTEL,
        sizeof(type),
        &type,
        nullptr);
    return type != CL_MEM_TYPE_UNKNOWN_INTEL;
}

static bool checkSVMCaps(
    cl_svm_capabilities_exp caps,
    cl_svm_capabilities_exp enabled,
    cl_svm_capabilities_exp disabled = 0)
{
    if ((caps & enabled) != enabled) {
        return false;
    }
    if ((~caps & disabled) != disabled) {
        return false;
    }
    return true;
}

static cl_device_id getAssociatedDeviceFromProperties(
    const cl_svm_alloc_properties_exp* props)
{
    if (props) {
        while(props[0] != 0) {
            cl_int property = (cl_int)props[0];
            switch(property) {
            case CL_SVM_ALLOC_ASSOCIATED_DEVICE_HANDLE_EXP: {
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

static cl_svm_capabilities_exp getDeviceUSMCaps(cl_device_id device)
{
    cl_svm_capabilities_exp ret = 0;

    cl_device_unified_shared_memory_capabilities_intel usmCaps = 0;
    g_pNextDispatch->clGetDeviceInfo(
        device,
        CL_DEVICE_DEVICE_MEM_CAPABILITIES_INTEL,
        sizeof(usmCaps),
        &usmCaps,
        nullptr);

    if (usmCaps != 0) {
        ret = CL_SVM_TYPE_MACRO_DEVICE_EXP;

        assert(usmCaps & CL_UNIFIED_SHARED_MEMORY_ACCESS_INTEL);
        assert(usmCaps & CL_UNIFIED_SHARED_MEMORY_ATOMIC_ACCESS_INTEL);

        if (usmCaps & CL_UNIFIED_SHARED_MEMORY_CONCURRENT_ACCESS_INTEL) {
            ret |= CL_SVM_CAPABILITY_CONCURRENT_ACCESS_EXP;
        }
        if (usmCaps & CL_UNIFIED_SHARED_MEMORY_CONCURRENT_ATOMIC_ACCESS_INTEL) {
            ret |= CL_SVM_CAPABILITY_CONCURRENT_ATOMIC_ACCESS_EXP;
        }
    }

    return ret;
}

static cl_svm_capabilities_exp getHostUSMCaps(cl_device_id device)
{
    cl_svm_capabilities_exp ret = 0;

    cl_device_unified_shared_memory_capabilities_intel usmCaps = 0;
    g_pNextDispatch->clGetDeviceInfo(
        device,
        CL_DEVICE_HOST_MEM_CAPABILITIES_INTEL,
        sizeof(usmCaps),
        &usmCaps,
        nullptr);

    if (usmCaps != 0) {
        ret = CL_SVM_TYPE_MACRO_HOST_EXP;

        assert(usmCaps & CL_UNIFIED_SHARED_MEMORY_ACCESS_INTEL);

        if (usmCaps & CL_UNIFIED_SHARED_MEMORY_ATOMIC_ACCESS_INTEL) {
            ret |= CL_SVM_CAPABILITY_DEVICE_ATOMIC_ACCESS_EXP;
        }
        if (usmCaps & CL_UNIFIED_SHARED_MEMORY_CONCURRENT_ACCESS_INTEL) {
            ret |= CL_SVM_CAPABILITY_CONCURRENT_ACCESS_EXP;
        }
        if (usmCaps & CL_UNIFIED_SHARED_MEMORY_CONCURRENT_ATOMIC_ACCESS_INTEL) {
            ret |= CL_SVM_CAPABILITY_CONCURRENT_ATOMIC_ACCESS_EXP;
        }
    }

    return ret;
}

static cl_svm_capabilities_exp getSingleDeviceSharedUSMCaps(cl_device_id device)
{
    cl_svm_capabilities_exp ret = 0;

    cl_device_unified_shared_memory_capabilities_intel usmCaps = 0;
    g_pNextDispatch->clGetDeviceInfo(
        device,
        CL_DEVICE_SINGLE_DEVICE_SHARED_MEM_CAPABILITIES_INTEL,
        sizeof(usmCaps),
        &usmCaps,
        nullptr);

    if (usmCaps != 0) {
        ret = CL_SVM_TYPE_MACRO_SINGLE_DEVICE_SHARED_EXP;

        assert(usmCaps & CL_UNIFIED_SHARED_MEMORY_ACCESS_INTEL);

        if (usmCaps & CL_UNIFIED_SHARED_MEMORY_ATOMIC_ACCESS_INTEL) {
            ret |= CL_SVM_CAPABILITY_DEVICE_ATOMIC_ACCESS_EXP;
        }
        if (usmCaps & CL_UNIFIED_SHARED_MEMORY_CONCURRENT_ACCESS_INTEL) {
            ret |= CL_SVM_CAPABILITY_CONCURRENT_ACCESS_EXP;
        }
        if (usmCaps & CL_UNIFIED_SHARED_MEMORY_CONCURRENT_ATOMIC_ACCESS_INTEL) {
            ret |= CL_SVM_CAPABILITY_CONCURRENT_ATOMIC_ACCESS_EXP;
        }
    }

    return ret;
}

static cl_svm_capabilities_exp getCoarseGrainSVMCaps(cl_device_id device)
{
    cl_svm_capabilities_exp ret = 0;

    cl_device_svm_capabilities svmCaps = 0;
    g_pNextDispatch->clGetDeviceInfo(
        device,
        CL_DEVICE_SVM_CAPABILITIES,
        sizeof(svmCaps),
        &svmCaps,
        nullptr);

    if (svmCaps & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER) {
        ret = CL_SVM_TYPE_MACRO_COARSE_GRAIN_BUFFER_EXP;
    }

    return ret;
}

static cl_svm_capabilities_exp getFineGrainSVMCaps(cl_device_id device)
{
    cl_svm_capabilities_exp ret = 0;

    cl_device_svm_capabilities svmCaps = 0;
    g_pNextDispatch->clGetDeviceInfo(
        device,
        CL_DEVICE_SVM_CAPABILITIES,
        sizeof(svmCaps),
        &svmCaps,
        nullptr);

    if (svmCaps & CL_DEVICE_SVM_FINE_GRAIN_BUFFER) {
        ret = CL_SVM_TYPE_MACRO_FINE_GRAIN_BUFFER_EXP;

        if (svmCaps & CL_DEVICE_SVM_ATOMICS) {
            ret |= CL_SVM_CAPABILITY_CONCURRENT_ATOMIC_ACCESS_EXP;
        }
    }

    return ret;
}

static cl_svm_capabilities_exp getSystemSVMCaps(cl_device_id device)
{
    cl_svm_capabilities_exp ret = 0;

    cl_device_svm_capabilities svmCaps = 0;
    g_pNextDispatch->clGetDeviceInfo(
        device,
        CL_DEVICE_SVM_CAPABILITIES,
        sizeof(svmCaps),
        &svmCaps,
        nullptr);

    cl_device_unified_shared_memory_capabilities_intel usmCaps = 0;
    g_pNextDispatch->clGetDeviceInfo(
        device,
        CL_DEVICE_SHARED_SYSTEM_MEM_CAPABILITIES_INTEL,
        sizeof(usmCaps),
        &usmCaps,
        nullptr);

    if (svmCaps & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM || usmCaps != 0) {
        ret = CL_SVM_TYPE_MACRO_SYSTEM_EXP;
    }

    return ret;
}

static std::vector<cl_device_svm_type_capabilities_exp> getSVMTypeCaps(cl_device_id device)
{
    std::vector<cl_device_svm_type_capabilities_exp> types;

    // USM Types

    cl_svm_capabilities_exp caps = getDeviceUSMCaps(device);
    if (caps != 0) {
        types.emplace_back();
        cl_device_svm_type_capabilities_exp& type = types.back();
        type.capabilities = caps & CL_SVM_TYPE_MACRO_DEVICE_EXP;
        type.optional_capabilities = caps & (
            CL_SVM_CAPABILITY_CONCURRENT_ACCESS_EXP |
            CL_SVM_CAPABILITY_CONCURRENT_ATOMIC_ACCESS_EXP);
    }

    caps = getHostUSMCaps(device);
    if (caps != 0) {
        types.emplace_back();
        cl_device_svm_type_capabilities_exp& type = types.back();
        type.capabilities = caps & CL_SVM_TYPE_MACRO_HOST_EXP;
        type.optional_capabilities = caps & (
            CL_SVM_CAPABILITY_DEVICE_ATOMIC_ACCESS_EXP |
            CL_SVM_CAPABILITY_CONCURRENT_ACCESS_EXP |
            CL_SVM_CAPABILITY_CONCURRENT_ATOMIC_ACCESS_EXP);
    }

    caps = getSingleDeviceSharedUSMCaps(device);
    if (caps != 0) {
        types.emplace_back();
        cl_device_svm_type_capabilities_exp& type = types.back();
        type.capabilities = caps & CL_SVM_TYPE_MACRO_SINGLE_DEVICE_SHARED_EXP;
        type.optional_capabilities = caps & (
            CL_SVM_CAPABILITY_DEVICE_ATOMIC_ACCESS_EXP |
            CL_SVM_CAPABILITY_CONCURRENT_ACCESS_EXP |
            CL_SVM_CAPABILITY_CONCURRENT_ATOMIC_ACCESS_EXP);
    }

    // SVM Types

    caps = getCoarseGrainSVMCaps(device);
    if (caps != 0) {
        types.emplace_back();
        cl_device_svm_type_capabilities_exp& type = types.back();
        type.capabilities = CL_SVM_TYPE_MACRO_COARSE_GRAIN_BUFFER_EXP;
        type.optional_capabilities = 0;
    }

    caps = getFineGrainSVMCaps(device);
    if (caps != 0) {
        types.emplace_back();
        cl_device_svm_type_capabilities_exp& type = types.back();
        type.capabilities = CL_SVM_TYPE_MACRO_FINE_GRAIN_BUFFER_EXP;
        type.optional_capabilities = caps & (
            CL_SVM_CAPABILITY_CONCURRENT_ATOMIC_ACCESS_EXP);
    }

    caps = getSystemSVMCaps(device);
    if (caps != 0) {
        types.emplace_back();
        cl_device_svm_type_capabilities_exp& type = types.back();
        type.capabilities = CL_SVM_TYPE_MACRO_SYSTEM_EXP;
        type.optional_capabilities = 0;
    }

    return types;
}

void* CL_API_CALL clSVMAllocWithPropertiesEXP_EMU(
    cl_context context,
    const cl_svm_alloc_properties_exp* properties,
    cl_svm_capabilities_exp capabilities,
    cl_mem_flags flags,
    size_t size,
    cl_uint alignment,
    cl_int* errcode_ret)
{
    cl_device_id device = getAssociatedDeviceFromProperties(properties);

    // TODO: Validate requested capabilities are supported!

    const bool isDeviceUSM = checkSVMCaps(
        capabilities,
        CL_SVM_CAPABILITY_DEVICE_OWNED_EXP);
    if (isDeviceUSM) {
        // note: currently ignores flags!
        return clDeviceMemAllocINTEL(
            context,
            device,
            nullptr,
            size,
            alignment,
            errcode_ret);
    }

    const bool isHostUSM = checkSVMCaps(
        capabilities,
        CL_SVM_CAPABILITY_HOST_OWNED_EXP);
    if (isHostUSM) {
        // note: currently ignores flags!
        return clHostMemAllocINTEL(
            context,
            nullptr,
            size,
            alignment,
            errcode_ret);
    }

    const bool isSingleDeviceSharedUSM = checkSVMCaps(
        capabilities,
        0,
        CL_SVM_CAPABILITY_DEVICE_OWNED_EXP |
        CL_SVM_CAPABILITY_DEVICE_UNASSOCIATED_EXP |
        CL_SVM_CAPABILITY_HOST_OWNED_EXP);
    if (isSingleDeviceSharedUSM) {
        // note: currently ignores flags!
        return clSharedMemAllocINTEL(
            context,
            device,
            nullptr,
            size,
            alignment,
            errcode_ret);
    }

    const bool isFineGrainBufferSVM = checkSVMCaps(
        capabilities,
        CL_SVM_CAPABILITY_DEVICE_UNASSOCIATED_EXP |
        CL_SVM_CAPABILITY_CONCURRENT_ACCESS_EXP,
        CL_SVM_CAPABILITY_DEVICE_OWNED_EXP |
        CL_SVM_CAPABILITY_HOST_OWNED_EXP);
    if (isFineGrainBufferSVM) {
        cl_svm_mem_flags svmFlags = flags | CL_MEM_SVM_FINE_GRAIN_BUFFER;
        if (capabilities & CL_SVM_CAPABILITY_CONCURRENT_ATOMIC_ACCESS_EXP) {
            svmFlags |= CL_MEM_SVM_ATOMICS;
        }
        void* ret = g_pNextDispatch->clSVMAlloc(
            context,
            svmFlags,
            size,
            alignment);
        if (errcode_ret) {
            errcode_ret[0] = ret ? CL_SUCCESS : CL_INVALID_VALUE;
        }
        return ret;
    }

    const bool isCoarseGrainBufferSVM = checkSVMCaps(
        capabilities,
        CL_SVM_CAPABILITY_DEVICE_UNASSOCIATED_EXP,
        CL_SVM_CAPABILITY_DEVICE_OWNED_EXP |
        CL_SVM_CAPABILITY_HOST_OWNED_EXP |
        CL_SVM_CAPABILITY_CONCURRENT_ACCESS_EXP);
    if (isCoarseGrainBufferSVM) {
        cl_svm_mem_flags svmFlags = flags;
        void* ret = g_pNextDispatch->clSVMAlloc(
            context,
            svmFlags,
            size,
            alignment);
        if (errcode_ret) {
            errcode_ret[0] = ret ? CL_SUCCESS : CL_INVALID_VALUE;
        }
        return ret;
    }

    if (errcode_ret) {
        errcode_ret[0] = CL_INVALID_VALUE;
    }
    return nullptr;
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

static cl_svm_capabilities_exp getSuggestedSVMCapabilitiesForDevice(
    cl_device_id device,
    cl_svm_capabilities_exp capabilities)
{
    auto supported = getSVMTypeCaps(device);
    for(const auto& type : supported) {
        auto supported_caps = type.capabilities | type.optional_capabilities;
        if ((supported_caps & capabilities) == capabilities) {
            return supported_caps;
        }
    }

    return 0;
}

cl_int CL_API_CALL clGetSuggestedSVMCapabilitiesEXP_EMU(
    cl_context context,
    cl_uint num_devices,
    const cl_device_id* devices,
    cl_svm_capabilities_exp required_capabilities,
    cl_svm_capabilities_exp* suggested_capabilities)
{
    if (suggested_capabilities == nullptr) {
        return CL_INVALID_VALUE;
    }
    if (num_devices > 0 && devices == nullptr ||
        num_devices == 0 && devices != nullptr) {
        return CL_INVALID_VALUE;
    }
    
    // not implemented
    return CL_INVALID_OPERATION;
}

cl_int CL_API_CALL clGetSVMInfoEXP_EMU(
    cl_context context,
    cl_device_id device,
    const void* ptr,
    cl_svm_info_exp param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    if (param_name == CL_SVM_INFO_CAPABILITIES_EXP) {
        cl_svm_capabilities_exp caps = 0;
        cl_unified_shared_memory_type_intel type = CL_MEM_TYPE_UNKNOWN_INTEL;
        clGetMemAllocInfoINTEL(
            context,
            ptr,
            CL_MEM_ALLOC_TYPE_INTEL,
            sizeof(type),
            &type,
            nullptr);
        cl_device_id associatedDevice = nullptr;
        clGetMemAllocInfoINTEL(
            context,
            ptr,
            CL_MEM_ALLOC_DEVICE_INTEL,
            sizeof(associatedDevice),
            &associatedDevice,
            nullptr);
        cl_device_id contextDevice = nullptr;
        clGetContextInfo(
            context, // note: assumes single-device context!
            CL_CONTEXT_DEVICES,
            sizeof(contextDevice),
            &contextDevice,
            nullptr);
        switch (type) {
        case CL_MEM_TYPE_DEVICE_INTEL:
            caps = getDeviceUSMCaps(associatedDevice);
            break;
        case CL_MEM_TYPE_HOST_INTEL:
            caps = getHostUSMCaps(contextDevice);
            break;
        case CL_MEM_TYPE_SHARED_INTEL:
            caps = getSingleDeviceSharedUSMCaps(associatedDevice);
            break;
        default:
            break;
        }
        auto ptr = (cl_svm_capabilities_exp*)param_value;
        return writeParamToMemory(
            param_value_size,
            caps,
            param_value_size_ret,
            ptr);
    }

    if (isUSMPtr(context, ptr)) {
        return clGetMemAllocInfoINTEL(
            context,
            ptr,
            param_name,
            param_value_size,
            param_value,
            param_value_size_ret);
    }

    // Note: do not currently support querying SVM pointers!
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
                newExtensions += CL_EXP_UNIFIED_SVM_EXTENSION_NAME;

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
    case CL_DEVICE_SVM_TYPE_CAPABILITIES_EXP:
        {
            auto svmTypeCaps = getSVMTypeCaps(device);
            auto ptr = (cl_device_svm_type_capabilities_exp*)param_value;
            return writeVectorToMemory(
                param_value_size,
                svmTypeCaps,
                param_value_size_ret,
                ptr);
        }
        break;
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

cl_int CL_API_CALL clSetKernelExecInfo_override(
    cl_kernel kernel,
    cl_kernel_exec_info param_name,
    size_t param_value_size,
    const void* param_value)
{
    switch (param_name) {
    case CL_KERNEL_EXEC_INFO_SVM_INDIRECT_ACCESS_EXP:
        {
            cl_int ret = CL_INVALID_OPERATION;
            cl_int check = CL_INVALID_OPERATION;
            check = g_pNextDispatch->clSetKernelExecInfo(
                kernel,
                CL_KERNEL_EXEC_INFO_INDIRECT_HOST_ACCESS_INTEL,
                param_value_size,
                param_value);
            ret = (check == CL_SUCCESS) ? CL_SUCCESS : ret;
            check = g_pNextDispatch->clSetKernelExecInfo(
                kernel,
                CL_KERNEL_EXEC_INFO_INDIRECT_DEVICE_ACCESS_INTEL,
                param_value_size,
                param_value);
            ret = (check == CL_SUCCESS) ? CL_SUCCESS : ret;
            check = g_pNextDispatch->clSetKernelExecInfo(
                kernel,
                CL_KERNEL_EXEC_INFO_INDIRECT_SHARED_ACCESS_INTEL,
                param_value_size,
                param_value);
            ret = (check == CL_SUCCESS) ? CL_SUCCESS : ret;
            return check;
        }
    default: break;
    }

    return g_pNextDispatch->clSetKernelExecInfo(
        kernel,
        param_name,
        param_value_size,
        param_value);
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
    cl_svm_advice_exp advice,
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
