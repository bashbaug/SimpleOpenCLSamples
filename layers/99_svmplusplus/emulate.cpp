/*
// Copyright (c) 2023 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <CL/cl_layer.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <cassert>

#include "layer_util.hpp"

#include "emulate.h"

static constexpr cl_version version_cl_khr_unified_svm =
    CL_MAKE_VERSION(0, 2, 0);

static cl_svm_capabilities_khr getDeviceUSMCaps(cl_device_id device)
{
    cl_svm_capabilities_khr ret = 0;

    cl_device_unified_shared_memory_capabilities_intel usmCaps = 0;
    g_pNextDispatch->clGetDeviceInfo(
        device,
        CL_DEVICE_DEVICE_MEM_CAPABILITIES_INTEL,
        sizeof(usmCaps),
        &usmCaps,
        nullptr);

    if (usmCaps != 0) {
        ret = CL_SVM_TYPE_MACRO_DEVICE_KHR;

        assert(usmCaps & CL_UNIFIED_SHARED_MEMORY_ACCESS_INTEL);
        assert(usmCaps & CL_UNIFIED_SHARED_MEMORY_ATOMIC_ACCESS_INTEL);

        if (usmCaps & CL_UNIFIED_SHARED_MEMORY_CONCURRENT_ACCESS_INTEL) {
            ret |= CL_SVM_CAPABILITY_CONCURRENT_ACCESS_KHR;
        }
        if (usmCaps & CL_UNIFIED_SHARED_MEMORY_CONCURRENT_ATOMIC_ACCESS_INTEL) {
            ret |= CL_SVM_CAPABILITY_CONCURRENT_ATOMIC_ACCESS_KHR;
        }
    }

    return ret;
}

static cl_svm_capabilities_khr getHostUSMCaps(cl_device_id device)
{
    cl_svm_capabilities_khr ret = 0;

    cl_device_unified_shared_memory_capabilities_intel usmCaps = 0;
    g_pNextDispatch->clGetDeviceInfo(
        device,
        CL_DEVICE_HOST_MEM_CAPABILITIES_INTEL,
        sizeof(usmCaps),
        &usmCaps,
        nullptr);

    if (usmCaps != 0) {
        ret = CL_SVM_TYPE_MACRO_HOST_KHR;

        assert(usmCaps & CL_UNIFIED_SHARED_MEMORY_ACCESS_INTEL);

        if (usmCaps & CL_UNIFIED_SHARED_MEMORY_ATOMIC_ACCESS_INTEL) {
            ret |= CL_SVM_CAPABILITY_DEVICE_ATOMIC_ACCESS_KHR;
        }
        if (usmCaps & CL_UNIFIED_SHARED_MEMORY_CONCURRENT_ACCESS_INTEL) {
            ret |= CL_SVM_CAPABILITY_CONCURRENT_ACCESS_KHR;
        }
        if (usmCaps & CL_UNIFIED_SHARED_MEMORY_CONCURRENT_ATOMIC_ACCESS_INTEL) {
            ret |= CL_SVM_CAPABILITY_CONCURRENT_ATOMIC_ACCESS_KHR;
        }
    }

    return ret;
}

static cl_svm_capabilities_khr getSingleDeviceSharedUSMCaps(cl_device_id device)
{
    cl_svm_capabilities_khr ret = 0;

    cl_device_unified_shared_memory_capabilities_intel usmCaps = 0;
    g_pNextDispatch->clGetDeviceInfo(
        device,
        CL_DEVICE_SINGLE_DEVICE_SHARED_MEM_CAPABILITIES_INTEL,
        sizeof(usmCaps),
        &usmCaps,
        nullptr);

    if (usmCaps != 0) {
        ret = CL_SVM_TYPE_MACRO_SINGLE_DEVICE_SHARED_KHR;

        assert(usmCaps & CL_UNIFIED_SHARED_MEMORY_ACCESS_INTEL);

        if (usmCaps & CL_UNIFIED_SHARED_MEMORY_ATOMIC_ACCESS_INTEL) {
            ret |= CL_SVM_CAPABILITY_DEVICE_ATOMIC_ACCESS_KHR;
        }
        if (usmCaps & CL_UNIFIED_SHARED_MEMORY_CONCURRENT_ACCESS_INTEL) {
            ret |= CL_SVM_CAPABILITY_CONCURRENT_ACCESS_KHR;
        }
        if (usmCaps & CL_UNIFIED_SHARED_MEMORY_CONCURRENT_ATOMIC_ACCESS_INTEL) {
            ret |= CL_SVM_CAPABILITY_CONCURRENT_ATOMIC_ACCESS_KHR;
        }
    }

    return ret;
}

static cl_svm_capabilities_khr getCoarseGrainSVMCaps(cl_device_id device)
{
    cl_svm_capabilities_khr ret = 0;

    cl_device_svm_capabilities svmCaps = 0;
    g_pNextDispatch->clGetDeviceInfo(
        device,
        CL_DEVICE_SVM_CAPABILITIES,
        sizeof(svmCaps),
        &svmCaps,
        nullptr);

    if (svmCaps & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER) {
        ret = CL_SVM_TYPE_MACRO_COARSE_GRAIN_BUFFER_KHR;
    }

    return ret;
}

static cl_svm_capabilities_khr getFineGrainSVMCaps(cl_device_id device)
{
    cl_svm_capabilities_khr ret = 0;

    cl_device_svm_capabilities svmCaps = 0;
    g_pNextDispatch->clGetDeviceInfo(
        device,
        CL_DEVICE_SVM_CAPABILITIES,
        sizeof(svmCaps),
        &svmCaps,
        nullptr);

    if (svmCaps & CL_DEVICE_SVM_FINE_GRAIN_BUFFER) {
        ret = CL_SVM_TYPE_MACRO_FINE_GRAIN_BUFFER_KHR;

        if (svmCaps & CL_DEVICE_SVM_ATOMICS) {
            ret |= CL_SVM_CAPABILITY_CONCURRENT_ATOMIC_ACCESS_KHR;
        }
    }

    return ret;
}

static cl_svm_capabilities_khr getSystemSVMCaps(cl_device_id device)
{
    cl_svm_capabilities_khr ret = 0;

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
        ret = CL_SVM_TYPE_MACRO_SYSTEM_KHR;
    }

    return ret;
}

struct SLayerContext
{
    SLayerContext() {
        cl_uint numPlatforms = 0;
        g_pNextDispatch->clGetPlatformIDs(
            0,
            nullptr,
            &numPlatforms);

        std::vector<cl_platform_id> platforms;
        platforms.resize(numPlatforms);
        g_pNextDispatch->clGetPlatformIDs(
            numPlatforms,
            platforms.data(),
            nullptr);

        for (auto platform: platforms) {
            getSVMTypesForPlatform(platform);
        }
    }

    std::map<cl_platform_id, std::vector<cl_svm_capabilities_khr>>  TypeCapsPlatform;
    std::map<cl_device_id, std::vector<cl_svm_capabilities_khr>>    TypeCapsDevice;

private:
    void getSVMTypesForPlatform(cl_platform_id platform)
    {
        std::vector<cl_svm_capabilities_khr> typeCapsPlatform;

        cl_uint numDevices = 0;
        g_pNextDispatch->clGetDeviceIDs(
            platform,
            CL_DEVICE_TYPE_ALL,
            0,
            nullptr,
            &numDevices);

        std::vector<cl_device_id> devices;
        devices.resize(numDevices);
        g_pNextDispatch->clGetDeviceIDs(
            platform,
            CL_DEVICE_TYPE_ALL,
            numDevices,
            devices.data(),
            nullptr);

        // USM Types:

        cl_svm_capabilities_khr combinedCaps = 0;
        for (auto device: devices) {
            cl_svm_capabilities_khr caps = getDeviceUSMCaps(device);
            if (caps != 0) {
                combinedCaps = (combinedCaps == 0) ? caps : (combinedCaps & caps);
            }
        }
        if (combinedCaps != 0) {
            typeCapsPlatform.push_back(combinedCaps);
        }

        combinedCaps = 0;
        for (auto device: devices) {
            cl_svm_capabilities_khr caps = getHostUSMCaps(device);
            if (caps != 0) {
                combinedCaps = (combinedCaps == 0) ? caps : (combinedCaps & caps);
            }
        }
        if (combinedCaps != 0) {
            typeCapsPlatform.push_back(combinedCaps);
        }

        combinedCaps = 0;
        for (auto device: devices) {
            cl_svm_capabilities_khr caps = getSingleDeviceSharedUSMCaps(device);
            if (caps != 0) {
                combinedCaps = (combinedCaps == 0) ? caps : (combinedCaps & caps);
            }
        }
        if (combinedCaps != 0) {
            typeCapsPlatform.push_back(combinedCaps);
        }

        // SVM Types:

        combinedCaps = 0;
        for (auto device: devices) {
            cl_svm_capabilities_khr caps = getCoarseGrainSVMCaps(device);
            if (caps != 0) {
                combinedCaps = (combinedCaps == 0) ? caps : (combinedCaps & caps);
            }
        }
        if (combinedCaps != 0) {
            typeCapsPlatform.push_back(combinedCaps);
        }

        combinedCaps = 0;
        for (auto device: devices) {
            cl_svm_capabilities_khr caps = getFineGrainSVMCaps(device);
            if (caps != 0) {
                combinedCaps = (combinedCaps == 0) ? caps : (combinedCaps & caps);
            }
        }
        if (combinedCaps != 0) {
            typeCapsPlatform.push_back(combinedCaps);
        }

        combinedCaps = 0;
        for (auto device: devices) {
            cl_svm_capabilities_khr caps = getSystemSVMCaps(device);
            if (caps != 0) {
                combinedCaps = (combinedCaps == 0) ? caps : (combinedCaps & caps);
            }
        }
        if (combinedCaps != 0) {
            typeCapsPlatform.push_back(combinedCaps);
        }

        TypeCapsPlatform[platform] = typeCapsPlatform;

        for (auto device: devices) {
            std::vector<cl_svm_capabilities_khr> typeCapsDevice;

            for (auto caps: typeCapsPlatform) {
                if ((caps & CL_SVM_TYPE_MACRO_DEVICE_KHR) == CL_SVM_TYPE_MACRO_DEVICE_KHR) {
                    typeCapsDevice.push_back(getDeviceUSMCaps(device));
                }
                else if ((caps & CL_SVM_TYPE_MACRO_HOST_KHR) == CL_SVM_TYPE_MACRO_HOST_KHR) {
                    typeCapsDevice.push_back(getHostUSMCaps(device));
                }
                else if ((caps & CL_SVM_TYPE_MACRO_SYSTEM_KHR) == CL_SVM_TYPE_MACRO_SYSTEM_KHR) {
                    typeCapsDevice.push_back(getSystemSVMCaps(device));
                }
                else if ((caps & CL_SVM_TYPE_MACRO_SINGLE_DEVICE_SHARED_KHR) == CL_SVM_TYPE_MACRO_SINGLE_DEVICE_SHARED_KHR) {
                    typeCapsDevice.push_back(getSingleDeviceSharedUSMCaps(device));
                }
                else if ((caps & CL_SVM_TYPE_MACRO_FINE_GRAIN_BUFFER_KHR) == CL_SVM_TYPE_MACRO_FINE_GRAIN_BUFFER_KHR) {
                    typeCapsDevice.push_back(getFineGrainSVMCaps(device));
                }
                else if ((caps & CL_SVM_TYPE_MACRO_COARSE_GRAIN_BUFFER_KHR) == CL_SVM_TYPE_MACRO_COARSE_GRAIN_BUFFER_KHR) {
                    typeCapsDevice.push_back(getCoarseGrainSVMCaps(device));
                }
                else {
                    assert(0 && "unknown platform SVM type");
                }
            }

            TypeCapsDevice[device] = typeCapsDevice;
        }
    }
};

SLayerContext& getLayerContext(void)
{
    static SLayerContext c;
    return c;
}

static cl_platform_id getPlatform(
    cl_device_id device)
{
    cl_platform_id platform = nullptr;
    g_pNextDispatch->clGetDeviceInfo(
        device,
        CL_DEVICE_PLATFORM,
        sizeof(platform),
        &platform,
        nullptr);
    return platform;
}

static cl_platform_id getPlatform(
    cl_context context)
{
    cl_uint numDevices = 0;
    g_pNextDispatch->clGetContextInfo(
        context,
        CL_CONTEXT_NUM_DEVICES,
        sizeof(numDevices),
        &numDevices,
        nullptr );

    if (numDevices == 1) {  // fast path, no dynamic allocation
        cl_device_id    device = nullptr;
        g_pNextDispatch->clGetContextInfo(
            context,
            CL_CONTEXT_DEVICES,
            sizeof(cl_device_id),
            &device,
            nullptr );
        return getPlatform(device);
    }

    // slower path, dynamic allocation
    std::vector<cl_device_id> devices(numDevices);
    g_pNextDispatch->clGetContextInfo(
        context,
        CL_CONTEXT_DEVICES,
        numDevices * sizeof(cl_device_id),
        devices.data(),
        nullptr );
    return getPlatform(devices[0]);
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

static void parseSVMAllocProperties(
    const cl_svm_alloc_properties_khr* props,
    cl_device_id& device,
    cl_svm_alloc_access_flags_khr& flags,
    size_t& alignment)
{
    device = nullptr;
    flags = 0;
    alignment = 0;

    if (props) {
        while(props[0] != 0) {
            cl_int property = (cl_int)props[0];
            switch(property) {
            case CL_SVM_ALLOC_ASSOCIATED_DEVICE_HANDLE_KHR: {
                    device = *(const cl_device_id*)(props + 1);
                    props += 2;
                }
                break;
            case CL_SVM_ALLOC_ACCESS_FLAGS_KHR: {
                    flags = *(const cl_svm_alloc_access_flags_khr*)(props + 1);
                    props += 2;
                }
                break;
            case CL_SVM_ALLOC_ALIGNMENT_KHR: {
                    alignment = *(const size_t*)(props + 1);
                    props += 2;
                }
                break;
            default:
                // Not supporting other properties currently.
                assert(0 && "unknown SVM mem property");
                break;
            }
        }
    }
}

void* CL_API_CALL clSVMAllocWithPropertiesKHR_EMU(
    cl_context context,
    const cl_svm_alloc_properties_khr* properties,
    cl_uint svm_type_index,
    size_t size,
    cl_int* errcode_ret)
{
    cl_platform_id platform = getPlatform(context);

    SLayerContext& layerContext = getLayerContext();
    const auto& typeCapsPlatform = layerContext.TypeCapsPlatform[platform];
    if (svm_type_index >= typeCapsPlatform.size()) {
        if (errcode_ret) {
            errcode_ret[0] = CL_INVALID_VALUE;
        }
        return nullptr;
    }

    cl_device_id device = nullptr;
    cl_svm_alloc_access_flags_khr flags = 0;
    size_t alignment = 0;
    parseSVMAllocProperties(properties, device, flags, alignment);

    switch(typeCapsPlatform[svm_type_index]) {
    case CL_SVM_TYPE_MACRO_DEVICE_KHR:
        return clDeviceMemAllocINTEL(
            context,
            device,
            nullptr,
            size,
            alignment,
            errcode_ret);
    case CL_SVM_TYPE_MACRO_HOST_KHR:
        return clHostMemAllocINTEL(
            context,
            nullptr,
            size,
            alignment,
            errcode_ret);
    case CL_SVM_TYPE_MACRO_SINGLE_DEVICE_SHARED_KHR:
        return clSharedMemAllocINTEL(
            context,
            device,
            nullptr,
            size,
            alignment,
            errcode_ret);
    case CL_SVM_TYPE_MACRO_COARSE_GRAIN_BUFFER_KHR: {
            void* ret = g_pNextDispatch->clSVMAlloc(
                context,
                CL_MEM_READ_WRITE,
                size,
                alignment);
            if (errcode_ret) {
                errcode_ret[0] = ret ? CL_SUCCESS : CL_INVALID_VALUE;
            }
            return ret;
        }
    case CL_SVM_TYPE_MACRO_FINE_GRAIN_BUFFER_KHR: {
            cl_svm_mem_flags svmFlags = CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER;

            const auto& typeCapsDevice = layerContext.TypeCapsDevice[device];
            if (typeCapsDevice[svm_type_index] & CL_SVM_CAPABILITY_CONCURRENT_ATOMIC_ACCESS_KHR) {
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
    default:
        assert(0 && "unknown SVM type");
        break;
    }

    if (errcode_ret) {
        errcode_ret[0] = CL_INVALID_OPERATION;
    }
    return nullptr;
}

cl_int CL_API_CALL clSVMFreeWithPropertiesKHR_EMU(
    cl_context context,
    const cl_svm_free_properties_khr* properties,
    cl_svm_free_flags_khr flags,
    void* ptr)
{
    if (isUSMPtr(context, ptr)) {
        if (flags & CL_SVM_FREE_BLOCKING_KHR) {
            return clMemBlockingFreeINTEL(
                context,
                ptr);
        }
        return clMemFreeINTEL(
            context,
            ptr);
    }

    assert(!(flags & CL_SVM_FREE_BLOCKING_KHR) && "blocking SVM free is currently unsupported");
    g_pNextDispatch->clSVMFree(
        context,
        ptr);
    return CL_SUCCESS;
}

cl_int CL_API_CALL clGetSVMSuggestedTypeIndexKHR_EMU(
    cl_context context,
    cl_svm_capabilities_khr required_capabilities,
    cl_svm_capabilities_khr desired_capabilities,
    const cl_svm_alloc_properties_khr* properties,
    size_t size,
    cl_uint* suggested_svm_type_index)
{
    if (suggested_svm_type_index == nullptr) {
        return CL_INVALID_VALUE;
    }

    cl_device_id associatedDevice = nullptr;
    cl_svm_alloc_access_flags_khr flags = 0;
    size_t alignment = 0;
    parseSVMAllocProperties(properties, associatedDevice, flags, alignment);

    std::vector<cl_device_id> checkDevices;
    if (associatedDevice) {
        checkDevices.push_back(associatedDevice);
    } else {
        cl_uint numDevices = 0;
        g_pNextDispatch->clGetContextInfo(
            context,
            CL_CONTEXT_NUM_DEVICES,
            sizeof(numDevices),
            &numDevices,
            nullptr );

        checkDevices.resize(numDevices);
        g_pNextDispatch->clGetContextInfo(
            context,
            CL_CONTEXT_DEVICES,
            numDevices * sizeof(cl_device_id),
            checkDevices.data(),
            nullptr );
    }

    // Note: this currently ignores the desired capabilities.
    // It would be better to score each set of supported capabilities that
    // match the required capabilities against the desired capabilities,
    // then return the match with the highest score, possibly looking for
    // the highest total score across multiple devices.

    cl_uint ret = CL_UINT_MAX;
    for (auto device: checkDevices) {
        SLayerContext& layerContext = getLayerContext();
        const auto& supported = layerContext.TypeCapsDevice[device];
        for (size_t ci = 0; ci < supported.size(); ci++) {
            if ((supported[ci] & required_capabilities) == required_capabilities) {
                ret = static_cast<cl_uint>(ci);
                break;
            }
        }
        if (ret != CL_UINT_MAX) {
            break;
        }
    }

    suggested_svm_type_index[0] = ret;
    return CL_SUCCESS;
}

cl_int CL_API_CALL clGetSVMPointerInfoKHR_EMU(
    cl_context context,
    cl_device_id device,
    const void* ptr,
    cl_svm_pointer_info_khr param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    if (param_name == CL_SVM_INFO_TYPE_INDEX_KHR) {
        cl_unified_shared_memory_type_intel type = CL_MEM_TYPE_UNKNOWN_INTEL;
        clGetMemAllocInfoINTEL(
            context,
            ptr,
            CL_MEM_ALLOC_TYPE_INTEL,
            sizeof(type),
            &type,
            nullptr);
        cl_svm_capabilities_khr search = 0;
        switch (type) {
        case CL_MEM_TYPE_DEVICE_INTEL:
            search = CL_SVM_TYPE_MACRO_DEVICE_KHR;
            break;
        case CL_MEM_TYPE_HOST_INTEL:
            search = CL_SVM_TYPE_MACRO_HOST_KHR;
            break;
        case CL_MEM_TYPE_SHARED_INTEL:
            search = CL_SVM_TYPE_MACRO_SINGLE_DEVICE_SHARED_KHR;
            break;
        default:
            break;
        }

        SLayerContext& layerContext = getLayerContext();
        cl_platform_id platform = getPlatform(context);
        const auto& platformSVMCaps = layerContext.TypeCapsPlatform[platform];

        cl_uint index = CL_UINT_MAX;
        for (size_t ci = 0; ci < platformSVMCaps.size(); ci++) {
            if (platformSVMCaps[ci] == search) {
                index = static_cast<cl_uint>(ci);
                break;
            }
        }

        auto ptr = (cl_uint*)param_value;
        return writeParamToMemory(
            param_value_size,
            index,
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
                    CL_KHR_UNIFIED_SVM_EXTENSION_NAME ) == false )
            {
                std::string newExtensions;
                newExtensions += CL_KHR_UNIFIED_SVM_EXTENSION_NAME;

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
                if( strcmp(extension.name, CL_KHR_UNIFIED_SVM_EXTENSION_NAME) == 0 )
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
                strcpy(extension.name, CL_KHR_UNIFIED_SVM_EXTENSION_NAME);

                extension.version = version_cl_khr_unified_svm;

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
    case CL_DEVICE_SVM_TYPE_CAPABILITIES_KHR:
        {
            SLayerContext& layerContext = getLayerContext();
            const auto& deviceSVMCaps = layerContext.TypeCapsDevice[device];
            auto ptr = (cl_svm_capabilities_khr*)param_value;
            return writeVectorToMemory(
                param_value_size,
                deviceSVMCaps,
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
                    CL_KHR_UNIFIED_SVM_EXTENSION_NAME ) == false )
            {
                std::string newExtensions;
                newExtensions += CL_KHR_UNIFIED_SVM_EXTENSION_NAME;

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
                if( strcmp(extension.name, CL_KHR_UNIFIED_SVM_EXTENSION_NAME) == 0 )
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
                strcpy(extension.name, CL_KHR_UNIFIED_SVM_EXTENSION_NAME);

                extension.version = version_cl_khr_unified_svm;

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
    case CL_PLATFORM_SVM_TYPE_CAPABILITIES_KHR:
        {
            SLayerContext& layerContext = getLayerContext();
            const auto& platformSVMCaps = layerContext.TypeCapsPlatform[platform];
            auto ptr = (cl_svm_capabilities_khr*)param_value;
            return writeVectorToMemory(
                param_value_size,
                platformSVMCaps,
                param_value_size_ret,
                ptr);
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
    case CL_KERNEL_EXEC_INFO_SVM_INDIRECT_ACCESS_KHR:
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
    } else {
        g_pNextDispatch->clSVMFree(context, ptr);
    }
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
