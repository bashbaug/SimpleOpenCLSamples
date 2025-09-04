/*
// Copyright (c) 2026 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <CL/cl_layer.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <map>
#include <string>
#include <vector>

#include "layer_util.hpp"

#include "emulate.h"

struct SPlatformInfo
{
    bool emulate_OpenCL31 = false;
};

struct SDeviceInfo
{
    bool emulate_OpenCL31 = false;

    clGetKernelSuggestedLocalWorkSizeKHR_fn
        clGetKernelSuggestedLocalWorkSizeKHR = nullptr;
};

struct SLayerContext
{
    SLayerContext()
    {
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

        for (auto platform : platforms) {
            checkPlatform(platform);
        }
    }

    const SPlatformInfo& getPlatformInfo(cl_platform_id platform)
    {
        return m_PlatformInfo[platform];
    }

    const SDeviceInfo& getDeviceInfo(cl_device_id device)
    {
        // TODO: query the parent device if this is a sub-device?
        return m_DeviceInfo[device];
    }

    const SDeviceInfo& getDeviceInfo(cl_context context)
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
            devices.size() * sizeof(cl_device_id),
            devices.data(),
            nullptr);

        return getDeviceInfo(devices[0]);
    }

    const SDeviceInfo& getDeviceInfo(cl_command_queue queue)
    {
        cl_device_id device = nullptr;
        g_pNextDispatch->clGetCommandQueueInfo(
            queue,
            CL_QUEUE_DEVICE,
            sizeof(device),
            &device,
            nullptr);

        return getDeviceInfo(device);
    }

    const SDeviceInfo& getDeviceInfo(cl_program program)
    {
        cl_context context = nullptr;
        g_pNextDispatch->clGetProgramInfo(
            program,
            CL_PROGRAM_CONTEXT,
            sizeof(context),
            &context,
            nullptr);

        return getDeviceInfo(context);
    }

private:
    std::map<cl_platform_id, SPlatformInfo> m_PlatformInfo;
    std::map<cl_device_id, SDeviceInfo>     m_DeviceInfo;

    void checkPlatform(cl_platform_id platform)
    {
        SPlatformInfo& platformInfo = m_PlatformInfo[platform];

        cl_uint numDevices = 0;
        g_pNextDispatch->clGetDeviceIDs(
            platform,
            CL_DEVICE_TYPE_ALL,
            0,
            nullptr,
            &numDevices);

        std::vector<cl_device_id> devices(numDevices);
        g_pNextDispatch->clGetDeviceIDs(
            platform,
        CL_DEVICE_TYPE_ALL,
            numDevices,
            devices.data(),
            nullptr);

        for (auto device : devices) {
            checkDevice(device);
        }

        platformInfo.emulate_OpenCL31 = std::any_of(
            devices.begin(),
            devices.end(),
            [this](cl_device_id device) { return getDeviceInfo(device).emulate_OpenCL31; });
    }

    void checkDevice(cl_device_id device)
    {
        SDeviceInfo& deviceInfo = m_DeviceInfo[device];

        size_t size = 0;

        // Skip emulation for devices prior to OpenCL 3.0, or when OpenCL 3.1 or
        // newer is already supported:
        cl_version  version = CL_MAKE_VERSION(0, 0, 0);
        g_pNextDispatch->clGetDeviceInfo(
            device,
            CL_DEVICE_NUMERIC_VERSION,
            sizeof(version),
            &version,
            nullptr);
        if (version < CL_MAKE_VERSION(3, 0, 0) ||
            version >= CL_MAKE_VERSION(3, 1, 0)) {
            return;
        }

        // Skip emulation if sub-groups are not supported:
        cl_uint maxNumSubGroups = 0;
        g_pNextDispatch->clGetDeviceInfo(
            device,
            CL_DEVICE_MAX_NUM_SUB_GROUPS,
            sizeof(maxNumSubGroups),
            &maxNumSubGroups,
            nullptr);
        if (maxNumSubGroups == 0) {
            return;
        }

        // Skip emulation if required extensions are not supported:
        g_pNextDispatch->clGetDeviceInfo(
            device,
            CL_DEVICE_EXTENSIONS_WITH_VERSION,
            0,
            nullptr,
            &size );

        const size_t numExtensions = size / sizeof(cl_name_version);
        std::vector<cl_name_version>    extensions(numExtensions);
        g_pNextDispatch->clGetDeviceInfo(
            device,
            CL_DEVICE_EXTENSIONS_WITH_VERSION,
            size,
            extensions.data(),
            nullptr );

        const std::vector<const char*>  requiredExtensions{
            "cl_khr_spirv_queries",
            "cl_khr_extended_bit_ops",
            "cl_khr_suggested_local_work_size",
            "cl_khr_device_uuid",
            "cl_khr_integer_dot_product",
            "cl_khr_subgroup_extended_types",
            "cl_khr_subgroup_rotate",
            "cl_khr_subgroup_shuffle",
            "cl_khr_subgroup_shuffle_relative",
        };
        for (const auto& check : requiredExtensions) {
            if (!checkForSupport(extensions, check)) {
                return;
            }
        }

        // Skip emulation if required SPIR-V versions are not supported:
        g_pNextDispatch->clGetDeviceInfo(
            device,
            CL_DEVICE_ILS_WITH_VERSION,
            0,
            nullptr,
            &size );

        const size_t numILs = size / sizeof(cl_name_version);
        std::vector<cl_name_version>    ils(numILs);
        g_pNextDispatch->clGetDeviceInfo(
            device,
            CL_DEVICE_ILS_WITH_VERSION,
            size,
            ils.data(),
            nullptr );

        const std::vector<cl_version>   requiredILVersions{
            CL_MAKE_VERSION(1, 0, 0),
            CL_MAKE_VERSION(1, 1, 0),
            CL_MAKE_VERSION(1, 2, 0),
            CL_MAKE_VERSION(1, 3, 0),
            CL_MAKE_VERSION(1, 4, 0),
        };
        for (const auto& check : requiredILVersions) {
            if (!checkForSupport(ils, "SPIR-V", check)) {
                return;
            }
        }

        cl_platform_id platform = nullptr;
        g_pNextDispatch->clGetDeviceInfo(
            device,
            CL_DEVICE_PLATFORM,
            sizeof(platform),
            &platform,
            nullptr );

        deviceInfo.emulate_OpenCL31 = true;
        deviceInfo.clGetKernelSuggestedLocalWorkSizeKHR =
            (clGetKernelSuggestedLocalWorkSizeKHR_fn)
                g_pNextDispatch->clGetExtensionFunctionAddressForPlatform(
                    platform,
                    "clGetKernelSuggestedLocalWorkSizeKHR");
    }
};

SLayerContext& getLayerContext(void)
{
    static SLayerContext c;
    return c;
}

static inline bool doEmulation(cl_device_id device)
{
    const auto& deviceInfo = getLayerContext().getDeviceInfo(device);
    return deviceInfo.emulate_OpenCL31;
}

static inline bool doEmulation(cl_command_queue queue)
{
    const auto& deviceInfo = getLayerContext().getDeviceInfo(queue);
    return deviceInfo.emulate_OpenCL31;
}

static inline bool doEmulation(cl_platform_id platform)
{
    const auto& platformInfo = getLayerContext().getPlatformInfo(platform);
    return platformInfo.emulate_OpenCL31;
}

cl_int CL_API_CALL clGetKernelSuggestedLocalWorkSize_EMU(
    cl_command_queue queue,
    cl_kernel kernel,
    cl_uint work_dim,
    const size_t* global_work_offset,
    const size_t* global_work_size,
    size_t* suggested_local_work_size)
{
    if (!doEmulation(queue)) {
        return CL_INVALID_OPERATION;
    }

    const auto& deviceInfo = getLayerContext().getDeviceInfo(queue);
    return deviceInfo.clGetKernelSuggestedLocalWorkSizeKHR(
        queue,
        kernel,
        work_dim,
        global_work_offset,
        global_work_size,
        suggested_local_work_size);
}

bool clGetDeviceInfo_override(
    cl_device_id device,
    cl_device_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret,
    cl_int* errcode_ret)
{
    if (!doEmulation(device)) {
        return false;
    }

    switch(param_name) {
    case CL_DEVICE_VERSION:
        {
            size_t size = 0;
            g_pNextDispatch->clGetDeviceInfo(
                device,
                CL_DEVICE_VERSION,
                0,
                nullptr,
                &size);

            std::string version;
            version.resize(size);
            g_pNextDispatch->clGetDeviceInfo(
                device,
                CL_DEVICE_VERSION,
                size,
                &version[0],
                nullptr);
            version.pop_back();

            std::string newVersion = "OpenCL 3.1 over " + version;
            auto pVersion = (char*)param_value;
            cl_int errorCode = writeStringToMemory(
                param_value_size,
                newVersion.c_str(),
                param_value_size_ret,
                pVersion);
            if (errcode_ret) {
                errcode_ret[0] = errorCode;
            }
            return true;
        }
        break;
    case CL_DEVICE_NUMERIC_VERSION:
        {
            const cl_version version_31 = CL_MAKE_VERSION(3, 1, 0);
            auto pVersion = (cl_version*)param_value;
            cl_int errorCode = writeParamToMemory(
                sizeof(version_31),
                version_31,
                param_value_size_ret,
                pVersion);
            if (errcode_ret) {
                errcode_ret[0] = errorCode;
            }
            return true;
        }
        break;
    case CL_DEVICE_OPENCL_C_ALL_VERSIONS:
        {
            size_t size = 0;
            g_pNextDispatch->clGetDeviceInfo(
                device,
                CL_DEVICE_OPENCL_C_ALL_VERSIONS,
                0,
                nullptr,
                &size );

            const size_t count = size / sizeof(cl_name_version);
            std::vector<cl_name_version>    clcs(count);
            g_pNextDispatch->clGetDeviceInfo(
                device,
                CL_DEVICE_OPENCL_C_ALL_VERSIONS,
                size,
                clcs.data(),
                nullptr );

            clcs.push_back({CL_MAKE_VERSION(3, 1, 0), "OpenCL C"});

            auto pCLCs = (cl_name_version*)param_value;
            cl_int errorCode = writeVectorToMemory(
                param_value_size,
                clcs,
                param_value_size_ret,
                pCLCs);
            if (errcode_ret) {
                errcode_ret[0] = errorCode;
            }
            return true;
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
    if (!doEmulation(platform)) {
        return false;
    }

    switch(param_name) {
    case CL_PLATFORM_VERSION:
        {
            size_t size = 0;
            g_pNextDispatch->clGetPlatformInfo(
                platform,
                CL_PLATFORM_VERSION,
                0,
                nullptr,
                &size);

            std::string version;
            version.resize(size);
            g_pNextDispatch->clGetPlatformInfo(
                platform,
                CL_PLATFORM_VERSION,
                size,
                &version[0],
                nullptr);
            version.pop_back();

            std::string newVersion = "OpenCL 3.1 over " + version;
            auto pVersion = (char*)param_value;
            cl_int errorCode = writeStringToMemory(
                param_value_size,
                newVersion.c_str(),
                param_value_size_ret,
                pVersion);
            if (errcode_ret) {
                errcode_ret[0] = errorCode;
            }
            return true;
        }
        break;
    case CL_PLATFORM_NUMERIC_VERSION:
        {
            const cl_version version_31 = CL_MAKE_VERSION(3, 1, 0);
            auto pVersion = (cl_version*)param_value;
            cl_int errorCode = writeParamToMemory(
                sizeof(version_31),
                version_31,
                param_value_size_ret,
                pVersion);
            if (errcode_ret) {
                errcode_ret[0] = errorCode;
            }
            return true;
        }
        break;
    default: break;
    }

    return false;
}

std::string replaceBuildOptions(
    cl_program program,
    const char* options)
{
    std::string newOptions;

    const auto& deviceInfo = getLayerContext().getDeviceInfo(program);
    if (deviceInfo.emulate_OpenCL31 && options) {
        const std::string find("-cl-std=CL3.1");
        const std::string replace("-cl-std=CL3.0");
        size_t pos = 0;
        newOptions = options;
        while((pos = newOptions.find(find, pos)) != std::string::npos) {
            newOptions.replace(pos, find.length(), replace);
            pos += replace.length();
        }
    }

    return newOptions;
}
