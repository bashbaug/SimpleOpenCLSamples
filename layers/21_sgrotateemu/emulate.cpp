/*
// Copyright (c) 2025 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <CL/cl.h>
#include <CL/cl_layer.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <map>
#include <string>
#include <vector>

#include "layer_util.hpp"

#include "emulate.h"
#include "sgrotate.cl.h"

static constexpr cl_version version_cl_khr_subgroup_rotate =
    CL_MAKE_VERSION(1, 0, 0);

struct SDeviceInfo
{
    bool supports_cl_khr_subgroup_rotate = true;
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
            getDeviceInfoForPlatform(platform);
        }
    }

    const SDeviceInfo& getDeviceInfo(cl_device_id device) const
    {
        // TODO: query the parent device if this is a sub-device?
        return m_DeviceInfo.at(device);
    }

    const SDeviceInfo& getDeviceInfo(cl_device_id device)
    {
        // TODO: query the parent device if this is a sub-device?
        return m_DeviceInfo[device];
    }

private:
    std::map<cl_device_id, SDeviceInfo>     m_DeviceInfo;

    void getDeviceInfoForPlatform(cl_platform_id platform)
    {
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

        for (auto device : devices) {
            SDeviceInfo& deviceInfo = m_DeviceInfo[device];

            size_t size = 0;

            std::string deviceExtensions;
            g_pNextDispatch->clGetDeviceInfo(
                device,
                CL_DEVICE_EXTENSIONS,
                0,
                nullptr,
                &size);
            if (size) {
                deviceExtensions.resize(size);
                g_pNextDispatch->clGetDeviceInfo(
                    device,
                    CL_DEVICE_EXTENSIONS,
                    size,
                    &deviceExtensions[0],
                    nullptr);
                deviceExtensions.pop_back();
                deviceInfo.supports_cl_khr_subgroup_rotate =
                    checkStringForExtension(
                        deviceExtensions.c_str(),
                        CL_KHR_SUBGROUP_ROTATE_EXTENSION_NAME);
            }
        }
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
    return !deviceInfo.supports_cl_khr_subgroup_rotate;
}

static inline bool doEmulation(cl_context context)
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
        [](cl_device_id device) { return doEmulation(device); });
}

cl_program clCreateProgramWithSource_override(
    cl_context context,
    cl_uint count,
    const char** strings,
    const size_t* lengths,
    cl_int* errcode_ret)
{
    if (!doEmulation(context)) {
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
    newStrings.insert(newStrings.end(), g_SubgroupRotateString);
    newStrings.insert(newStrings.end(), strings, strings + count);

    std::vector<size_t> newLengths;
    if (lengths != nullptr) {
        newLengths.reserve(count + 1);
        newLengths.insert(newLengths.end(), 0);  // g_SubgroupRotateString is nul-terminated
        newLengths.insert(newLengths.end(), lengths, lengths + count);
    }

    return g_pNextDispatch->clCreateProgramWithSource(
        context,
        count + 1,
        newStrings.data(),
        newLengths.size() ? newLengths.data() : nullptr,
        errcode_ret);
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
                    CL_KHR_SUBGROUP_ROTATE_EXTENSION_NAME ) == false &&
                checkStringForExtension(
                    deviceExtensions.data(),
                    CL_KHR_SUBGROUP_SHUFFLE_EXTENSION_NAME ) )
            {
                std::string newExtensions;
                newExtensions += CL_KHR_SUBGROUP_ROTATE_EXTENSION_NAME;

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

            bool supports_cl_khr_subgroup_rotate = false;
            bool supports_cl_khr_subgroup_shuffle = false;
            for( const auto& extension : extensions )
            {
                if( strcmp(extension.name, CL_KHR_SUBGROUP_ROTATE_EXTENSION_NAME) == 0 )
                {
                    supports_cl_khr_subgroup_rotate = true;
                }
                if( strcmp(extension.name, CL_KHR_SUBGROUP_SHUFFLE_EXTENSION_NAME) == 0 )
                {
                    supports_cl_khr_subgroup_shuffle = true;
                }
            }

            if( supports_cl_khr_subgroup_rotate == false &&
                supports_cl_khr_subgroup_shuffle )
            {
                extensions.emplace_back();
                cl_name_version& extension = extensions.back();

                memset(extension.name, 0, CL_NAME_VERSION_MAX_NAME_SIZE);
                strcpy(extension.name, CL_KHR_SUBGROUP_ROTATE_EXTENSION_NAME);

                extension.version = version_cl_khr_subgroup_rotate;

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
