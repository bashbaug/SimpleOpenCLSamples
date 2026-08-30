/*
// Copyright (c) 2021-2026 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/
#pragma once

#include <CL/opencl.hpp>

#include <cctype>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>

static cl_version getDeviceOpenCLVersion(
    const cl::Device& device)
{
    cl_uint major = 0;
    cl_uint minor = 0;

    std::string version = device.getInfo<CL_DEVICE_VERSION>();

    // The device version string has the form:
    //   OpenCL <Major>.<Minor> <Vendor Specific Info>
    const std::string prefix{"OpenCL "};
    if (!version.compare(0, prefix.length(), prefix)) {
        const char* check = version.c_str() + prefix.length();
        while (isdigit(check[0])) {
            major *= 10;
            major += check[0] - '0';
            ++check;
        }
        if (check[0] == '.') {
            ++check;
        }
        while (isdigit(check[0])) {
            minor *= 10;
            minor += check[0] - '0';
            ++check;
        }
    }

    return CL_MAKE_VERSION(major, minor, 0);
}

static bool checkDeviceForExtension(
    const cl::Device& device,
    const char* extensionName)
{
    bool    supported = false;

    if (extensionName && !strchr(extensionName, ' ')) {
        std::string deviceExtensions = device.getInfo<CL_DEVICE_EXTENSIONS>();

        const char* start = deviceExtensions.c_str();
        while (true) {
            const char* where = strstr(start, extensionName);
            if (!where) {
                break;
            }
            const char* terminator = where + strlen(extensionName);
            if (where == start || *(where - 1) == ' ') {
                if (*terminator == ' ' || *terminator == '\0') {
                    supported = true;
                    break;
                }
            }
            start = terminator;
        }
    }

    return supported;
}

static std::string readStringFromFile(
    const std::string& filename )
{
    std::ifstream is(filename, std::ios::binary);
    if (!is.good()) {
        printf("Couldn't open file '%s'!\n", filename.c_str());
        return "";
    }

    std::string source{
        std::istreambuf_iterator<char>(is),
        std::istreambuf_iterator<char>() };

    return source;
}

static bool checkPlatformIndex(
    const std::vector<cl::Platform>& platforms,
    int platformIndex)
{
    if (platforms.size() == 0) {
        fprintf(stderr, "Error: No OpenCL platforms found.\n");
        return false;
    }
    if (platformIndex >= (int)platforms.size()) {
        fprintf(stderr, "Error: Invalid platform index %d specified (max %d)\n",
            platformIndex,
            (int)(platforms.size() - 1) );
        return false;
    }
    return true;
}

static bool checkDeviceIndex(
    const std::vector<cl::Device>& devices,
    int deviceIndex)
{
    if (devices.size() == 0) {
        fprintf(stderr, "Error: No OpenCL devices found.\n");
        return false;
    }
    if (deviceIndex >= (int)devices.size()) {
        fprintf(stderr, "Error: Invalid device index %d specified (max %d)\n",
            deviceIndex,
            (int)(devices.size() - 1) );
        return false;
    }
    return true;
}

static bool setupPlatformAndDevice(
    cl::Platform& platform,
    cl::Device& device,
    int platformIndex,
    int deviceIndex,
    bool verbose = false)
{
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    if (!checkPlatformIndex(platforms, platformIndex)) {
        return false;
    }

    platform = std::move(platforms[platformIndex]);
    printf("Running on platform: %s\n",
        platform.getInfo<CL_PLATFORM_NAME>().c_str() );

    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);  

    if (!checkDeviceIndex(devices, deviceIndex)) {
        return false;
    }

    device = std::move(devices[deviceIndex]);
    if (verbose) {
        printf("Running on device: %s (%uCUs, %uMHz)\n",
            device.getInfo<CL_DEVICE_NAME>().c_str(),
            device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>(),
            device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() );
        printf("Device version: %s\n",
            device.getInfo<CL_DEVICE_VERSION>().c_str() );
        printf("Driver version: %s\n",
            device.getInfo<CL_DRIVER_VERSION>().c_str() );
    } else {
        printf("Running on device: %s\n",
            device.getInfo<CL_DEVICE_NAME>().c_str() );
    }

    return true;
}
