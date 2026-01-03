/*
// Copyright (c) 2021-2026 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/
#pragma once

#include <CL/opencl.hpp>
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
