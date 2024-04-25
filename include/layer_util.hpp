/*
// Copyright (c) 2022 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/
#pragma once

#include <CL/cl_layer.h>

#include <cstring>
#include <cctype>
#include <vector>

template<class T>
cl_int writeParamToMemory(
    size_t param_value_size,
    T param,
    size_t* param_value_size_ret,
    T* pointer)
{
    if (pointer != nullptr) {
        if (param_value_size < sizeof(param)) {
            return CL_INVALID_VALUE;
        }
        *pointer = param;
    }

    if (param_value_size_ret != nullptr) {
        *param_value_size_ret = sizeof(param);
    }

    return CL_SUCCESS;
}

template<class T>
cl_int writeVectorToMemory(
    size_t param_value_size,
    const std::vector<T>& param,
    size_t *param_value_size_ret,
    T* pointer )
{
    size_t  size = param.size() * sizeof(T);

    if (pointer != nullptr) {
        if (param_value_size < size) {
            return CL_INVALID_VALUE;
        }
        memcpy(pointer, param.data(), size);
    }

    if (param_value_size_ret != nullptr) {
        *param_value_size_ret = size;
    }

    return CL_SUCCESS;
}

static inline cl_int writeStringToMemory(
    size_t param_value_size,
    const char* param,
    size_t* param_value_size_ret,
    char* pointer )
{
    size_t  size = strlen(param) + 1;

    if (pointer != nullptr) {
        if (param_value_size < size) {
            return CL_INVALID_VALUE;
        }
        strcpy(pointer, param);
    }

    if (param_value_size_ret != nullptr) {
        *param_value_size_ret = size;
    }

    return CL_SUCCESS;
}

static cl_uint getOpenCLVersionFromString(
    const char* str)
{
    cl_uint major = 0;
    cl_uint minor = 0;

    // The device version string has the form:
    //   OpenCL <Major>.<Minor> <Vendor Specific Info>
    const char* prefix = "OpenCL ";
    size_t sz = strlen(prefix);
    if (strlen(str) > sz &&
        strncmp(str, prefix, sz) == 0) {
        const char* check = str + sz;
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

    return (major << 16) | minor;
}

static inline bool checkStringForExtension(
    const char* str,
    const char* extensionName )
{
    bool    supported = false;

    if (extensionName && !strchr(extensionName, ' ')) {
        const char* start = str;
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
