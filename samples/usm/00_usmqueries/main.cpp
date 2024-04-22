/*
// Copyright (c) 2020-2024 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <popl/popl.hpp>

#include <CL/opencl.hpp>

void PrintUSMCaps(
    const char* label,
    cl_device_unified_shared_memory_capabilities_intel usmcaps )
{
    printf("%s: %s%s%s%s\n",
        label,
        ( usmcaps & CL_UNIFIED_SHARED_MEMORY_ACCESS_INTEL                   ) ? "\n\t\tCL_UNIFIED_SHARED_MEMORY_ACCESS_INTEL"                   : "",
        ( usmcaps & CL_UNIFIED_SHARED_MEMORY_ATOMIC_ACCESS_INTEL            ) ? "\n\t\tCL_UNIFIED_SHARED_MEMORY_ATOMIC_ACCESS_INTEL"            : "",
        ( usmcaps & CL_UNIFIED_SHARED_MEMORY_CONCURRENT_ACCESS_INTEL        ) ? "\n\t\tCL_UNIFIED_SHARED_MEMORY_CONCURRENT_ACCESS_INTEL"        : "",
        ( usmcaps & CL_UNIFIED_SHARED_MEMORY_CONCURRENT_ATOMIC_ACCESS_INTEL ) ? "\n\t\tCL_UNIFIED_SHARED_MEMORY_CONCURRENT_ATOMIC_ACCESS_INTEL" : "" );
}

int main(
    int argc,
    char** argv )
{
    int platformIndex = 0;
    int deviceIndex = 0;

    {
        popl::OptionParser op("Supported Options");
        op.add<popl::Value<int>>("p", "platform", "Platform Index", platformIndex, &platformIndex);
        op.add<popl::Value<int>>("d", "device", "Device Index", deviceIndex, &deviceIndex);
        bool printUsage = false;
        try {
            op.parse(argc, argv);
        } catch (std::exception& e) {
            fprintf(stderr, "Error: %s\n\n", e.what());
            printUsage = true;
        }

        if (printUsage || !op.unknown_options().empty() || !op.non_option_args().empty()) {
            fprintf(stderr,
                "Usage: usmqueries [options]\n"
                "%s", op.help().c_str());
            return -1;
        }
    }

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    printf("Running on platform: %s\n",
        platforms[platformIndex].getInfo<CL_PLATFORM_NAME>().c_str() );

    std::vector<cl::Device> devices;
    platforms[platformIndex].getDevices(CL_DEVICE_TYPE_ALL, &devices);

    printf("Running on device: %s\n",
        devices[deviceIndex].getInfo<CL_DEVICE_NAME>().c_str() );

    cl_device_unified_shared_memory_capabilities_intel usmcaps = 0;

    clGetDeviceInfo(
        devices[deviceIndex](),
        CL_DEVICE_HOST_MEM_CAPABILITIES_INTEL,
        sizeof(usmcaps),
        &usmcaps,
        nullptr );
    PrintUSMCaps( "CL_DEVICE_HOST_MEM_CAPABILITIES_INTEL", usmcaps );

    clGetDeviceInfo(
        devices[deviceIndex](),
        CL_DEVICE_DEVICE_MEM_CAPABILITIES_INTEL,
        sizeof(usmcaps),
        &usmcaps,
        nullptr );
    PrintUSMCaps( "CL_DEVICE_DEVICE_MEM_CAPABILITIES_INTEL", usmcaps );

    clGetDeviceInfo(
        devices[deviceIndex](),
        CL_DEVICE_SINGLE_DEVICE_SHARED_MEM_CAPABILITIES_INTEL,
        sizeof(usmcaps),
        &usmcaps,
        nullptr );
    PrintUSMCaps( "CL_DEVICE_SINGLE_DEVICE_SHARED_MEM_CAPABILITIES_INTEL", usmcaps );

    clGetDeviceInfo(
        devices[deviceIndex](),
        CL_DEVICE_CROSS_DEVICE_SHARED_MEM_CAPABILITIES_INTEL,
        sizeof(usmcaps),
        &usmcaps,
        nullptr );
    PrintUSMCaps( "CL_DEVICE_CROSS_DEVICE_SHARED_MEM_CAPABILITIES_INTEL", usmcaps );

    clGetDeviceInfo(
        devices[deviceIndex](),
        CL_DEVICE_SHARED_SYSTEM_MEM_CAPABILITIES_INTEL,
        sizeof(usmcaps),
        &usmcaps,
        nullptr );
    PrintUSMCaps( "CL_DEVICE_SHARED_SYSTEM_MEM_CAPABILITIES_INTEL", usmcaps );

    printf("Cleaning up...\n");

    return 0;
}