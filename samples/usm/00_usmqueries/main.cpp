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
    printf("\t%s: %s%s%s%s\n",
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
    {
        popl::OptionParser op("Supported Options");

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

    for( size_t i = 0; i < platforms.size(); i++ )
    {
        printf( "Platform[%zu]: %s\n",
            i,
            platforms[i].getInfo<CL_PLATFORM_NAME>().c_str());

        std::vector<cl::Device> devices;
        platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &devices);

        for( size_t d = 0; d < devices.size(); d++ )
        {
            printf("\tDevice[%zu]: %s\n",
                d,
                devices[d].getInfo<CL_DEVICE_NAME>().c_str());

            cl_device_unified_shared_memory_capabilities_intel usmcaps = 0;

            usmcaps = devices[d].getInfo<CL_DEVICE_HOST_MEM_CAPABILITIES_INTEL>();
            PrintUSMCaps( "CL_DEVICE_HOST_MEM_CAPABILITIES_INTEL", usmcaps );

            usmcaps = devices[d].getInfo<CL_DEVICE_DEVICE_MEM_CAPABILITIES_INTEL>();
            PrintUSMCaps( "CL_DEVICE_DEVICE_MEM_CAPABILITIES_INTEL", usmcaps );

            usmcaps = devices[d].getInfo<CL_DEVICE_SINGLE_DEVICE_SHARED_MEM_CAPABILITIES_INTEL>();
            PrintUSMCaps( "CL_DEVICE_SINGLE_DEVICE_SHARED_MEM_CAPABILITIES_INTEL", usmcaps );

            usmcaps = devices[d].getInfo<CL_DEVICE_CROSS_DEVICE_SHARED_MEM_CAPABILITIES_INTEL>();
            PrintUSMCaps( "CL_DEVICE_CROSS_DEVICE_SHARED_MEM_CAPABILITIES_INTEL", usmcaps );

            usmcaps = devices[d].getInfo<CL_DEVICE_SHARED_SYSTEM_MEM_CAPABILITIES_INTEL>();
            PrintUSMCaps( "CL_DEVICE_SHARED_SYSTEM_MEM_CAPABILITIES_INTEL", usmcaps );

            printf( "\n" );
        }
    }

    printf("Cleaning up...\n");

    return 0;
}