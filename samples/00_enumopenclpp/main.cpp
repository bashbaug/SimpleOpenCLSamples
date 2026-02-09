/*
// Copyright (c) 2019-2026 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <stdio.h>
#include <vector>
#include <popl/popl.hpp>

#include <CL/opencl.hpp>

static cl_int PrintPlatformInfoSummary(
    cl::Platform platform )
{
    printf("\tName:             %s\n", platform.getInfo<CL_PLATFORM_NAME>().c_str() );
    printf("\tVendor:           %s\n", platform.getInfo<CL_PLATFORM_VENDOR>().c_str() );
    printf("\tPlatform Version: %s\n", platform.getInfo<CL_PLATFORM_VERSION>().c_str() );

    return CL_SUCCESS;
}

static void PrintDeviceType(
    const char* label,
    cl_device_type type )
{
    printf("%s%s%s%s%s%s\n",
        label,
        ( type & CL_DEVICE_TYPE_DEFAULT     ) ? "DEFAULT "      : "",
        ( type & CL_DEVICE_TYPE_CPU         ) ? "CPU "          : "",
        ( type & CL_DEVICE_TYPE_GPU         ) ? "GPU "          : "",
        ( type & CL_DEVICE_TYPE_ACCELERATOR ) ? "ACCELERATOR "  : "",
        ( type & CL_DEVICE_TYPE_CUSTOM      ) ? "CUSTOM "       : "");
}

static cl_int PrintDeviceInfoSummary(
    const std::vector<cl::Device>& devices )
{
    for( size_t i = 0; i < devices.size(); i++ )
    {
        printf("Device[%zu]:\n", i );

        cl_device_type  deviceType = devices[i].getInfo<CL_DEVICE_TYPE>();
        PrintDeviceType("\tType:           ", deviceType);

        printf("\tName:           %s\n", devices[i].getInfo<CL_DEVICE_NAME>().c_str() );
        printf("\tVendor:         %s\n", devices[i].getInfo<CL_DEVICE_VENDOR>().c_str() );
        printf("\tDevice Version: %s\n", devices[i].getInfo<CL_DEVICE_VERSION>().c_str() );
        printf("\tDriver Version: %s\n", devices[i].getInfo<CL_DRIVER_VERSION>().c_str() );
    }

    return CL_SUCCESS;
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
                "Usage: enumopenclpp [options]\n"
                "%s", op.help().c_str());
            return -1;
        }
    }

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    for( size_t i = 0; i < platforms.size(); i++ )
    {
        printf( "Platform[%zu]:\n", i );
        PrintPlatformInfoSummary( platforms[i] );

        std::vector<cl::Device> devices;
        platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &devices);

        PrintDeviceInfoSummary( devices );
        printf( "\n" );
    }

    printf( "Done.\n" );

    return 0;
}
