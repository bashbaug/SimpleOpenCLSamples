/*
// Copyright (c) 2019-2025 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <stdio.h>
#include <vector>
#include <popl/popl.hpp>

#include <CL/cl.h>

static cl_int AllocateAndGetPlatformInfoString(
    cl_platform_id platformId,
    cl_platform_info param_name,
    char*& param_value )
{
    cl_int  errorCode = CL_SUCCESS;
    size_t  size = 0;

    if( errorCode == CL_SUCCESS )
    {
        if( param_value != NULL )
        {
            delete [] param_value;
            param_value = NULL;
        }
    }

    if( errorCode == CL_SUCCESS )
    {
        errorCode = clGetPlatformInfo(
            platformId,
            param_name,
            0,
            NULL,
            &size );
    }

    if( errorCode == CL_SUCCESS )
    {
        if( size != 0 )
        {
            param_value = new char[ size ];
            if( param_value == NULL )
            {
                errorCode = CL_OUT_OF_HOST_MEMORY;
            }
        }
    }

    if( errorCode == CL_SUCCESS )
    {
        errorCode = clGetPlatformInfo(
            platformId,
            param_name,
            size,
            param_value,
            NULL );
    }

    if( errorCode != CL_SUCCESS )
    {
        delete [] param_value;
        param_value = NULL;
    }

    return errorCode;
}

static cl_int PrintPlatformInfoSummary(
    cl_platform_id platformId )
{
    cl_int  errorCode = CL_SUCCESS;

    char*           platformName = NULL;
    char*           platformVendor = NULL;
    char*           platformVersion = NULL;

    errorCode |= AllocateAndGetPlatformInfoString(
        platformId,
        CL_PLATFORM_NAME,
        platformName );
    errorCode |= AllocateAndGetPlatformInfoString(
        platformId,
        CL_PLATFORM_VENDOR,
        platformVendor );
    errorCode |= AllocateAndGetPlatformInfoString(
        platformId,
        CL_PLATFORM_VERSION,
        platformVersion );

    printf("\tName:           %s\n", platformName );
    printf("\tVendor:         %s\n", platformVendor );
    printf("\tDriver Version: %s\n", platformVersion );

    delete [] platformName;
    delete [] platformVendor;
    delete [] platformVersion;

    platformName = NULL;
    platformVendor = NULL;
    platformVersion = NULL;

    return errorCode;
}

static cl_int AllocateAndGetDeviceInfoString(
    cl_device_id    device,
    cl_device_info  param_name,
    char*&          param_value )
{
    cl_int  errorCode = CL_SUCCESS;
    size_t  size = 0;

    if( errorCode == CL_SUCCESS )
    {
        if( param_value != NULL )
        {
            delete [] param_value;
            param_value = NULL;
        }
    }

    if( errorCode == CL_SUCCESS )
    {
        errorCode = clGetDeviceInfo(
            device,
            param_name,
            0,
            NULL,
            &size );
    }

    if( errorCode == CL_SUCCESS )
    {
        if( size != 0 )
        {
            param_value = new char[ size ];
            if( param_value == NULL )
            {
                errorCode = CL_OUT_OF_HOST_MEMORY;
            }
        }
    }

    if( errorCode == CL_SUCCESS )
    {
        errorCode = clGetDeviceInfo(
            device,
            param_name,
            size,
            param_value,
            NULL );
    }

    if( errorCode != CL_SUCCESS )
    {
        delete [] param_value;
        param_value = NULL;
    }

    return errorCode;
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
    cl_device_id* devices,
    cl_uint numDevices )
{
    cl_int  errorCode = CL_SUCCESS;

    cl_device_type  deviceType;
    char*           deviceName = NULL;
    char*           deviceVendor = NULL;
    char*           deviceVersion = NULL;
    char*           driverVersion = NULL;

    for( cl_uint i = 0; i < numDevices; i++ )
    {
        errorCode |= clGetDeviceInfo(
            devices[i],
            CL_DEVICE_TYPE,
            sizeof( deviceType ),
            &deviceType,
            NULL );
        errorCode |= AllocateAndGetDeviceInfoString(
            devices[i],
            CL_DEVICE_NAME,
            deviceName );
        errorCode |= AllocateAndGetDeviceInfoString(
            devices[i],
            CL_DEVICE_VENDOR,
            deviceVendor );
        errorCode |= AllocateAndGetDeviceInfoString(
            devices[i],
            CL_DEVICE_VERSION,
            deviceVersion );
        errorCode |= AllocateAndGetDeviceInfoString(
            devices[i],
            CL_DRIVER_VERSION,
            driverVersion );

        if( errorCode == CL_SUCCESS )
        {
            printf("Device[%u]:\n", i );

            PrintDeviceType("\tType:           ", deviceType);

            printf("\tName:           %s\n", deviceName );
            printf("\tVendor:         %s\n", deviceVendor );
            printf("\tDevice Version: %s\n", deviceVersion );
            printf("\tDriver Version: %s\n", driverVersion );
        }
        else
        {
            fprintf(stderr, "Error getting device info for device %u.\n", i );
        }

        delete [] deviceName;
        delete [] deviceVendor;
        delete [] deviceVersion;
        delete [] driverVersion;

        deviceName = NULL;
        deviceVendor = NULL;
        deviceVersion = NULL;
        driverVersion = NULL;
    }

    return errorCode;
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
                "Usage: enumopencl [options]\n"
                "%s", op.help().c_str());
            return -1;
        }
    }

    cl_uint numPlatforms = 0;
    clGetPlatformIDs( 0, NULL, &numPlatforms );
    printf( "Enumerated %u platforms.\n\n", numPlatforms );

    std::vector<cl_platform_id> platforms;
    platforms.resize( numPlatforms );
    clGetPlatformIDs( numPlatforms, platforms.data(), NULL );

    for( cl_uint i = 0; i < numPlatforms; i++ )
    {
        printf( "Platform[%u]:\n", i );
        PrintPlatformInfoSummary( platforms[i] );

        cl_uint numDevices = 0;
        clGetDeviceIDs( platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices );

        std::vector<cl_device_id> devices;
        devices.resize( numDevices );
        clGetDeviceIDs( platforms[i], CL_DEVICE_TYPE_ALL, numDevices, devices.data(), NULL );

        PrintDeviceInfoSummary( devices.data(), numDevices );
        printf( "\n" );
    }

    printf( "Done.\n" );

    return 0;
}
