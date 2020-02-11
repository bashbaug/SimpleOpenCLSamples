/*
// Copyright (c) 2019 Ben Ashbaugh
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
*/

#include <stdio.h>
#include <vector>

#include <CL/cl.h>
#include <CL/cl_ext.h>

#if defined(CL_NAME_VERSION_MAX_NAME_SIZE_KHR)

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

    {
        cl_version_khr version;
        clGetPlatformInfo(
            platformId,
            CL_PLATFORM_NUMERIC_VERSION_KHR,
            sizeof(version),
            &version,
            nullptr );
        printf("\tPlatform Numeric Version: %u.%u.%u\n",
            CL_VERSION_MAJOR_KHR(version),
            CL_VERSION_MINOR_KHR(version),
            CL_VERSION_PATCH_KHR(version));
    }
    {
        size_t  sizeInBytes = 0;
        clGetPlatformInfo(
            platformId,
            CL_PLATFORM_EXTENSIONS_WITH_VERSION_KHR,
            0,
            nullptr,
            &sizeInBytes );

        size_t  numExtensions = sizeInBytes / sizeof(cl_name_version_khr);
        std::vector<cl_name_version_khr>  extensions{ numExtensions };
        clGetPlatformInfo(
            platformId,
            CL_PLATFORM_EXTENSIONS_WITH_VERSION_KHR,
            sizeInBytes,
            extensions.data(),
            nullptr );

        for( auto& extension : extensions )
        {
            printf("\t\tExtension (version): %s (%u.%u.%u)\n",
                extension.name,
                CL_VERSION_MAJOR_KHR(extension.version),
                CL_VERSION_MINOR_KHR(extension.version),
                CL_VERSION_PATCH_KHR(extension.version) );
        }
    }

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

static cl_int PrintDeviceInfoSummary(
    cl_device_id* devices,
    size_t numDevices )
{
    cl_int  errorCode = CL_SUCCESS;

    cl_device_type  deviceType;
    char*           deviceName = NULL;
    char*           deviceVendor = NULL;
    char*           deviceVersion = NULL;
    char*           deviceOpenCLCVersion = NULL;
    char*           driverVersion = NULL;

    size_t  i = 0;
    for( i = 0; i < numDevices; i++ )
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
            CL_DEVICE_OPENCL_C_VERSION,
            deviceOpenCLCVersion );
        errorCode |= AllocateAndGetDeviceInfoString(
            devices[i],
            CL_DRIVER_VERSION,
            driverVersion );

        if( errorCode == CL_SUCCESS )
        {
            printf("Device[%d]:\n", (int)i );

            switch( deviceType )
            {
            case CL_DEVICE_TYPE_DEFAULT:    printf("\tType:           %s\n", "DEFAULT" );      break;
            case CL_DEVICE_TYPE_CPU:        printf("\tType:           %s\n", "CPU" );          break;
            case CL_DEVICE_TYPE_GPU:        printf("\tType:           %s\n", "GPU" );          break;
            case CL_DEVICE_TYPE_ACCELERATOR:printf("\tType:           %s\n", "ACCELERATOR" );  break;
            default:                        printf("\tType:           %s\n", "***UNKNOWN***" );break;
            }

            printf("\tName:             %s\n", deviceName );
            printf("\tVendor:           %s\n", deviceVendor );
            printf("\tDevice Version:   %s\n", deviceVersion );
            printf("\tOpenCL C Version: %s\n", deviceOpenCLCVersion );
            printf("\tDriver Version:   %s\n", driverVersion );

            {
                cl_version_khr version;
                clGetDeviceInfo(
                    devices[i],
                    CL_DEVICE_NUMERIC_VERSION_KHR,
                    sizeof(version),
                    &version,
                    nullptr );
                printf("\tDevice Numeric Version:   %u.%u.%u\n",
                    CL_VERSION_MAJOR_KHR(version),
                    CL_VERSION_MINOR_KHR(version),
                    CL_VERSION_PATCH_KHR(version));
            }
            {
                cl_version_khr version;
                clGetDeviceInfo(
                    devices[i],
                    CL_DEVICE_OPENCL_C_NUMERIC_VERSION_KHR,
                    sizeof(version),
                    &version,
                    nullptr );
                printf("\tOpenCL C Numeric Version: %u.%u.%u\n",
                    CL_VERSION_MAJOR_KHR(version),
                    CL_VERSION_MINOR_KHR(version),
                    CL_VERSION_PATCH_KHR(version));
            }
            {
                size_t  sizeInBytes = 0;
                clGetDeviceInfo(
                    devices[i],
                    CL_DEVICE_EXTENSIONS_WITH_VERSION_KHR,
                    0,
                    nullptr,
                    &sizeInBytes );

                size_t  numExtensions = sizeInBytes / sizeof(cl_name_version_khr);
                std::vector<cl_name_version_khr>  extensions{ numExtensions };
                clGetDeviceInfo(
                    devices[i],
                    CL_DEVICE_EXTENSIONS_WITH_VERSION_KHR,
                    sizeInBytes,
                    extensions.data(),
                    nullptr );

                for( auto& extension : extensions )
                {
                    printf("\t\tExtension (version): %s (%u.%u.%u)\n",
                        extension.name,
                        CL_VERSION_MAJOR_KHR(extension.version),
                        CL_VERSION_MINOR_KHR(extension.version),
                        CL_VERSION_PATCH_KHR(extension.version) );
                }
            }
            {
                size_t  sizeInBytes = 0;
                clGetDeviceInfo(
                    devices[i],
                    CL_DEVICE_ILS_WITH_VERSION_KHR,
                    0,
                    nullptr,
                    &sizeInBytes );

                size_t  numILs = sizeInBytes / sizeof(cl_name_version_khr);
                std::vector<cl_name_version_khr> ils{ numILs };
                clGetDeviceInfo(
                    devices[i],
                    CL_DEVICE_ILS_WITH_VERSION_KHR,
                    sizeInBytes,
                    ils.data(),
                    nullptr );

                for( auto& il : ils )
                {
                    printf("\t\tIL (version): %s (%u.%u.%u)\n",
                        il.name,
                        CL_VERSION_MAJOR_KHR(il.version),
                        CL_VERSION_MINOR_KHR(il.version),
                        CL_VERSION_PATCH_KHR(il.version) );
                }
            }
            {
                size_t  sizeInBytes = 0;
                clGetDeviceInfo(
                    devices[i],
                    CL_DEVICE_BUILT_IN_KERNELS_WITH_VERSION_KHR,
                    0,
                    nullptr,
                    &sizeInBytes );

                size_t  numBuiltInKernels = sizeInBytes / sizeof(cl_name_version_khr);
                std::vector<cl_name_version_khr> builtInKernels{ numBuiltInKernels };
                clGetDeviceInfo(
                    devices[i],
                    CL_DEVICE_BUILT_IN_KERNELS_WITH_VERSION_KHR,
                    sizeInBytes,
                    builtInKernels.data(),
                    nullptr );

                for( auto& builtInFunction : builtInKernels )
                {
                    printf("\t\tBuilt-in Function (version): %s (%u.%u.%u)\n",
                        builtInFunction.name,
                        CL_VERSION_MAJOR_KHR(builtInFunction.version),
                        CL_VERSION_MINOR_KHR(builtInFunction.version),
                        CL_VERSION_PATCH_KHR(builtInFunction.version) );
                }
            }
        }
        else
        {
            fprintf(stderr, "Error getting device info for device %d.\n", (int)i );
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
    bool printUsage = false;

    int i = 0;

    if( argc < 1 )
    {
        printUsage = true;
    }
    else
    {
        for( i = 1; i < argc; i++ )
        {
            {
                printUsage = true;
            }
        }
    }
    if( printUsage )
    {
        fprintf(stderr,
            "Usage: newqueries  [options]\n"
            "Options:\n"
            );

        return -1;
    }

    cl_uint numPlatforms = 0;
    clGetPlatformIDs( 0, NULL, &numPlatforms );
    printf( "Enumerated %u platforms.\n\n", numPlatforms );

    std::vector<cl_platform_id> platforms;
    platforms.resize( numPlatforms );
    clGetPlatformIDs( numPlatforms, platforms.data(), NULL );

    for( auto& platform : platforms )
    {
        printf( "Platform:\n" );
        PrintPlatformInfoSummary( platform );

        cl_uint numDevices = 0;
        clGetDeviceIDs( platform, CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices );

        std::vector<cl_device_id> devices;
        devices.resize( numDevices );
        clGetDeviceIDs( platform, CL_DEVICE_TYPE_ALL, numDevices, devices.data(), NULL );

        PrintDeviceInfoSummary( devices.data(), numDevices );
        printf( "\n" );
    }

    printf( "Done.\n" );

    return 0;
}

#else

#pragma message("newqueries: cl_khr_extended_versioning APIS not found.  Please update your OpenCL headers.")

int main()
{
    printf("newqueries: cl_khr_extended_versioning APIS not found.  Please update your OpenCL headers.\n");
    return 0;
};

#endif
