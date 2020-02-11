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

#include <CL/cl2.hpp>

#if defined(CL_NAME_VERSION_MAX_NAME_SIZE_KHR)

static cl_int PrintPlatformInfoSummary(
    cl::Platform platform )
{
    printf("\tName:           %s\n", platform.getInfo<CL_PLATFORM_NAME>().c_str() );
    printf("\tVendor:         %s\n", platform.getInfo<CL_PLATFORM_VENDOR>().c_str() );
    printf("\tDriver Version: %s\n", platform.getInfo<CL_PLATFORM_VERSION>().c_str() );

    cl_version_khr platformVersion = 
        platform.getInfo<CL_PLATFORM_NUMERIC_VERSION_KHR>();
    printf("\tPlatform Numeric Version: %u.%u.%u\n",
        CL_VERSION_MAJOR_KHR(platformVersion),
        CL_VERSION_MINOR_KHR(platformVersion),
        CL_VERSION_PATCH_KHR(platformVersion));

    std::vector<cl_name_version_khr>  extensions =
        platform.getInfo<CL_PLATFORM_EXTENSIONS_WITH_VERSION_KHR>();
    for( auto& extension : extensions )
    {
        printf("\t\tExtension (version): %s (%u.%u.%u)\n",
            extension.name,
            CL_VERSION_MAJOR_KHR(extension.version),
            CL_VERSION_MINOR_KHR(extension.version),
            CL_VERSION_PATCH_KHR(extension.version) );
    }

    return CL_SUCCESS;
}

static cl_int PrintDeviceInfoSummary(
    const std::vector<cl::Device> devices )
{
    size_t  i = 0;
    for( i = 0; i < devices.size(); i++ )
    {
        printf("Device[%d]:\n", (int)i );

        cl_device_type  deviceType = devices[i].getInfo<CL_DEVICE_TYPE>();
        switch( deviceType )
        {
        case CL_DEVICE_TYPE_DEFAULT:    printf("\tType:             %s\n", "DEFAULT" );      break;
        case CL_DEVICE_TYPE_CPU:        printf("\tType:             %s\n", "CPU" );          break;
        case CL_DEVICE_TYPE_GPU:        printf("\tType:             %s\n", "GPU" );          break;
        case CL_DEVICE_TYPE_ACCELERATOR:printf("\tType:             %s\n", "ACCELERATOR" );  break;
        default:                        printf("\tType:             %s\n", "***UNKNOWN***" );break;
        }

        printf("\tName:             %s\n", devices[i].getInfo<CL_DEVICE_NAME>().c_str() );
        printf("\tVendor:           %s\n", devices[i].getInfo<CL_DEVICE_VENDOR>().c_str() );
        printf("\tDevice Version:   %s\n", devices[i].getInfo<CL_DEVICE_VERSION>().c_str() );
        printf("\tOpenCL C Version: %s\n", devices[i].getInfo<CL_DEVICE_OPENCL_C_VERSION>().c_str() );
        printf("\tDriver Version:   %s\n", devices[i].getInfo<CL_DRIVER_VERSION>().c_str() );

        cl_version_khr deviceVersion = 
            devices[i].getInfo<CL_DEVICE_NUMERIC_VERSION_KHR>();
        printf("\tDevice Numeric Version:   %u.%u.%u\n",
            CL_VERSION_MAJOR_KHR(deviceVersion),
            CL_VERSION_MINOR_KHR(deviceVersion),
            CL_VERSION_PATCH_KHR(deviceVersion));

        cl_version_khr deviceOpenCLCVersion = 
            devices[i].getInfo<CL_DEVICE_OPENCL_C_NUMERIC_VERSION_KHR>();
        printf("\tOpenCL C Numeric Version: %u.%u.%u\n",
            CL_VERSION_MAJOR_KHR(deviceOpenCLCVersion),
            CL_VERSION_MINOR_KHR(deviceOpenCLCVersion),
            CL_VERSION_PATCH_KHR(deviceOpenCLCVersion));

        std::vector<cl_name_version_khr> extensions =
            devices[i].getInfo<CL_DEVICE_EXTENSIONS_WITH_VERSION_KHR>();
        for( auto& extension : extensions )
        {
            printf("\t\tExtension (version): %s (%u.%u.%u)\n",
                extension.name,
                CL_VERSION_MAJOR_KHR(extension.version),
                CL_VERSION_MINOR_KHR(extension.version),
                CL_VERSION_PATCH_KHR(extension.version) );
        }

        std::vector<cl_name_version_khr> ils =
            devices[i].getInfo<CL_DEVICE_ILS_WITH_VERSION_KHR>();
        for( auto& il : ils )
        {
            printf("\t\tIL (version): %s (%u.%u.%u)\n",
                il.name,
                CL_VERSION_MAJOR_KHR(il.version),
                CL_VERSION_MINOR_KHR(il.version),
                CL_VERSION_PATCH_KHR(il.version) );
        }

        std::vector<cl_name_version_khr> builtInKernels =
            devices[i].getInfo<CL_DEVICE_BUILT_IN_KERNELS_WITH_VERSION_KHR>();
        for( auto& builtInFunction : builtInKernels )
        {
            printf("\t\tBuilt-in Function (version): %s (%u.%u.%u)\n",
                builtInFunction.name,
                CL_VERSION_MAJOR_KHR(builtInFunction.version),
                CL_VERSION_MINOR_KHR(builtInFunction.version),
                CL_VERSION_PATCH_KHR(builtInFunction.version) );
        }
    }

    return CL_SUCCESS;
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

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    for( auto& platform : platforms )
    {
        printf( "Platform:\n" );
        PrintPlatformInfoSummary( platform );

        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

        PrintDeviceInfoSummary( devices );
        printf( "\n" );
    }

    printf( "Done.\n" );

    return 0;
}

#else

#pragma message("newqueriespp: cl_khr_extended_versioning APIS not found.  Please update your OpenCL headers.")

int main()
{
    printf("newqueriespp: cl_khr_extended_versioning APIS not found.  Please update your OpenCL headers.\n");
    return 0;
};

#endif
