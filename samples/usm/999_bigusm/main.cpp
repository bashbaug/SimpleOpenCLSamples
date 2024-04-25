/*
// Copyright (c) 2020 Ben Ashbaugh
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

#include <CL/opencl.hpp>

constexpr size_t BIG_ALLOC = 5ULL * 1024 * 1024 * 1024;    // 5GB

#define CL_MEM_FLAGS_INTEL 0x10001
#define CL_MEM_ALLOW_UNRESTRICTED_SIZE_INTEL (1 << 23)

const cl_mem_properties_intel big_alloc_properties[] = {
    CL_MEM_FLAGS_INTEL, CL_MEM_ALLOW_UNRESTRICTED_SIZE_INTEL,
    0,
};

int main(
    int argc,
    char** argv )
{
    bool printUsage = false;
    int platformIndex = 0;
    int deviceIndex = 0;

    if( argc < 1 )
    {
        printUsage = true;
    }
    else
    {
        for( size_t i = 1; i < argc; i++ )
        {
            if( !strcmp( argv[i], "-d" ) )
            {
                if( ++i < argc )
                {
                    deviceIndex = strtol(argv[i], NULL, 10);
                }
            }
            else if( !strcmp( argv[i], "-p" ) )
            {
                if( ++i < argc )
                {
                    platformIndex = strtol(argv[i], NULL, 10);
                }
            }
            else
            {
                printUsage = true;
            }
        }
    }
    if( printUsage )
    {
        fprintf(stderr,
            "Usage: bigusm [options]\n"
            "Options:\n"
            "      -d: Device Index (default = 0)\n"
            "      -p: Platform Index (default = 0)\n"
            );

        return -1;
    }

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    printf("Running on platform: %s\n",
        platforms[platformIndex].getInfo<CL_PLATFORM_NAME>().c_str() );

    std::vector<cl::Device> devices;
    platforms[platformIndex].getDevices(CL_DEVICE_TYPE_ALL, &devices);

    printf("Running on device: %s\n",
        devices[deviceIndex].getInfo<CL_DEVICE_NAME>().c_str() );

    cl::Context context{devices[deviceIndex]};

    cl_device_unified_shared_memory_capabilities_intel usmcaps = 0;
    cl_int errCode;
    
    errCode = clGetDeviceInfo(
        devices[deviceIndex](),
        CL_DEVICE_HOST_MEM_CAPABILITIES_INTEL,
        sizeof(usmcaps),
        &usmcaps,
        nullptr );
    if( errCode == CL_SUCCESS && usmcaps != 0 )
    {
        printf("\nTesting Host Allocations:\n");

        void* ptr0 = clHostMemAllocINTEL(
            context(),
            nullptr,
            BIG_ALLOC,
            0,
            nullptr );
        printf("Allocating %zu bytes with no properties returned: %p\n", BIG_ALLOC, ptr0);
        clMemFreeINTEL(
            context(),
            ptr0 );

        void* ptr1 = clHostMemAllocINTEL(
            context(),
            big_alloc_properties,
            BIG_ALLOC,
            0,
            nullptr );
        printf("Allocating %zu bytes with properties returned %p\n", BIG_ALLOC, ptr1);
        clMemFreeINTEL(
            context(),
            ptr1 );

        printf("done!\n");
    }
    else
    {
        printf("\nThis device does not support HOST allocations.\n");
    }

    errCode = clGetDeviceInfo(
        devices[deviceIndex](),
        CL_DEVICE_DEVICE_MEM_CAPABILITIES_INTEL,
        sizeof(usmcaps),
        &usmcaps,
        nullptr );
    if( errCode == CL_SUCCESS && usmcaps != 0 )
    {
        printf("\nTesting Device Allocations:\n");

        printf("Associated Device is: %p (%s)\n",
            devices[deviceIndex](),
            devices[deviceIndex].getInfo<CL_DEVICE_NAME>().c_str());

        void* ptr0 = clDeviceMemAllocINTEL(
            context(),
            devices[deviceIndex](),
            nullptr,
            BIG_ALLOC,
            0,
            nullptr );
        printf("Allocating %zu bytes with no properties returned: %p\n", BIG_ALLOC, ptr0);
        clMemFreeINTEL(
            context(),
            ptr0 );

        void* ptr1 = clDeviceMemAllocINTEL(
            context(),
            devices[deviceIndex](),
            big_alloc_properties,
            BIG_ALLOC,
            0,
            nullptr );
        printf("Allocating %zu bytes with properties returned %p\n", BIG_ALLOC, ptr1);
        clMemFreeINTEL(
            context(),
            ptr1 );

        printf("done!\n");
    }
    else
    {
        printf("\nThis device does not support DEVICE allocations.\n");
    }

    errCode = clGetDeviceInfo(
        devices[deviceIndex](),
        CL_DEVICE_SINGLE_DEVICE_SHARED_MEM_CAPABILITIES_INTEL,
        sizeof(usmcaps),
        &usmcaps,
        nullptr );
    if( errCode == CL_SUCCESS && usmcaps != 0 )
    {
        printf("\nTesting Shared Allocations:\n");

        printf("Associated Device is: %p (%s)\n",
            devices[deviceIndex](),
            devices[deviceIndex].getInfo<CL_DEVICE_NAME>().c_str());

        void* ptr0 = clSharedMemAllocINTEL(
            context(),
            devices[deviceIndex](),
            nullptr,
            BIG_ALLOC,
            0,
            nullptr );
        printf("Allocating %zu bytes with no properties returned: %p\n", BIG_ALLOC, ptr0);
        clMemFreeINTEL(
            context(),
            ptr0 );

        void* ptr1 = clSharedMemAllocINTEL(
            context(),
            devices[deviceIndex](),
            big_alloc_properties,
            BIG_ALLOC,
            0,
            nullptr );
        printf("Allocating %zu bytes with properties returned %p\n", BIG_ALLOC, ptr1);
        clMemFreeINTEL(
            context(),
            ptr1 );

        printf("done!\n");
    }
    else
    {
        printf("\nThis device does not support SHARED allocations.\n");
    }

    return 0;
}