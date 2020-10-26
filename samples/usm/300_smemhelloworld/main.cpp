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
#include "libusm.h"

cl::CommandQueue commandQueue;
cl::Kernel kernel;
cl_uint* s_src;
cl_uint* s_dst;

size_t  gwx = 1024*1024;

static const char kernelString[] = R"CLC(
kernel void CopyBuffer( global uint* dst, global uint* src )
{
    uint id = get_global_id(0);
    dst[id] = src[id];
}
)CLC";

static void init( void )
{
    for( size_t i = 0; i < gwx; i++ )
    {
        s_src[i] = (cl_uint)(i);
    }

    memset( s_dst, 0, gwx * sizeof(cl_uint) );
}

static void go()
{
    clSetKernelArgMemPointerINTEL(
        kernel(),
        0,
        s_dst );
    clSetKernelArgMemPointerINTEL(
        kernel(),
        1,
        s_src );

    commandQueue.enqueueNDRangeKernel(
        kernel,
        cl::NullRange,
        cl::NDRange{gwx} );
}

static void checkResults()
{
    commandQueue.finish();

    unsigned int    mismatches = 0;

    for( size_t i = 0; i < gwx; i++ )
    {
        if( s_dst[i] != i )
        {
            if( mismatches < 16 )
            {
                fprintf(stderr, "MisMatch!  dst[%d] == %08X, want %08X\n",
                    (unsigned int)i,
                    s_dst[i],
                    (unsigned int)i );
            }
            mismatches++;
        }
    }

    if( mismatches )
    {
        fprintf(stderr, "Error: Found %d mismatches / %d values!!!\n",
            mismatches,
            (unsigned int)gwx );
    }
    else
    {
        printf("Success.\n");
    }
}

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
            "Usage: smemhelloworld  [options]\n"
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
    libusm::initialize(platforms[platformIndex]());

    std::vector<cl::Device> devices;
    platforms[platformIndex].getDevices(CL_DEVICE_TYPE_ALL, &devices);

    printf("Running on device: %s\n",
        devices[deviceIndex].getInfo<CL_DEVICE_NAME>().c_str() );

    cl::Context context{devices[deviceIndex]};
    commandQueue = cl::CommandQueue{context, devices[deviceIndex]};

    cl::Program program{ context, kernelString };
    program.build();
#if 0
    for( auto& device : program.getInfo<CL_PROGRAM_DEVICES>() )
    {
        printf("Program build log for device %s:\n",
            device.getInfo<CL_DEVICE_NAME>().c_str() );
        printf("%s\n",
            program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device).c_str() );
    }
#endif
    kernel = cl::Kernel{ program, "CopyBuffer" };

    s_src = (cl_uint*)clSharedMemAllocINTEL(
        context(),
        devices[deviceIndex](),
        nullptr,
        gwx * sizeof(cl_uint),
        0,
        nullptr );
    s_dst = (cl_uint*)clSharedMemAllocINTEL(
        context(),
        devices[deviceIndex](),
        nullptr,
        gwx * sizeof(cl_uint),
        0,
        nullptr );

    if( s_src && s_dst )
    {
        init();
        go();
        checkResults();
    }
    else
    {
        printf("Allocation failed - does this device support Unified Shared Memory?\n");
    }

    printf("Cleaning up...\n");

    clMemFreeINTEL(
        context(),
        s_src );
    clMemFreeINTEL(
        context(),
        s_dst );

    return 0;
}