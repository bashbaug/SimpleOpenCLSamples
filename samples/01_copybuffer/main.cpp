/*
// Copyright (c) 2018 Intel Corporation
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

#include "CL/cl_static.h"
#include "CL/cl2.hpp"

cl::CommandQueue commandQueue;
cl::Buffer deviceMemSrc;
cl::Buffer deviceMemDst;

size_t  gwx = 1024*1024;

static void init( void )
{
    cl_uint*    pSrc = (cl_uint*)commandQueue.enqueueMapBuffer(
        deviceMemSrc,
        CL_TRUE,
        CL_MAP_WRITE_INVALIDATE_REGION,
        0,
        gwx * sizeof(cl_uint) );

    for( size_t i = 0; i < gwx; i++ )
    {
        pSrc[i] = (cl_uint)(i);
    }

    commandQueue.enqueueUnmapMemObject(
        deviceMemSrc,
        pSrc );
}

static void go()
{
    commandQueue.enqueueCopyBuffer(
        deviceMemSrc,
        deviceMemDst,
        0,
        0,
        gwx * sizeof(cl_uint) );
}

static void checkResults()
{
    const cl_uint*  pDst = (const cl_uint*)commandQueue.enqueueMapBuffer(
        deviceMemDst,
        CL_TRUE,
        CL_MAP_READ,
        0,
        gwx * sizeof(cl_uint) );

    unsigned int    mismatches = 0;

    for( size_t i = 0; i < gwx; i++ )
    {
        if( pDst[i] != i )
        {
            if( mismatches < 16 )
            {
                fprintf(stderr, "MisMatch!  dst[%d] == %08X, want %08X\n",
                    (unsigned int)i,
                    pDst[i],
                    (unsigned int)i );
            }
            mismatches++;
        }
    }

    if( mismatches )
    {
        fprintf(stderr,"Error: Found %d mismatches / %d values!!!\n",
            mismatches,
            (unsigned int)gwx );
    }
    else
    {
        printf("Success.\n");
    }

    commandQueue.enqueueUnmapMemObject(
        deviceMemDst,
        (void*)pDst ); // TODO: Why isn't this a const void* in the API?
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
                ++i;
                if( i < argc )
                {
                    deviceIndex = strtol(argv[i], NULL, 10);
                }
            }
            else if( !strcmp( argv[i], "-p" ) )
            {
                ++i;
                if( i < argc )
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
            "Usage: copybuffer      [options]\n"
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

    cl::Context context{devices};
    commandQueue = cl::CommandQueue{context, devices[deviceIndex]};

    deviceMemSrc = cl::Buffer{
        context,
        CL_MEM_ALLOC_HOST_PTR,
        gwx * sizeof( cl_uint ) };

    deviceMemDst = cl::Buffer{
        context,
        CL_MEM_ALLOC_HOST_PTR,
        gwx * sizeof( cl_uint ) };

    init();
    go();
    checkResults();

    return 0;
}