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

#include <CL/cl2.hpp>
#include "libusm.h"

cl::CommandQueue commandQueue;
cl::Kernel kernel;

cl_uint numNodes = 4;

struct Node {
    Node() :
        pNext( nullptr ),
        Num( 0xDEADBEEF ) {}

    Node*   pNext;
    cl_uint Num;
};

Node*   d_head;

static const char kernelString[] = R"CLC(
struct Node {
    global struct Node* pNext;
    uint Num;
};
kernel void WalkLinkedList( global struct Node* pHead )
{
    uint count = 0;
    while( pHead )
    {
        ++count;
        pHead->Num = pHead->Num * 2 + 1;
        pHead = pHead->pNext;
    }
}
)CLC";

static void init( cl::Context& context, cl::Device& device )
{
    Node*   d_cur = nullptr;
    Node    h_cur;

    for( cl_uint i = 0; i < numNodes; i++ )
    {
        if( i == 0 )
        {
            d_head = (Node*)clDeviceMemAllocINTEL(
                context(),
                device(),
                nullptr,
                sizeof(Node),
                0,
                nullptr );
            d_cur = d_head;
        }

        h_cur.Num = i * 2;

        if( i != numNodes - 1 )
        {
            h_cur.pNext = (Node*)clDeviceMemAllocINTEL(
                context(),
                device(),
                nullptr,
                sizeof(Node),
                0,
                nullptr );
        }
        else
        {
            h_cur.pNext = nullptr;
        }

        clEnqueueMemcpyINTEL(
            commandQueue(),
            CL_TRUE,
            d_cur,
            &h_cur,
            sizeof(Node),
            0,
            nullptr,
            nullptr );

        d_cur = h_cur.pNext;
    }
}

static void go()
{
    clSetKernelArgMemPointerINTEL(
        kernel(),
        0,
        d_head );

    commandQueue.enqueueNDRangeKernel(
        kernel,
        cl::NullRange,
        cl::NDRange{1} );
}

static void checkResults()
{
    const Node* d_cur = d_head;
    Node    h_cur;

    unsigned int    mismatches = 0;
    for( cl_uint i = 0; i < numNodes; i++ )
    {
        clEnqueueMemcpyINTEL(
            commandQueue(),
            CL_TRUE,
            &h_cur,
            d_cur,
            sizeof(Node),
            0,
            nullptr,
            nullptr );

        const cl_uint want = i * 4 + 1;
        if( h_cur.Num != want )
        {
            if( mismatches < 16 )
            {
                fprintf(stderr, "MisMatch at node %u!  got %08X, want %08X\n",
                    i,
                    h_cur.Num,
                    want );
            }
            mismatches++;
        }

        d_cur = h_cur.pNext;
    }

    if( mismatches )
    {
        fprintf(stderr, "Error: Found %d mismatches / %d values!!!\n",
            mismatches,
            numNodes );
    }
    else
    {
        printf("Success.\n");
    }
}

void cleanup( cl::Context& context )
{
    Node*   d_cur = d_head;
    Node    h_cur;

    while( d_cur != nullptr )
    {
        clEnqueueMemcpyINTEL(
            commandQueue(),
            CL_TRUE,
            &h_cur,
            d_cur,
            sizeof(Node),
            0,
            nullptr,
            nullptr );

        clMemFreeINTEL(
            context(),
            d_cur );

        d_cur = h_cur.pNext;
    }

    d_head = nullptr;
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
            else if (!strcmp(argv[i], "-n"))
            {
                ++i;
                if (i < argc)
                {
                    numNodes = strtol(argv[i], NULL, 10);
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
            "Usage: dmemlinkedlist  [options]\n"
            "Options:\n"
            "      -d: Device Index (default = 0)\n"
            "      -p: Platform Index (default = 0)\n"
            "      -n: Number of Linked List nodes\n"
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
    kernel = cl::Kernel{ program, "WalkLinkedList" };
    cl_bool enableIndirectAccess = CL_TRUE;
    clSetKernelExecInfo(
        kernel(),
        CL_KERNEL_EXEC_INFO_INDIRECT_DEVICE_ACCESS_INTEL,
        sizeof(enableIndirectAccess),
        &enableIndirectAccess );

    init( context, devices[deviceIndex] );
    go();
    checkResults();

    printf("Cleaning up...\n");
    cleanup( context );

    return 0;
}