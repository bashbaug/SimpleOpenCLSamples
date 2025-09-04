/*
// Copyright (c) 2024-2025 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <popl/popl.hpp>

#include <CL/opencl.hpp>

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

Node*   h_head;
std::vector<void*>  nodes;

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
    Node*   h_cur;

    for( cl_uint i = 0; i < numNodes; i++ )
    {
        if( i == 0 )
        {
            h_head = (Node*)clSVMAlloc(
                context(),
                CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER,
                sizeof(Node),
                0 );
            nodes.push_back( h_head );
            h_cur = h_head;
        }

        if( h_cur != nullptr )
        {
            h_cur->Num = i * 2;

            if( i != numNodes - 1 )
            {
                h_cur->pNext = (Node*)clSVMAlloc(
                    context(),
                    CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER,
                    sizeof(Node),
                    0 );
                nodes.push_back( h_cur->pNext );
            }
            else
            {
                h_cur->pNext = nullptr;
            }

            h_cur = h_cur->pNext;
        }
    }
}

static void go()
{
    kernel.setArg(0, h_head );
    kernel.setSVMPointers(nodes);
    commandQueue.enqueueNDRangeKernel(
        kernel,
        cl::NullRange,
        cl::NDRange{1} );
}

static void checkResults()
{
    commandQueue.finish();

    const Node* h_cur = h_head;

    unsigned int    mismatches = 0;
    for( cl_uint i = 0; i < numNodes; i++ )
    {
        const cl_uint want = i * 4 + 1;
        if( h_cur == nullptr )
        {
            if( mismatches < 16 )
            {
                fprintf(stderr, "Node %u is NULL!\n", i);
            }
            mismatches++;
        }
        else
        {
            if( h_cur->Num != want )
            {
                if( mismatches < 16 )
                {
                    fprintf(stderr, "MisMatch at node %u!  got %08X, want %08X\n",
                        i,
                        h_cur->Num,
                        want );
                }
                mismatches++;
            }

            h_cur = h_cur->pNext;
        }
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
    Node*   h_cur = h_head;

    while( h_cur != nullptr )
    {
        Node*   h_next = h_cur->pNext;

        clSVMFree(
            context(),
            h_cur );

        h_cur = h_next;
    }

    h_head = nullptr;
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
        op.add<popl::Value<cl_uint>>("n", "nodes", "Number of Linked List Nodes", numNodes, &numNodes);

        bool printUsage = false;
        try {
            op.parse(argc, argv);
        } catch (std::exception& e) {
            fprintf(stderr, "Error: %s\n\n", e.what());
            printUsage = true;
        }

        if (printUsage || !op.unknown_options().empty() || !op.non_option_args().empty()) {
            fprintf(stderr,
                "Usage: dmemlinkedlist [options]\n"
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

    cl_device_svm_capabilities svmcaps = devices[deviceIndex].getInfo<CL_DEVICE_SVM_CAPABILITIES>();
    if( svmcaps & CL_DEVICE_SVM_FINE_GRAIN_BUFFER ) {
        printf("Device supports CL_DEVICE_SVM_FINE_GRAIN_BUFFER.\n");
    } else {
        printf("Device does not support CL_DEVICE_SVM_FINE_GRAIN_BUFFER, exiting.\n");
        return -1;
    }

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

    init( context, devices[deviceIndex] );
    go();
    checkResults();

    printf("Cleaning up...\n");
    cleanup( context );

    return 0;
}
