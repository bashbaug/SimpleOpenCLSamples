/*
// Copyright (c) 2023 Ben Ashbaugh
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

Node*   s_head;

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
    const cl_svm_alloc_properties_khr props[] = {
        CL_SVM_ALLOC_ASSOCIATED_DEVICE_HANDLE_KHR, (cl_svm_alloc_properties_khr)device(),
        0,
    };

    cl_uint index = CL_UINT_MAX;
    clGetSVMSuggestedTypeIndexKHR(
        context(),
        CL_SVM_TYPE_MACRO_SINGLE_DEVICE_SHARED_KHR,
        0,
        props,
        0,
        &index);

    Node*   s_cur = nullptr;

    for( cl_uint i = 0; i < numNodes; i++ )
    {
        if( i == 0 )
        {
            s_head = (Node*)clSVMAllocWithPropertiesKHR(
                context(),
                props,
                index,
                sizeof(Node),
                nullptr );
            s_cur = s_head;
        }

        if( s_cur != nullptr )
        {
            s_cur->Num = i * 2;

            if( i != numNodes - 1 )
            {
                s_cur->pNext = (Node*)clSVMAllocWithPropertiesKHR(
                    context(),
                    props,
                    index,
                    sizeof(Node),
                    nullptr );
            }
            else
            {
                s_cur->pNext = nullptr;
            }

            s_cur = s_cur->pNext;
        }
    }
}

static void go()
{
    kernel.setArg(0, s_head);
    commandQueue.enqueueNDRangeKernel(
        kernel,
        cl::NullRange,
        cl::NDRange{1} );
}

static void checkResults()
{
    commandQueue.finish();

    const Node* s_cur = s_head;

    unsigned int    mismatches = 0;
    for( cl_uint i = 0; i < numNodes; i++ )
    {
        const cl_uint want = i * 4 + 1;
        if( s_cur == nullptr )
        {
            if( mismatches < 16 )
            {
                fprintf(stderr, "Node %u is NULL!\n", i);
            }
            mismatches++;
        }
        else
        {
            if( s_cur->Num != want )
            {
                if( mismatches < 16 )
                {
                    fprintf(stderr, "MisMatch at node %u!  got %08X, want %08X\n",
                        i,
                        s_cur->Num,
                        want );
                }
                mismatches++;
            }

            s_cur = s_cur->pNext;
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
    Node*   s_cur = s_head;

    while( s_cur != nullptr )
    {
        Node*   s_next = s_cur->pNext;

        clSVMFree(
            context(),
            s_cur );

        s_cur = s_next;
    }

    s_head = nullptr;
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
                "Usage: usmemlinkedlist [options]\n"
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
    cl_bool enable = CL_TRUE;
    clSetKernelExecInfo(
        kernel(),
        CL_KERNEL_EXEC_INFO_SVM_INDIRECT_ACCESS_KHR,
        sizeof(enable),
        &enable );

    init( context, devices[deviceIndex] );
    go();
    checkResults();

    printf("Cleaning up...\n");
    cleanup( context );

    return 0;
}