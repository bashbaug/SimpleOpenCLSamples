/*
// Copyright (c) 2022 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <popl/popl.hpp>

#include <CL/opencl.hpp>

#include <cinttypes>

#include "util.hpp"

// This is a new enum that might not be in all headers yet.
#ifndef CL_COMMAND_BUFFER_CONTEXT_KHR
#define CL_COMMAND_BUFFER_CONTEXT_KHR 0x1299
#endif

const size_t    gwx = 1024*1024;

static const char kernelString[] = R"CLC(
kernel void CopyBuffer( global uint* dst, global uint* src )
{
    uint id = get_global_id(0);
    dst[id] = src[id];
}
)CLC";

static void PrintCommandBufferCapabilities(
    cl_device_command_buffer_capabilities_khr caps )
{
    if (caps & CL_COMMAND_BUFFER_CAPABILITY_KERNEL_PRINTF_KHR       ) printf("\t\tCL_COMMAND_BUFFER_CAPABILITY_KERNEL_PRINTF_KHR\n");
    if (caps & CL_COMMAND_BUFFER_CAPABILITY_DEVICE_SIDE_ENQUEUE_KHR ) printf("\t\tCL_COMMAND_BUFFER_CAPABILITY_DEVICE_SIDE_ENQUEUE_KHR\n");
    if (caps & CL_COMMAND_BUFFER_CAPABILITY_SIMULTANEOUS_USE_KHR    ) printf("\t\tCL_COMMAND_BUFFER_CAPABILITY_SIMULTANEOUS_USE_KHR\n");
    if (caps & CL_COMMAND_BUFFER_CAPABILITY_OUT_OF_ORDER_KHR        ) printf("\t\tCL_COMMAND_BUFFER_CAPABILITY_OUT_OF_ORDER_KHR\n");

    cl_device_command_buffer_capabilities_khr extra = caps & ~(
        CL_COMMAND_BUFFER_CAPABILITY_KERNEL_PRINTF_KHR |
        CL_COMMAND_BUFFER_CAPABILITY_DEVICE_SIDE_ENQUEUE_KHR |
        CL_COMMAND_BUFFER_CAPABILITY_SIMULTANEOUS_USE_KHR |
        CL_COMMAND_BUFFER_CAPABILITY_OUT_OF_ORDER_KHR );
    if (extra) {
        printf("\t\t(Unknown capability: %016" PRIx64 ")\n", extra);
    }
}

static void PrintCommandBufferRequiredQueueProperties(
    cl_command_queue_properties props )
{
    if (props & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE  ) printf("\t\tCL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE\n");
    if (props & CL_QUEUE_PROFILING_ENABLE               ) printf("\t\tCL_QUEUE_PROFILING_ENABLE\n");
#ifdef CL_VERSION_2_0
    if (props & CL_QUEUE_ON_DEVICE                      ) printf("\t\tCL_QUEUE_ON_DEVICE\n");
    if (props & CL_QUEUE_ON_DEVICE_DEFAULT              ) printf("\t\tCL_QUEUE_ON_DEVICE_DEFAULT\n");
#endif

    cl_command_queue_properties extra = props & ~(
        CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE |
        CL_QUEUE_PROFILING_ENABLE |
#ifdef CL_VERSION_2_0
        CL_QUEUE_ON_DEVICE |
        CL_QUEUE_ON_DEVICE_DEFAULT |
#endif
        0);
    if (extra) {
        printf("\t\t(Unknown property: %016" PRIx64 ")\n", extra);
    }
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

        bool printUsage = false;
        try {
            op.parse(argc, argv);
        } catch (std::exception& e) {
            fprintf(stderr, "Error: %s\n\n", e.what());
            printUsage = true;
        }

        if (printUsage || !op.unknown_options().empty() || !op.non_option_args().empty()) {
            fprintf(stderr,
                "Usage: commandbuffers [options]\n"
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

    // device queries:

    bool has_cl_khr_command_buffer =
        checkDeviceForExtension(devices[deviceIndex], CL_KHR_COMMAND_BUFFER_EXTENSION_NAME);
    if (has_cl_khr_command_buffer) {
        printf("Device supports " CL_KHR_COMMAND_BUFFER_EXTENSION_NAME ".\n");
    } else {
        printf("Device does not support " CL_KHR_COMMAND_BUFFER_EXTENSION_NAME ", exiting.\n");
        return -1;
    }

    cl_device_command_buffer_capabilities_khr caps = 0;
    clGetDeviceInfo(
        devices[deviceIndex](),
        CL_DEVICE_COMMAND_BUFFER_CAPABILITIES_KHR,
        sizeof(caps),
        &caps,
        NULL );
    printf("\tCommand Buffer Capabilities:\n");
    PrintCommandBufferCapabilities(caps);

    cl_command_queue_properties requiredProps = 0;
    clGetDeviceInfo(
        devices[deviceIndex](),
        CL_DEVICE_COMMAND_BUFFER_REQUIRED_QUEUE_PROPERTIES_KHR,
        sizeof(requiredProps),
        &requiredProps,
        NULL );
    printf("\tCommand Buffer Required Queue Properties:\n");
    PrintCommandBufferRequiredQueueProperties(requiredProps);

    cl::Context context{devices[deviceIndex]};
    cl::CommandQueue commandQueue = cl::CommandQueue{context, devices[deviceIndex]};

    cl::Program program{ context, kernelString };
    program.build();
    cl::Kernel kernel = cl::Kernel{ program, "CopyBuffer" };

    cl::Buffer deviceMemSrc = cl::Buffer{
        context,
        CL_MEM_ALLOC_HOST_PTR,
        gwx * sizeof( cl_uint ) };

    cl::Buffer deviceMemDst = cl::Buffer{
        context,
        CL_MEM_ALLOC_HOST_PTR,
        gwx * sizeof( cl_uint ) };

    // initialization
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

    cl_command_buffer_khr cmdbuf = clCreateCommandBufferKHR(
        1,
        &commandQueue(),
        NULL,
        NULL);

    // command buffer queries:
    {
        printf("\tCommand Buffer Info:\n");

        cl_uint numQueues = 0;
        clGetCommandBufferInfoKHR(
            cmdbuf,
            CL_COMMAND_BUFFER_NUM_QUEUES_KHR,
            sizeof(numQueues),
            &numQueues,
            NULL );
        printf("\t\tCL_COMMAND_BUFFER_NUM_QUEUES_KHR: %u\n", numQueues);

        cl_command_queue testQueue = NULL;
        clGetCommandBufferInfoKHR(
            cmdbuf,
            CL_COMMAND_BUFFER_QUEUES_KHR,
            sizeof(testQueue),
            &testQueue,
            NULL );
        printf("\t\tCL_COMMAND_BUFFER_QUEUES_KHR: %p (%s)\n",
            testQueue,
            testQueue == commandQueue() ? "matches" : "MISMATCH!");

        cl_context testContext = NULL;
        clGetCommandBufferInfoKHR(
            cmdbuf,
            CL_COMMAND_BUFFER_CONTEXT_KHR,
            sizeof(testContext),
            &testContext,
            NULL );
        printf("\t\tCL_COMMAND_BUFFER_CONTEXT: %p (%s)\n",
            testContext,
            testContext == context() ? "matches" : "MISMATCH!");

        cl_uint refCount = 0;
        clGetCommandBufferInfoKHR(
            cmdbuf,
            CL_COMMAND_BUFFER_REFERENCE_COUNT_KHR,
            sizeof(refCount),
            &refCount,
            NULL );
        printf("\t\tCL_COMMAND_BUFFER_REFERENCE_COUNT_KHR: %u\n", refCount);

        cl_command_buffer_state_khr state = 0;
        clGetCommandBufferInfoKHR(
            cmdbuf,
            CL_COMMAND_BUFFER_STATE_KHR,
            sizeof(state),
            &state,
            NULL );
        printf("\t\tCL_COMMAND_BUFFER_STATE_KHR: %s\n",
            state == CL_COMMAND_BUFFER_STATE_RECORDING_KHR ? "RECORDING" :
            state == CL_COMMAND_BUFFER_STATE_EXECUTABLE_KHR ? "EXECUTABLE" :
            state == CL_COMMAND_BUFFER_STATE_PENDING_KHR ? "PENDING" :
            "UNKNOWN!");

        size_t propsSize = 0;
        clGetCommandBufferInfoKHR(
            cmdbuf,
            CL_COMMAND_BUFFER_PROPERTIES_ARRAY_KHR,
            0,
            NULL,
            &propsSize );
        printf("\t\tCL_COMMAND_BUFFER_PROPERTIES_ARRAY_KHR size: %zu\n", propsSize);
    }

    kernel.setArg(0, deviceMemDst);
    kernel.setArg(1, deviceMemSrc);
    cl_sync_point_khr sync_point;
    clCommandNDRangeKernelKHR(
        cmdbuf,
        NULL,
        NULL,
        kernel(),
        1,
        NULL,
        &gwx,
        NULL,
        0,
        NULL,
        &sync_point,
        NULL);
    clFinalizeCommandBufferKHR(cmdbuf);

    clEnqueueCommandBufferKHR(
        0,
        NULL,
        cmdbuf,
        0,
        NULL,
        NULL);

    // verification
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
            fprintf(stderr, "Error: Found %d mismatches / %d values!!!\n",
                mismatches,
                (unsigned int)gwx );
        }
        else
        {
            printf("Success.\n");
        }

        commandQueue.enqueueUnmapMemObject(
            deviceMemDst,
            (void*)pDst );
    }

    clReleaseCommandBufferKHR(cmdbuf);

    return 0;
}
