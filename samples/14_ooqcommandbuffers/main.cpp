/*
// Copyright (c) 2022 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <popl/popl.hpp>

#include <CL/opencl.hpp>

#include <cinttypes>

#include "util.hpp"

const size_t    gwx = 1024*1024;

static const char kernelString[] = R"CLC(
kernel void SlowFillBuffer( global uint* dst, uint pattern, uint size )
{
    for (uint i = 0; i < size; i++) {
        dst[i] = pattern;
    }
}

kernel void AddBuffers( global uint* dst, global uint* srcA, global uint* srcB )
{
    uint id = get_global_id(0);
    dst[id] = srcA[id] + srcB[id];
}
)CLC";

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
                "Usage: copybufferkernel [options]\n"
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

    cl_command_queue_properties cmdqprops =
        devices[deviceIndex].getInfo<CL_DEVICE_QUEUE_PROPERTIES>();
    if (cmdqprops & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) {
        printf("Device supports out-of-order queues.\n");
    } else {
        printf("Device does not support out-of-order queues, exiting.\n");
        return -1;
    }

    cl_device_command_buffer_capabilities_khr cmdbufcaps =
        devices[deviceIndex].getInfo<CL_DEVICE_COMMAND_BUFFER_CAPABILITIES_KHR>();
    if (cmdbufcaps & CL_COMMAND_BUFFER_CAPABILITY_OUT_OF_ORDER_KHR) {
        printf("Device supports out-of-order command buffers.\n");
    } else {
        printf("Device does not support out-of-order command buffers, exiting.\n");
        return -1;
    }

    cl::Context context{devices[deviceIndex]};
    cl::CommandQueue commandQueue{context, devices[deviceIndex], CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE};

    cl::Program program{ context, kernelString };
    program.build();
    cl::Kernel fillKernel = cl::Kernel{ program, "SlowFillBuffer" };
    cl::Kernel addKernel = cl::Kernel{ program, "AddBuffers" };

    cl::Buffer deviceMemSrcA = cl::Buffer{
        context,
        CL_MEM_ALLOC_HOST_PTR,
        gwx * sizeof( cl_uint ) };

    cl::Buffer deviceMemSrcB = cl::Buffer{
        context,
        CL_MEM_ALLOC_HOST_PTR,
        gwx * sizeof( cl_uint ) };

    cl::Buffer deviceMemDst = cl::Buffer{
        context,
        CL_MEM_ALLOC_HOST_PTR,
        gwx * sizeof( cl_uint ) };

    cl::CommandBuffer cmdbuf{clCreateCommandBufferKHR(
        1,
        &commandQueue(),
        nullptr,
        nullptr)};

    const size_t one = 1;

    cl_sync_point_khr writeA = 0;
    fillKernel.setArg(0, deviceMemSrcA);
    fillKernel.setArg(1, static_cast<cl_uint>(1));
    fillKernel.setArg(2, static_cast<cl_uint>(gwx));
    clCommandNDRangeKernelKHR(
        cmdbuf(),
        nullptr,
        nullptr,
        fillKernel(),
        1,
        nullptr,
        &one,
        nullptr,
        0,
        nullptr,
        &writeA,
        nullptr);

    cl_sync_point_khr writeB = 0;
    fillKernel.setArg(0, deviceMemSrcB);
    fillKernel.setArg(1, static_cast<cl_uint>(2));
    fillKernel.setArg(2, static_cast<cl_uint>(gwx));
    clCommandNDRangeKernelKHR(
        cmdbuf(),
        nullptr,
        nullptr,
        fillKernel(),
        1,
        nullptr,
        &one,
        nullptr,
        0,
        nullptr,
        &writeB,
        nullptr);

    std::vector<cl_sync_point_khr> waitList({writeA, writeB});
    addKernel.setArg(0, deviceMemDst);
    addKernel.setArg(1, deviceMemSrcA);
    addKernel.setArg(2, deviceMemSrcB);
    clCommandNDRangeKernelKHR(
        cmdbuf(),
        nullptr,
        nullptr,
        addKernel(),
        1,
        nullptr,
        &gwx,
        nullptr,
        static_cast<cl_uint>(waitList.size()),
        waitList.data(),
        nullptr,
        nullptr);
    cmdbuf.finalize();

    clEnqueueCommandBufferKHR(
        0,
        nullptr,
        cmdbuf(),
        0,
        nullptr,
        nullptr);

    // verification
    {
        commandQueue.finish();

        const cl_uint*  pDst = (const cl_uint*)commandQueue.enqueueMapBuffer(
            deviceMemDst,
            CL_TRUE,
            CL_MAP_READ,
            0,
            gwx * sizeof(cl_uint) );

        unsigned int    mismatches = 0;

        for( size_t i = 0; i < gwx; i++ )
        {
            const cl_uint got = pDst[0];
            const cl_uint want = 3;
            if( got != want )
            {
                if( mismatches < 16 )
                {
                    fprintf(stderr, "MisMatch!  dst[%d] == %08X, want %08X\n",
                        (unsigned int)i,
                        got,
                        want );
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
        commandQueue.finish();
    }

    return 0;
}
