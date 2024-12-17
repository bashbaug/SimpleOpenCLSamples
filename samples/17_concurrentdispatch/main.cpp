/*
// Copyright (c) 2024 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <popl/popl.hpp>

#include <CL/opencl.hpp>

#include "util.hpp"

// TODO: clean this up once support is in the upstream headers.
#if !defined(cl_intel_concurrent_dispatch)

#define cl_intel_concurrent_dispatch 1
#define CL_INTEL_CONCURRENT_DISPATCH_EXTENSION_NAME \
    "cl_intel_concurrent_dispatch"

#define CL_INTEL_CONCURRENT_DISPATCH_EXTENSION_VERSION CL_MAKE_VERSION(1, 0, 0)

/* cl_kernel_exec_info */
#define CL_KERNEL_EXEC_INFO_DISPATCH_TYPE_INTEL             0x4257

typedef cl_uint             cl_kernel_exec_info_dispatch_type_intel;

/* cl_kernel_exec_info_dispatch_type_intel */
#define CL_KERNEL_EXEC_INFO_DISPATCH_TYPE_DEFAULT_INTEL     0
#define CL_KERNEL_EXEC_INFO_DISPATCH_TYPE_CONCURRENT_INTEL  1

typedef cl_int CL_API_CALL
clGetKernelMaxConcurrentWorkGroupCountINTEL_t(
    cl_command_queue command_queue,
    cl_kernel kernel,
    cl_uint work_dim,
    const size_t* global_work_offset,
    const size_t* local_work_size,
    size_t* max_work_group_count);

typedef clGetKernelMaxConcurrentWorkGroupCountINTEL_t *
clGetKernelMaxConcurrentWorkGroupCountINTEL_fn ;

#if !defined(CL_NO_NON_ICD_DISPATCH_EXTENSION_PROTOTYPES)

extern CL_API_ENTRY cl_int CL_API_CALL
clGetKernelMaxConcurrentWorkGroupCountINTEL(
    cl_command_queue command_queue,
    cl_kernel kernel,
    cl_uint work_dim,
    const size_t* global_work_offset,
    const size_t* local_work_size,
    size_t* max_work_group_count) ;

#endif /* !defined(CL_NO_NON_ICD_DISPATCH_EXTENSION_PROTOTYPES) */

#endif // !defined(cl_intel_concurrent_dispatch)

static const char kernelString[] = R"CLC(
#pragma OPENCL EXTENSION cl_intel_concurrent_dispatch : enable
kernel void DeviceBarrierTest( global uint* dst )
{
    const size_t gws = get_global_size(0);
    atomic_add( &dst[gws], 1 );

    //if (intel_is_device_barrier_valid()) {
        //intel_device_barrier( CLK_LOCAL_MEM_FENCE );    // TODO: check fence flags
        //intel_device_barrier( CLK_LOCAL_MEM_FENCE, memory_scope_device );    // TODO: check fence flags
    //}

    const uint id = get_global_id(0);
    dst[id] = dst[gws] + 1;
}
)CLC";

int main(int argc, char** argv)
{
    int platformIndex = 0;
    int deviceIndex = 0;

    size_t iterations = 16;
    size_t lws = 64;
    size_t wgCount = 0;

    {
        popl::OptionParser op("Supported Options");
        op.add<popl::Value<int>>("p", "platform", "Platform Index", platformIndex, &platformIndex);
        op.add<popl::Value<int>>("d", "device", "Device Index", deviceIndex, &deviceIndex);
        op.add<popl::Value<size_t>>("i", "iterations", "Iterations", iterations, &iterations);
        op.add<popl::Value<size_t>>("", "lws", "Local Work-Group Size", lws, &lws);

        bool printUsage = false;
        try {
            op.parse(argc, argv);
        } catch (std::exception& e) {
            fprintf(stderr, "Error: %s\n\n", e.what());
            printUsage = true;
        }

        if (printUsage || !op.unknown_options().empty() || !op.non_option_args().empty()) {
            fprintf(stderr,
                "Usage: concurrentdispatch [options]\n"
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

    if (checkDeviceForExtension(devices[deviceIndex], CL_INTEL_CONCURRENT_DISPATCH_EXTENSION_NAME)) {
        printf("Device supports " CL_INTEL_CONCURRENT_DISPATCH_EXTENSION_NAME ".\n");
    } else {
        printf("Device does not support " CL_INTEL_CONCURRENT_DISPATCH_EXTENSION_NAME ".\n");
        return -1;
    }

    cl::Context context{devices[deviceIndex]};
    cl::CommandQueue commandQueue{context, devices[deviceIndex]};

    cl::Program program{ context, kernelString };
    program.build("-cl-std=CL3.0");
    cl::Kernel kernel = cl::Kernel{ program, "DeviceBarrierTest" };

    cl_kernel_exec_info_dispatch_type_intel dispatchType =
        CL_KERNEL_EXEC_INFO_DISPATCH_TYPE_CONCURRENT_INTEL;
    kernel.setExecInfo(CL_KERNEL_EXEC_INFO_DISPATCH_TYPE_INTEL, dispatchType);

    auto clGetKernelMaxConcurrentWorkGroupCountINTEL_ = (clGetKernelMaxConcurrentWorkGroupCountINTEL_fn)
        clGetExtensionFunctionAddressForPlatform(
            platforms[platformIndex](),
            "clGetKernelMaxConcurrentWorkGroupCountINTEL");
    clGetKernelMaxConcurrentWorkGroupCountINTEL_(
        commandQueue(),
        kernel(),
        1,
        nullptr,
        &lws,
        &wgCount);

    printf("Max concurrent work-group count for local work size %zu is %zu.\n",
        lws, wgCount);

    const size_t gws = lws * wgCount;

    cl::Buffer dst = cl::Buffer{
        context,
        CL_MEM_READ_WRITE,
        (gws + 1) * sizeof(cl_uint) };

    // execution
    {
        kernel.setArg(0, dst);

        commandQueue.finish();

        auto start = std::chrono::system_clock::now();
        for (size_t i = 0; i < iterations; i++) {
            cl_uint zero = 0;
            commandQueue.enqueueFillBuffer(
                dst,
                zero,
                0,
                (gws + 1) * sizeof(cl_uint));
            commandQueue.enqueueNDRangeKernel(
                kernel,
                cl::NullRange,
                cl::NDRange{gws},
                cl::NDRange{lws});
        }

        commandQueue.finish();

        auto end = std::chrono::system_clock::now();
        std::chrono::duration<float> elapsed_seconds = end - start;
        printf("Finished in %f seconds\n", elapsed_seconds.count());
    }

    // verification
    {
        const cl_uint*  pDst = (const cl_uint*)commandQueue.enqueueMapBuffer(
            dst,
            CL_TRUE,
            CL_MAP_READ,
            0,
            (gws + 1) * sizeof(cl_uint) );

        size_t mismatches = 0;

        for( size_t i = 0; i < gws + 1; i++ )
        {
            uint check = (i == gws) ? gws : gws + 1;
            if( pDst[i] != check )
            {
                if( mismatches < 16 )
                {
                    fprintf(stderr, "MisMatch!  dst[%zu] == %08X, want %08X\n",
                        i,
                        pDst[i],
                        check );
                }
                mismatches++;
            }
        }

        if( mismatches )
        {
            fprintf(stderr, "Error: Found %zu mismatches / %zu values!!!\n",
                mismatches,
                gws + 1 );
        }
        else
        {
            printf("Success.\n");
        }

        commandQueue.enqueueUnmapMemObject(
            dst,
            (void*)pDst );
    }

    return 0;
}
