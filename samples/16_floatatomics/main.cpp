/*
// Copyright (c) 2019-2024 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <popl/popl.hpp>

#include <CL/opencl.hpp>

#include <chrono>
#include <cinttypes>

#include "util.hpp"

static const char kernelString[] = R"CLC(
inline float atomic_add_f(volatile global float* addr, float val)
{
    #if defined(__opencl_c_ext_fp32_global_atomic_add) && !defined(EMULATE)
        //#pragma message("using cl_ext_float_atomics")
        return atomic_fetch_add_explicit((volatile global atomic_float*)addr, val, memory_order_relaxed);
    #elif defined(cl_nv_pragma_unroll) && !defined(EMULATE)
        //#pragma message("using PTX atomics")
        float ret; asm volatile("atom.global.add.f32 %0,[%1],%2;":"=f"(ret):"l"(addr),"f"(val):"memory");
        return ret;
    #else // fallback, see: https://forums.developer.nvidia.com/t/atomicadd-float-float-atomicmul-float-float/14639/7
        //#pragma message("using emulated float atomics")
        float ret = atomic_xchg(addr, 0.0f);
        float old = ret + val;
        while((old = atomic_xchg(addr, old)) != 0.0f) {
            old = atomic_xchg(addr, 0.0f) + old;
        }
        return ret;
    #endif
}

kernel void FloatAtomicTest(global float* dst)
{
    atomic_add_f(dst, 1.0f);
}
)CLC";

static void PrintFloatAtomicCapabilities(
    cl_device_fp_atomic_capabilities_ext caps )
{
    if (caps & CL_DEVICE_GLOBAL_FP_ATOMIC_LOAD_STORE_EXT ) printf("\t\tCL_DEVICE_GLOBAL_FP_ATOMIC_LOAD_STORE_EXT\n");
    if (caps & CL_DEVICE_GLOBAL_FP_ATOMIC_ADD_EXT        ) printf("\t\tCL_DEVICE_GLOBAL_FP_ATOMIC_ADD_EXT\n");
    if (caps & CL_DEVICE_GLOBAL_FP_ATOMIC_MIN_MAX_EXT    ) printf("\t\tCL_DEVICE_GLOBAL_FP_ATOMIC_MIN_MAX_EXT\n");
    if (caps & CL_DEVICE_LOCAL_FP_ATOMIC_LOAD_STORE_EXT  ) printf("\t\tCL_DEVICE_LOCAL_FP_ATOMIC_LOAD_STORE_EXT\n");
    if (caps & CL_DEVICE_LOCAL_FP_ATOMIC_ADD_EXT         ) printf("\t\tCL_DEVICE_LOCAL_FP_ATOMIC_ADD_EXT\n");
    if (caps & CL_DEVICE_LOCAL_FP_ATOMIC_MIN_MAX_EXT     ) printf("\t\tCL_DEVICE_LOCAL_FP_ATOMIC_MIN_MAX_EXT\n");

    cl_device_command_buffer_capabilities_khr extra = caps & ~(
        CL_DEVICE_GLOBAL_FP_ATOMIC_LOAD_STORE_EXT |
        CL_DEVICE_GLOBAL_FP_ATOMIC_ADD_EXT |
        CL_DEVICE_GLOBAL_FP_ATOMIC_MIN_MAX_EXT |
        CL_DEVICE_LOCAL_FP_ATOMIC_LOAD_STORE_EXT |
        CL_DEVICE_LOCAL_FP_ATOMIC_ADD_EXT |
        CL_DEVICE_LOCAL_FP_ATOMIC_MIN_MAX_EXT );
    if (extra) {
        printf("\t\t(Unknown capability: %016" PRIx64 ")\n", extra);
    }
}

int main(
    int argc,
    char** argv )
{
    int platformIndex = 0;
    int deviceIndex = 0;

    size_t iterations = 16;
    size_t gwx = 1024 * 1024;

    bool emulate = false;

    {
        popl::OptionParser op("Supported Options");
        op.add<popl::Value<int>>("p", "platform", "Platform Index", platformIndex, &platformIndex);
        op.add<popl::Value<int>>("d", "device", "Device Index", deviceIndex, &deviceIndex);
        op.add<popl::Value<size_t>>("i", "iterations", "Iterations", iterations, &iterations);
        op.add<popl::Value<size_t>>("", "gwx", "Global Work Size X AKA Number of Atomics", gwx, &gwx);
        op.add<popl::Switch>("e", "emulate", "Unconditionally Emulate Float Atomics", &emulate);

        bool printUsage = false;
        try {
            op.parse(argc, argv);
        } catch (std::exception& e) {
            fprintf(stderr, "Error: %s\n\n", e.what());
            printUsage = true;
        }

        if (printUsage || !op.unknown_options().empty() || !op.non_option_args().empty()) {
            fprintf(stderr,
                "Usage: floatatomics [options]\n"
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

    if (checkDeviceForExtension(devices[deviceIndex], CL_EXT_FLOAT_ATOMICS_EXTENSION_NAME)) {
        printf("Device supports " CL_EXT_FLOAT_ATOMICS_EXTENSION_NAME ".\n");

        cl_device_fp_atomic_capabilities_ext spcaps =
            devices[deviceIndex].getInfo<CL_DEVICE_SINGLE_FP_ATOMIC_CAPABILITIES_EXT>();
        printf("CL_DEVICE_SINGLE_FP_ATOMIC_CAPABILITIES_EXT:\n");
        PrintFloatAtomicCapabilities(spcaps);

        cl_device_fp_atomic_capabilities_ext dpcaps =
            devices[deviceIndex].getInfo<CL_DEVICE_DOUBLE_FP_ATOMIC_CAPABILITIES_EXT>();
        printf("CL_DEVICE_DOUBLE_FP_ATOMIC_CAPABILITIES_EXT:\n");
        PrintFloatAtomicCapabilities(dpcaps);

        cl_device_fp_atomic_capabilities_ext hpcaps =
            devices[deviceIndex].getInfo<CL_DEVICE_HALF_FP_ATOMIC_CAPABILITIES_EXT>();
        printf("CL_DEVICE_HALF_FP_ATOMIC_CAPABILITIES_EXT:\n");
        PrintFloatAtomicCapabilities(hpcaps);

        if (spcaps & CL_DEVICE_GLOBAL_FP_ATOMIC_ADD_EXT == 0) {
            printf("Device does not support fp32 atomic add.\n");
        }
    } else {
        printf("Device does not support " CL_EXT_FLOAT_ATOMICS_EXTENSION_NAME ".\n");
    }

    cl::Context context{devices[deviceIndex]};
    cl::CommandQueue commandQueue{context, devices[deviceIndex]};

    cl::Program program{ context, kernelString };
    std::string buildOptions = "-cl-std=CL3.0";

    if (emulate) {
        printf("Forcing emulation.\n");
        buildOptions += " -DEMULATE";
    }

    program.build(buildOptions);
    cl::Kernel kernel = cl::Kernel{ program, "FloatAtomicTest" };

    cl::Buffer deviceMemDst = cl::Buffer{
        context,
        CL_MEM_READ_WRITE,
        sizeof(cl_float) };

    // execution
    {
        kernel.setArg(0, deviceMemDst);

        // Ensure the queue is empty and no processing is happening
        // on the device before starting the timer.
        commandQueue.finish();

        auto start = std::chrono::system_clock::now();
        for( size_t i = 0; i < iterations; i++ )
        {
            cl_float zero = 0.0f;
            commandQueue.enqueueFillBuffer(
                deviceMemDst,
                zero,
                0,
                sizeof(zero));
            commandQueue.enqueueNDRangeKernel(
                kernel,
                cl::NullRange,
                cl::NDRange{gwx});
        }

        // Ensure all processing is complete before stopping the timer.
        commandQueue.finish();

        auto end = std::chrono::system_clock::now();
        std::chrono::duration<float> elapsed_seconds = end - start;
        printf("Finished in %f seconds\n", elapsed_seconds.count());
    }

    // validation
    {
        cl_float result = 0.0f;
        commandQueue.enqueueReadBuffer(
            deviceMemDst,
            CL_TRUE,
            0,
            sizeof(result),
            &result);
        if (result != (float)gwx) {
            printf("Error: expected %f, got %f!\n", (float)gwx, result);
        } else {
            printf("Success.\n");
        }
    }

    return 0;
}
