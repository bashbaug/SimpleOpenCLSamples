/*
// Copyright (c) 2019-2024 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <popl/popl.hpp>

#include <CL/opencl.hpp>

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <vector>

#include "util.hpp"

static const char kernelString[] = R"CLC(
float atomic_add_f(volatile global float* addr, float val)
{
    #if defined(__opencl_c_ext_fp32_global_atomic_add) && !defined(EMULATE)
        //#pragma message("using cl_ext_float_atomics")
        return atomic_fetch_add_explicit((volatile global atomic_float*)addr, val, memory_order_relaxed);
    #elif defined(cl_nv_pragma_unroll) && !defined(EMULATE)
        //#pragma message("using PTX atomics")
        float ret; asm volatile("atom.global.add.f32 %0,[%1],%2;":"=f"(ret):"l"(addr),"f"(val):"memory");
        return ret;
    #elif __has_builtin(__builtin_amdgcn_global_atomic_fadd_f32) && !defined(EMULATE)
        //#pragma message("using AMD atomics")
        return __builtin_amdgcn_global_atomic_fadd_f32(addr, val);
    #elif !defined(SLOW_EMULATE)
        // fallback, see: https://forums.developer.nvidia.com/t/atomicadd-float-float-atomicmul-float-float/14639/7
        //#pragma message("using emulated float atomics")
        float old = val; while((old=atomic_xchg(addr, atomic_xchg(addr, 0.0f)+old))!=0.0f);
        // Note: this emulated version cannot reliably return the previous value!
        // This makes it unsuitable for general-purpose use, but it is sufficient
        // for some cases, such as reductions.
        return 0.0f;
    #else
        // This is the traditional fallback that uses a compare and exchange loop.
        // It is much slower, but it supports returning the previous value.
        //#pragma message("using slow emulated float atomics")
        volatile global int* iaddr = (volatile global int*)addr;
        int old;
        int check;
        do {
            old = atomic_or(iaddr, 0);  // emulated atomic load
            int new = as_int(as_float(old) + val);
            check = atomic_cmpxchg(iaddr, old, new);
        } while (check != old);
        return as_float(old);
    #endif
}

kernel void FloatAtomicTest(global float* dst, global float* results)
{
    int index = get_global_id(0);
    results[index] = atomic_add_f(dst, 1.0f);
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
    size_t gwx = 64 * 1024;

    bool emulate = false;
    bool slowEmulate = false;
    bool check = false;

    {
        popl::OptionParser op("Supported Options");
        op.add<popl::Value<int>>("p", "platform", "Platform Index", platformIndex, &platformIndex);
        op.add<popl::Value<int>>("d", "device", "Device Index", deviceIndex, &deviceIndex);
        op.add<popl::Value<size_t>>("i", "iterations", "Iterations", iterations, &iterations);
        op.add<popl::Value<size_t>>("", "gwx", "Global Work Size X AKA Number of Atomics", gwx, &gwx);
        op.add<popl::Switch>("e", "emulate", "Unconditionally Emulate Float Atomics", &emulate);
        op.add<popl::Switch>("s", "slow-emulate", "Unconditionally Emulate Float Atomics (slowly and safely)", &slowEmulate);
        op.add<popl::Switch>("c", "check", "Check Intermediate Results", &check);

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

    // On some implementations, the feature test macros for float atomics are
    // only defined when compiling for OpenCL C 3.0 or newer.
    std::string buildOptions = "-cl-std=CL3.0";
    if (slowEmulate) {
        printf("Forcing slow and safe emulation.\n");
        buildOptions += " -DEMULATE -DSLOW_EMULATE";
    } else if (emulate) {
        printf("Forcing emulation.\n");
        buildOptions += " -DEMULATE";
    } else if (!checkDeviceForExtension(devices[deviceIndex], CL_EXT_FLOAT_ATOMICS_EXTENSION_NAME)) {
        printf("Device does not support " CL_EXT_FLOAT_ATOMICS_EXTENSION_NAME ".\n");
    } else {
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
    }

    cl::Context context{devices[deviceIndex]};
    cl::CommandQueue commandQueue{context, devices[deviceIndex]};

    cl::Program program{ context, kernelString };
    program.build(buildOptions);
    cl::Kernel kernel = cl::Kernel{ program, "FloatAtomicTest" };

    cl::Buffer dst = cl::Buffer{
        context,
        CL_MEM_READ_WRITE,
        sizeof(cl_float) };
    cl::Buffer intermediates = cl::Buffer{
        context,
        CL_MEM_READ_WRITE,
        gwx * sizeof(cl_float) };

    // execution
    {
        kernel.setArg(0, dst);
        kernel.setArg(1, intermediates);

        commandQueue.finish();

        auto start = std::chrono::system_clock::now();
        for (size_t i = 0; i < iterations; i++) {
            cl_float zero = 0.0f;
            commandQueue.enqueueFillBuffer(
                dst,
                zero,
                0,
                sizeof(zero));
            commandQueue.enqueueNDRangeKernel(
                kernel,
                cl::NullRange,
                cl::NDRange{gwx});
        }

        commandQueue.finish();

        auto end = std::chrono::system_clock::now();
        std::chrono::duration<float> elapsed_seconds = end - start;
        printf("Finished in %f seconds\n", elapsed_seconds.count());
    }

    // basic validation
    {
        cl_float check = 0.0f;
        for (size_t i = 0; i < gwx; i++) {
            check += 1.0f;
        }

        cl_float result = 0.0f;
        commandQueue.enqueueReadBuffer(
            dst,
            CL_TRUE,
            0,
            sizeof(result),
            &result);
        if (result != check) {
            printf("Error: expected %f, got %f!\n", check, result);
        } else {
            printf("Basic Validation: Success.\n");
        }
    }

    // intermediate results validation
    if (check) {
        if (emulate && !slowEmulate) {
            printf("The emulated float atomic add does not support intermediate results.\n");
        } else {
            std::vector<cl_float> test(gwx);
            commandQueue.enqueueReadBuffer(
                intermediates,
                CL_TRUE,
                0,
                gwx * sizeof(cl_float),
                test.data());

            std::sort(test.begin(), test.end());

            size_t mismatches = 0;
            for (size_t i = 0; i < gwx; i++) {
                if (i == 0 && !(test[i] == 0.0f)) {
                    if (mismatches < 16) {
                        printf("Error at index %zu: expected %f, got %f!\n", i, 0.0f, test[i]);
                    }
                    mismatches++;
                } else if (i > 0 && !(test[i] > test[i-1])) {
                    if (mismatches < 16) {
                        printf("Error at index %zu: expected %f > %f!\n", i, test[i], test[i-1]);
                    }
                    mismatches++;
                }
            }

            if (mismatches) {
                printf("Intermediate Results Validation: Found %zu mismatches / %zu values!!!\n", mismatches, gwx);
            } else {
                printf("Intermediate Results Validation: Success.\n");
            }
        }
    }

    return 0;
}
