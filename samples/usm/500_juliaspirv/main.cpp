/*
// Copyright (c) 2022 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <popl/popl.hpp>

#include <CL/opencl.hpp>
#include <CL/cl_ext.h>
#include "bmp.hpp"
#include "util.hpp"

#include <chrono>

const char* bmp_filename = "julia.bmp";
const char* spv_filename = "julia.spv";

const float cr = -0.123f;
const float ci =  0.745f;

int main(
    int argc,
    char** argv )
{
    int platformIndex = 0;
    int deviceIndex = 0;

    std::string buildOptions;

    size_t iterations = 16;
    size_t gwx = 512;
    size_t gwy = 512;
    size_t lwx = 0;
    size_t lwy = 0;

    {
        popl::OptionParser op("Supported Options");
        op.add<popl::Value<int>>("p", "platform", "Platform Index", platformIndex, &platformIndex);
        op.add<popl::Value<int>>("d", "device", "Device Index", deviceIndex, &deviceIndex);
        op.add<popl::Value<std::string>>("", "options", "Program Build Options", buildOptions, &buildOptions);
        op.add<popl::Value<size_t>>("i", "iterations", "Iterations", iterations, &iterations);
        op.add<popl::Value<size_t>>("", "gwx", "Global Work Size X AKA Image Width", gwx, &gwx);
        op.add<popl::Value<size_t>>("", "gwy", "Global Work Size Y AKA Image Height", gwy, &gwy);
        op.add<popl::Value<size_t>>("", "lwx", "Local Work Size X", lwx, &lwx);
        op.add<popl::Value<size_t>>("", "lwy", "Local Work Size Y", lwy, &lwy);

        bool printUsage = false;
        try {
            op.parse(argc, argv);
        } catch (std::exception& e) {
            fprintf(stderr, "Error: %s\n\n", e.what());
            printUsage = true;
        }

        if (printUsage || !op.unknown_options().empty() || !op.non_option_args().empty()) {
            fprintf(stderr,
                "Usage: juliaspirv [options]\n"
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

    // Check for SPIR-V support.  This sample requires OpenCL 2.1 or newer
    // for the core clCreateProgramWithIL API.
    // Note: This could look for "SPIR-V" in CL_DEVICE_IL_VERSION.
    if (getDeviceOpenCLVersion(devices[deviceIndex]) >= 0x00020001 &&
        !devices[deviceIndex].getInfo<CL_DEVICE_IL_VERSION>().empty()) {
        printf("Device supports OpenCL 2.1 or newer, using clCreateProgramWithIL.\n");
    } else {
        printf("Device does not support SPIR-V, exiting.\n");
        return -1;
    }

    cl::Context context{devices[deviceIndex]};
    cl::CommandQueue commandQueue = cl::CommandQueue{context, devices[deviceIndex]};

    printf("Reading SPIR-V from file: %s\n", spv_filename);
    std::vector<uint8_t> spirv = readSPIRVFromFile(spv_filename);

    cl::Program program{ clCreateProgramWithIL(context(), spirv.data(), spirv.size(), nullptr) };
    program.build(buildOptions.c_str());
    cl::Kernel kernel = cl::Kernel{ program, "Julia" };

    auto* pDst = (uint32_t*)clHostMemAllocINTEL(
        context(),
        nullptr,
        gwx * gwy * sizeof(uint32_t),
        0,
        nullptr);

    // execution
    if( pDst )
    {
        clSetKernelArgMemPointerINTEL(kernel(), 0, pDst);
        kernel.setArg(1, cr);
        kernel.setArg(2, ci);

        cl::NDRange lws;    // NullRange by default.

        printf("Executing the kernel %d times\n", (int)iterations);
        printf("Global Work Size = ( %d, %d )\n", (int)gwx, (int)gwy);
        if( lwx > 0 && lwy > 0 )
        {
            printf("Local Work Size = ( %d, %d )\n", (int)lwx, (int)lwy);
            lws = cl::NDRange{lwx, lwy};
        }
        else
        {
            printf("Local work size = NULL\n");
        }

        // Ensure the queue is empty and no processing is happening
        // on the device before starting the timer.
        commandQueue.finish();

        auto start = std::chrono::system_clock::now();
        for( int i = 0; i < iterations; i++ )
        {
            commandQueue.enqueueNDRangeKernel(
                kernel,
                cl::NullRange,
                cl::NDRange{gwx, gwy},
                lws);
        }

        // Ensure all processing is complete before stopping the timer.
        commandQueue.finish();

        auto end = std::chrono::system_clock::now();
        std::chrono::duration<float> elapsed_seconds = end - start;
        printf("Finished in %f seconds\n", elapsed_seconds.count());
    }
    else
    {
        printf("Allocation failed - does this device support Unified Shared Memory?\n");
    }

    // save bitmap
    {
        BMP::save_image(pDst, gwx, gwy, bmp_filename);
        printf("Wrote image file %s\n", bmp_filename);
    }

    printf("Cleaning up...\n");

    clMemFreeINTEL(context(), pDst);

    return 0;
}
