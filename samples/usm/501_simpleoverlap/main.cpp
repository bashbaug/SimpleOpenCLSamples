/*
// Copyright (c) 2022 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <popl/popl.hpp>

#include <CL/opencl.hpp>
#include <CL/cl_ext.h>
#include "util.hpp"

#include <chrono>
#include <fstream>

using test_clock = std::chrono::high_resolution_clock;

constexpr int maxKernels = 64;
constexpr int testIterations = 32;

int numKernels = 8;
int numIterations = 1;
size_t numElements = 1;

const char* spv_filename = "timesink.spv";

static const char kernelString[] = R"CLC(
kernel void TimeSink( global float* dst, int numIterations )
{
    float result;
    for (int i = 0; i < numIterations; i++) {
        result = 0.0f;
        while (result < 1.0f) result += 1e-6f;
    }
    dst[get_global_id(0)] = result;
}
)CLC";

static std::vector<cl_uchar> readSPIRVFromFile(
    const std::string& filename )
{
    std::ifstream is(filename, std::ios::binary);
    std::vector<cl_uchar> ret;
    if (!is.good()) {
        printf("Couldn't open file '%s'!\n", filename.c_str());
        return ret;
    }

    size_t filesize = 0;
    is.seekg(0, std::ios::end);
    filesize = (size_t)is.tellg();
    is.seekg(0, std::ios::beg);

    ret.reserve(filesize);
    ret.insert(
        ret.begin(),
        std::istreambuf_iterator<char>(is),
        std::istreambuf_iterator<char>() );

    return ret;
}

int main(
    int argc,
    char** argv )
{
    int platformIndex = 0;
    int deviceIndex = 0;

    bool buildFromSource = false;

    {
        popl::OptionParser op("Supported Options");
        op.add<popl::Value<int>>("p", "platform", "Platform Index", platformIndex, &platformIndex);
        op.add<popl::Value<int>>("d", "device", "Device Index", deviceIndex, &deviceIndex);
        op.add<popl::Switch>("", "source", "Build from Source", &buildFromSource);
        op.add<popl::Value<int>>("k", "kernels", "Kernel to Execute", numKernels, &numKernels);
        op.add<popl::Value<int>>("i", "iterations", "Kernel Iterations", numIterations, &numIterations);
        op.add<popl::Value<size_t>>("e", "elements", "Number of ND-Range Elements", numElements, &numElements);

        bool printUsage = false;
        try {
            op.parse(argc, argv);
        } catch (std::exception& e) {
            fprintf(stderr, "Error: %s\n\n", e.what());
            printUsage = true;
        }

        if (printUsage || !op.unknown_options().empty() || !op.non_option_args().empty()) {
            fprintf(stderr,
                "Usage: simpleoverlap [options]\n"
                "%s", op.help().c_str());
            return -1;
        }
    }

    if (numKernels > maxKernels) {
        printf("Number of kernels is %d, which exceeds the maximum of %d.\n", numKernels, maxKernels);
        printf("The number of kernels will be set to %d instead.\n", maxKernels);
        numKernels = maxKernels;
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

    cl::Program program;
    if (buildFromSource) {
        program = cl::Program(kernelString);
    }
    else {
        printf("Reading SPIR-V from file: %s\n", spv_filename);
        std::vector<uint8_t> spirv = readSPIRVFromFile(spv_filename);
        program = cl::Program(clCreateProgramWithIL(context(), spirv.data(), spirv.size(), nullptr));
    }
    program.build();

    cl::Kernel kernel = cl::Kernel{ program, "TimeSink" };

    void* dptr = clHostMemAllocINTEL(
        context(),
        nullptr,
        numElements * sizeof(float),
        0,
        nullptr);
    clSetKernelArgMemPointerINTEL(kernel(), 0, dptr);
    kernel.setArg(1, numIterations);

    // execution
    {
        printf("Running %d iterations of %d kernels...\n", testIterations, numKernels);

        cl::CommandQueue queue(context, devices[deviceIndex], CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);

        float best = 999.0f;
        for (int test = 0; test < testIterations; test++) {
            auto start = test_clock::now();
            for (int i = 0; i < numKernels; i++) {
                queue.enqueueNDRangeKernel(
                    kernel,
                    cl::NullRange,
                    cl::NDRange{numElements});
            }
            queue.finish();

            auto end = test_clock::now();
            std::chrono::duration<float> elapsed_seconds = end - start;
            best = std::min(best, elapsed_seconds.count());
        }

        printf("Best time was %f seconds.\n", best);
    }

    printf("Cleaning up...\n");

    clMemFreeINTEL(context(), dptr);

    return 0;
}
