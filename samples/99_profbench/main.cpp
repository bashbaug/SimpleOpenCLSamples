/*
// Copyright (c) 2026 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <popl/popl.hpp>
#include <CL/opencl.hpp>
#include <chrono>
#include "util.hpp"

static const char kernelString[] = R"CLC(
kernel void inc_buffer(global int* dst)
{
    atomic_inc(dst);
}
)CLC";

int main(int argc, char** argv )
{
    int platformIndex = 0;
    int deviceIndex = 0;

    size_t numEvents = 1024 * 1024;

    {
        popl::OptionParser op("Supported Options");
        op.add<popl::Value<int>>("p", "platform", "Platform Index", platformIndex, &platformIndex);
        op.add<popl::Value<int>>("d", "device", "Device Index", deviceIndex, &deviceIndex);
        op.add<popl::Value<size_t>>("n", "numevents", "Number of Events", numEvents, &numEvents);

        bool printUsage = false;
        try {
            op.parse(argc, argv);
        } catch (std::exception& e) {
            fprintf(stderr, "Error: %s\n\n", e.what());
            printUsage = true;
        }

        if (printUsage || !op.unknown_options().empty() || !op.non_option_args().empty()) {
            fprintf(stderr,
                "Usage: profbench [options]\n"
                "%s", op.help().c_str());
            return -1;
        }
    }

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    if (!checkPlatformIndex(platforms, platformIndex)) {
        return -1;
    }

    printf("Running on platform: %s\n",
        platforms[platformIndex].getInfo<CL_PLATFORM_NAME>().c_str() );

    std::vector<cl::Device> devices;
    platforms[platformIndex].getDevices(CL_DEVICE_TYPE_ALL, &devices);

    printf("Running on device: %s\n",
        devices[deviceIndex].getInfo<CL_DEVICE_NAME>().c_str() );

    cl::Context context{devices[deviceIndex]};
    cl::CommandQueue commandQueue{context, devices[deviceIndex], CL_QUEUE_PROFILING_ENABLE};

    cl::Program program{ context, kernelString };
    program.build();
    cl::Kernel kernel = cl::Kernel{ program, "inc_buffer" };

    cl::Buffer buf = cl::Buffer{
        context,
        CL_MEM_ALLOC_HOST_PTR,
        sizeof(cl_int) };

    kernel.setArg(0, buf);

    const cl_int zero = 0;
    commandQueue.enqueueFillBuffer(buf, zero, 0, sizeof(zero));

    std::vector<cl::Event>  events;
    events.reserve(numEvents);

    printf("Enqueueing kernels to create %zu events...\n", numEvents);
    for (int i = 0; i < numEvents; i++) {
        cl::Event event;
        commandQueue.enqueueNDRangeKernel(
            kernel,
            cl::NullRange,
            cl::NDRange{1},
            cl::NullRange,
            nullptr,
            &event);
        events.push_back(std::move(event));
    }

    printf("Waiting for %zu kernels to complete...\n", numEvents);
    commandQueue.finish();

    cl_ulong totalTimeNS = 0;
    printf("Querying profiling data for %zu events...\n", numEvents);

    auto start = std::chrono::system_clock::now();

    for (const auto& event : events) {
        totalTimeNS +=
            event.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
            event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    }

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<float> queryTimeS = end - start;
    printf("Querying profiling data took %f s (%f us per event)\n",
        queryTimeS.count(), queryTimeS.count() * 1000000 / numEvents);

    int result = 0;
    commandQueue.enqueueReadBuffer(
        buf,
        CL_TRUE,
        0,
        sizeof(result),
        &result);

    if (result == numEvents) {
        printf("Success.\n");
    } else {
        printf("Unexpected result: %d\n", result);
    }

    return 0;
}
