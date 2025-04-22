/*
// Copyright (c) 2019-2025 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <popl/popl.hpp>

#include <CL/opencl.hpp>

#include "util.hpp"

#include <chrono>
#include <thread>

using test_clock = std::chrono::high_resolution_clock;

static const char kernelString[] = R"CLC(
kernel void TimeSink(global float* dst)
{
    float result = 0.0f;
    while (result < 1.0f) result += 1e-6f;
    dst[get_global_id(0)] = result;
}
)CLC";

typedef cl_int CL_API_CALL
clEnqueueHostTaskEXP_t(
    cl_command_queue queue,
    void(CL_CALLBACK* user_func)(void*),
    void* user_data,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event);

typedef clEnqueueHostTaskEXP_t *
clEnqueueHostTaskEXP_fn ;

void HostTask(void* user_data)
{
    cl_uint count = 2;
    if (user_data) {
        count = *(cl_uint*)user_data;
    }
    std::this_thread::sleep_for(std::chrono::seconds(count));
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
                "Usage: hosttask [options]\n"
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

    auto clEnqueueHostTaskEXP = (clEnqueueHostTaskEXP_fn)
        clGetExtensionFunctionAddressForPlatform(
            platforms[platformIndex](),
            "clEnqueueHostTaskEXP");

    cl::Context context{devices[deviceIndex]};
    cl::CommandQueue commandQueue{context, devices[deviceIndex]};

    cl::Program program{context, kernelString};
    program.build();
    cl::Kernel kernel = cl::Kernel{program, "TimeSink"};

    cl::Buffer buf = cl::Buffer{
        context,
        CL_MEM_ALLOC_HOST_PTR,
        sizeof(cl_float)};

    kernel.setArg(0, buf);
    commandQueue.enqueueNDRangeKernel(
        kernel,
        cl::NullRange,
        cl::NDRange{1} );

    clEnqueueHostTaskEXP(
        commandQueue(),
        HostTask,
        nullptr,
        0,
        nullptr,
        nullptr);

    commandQueue.enqueueNDRangeKernel(
        kernel,
        cl::NullRange,
        cl::NDRange{1} );

    cl_uint one = 1;
    clEnqueueHostTaskEXP(
        commandQueue(),
        HostTask,
        &one,
        0,
        nullptr,
        nullptr);

    commandQueue.enqueueNDRangeKernel(
        kernel,
        cl::NullRange,
        cl::NDRange{1} );

    cl_uint three = 3;
    clEnqueueHostTaskEXP(
        commandQueue(),
        HostTask,
        &three,
        0,
        nullptr,
        nullptr);

    commandQueue.enqueueNDRangeKernel(
        kernel,
        cl::NullRange,
        cl::NDRange{1} );

    commandQueue.finish();

    return 0;
}
