/*
// Copyright (c) 2019-2021 Ben Ashbaugh
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
*/

#include <popl/popl.hpp>

#include <CL/opencl.hpp>

#include "util.hpp"

#include <chrono>

using test_clock = std::chrono::high_resolution_clock;

constexpr int maxKernels = 64;
constexpr int testIterations = 32;

int numIterations = 1;
size_t numElements = 1;

std::vector<cl::Kernel> kernels;
std::vector<cl::Buffer> buffers;

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

static void init(cl::Context& context, cl::Device& device)
{
    cl::CommandQueue queue(context, device);
    for (auto& buffer : buffers) {
        float pattern = 0.0f;
        queue.enqueueFillBuffer(buffer, pattern, 0, numElements * sizeof(pattern));
    }
    queue.finish();
}

static void go_kernelxN( cl::Context& context, cl::Device& device, const int numKernels )
{
    init(context, device);

    printf("%s (n=%d): ", __FUNCTION__, numKernels); fflush(stdout);

    cl::CommandQueue queue(context, device);

    float best = 999.0f;
    for (int test = 0; test < testIterations; test++) {
        auto start = test_clock::now();
        for (int i = 0; i < numKernels; i++) {
            queue.enqueueNDRangeKernel(
                kernels[i],
                cl::NullRange,
                cl::NDRange{numElements});
        }
        queue.finish();

        auto end = test_clock::now();
        std::chrono::duration<float> elapsed_seconds = end - start;
        best = std::min(best, elapsed_seconds.count());
    }
    printf("Finished in %f seconds\n", best);
}

static void go_kernelxN_ooq( cl::Context& context, cl::Device& device, const int numKernels )
{
    init(context, device);

    printf("%s (n=%d): ", __FUNCTION__, numKernels); fflush(stdout);

    cl_command_queue_properties props = device.getInfo<CL_DEVICE_QUEUE_PROPERTIES>();
    if ((props & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) == 0) {
        printf("Skipping (device does not support out-of-order queues).\n");
        return;
    }

    cl::CommandQueue queue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);

    float best = 999.0f;
    for (int test = 0; test < testIterations; test++) {
        auto start = test_clock::now();
        for (int i = 0; i < numKernels; i++) {
            queue.enqueueNDRangeKernel(
                kernels[i],
                cl::NullRange,
                cl::NDRange{numElements});
        }
        queue.finish();

        auto end = test_clock::now();
        std::chrono::duration<float> elapsed_seconds = end - start;
        best = std::min(best, elapsed_seconds.count());
    }
    printf("Finished in %f seconds\n", best);
}

static void go_kernelxN_ooq_events( cl::Context& context, cl::Device& device, const int numKernels )
{
    init(context, device);

    printf("%s (n=%d): ", __FUNCTION__, numKernels); fflush(stdout);

    cl_command_queue_properties props = device.getInfo<CL_DEVICE_QUEUE_PROPERTIES>();
    if ((props & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) == 0) {
        printf("Skipping (device does not support out-of-order queues).\n");
        return;
    }

    cl::CommandQueue queue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);

    float best = 999.0f;
    for (int test = 0; test < testIterations; test++) {
        std::vector<cl::Event> events(numKernels);
        auto start = test_clock::now();
        for (int i = 0; i < numKernels; i++) {
            queue.enqueueNDRangeKernel(
                kernels[i],
                cl::NullRange,
                cl::NDRange{numElements},
                cl::NullRange,
                nullptr,
                &events[i]);
        }
        cl::Event::waitForEvents(events);

        auto end = test_clock::now();
        std::chrono::duration<float> elapsed_seconds = end - start;
        best = std::min(best, elapsed_seconds.count());
    }
    printf("Finished in %f seconds\n", best);
}

static void go_kernel_ioqxN( cl::Context& context, cl::Device& device, const int numKernels )
{
    init(context, device);

    printf("%s (n=%d): ", __FUNCTION__, numKernels); fflush(stdout);

    cl_command_queue_properties props = device.getInfo<CL_DEVICE_QUEUE_PROPERTIES>();
    if (props & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) {
        // Create a dummy out-of-order queue to enable command aggregation.
        cl::CommandQueue dummy{context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE};
    }

    std::vector<cl::CommandQueue> queues;
    for (int i = 0; i < numKernels; i++) {
        queues.push_back(cl::CommandQueue{context, device});
    }

    float best = 999.0f;
    for (int test = 0; test < testIterations; test++) {
        auto start = test_clock::now();
        for (int i = 0; i < numKernels; i++) {
            queues[i].enqueueNDRangeKernel(
                kernels[i],
                cl::NullRange,
                cl::NDRange{numElements});
        }
        for (int i = 0; i < numKernels; i++) {
            queues[i].flush();
        }
        for (int i = 0; i < numKernels; i++) {
            queues[i].finish();
        }

        auto end = test_clock::now();
        std::chrono::duration<float> elapsed_seconds = end - start;
        best = std::min(best, elapsed_seconds.count());
    }
    printf("Finished in %f seconds\n", best);
}

static void findQueueFamily( cl::Device& device, cl_uint& family, cl_uint& numQueues )
{
    numQueues = 0;

    std::vector<cl_queue_family_properties_intel>   qfprops =
        device.getInfo<CL_DEVICE_QUEUE_FAMILY_PROPERTIES_INTEL>();
    for (cl_uint q = 0; q < qfprops.size(); q++) {
        if (qfprops[q].capabilities == CL_QUEUE_DEFAULT_CAPABILITIES_INTEL &&
            qfprops[q].count > numQueues) {
            family = q;
            numQueues = qfprops[q].count;
        }
    }
}

static void go_kernel_qf_ioqxN( cl::Context& context, cl::Device& device, const int numKernels )
{
    init(context, device);

    printf("%s (n=%d): ", __FUNCTION__, numKernels); fflush(stdout);

    if (!checkDeviceForExtension(device, "cl_intel_command_queue_families")) {
        printf("Skipping (device does not support cl_intel_command_queue_families).\n");
        return;
    }

    cl_uint family = 0;
    cl_uint numQueues = 0;
    findQueueFamily(device, family, numQueues);
    if (numQueues == 0) {
        printf("Skipping (no queues found?).\n");
        return;
    } else {
        printf("Using queue family %u with %u queue(s): ", family, numQueues); fflush(stdout);
    }

    cl_command_queue_properties props = device.getInfo<CL_DEVICE_QUEUE_PROPERTIES>();
    if (props & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) {
        // Create a dummy out-of-order queue to enable command aggregation.
        cl::CommandQueue dummy{context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE};
    }

    std::vector<cl::CommandQueue> queues;
    for (int i = 0; i < numKernels; i++) {
        cl_queue_properties props[] = {
            CL_QUEUE_FAMILY_INTEL, family,
            CL_QUEUE_INDEX_INTEL, i % numQueues,
            0
        };
        queues.push_back(cl::CommandQueue{
            clCreateCommandQueueWithProperties(context(), device(), props, NULL)});
    }

    float best = 999.0f;
    for (int test = 0; test < testIterations; test++) {
        auto start = test_clock::now();
        for (int i = 0; i < numKernels; i++) {
            queues[i].enqueueNDRangeKernel(
                kernels[i],
                cl::NullRange,
                cl::NDRange{numElements});
        }
        for (int i = 0; i < numKernels; i++) {
            queues[i].flush();
        }
        for (int i = 0; i < numKernels; i++) {
            queues[i].finish();
        }

        auto end = test_clock::now();
        std::chrono::duration<float> elapsed_seconds = end - start;
        best = std::min(best, elapsed_seconds.count());
    }
    printf("Finished in %f seconds\n", best);
}

static void go_kernel_qfs_ioqxN( cl::Context& context, cl::Device& device, const int numKernels )
{
    init(context, device);

    printf("%s (n=%d): ", __FUNCTION__, numKernels); fflush(stdout);

    if (!checkDeviceForExtension(device, "cl_intel_command_queue_families")) {
        printf("Skipping (device does not support cl_intel_command_queue_families).\n");
        return;
    }

    cl_uint family = 0;
    cl_uint numQueues = 0;
    findQueueFamily(device, family, numQueues);
    if (numQueues == 0) {
        printf("Skipping (no queues found?).\n");
        return;
    } else {
        printf("Using queue family %u with %u queue(s): ", family, numQueues); fflush(stdout);
    }

    cl_command_queue_properties props = device.getInfo<CL_DEVICE_QUEUE_PROPERTIES>();
    if (props & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) {
        // Create a dummy out-of-order queue to enable command aggregation.
        cl::CommandQueue dummy{context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE};
    }

    std::vector<cl::CommandQueue> queues;
    for (int i = 0; i < numKernels; i++) {
        cl_queue_properties props[] = {
            CL_QUEUE_FAMILY_INTEL, family,
            CL_QUEUE_INDEX_INTEL, 0,    // always use queue index zero
            0
        };
        queues.push_back(cl::CommandQueue{
            clCreateCommandQueueWithProperties(context(), device(), props, NULL)});
    }

    float best = 999.0f;
    for (int test = 0; test < testIterations; test++) {
        auto start = test_clock::now();
        for (int i = 0; i < numKernels; i++) {
            queues[i].enqueueNDRangeKernel(
                kernels[i],
                cl::NullRange,
                cl::NDRange{numElements});
        }
        for (int i = 0; i < numKernels; i++) {
            queues[i].flush();
        }
        for (int i = 0; i < numKernels; i++) {
            queues[i].finish();
        }

        auto end = test_clock::now();
        std::chrono::duration<float> elapsed_seconds = end - start;
        best = std::min(best, elapsed_seconds.count());
    }
    printf("Finished in %f seconds\n", best);
}

static void go_kernel_ctx_ioqxN( cl::Device& device, const int numKernels )
{
    printf("%s (n=%d): ", __FUNCTION__, numKernels); fflush(stdout);

    std::vector<cl::Context> contexts;
    std::vector<cl::Program> programs;
    std::vector<cl::Kernel> kernels;
    std::vector<cl::Buffer> buffers;
    std::vector<cl::CommandQueue> queues;

    for (int i = 0; i < numKernels; i++) {
        contexts.push_back(cl::Context{device});

        programs.push_back(cl::Program{ contexts[i], kernelString });
        programs[i].build();

        buffers.push_back(cl::Buffer{contexts[i], CL_MEM_ALLOC_HOST_PTR, numElements * sizeof(float)});
        kernels.push_back(cl::Kernel{programs[i], "TimeSink"});
        queues.push_back(cl::CommandQueue{contexts[i], device});

        kernels[i].setArg(0, buffers[i]);
        kernels[i].setArg(1, numIterations);

        float pattern = 0.0f;
        queues[i].enqueueFillBuffer(buffers[i], pattern, 0, numElements * sizeof(pattern));
        queues[i].finish();

        cl_command_queue_properties props = device.getInfo<CL_DEVICE_QUEUE_PROPERTIES>();
        if (props & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) {
            // Create a dummy out-of-order queue to enable command aggregation.
            cl::CommandQueue dummy{contexts[i], device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE};
        }
    }

    float best = 999.0f;
    for (int test = 0; test < testIterations; test++) {
        auto start = test_clock::now();
        for (int i = 0; i < numKernels; i++) {
            queues[i].enqueueNDRangeKernel(
                kernels[i],
                cl::NullRange,
                cl::NDRange{numElements});
        }
        for (int i = 0; i < numKernels; i++) {
            queues[i].flush();
        }
        for (int i = 0; i < numKernels; i++) {
            queues[i].finish();
        }

        auto end = test_clock::now();
        std::chrono::duration<float> elapsed_seconds = end - start;
        best = std::min(best, elapsed_seconds.count());
    }
    printf("Finished in %f seconds\n", best);
}

int main(
    int argc,
    char** argv )
{
    int platformIndex = 0;
    int deviceIndex = 0;
    int numKernels = 0;

    {
        popl::OptionParser op("Supported Options");
        op.add<popl::Value<int>>("p", "platform", "Platform Index", platformIndex, &platformIndex);
        op.add<popl::Value<int>>("d", "device", "Device Index", deviceIndex, &deviceIndex);
        op.add<popl::Value<int>>("k", "kernels", "Kernel to Execute (<=0 to sweep)", numKernels, &numKernels);
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
                "Usage: queueexperiments [options]\n"
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

    cl::Device& device = devices[deviceIndex];
    printf("Running on device: %s\n",
        device.getInfo<CL_DEVICE_NAME>().c_str() );

    cl::Context context{ device };

    cl::Program program{ context, kernelString };
    program.build();

    for (int i = 0; i < maxKernels; i++) {
        buffers.push_back(cl::Buffer{context, CL_MEM_ALLOC_HOST_PTR, numElements * sizeof(float)});
        kernels.push_back(cl::Kernel{program, "TimeSink"});

        kernels[i].setArg(0, buffers[i]);
        kernels[i].setArg(1, numIterations);
    }

    std::vector<int> counts;
    if (numKernels <= 0) {
        counts.assign({1, 2, 4, 8, 16, 32, 64});
    } else {
        counts.assign({numKernels});
    }

    for (auto count : counts) {
        go_kernelxN(context, device, count);
    }

    for (auto count : counts) {
        go_kernelxN_ooq(context, device, count);
    }

    for (auto count : counts) {
        go_kernelxN_ooq_events(context, device, count);
    }

    for (auto count : counts) {
        go_kernel_ioqxN(context, device, count);
    }

    for (auto count : counts) {
        go_kernel_qf_ioqxN(context, device, count);
    }

    for (auto count : counts) {
        go_kernel_qfs_ioqxN(context, device, count);
    }

    for (auto count : counts) {
        go_kernel_ctx_ioqxN(device, count);
    }

    return 0;
}
