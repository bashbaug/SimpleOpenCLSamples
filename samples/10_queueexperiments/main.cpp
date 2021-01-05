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

#include <CL/opencl.hpp>

#include <chrono>

using test_clock = std::chrono::high_resolution_clock;

constexpr int maxKernels = 16;
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

static void go_kernelx1( cl::Context& context, cl::Device& device )
{
    constexpr int numKernels = 1;
    init(context, device);

    printf("%s: ", __FUNCTION__); fflush(stdout);

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

static void go_kernelx2( cl::Context& context, cl::Device& device )
{
    constexpr int numKernels = 2;
    init(context, device);

    printf("%s: ", __FUNCTION__); fflush(stdout);

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

static void go_kernelx4( cl::Context& context, cl::Device& device )
{
    constexpr int numKernels = 4;
    init(context, device);

    printf("%s: ", __FUNCTION__); fflush(stdout);

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

static void go_kernelx1_ooq( cl::Context& context, cl::Device& device )
{
    constexpr int numKernels = 1;
    init(context, device);

    printf("%s: ", __FUNCTION__); fflush(stdout);

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

static void go_kernelx2_ooq( cl::Context& context, cl::Device& device )
{
    constexpr int numKernels = 2;
    init(context, device);

    printf("%s: ", __FUNCTION__); fflush(stdout);

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

static void go_kernelx4_ooq( cl::Context& context, cl::Device& device )
{
    constexpr int numKernels = 4;
    init(context, device);

    printf("%s: ", __FUNCTION__); fflush(stdout);

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

static void go_kernelx4_ooq_events( cl::Context& context, cl::Device& device )
{
    constexpr int numKernels = 4;
    init(context, device);

    printf("%s: ", __FUNCTION__); fflush(stdout);

    cl_command_queue_properties props = device.getInfo<CL_DEVICE_QUEUE_PROPERTIES>();
    if ((props & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) == 0) {
        printf("Skipping (device does not support out-of-order queues).\n");
        return;
    }

    cl::CommandQueue queue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);

    float best = 999.0f;
    for (int test = 0; test < testIterations; test++) {
        std::vector<cl::Event> events(4);
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

static void go_kernel_ioqx4( cl::Context& context, cl::Device& device )
{
    constexpr int numKernels = 4;
    init(context, device);

    printf("%s: ", __FUNCTION__); fflush(stdout);

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

int main(
    int argc,
    char** argv )
{
    bool printUsage = false;
    int platformIndex = 0;
    int deviceIndex = 0;

    if( argc < 1 )
    {
        printUsage = true;
    }
    else
    {
        for( size_t i = 1; i < argc; i++ )
        {
            if( !strcmp( argv[i], "-d" ) )
            {
                ++i;
                if( i < argc )
                {
                    deviceIndex = strtol(argv[i], NULL, 10);
                }
            }
            else if( !strcmp( argv[i], "-p" ) )
            {
                ++i;
                if( i < argc )
                {
                    platformIndex = strtol(argv[i], NULL, 10);
                }
            }
            else if( !strcmp( argv[i], "-i" ) )
            {
                if( ++i < argc )
                {
                    numIterations = strtol(argv[i], NULL, 10);
                }
            }
            else if( !strcmp( argv[i], "-e" ) )
            {
                if( ++i < argc )
                {
                    numElements = strtol(argv[i], NULL, 10);
                }
            }
            else
            {
                printUsage = true;
            }
        }
    }
    if( printUsage )
    {
        fprintf(stderr,
            "Usage: queueexperiments [options]\n"
            "Options:\n"
            "      -d: Device Index (default = 0)\n"
            "      -p: Platform Index (default = 0)\n"
            "      -i: Kernel Iterations (default = 1)\n"
            "      -e: Number of ND-Range Elements (default = 1)\n"
            );

        return -1;
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

    go_kernelx1(context, device);
    go_kernelx2(context, device);
    go_kernelx4(context, device);

    go_kernelx1_ooq(context, device);
    go_kernelx2_ooq(context, device);
    go_kernelx4_ooq(context, device);

    go_kernelx4_ooq_events(context, device);

    go_kernel_ioqx4(context, device);

    return 0;
}