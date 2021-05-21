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

#include <chrono>

#define CL_MEM_FLAGS_INTEL 0x10001
#define CL_MEM_LOCALLY_UNCACHED_RESOURCE (1 << 18)
#define CL_MEM_LOCALLY_UNCACHED_SURFACE_STATE_RESOURCE (1 << 25)

using test_clock = std::chrono::high_resolution_clock;

constexpr int testIterations = 32;
int numIterations = 1;

size_t  gwx = 16*1024*1024;
size_t  tile = gwx;

static const char kernelString[] = R"CLC(
kernel void Add1( global uint4* ptr, uint start )
{
    uint id = get_global_id(0) + start;
    ptr[id] = ptr[id] + 1;
}
kernel void Add2( global uint4* ptr, uint start )
{
    uint id = get_global_id(0) + start;
    ptr[id] = ptr[id] + 2;
}
kernel void Add3( global uint4* ptr, uint start )
{
    uint id = get_global_id(0) + start;
    ptr[id] = ptr[id] + 3;
}
)CLC";

static void go_ioq(cl::Context& context, cl::Device& device, cl::Buffer& buffer, std::vector<cl::Kernel>& kernels)
{
    printf("%s: ", __FUNCTION__); fflush(stdout);

    cl_command_queue_properties props = device.getInfo<CL_DEVICE_QUEUE_PROPERTIES>();
    if (props & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) {
        // Create a dummy out-of-order queue to enable command aggregation.
        cl::CommandQueue dummy{context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE};
    }

    cl::CommandQueue queue(context, device);

    for (auto& kernel : kernels) {
        kernel.setArg(0, buffer);
    }

    int pattern = 0;
    queue.enqueueFillBuffer(buffer, pattern, 0, gwx * sizeof(pattern));
    queue.finish();

    float best = 999.0f;
    for (int test = 0; test < testIterations; test++) {
        auto start = test_clock::now();
        for (size_t start = 0; start < gwx; start += tile) {
            for (auto& kernel : kernels) {
                kernel.setArg(1, (int)start);
                queue.enqueueNDRangeKernel(
                    kernel,
                    cl::NullRange,
                    cl::NDRange{tile});
            }
        }
        queue.finish();

        auto end = test_clock::now();
        std::chrono::duration<float> elapsed_seconds = end - start;
        best = std::min(best, elapsed_seconds.count());
    }
    printf("Finished in %f seconds\n", best);

    // verification
    {
        cl_uint* p = (cl_uint*)queue.enqueueMapBuffer(
            buffer,
            CL_TRUE,
            CL_MAP_READ,
            0,
            gwx * sizeof(cl_uint4) );
        printf("Check: [0] = %u, [1] = %u, ... [n-2] = %u, [n-1] = %u\n",
            p[0], p[1], p[gwx * 4 - 2], p[gwx * 4 - 1]);
        queue.enqueueUnmapMemObject(
            buffer,
            p );
        queue.finish();
    }
}

static void go_ioq_gwo(cl::Context& context, cl::Device& device, cl::Buffer& buffer, std::vector<cl::Kernel>& kernels)
{
    printf("%s: ", __FUNCTION__); fflush(stdout);

    cl_command_queue_properties props = device.getInfo<CL_DEVICE_QUEUE_PROPERTIES>();
    if (props & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) {
        // Create a dummy out-of-order queue to enable command aggregation.
        cl::CommandQueue dummy{context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE};
    }

    cl::CommandQueue queue(context, device);

    for (auto& kernel : kernels) {
        kernel.setArg(0, buffer);
        kernel.setArg(1, 0);
    }

    int pattern = 0;
    queue.enqueueFillBuffer(buffer, pattern, 0, gwx * sizeof(pattern));
    queue.finish();

    float best = 999.0f;
    for (int test = 0; test < testIterations; test++) {
        auto start = test_clock::now();
        for (size_t start = 0; start < gwx; start += tile) {
            for (auto& kernel : kernels) {
                queue.enqueueNDRangeKernel(
                    kernel,
                    cl::NDRange{start},
                    cl::NDRange{tile});
            }
        }
        queue.finish();

        auto end = test_clock::now();
        std::chrono::duration<float> elapsed_seconds = end - start;
        best = std::min(best, elapsed_seconds.count());
    }
    printf("Finished in %f seconds\n", best);

    // verification
    {
        cl_uint* p = (cl_uint*)queue.enqueueMapBuffer(
            buffer,
            CL_TRUE,
            CL_MAP_READ,
            0,
            gwx * sizeof(cl_uint4) );
        printf("Check: [0] = %u, [1] = %u, ... [n-2] = %u, [n-1] = %u\n",
            p[0], p[1], p[gwx * 4 - 2], p[gwx * 4 - 1]);
        queue.enqueueUnmapMemObject(
            buffer,
            p );
        queue.finish();
    }
}

int main(
    int argc,
    char** argv )
{
    int platformIndex = 0;
    int deviceIndex = 0;
    bool uncached = false;

    {
        popl::OptionParser op("Supported Options");
        op.add<popl::Value<int>>("p", "platform", "Platform Index", platformIndex, &platformIndex);
        op.add<popl::Value<int>>("d", "device", "Device Index", deviceIndex, &deviceIndex);
        op.add<popl::Value<int>>("i", "iterations", "Kernel Iterations", numIterations, &numIterations);
        op.add<popl::Value<size_t>>("s", "size", "Total Buffer Size", gwx, &gwx);
        op.add<popl::Value<size_t>>("t", "tile", "Tile Size", tile, &tile);
        op.add<popl::Switch>("", "uncached", "Allocate an Uncached Buffer", &uncached);

        bool printUsage = false;
        try {
            op.parse(argc, argv);
        } catch (std::exception& e) {
            fprintf(stderr, "Error: %s\n\n", e.what());
            printUsage = true;
        }

        if (printUsage || !op.unknown_options().empty() || !op.non_option_args().empty()) {
            fprintf(stderr,
                "Usage: pipeline [options]\n"
                "%s", op.help().c_str());
            return -1;
        }
    }

    printf("Buffer Size is: %zu\n", gwx);
    printf("Tile Size is: %zu\n", tile);

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    printf("Running on platform: %s\n",
        platforms[platformIndex].getInfo<CL_PLATFORM_NAME>().c_str() );

    std::vector<cl::Device> devices;
    platforms[platformIndex].getDevices(CL_DEVICE_TYPE_ALL, &devices);

    cl::Device& device = devices[deviceIndex];
    printf("Running on device: %s\n",
        devices[deviceIndex].getInfo<CL_DEVICE_NAME>().c_str() );

    cl::Context context{devices[deviceIndex]};
    cl::CommandQueue commandQueue = cl::CommandQueue{context, devices[deviceIndex]};

    cl::Program program{ context, kernelString };
    program.build();
    cl::Kernel Add1 = cl::Kernel{ program, "Add1" };
    cl::Kernel Add2 = cl::Kernel{ program, "Add2" };
    cl::Kernel Add3 = cl::Kernel{ program, "Add3" };

    cl::Buffer buf;
    if (uncached) {
        cl_mem_properties props[] = {
            CL_MEM_FLAGS_INTEL, CL_MEM_LOCALLY_UNCACHED_SURFACE_STATE_RESOURCE,
            0
        };
        buf = cl::Buffer{
            clCreateBufferWithProperties(
                context(),
                props,
                CL_MEM_ALLOC_HOST_PTR,
                gwx * sizeof(cl_uint4),
                NULL,
                NULL) };
    } else {
        buf = cl::Buffer{
            context,
            CL_MEM_ALLOC_HOST_PTR,
            gwx * sizeof(cl_uint4) };
    }

    std::vector<cl::Kernel> pipeline;
    pipeline.push_back(Add1);
    pipeline.push_back(Add2);
    pipeline.push_back(Add3);

    //go_ioq(context, device, buf, pipeline);
    go_ioq_gwo(context, device, buf, pipeline);

    return 0;
}
