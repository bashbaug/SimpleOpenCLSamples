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

using element = cl_uint4;

size_t  gwx = 4096;
size_t  tile = gwx;

static const char kernelString[] = R"CLC(
kernel void Add1( read_write image2d_t img, int startX, int startY )
{
    int x = get_global_id(0) + startX;
    int y = get_global_id(1) + startY;
    int2 coord = (int2)(x, y);
    uint4 data = read_imageui(img, coord) + 1;
    write_imageui(img, coord, data);
}
kernel void Add2( read_write image2d_t img, int startX, int startY )
{
    int x = get_global_id(0) + startX;
    int y = get_global_id(1) + startY;
    int2 coord = (int2)(x, y);
    uint4 data = read_imageui(img, coord) + 2;
    write_imageui(img, coord, data);
}
kernel void Add3( read_write image2d_t img, int startX, int startY )
{
    uint x = get_global_id(0) + startX;
    uint y = get_global_id(1) + startY;
    int2 coord = (int2)(x, y);
    uint4 data = read_imageui(img, coord) + 3;
    write_imageui(img, coord, data);
}
)CLC";

static void go_ioq(cl::Context& context, cl::Device& device, cl::Image2D& img, std::vector<cl::Kernel>& kernels)
{
    printf("%s: ", __FUNCTION__); fflush(stdout);

    cl_command_queue_properties props = device.getInfo<CL_DEVICE_QUEUE_PROPERTIES>();
    if (props & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) {
        // Create a dummy out-of-order queue to enable command aggregation.
        cl::CommandQueue dummy{context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE};
    }

    cl::CommandQueue queue(context, device);

    for (auto& kernel : kernels) {
        kernel.setArg(0, img);
    }

    cl_uint4 pattern = {};
    queue.enqueueFillImage(img, pattern, {0, 0, 0}, {gwx, gwx, 1});
    queue.finish();

    float best = 999.0f;
    for (int test = 0; test < testIterations; test++) {
        auto start = test_clock::now();
        for (size_t startX = 0; startX < gwx; startX += tile) {
            for (size_t startY = 0; startY < gwx; startY += tile) {
                for (auto& kernel : kernels) {
                    kernel.setArg(1, (cl_int)startX);
                    kernel.setArg(2, (cl_int)startY);
                    queue.enqueueNDRangeKernel(
                        kernel,
                        cl::NullRange,
                        cl::NDRange{tile, tile});
                }
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
        size_t rowPitch;
        cl_uint* p = (cl_uint*)queue.enqueueMapImage(
            img,
            CL_TRUE,
            CL_MAP_READ,
            {0, 0, 0},
            {gwx, gwx, 1},
            &rowPitch,
            nullptr);
        printf("Check: [0] = %u, [1] = %u, ... [n-2] = %u, [n-1] = %u\n",
            p[0], p[1], p[gwx * gwx * 4 - 2], p[gwx * gwx * 4 - 1]);
        queue.enqueueUnmapMemObject(
            img,
            p );
        queue.finish();
    }
}

static void go_ioq_gwo(cl::Context& context, cl::Device& device, cl::Image2D& img, std::vector<cl::Kernel>& kernels)
{
    printf("%s: ", __FUNCTION__); fflush(stdout);

    cl_command_queue_properties props = device.getInfo<CL_DEVICE_QUEUE_PROPERTIES>();
    if (props & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) {
        // Create a dummy out-of-order queue to enable command aggregation.
        cl::CommandQueue dummy{context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE};
    }

    cl::CommandQueue queue(context, device);

    for (auto& kernel : kernels) {
        kernel.setArg(0, img);
        kernel.setArg(1, 0);
        kernel.setArg(2, 0);
    }

    cl_uint4 pattern = {};
    queue.enqueueFillImage(img, pattern, {0, 0, 0}, {gwx, gwx, 1});
    queue.finish();

    float best = 999.0f;
    for (int test = 0; test < testIterations; test++) {
        auto start = test_clock::now();
        for (size_t startX = 0; startX < gwx; startX += tile) {
            for (size_t startY = 0; startY < gwx; startY += tile) {
                for (auto& kernel : kernels) {
                    queue.enqueueNDRangeKernel(
                        kernel,
                        cl::NDRange{startX, startY},
                        cl::NDRange{tile, tile});
                }
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
        size_t rowPitch;
        cl_uint* p = (cl_uint*)queue.enqueueMapImage(
            img,
            CL_TRUE,
            CL_MAP_READ,
            {0, 0, 0},
            {gwx, gwx, 1},
            &rowPitch,
            nullptr);
        printf("Check: [0] = %u, [1] = %u, ... [n-2] = %u, [n-1] = %u\n",
            p[0], p[1], p[gwx * gwx * 4 - 2], p[gwx * gwx * 4 - 1]);
        queue.enqueueUnmapMemObject(
            img,
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
    std::string buildOptions;

    {
        popl::OptionParser op("Supported Options");
        op.add<popl::Value<int>>("p", "platform", "Platform Index", platformIndex, &platformIndex);
        op.add<popl::Value<int>>("d", "device", "Device Index", deviceIndex, &deviceIndex);
        op.add<popl::Value<std::string>>("", "options", "Program Build Options", buildOptions, &buildOptions);
        op.add<popl::Value<int>>("i", "iterations", "Kernel Iterations", numIterations, &numIterations);
        op.add<popl::Value<size_t>>("s", "size", "Total Buffer Size", gwx, &gwx);
        op.add<popl::Value<size_t>>("t", "tile", "Tile Size", tile, &tile);
        //op.add<popl::Switch>("", "uncached", "Allocate an Uncached Buffer", &uncached);

        bool printUsage = false;
        try {
            op.parse(argc, argv);
        } catch (std::exception& e) {
            fprintf(stderr, "Error: %s\n\n", e.what());
            printUsage = true;
        }

        if (printUsage || !op.unknown_options().empty() || !op.non_option_args().empty()) {
            fprintf(stderr,
                "Usage: pipeline2dimg [options]\n"
                "%s", op.help().c_str());
            return -1;
        }
    }

    printf("Image Dimensions are: %zu x %zu\n", gwx, gwx);
    printf("Total Buffer Size is: %zu bytes\n", sizeof(element) * gwx * gwx);
    printf("Tile Dimensions are: %zu x %zu\n", tile, tile);
    printf("Total Tile Size is: %zu bytes\n", sizeof(element) * tile * tile);

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

    printf("Building program with build options: %s\n",
        buildOptions.empty() ? "(none)" : buildOptions.c_str() );
    cl::Program program{ context, kernelString };
    program.build(buildOptions.c_str());
    cl::Kernel Add1 = cl::Kernel{ program, "Add1" };
    cl::Kernel Add2 = cl::Kernel{ program, "Add2" };
    cl::Kernel Add3 = cl::Kernel{ program, "Add3" };

    cl::Image2D img{
        context,
        CL_MEM_READ_WRITE,
        cl::ImageFormat{CL_RGBA, CL_UNSIGNED_INT32},
        gwx, gwx };

    std::vector<cl::Kernel> pipeline;
    pipeline.push_back(Add1);
    pipeline.push_back(Add2);
    pipeline.push_back(Add3);

    //go_ioq(context, device, img, pipeline);
    go_ioq_gwo(context, device, img, pipeline);

    return 0;
}
