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

const size_t    gwx = 1024*1024;

static const char kernelString[] = R"CLC(
kernel void Empty( global uint* ptr ) {}
)CLC";

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
                "Usage: overlapmem [options]\n"
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

    cl::Context context{devices[deviceIndex]};
    cl::CommandQueue commandQueue = cl::CommandQueue{context, devices[deviceIndex]};

    cl::Program program{ context, kernelString };
    program.build();
    cl::Kernel kernel = cl::Kernel{ program, "Empty" };

    // Allocate some host memory.
    const size_t elements = 1024 * 1024;
    std::vector<int> v(elements);

    printf("Creating a buffer for the entire host memory allocation...\n");
    cl::Buffer everything = cl::Buffer{
        context,
        CL_MEM_USE_HOST_PTR,
        elements * sizeof(v[0]),
        v.data()};
    printf("... buffer is %p\n\n", everything());

    printf("Creating a buffer for the first half of the allocation...\n");
    cl::Buffer first = cl::Buffer{
        context,
        CL_MEM_USE_HOST_PTR,
        elements / 2 * sizeof(v[0]),
        v.data()};
    printf("... buffer is %p\n\n", first());

    printf("Creating a buffer for the second half of the allocation...\n");
    cl::Buffer second = cl::Buffer{
        context,
        CL_MEM_USE_HOST_PTR,
        elements / 2 * sizeof(v[0]),
        v.data() + elements / 2};
    printf("... buffer is %p\n\n", second());

    printf("Creating a buffer for the middle of the allocation...\n");
    cl::Buffer middle = cl::Buffer{
        context,
        CL_MEM_USE_HOST_PTR,
        elements / 2 * sizeof(v[0]),
        v.data() + elements / 4};
    printf("... buffer is %p\n\n", middle());

    printf("Using the entire buffer...\n");
    kernel.setArg(0, everything);
    commandQueue.enqueueNDRangeKernel(
        kernel,
        cl::NullRange,
        cl::NDRange{256});
    commandQueue.finish();

    printf("Using the first half buffer...\n");
    kernel.setArg(0, first);
    commandQueue.enqueueNDRangeKernel(
        kernel,
        cl::NullRange,
        cl::NDRange{256});
    commandQueue.finish();

    printf("Using the second half buffer...\n");
    kernel.setArg(0, second);
    commandQueue.enqueueNDRangeKernel(
        kernel,
        cl::NullRange,
        cl::NDRange{256});
    commandQueue.finish();

    printf("Using the middle buffer...\n");
    kernel.setArg(0, middle);
    commandQueue.enqueueNDRangeKernel(
        kernel,
        cl::NullRange,
        cl::NDRange{256});
    commandQueue.finish();

    printf("Done!\n");

    return 0;
}
