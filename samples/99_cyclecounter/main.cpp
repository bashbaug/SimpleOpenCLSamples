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

static const char kernelString[] = R"CLC(
ulong __attribute__((overloadable)) intel_get_cycle_counter(void);

kernel void cyclecounter(global float* r)
{
    ulong start = intel_get_cycle_counter();

    // waste a fair bit of time
    float reg;
    for (int i = 0; i < 10; i++) {
        reg = 0.0f;
        while (reg < 1.0f) {
            reg += 1e-7f;
        }
    }
    ulong end = intel_get_cycle_counter();

    r[0] = start;
    r[1] = end;
    r[2] = reg;
}
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
                "Usage: cyclecounter [options]\n"
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
    cl::Kernel kernel = cl::Kernel{ program, "cyclecounter" };

    cl::Buffer result = cl::Buffer{
        context,
        CL_MEM_ALLOC_HOST_PTR,
        3 * sizeof(cl_float) };
    float pattern = 42.0f;
    commandQueue.enqueueFillBuffer(result, pattern, 0, 3 * sizeof(pattern));

    // execution
    kernel.setArg(0, result);
    commandQueue.enqueueNDRangeKernel(
        kernel,
        cl::NullRange,
        cl::NDRange{1});

    {
        auto ptr = (const cl_float*)commandQueue.enqueueMapBuffer(
            result,
            CL_TRUE,
            CL_MAP_READ,
            0,
            3 * sizeof(cl_float) );

        printf("Result = %f\n", ptr[2]);
        printf("Start = %f, End = %f, Delta = %f\n", ptr[0], ptr[1], ptr[1] - ptr[0]);

        commandQueue.enqueueUnmapMemObject(
            result,
            (void*)ptr );
        commandQueue.finish();
    }

    printf("Done.\n");

    return 0;
}
