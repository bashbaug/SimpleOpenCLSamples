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
kernel void early_exit(global float* dst, int iterations)
{
    int index = get_group_id(0) * get_local_size(0) + get_local_id(0);
    if (get_local_id(0) == 0) {
        float result;
        for (int i = 0; i < iterations; i++) {
            result = 0.0f;
            while (result < 1.0f) result += 1e-3f;
        }
        dst[index] = result;

        // divergent
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

kernel void divergent_then_convergent(global float* dst, int iterations)
{
    int index = get_group_id(0) * get_local_size(0) + get_local_id(0);
    if (get_local_id(0) == 0) {
        float result;
        for (int i = 0; i < iterations; i++) {
            result = 0.0f;
            while (result < 1.0f) result += 1e-3f;
        }
        dst[index] = result;

        // divergent
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // convergent
    barrier(CLK_LOCAL_MEM_FENCE);
    dst[index] += 1.0f;
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
                "Usage: barriers [options]\n"
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
    cl::Kernel early_exit = cl::Kernel{ program, "early_exit" };
    cl::Kernel divergent_then_convergent = cl::Kernel{ program, "divergent_then_convergent" };

    cl::Buffer p = cl::Buffer{context, CL_MEM_ALLOC_HOST_PTR, 1024 * sizeof(float)};

    printf("Running the early_exit kernel..."); fflush(stdout);
    early_exit.setArg(0, p);
    early_exit.setArg(1, 10);
    commandQueue.enqueueNDRangeKernel(
        early_exit,
        cl::NullRange,
        cl::NDRange{1024},
        cl::NDRange{256});
    commandQueue.finish();
    printf(" done!\n");

    printf("Running the divergent_then_convergent kernel..."); fflush(stdout);
    divergent_then_convergent.setArg(0, p);
    divergent_then_convergent.setArg(1, 10);
    commandQueue.enqueueNDRangeKernel(
        divergent_then_convergent,
        cl::NullRange,
        cl::NDRange{1024},
        cl::NDRange{256});
    commandQueue.finish();
    printf(" done!\n");

    return 0;
}
