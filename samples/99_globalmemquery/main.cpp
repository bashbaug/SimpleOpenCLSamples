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
kernel void TouchBuffer(global int* buf, ulong sz)
{
    sz /= sizeof(int);

    buf[0]      = 0x12345678;
    buf[sz - 1] = 0xABCDEF01;
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
                "Usage: globalmemquery [options]\n"
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

    size_t globalMemSize = devices[deviceIndex].getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
    printf("CL_DEVICE_GLOBAL_MEM_SIZE for this device is %zu bytes.\n", globalMemSize);

    size_t maxMemAllocSize = devices[deviceIndex].getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
    printf("CL_DEVICE_MAX_MEM_ALLOC_SIZE for this device is %zu bytes.\n", maxMemAllocSize);

    cl::Context context{devices[deviceIndex]};
    cl::CommandQueue queue = cl::CommandQueue{context, devices[deviceIndex]};
    cl::Program program{context, kernelString};

    program.build();

    cl::Kernel kernel{program, "TouchBuffer"};

    for (size_t i = 2; i < 64; i++) {
        size_t allocSize = (size_t)1 << i;
        if (allocSize > globalMemSize) {
            break;
        }
        if (allocSize > maxMemAllocSize) {
            printf("Size is greater than max mem alloc size, this might not work...\n");
        }
        printf("Trying to allocate %zu bytes... ", allocSize); fflush(stdout);
        int errorCode = CL_SUCCESS;
        cl::Buffer buf = cl::Buffer{
            context,
            CL_MEM_READ_WRITE,
            allocSize,
            nullptr,
            &errorCode};
        if (errorCode == CL_SUCCESS) {
            printf("success!\n");
        } else {
            printf("failed.\n");
            continue;
        }

        kernel.setArg(0, buf);
        kernel.setArg(1, (cl_ulong)allocSize);
        errorCode = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange{1});
        if (errorCode != CL_SUCCESS) {
            printf("clEnqueueNDRangeKernel failed.\n");
        }
        errorCode = queue.finish();
        if (errorCode != CL_SUCCESS) {
            printf("clFinish failed.\n");
        }
    }

    return 0;
}