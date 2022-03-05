/*
// Copyright (c) 2019-2022 Ben Ashbaugh
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

void test_clGetDeviceIDs(cl::Platform& platform)
{
    const size_t iterations = 1024 * 1024;
    printf("Testing %s (%zu): ", __FUNCTION__, iterations); fflush(stdout);

    cl_platform_id p = platform();

    auto start = std::chrono::system_clock::now();
    for( int i = 0; i < iterations; i++ )
    {
        cl_uint numDevices = 0;
        clGetDeviceIDs(
            p,
            CL_DEVICE_TYPE_ALL,
            0,
            NULL,
            &numDevices);
    }
    auto end = std::chrono::system_clock::now();

    std::chrono::duration<float> elapsed_seconds = end - start;
    float ms = elapsed_seconds.count() * 1000;
    printf("finished in %f ms\n", ms);
}

void test_clGetDeviceInfo(cl::Device& device)
{
    const size_t iterations = 1024 * 1024;
    printf("Testing %s (%zu): ", __FUNCTION__, iterations); fflush(stdout);

    cl_device_id d = device();

    auto start = std::chrono::system_clock::now();
    for( int i = 0; i < iterations; i++ )
    {
        cl_device_type type = 0;
        clGetDeviceInfo(
            d,
            CL_DEVICE_TYPE,
            sizeof(type),
            &type,
            NULL);
    }
    auto end = std::chrono::system_clock::now();

    std::chrono::duration<float> elapsed_seconds = end - start;
    float ms = elapsed_seconds.count() * 1000;
    printf("finished in %f ms\n", ms);
}

void test_clSetKernelArg(cl::Device& device)
{
    const size_t iterations = 1024 * 1024;
    printf("Testing %s (%zu): ", __FUNCTION__, iterations); fflush(stdout);

    cl::Context context{device};

    static const char kernelString[] = R"CLC( kernel void Empty(int a) {} )CLC";

    cl::Program program{context, kernelString};
    program.build();
    cl::Kernel kernel = cl::Kernel{program, "Empty"};

    cl_kernel k = kernel();

    auto start = std::chrono::system_clock::now();
    for( int i = 0; i < iterations; i++ )
    {
        int x = 0;
        clSetKernelArg(
            k,
            0,
            sizeof(x),
            &x);
    }
    auto end = std::chrono::system_clock::now();

    std::chrono::duration<float> elapsed_seconds = end - start;
    float ms = elapsed_seconds.count() * 1000;
    printf("finished in %f ms\n", ms);
}

int main(
    int argc,
    char** argv )
{
    int platformIndex = 0;
    int deviceIndex = 0;
    int partitionType = 0;

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
                "Usage: subdevices[options]\n"
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

    //test_clGetDeviceIDs(platforms[platformIndex]);
    //test_clGetDeviceInfo(devices[deviceIndex]);
    test_clSetKernelArg(devices[deviceIndex]);

    return 0;
 }