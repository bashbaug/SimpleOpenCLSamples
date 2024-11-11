/*
// Copyright (c) 2019-2024 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <popl/popl.hpp>

#include <CL/opencl.hpp>

#include <chrono>

void test_clGetDeviceIDs(cl::Platform& platform)
{
    const size_t iterations = 1024 * 1024;
    printf("Testing %s (%zu): ", __FUNCTION__, iterations); fflush(stdout);

    cl_platform_id p = platform();

    float ms = 0.0f;
    for( int w = 0; w < 2; w++ )
    {
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
        ms = elapsed_seconds.count() * 1000;
    }

    printf("finished in %f ms (%f ns/iteration)\n", ms, ms * iterations / 1000000.0f);
}

void test_clGetDeviceInfo(cl::Device& device)
{
    const size_t iterations = 1024 * 1024;
    printf("Testing %s (%zu): ", __FUNCTION__, iterations); fflush(stdout);

    cl_device_id d = device();

    float ms = 0.0f;
    for( int w = 0; w < 2; w++ )
    {
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
        ms = elapsed_seconds.count() * 1000;
    }

    printf("finished in %f ms (%f ns/iteration)\n", ms, ms * iterations / 1000000.0f);
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

    float ms = 0.0f;
    for( int w = 0; w < 2; w++ )
    {
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
        ms = elapsed_seconds.count() * 1000;
    }

    printf("finished in %f ms (%f ns/iteration)\n", ms, ms * iterations / 1000000.0f);
}

void test_clSetKernelArgSVMPointer(cl::Device& device)
{
    const size_t iterations = 1024 * 1024;
    printf("Testing %s (%zu): ", __FUNCTION__, iterations); fflush(stdout);

    cl::Context context{device};

    std::vector<void*>  ptrs;
    ptrs.push_back(clSVMAlloc(context(), CL_MEM_READ_WRITE, 128, 0));
    ptrs.push_back(clSVMAlloc(context(), CL_MEM_READ_WRITE, 128, 0));

    static const char kernelString[] = R"CLC( kernel void Empty(global int* a) {} )CLC";

    cl::Program program{context, kernelString};
    program.build();
    cl::Kernel kernel = cl::Kernel{program, "Empty"};

    cl_kernel k = kernel();

    float ms = 0.0f;
    for( int w = 0; w < 2; w++ )
    {
        auto start = std::chrono::system_clock::now();
        for( int i = 0; i < iterations; i++ )
        {
            clSetKernelArgSVMPointer(
                k,
                0,
                ptrs[i&1] );
        }
        auto end = std::chrono::system_clock::now();

        std::chrono::duration<float> elapsed_seconds = end - start;
        ms = elapsed_seconds.count() * 1000;
    }

    printf("finished in %f ms (%f ns/iteration)\n", ms, ms * iterations / 1000000.0f);

    for (auto ptr : ptrs)
    {
        clSVMFree(context(), ptr);
    }
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

    test_clGetDeviceIDs(platforms[platformIndex]);
    test_clGetDeviceInfo(devices[deviceIndex]);
    test_clSetKernelArg(devices[deviceIndex]);
    test_clSetKernelArgSVMPointer(devices[deviceIndex]);

    return 0;
 }