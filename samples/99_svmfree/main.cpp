/*
// Copyright (c) 2025 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <popl/popl.hpp>

#include <CL/opencl.hpp>

#include <chrono>

using test_clock = std::chrono::steady_clock;

static const char kernelString[] = R"CLC(
kernel void eat_time(global int* ptr, int kernelOperationsCount )
{
    volatile int value = kernelOperationsCount;
    while (--value) {}
    if (get_global_id(0) > 10000000) {
        ptr[0] = 0;
    }
}
)CLC";

int kernelOperationsCount = 10000000;

static void test_malloc(cl::Context& context, cl::CommandQueue& queue, cl::Kernel& kernel, size_t count)
{
    std::cout << "Testing SVM alloc for " << count * sizeof(int) << " bytes:\n";

    auto ptr = (int*)clSVMAlloc(
        context(),
        CL_MEM_READ_WRITE,
        count * sizeof(int),
        0);
    
    kernel.setArg(0, ptr);
    kernel.setArg(1, kernelOperationsCount);

    auto start = test_clock::now();
    queue.enqueueNDRangeKernel(
        kernel,
        cl::NullRange,
        cl::NDRange{1024},
        cl::NDRange{1});
    auto submit = test_clock::now();
    auto another = (int*)clSVMAlloc(
        context(),
        CL_MEM_READ_WRITE,
        count * sizeof(int),
        0);
    auto alloc = test_clock::now();
    queue.finish();
    auto wait = test_clock::now();

    std::cout << "\tAlloc time: " << std::chrono::duration_cast<std::chrono::microseconds>(alloc - submit).count() << " us" << std::endl;
    std::cout << "\tWait time: " << std::chrono::duration_cast<std::chrono::microseconds>(wait - alloc).count() << " us" << std::endl;

    clSVMFree(context(), ptr);
    clSVMFree(context(), another);
}

static void test_free(cl::Context& context, cl::CommandQueue& queue, cl::Kernel& kernel, size_t count, bool inUse)
{
    std::cout << "Testing SVM free while pointer is" << (inUse ? "" : " NOT") << " in use for " << count * sizeof(int) << " bytes:\n";

    auto ptr = (int*)clSVMAlloc(
        context(),
        CL_MEM_READ_WRITE,
        count * sizeof(int),
        0);
    auto another = (int*)clSVMAlloc(
        context(),
        CL_MEM_READ_WRITE,
        count * sizeof(int),
        0);

    int kernelOperationsCount = 10000000;

    kernel.setArg(0, inUse ? ptr : another);
    kernel.setArg(1, kernelOperationsCount);

    auto start = test_clock::now();
    queue.enqueueNDRangeKernel(
        kernel,
        cl::NullRange,
        cl::NDRange{1024},
        cl::NDRange{1});
    auto submit = test_clock::now();
    clSVMFree(context(), ptr);
    auto free = test_clock::now();
    queue.finish();
    auto wait = test_clock::now();

    std::cout << "\tFree time: " << std::chrono::duration_cast<std::chrono::microseconds>(free - submit).count() << " us" << std::endl;
    std::cout << "\tWait time: " << std::chrono::duration_cast<std::chrono::microseconds>(wait - free).count() << " us" << std::endl;

    clSVMFree(context(), another);
}

int main(int argc, char** argv)
{
    int platformIndex = 0;
    int deviceIndex = 0;

    size_t smallCount = 1024;
    size_t largeCount = 32 * 1024 * 1024;

    {
        popl::OptionParser op("Supported Options");
        op.add<popl::Value<int>>("p", "platform", "Platform Index", platformIndex, &platformIndex);
        op.add<popl::Value<int>>("d", "device", "Device Index", deviceIndex, &deviceIndex);
        op.add<popl::Value<int>>("t", "timer", "Timer Value", kernelOperationsCount, &kernelOperationsCount);
        op.add<popl::Value<size_t>>("s", "small", "Small Count", smallCount, &smallCount);
        op.add<popl::Value<size_t>>("l", "large", "Large Count", largeCount, &largeCount);

        bool printUsage = false;
        try {
            op.parse(argc, argv);
        } catch (std::exception& e) {
            fprintf(stderr, "Error: %s\n\n", e.what());
            printUsage = true;
        }

        if (printUsage || !op.unknown_options().empty() || !op.non_option_args().empty()) {
            fprintf(stderr,
                "Usage: svmfree [options]\n"
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
    cl::CommandQueue commandQueue{context, devices[deviceIndex]};

    cl::Program program{ context, kernelString };
    program.build();
    cl::Kernel kernel = cl::Kernel{ program, "eat_time" };

    std::cout << "Testing small SVM allocations...\n";
    test_malloc(context, commandQueue, kernel, smallCount);
    test_free(context, commandQueue, kernel, smallCount, false);
    test_free(context, commandQueue, kernel, smallCount, true);

    std::cout << "Testing large SVM allocations...\n";
    test_malloc(context, commandQueue, kernel, largeCount);
    test_free(context, commandQueue, kernel, largeCount, false);
    test_free(context, commandQueue, kernel, largeCount, true);

    return 0;
}
