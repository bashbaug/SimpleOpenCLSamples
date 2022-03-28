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

#include <benchmark/benchmark.h>
#include <popl/popl.hpp>

#include <CL/opencl.hpp>
#include "util.hpp"

struct OpenCLBenchmarkEnvironment
{
    void ParseArgs(int argc, char** argv)
    {
        int platformIndex = 0;
        int deviceIndex = 0;

        bool printHelp = false;

        popl::OptionParser op("OpenCL Fixture Options");
        op.add<popl::Value<int>>("p", "platform", "Platform Index", platformIndex, &platformIndex);
        op.add<popl::Value<int>>("d", "device", "Device Index", deviceIndex, &deviceIndex);
        op.add<popl::Switch, popl::Attribute::hidden>("h", "help", "Print Help", &printHelp);
        bool printUsage = false;
        try {
            op.parse(argc, argv);
        } catch (std::exception& e) {
            fprintf(stderr, "Error: %s\n\n", e.what());
            printUsage = true;
        }
        if (printUsage || printHelp) {
            fprintf(stderr, "%s", op.help().c_str());
        }

        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);

        platform = platforms[platformIndex];
        printf("Running on platform: %s\n",
            platform.getInfo<CL_PLATFORM_NAME>().c_str() );

        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

        device = devices[deviceIndex];
        printf("Running on device: %s\n",
            device.getInfo<CL_DEVICE_NAME>().c_str() );

        context = cl::Context{device};

        ioq = cl::CommandQueue{context};
        ooq = cl::CommandQueue{context, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE};
    }

    cl::Platform platform;
    cl::Device device;
    cl::Context context;
    cl::CommandQueue ioq;
    cl::CommandQueue ooq;
};

OpenCLBenchmarkEnvironment env;

struct Platform : public benchmark::Fixture
{
    cl::Platform platform;

    virtual void SetUp(benchmark::State& state) override {
        platform = env.platform;
    }
    virtual void TearDown(benchmark::State& state) override {
        platform = NULL;
    }
};

BENCHMARK_DEFINE_F(Platform, clGetDeviceIDs)(benchmark::State& state)
{
    while(state.KeepRunning()) {
        cl_uint numDevices = 0;
        clGetDeviceIDs(
            platform(),
            CL_DEVICE_TYPE_ALL,
            0,
            NULL,
            &numDevices);
    }
}
BENCHMARK_REGISTER_F(Platform, clGetDeviceIDs);

struct Device : public benchmark::Fixture
{
    cl::Device device;

    virtual void SetUp(benchmark::State& state) override {
        device = env.device;
    }
    virtual void TearDown(benchmark::State& state) override {
        device = NULL;
    }
};

BENCHMARK_DEFINE_F(Device, clGetDeviceInfo)(benchmark::State& state)
{
    while(state.KeepRunning()) {
        cl_device_type type = 0;
        clGetDeviceInfo(
            device(),
            CL_DEVICE_TYPE,
            sizeof(type),
            &type,
            NULL);
    }
}
BENCHMARK_REGISTER_F(Device, clGetDeviceInfo);

struct Kernel : public benchmark::Fixture
{
    cl::Program program;
    cl::Kernel kernel;

    virtual void SetUp(benchmark::State& state) override {
        static const char kernelString[] = R"CLC( kernel void Empty(int a) {} )CLC";

        program = cl::Program{env.context, kernelString};

        program.build();
        kernel = cl::Kernel{program, "Empty"};
    }
    virtual void TearDown(benchmark::State& state) override {
        program = NULL;
        kernel = NULL;
    }
};

BENCHMARK_DEFINE_F(Kernel, clSetKernelArg)(benchmark::State& state)
{
    while(state.KeepRunning()) {
        int x = 0;
        clSetKernelArg(
            kernel(),
            0,
            sizeof(x),
            &x);
    }
}
BENCHMARK_REGISTER_F(Kernel, clSetKernelArg);

struct SVMKernel : public benchmark::Fixture
{
    cl::Program program;
    cl::Kernel kernel;

    std::vector<void*>  ptrs;

    virtual void SetUp(benchmark::State& state) override {
        static const char kernelString[] = R"CLC( kernel void Empty(global int* a) {} )CLC";

        program = cl::Program{env.context, kernelString};

        program.build();
        kernel = cl::Kernel{program, "Empty"};

        for (int i = 0; i < 16; i++) {
            ptrs.push_back(clSVMAlloc(env.context(), CL_MEM_READ_WRITE, 128, 0));
        }
    }
    virtual void TearDown(benchmark::State& state) override {
        program = NULL;
        kernel = NULL;

        for(auto ptr : ptrs) {
            clSVMFree(env.context(), ptr);
        }
    }
};

BENCHMARK_DEFINE_F(SVMKernel, clSetKernelArgSVMPointer)(benchmark::State& state)
{
    const int mask = (int)state.range(0) - 1;
    int i = 0;
    while(state.KeepRunning()) {
        clSetKernelArgSVMPointer(
            kernel(),
            0,
            ptrs[i&mask]);
        ++i;
    }
}
BENCHMARK_REGISTER_F(SVMKernel, clSetKernelArgSVMPointer)->Arg(1)->Arg(2)->Arg(4);

struct USMMemCpy : public benchmark::Fixture
{
    bool    hasSupport = false;

    cl::CommandQueue queue;

    std::vector<void*>  dptrs;
    std::vector<void*>  hptrs;
    std::vector<void*>  sptrs;

    virtual void SetUp(benchmark::State& state) override {
        queue = env.ioq;

        for (int i = 0; i < 16; i++) {
            dptrs.push_back(clDeviceMemAllocINTEL(env.context(), env.device(), NULL, 128, 0, NULL));
            hptrs.push_back(clHostMemAllocINTEL(env.context(), NULL, 128, 0, NULL));
            sptrs.push_back(clSharedMemAllocINTEL(env.context(), env.device(), NULL, 128, 0, NULL));
        }
    }
    virtual void TearDown(benchmark::State& state) override {
        queue = NULL;

        for(auto ptr : dptrs) {
            clMemBlockingFreeINTEL(env.context(), ptr);
        }
        for(auto ptr : hptrs) {
            clMemBlockingFreeINTEL(env.context(), ptr);
        }
        for(auto ptr : sptrs) {
            clMemBlockingFreeINTEL(env.context(), ptr);
        }
    }
};

BENCHMARK_DEFINE_F(USMMemCpy, clEnqueueMemcpyINTEL_device_blocking)(benchmark::State& state)
{
    if (dptrs[0] == NULL || dptrs[1] == NULL) {
        state.SkipWithError("unsupported");
    }
    while(state.KeepRunning()) {
        clEnqueueMemcpyINTEL(
            queue(),
            CL_TRUE,
            dptrs[0],
            dptrs[1],
            128,
            0,
            NULL,
            NULL);
    }
}
BENCHMARK_REGISTER_F(USMMemCpy, clEnqueueMemcpyINTEL_device_blocking);

int main(int argc, char** argv)
{
    env.ParseArgs(argc, argv);

    ::benchmark::Initialize(&argc, argv);
    //if (::benchmark::ReportUnrecognizedArguments(argc, argv)) {
    //    return 1;
    //}
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    return 0;
}
