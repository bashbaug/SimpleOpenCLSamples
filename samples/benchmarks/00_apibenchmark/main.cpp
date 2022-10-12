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
            fprintf(stderr, "Pass '--help' to view Google Benchmark options.\n");
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
    for(auto _ : state) {
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
    for(auto _ : state) {
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

struct Context : public benchmark::Fixture
{
    cl::Context context;

    virtual void SetUp(benchmark::State& state) override {
        context = env.context;
    }
    virtual void TearDown(benchmark::State& state) override {
        context = NULL;
    }
};

BENCHMARK_DEFINE_F(Context, clCreateBuffer)(benchmark::State& state)
{
    const size_t bufferSize = state.range(0);
    std::vector<cl_uchar> data(bufferSize, 0);

    const size_t maxNumBuffers = 128;
    std::array<cl_mem, maxNumBuffers> buffers;

    size_t count = 0;
    for(auto _ : state) {
        buffers[count++] = clCreateBuffer(
            context(),
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            data.size(),
            data.data(),
            nullptr);

        if (count >= maxNumBuffers) {
            for(size_t i = 0; i < count; i++) {
                clReleaseMemObject(buffers[i]);
            }
            count = 0;
        }
    }

    for(size_t i = 0; i < count; i++) {
        clReleaseMemObject(buffers[i]);
    }
}
BENCHMARK_REGISTER_F(Context, clCreateBuffer)->Arg(64);

BENCHMARK_DEFINE_F(Context, clCreateBuffer_ForceHostMem)(benchmark::State& state)
{
    const size_t bufferSize = state.range(0);
    std::vector<cl_uchar> data(bufferSize, 0);

    const size_t maxNumBuffers = 128;
    std::array<cl_mem, maxNumBuffers> buffers;

    cl_mem test = clCreateBuffer(
        context(),
        CL_MEM_FORCE_HOST_MEMORY_INTEL,
        bufferSize,
        nullptr,
        nullptr);
    if (test) {
        clReleaseMemObject(test);
    } else {
        state.SkipWithError("Couldn't create buffer with CL_MEM_FORCE_HOST_MEMORY_INTEL");
    }

    size_t count = 0;
    for(auto _ : state) {
        buffers[count++] = clCreateBuffer(
            context(),
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_FORCE_HOST_MEMORY_INTEL,
            data.size(),
            data.data(),
            nullptr);

        if (count >= maxNumBuffers) {
            for(size_t i = 0; i < count; i++) {
                clReleaseMemObject(buffers[i]);
            }
            count = 0;
        }
    }

    for(size_t i = 0; i < count; i++) {
        clReleaseMemObject(buffers[i]);
    }
}
BENCHMARK_REGISTER_F(Context, clCreateBuffer_ForceHostMem)->Arg(64);

struct Kernel : public benchmark::Fixture
{
    cl::CommandQueue queue;
    cl::Program program;
    cl::Kernel kernel;

    virtual void SetUp(benchmark::State& state) override {
        queue = env.ioq;

        static const char kernelString[] = R"CLC(
            kernel void Silly(global int* dst) {
                size_t sum = get_local_id(0) + get_local_id(1) + get_local_id(2);
                if (sum > 99999) {
                    dst[0] = 0;
                }
            } )CLC";

        program = cl::Program{env.context, kernelString};

        program.build();
        kernel = cl::Kernel{program, "Silly"};

        kernel.setArg(0, nullptr);
    }
    virtual void TearDown(benchmark::State& state) override {
        program = NULL;
        kernel = NULL;
    }
};

BENCHMARK_DEFINE_F(Kernel, clSetKernelArg)(benchmark::State& state)
{
    for(auto _ : state) {
        int x = 0;
        clSetKernelArg(
            kernel(),
            0,
            sizeof(x),
            &x);
    }
}
BENCHMARK_REGISTER_F(Kernel, clSetKernelArg);

BENCHMARK_DEFINE_F(Kernel, clEnqueueNDRangeKernel_NullQueueError)(benchmark::State& state)
{
    const size_t work_dim = 1;
    const size_t global_work_size[work_dim] = { 1 };
    const size_t local_work_size[work_dim] = { 1 };
    const size_t global_work_offset[work_dim] = { 0 };
    for(auto _ : state) {
        clEnqueueNDRangeKernel(
            NULL,
            kernel(),
            work_dim,
            NULL,
            global_work_size,
            NULL,
            0,
            NULL,
            NULL );
    }
}
BENCHMARK_REGISTER_F(Kernel, clEnqueueNDRangeKernel_NullQueueError);

BENCHMARK_DEFINE_F(Kernel, clEnqueueNDRangeKernel_NullKernelError)(benchmark::State& state)
{
    const size_t work_dim = 1;
    const size_t global_work_size[work_dim] = { 1 };
    const size_t local_work_size[work_dim] = { 1 };
    const size_t global_work_offset[work_dim] = { 0 };
    for(auto _ : state) {
        clEnqueueNDRangeKernel(
            queue(),
            NULL,
            work_dim,
            NULL,
            global_work_size,
            NULL,
            0,
            NULL,
            NULL );
    }
}
BENCHMARK_REGISTER_F(Kernel, clEnqueueNDRangeKernel_NullKernelError);

BENCHMARK_DEFINE_F(Kernel, clEnqueueNDRangeKernel_1x1_NoEvent)(benchmark::State& state)
{
    const int flushFrequency = (int)state.range(0);

    const size_t work_dim = 1;
    const size_t global_work_size[work_dim] = { 1 };
    const size_t local_work_size[work_dim] = { 1 };
    const size_t global_work_offset[work_dim] = { 0 };

    size_t count = 0;

    for(auto _ : state) {
        clEnqueueNDRangeKernel(
            queue(),
            kernel(),
            work_dim,
            NULL,
            global_work_size,
            local_work_size,
            0,
            NULL,
            NULL );
        if (++count % flushFrequency) {
            clFlush(queue());
        }
    }

    clFinish(queue());
}
BENCHMARK_REGISTER_F(Kernel, clEnqueueNDRangeKernel_1x1_NoEvent)->Arg(1)->Arg(32)->Arg(512)->Arg(2048);

BENCHMARK_DEFINE_F(Kernel, clEnqueueNDRangeKernel_1x1_Event)(benchmark::State& state)
{
    const int flushFrequency = (int)state.range(0);

    const size_t work_dim = 1;
    const size_t global_work_size[work_dim] = { 1 };
    const size_t local_work_size[work_dim] = { 1 };
    const size_t global_work_offset[work_dim] = { 0 };

    size_t count = 0;

    for(auto _ : state) {
        cl_event event = NULL;
        clEnqueueNDRangeKernel(
            queue(),
            kernel(),
            work_dim,
            NULL,
            global_work_size,
            local_work_size,
            0,
            NULL,
            &event );
        clReleaseEvent(event);
        if (++count % flushFrequency) {
            clFlush(queue());
        }
    }

    clFinish(queue());
}
BENCHMARK_REGISTER_F(Kernel, clEnqueueNDRangeKernel_1x1_Event)->Arg(1)->Arg(32)->Arg(512)->Arg(2048);

BENCHMARK_DEFINE_F(Kernel, clEnqueueNDRangeKernel_overhead)(benchmark::State& state)
{
    size_t maxWGS = kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(env.device);
    if (maxWGS < 256) {
        state.SkipWithError("kernel work-group size is too small");
    }

    const bool npot = state.range(0) == 0;
    const int nwgs = (int)state.range(1);

    const size_t lwx = npot ? 7 : 8;
    const size_t lwy = npot ? 7 : 8;
    const size_t lwz = npot ? 5 : 4;

    const size_t work_dim = 3;
    const size_t global_work_size[work_dim] = { nwgs * lwx, lwy, lwz };
    const size_t local_work_size[work_dim] = { lwx, lwy, lwz };

    for(auto _ : state) {
        clEnqueueNDRangeKernel(
            queue(),
            kernel(),
            work_dim,
            NULL,
            global_work_size,
            local_work_size,
            0,
            NULL,
            NULL );
        clFinish(queue());
    }
}
BENCHMARK_REGISTER_F(Kernel, clEnqueueNDRangeKernel_overhead)->ArgsProduct({{0, 1}, {1, 32*1024*1024}});

BENCHMARK_DEFINE_F(Kernel, clSetKernelArgSVMPointer_null)(benchmark::State& state)
{
    for(auto _ : state) {
        clSetKernelArgSVMPointer(
            kernel(),
            0,
            nullptr);
    }
}
BENCHMARK_REGISTER_F(Kernel, clSetKernelArgSVMPointer_null);

BENCHMARK_DEFINE_F(Kernel, clSetKernelArgMemPointerINTEL_null)(benchmark::State& state)
{
    for(auto _ : state) {
        clSetKernelArgMemPointerINTEL(
            kernel(),
            0,
            nullptr);
    }
}
BENCHMARK_REGISTER_F(Kernel, clSetKernelArgMemPointerINTEL_null);

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
    for(auto _ : state) {
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
    for(auto _ : state) {
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

struct USMMemFill : public benchmark::Fixture
{
    const size_t sz = 1 * 1024 * 1024;

    cl::CommandQueue queue;

    void* dptr = NULL;
    void* hptr = NULL;
    void* sptr = NULL;

    virtual void SetUp(benchmark::State& state) override {
        queue = env.ioq;

        dptr = clDeviceMemAllocINTEL(env.context(), env.device(), NULL, sz, 0, NULL);
        hptr = clHostMemAllocINTEL(env.context(), NULL, sz, 0, NULL);
        sptr = clSharedMemAllocINTEL(env.context(), env.device(), NULL, sz, 0, NULL);
    }
    virtual void TearDown(benchmark::State& state) override {
        queue = NULL;

        clMemBlockingFreeINTEL(env.context(), dptr);
        clMemBlockingFreeINTEL(env.context(), hptr);
        clMemBlockingFreeINTEL(env.context(), sptr);
    }
};

BENCHMARK_DEFINE_F(USMMemFill, clEnqueueMemsetINTEL)(benchmark::State& state)
{
    void* dst = nullptr;

    switch (state.range(0)) {
    case 0: dst = dptr; break;
    case 1: dst = hptr; break;
    case 2: dst = sptr; break;
    default: state.SkipWithError("unknown mem type");
    }
    if (dst == nullptr) {
        state.SkipWithError("unsupported mem type");
    }
    for(auto _ : state) {
        clEnqueueMemsetINTEL(
            queue(),
            dst,
            0,
            sz,
            0,
            NULL,
            NULL);
        queue.finish();
    }
}
BENCHMARK_REGISTER_F(USMMemFill, clEnqueueMemsetINTEL)->ArgsProduct({{0, 1, 2}});

BENCHMARK_DEFINE_F(USMMemFill, clEnqueueMemFillINTEL)(benchmark::State& state)
{
    void* dst = nullptr;
    switch (state.range(1)) {
    case 0: dst = dptr; break;
    case 1: dst = hptr; break;
    case 2: dst = sptr; break;
    default: state.SkipWithError("unknown mem type"); break;
    }
    if (dst == nullptr) {
        state.SkipWithError("unsupported mem type");
    }

    const cl_ulong pattern = 0;
    const size_t patternSize = state.range(0);
    for(auto _ : state) {
        clEnqueueMemFillINTEL(
            queue(),
            dptr,
            &pattern,
            patternSize,
            sz,
            0,
            NULL,
            NULL);
        queue.finish();
    }
}
BENCHMARK_REGISTER_F(USMMemFill, clEnqueueMemFillINTEL)->ArgsProduct({{1, 4, 8, 16}, {0, 1, 2}});

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
