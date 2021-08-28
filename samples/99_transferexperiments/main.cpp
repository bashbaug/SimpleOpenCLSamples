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
#include <numeric>

static constexpr size_t MB = 1024 * 1024;

int main(
    int argc,
    char** argv )
{
    int platformIndex = 0;
    int deviceIndex = 0;

    size_t iterations = 16;
    size_t size = 256;
    bool hostptr = false;
    bool finish = false;

    {
        popl::OptionParser op("Supported Options");
        op.add<popl::Value<int>>("p", "platform", "Platform Index", platformIndex, &platformIndex);
        op.add<popl::Value<int>>("d", "device", "Device Index", deviceIndex, &deviceIndex);
        op.add<popl::Value<size_t>>("i", "iterations", "Iterations", iterations, &iterations);
        op.add<popl::Value<size_t>>("", "size", "Buffer Size in MB", size, &size);
        op.add<popl::Switch>("h", "hostptr", "create buffers with CL_MEM_ALLOC_HOST_PTR", &hostptr);
        op.add<popl::Switch>("f", "finish", "call clFinish each iteration", &finish);
        bool printUsage = false;
        try {
            op.parse(argc, argv);
        } catch (std::exception& e) {
            fprintf(stderr, "Error: %s\n\n", e.what());
            printUsage = true;
        }

        if (printUsage || !op.unknown_options().empty() || !op.non_option_args().empty()) {
            fprintf(stderr,
                "Usage: transfertester [options]\n"
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

    std::vector<cl_uchar> hostMem(size * MB);
    std::iota(hostMem.begin(), hostMem.end(), 0);

    cl::Buffer deviceMemSrc = cl::Buffer{
        context,
        CL_MEM_COPY_HOST_PTR | (hostptr ? CL_MEM_ALLOC_HOST_PTR : 0U),
        size * MB,
        hostMem.data() };

    cl::Buffer deviceMemDst = cl::Buffer{
        context,
        hostptr ? CL_MEM_ALLOC_HOST_PTR : 0U,
        size * MB };

    cl::CommandQueue commandQueue = cl::CommandQueue{context, devices[deviceIndex]};

    // clEnqueueWriteBuffer
    {
        auto start = std::chrono::system_clock::now();
        for (int i = 0; i < iterations; i++) {
            commandQueue.enqueueWriteBuffer(
                deviceMemDst,
                CL_FALSE,
                0,
                size * MB,
                hostMem.data() );
            if (finish) {
                commandQueue.finish();
            }
        }
        commandQueue.finish();
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<float> elapsed_seconds = end - start;
        printf("clEnqueueWriteBuffer: Transferred a %zu byte buffer %zu times in %f seconds (%.2f GB/s)\n", 
            size * MB,
            iterations,
            elapsed_seconds.count(),
            size * MB * iterations / elapsed_seconds.count() / MB / 1024);
    }

    // clEnqueueReadBuffer
    {
        auto start = std::chrono::system_clock::now();
        for (int i = 0; i < iterations; i++) {
            commandQueue.enqueueReadBuffer(
                deviceMemDst,
                CL_FALSE,
                0,
                size * MB,
                hostMem.data() );
            if (finish) {
                commandQueue.finish();
            }
        }
        commandQueue.finish();
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<float> elapsed_seconds = end - start;
        printf("clEnqueueReadBuffer: Transferred a %zu byte buffer %zu times in %f seconds (%.2f GB/s)\n", 
            size * MB,
            iterations,
            elapsed_seconds.count(),
            size * MB * iterations / elapsed_seconds.count() / MB / 1024);
    }

    // clEnqueueCopyBuffer
    {
        auto start = std::chrono::system_clock::now();
        for (int i = 0; i < iterations; i++) {
            commandQueue.enqueueCopyBuffer(
                deviceMemSrc,
                deviceMemDst,
                CL_FALSE,
                0,
                size * MB );
            if (finish) {
                commandQueue.finish();
            }
        }
        commandQueue.finish();
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<float> elapsed_seconds = end - start;
        printf("clEnqueueCopyBuffer: Transferred a %zu byte buffer %zu times in %f seconds (%.2f GB/s)\n", 
            size * MB,
            iterations,
            elapsed_seconds.count(),
            size * MB * iterations / elapsed_seconds.count() / MB / 1024);
    }

    // verify results by printing the first few values
    if (size > 0) {
        auto ptr = (const cl_uint*)commandQueue.enqueueMapBuffer(
            deviceMemDst,
            CL_TRUE,
            CL_MAP_READ,
            0,
            size * MB );

        printf("First few values: [0] = %08X, [1] = %08X, [2] = %08X\n", ptr[0], ptr[1], ptr[2]);

        commandQueue.enqueueUnmapMemObject(
            deviceMemDst,
            (void*)ptr );
    }

    commandQueue.finish();

    return 0;
}
