/*
// Copyright (c) 2026 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <popl/popl.hpp>

#include <CL/opencl.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <random>
#include <thread>
#include <vector>

#include "util.hpp"

static const char kernelString[] = R"CLC(
kernel void Increment( global uint* dst )
{
    dst[0] = dst[0] + 1;
}
)CLC";

static std::atomic<bool> stop{false};

static void ThreadFunc(
    int threadId,
    bool useEvents,
    const cl::Context& context,
    const cl::Device& device,
    const cl::Program& program )
{
    int checks = 0;
    bool quiet = false;

    // Each thread creates its own command-queue, buffer, and kernel.
    cl::CommandQueue commandQueue{context, device};
    cl::Buffer buffer{context, CL_MEM_READ_WRITE, sizeof(cl_uint)};
    cl::Kernel kernel{program, "Increment"};
    kernel.setArg(0, buffer);

    // Seed the random number generator differently for each thread.
    std::mt19937 rng{static_cast<std::mt19937::result_type>(threadId)};
    std::uniform_int_distribution<int> dist{0, 99};

    cl_uint expected = 0;
    commandQueue.enqueueFillBuffer(
        buffer,
        expected,
        0,
        sizeof(cl_uint) );

    while (!stop.load(std::memory_order_relaxed)) {
        int r = dist(rng);
        if (r < 70) {
            // 70% probability: enqueue the kernel.
            cl::Event event;
            if (useEvents) {
                commandQueue.enqueueNDRangeKernel(
                    kernel,
                    cl::NullRange,
                    cl::NDRange{1},
                    cl::NullRange,
                    nullptr,
                    &event );
            } else {
                commandQueue.enqueueNDRangeKernel(
                    kernel,
                    cl::NullRange,
                    cl::NDRange{1},
                    cl::NullRange );
            }
            ++expected;
        } else if (r < 80) {
            // 10% probability: flush the queue.
            commandQueue.flush();
        } else if (r < 90) {
            // 10% probability: finish the queue.
            commandQueue.finish();
        } else {
            // 10% probability: read and check the result.
            cl_uint result = 0;
            commandQueue.enqueueReadBuffer(
                buffer,
                CL_TRUE,
                0,
                sizeof(result),
                &result );
            ++checks;
            if (result != expected && quiet == false) {
                fprintf(stderr, "Mismatch on thread %d on check %d: Expected %u, got %u!\n",
                    threadId, checks, expected, result);
                quiet = true;
            }
        }
    }

    // Make sure all outstanding work is complete before the thread exits.
    commandQueue.finish();
}

int main(int argc, char** argv)
{
    int platformIndex = 0;
    int deviceIndex = 0;

    int numThreads = 8;
    int seconds = 1;
    bool useEvents = false;

    {
        popl::OptionParser op("Supported Options");
        op.add<popl::Value<int>>("p", "platform", "Platform Index", platformIndex, &platformIndex);
        op.add<popl::Value<int>>("d", "device", "Device Index", deviceIndex, &deviceIndex);
        op.add<popl::Value<int>>("t", "threads", "Number of Threads", numThreads, &numThreads);
        op.add<popl::Value<int>>("s", "seconds", "Number of Seconds to Run", seconds, &seconds);
        op.add<popl::Switch>("e", "events", "Use Events", &useEvents);

        bool printUsage = false;
        try {
            op.parse(argc, argv);
        } catch (std::exception& e) {
            fprintf(stderr, "Error: %s\n\n", e.what());
            printUsage = true;
        }

        if (printUsage || !op.unknown_options().empty() || !op.non_option_args().empty()) {
            fprintf(stderr,
                "Usage: multithreading [options]\n"
                "%s", op.help().c_str());
            return -1;
        }
    }

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    if (!checkPlatformIndex(platforms, platformIndex)) {
        return -1;
    }

    printf("Running on platform: %s\n",
        platforms[platformIndex].getInfo<CL_PLATFORM_NAME>().c_str() );

    std::vector<cl::Device> devices;
    platforms[platformIndex].getDevices(CL_DEVICE_TYPE_ALL, &devices);

    printf("Running on device: %s\n",
        devices[deviceIndex].getInfo<CL_DEVICE_NAME>().c_str() );

    printf("Running with %d thread(s) for %d second(s).\n",
        numThreads, seconds );

    // Setup: create the context and build the kernel on the main thread.
    cl::Context context{devices[deviceIndex]};

    cl::Program program{context, kernelString};
    program.build();

    // Spawn the worker threads.
    std::vector<std::thread> threads;
    threads.reserve(numThreads);
    for (int i = 0; i < numThreads; i++) {
        threads.emplace_back(
            ThreadFunc,
            i,
            useEvents,
            std::cref(context),
            std::cref(devices[deviceIndex]),
            std::cref(program) );
    }

    // Let the threads run for the requested amount of time, then signal them
    // to stop and wait for them to exit.
    std::this_thread::sleep_for(std::chrono::seconds(seconds));

    printf("All done, signaling threads to complete.\n");
    stop.store(true, std::memory_order_relaxed);

    for (auto& thread : threads) {
        thread.join();
    }

    printf("All threads complete.\n");

    return 0;
}
