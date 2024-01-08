/*
// Copyright (c) 2024 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <popl/popl.hpp>

#include <CL/opencl.hpp>

const size_t    gwx = 1024*1024;

static const char kernelString[] = R"CLC(
  __kernel void copy(__global int* in, __global int* out, __global int* offset) {
      size_t id = get_global_id(0);
      int ind = offset[0] + id;
      out[ind] = in[ind];
  }
)CLC";

int main(int argc, char** argv )
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
                "Usage: copybufferkernel [options]\n"
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
    cl::CommandQueue commandQueue{context, devices[deviceIndex], /*CL_QUEUE_PROFILING_ENABLE*/};

    cl::Program program{ context, kernelString };
    program.build();
    cl::Kernel kernel{ program, "copy" };

    constexpr size_t numElements = 16384;
    cl::Buffer in_mem{context, CL_MEM_READ_ONLY, numElements * 2 * sizeof(cl_int)};
    cl::Buffer out_mem{context, CL_MEM_WRITE_ONLY, numElements * 2 * sizeof(cl_int)};
    cl_int offset = 0;
    cl::Buffer off_mem{context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(offset), &offset};

    kernel.setArg(0, in_mem);
    kernel.setArg(1, out_mem);
    kernel.setArg(2, off_mem);

    //cl_int pattern = 0xA;
    //commandQueue.enqueueFillBuffer(out_mem, pattern, 0, numElements);
    //commandQueue.enqueueFillBuffer(off_mem, static_cast<cl_int>(0), 0, 1);

    cl::UserEvent blockEvent{context};
    std::vector<cl::Event> waitEvents;
    waitEvents.push_back(blockEvent);

    cl::Event startEvent0;
    cl::Event endEvent0;
    commandQueue.enqueueBarrierWithWaitList(&waitEvents, &startEvent0);
    commandQueue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange{numElements});
    //commandQueue.enqueueBarrierWithWaitList(nullptr, &endEvent0);

    //std::vector<cl_int> results0(numElements);
    //commandQueue.enqueueReadBuffer(out_mem, CL_FALSE, 0, numElements * sizeof(cl_int), results0.data());

    //commandQueue.enqueueFillBuffer(out_mem, pattern, numElements, numElements);
    //commandQueue.enqueueFillBuffer(off_mem, static_cast<cl_int>(numElements), 0, 1);

    //cl::Event startEvent1;
    //cl::Event endEvent1;
    //commandQueue.enqueueBarrierWithWaitList(&waitEvents, &startEvent1);
    //commandQueue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange{numElements});
    //commandQueue.enqueueBarrierWithWaitList(nullptr, &endEvent1);

    //std::vector<cl_int> results1(numElements);
    //commandQueue.enqueueReadBuffer(out_mem, CL_FALSE, 0, numElements * sizeof(cl_int), results1.data());

    blockEvent.setStatus(CL_COMPLETE);

    printf("Calling clFinish()...\n");
    commandQueue.finish();

    printf("Done!\n");

    return 0;
}
