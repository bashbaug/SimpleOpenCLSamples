/*
// Copyright (c) 2023 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <popl/popl.hpp>

#include <CL/opencl.hpp>

#define checkCLErrors(val) __checkCLErrors((val), #val, __FILE__, __LINE__)

static inline void __checkCLErrors(cl_int errCode, const char* func, const char* file, int line)
{
    if (errCode) {
        fprintf(stderr, "Error: %s returned %i\n", func, errCode);
    }
}

int main(int argc, char** argv )
{
    constexpr size_t size = 1024;
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
                "Usage: mapunmap [options]\n"
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

    cl::Buffer buf = cl::Buffer{
        context,
        CL_MEM_ALLOC_HOST_PTR,
        size };

    printf("Initially: map count is %u\n", buf.getInfo<CL_MEM_MAP_COUNT>());

    void* ptr0 = commandQueue.enqueueMapBuffer(buf, CL_TRUE, CL_MAP_READ, 0, size);
    printf("After blocking map: map count is %u\n", buf.getInfo<CL_MEM_MAP_COUNT>());

    void* ptr1 = commandQueue.enqueueMapBuffer(buf, CL_FALSE, CL_MAP_READ, 0, size);
    printf("After non-blocking map: map count is %u\n", buf.getInfo<CL_MEM_MAP_COUNT>());

    void* ptr2 = commandQueue.enqueueMapBuffer(buf, CL_FALSE, CL_MAP_READ, 0, size);
    printf("After another non-blocking map: map count is %u\n", buf.getInfo<CL_MEM_MAP_COUNT>());

    checkCLErrors(commandQueue.finish());
    printf("After finish: map count is %u\n", buf.getInfo<CL_MEM_MAP_COUNT>());

    checkCLErrors(commandQueue.enqueueUnmapMemObject(buf, ptr0));
    printf("After unmap: map count is %u\n", buf.getInfo<CL_MEM_MAP_COUNT>());

    checkCLErrors(commandQueue.finish());
    printf("After finish: map count is %u\n", buf.getInfo<CL_MEM_MAP_COUNT>());

    cl::UserEvent event0{context};
    std::vector<cl::Event> deps0{event0};
    checkCLErrors(commandQueue.enqueueUnmapMemObject(buf, ptr1, &deps0));
    printf("After unmap with event dependency: map count is %u\n", buf.getInfo<CL_MEM_MAP_COUNT>());

    cl::UserEvent event1{context};
    std::vector<cl::Event> deps1{event1};
    checkCLErrors(commandQueue.enqueueUnmapMemObject(buf, ptr2, &deps1));
    printf("After unmap with event dependency: map count is %u\n", buf.getInfo<CL_MEM_MAP_COUNT>());

    cl::UserEvent event2{context};
    std::vector<cl::Event> deps2{event2};
    checkCLErrors(commandQueue.enqueueUnmapMemObject(buf, ptr2, &deps2));
    printf("After duplicate unmap with event dependency: map count is %u\n", buf.getInfo<CL_MEM_MAP_COUNT>());

    checkCLErrors(event0.setStatus(CL_COMPLETE));
    checkCLErrors(event1.setStatus(CL_COMPLETE));
    checkCLErrors(event2.setStatus(CL_COMPLETE));
    printf("After setting event status to complete: map count is %u\n", buf.getInfo<CL_MEM_MAP_COUNT>());

    checkCLErrors(commandQueue.finish());
    printf("After finish: map count is %u\n", buf.getInfo<CL_MEM_MAP_COUNT>());

    return 0;
}
