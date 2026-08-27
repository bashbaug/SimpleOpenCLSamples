/*
// Copyright (c) 2019-2026 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <popl/popl.hpp>

#include <CL/opencl.hpp>

#include "util.hpp"

int main(int argc, char** argv)
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
                "Usage: copybuffer [options]\n"
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

    constexpr size_t sz = 1024;
    cl_int err = CL_SUCCESS;
    cl::Context context{devices[deviceIndex]};

    {
        printf("Creating a buffer with CL_MEM_READ_WRITE...\n");
        cl::Buffer buf = cl::Buffer{
            context,
            CL_MEM_READ_WRITE,
            sz,
            nullptr,
            &err};
        printf("  ... returned %d\n", err);
    }

    {
        printf("Creating a buffer with CL_MEM_KERNEL_READ_AND_WRITE...\n");
        cl::Buffer buf = cl::Buffer{
            context,
            CL_MEM_KERNEL_READ_AND_WRITE,
            sz,
            nullptr,
            &err};
        printf("  ... returned %d\n", err);
    }

    {
        printf("Creating an image with CL_MEM_READ_WRITE...\n");
        cl::Image2D img = cl::Image2D{
            context,
            CL_MEM_READ_WRITE,
            cl::ImageFormat{CL_RGBA, CL_UNSIGNED_INT8},
            16, 16, 0,
            nullptr,
            &err};
        printf("  ... returned %d\n", err);
    }

    {
        printf("Creating an image with CL_MEM_KERNEL_READ_AND_WRITE...\n");
        cl::Image2D img = cl::Image2D{
            context,
            CL_MEM_KERNEL_READ_AND_WRITE,
            cl::ImageFormat{CL_RGBA, CL_UNSIGNED_INT8},
            16, 16, 0,
            nullptr,
            &err};
        printf("  ... returned %d\n", err);
    }

    return 0;
}
