/*
// Copyright (c) 2019-2026 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <popl/popl.hpp>

#include <CL/opencl.hpp>

#include "util.hpp"

static const char kernelString[] = R"CLC(
kernel void test(global uint* dst)
{
    uint id = get_global_id(0);
    dst[id] = id;
}
)CLC";

#if !defined(cl_khr_suggested_local_work_size)
#pragma message("suggestedlocalworksize: cl_khr_suggested_local_work_size.  Please update your OpenCL headers.")
#endif

int main(int argc, char** argv)
{
    int platformIndex = 0;
    int deviceIndex = 0;

    size_t gws = 1024*1024;

    {
        popl::OptionParser op("Supported Options");
        op.add<popl::Value<int>>("p", "platform", "Platform Index", platformIndex, &platformIndex);
        op.add<popl::Value<int>>("d", "device", "Device Index", deviceIndex, &deviceIndex);
        op.add<popl::Value<size_t>>("g", "gws", "Global Work Size", gws, &gws);

        bool printUsage = false;
        try {
            op.parse(argc, argv);
        } catch (std::exception& e) {
            fprintf(stderr, "Error: %s\n\n", e.what());
            printUsage = true;
        }

        if (printUsage || !op.unknown_options().empty() || !op.non_option_args().empty()) {
            fprintf(stderr,
                "Usage: suggestedlocalworksize [options]\n"
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

    if (checkDeviceForExtension(devices[deviceIndex], CL_KHR_SUGGESTED_LOCAL_WORK_SIZE_EXTENSION_NAME)) {
        printf("Device supports %s.\n", CL_KHR_SUGGESTED_LOCAL_WORK_SIZE_EXTENSION_NAME);
    } else {
        printf("Device does not support %s, exiting.\n", CL_KHR_SUGGESTED_LOCAL_WORK_SIZE_EXTENSION_NAME);
        return -1;
    }

    clGetKernelSuggestedLocalWorkSizeKHR_fn clGetKernelSuggestedLocalWorkSizeKHR =
        (clGetKernelSuggestedLocalWorkSizeKHR_fn)clGetExtensionFunctionAddressForPlatform(
            platforms[platformIndex](),
            "clGetKernelSuggestedLocalWorkSizeKHR" );
    if (clGetKernelSuggestedLocalWorkSizeKHR == nullptr) {
        fprintf(stderr, "Failed to get function pointer for clGetKernelSuggestedLocalWorkSizeKHR, exiting.\n");
        return -1;
    }

    cl::Context context{devices[deviceIndex]};
    cl::CommandQueue commandQueue{context, devices[deviceIndex]};

    cl::Program program{ context, kernelString };
    program.build();
    cl::Kernel kernel = cl::Kernel{ program, "test" };

    cl::Buffer buf = cl::Buffer{
        context,
        CL_MEM_ALLOC_HOST_PTR,
        gws * sizeof( cl_uint ) };

    kernel.setArg(0, buf);

    size_t suggestedLocalWorkSize = 0;
    cl_int errorCode = clGetKernelSuggestedLocalWorkSizeKHR(
        commandQueue(),
        kernel(),
        1,
        nullptr,
        &gws,
        &suggestedLocalWorkSize );

    if (errorCode != CL_SUCCESS) {
        fprintf(stderr, "clGetKernelSuggestedLocalWorkSizeKHR failed with error code %d, exiting.\n", errorCode);
        return -1;
    }

    printf("Suggested local work size for global work size %zu is: %zu\n", gws, suggestedLocalWorkSize);
    return 0;
}
