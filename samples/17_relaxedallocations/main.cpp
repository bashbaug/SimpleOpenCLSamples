/*
// Copyright (c) 2025 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <cinttypes>
#include <popl/popl.hpp>

#include <CL/opencl.hpp>

#include "util.hpp"

#if !defined(CL_INTEL_RELAX_ALLOCATION_LIMITS_EXTENSION_NAME)
#define CL_INTEL_RELAX_ALLOCATION_LIMITS_EXTENSION_NAME \
    "cl_intel_relax_allocation_limits"
#endif
#if !defined(CL_MEM_ALLOW_UNRESTRICTED_SIZE_INTEL)
#define CL_MEM_ALLOW_UNRESTRICTED_SIZE_INTEL                ( 1 << 23 )
#endif

static const char kernelString[] = R"CLC(
kernel void touch(global uint* buf)
{
    size_t id = get_global_id(0);
    for (size_t i = 0; i < 1024; i++) {
        buf[id * 1024 + i] += 2;
    }
}
)CLC";

int main(
    int argc,
    char** argv )
{
    int platformIndex = 0;
    int deviceIndex = 0;

    size_t sz = 2;
    bool relaxAllocationLimits = false;
    bool useSVM = false;
    bool useUSM = false;

    {
        popl::OptionParser op("Supported Options");
        op.add<popl::Value<int>>("p", "platform", "Platform Index", platformIndex, &platformIndex);
        op.add<popl::Value<int>>("d", "device", "Device Index", deviceIndex, &deviceIndex);
        op.add<popl::Value<size_t>>("s", "size", "Allocation Size (GB)", sz, &sz);
        op.add<popl::Switch>("r", "relax", "Relax Allocation Limits", &relaxAllocationLimits);
        op.add<popl::Switch>("", "svm", "Use Coarse-grain SVM Allocations", &useSVM);
        op.add<popl::Switch>("", "usm", "Use Device USM Allocations", &useUSM);

        bool printUsage = false;
        try {
            op.parse(argc, argv);
        } catch (std::exception& e) {
            fprintf(stderr, "Error: %s\n\n", e.what());
            printUsage = true;
        }

        if (printUsage || !op.unknown_options().empty() || !op.non_option_args().empty()) {
            fprintf(stderr,
                "Usage: relaxedallocations [options]\n"
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

    bool has_cl_intel_relax_allocation_limits =
        checkDeviceForExtension(devices[deviceIndex], CL_INTEL_RELAX_ALLOCATION_LIMITS_EXTENSION_NAME);
    if (has_cl_intel_relax_allocation_limits) {
        printf("Device supports " CL_INTEL_RELAX_ALLOCATION_LIMITS_EXTENSION_NAME ".\n");
    } else {
        printf("Device does not support " CL_INTEL_RELAX_ALLOCATION_LIMITS_EXTENSION_NAME ".\n");
    }

    cl::Context context{devices[deviceIndex]};
    cl::CommandQueue commandQueue{context, devices[deviceIndex]};

    printf("For this device:\n");
    printf("\tCL_DEVICE_GLOBAL_MEM_SIZE is %f GB\n",
        devices[deviceIndex].getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() / (1024.0f * 1024.0f * 1024.0f));
    printf("\tCL_DEVICE_MAX_MEM_ALLOC_SIZE is %f GB\n",
        devices[deviceIndex].getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>() / (1024.0f * 1024.0f * 1024.0f));

    size_t allocSize = (size_t)sz * 1024 * 1024 * 1024;
    size_t gwx = allocSize / 1024 / sizeof(cl_uint);

    printf("Testing allocation size %zu GB (%zu 32-bit values).\n", sz, allocSize / sizeof(cl_uint));
    if (relaxAllocationLimits) {
        printf("Testing with relaxed allocation limits.\n");
    } else if (allocSize > devices[deviceIndex].getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>()) {
        printf("Allocation may fail, allocation size exceeds CL_DEVICE_MAX_MEM_ALLOC_SIZE!\n");
    }

    std::vector<cl_uint> h_buf(allocSize / sizeof(cl_uint));
    for (size_t i = 0; i < h_buf.size(); i++) {
        h_buf[i] = static_cast<cl_uint>(i);
    }

    // initialization

    cl::Program program{ context, kernelString };
    program.build(relaxAllocationLimits ? "-cl-intel-greater-than-4GB-buffer-required" : "");
    cl::Kernel kernel = cl::Kernel{ program, "touch" };

    cl_uint* dptr = nullptr;
    cl::Buffer mem;

    if (useSVM) {
        const cl_mem_flags flags =
            relaxAllocationLimits ? CL_MEM_ALLOW_UNRESTRICTED_SIZE_INTEL : 0;
        dptr = (cl_uint *)clSVMAlloc(
            context(),
            flags,
            allocSize, 0);
        if (dptr == nullptr) {
            printf("SVM allocation failed!\n");
        } else {
            commandQueue.enqueueMemcpySVM(dptr, h_buf.data(), CL_TRUE, allocSize);
            kernel.setArg(0, dptr);
        }
    } else if (useUSM) {
        const cl_mem_properties_intel props[] = {
            CL_MEM_FLAGS,
            CL_MEM_ALLOW_UNRESTRICTED_SIZE_INTEL,
            0
        };
        dptr = (cl_uint*)clDeviceMemAllocINTEL(
            context(),
            devices[deviceIndex](),
            relaxAllocationLimits ? props : nullptr,
            allocSize,
            0,
            nullptr);
        if (dptr == nullptr) {
            printf("USM allocation failed!\n");
        } else {
            clEnqueueMemcpyINTEL(
                commandQueue(),
                CL_TRUE,
                dptr,
                h_buf.data(),
                allocSize,
                0,
                nullptr,
                nullptr);
            clSetKernelArgMemPointerINTEL(
                kernel(),
                0,
                dptr);
        }
    } else {
        const cl_mem_flags flags =
            relaxAllocationLimits ? CL_MEM_ALLOW_UNRESTRICTED_SIZE_INTEL : 0;
        mem = cl::Buffer{
            context,
            flags,
            allocSize};
        if (mem() == nullptr) {
            printf("Buffer allocation failed!\n");
        } else {
            commandQueue.enqueueWriteBuffer(
                mem,
                CL_TRUE,
                0,
                allocSize,
                h_buf.data());
            kernel.setArg(0, mem);
        }
    }

    // execution

    commandQueue.enqueueNDRangeKernel(
        kernel,
        cl::NullRange,
        cl::NDRange{gwx});

    // validation

    if (useSVM) {
        commandQueue.enqueueMemcpySVM(
            h_buf.data(),
            dptr,
            CL_TRUE,
            allocSize);
        clSVMFree(context(), dptr);
        dptr = nullptr;
    } else if (useUSM) {
        clEnqueueMemcpyINTEL(
            commandQueue(),
            CL_TRUE,
            h_buf.data(),
            dptr,
            allocSize,
            0,
            nullptr,
            nullptr);
        clMemFreeINTEL(context(), dptr);
        dptr = nullptr;
    } else {
        commandQueue.enqueueReadBuffer(
            mem,
            CL_TRUE,
            0,
            allocSize,
            h_buf.data());
    }

    cl_uint mismatches = 0;
    for (size_t i = 0; i < h_buf.size(); i++) {
        cl_uint want = static_cast<cl_uint>(i + 2);
        if (h_buf[i] != want) {
            if (mismatches < 16) {
                printf("Error at index %zu: expected %u, got %u!\n", i, want, h_buf[i]);
            }
            mismatches++;
        }
    }
    if (mismatches) {
        printf("Error: Found %u mismatches / %zu values!!!\n", mismatches, h_buf.size());
    } else {
        printf("Success.\n");
    }

    return 0;
}
