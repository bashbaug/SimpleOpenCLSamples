/*
// Copyright (c) 2025 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <popl/popl.hpp>

#include <CL/opencl.hpp>

#include <memory>
#include <numeric>

const size_t sz = 1024;

static const char kernelString[] = R"CLC(
struct s { const global uint* ptr; };
kernel void test_IndirectAccessRead(
    const global struct s* src,
    global uint* dst,
    global uint* dummy)
{
    dst[get_global_id(0)] = src->ptr[get_global_id(0)];
}
)CLC";

int main(
    int argc,
    char** argv )
{
    int platformIndex = 0;
    int deviceIndex = 0;

    bool enableIndirectAccess = false;
    bool setAsKernelArg = false;
    bool useHostUSM = false;

    {
        popl::OptionParser op("Supported Options");
        op.add<popl::Value<int>>("p", "platform", "Platform Index", platformIndex, &platformIndex);
        op.add<popl::Value<int>>("d", "device", "Device Index", deviceIndex, &deviceIndex);
        op.add<popl::Switch>("i", "indirect", "Enable Indirect Access", &enableIndirectAccess);
        op.add<popl::Switch>("a", "argument", "Set as Kernel Argument", &setAsKernelArg);
        op.add<popl::Switch>("h", "host", "Use Host USM", &useHostUSM);
        bool printUsage = false;
        try {
            op.parse(argc, argv);
        } catch (std::exception& e) {
            fprintf(stderr, "Error: %s\n\n", e.what());
            printUsage = true;
        }

        if (printUsage || !op.unknown_options().empty() || !op.non_option_args().empty()) {
            fprintf(stderr,
                "Usage: dmemindirect [options]\n"
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
    cl::Kernel kernel = cl::Kernel{ program, "test_IndirectAccessRead" };

    cl_uint* d_src = useHostUSM ?
        (cl_uint*)clHostMemAllocINTEL(
            context(),
            nullptr,
            sz * sizeof(cl_uint),
            0,
            nullptr) :
        (cl_uint*)clDeviceMemAllocINTEL(
            context(),
            devices[deviceIndex](),
            nullptr,
            sz * sizeof(cl_uint),
            0,
            nullptr);

    // destination buffer
    cl::Buffer dst = cl::Buffer{
        context,
        CL_MEM_READ_WRITE,
        sz * sizeof(cl_uint)};

    // indirect source buffer
    cl::Buffer src = cl::Buffer{
        context,
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        sizeof(d_src),
        &d_src};

    if (d_src) {
        if (enableIndirectAccess) {
            cl_bool enable = CL_TRUE;
            clSetKernelExecInfo(
                kernel(),
                CL_KERNEL_EXEC_INFO_INDIRECT_DEVICE_ACCESS_INTEL,
                sizeof(cl_bool),
                &enable);
        }

        clSetKernelExecInfo(
            kernel(),
            CL_KERNEL_EXEC_INFO_USM_PTRS_INTEL,
            sizeof(d_src),
            &d_src);

        kernel.setArg(0, src);
        kernel.setArg(1, dst);
        if (setAsKernelArg) {
            clSetKernelArgMemPointerINTEL(
                kernel(),
                2,
                d_src);
        } else {
            kernel.setArg(2, nullptr);
        }

        cl_int errorCode = commandQueue.enqueueNDRangeKernel(
            kernel,
            cl::NullRange,
            cl::NDRange(sz),
            cl::NullRange);
        printf("clEnqueueNDRangeKernel returned: %d\n", errorCode);

        commandQueue.finish();
    }
    else
    {
        printf("Allocation failed - does this device support Unified Shared Memory?\n");
    }

    printf("Cleaning up...\n");

    clMemFreeINTEL(context(), d_src);

    return 0;
}
