/*
// Copyright (c) 2026 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <memory>
#include <popl/popl.hpp>

#include <CL/opencl.hpp>

#include "util.hpp"

const size_t    gwx = 1024*1024;

static const char kernelString[] = R"CLC(
kernel void SillyCopy( global uint* dst0, global uint* src0, global uint* dst1, global uint* src1 )
{
    *dst0 = *src0;
    *dst1 = *src1;
}
)CLC";

struct SVMDeleter
{
    SVMDeleter(cl::Context& _c) : context(_c) {}
    void operator()(void* ptr) {
        clSVMFree(context(), ptr);
    }
    cl::Context context;
};

int main(
    int argc,
    char** argv )
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
                "Usage: dmemhelloworld [options]\n"
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

    cl_device_svm_capabilities svmcaps = devices[deviceIndex].getInfo<CL_DEVICE_SVM_CAPABILITIES>();
    if( svmcaps & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER ) {
        printf("Device supports CL_DEVICE_SVM_COARSE_GRAIN_BUFFER.\n");
    } else {
        printf("Device does not support CL_DEVICE_SVM_COARSE_GRAIN_BUFFER, exiting.\n");
        return -1;
    }

    cl::Context context{devices[deviceIndex]};
    cl::CommandQueue commandQueue{context, devices[deviceIndex]};

    cl::Program program{ context, kernelString };
    program.build();
    cl::Kernel kernel = cl::Kernel{ program, "SillyCopy" };

    {
        constexpr size_t count = 5;

        std::unique_ptr<cl_int[], SVMDeleter> mem(
            (cl_int*)clSVMAlloc(
                context(),
                CL_MEM_READ_WRITE,
                count * sizeof(cl_int),
                0),
            SVMDeleter(context));

        if (mem) {
            // initialization
            {
                commandQueue.enqueueMapSVM(
                    mem,
                    CL_TRUE,
                    CL_MAP_WRITE_INVALIDATE_REGION,
                    count * sizeof(cl_int) );
                for( size_t i = 0; i < count; i++ )
                {
                    auto val = static_cast<cl_int>(i);
                    mem[i] = val;
                }

                commandQueue.enqueueUnmapSVM( mem );
            }

            // execution
            kernel.setArg( 0, mem.get() + 1 );
            kernel.setArg( 1, mem.get() + 2 );
            kernel.setArg( 2, mem.get() + 3 );
            kernel.setArg( 3, mem.get() + 4 );
            commandQueue.enqueueNDRangeKernel(
                kernel,
                cl::NullRange,
                cl::NDRange{1} );

            // verification
            {
                commandQueue.enqueueMapSVM(
                    mem,
                    CL_TRUE,
                    CL_MAP_READ,
                    count * sizeof(cl_uint) );

                printf("Values are: [%u, %u, %u, %u, %u]\n",
                    mem[0], mem[1], mem[2], mem[3], mem[4]);

                commandQueue.enqueueUnmapSVM( mem );
            }
        } else {
            printf("Allocation failed - does this device support SVM?\n");
        }

        printf("Cleaning up...\n");
    }

    return 0;
}
