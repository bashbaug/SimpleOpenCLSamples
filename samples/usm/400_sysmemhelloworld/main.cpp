/*
// Copyright (c) 2020-2026 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <popl/popl.hpp>

#include <CL/opencl.hpp>

#include "util.hpp"

const size_t    gwx = 1024*1024;

static const char kernelString[] = R"CLC(
kernel void CopyBuffer( global uint* dst, global uint* src )
{
    uint id = get_global_id(0);
    dst[id] = src[id];
}
)CLC";

int main(int argc, char** argv)
{
    bool verbose = false;
    int platformIndex = 0;
    int deviceIndex = 0;

    {
        popl::OptionParser op("Supported Options");
        op.add<popl::Switch, popl::Attribute::advanced>("v", "verbose", "Verbose Output", &verbose);
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
                "Usage: sysmemhelloworld [options]\n"
                "%s", op.help().c_str());
            return -1;
        }
    }

    cl::Device device;
    if (!setupDevice(device, platformIndex, deviceIndex, verbose)) {
        return -1;
    }

    // Check that this device supports system USM.
    // For this sample we only require ACCESS capabilities.
    cl_device_unified_shared_memory_capabilities_intel usmcaps = 0;
    clGetDeviceInfo(
        device(),
        CL_DEVICE_SHARED_SYSTEM_MEM_CAPABILITIES_INTEL,
        sizeof(usmcaps),
        &usmcaps,
        nullptr );
    if ((usmcaps & CL_UNIFIED_SHARED_MEMORY_ACCESS_INTEL) == 0) {
        printf("Device does not support system USM, exiting.\n");
        return -1;
    }

    cl::Context context{device};
    cl::CommandQueue commandQueue{context, device};

    cl::Program program{ context, kernelString };
    program.build();
    cl::Kernel kernel = cl::Kernel{ program, "CopyBuffer" };

    // For this sample we will use "malloc" as our system allocator.
    // We could also use "new" or a C++ type like a std::vector.
    cl_uint* src = (cl_uint*)malloc(gwx * sizeof(cl_uint));
    cl_uint* dst = (cl_uint*)malloc(gwx * sizeof(cl_uint));

    if( src && dst )
    {
        // initialization
        {
            for( size_t i = 0; i < gwx; i++ )
            {
                src[i] = (cl_uint)(i);
            }

            memset( dst, 0, gwx * sizeof(cl_uint) );
        }

        // execution
        clSetKernelArgMemPointerINTEL(
            kernel(),
            0,
            dst );
        clSetKernelArgMemPointerINTEL(
            kernel(),
            1,
            src );
        commandQueue.enqueueNDRangeKernel(
            kernel,
            cl::NullRange,
            cl::NDRange{gwx} );

        // verification
        {
            commandQueue.finish();

            unsigned int    mismatches = 0;

            for( size_t i = 0; i < gwx; i++ )
            {
                if( dst[i] != i )
                {
                    if( mismatches < 16 )
                    {
                        fprintf(stderr, "MisMatch!  dst[%d] == %08X, want %08X\n",
                            (unsigned int)i,
                            dst[i],
                            (unsigned int)i );
                    }
                    mismatches++;
                }
            }

            if( mismatches )
            {
                fprintf(stderr, "Error: Found %d mismatches / %d values!!!\n",
                    mismatches,
                    (unsigned int)gwx );
            }
            else
            {
                printf("Success.\n");
            }
        }
    }
    else
    {
        printf("Allocation failed.\n");
    }

    printf("Cleaning up...\n");

    free(src);
    free(dst);

    return 0;
}
