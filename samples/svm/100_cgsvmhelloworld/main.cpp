/*
// Copyright (c) 2024-2025 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <popl/popl.hpp>

#include <CL/opencl.hpp>

const size_t    gwx = 1024*1024;

static const char kernelString[] = R"CLC(
kernel void CopyBuffer( global uint* dst, global uint* src )
{
    uint id = get_global_id(0);
    dst[id] = src[id];
}
)CLC";

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
    cl::Kernel kernel = cl::Kernel{ program, "CopyBuffer" };

    cl_uint* src = (cl_uint*)clSVMAlloc(
        context(),
        CL_MEM_READ_WRITE,
        gwx * sizeof(cl_uint),
        0 );
    cl_uint* dst = (cl_uint*)clSVMAlloc(
        context(),
        CL_MEM_READ_WRITE,
        gwx * sizeof(cl_uint),
        0 );

    if( src && dst )
    {
        // initialization
        {
            commandQueue.enqueueMapSVM(
                src,
                CL_TRUE,
                CL_MAP_WRITE_INVALIDATE_REGION,
                gwx * sizeof(cl_uint) );
            for( size_t i = 0; i < gwx; i++ )
            {
                src[i] = (cl_uint)(i);
            }

            commandQueue.enqueueUnmapSVM( src );
        }

        // execution
        kernel.setArg( 0, dst );
        kernel.setArg( 1, src );
        commandQueue.enqueueNDRangeKernel(
            kernel,
            cl::NullRange,
            cl::NDRange{gwx} );

        // verification
        {
            commandQueue.enqueueMapSVM(
                dst,
                CL_TRUE,
                CL_MAP_READ,
                gwx * sizeof(cl_uint) );

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

            commandQueue.enqueueUnmapSVM( dst );

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
        printf("Allocation failed - does this device support SVM?\n");
    }

    printf("Cleaning up...\n");

    clSVMFree( context(), src );
    clSVMFree( context(), dst );

    return 0;
}
