/*
// Copyright (c) 2020-2025 Ben Ashbaugh
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

    cl::Context context{devices[deviceIndex]};
    cl::CommandQueue commandQueue{context, devices[deviceIndex]};

    cl::Program program{ context, kernelString };
    program.build();
    cl::Kernel kernel = cl::Kernel{ program, "CopyBuffer" };

    cl_uint* h_buf = new cl_uint[gwx];

    cl_uint* d_src = (cl_uint*)clDeviceMemAllocINTEL(
        context(),
        devices[deviceIndex](),
        nullptr,
        gwx * sizeof(cl_uint),
        0,
        nullptr );
    cl_uint* d_dst = (cl_uint*)clDeviceMemAllocINTEL(
        context(),
        devices[deviceIndex](),
        nullptr,
        gwx * sizeof(cl_uint),
        0,
        nullptr );

    if( h_buf && d_src && d_dst )
    {
        // initialization
        {
            for( size_t i = 0; i < gwx; i++ )
            {
                h_buf[i] = (cl_uint)(i);
            }

            clEnqueueMemcpyINTEL(
                commandQueue(),
                CL_TRUE,
                d_src,
                h_buf,
                gwx * sizeof(cl_uint),
                0,
                nullptr,
                nullptr );

            memset( h_buf, 0, gwx * sizeof(cl_uint) );
        }

        // execution
        clSetKernelArgMemPointerINTEL(
            kernel(),
            0,
            d_dst );
        clSetKernelArgMemPointerINTEL(
            kernel(),
            1,
            d_src );
        commandQueue.enqueueNDRangeKernel(
            kernel,
            cl::NullRange,
            cl::NDRange{gwx} );

        // verification
        {
            clEnqueueMemcpyINTEL(
                commandQueue(),
                CL_TRUE,
                h_buf,
                d_dst,
                gwx * sizeof(cl_uint),
                0,
                nullptr,
                nullptr );

            unsigned int    mismatches = 0;

            for( size_t i = 0; i < gwx; i++ )
            {
                if( h_buf[i] != i )
                {
                    if( mismatches < 16 )
                    {
                        fprintf(stderr, "MisMatch!  dst[%d] == %08X, want %08X\n",
                            (unsigned int)i,
                            h_buf[i],
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
        printf("Allocation failed - does this device support Unified Shared Memory?\n");
    }

    printf("Cleaning up...\n");

    delete [] h_buf;

    clMemFreeINTEL(
        context(),
        d_src );
    clMemFreeINTEL(
        context(),
        d_dst );

    return 0;
}
