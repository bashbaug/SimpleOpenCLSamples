/*
// Copyright (c) 2020 Ben Ashbaugh
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
*/

#include <popl/popl.hpp>

#include <CL/opencl.hpp>
#include "libusm.h"

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
                "Usage: usmmigratemem [options]\n"
                "%s", op.help().c_str());
            return -1;
        }
    }

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    printf("Running on platform: %s\n",
        platforms[platformIndex].getInfo<CL_PLATFORM_NAME>().c_str() );
    libusm::initialize(platforms[platformIndex]());

    std::vector<cl::Device> devices;
    platforms[platformIndex].getDevices(CL_DEVICE_TYPE_ALL, &devices);

    printf("Running on device: %s\n",
        devices[deviceIndex].getInfo<CL_DEVICE_NAME>().c_str() );

    cl::Context context{devices[deviceIndex]};
    cl::CommandQueue commandQueue = cl::CommandQueue{context, devices[deviceIndex]};

    cl::Program program{ context, kernelString };
    program.build();
    cl::Kernel kernel = cl::Kernel{ program, "CopyBuffer" };

    cl_uint* s_src = (cl_uint*)clSharedMemAllocINTEL(
        context(),
        devices[deviceIndex](),
        nullptr,
        gwx * sizeof(cl_uint),
        0,
        nullptr );
    cl_uint* s_dst = (cl_uint*)clSharedMemAllocINTEL(
        context(),
        devices[deviceIndex](),
        nullptr,
        gwx * sizeof(cl_uint),
        0,
        nullptr );

    if( s_src && s_dst )
    {
        // initialization
        {
            for( size_t i = 0; i < gwx; i++ )
            {
                s_src[i] = (cl_uint)(i);
            }

            memset( s_dst, 0, gwx * sizeof(cl_uint) );
        }

        // execution
        clEnqueueMigrateMemINTEL(
            commandQueue(),
            s_src,
            gwx * sizeof(cl_uint),
            0,
            0,
            nullptr,
            nullptr );
        clEnqueueMigrateMemINTEL(
            commandQueue(),
            s_dst,
            gwx * sizeof(cl_uint),
            CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED,
            0,
            nullptr,
            nullptr );

        clSetKernelArgMemPointerINTEL(
            kernel(),
            0,
            s_dst );
        clSetKernelArgMemPointerINTEL(
            kernel(),
            1,
            s_src );
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
                if( s_dst[i] != i )
                {
                    if( mismatches < 16 )
                    {
                        fprintf(stderr, "MisMatch!  dst[%d] == %08X, want %08X\n",
                            (unsigned int)i,
                            s_dst[i],
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

    clMemFreeINTEL(
        context(),
        s_src );
    clMemFreeINTEL(
        context(),
        s_dst );

    return 0;
}