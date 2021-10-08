/*
// Copyright (c) 2020-2021 Ben Ashbaugh
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
                "Usage: sysmemhelloworld [options]\n"
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

    // Check that this device supports system USM.
    // For this sample we only require ACCESS capabilities.
    cl_device_unified_shared_memory_capabilities_intel usmcaps = 0;
    clGetDeviceInfo(
        devices[deviceIndex](),
        CL_DEVICE_SHARED_SYSTEM_MEM_CAPABILITIES_INTEL,
        sizeof(usmcaps),
        &usmcaps,
        nullptr );
    if ((usmcaps & CL_UNIFIED_SHARED_MEMORY_ACCESS_INTEL) == 0) {
        printf("Device does not support system USM, exiting.\n");
        return -1;
    }

    cl::Context context{devices[deviceIndex]};
    cl::CommandQueue commandQueue = cl::CommandQueue{context, devices[deviceIndex]};

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
