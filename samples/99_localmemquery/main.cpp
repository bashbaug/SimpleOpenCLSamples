/*
// Copyright (c) 2019-2020 Ben Ashbaugh
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

#include <CL/opencl.hpp>

cl::Kernel kernel;

size_t  gwx = 1024*1024;

static const char kernelString[] = R"CLC(
kernel void LocalMemTest( global uint* dst, local uint* dynamic_allocation ) {
    int lid = get_local_id(0);
    int gid = get_global_id(0);

    local uint static_allocation[256];

    static_allocation[lid] = gid;
    dynamic_allocation[lid] = static_allocation[lid];
    dst[gid] = dynamic_allocation[lid];
}
)CLC";

int main(
    int argc,
    char** argv )
{
    bool printUsage = false;
    int platformIndex = 0;
    int deviceIndex = 0;

    if( argc < 1 )
    {
        printUsage = true;
    }
    else
    {
        for( size_t i = 1; i < argc; i++ )
        {
            if( !strcmp( argv[i], "-d" ) )
            {
                ++i;
                if( i < argc )
                {
                    deviceIndex = strtol(argv[i], NULL, 10);
                }
            }
            else if( !strcmp( argv[i], "-p" ) )
            {
                ++i;
                if( i < argc )
                {
                    platformIndex = strtol(argv[i], NULL, 10);
                }
            }
            else
            {
                printUsage = true;
            }
        }
    }
    if( printUsage )
    {
        fprintf(stderr,
            "Usage: localmemquery   [options]\n"
            "Options:\n"
            "      -d: Device Index (default = 0)\n"
            "      -p: Platform Index (default = 0)\n"
            );

        return -1;
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
    cl::Program program{ context, kernelString };
    program.build();
    for( auto& device : program.getInfo<CL_PROGRAM_DEVICES>() )
    {
        printf("Program build log for device %s:\n",
            device.getInfo<CL_DEVICE_NAME>().c_str() );
        printf("%s\n",
            program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device).c_str() );
    }
    kernel = cl::Kernel{ program, "LocalMemTest" };

    size_t lms;
    lms = kernel.getWorkGroupInfo<CL_KERNEL_LOCAL_MEM_SIZE>(devices[deviceIndex]);
    printf("Initially:\n\tLOCAL_MEM_SIZE = %zu\n", lms);

    kernel.setArg(1, 1024);
    lms = kernel.getWorkGroupInfo<CL_KERNEL_LOCAL_MEM_SIZE>(devices[deviceIndex]);
    printf("After setting arg to 1K:\n\tLOCAL_MEM_SIZE = %zu\n", lms);

    kernel.setArg(1, 2048);
    lms = kernel.getWorkGroupInfo<CL_KERNEL_LOCAL_MEM_SIZE>(devices[deviceIndex]);
    printf("After setting arg to 2K:\n\tLOCAL_MEM_SIZE = %zu\n", lms);

    return 0;
}