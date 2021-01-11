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

#include <fstream>
#include <string>

size_t gwx = 32771;

cl::CommandQueue commandQueue;
cl::Kernel kernel;
cl::Buffer deviceMemDst;

typedef CL_API_ENTRY
cl_int (CL_API_CALL *clGetKernelSuggestedLocalWorkSizeINTEL_fn)(
    cl_command_queue commandQueue,
    cl_kernel kernel,
    cl_uint workDim,
    const size_t *globalWorkOffset,
    const size_t *globalWorkSize,
    size_t *suggestedLocalWorkSize);

clGetKernelSuggestedLocalWorkSizeINTEL_fn clGetKernelSuggestedLocalWorkSizeINTEL;

static std::string readStringFromFile(
    const std::string& filename )
{
    std::ifstream is(filename, std::ios::binary);
    if (!is.good()) {
        printf("Couldn't open file '%s'!\n", filename.c_str());
        return "";
    }

    size_t filesize = 0;
    is.seekg(0, std::ios::end);
    filesize = (size_t)is.tellg();
    is.seekg(0, std::ios::beg);

    std::string source{
        std::istreambuf_iterator<char>(is),
        std::istreambuf_iterator<char>() };

    return source;
}

static void init( void )
{
    // No initialization is needed for this sample.
}

static void go()
{
    if (clGetKernelSuggestedLocalWorkSizeINTEL) {
        size_t gwo = 0;
        size_t slws = 0;
        cl_int errorCode = clGetKernelSuggestedLocalWorkSizeINTEL(
            commandQueue(),
            kernel(),
            1,
            &gwo,
            &gwx,
            &slws);
        if (errorCode != CL_SUCCESS) {
            printf("Initial call: clGetKernelSuggestedLocalWorkSizeINTEL returned %d\n",
                errorCode);
        } else {
            printf("Initial call: Suggested local work size is %d.\n",
                (int)slws);
        }
    }

    kernel.setArg(0, deviceMemDst);

    if (clGetKernelSuggestedLocalWorkSizeINTEL) {
        size_t gwo = 0;
        size_t slws = 0;
        cl_int errorCode = clGetKernelSuggestedLocalWorkSizeINTEL(
            commandQueue(),
            kernel(),
            1,
            &gwo,
            &gwx,
            &slws);
        if (errorCode != CL_SUCCESS) {
            printf("After setting kernel args: clGetKernelSuggestedLocalWorkSizeINTEL returned %d\n",
                errorCode);
        } else {
            printf("After setting kernel args: Suggested local work size is %d.\n",
                (int)slws);
        }
    }

    commandQueue.enqueueNDRangeKernel(
        kernel,
        cl::NullRange,
        cl::NDRange{gwx});

    if (clGetKernelSuggestedLocalWorkSizeINTEL) {
        size_t gwo = 0;
        size_t slws = 0;
        cl_int errorCode = clGetKernelSuggestedLocalWorkSizeINTEL(
            commandQueue(),
            kernel(),
            1,
            &gwo,
            &gwx,
            &slws);
        if (errorCode != CL_SUCCESS) {
            printf("After enqueue: clGetKernelSuggestedLocalWorkSizeINTEL returned %d\n",
                errorCode);
        } else {
            printf("After enqueue: Suggested local work size is %d.\n",
                (int)slws);
        }
    }
}

static void checkResults()
{
    // No results to check for this sample, but do verify that execution
    // has completed.
    commandQueue.finish();
}

int main(
    int argc,
    char** argv )
{
    bool printUsage = false;
    int platformIndex = 0;
    int deviceIndex = 0;

    const char* fileName = "sample_kernel_slws.cl";
    const char* kernelName = "Test";
    const char* buildOptions = NULL;

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
                if( ++i < argc )
                {
                    deviceIndex = strtol(argv[i], NULL, 10);
                }
            }
            else if( !strcmp( argv[i], "-p" ) )
            {
                if( ++i < argc )
                {
                    platformIndex = strtol(argv[i], NULL, 10);
                }
            }
            else if( !strcmp( argv[i], "-file" ) )
            {
                if( ++i < argc )
                {
                    fileName = argv[i];
                }
            }
            else if( !strcmp( argv[i], "-name" ) )
            {
                if( ++i < argc )
                {
                    kernelName = argv[i];
                }
            }
            else if( !strcmp( argv[i], "-options" ) )
            {
                if( ++i < argc )
                {
                    buildOptions = argv[i];
                }
            }
            else if( !strcmp( argv[i], "-gwx" ) )
            {
                if( ++i < argc )
                {
                    gwx = strtol(argv[i], NULL, 10);
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
            "Usage: suggestedlws [options]\n"
            "Options:\n"
            "      -d: Device Index (default = 0)\n"
            "      -p: Platform Index (default = 0)\n"
            "      -file: Kernel File Name (default = sample_kernel_slws.cl)\n"
            "      -name: Kernel Name (default = Test)\n"
            "      -options: Program Build Options (default = NULL)\n"
            "      -gwx: Global Work Size (default = 512)\n"
            );

        return -1;
    }

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    printf("Running on platform: %s\n",
        platforms[platformIndex].getInfo<CL_PLATFORM_NAME>().c_str() );

    clGetKernelSuggestedLocalWorkSizeINTEL = (clGetKernelSuggestedLocalWorkSizeINTEL_fn)
        clGetExtensionFunctionAddressForPlatform(
            platforms[platformIndex](),
            "clGetKernelSuggestedLocalWorkSizeINTEL");
    if (clGetKernelSuggestedLocalWorkSizeINTEL == NULL) {
        printf("Couldn't get function pointer for clGetKernelSuggestedLocalWorkSizeINTEL!\n");
    }

    std::vector<cl::Device> devices;
    platforms[platformIndex].getDevices(CL_DEVICE_TYPE_ALL, &devices);

    printf("Running on device: %s\n",
        devices[deviceIndex].getInfo<CL_DEVICE_NAME>().c_str() );

    cl::Context context{devices[deviceIndex]};
    commandQueue = cl::CommandQueue{context, devices[deviceIndex]};

    printf("Reading program source from file: %s\n", fileName );
    std::string kernelString = readStringFromFile(fileName);

    printf("Building program with build options: %s\n",
        buildOptions ? buildOptions : "(none)" );
    cl::Program program{ context, kernelString };
    program.build(buildOptions);
    for( auto& device : program.getInfo<CL_PROGRAM_DEVICES>() )
    {
        printf("Program build log for device %s:\n",
            device.getInfo<CL_DEVICE_NAME>().c_str() );
        printf("%s\n",
            program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device).c_str() );
    }
    printf("Creating kernel: %s\n", kernelName );
    kernel = cl::Kernel{ program, kernelName };

    deviceMemDst = cl::Buffer{
        context,
        CL_MEM_ALLOC_HOST_PTR,
        gwx * sizeof( cl_uint ) };

    init();
    go();
    checkResults();

    printf("Done.\n");

    return 0;
}