/*
// Copyright (c) 2021 Ben Ashbaugh
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

cl::CommandQueue commandQueue;
cl::Kernel kernel;
cl::Image1D imageSrc;
cl::Buffer memDst;

static const char kernelString[] = R"CLC(
sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE  | CLK_ADDRESS_NONE | CLK_FILTER_LINEAR;

kernel void Test( global float4* dst, read_only image1d_t src )
{
    dst[0] = read_imagef(src, sampler, 3.5f);
}
)CLC";

static void init( void )
{
    // empty
}

static void go()
{
    kernel.setArg(0, memDst);
    kernel.setArg(1, imageSrc);

    commandQueue.enqueueNDRangeKernel(
        kernel,
        cl::NullRange,
        cl::NDRange{1} );
}

static void checkResults()
{
    const cl_float* pDst = (const cl_float*)commandQueue.enqueueMapBuffer(
        memDst,
        CL_TRUE,
        CL_MAP_READ,
        0,
        4 * sizeof(cl_float) );

    printf("Got: [%f, %f, %f, %f]\n", pDst[0], pDst[1], pDst[2], pDst[3]);

    commandQueue.enqueueUnmapMemObject(
        memDst,
        (void*)pDst ); // TODO: Why isn't this a const void* in the API?
}

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
                "Usage: copybufferkernel [options]\n"
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
    commandQueue = cl::CommandQueue{context, devices[deviceIndex]};

    cl::Program program{ context, kernelString };
    program.build();
    kernel = cl::Kernel{ program, "Test" };

    float imageData[] =
    {
        0.2f, 0.4f, 0.6f, 0.8f,
        0.6f, 0.4f, 0.2f, 0.0f,
        0.2f, 0.4f, 0.6f, 0.8f,
        0.6f, 0.4f, 0.2f, 0.0f,
    };

    imageSrc = cl::Image1D{
        context,
        CL_MEM_COPY_HOST_PTR,
        cl::ImageFormat{CL_RGBA, CL_FLOAT},
        4,
        imageData };

    memDst = cl::Buffer{
        context,
        CL_MEM_ALLOC_HOST_PTR,
        4 * sizeof( cl_float ) };

    init();
    go();
    checkResults();

    return 0;
}