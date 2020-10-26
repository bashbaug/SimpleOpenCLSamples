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
#include "bmp.hpp"

const char* filename = "mandelbrot.bmp";

const cl_uint width = 768;
const cl_uint height = 512;

const int maxIterations = 256;

cl::CommandQueue commandQueue;
cl::Kernel kernel;
cl::Buffer deviceMemDst;

static const char kernelString[] = R"CLC(
static inline int mandel(float c_re, float c_im, int count) {
    float z_re = c_re, z_im = c_im;
    int i;
    for (i = 0; i < count; ++i) {
        if (z_re * z_re + z_im * z_im > 4.)
            break;

        float new_re = z_re*z_re - z_im*z_im;
        float new_im = 2.f * z_re * z_im;

        z_re = c_re + new_re;
        z_im = c_im + new_im;
    }

    return i;
}
kernel void Mandelbrot(
    float x0, float y0,
    float x1, float y1,
    int width, int height,
    int maxIterations,
    global int* output)
{
    float dx = (x1 - x0) / width;
    float dy = (y1 - y0) / height;

    float x = x0 + get_global_id(0) * dx;
    float y = y0 + get_global_id(1) * dy;

    int index = get_global_id(1) * width + get_global_id(0);
    output[index] = mandel(x, y, maxIterations);
}
)CLC";

static void init( void )
{
    // No initialization is needed for this sample.
}

static void go()
{
    kernel.setArg(0, -2.0f);    // x0
    kernel.setArg(1, -1.0f);    // y0
    kernel.setArg(2, 1.0f);     // x1
    kernel.setArg(3, 1.0f);     // y1
    kernel.setArg(4, width);
    kernel.setArg(5, height);
    kernel.setArg(6, maxIterations);
    kernel.setArg(7, deviceMemDst);

    commandQueue.enqueueNDRangeKernel(
        kernel,
        cl::NullRange,
        cl::NDRange{width, height} );
}

static void checkResults()
{
    const cl_int*  buf = (const cl_int*)commandQueue.enqueueMapBuffer(
        deviceMemDst,
        CL_TRUE,
        CL_MAP_READ,
        0,
        width * height * sizeof(cl_int) );

    std::vector<uint8_t> colors;
    colors.resize(width * height);
    for (int i = 0; i < width * height; ++i) {
        // Map the iteration count to colors by just alternating between
        // two greys.
        colors[i] = (buf[i] & 0x1) ? 240 : 20;
    }
    BMP::save_image(colors.data(), width, height, filename);
    printf("Wrote image file %s\n", filename);

    commandQueue.enqueueUnmapMemObject(
        deviceMemDst,
        (void*)buf ); // TODO: Why isn't this a const void* in the API?
}

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
            "Usage: mandelbrot  [options]\n"
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
    commandQueue = cl::CommandQueue{context, devices[deviceIndex]};

    cl::Program program{ context, kernelString };
    program.build();
#if 0
    for( auto& device : program.getInfo<CL_PROGRAM_DEVICES>() )
    {
        printf("Program build log for device %s:\n",
            device.getInfo<CL_DEVICE_NAME>().c_str() );
        printf("%s\n",
            program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device).c_str() );
    }
#endif
    kernel = cl::Kernel{ program, "Mandelbrot" };

    deviceMemDst = cl::Buffer{
        context,
        CL_MEM_ALLOC_HOST_PTR,
        width * height * sizeof( cl_int ) };

    init();
    go();
    checkResults();

    return 0;
}