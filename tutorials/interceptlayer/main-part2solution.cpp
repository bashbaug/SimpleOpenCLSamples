/*
// Copyright (c) 2019-2023 Ben Ashbaugh
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

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>

#include <CL/opencl.hpp>

#include <chrono>

const char* filename = "sinjulia.bmp";

size_t iterations = 16;
// Part 4: Fix the default global work size.
// Since we are compiling our kernels for the default OpenCL C 1.2 we require
// uniform work groups. Unfortunately, this chosen global work size is prime, so
// the only uniform local work-group size is one work-item, which is not very
// efficient! Can we choose a different global work size that will perform
// better?
// Note: 4K resolution is 3840 x 2160.
size_t gwx = 3847;
size_t gwy = 2161;
size_t lwx = 0; // NULL local work size.
size_t lwy = 0;

float cr = 1.0f;
float ci = 0.3f;

cl::CommandQueue commandQueue;
cl::Kernel kernel;
cl::Buffer deviceMemDst;

// Part 2: Fix the OpenCL C program source.
// Solution: Fix the typo.
static const char kernelString[] = R"CLC(
kernel void SinJulia(global uchar4* dst, float cr, float ci)
{
    const int xMax = get_global_size(0);
    const int yMax = get_global_size(1);

    const int x = get_global_id(0);
    const int y = get_global_id(1);

    const float zMin = -M_PI_F / 2;
    const float zMax =  M_PI_F / 2;

    float zr = (float)x / xMax * (zMax - zMin) + zMin;
    float zi = (float)y / yMax * (zMax - zMin) + zMin;

    const int cIterations = 64;
    const float cThreshold = 50.0f;

    float result = 0.0f;
    for( int i = 0; i < cIterations; i++ ) {
        if(fabs(zi) > cThreshold) {
            break;
        }

        // zn = sin(z)
        float zrn = sin(zr) * cosh(zi);
        float zin = cos(zr) * sinh(zi);

        // z = c * zn = c * sin(z)
        zr = cr * zrn - ci * zin;
        zi = cr * zin + ci * zrn;

        result += 1.0f / cIterations;
    }

    result = max(result, 0.0f);
    result = min(result, 1.0f);

    // RGBA
    float4 color = (float4)(
        result * result,
        result,
        1.0f,
        1.0f );

    dst[ y * xMax + x ] = convert_uchar4(color * 255.0f);
}
)CLC";

static void init( void )
{
    // No initialization is needed for this sample.
}

static void go()
{
    kernel.setArg(0, deviceMemDst);
    kernel.setArg(1, cr);
    kernel.setArg(2, ci);

    cl::NDRange lws;    // NullRange by default.

    printf("Executing the kernel %d times\n", (int)iterations);
    printf("Global Work Size = ( %d, %d )\n", (int)gwx, (int)gwy);
    if( lwx > 0 && lwy > 0 )
    {
        printf("Local Work Size = ( %d, %d )\n", (int)lwx, (int)lwy);
        lws = cl::NDRange{lwx, lwy};
    }
    else
    {
        printf("Local work size = NULL\n");
    }

    // Ensure the queue is empty and no processing is happening
    // on the device before starting the timer.
    commandQueue.finish();

    auto start = std::chrono::system_clock::now();
    for( size_t i = 0; i < iterations; i++ )
    {
        commandQueue.enqueueNDRangeKernel(
            kernel,
            cl::NullRange,
            cl::NDRange{gwx, gwy},
            lws);
        commandQueue.flush();
    }

    // Enqueue all processing is complete before stopping the timer.
    commandQueue.finish();

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<float> elapsed_seconds = end - start;
    printf("Finished in %f seconds\n", elapsed_seconds.count());
}

static void checkResults()
{
    // Part 3: Fix the map flags.
    // We want to read the results of our kernel and save them to a bitmap. The
    // map flags below are more typically used to initialize a buffer. What map
    // flag should we use instead?
    auto buf = reinterpret_cast<const uint32_t*>(
        commandQueue.enqueueMapBuffer(
            deviceMemDst,
            CL_TRUE,
            CL_MAP_WRITE_INVALIDATE_REGION,
            0,
            gwx * gwy * sizeof(cl_uchar4) ) );

    stbi_write_bmp(filename, gwx, gwy, 4, buf);
    printf("Wrote image file %s\n", filename);

    commandQueue.enqueueUnmapMemObject(
        deviceMemDst,
        (void*)buf );
    commandQueue.finish();
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
        op.add<popl::Value<size_t>>("i", "iterations", "Iterations", iterations, &iterations);
        op.add<popl::Value<size_t>>("", "gwx", "Global Work Size X", gwx, &gwx);
        op.add<popl::Value<size_t>>("", "gwy", "Global Work Size Y", gwy, &gwy);
        op.add<popl::Value<size_t>>("", "lwx", "Local Work Size X", lwx, &lwx);
        op.add<popl::Value<size_t>>("", "lwy", "Local Work Size Y", lwy, &lwy);

        bool printUsage = false;
        try {
            op.parse(argc, argv);
        } catch (std::exception& e) {
            fprintf(stderr, "Error: %s\n\n", e.what());
            printUsage = true;
        }

        if (printUsage || !op.unknown_options().empty() || !op.non_option_args().empty()) {
            fprintf(stderr,
                "Usage: sinjulia [options]\n"
                "%s", op.help().c_str());
            return -1;
        }
    }

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    printf("Running on platform: %s\n",
        platforms[platformIndex].getInfo<CL_PLATFORM_NAME>().c_str() );

    std::vector<cl::Device> devices;
    // Part 1: Query the devices in this platform.
    // Solution: Query for CL_DEVICE_TYPE_ALL, not CL_DEVICE_TYPE.
    platforms[platformIndex].getDevices(CL_DEVICE_TYPE_ALL, &devices);

    printf("Running on device: %s\n",
        devices[deviceIndex].getInfo<CL_DEVICE_NAME>().c_str() );

    cl::Context context{devices[deviceIndex]};
    commandQueue = cl::CommandQueue{context, devices[deviceIndex]};

    cl::Program program{ context, kernelString };

    // Part 5: Experiment with build options.
    // By default, OpenCL kernels use precise math functions. For images like
    // the ones we are generating, though, fast math is usually sufficient. If
    // we pass build options to use fast math does the image still look OK? Does
    // using fast math improve performance?
    program.build();

    kernel = cl::Kernel{ program, "SinJulia" };

    deviceMemDst = cl::Buffer{
        context,
        CL_MEM_READ_WRITE,
        gwx * gwy * sizeof(cl_uchar4) };

    init();
    go();
    checkResults();

    return 0;
}
