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

#include <chrono>
#include <ctime>

const char* filename = "sinjulia.bmp";

size_t iterations = 16;
// Part 5: Fix the default global work size.
// Only some devices support non-uniform work groups, and non-uniform work
// groups require OpenCL C 2.0 or newer.  Since we are compiling our kernels for
// the default OpenCL C 1.2 we require uniform work groups. Unfortunately, this
// chosen global work size is prime, so the only uniform work group size is one
// work-item, which is not very efficient!  Choosing a different global work
// size will likely perform much better.
// Note: 4K resolution is 3840 x 2160.
size_t gwx = 3847;  // this is a weird number...
size_t gwy = 2161;  // this is a weird number also...
size_t lwx = 0;
size_t lwy = 0;

float cr = 1.0f;
float ci = 0.3f;

cl::CommandQueue commandQueue;
cl::Kernel kernel;
cl::Buffer deviceMemDst;

// Part 2: Fix the program.
// There are a few typos in the program below that need to be fixed.
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

    // BGRA
    float4 color = (float4)(
        1.0f,
        result,
        result * result,
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
    for( int i = 0; i < iterations; i++ )
    {
        // Part 3: Fix the ND-Range.
        // The C++ bindings are great and save a lot of typing, but sometimes
        // they can save too much!  The arguments below are incorrect and
        // generate a bad call to clEnqueueNDRangeKernel.  What is needed to fix
        // it?
        commandQueue.enqueueNDRangeKernel(
            kernel,
            cl::NDRange{gwx, gwy},
            lws);
    }

    // Enqueue all processing is complete before stopping the timer.
    commandQueue.finish();

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<float> elapsed_seconds = end - start;
    printf("Finished in %f seconds\n", elapsed_seconds.count());
}

static void checkResults()
{
    // Part 4: Fix the map flags.
    // We want to read the results of our kernel and save them to a bitmap. The
    // map flags below are more typically used to initialize a buffer. What map
    // flag should be used instead?
    auto buf = reinterpret_cast<const uint32_t*>(
        commandQueue.enqueueMapBuffer(
            deviceMemDst,
            CL_TRUE,
            CL_MAP_WRITE_INVALIDATE_REGION,
            0,
            gwx * gwy * sizeof(cl_uchar4) ) );

    BMP::save_image(buf, gwx, gwy, filename);
    printf("Wrote image file %s\n", filename);

    commandQueue.enqueueUnmapMemObject(
        deviceMemDst,
        (void*)buf );
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
            else if( !strcmp( argv[i], "-p" ) )
            {
                if( ++i < argc )
                {
                    platformIndex = strtol(argv[i], NULL, 10);
                }
            }
            else if( !strcmp( argv[i], "-i" ) )
            {
                if( ++i < argc )
                {
                    iterations = strtol(argv[i], NULL, 10);
                }
            }
            else if( !strcmp( argv[i], "-gwx" ) )
            {
                if( ++i < argc )
                {
                    gwx = strtol(argv[i], NULL, 10);
                }
            }
            else if( !strcmp( argv[i], "-gwy" ) )
            {
                if( ++i < argc )
                {
                    gwy = strtol(argv[i], NULL, 10);
                }
            }
            else if( !strcmp( argv[i], "-lwx" ) )
            {
                if( ++i < argc )
                {
                    lwx = strtol(argv[i], NULL, 10);
                }
            }
            else if( !strcmp( argv[i], "-lwy" ) )
            {
                if( ++i < argc )
                {
                    lwy = strtol(argv[i], NULL, 10);
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
            "Usage: sinjulia [options]\n"
            "Options:\n"
            "      -d: Device Index (default = 0)\n"
            "      -p: Platform Index (default = 0)\n"
            "      -i: Number of Iterations (default = 16)\n"
            "      -gwx: Global Work Size X AKA Image Width\n"
            "      -gwy: Global Work Size Y AKA Image Height\n"
            "      -lwx: Local Work Size X (default = 0 = NULL Local Work Size)\n"
            "      -lwy: Local Work Size Y (default = 0 = Null Local Work size)\n"
            );

        return -1;
    }

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    printf("Running on platform: %s\n",
        platforms[platformIndex].getInfo<CL_PLATFORM_NAME>().c_str() );

    std::vector<cl::Device> devices;
    // Part 1: Query the devices in this platform.
    // We will usually pass in a specific device type, e.g. CL_DEVICE_TYPE_CPU.
    // We can also pass in CL_DEVICE_TYPE_ALL, which will get all devices.
    // Passing CL_DEVICE_TYPE isn't a valid device type...
    platforms[platformIndex].getDevices(CL_DEVICE_TYPE_ALL, &devices);

    printf("Running on device: %s\n",
        devices[deviceIndex].getInfo<CL_DEVICE_NAME>().c_str() );

    cl::Context context{devices[deviceIndex]};
    commandQueue = cl::CommandQueue{context, devices[deviceIndex]};

    cl::Program program{ context, kernelString };

    // Part 6: Experiment with build options.
    // By default, OpenCL kernels use precise math functions.  For images like
    // the ones we are generating though, fast math is usually sufficient.  If
    // you pass build options to use fast math do you see a performance
    // improvement?
    program.build();

    kernel = cl::Kernel{ program, "SinJulia" };

    deviceMemDst = cl::Buffer{
        context,
        CL_MEM_ALLOC_HOST_PTR,
        gwx * gwy * sizeof(cl_uchar4) };

    init();
    go();
    checkResults();

    return 0;
}
