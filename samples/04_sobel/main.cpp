/*
// Copyright (c) 2023-2025 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <popl/popl.hpp>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>

#include <CL/opencl.hpp>

#include <chrono>

const char* filename = "sobel.bmp";

const float cr = -0.123f;
const float ci =  0.745f;

static const char kernelString[] = R"CLC(
kernel void Julia( global uchar4* dst, float cr, float ci )
{
    const float cMinX = -1.5f;
    const float cMaxX =  1.5f;
    const float cMinY = -1.5f;
    const float cMaxY =  1.5f;

    const int cWidth = get_global_size(0);
    const int cHeight = get_global_size(1);
    const int cIterations = 16;

    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);

    float a = x * ( cMaxX - cMinX ) / cWidth + cMinX;
    float b = y * ( cMaxY - cMinY ) / cHeight + cMinY;

    float result = 0.0f;
    const float thresholdSquared = cIterations * cIterations / 64.0f;

    for( int i = 0; i < cIterations; i++ ) {
        float aa = a * a;
        float bb = b * b;

        float magnitudeSquared = aa + bb;
        if( magnitudeSquared >= thresholdSquared ) {
            break;
        }

        result += 1.0f / cIterations;
        b = 2 * a * b + ci;
        a = aa - bb + cr;
    }

    result = max( result, 0.0f );
    result = min( result, 1.0f );

    // RGBA
    float4 color = (float4)( result, sqrt(result), 1.0f, 1.0f );

    dst[ y * cWidth + x ] = convert_uchar4(color * 255.0f);
}

kernel void Sobel( global uchar4* src, global uchar4* dst )
{
    const int cWidth = get_global_size(0);
    const int cHeight = get_global_size(1);

    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);

    float4 texel_ul = x > 0 && y > 0 ?
        convert_float4(src[ (y - 1) * cWidth + (x - 1)]) : 0.0f;
    float4 texel_u  = y > 0 ?
        convert_float4(src[ (y - 1) * cWidth +  x     ]) : 0.0f;
    float4 texel_ur = x < cWidth - 1 && y > 0 ?
        convert_float4(src[ (y - 1) * cWidth + (x + 1)]) : 0.0f;

    float4 texel_l  = x > 0 ?
        convert_float4(src[  y      * cWidth + (x - 1)]) : 0.0f;
    float4 texel_r  = x < cWidth - 1 ?
        convert_float4(src[  y      * cWidth + (x + 1)]) : 0.0f;

    float4 texel_bl = x > 0 && y < cHeight - 1 ?
        convert_float4(src[ (y + 1) * cWidth + (x - 1)]) : 0.0f;
    float4 texel_b  = y < cHeight - 1 ?
        convert_float4(src[ (y + 1) * cWidth +  x     ]) : 0.0f;
    float4 texel_br = x < cWidth - 1 && y < cHeight - 1 ?
        convert_float4(src[ (y + 1) * cWidth + (x + 1)]) : 0.0f;

    float4 gx =
        texel_ul - texel_ur
        + 2.0f * texel_l - 2.0f * texel_r
        + texel_bl - texel_br;

    float4 gy =
        texel_ul + 2.0f * texel_u + texel_ur
        - texel_bl - 2.0f * texel_b - texel_br;

    float4 mag = sqrt(gx * gx + gy * gy);

    float grey = mag.x * 0.2126f + mag.y * 0.7152f + mag.z * 0.0722f;

    // RGBA
    float4 color = (float4)(grey, grey, grey, 255.0f);

    dst[ y * cWidth + x ] = convert_uchar4(color);
}
)CLC";

int main(
    int argc,
    char** argv )
{
    int platformIndex = 0;
    int deviceIndex = 0;

    size_t iterations = 16;
    size_t gwx = 512;
    size_t gwy = 512;
    size_t lwx = 0;
    size_t lwy = 0;

    {
        popl::OptionParser op("Supported Options");
        op.add<popl::Value<int>>("p", "platform", "Platform Index", platformIndex, &platformIndex);
        op.add<popl::Value<int>>("d", "device", "Device Index", deviceIndex, &deviceIndex);
        op.add<popl::Value<size_t>>("i", "iterations", "Iterations", iterations, &iterations);
        op.add<popl::Value<size_t>>("", "gwx", "Global Work Size X AKA Image Width", gwx, &gwx);
        op.add<popl::Value<size_t>>("", "gwy", "Global Work Size Y AKA Image Height", gwy, &gwy);
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
                "Usage: sobel [options]\n"
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
    cl::Kernel kernelJulia = cl::Kernel{ program, "Julia" };
    cl::Kernel kernelSobel = cl::Kernel{ program, "Sobel" };

    cl::Buffer deviceMemTmp = cl::Buffer{
        context,
        CL_MEM_ALLOC_HOST_PTR,
        gwx * gwy * sizeof( cl_uchar4 ) };

    cl::Buffer deviceMemDst = cl::Buffer{
        context,
        CL_MEM_ALLOC_HOST_PTR,
        gwx * gwy * sizeof( cl_uchar4 ) };

    // execution
    {
        kernelJulia.setArg(0, deviceMemTmp);
        kernelJulia.setArg(1, cr);
        kernelJulia.setArg(2, ci);

        kernelSobel.setArg(0, deviceMemTmp);
        kernelSobel.setArg(1, deviceMemDst);

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
                kernelJulia,
                cl::NullRange,
                cl::NDRange{gwx, gwy},
                lws);
            commandQueue.enqueueNDRangeKernel(
                kernelSobel,
                cl::NullRange,
                cl::NDRange{gwx, gwy},
                lws);
        }

        // Ensure all processing is complete before stopping the timer.
        commandQueue.finish();

        auto end = std::chrono::system_clock::now();
        std::chrono::duration<float> elapsed_seconds = end - start;
        printf("Finished in %f seconds\n", elapsed_seconds.count());
    }

    // save bitmap
    {
        auto buf = reinterpret_cast<const uint32_t*>(
            commandQueue.enqueueMapBuffer(
                deviceMemDst,
                CL_TRUE,
                CL_MAP_READ,
                0,
                gwx * gwy * sizeof(cl_uchar4) ) );

        stbi_write_bmp(filename, (int)gwx, (int)gwy, 4, buf);
        printf("Wrote image file %s\n", filename);

        commandQueue.enqueueUnmapMemObject(
            deviceMemDst,
            (void*)buf );
    }

    return 0;
}
