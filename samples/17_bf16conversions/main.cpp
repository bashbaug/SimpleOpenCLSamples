/*
// Copyright (c) 2024 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <popl/popl.hpp>

#include <CL/opencl.hpp>

#include <cmath>
#include <cstdint>

#include "util.hpp"

#ifndef CL_INTEL_BFLOAT16_CONVERSIONS_NAME
#define CL_INTEL_BFLOAT16_CONVERSIONS_NAME \
    "cl_intel_bfloat16_conversions"
#endif

static const char kernelString[] = R"CLC(
kernel void bf16_convert( global float* dst )
{
    uint id = get_global_id(0);
    dst[id] = intel_convert_as_bfloat16_float(id);;
}
)CLC";

static float bf16_to_float(const uint16_t a) {
    union {
        uint32_t intStorage;
        float floatValue;
    };
    intStorage = a << 16;
    return floatValue;
}

int main(
    int argc,
    char** argv )
{
    constexpr size_t gws = 65536;

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
                "Usage: bf16conversions [options]\n"
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

    bool has_cl_intel_bfloat16_conversions =
        checkDeviceForExtension(devices[deviceIndex], CL_INTEL_BFLOAT16_CONVERSIONS_NAME);
    if (has_cl_intel_bfloat16_conversions) {
        printf("Device supports " CL_INTEL_BFLOAT16_CONVERSIONS_NAME ".\n");
    } else {
        printf("Device does not support " CL_INTEL_BFLOAT16_CONVERSIONS_NAME ", exiting.\n");
        return -1;
    }

    cl::Context context{devices[deviceIndex]};
    cl::CommandQueue commandQueue{context, devices[deviceIndex]};

    cl::Program program{ context, kernelString };
    program.build();
    cl::Kernel kernel = cl::Kernel{ program, "bf16_convert" };

    cl::Buffer deviceMemDst = cl::Buffer{
        context,
        CL_MEM_ALLOC_HOST_PTR,
        gws * sizeof(cl_float) };

    kernel.setArg(0, deviceMemDst);
    commandQueue.enqueueNDRangeKernel(
        kernel,
        cl::NullRange,
        cl::NDRange{gws} );

    // verification
    {
        auto    pDst = (const float*)commandQueue.enqueueMapBuffer(
            deviceMemDst,
            CL_TRUE,
            CL_MAP_READ,
            0,
            gws * sizeof(cl_float) );

        unsigned int    mismatches = 0;

        for( size_t i = 0; i < gws; i++ )
        {
            auto result = pDst[i];
            auto check = bf16_to_float(static_cast<uint16_t>(i));
            if( (std::isnan(result) && !std::isnan(check)) ||
                (!std::isnan(result) && std::isnan(check)) ||
                (!std::isnan(result) && !std::isnan(check) && result != check) )
            {
                if( mismatches < 16 )
                {
                    fprintf(stderr, "MisMatch at index %zu: got %f (%08X), want %f (%08X)\n",
                        i,
                        result,
                        *(unsigned int*)&result,
                        check,
                        *(unsigned int*)&check );
                }
                mismatches++;
            }
        }

        if( mismatches )
        {
            fprintf(stderr, "Error: Found %d mismatches / %d values!!!\n",
                mismatches,
                (unsigned int)gws );
        }
        else
        {
            printf("Success.\n");
        }

        commandQueue.enqueueUnmapMemObject(
            deviceMemDst,
            (void*)pDst );
    }

    return 0;
}
