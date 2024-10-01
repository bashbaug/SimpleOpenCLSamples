/*
// Copyright (c) 2019-2024 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <popl/popl.hpp>

#include <CL/opencl.hpp>

#include "util.hpp"

#include <algorithm>

static const char kernelString[] = R"CLC(
#pragma OPENCL EXTENSION cl_khr_mipmap_image : enable

// For this sample, we can simply use an inline sampler, with
// default values for sampling mipmaps.  A more complicated
// sample would need to create a sampler with properties on
// the host and pass it into the kernel.

const sampler_t sampler =
    CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

kernel void GetMipData(read_only image2d_t img, global float* dst)
{
    int numMipLevels = get_image_num_mip_levels(img);
    dst[0] = numMipLevels;
    dst[1] = read_imagef(img, sampler, (float2)(0.5f, 0.5f)).x;

    numMipLevels = max(1, numMipLevels);
    for (int m = 0; m < numMipLevels + 3; m++) {
        float4 mipData = read_imagef(img, sampler, (float2)(0.5f, 0.5f), (float)m);
        dst[m + 2] = mipData.x;
    }
}
)CLC";

int main(
    int argc,
    char** argv )
{
    int platformIndex = 0;
    int deviceIndex = 0;

    cl_uint mipLevels = 6;

    {
        popl::OptionParser op("Supported Options");
        op.add<popl::Value<int>>("p", "platform", "Platform Index", platformIndex, &platformIndex);
        op.add<popl::Value<int>>("d", "device", "Device Index", deviceIndex, &deviceIndex);
        op.add<popl::Value<cl_uint>>("m", "miplevels", "Number of Mipmap Levels", mipLevels, &mipLevels);

        bool printUsage = false;
        try {
            op.parse(argc, argv);
        } catch (std::exception& e) {
            fprintf(stderr, "Error: %s\n\n", e.what());
            printUsage = true;
        }

        if (printUsage || !op.unknown_options().empty() || !op.non_option_args().empty()) {
            fprintf(stderr,
                "Usage: mipmapimage [options]\n"
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

    bool has_cl_khr_mipmap_image =
        checkDeviceForExtension(devices[deviceIndex], CL_KHR_MIPMAP_IMAGE_EXTENSION_NAME);
    if (has_cl_khr_mipmap_image) {
        printf("Device supports " CL_KHR_MIPMAP_IMAGE_EXTENSION_NAME ".\n");
    } else {
        printf("Device does not support " CL_KHR_MIPMAP_IMAGE_EXTENSION_NAME ", exiting.\n");
        return -1;
    }

    cl::Context context{devices[deviceIndex]};
    cl::CommandQueue commandQueue{context, devices[deviceIndex]};

    cl::Program program{ context, kernelString };
    program.build("-cl-std=CL3.0");
    cl::Kernel kernel = cl::Kernel{ program, "GetMipData" };

    if (mipLevels > 20) {
        printf("Maximum number of supported mip levels is 20.\n");
        mipLevels = 20;
    }

    size_t imageWidth = std::max(32, 1 << mipLevels);
    size_t imageHeight = std::max(32, 1 << mipLevels);

    cl_image_format imgFormat{};
    imgFormat.image_channel_order = CL_R;
    imgFormat.image_channel_data_type = CL_FLOAT;

    cl_image_desc imgDesc{};
    imgDesc.image_type = CL_MEM_OBJECT_IMAGE2D;
    imgDesc.image_width = imageWidth;
    imgDesc.image_height = imageHeight;
    imgDesc.num_mip_levels = mipLevels;

    printf("Creating a %zu x %zu image with %u mip levels...\n",
        imgDesc.image_width, imgDesc.image_height, imgDesc.num_mip_levels);

    cl::Image2D img = cl::Image2D{
        clCreateImage(
            context(),
            CL_MEM_READ_ONLY,
            &imgFormat,
            &imgDesc,
            nullptr,
            nullptr)};

    cl::Buffer buf = cl::Buffer{
        context,
        CL_MEM_ALLOC_HOST_PTR,
        64 * sizeof(cl_float)};

    // Things to add, things to test:
    // - How do you create a sampler with float properties?
    // - FillImage to nonzero mip levels.
    // - What should get_image_num_mip_levels return when num_mip_levels is zero?

    mipLevels = std::max(mipLevels, 1U);

    // initialization
    {
        printf("Initializing mipmap image...\n");
        for (cl_uint m = 0; m < mipLevels; m++) {
#if 0
            const cl_float v = (m + 1.0f) / mipLevels;
            const cl_float4 mipData{v, v, v, v};
            commandQueue.enqueueFillImage(
                img,
                mipData,
                {0, 0, m},
                {imageWidth >> m, imageHeight >> m, 1});
#else
            const cl_float v = (m + 1.0f) / mipLevels;
            size_t rowPitch = 0;
            auto pImage = (cl_float*)commandQueue.enqueueMapImage(
                img,
                CL_TRUE,
                CL_MAP_READ | CL_MAP_WRITE,
                {0, 0, m},
                {imageWidth >> m, imageHeight >> m, 1},
                &rowPitch,
                nullptr);

            for (size_t y = 0; y < imageHeight >> m; y++) {
                for (size_t x = 0; x < imageWidth >> m; x++) {
                    pImage[y * rowPitch / sizeof(cl_float) + x] = v;
                }
            }

            commandQueue.enqueueUnmapMemObject(
                img,
                pImage);
#endif
        }
        commandQueue.finish();
    }

    // execution
    kernel.setArg(0, img);
    kernel.setArg(1, buf);
    commandQueue.enqueueNDRangeKernel(
        kernel,
        cl::NullRange,
        cl::NDRange{1} );

    // verification
    {
        auto pData = (const float*)commandQueue.enqueueMapBuffer(
            buf,
            CL_TRUE,
            CL_MAP_READ,
            0,
            64 * sizeof(cl_float));

        printf("Found %.f mip levels.\n", pData[0]);
        printf("At base level: found %f, wanted %f\n", pData[1], 1.0f / mipLevels);

        for (cl_uint m = 0; m < mipLevels; m++) {
            const cl_float v = (m + 1.0f) / mipLevels;
            printf("At mip level %u: found %f, wanted %f\n", m, pData[m + 2], v);
        }

        for (cl_uint m = 0; m < 3; m++) {
            printf("At out-of-range mip level %u: found %f\n", m + mipLevels, pData[m + mipLevels + 2]);
        }

        commandQueue.enqueueUnmapMemObject(
            buf,
            (void*)pData);
    }

    return 0;
}
