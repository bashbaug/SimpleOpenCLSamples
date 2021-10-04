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

#include <algorithm>

static const char kernelString[] = R"CLC(
constant sampler_t samp =
    CLK_FILTER_NEAREST | CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE;
kernel void ImageToBuffer(global uchar* dst, read_only image2d_t img)
{
    uint w = get_global_id(0);
    uint h = get_global_id(1);
    uint p = get_image_width(img);

    float f = read_imagef(img, samp, (int2)(w, h)).x;
    dst[h * p + w] = convert_uchar(f * 255);
}
)CLC";

int main(
    int argc,
    char** argv )
{
    cl_int errorCode = CL_SUCCESS;

    int platformIndex = 0;
    int deviceIndex = 0;
    size_t width = 3;
    size_t height = 2;
    size_t pitch = 0;
    bool useKernel;

    {
        popl::OptionParser op("Supported Options");
        op.add<popl::Value<int>>("p", "platform", "Platform Index", platformIndex, &platformIndex);
        op.add<popl::Value<int>>("d", "device", "Device Index", deviceIndex, &deviceIndex);
        op.add<popl::Value<size_t>>("w", "width", "Image Width", width, &width);
        op.add<popl::Value<size_t>>("h", "height", "Image Height", height, &height);
        op.add<popl::Value<size_t>>("", "pitch", "Test Image Pitch", pitch, &pitch);
        op.add<popl::Switch>("k", "kernel", "Use a Kernel to Copy Image", &useKernel);

        bool printUsage = false;
        try {
            op.parse(argc, argv);
        } catch (std::exception& e) {
            fprintf(stderr, "Error: %s\n\n", e.what());
            printUsage = true;
        }

        if (printUsage || !op.unknown_options().empty() || !op.non_option_args().empty()) {
            fprintf(stderr,
                "Usage: rowpitchtester [options]\n"
                "%s", op.help().c_str());
            return -1;
        }
    }

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    printf("Running on platform: %s\n",
        platforms[platformIndex].getInfo<CL_PLATFORM_NAME>().c_str());

    std::vector<cl::Device> devices;
    platforms[platformIndex].getDevices(CL_DEVICE_TYPE_ALL, &devices);

    printf("Running on device: %s\n",
        devices[deviceIndex].getInfo<CL_DEVICE_NAME>().c_str());
    printf("CL_DEVICE_IMAGE_PITCH_ALIGNMENT is %u\n",
        devices[deviceIndex].getInfo<CL_DEVICE_IMAGE_PITCH_ALIGNMENT_KHR>());
    printf("CL_DEVICE_IMAGE_SUPPORT is %s\n",
        devices[deviceIndex].getInfo<CL_DEVICE_IMAGE_SUPPORT>() == CL_TRUE ? "CL_TRUE" : "CL_FALSE");

    if (pitch == 0) {
        pitch = width;
        printf("Calculated pitch as %zu.\n", pitch);
    }

    printf("Using: width = %zu, height = %zu, pitch = %zu\n", width, height, pitch);

    cl::Context context{devices[deviceIndex]};
    cl::CommandQueue commandQueue = cl::CommandQueue{context, devices[deviceIndex]};

#if 1
    // By default, allocate a bigger buffer to check if the image goes out-of-bounds.
    size_t maxPitch = std::min((size_t)1024, pitch) * 2;
    size_t maxHeight = height + 1;
#else
    // This will allocate a buffer that is exactly the size of the image, instead.
    size_t maxPitch = pitch;
    size_t maxHeight = height;
#endif

    cl::Buffer srcBuf = cl::Buffer{
        context,
        CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
        maxPitch * maxHeight };

    // initialization
    {
        auto pSrc = (cl_uchar*)commandQueue.enqueueMapBuffer(
            srcBuf,
            CL_TRUE,
            CL_MAP_WRITE_INVALIDATE_REGION,
            0,
            maxPitch * maxHeight);

        // valid image data: 0xFF
        // padding data (pitch): 0x80
        // padding data (other): 0x40

        for (size_t i = 0; i < maxPitch * maxHeight; i++) {
            pSrc[i] = 0x40;
        }

        for (size_t h = 0; h < height; h++) {
            for (size_t w = 0; w < pitch; w++) {
                if (w < width) {
                    pSrc[h * pitch + w] = 0xFF;
                } else {
                    pSrc[h * pitch + w] = 0x80;
                }
            }
        }

        commandQueue.enqueueUnmapMemObject(
            srcBuf,
            pSrc);
    }

    cl::Buffer dstBuf = cl::Buffer{
        context,
        CL_MEM_ALLOC_HOST_PTR,
        width * height };

    cl::Image2D srcImg = cl::Image2D{
        context,
        cl::ImageFormat{CL_R, CL_UNORM_INT8},
        srcBuf,
        width,
        height };

    if (useKernel) {
        cl::Program program{context, kernelString};
        program.build();
        cl::Kernel kernel{program, "ImageToBuffer"};
        kernel.setArg(0, dstBuf);
        kernel.setArg(1, srcImg);
        commandQueue.enqueueNDRangeKernel(
            kernel,
            cl::NullRange,
            cl::NDRange{width, height});
    } else {
        commandQueue.enqueueCopyImageToBuffer(
            srcImg,
            dstBuf,
            {0, 0, 0},
            {width, height, 1},
            0);
    }

    // verification via image
    {
        size_t rowPitch = 0;
        auto pDst = (const cl_uchar*)commandQueue.enqueueMapImage(
            srcImg,
            CL_TRUE,
            CL_MAP_READ,
            {0, 0, 0},
            {width, height, 1},
            &rowPitch,
            NULL );

        printf("Mapped image row pitch is %zu\n", rowPitch);

        unsigned int    mismatches = 0;

        for (size_t h = 0; h < height; h++) {
            for (size_t w = 0; w < width; w++) {
                if (pDst[h * rowPitch + w] != 0xFF) {
                    if (mismatches < 16) {
                        fprintf(stderr, "MisMatch!  dst[%zu, %zu] == %02X, want %02X\n",
                            w, h,
                            pDst[h * rowPitch + w],
                            0xFF );
                    }
                    mismatches++;
                }
            }
        }

        if (mismatches) {
            fprintf(stderr, "Image Verification Error: Found %d mismatches / %zu values!!!\n",
                mismatches,
                width * height);
        } else {
            printf("Image Verification Success.\n");
        }

        commandQueue.enqueueUnmapMemObject(
            dstBuf,
            (void*)pDst);
    }

    // verification via buffer
    {
        auto pDst = (const cl_uchar*)commandQueue.enqueueMapBuffer(
            dstBuf,
            CL_TRUE,
            CL_MAP_READ,
            0,
            width * height );

        unsigned int    mismatches = 0;

        for (size_t h = 0; h < height; h++) {
            for (size_t w = 0; w < width; w++) {
                if (pDst[h * width + w] != 0xFF) {
                    if (mismatches < 16) {
                        fprintf(stderr, "MisMatch!  dst[%zu, %zu] == %02X, want %02X\n",
                            w, h,
                            pDst[h * width + w],
                            0xFF );
                    }
                    mismatches++;
                }
            }
        }

        if (mismatches) {
            fprintf(stderr, "Buffer Verification Error: Found %d mismatches / %zu values!!!\n",
                mismatches,
                width * height);
        } else {
            printf("Buffer Verification Success.\n");
        }

        commandQueue.enqueueUnmapMemObject(
            dstBuf,
            (void*)pDst);
    }

    commandQueue.finish();
    return 0;
}
