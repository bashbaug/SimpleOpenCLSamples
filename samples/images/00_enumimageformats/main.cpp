/*
// Copyright (c) 2019 Ben Ashbaugh
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

#include <stdio.h>
#include <vector>

#include <CL/opencl.hpp>

#define CASE_TO_STRING(_e) case _e: return #_e;

const char* mem_object_type_to_string(cl_mem_object_type mem_object_type)
{
    switch (mem_object_type) {
    CASE_TO_STRING(CL_MEM_OBJECT_BUFFER);
    CASE_TO_STRING(CL_MEM_OBJECT_IMAGE2D);
    CASE_TO_STRING(CL_MEM_OBJECT_IMAGE3D);
#ifdef CL_VERSION_1_2
    CASE_TO_STRING(CL_MEM_OBJECT_IMAGE2D_ARRAY);
    CASE_TO_STRING(CL_MEM_OBJECT_IMAGE1D);
    CASE_TO_STRING(CL_MEM_OBJECT_IMAGE1D_ARRAY);
    CASE_TO_STRING(CL_MEM_OBJECT_IMAGE1D_BUFFER);
#endif
#ifdef CL_VERSION_2_0
    CASE_TO_STRING(CL_MEM_OBJECT_PIPE);
#endif
    default: return "Unknown cl_mem_object_type";
    }
}

const char* mem_flags_to_string(cl_mem_flags mem_flags)
{
    switch (mem_flags) {
    CASE_TO_STRING(CL_MEM_READ_WRITE);
    CASE_TO_STRING(CL_MEM_WRITE_ONLY);
    CASE_TO_STRING(CL_MEM_READ_ONLY);
    CASE_TO_STRING(CL_MEM_USE_HOST_PTR);
    CASE_TO_STRING(CL_MEM_ALLOC_HOST_PTR);
    CASE_TO_STRING(CL_MEM_COPY_HOST_PTR);
#ifdef CL_VERSION_1_2
    CASE_TO_STRING(CL_MEM_HOST_WRITE_ONLY)
    CASE_TO_STRING(CL_MEM_HOST_READ_ONLY);
    CASE_TO_STRING(CL_MEM_HOST_NO_ACCESS);
#endif
#ifdef CL_VERSION_2_0
    CASE_TO_STRING(CL_MEM_SVM_FINE_GRAIN_BUFFER);
    CASE_TO_STRING(CL_MEM_SVM_ATOMICS);
    CASE_TO_STRING(CL_MEM_KERNEL_READ_AND_WRITE);
#endif
    default: return "Unknown cl_mem_flags";
    }
}

const char* channel_order_to_string(cl_channel_order channel_order)
{
    switch (channel_order) {
    CASE_TO_STRING(CL_R);
    CASE_TO_STRING(CL_A);
    CASE_TO_STRING(CL_RG);
    CASE_TO_STRING(CL_RA);
    CASE_TO_STRING(CL_RGB);
    CASE_TO_STRING(CL_RGBA);
    CASE_TO_STRING(CL_BGRA);
    CASE_TO_STRING(CL_ARGB);
    CASE_TO_STRING(CL_INTENSITY);
    CASE_TO_STRING(CL_LUMINANCE);
#ifdef CL_VERSION_1_1
    CASE_TO_STRING(CL_Rx);
    CASE_TO_STRING(CL_RGx);
    CASE_TO_STRING(CL_RGBx);
#endif
#ifdef CL_VERSION_1_2
    CASE_TO_STRING(CL_DEPTH);
    CASE_TO_STRING(CL_DEPTH_STENCIL);
#endif
#ifdef CL_VERSION_2_0
    CASE_TO_STRING(CL_sRGB);
    CASE_TO_STRING(CL_sRGBx);
    CASE_TO_STRING(CL_sRGBA);
    CASE_TO_STRING(CL_sBGRA);
    CASE_TO_STRING(CL_ABGR);
#endif
    default: return "Unknown cl_channel_order";
    }
}

const char* channel_type_to_string(cl_channel_type channel_type)
{
    switch (channel_type) {
    CASE_TO_STRING(CL_SNORM_INT8);
    CASE_TO_STRING(CL_SNORM_INT16);
    CASE_TO_STRING(CL_UNORM_INT8);
    CASE_TO_STRING(CL_UNORM_INT16);
    CASE_TO_STRING(CL_UNORM_SHORT_565);
    CASE_TO_STRING(CL_UNORM_SHORT_555);
    CASE_TO_STRING(CL_UNORM_INT_101010);
    CASE_TO_STRING(CL_SIGNED_INT8);
    CASE_TO_STRING(CL_SIGNED_INT16);
    CASE_TO_STRING(CL_SIGNED_INT32);
    CASE_TO_STRING(CL_UNSIGNED_INT8);
    CASE_TO_STRING(CL_UNSIGNED_INT16);
    CASE_TO_STRING(CL_UNSIGNED_INT32);
    CASE_TO_STRING(CL_HALF_FLOAT);
    CASE_TO_STRING(CL_FLOAT);
#ifdef CL_VERSION_1_2
    CASE_TO_STRING(CL_UNORM_INT24);
#endif
#ifdef CL_VERSION_2_1
    CASE_TO_STRING(CL_UNORM_INT_101010_2);
#endif
    default: return "Unknown cl_channel_type";
    }
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
            "Usage: enumimageformats    [options]\n"
            "Options:\n"
            "      -d: Device Index (default = 0)\n"
            "      -p: Platform Index (default = 0)\n"
            );

        return -1;
    }

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    printf("Querying platform: %s\n",
        platforms[platformIndex].getInfo<CL_PLATFORM_NAME>().c_str() );

    std::vector<cl::Device> devices;
    platforms[platformIndex].getDevices(CL_DEVICE_TYPE_ALL, &devices);

    printf("Querying device: %s\n",
        devices[deviceIndex].getInfo<CL_DEVICE_NAME>().c_str() );

    const std::vector<cl_mem_flags> imageAccesses{
        {
            CL_MEM_READ_ONLY,
            CL_MEM_WRITE_ONLY,
            CL_MEM_READ_WRITE,
#ifdef CL_VERSION_2_0
            CL_MEM_KERNEL_READ_AND_WRITE,
#endif
        },
    };
    const std::vector<cl_mem_object_type> imageTypes{
        {
#ifdef CL_VERSION_1_2
            CL_MEM_OBJECT_IMAGE1D,
            CL_MEM_OBJECT_IMAGE1D_BUFFER,
#endif
            CL_MEM_OBJECT_IMAGE2D,
            CL_MEM_OBJECT_IMAGE3D,
#ifdef CL_VERSION_1_2
            CL_MEM_OBJECT_IMAGE1D_ARRAY,
            CL_MEM_OBJECT_IMAGE2D_ARRAY,
#endif
        }
    };

    cl::Context context{devices[deviceIndex]};

    for( auto& imageAccess : imageAccesses )
    {
        for( auto& imageType : imageTypes )
        {
            std::vector<cl::ImageFormat> imageFormats;
            context.getSupportedImageFormats( imageAccess, imageType, &imageFormats );

            printf("\nFor image access %s (%04X), image type %s (%04X):\n",
                mem_flags_to_string(imageAccess),
                (cl_uint)imageAccess,
                mem_object_type_to_string(imageType),
                (cl_uint)imageType);
            for( auto& imageFormat : imageFormats )
            {
                printf("\tChannel Order: %s (%04X),\tChannel Data Type: %s (%04X)\n",
                    channel_order_to_string(imageFormat.image_channel_order),
                    (cl_uint)imageFormat.image_channel_order,
                    channel_type_to_string(imageFormat.image_channel_data_type),
                    (cl_uint)imageFormat.image_channel_data_type);
            }
        }
    }

    printf( "Done.\n" );

    return 0;
}
