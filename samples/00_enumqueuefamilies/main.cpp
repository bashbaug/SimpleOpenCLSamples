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

#include <stdio.h>
#include <vector>

#include <CL/cl_ext.h>

// This is only needed until this extension is added to the upstream headers.
#ifndef cl_intel_command_queue_families

/***************************************************************
* cl_intel_command_queue_families
***************************************************************/
#define cl_intel_command_queue_families 1

typedef cl_bitfield         cl_command_queue_capabilities_intel;

#define CL_QUEUE_FAMILY_MAX_NAME_SIZE_INTEL                 64

typedef struct _cl_queue_family_properties_intel {
    cl_command_queue_properties properties;
    cl_command_queue_capabilities_intel capabilities;
    cl_uint count;
    char name[CL_QUEUE_FAMILY_MAX_NAME_SIZE_INTEL];
} cl_queue_family_properties_intel;

/* cl_device_info */
#define CL_DEVICE_QUEUE_FAMILY_PROPERTIES_INTEL             0x418B

/* cl_queue_properties */
#define CL_QUEUE_FAMILY_INTEL                               0x418C
#define CL_QUEUE_INDEX_INTEL                                0x418D

/* cl_command_queue_capabilities_intel */
#define CL_QUEUE_DEFAULT_CAPABILITIES_INTEL                 0
#define CL_QUEUE_CAPABILITY_CREATE_SINGLE_QUEUE_EVENTS_INTEL (1 << 0)
#define CL_QUEUE_CAPABILITY_CREATE_CROSS_QUEUE_EVENTS_INTEL (1 << 1)
#define CL_QUEUE_CAPABILITY_SINGLE_QUEUE_EVENT_WAIT_LIST_INTEL (1 << 2)
#define CL_QUEUE_CAPABILITY_CROSS_QUEUE_EVENT_WAIT_LIST_INTEL (1 << 3)
#define CL_QUEUE_CAPABILITY_TRANSFER_BUFFER_INTEL           (1 << 8)
#define CL_QUEUE_CAPABILITY_TRANSFER_BUFFER_RECT_INTEL      (1 << 9)
#define CL_QUEUE_CAPABILITY_MAP_BUFFER_INTEL                (1 << 10)
#define CL_QUEUE_CAPABILITY_FILL_BUFFER_INTEL               (1 << 11)
#define CL_QUEUE_CAPABILITY_TRANSFER_IMAGE_INTEL            (1 << 12)
#define CL_QUEUE_CAPABILITY_MAP_IMAGE_INTEL                 (1 << 13)
#define CL_QUEUE_CAPABILITY_FILL_IMAGE_INTEL                (1 << 14)
#define CL_QUEUE_CAPABILITY_TRANSFER_BUFFER_IMAGE_INTEL     (1 << 15)
#define CL_QUEUE_CAPABILITY_TRANSFER_IMAGE_BUFFER_INTEL     (1 << 16)
#define CL_QUEUE_CAPABILITY_MARKER_INTEL                    (1 << 24)
#define CL_QUEUE_CAPABILITY_BARRIER_INTEL                   (1 << 25)
#define CL_QUEUE_CAPABILITY_KERNEL_INTEL                    (1 << 26)

#endif

#include <CL/opencl.hpp>

#include "util.hpp"

static void PrintDeviceType(
    const char* label,
    cl_device_type type )
{
    printf("%s%s%s%s%s%s\n",
        label,
        ( type & CL_DEVICE_TYPE_DEFAULT     ) ? "DEFAULT "      : "",
        ( type & CL_DEVICE_TYPE_CPU         ) ? "CPU "          : "",
        ( type & CL_DEVICE_TYPE_GPU         ) ? "GPU "          : "",
        ( type & CL_DEVICE_TYPE_ACCELERATOR ) ? "ACCELERATOR "  : "",
        ( type & CL_DEVICE_TYPE_CUSTOM      ) ? "CUSTOM "       : "");
}

static void PrintQueueProperties(
    cl_command_queue_properties props )
{
    if (props & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE  ) printf("\t\t\tCL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE\n");
    if (props & CL_QUEUE_PROFILING_ENABLE               ) printf("\t\t\tCL_QUEUE_PROFILING_ENABLE\n");
#ifdef CL_VERSION_2_0
    if (props & CL_QUEUE_ON_DEVICE                      ) printf("\t\t\tCL_QUEUE_ON_DEVICE\n");
    if (props & CL_QUEUE_ON_DEVICE_DEFAULT              ) printf("\t\t\tCL_QUEUE_ON_DEVICE_DEFAULT\n");
#endif
}

static void PrintQueueCapabilities(
    cl_command_queue_capabilities_intel caps )
{
    if (caps == CL_QUEUE_DEFAULT_CAPABILITIES_INTEL) {
        printf("\t\t\tDEFAULT\n");
    } else {
        if (caps & CL_QUEUE_CAPABILITY_CREATE_SINGLE_QUEUE_EVENTS_INTEL    ) printf("\t\t\tCREATE_SINGLE_QUEUE_EVENTS_INTEL\n");
        if (caps & CL_QUEUE_CAPABILITY_CREATE_CROSS_QUEUE_EVENTS_INTEL     ) printf("\t\t\tCREATE_CROSS_QUEUE_EVENTS_INTEL\n");
        if (caps & CL_QUEUE_CAPABILITY_SINGLE_QUEUE_EVENT_WAIT_LIST_INTEL  ) printf("\t\t\tSINGLE_QUEUE_EVENT_WAIT_LIST_INTEL\n");
        if (caps & CL_QUEUE_CAPABILITY_CROSS_QUEUE_EVENT_WAIT_LIST_INTEL   ) printf("\t\t\tCROSS_QUEUE_EVENT_WAIT_LIST_INTEL\n");
        if (caps & CL_QUEUE_CAPABILITY_TRANSFER_BUFFER_INTEL               ) printf("\t\t\tTRANSFER_BUFFER_INTEL\n");
        if (caps & CL_QUEUE_CAPABILITY_TRANSFER_BUFFER_RECT_INTEL          ) printf("\t\t\tTRANSFER_BUFFER_RECT_INTEL\n");
        if (caps & CL_QUEUE_CAPABILITY_MAP_BUFFER_INTEL                    ) printf("\t\t\tMAP_BUFFER_INTEL\n");
        if (caps & CL_QUEUE_CAPABILITY_FILL_BUFFER_INTEL                   ) printf("\t\t\tFILL_BUFFER_INTEL\n");
        if (caps & CL_QUEUE_CAPABILITY_TRANSFER_IMAGE_INTEL                ) printf("\t\t\tTRANSFER_IMAGE_INTEL\n");
        if (caps & CL_QUEUE_CAPABILITY_MAP_IMAGE_INTEL                     ) printf("\t\t\tMAP_IMAGE_INTEL\n");
        if (caps & CL_QUEUE_CAPABILITY_FILL_IMAGE_INTEL                    ) printf("\t\t\tFILL_IMAGE_INTEL\n");
        if (caps & CL_QUEUE_CAPABILITY_TRANSFER_BUFFER_IMAGE_INTEL         ) printf("\t\t\tTRANSFER_BUFFER_IMAGE_INTEL\n");
        if (caps & CL_QUEUE_CAPABILITY_TRANSFER_IMAGE_BUFFER_INTEL         ) printf("\t\t\tTRANSFER_IMAGE_BUFFER_INTEL\n");
        if (caps & CL_QUEUE_CAPABILITY_MARKER_INTEL                        ) printf("\t\t\tMARKER_INTEL\n");
        if (caps & CL_QUEUE_CAPABILITY_BARRIER_INTEL                       ) printf("\t\t\tBARRIER_INTEL\n");
        if (caps & CL_QUEUE_CAPABILITY_KERNEL_INTEL                        ) printf("\t\t\tKERNEL_INTEL\n");
    }
}

static void PrintPlatformInfoSummary(
    cl::Platform platform )
{
    printf("\tName:           %s\n", platform.getInfo<CL_PLATFORM_NAME>().c_str() );
    printf("\tVendor:         %s\n", platform.getInfo<CL_PLATFORM_VENDOR>().c_str() );
    printf("\tDriver Version: %s\n", platform.getInfo<CL_PLATFORM_VERSION>().c_str() );
}

static void PrintDeviceInfoSummary(
    const std::vector<cl::Device> devices )
{
    size_t  i = 0;
    for( i = 0; i < devices.size(); i++ )
    {
        printf("Device[%d]:\n", (int)i );

        cl_device_type  deviceType = devices[i].getInfo<CL_DEVICE_TYPE>();
        PrintDeviceType("\tType:           ", deviceType);

        printf("\tName:           %s\n", devices[i].getInfo<CL_DEVICE_NAME>().c_str() );
        printf("\tVendor:         %s\n", devices[i].getInfo<CL_DEVICE_VENDOR>().c_str() );
        printf("\tDevice Version: %s\n", devices[i].getInfo<CL_DEVICE_VERSION>().c_str() );
        printf("\tDriver Version: %s\n", devices[i].getInfo<CL_DRIVER_VERSION>().c_str() );

        if (checkDeviceForExtension(devices[i], "cl_intel_command_queue_families")) {
            cl_int test;
            std::vector<cl_queue_family_properties_intel>   qfprops =
                devices[i].getInfo<CL_DEVICE_QUEUE_FAMILY_PROPERTIES_INTEL>(&test);
            for ( size_t q = 0; q < qfprops.size(); q++ ) {
                printf("\tQueue Family %i:\n", (int)q);
                printf("\t\tName:   %s\n", qfprops[q].name);
                printf("\t\tCount:  %u\n", qfprops[q].count);
                printf("\t\tProperties:\n");
                PrintQueueProperties(qfprops[q].properties);
                printf("\t\tCapabilities:\n");
                PrintQueueCapabilities(qfprops[q].capabilities);
            }
        } else {
            printf("\tThis device does not support queue families.\n");
        }
    }
}

int main(
    int argc,
    char** argv )
{
    bool printUsage = false;

    int i = 0;

    if( argc < 1 )
    {
        printUsage = true;
    }
    else
    {
        for( i = 1; i < argc; i++ )
        {
            {
                printUsage = true;
            }
        }
    }
    if( printUsage )
    {
        fprintf(stderr,
            "Usage: enumqueuefamilies [options]\n"
            "Options:\n"
            );

        return -1;
    }

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    for( auto& platform : platforms )
    {
        printf( "Platform:\n" );
        PrintPlatformInfoSummary( platform );

        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

        PrintDeviceInfoSummary( devices );
        printf( "\n" );
    }

    printf( "Done.\n" );

    return 0;
}
