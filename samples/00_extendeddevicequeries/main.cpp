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
#include <cinttypes>
#include <vector>
#include <popl/popl.hpp>

#include <CL/cl_ext.h>

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

static void PrintDeviceFeatureCapabilities(
    cl_device_feature_capabilities_intel caps )
{
    if (caps & CL_DEVICE_FEATURE_FLAG_DP4A_INTEL    ) printf("\t\t\tCL_DEVICE_FEATURE_FLAG_DP4A_INTEL\n");
    if (caps & CL_DEVICE_FEATURE_FLAG_DPAS_INTEL    ) printf("\t\t\tCL_DEVICE_FEATURE_FLAG_DPAS_INTEL\n");
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

        if (checkDeviceForExtension(devices[i], "cl_amd_device_attribute_query")) {
            printf("\tFor: cl_amd_device_attribute_query:\n");
            printf("\tDevice Profiling Timer Offset:          %" PRIu64 "\n", devices[i].getInfo<CL_DEVICE_PROFILING_TIMER_OFFSET_AMD>());
            //devices[i].getInfo<CL_DEVICE_TOPOLOGY_AMD>();
            //devices[i].getInfo<CL_DEVICE_BOARD_NAME_AMD>();
            printf("\tDevice Global Free Memory:              %zu\n", devices[i].getInfo<CL_DEVICE_GLOBAL_FREE_MEMORY_AMD>()[0]);
            printf("\tDevice SIMD Per Compute Unit:           %u\n", devices[i].getInfo<CL_DEVICE_SIMD_PER_COMPUTE_UNIT_AMD>());
            printf("\tDevice SIMD Width:                      %u\n", devices[i].getInfo<CL_DEVICE_SIMD_WIDTH_AMD>());
            printf("\tDevice SIMD Instruction Width:          %u\n", devices[i].getInfo<CL_DEVICE_SIMD_INSTRUCTION_WIDTH_AMD>());
            printf("\tDevice Wavefront Width:                 %u\n", devices[i].getInfo<CL_DEVICE_WAVEFRONT_WIDTH_AMD>());
            printf("\tDevice Global Mem Channels:             %u\n", devices[i].getInfo<CL_DEVICE_GLOBAL_MEM_CHANNELS_AMD>());
            printf("\tDevice Global Mem Channel Banks:        %u\n", devices[i].getInfo<CL_DEVICE_GLOBAL_MEM_CHANNEL_BANKS_AMD>());
            printf("\tDevice Global Mem Channel Bank Width:   %u\n", devices[i].getInfo<CL_DEVICE_GLOBAL_MEM_CHANNEL_BANK_WIDTH_AMD>());
            printf("\tDevice Local Mem Size Per Compute Unit: %u\n", devices[i].getInfo<CL_DEVICE_LOCAL_MEM_SIZE_PER_COMPUTE_UNIT_AMD>());
            printf("\tDevice local Mem Banks:                 %u\n", devices[i].getInfo<CL_DEVICE_LOCAL_MEM_BANKS_AMD>());
            //devices[i].getInfo<CL_DEVICE_THREAD_TRACE_SUPPORTED_AMD>();
            //devices[i].getInfo<CL_DEVICE_GFXIP_MAJOR_AMD>();
            //devices[i].getInfo<CL_DEVICE_GFXIP_MINOR_AMD>();
            //devices[i].getInfo<CL_DEVICE_AVAILABLE_ASYNC_QUEUES_AMD>();
            //devices[i].getInfo<CL_DEVICE_PREFERRED_WORK_GROUP_SIZE_AMD>();
            //devices[i].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE_AMD>();
            //devices[i].getInfo<CL_DEVICE_PREFERRED_CONSTANT_BUFFER_SIZE_AMD>();
            //devices[i].getInfo<CL_DEVICE_PCIE_ID_AMD>();
        } else {
            printf("\tThis devices does not support cl_amd_device_attribute_query.\n");
        }

        if (checkDeviceForExtension(devices[i], "cl_intel_device_attribute_query")) {
            printf("\tFor: cl_intel_device_attribute_query:\n");
            if (deviceType & CL_DEVICE_TYPE_GPU) {
                printf("\tDevice IP Version:               %08X\n", devices[i].getInfo<CL_DEVICE_IP_VERSION_INTEL>());
                printf("\tDevice ID:                       %04X\n", devices[i].getInfo<CL_DEVICE_ID_INTEL>());
                printf("\tDevice Num Slices:               %u\n", devices[i].getInfo<CL_DEVICE_NUM_SLICES_INTEL>());
                printf("\tDevice Num Sub-slices Per Slice: %u\n", devices[i].getInfo<CL_DEVICE_NUM_SUB_SLICES_PER_SLICE_INTEL>());
                printf("\tDevice Num EUs Per Sub-Slice:    %u\n", devices[i].getInfo<CL_DEVICE_NUM_EUS_PER_SUB_SLICE_INTEL>());
                printf("\tDevice Num Threads Per EU:       %u\n", devices[i].getInfo<CL_DEVICE_NUM_THREADS_PER_EU_INTEL>());
                printf("\tDevice Feature Capabilities:\n");
                PrintDeviceFeatureCapabilities(devices[i].getInfo<CL_DEVICE_FEATURE_CAPABILITIES_INTEL>());
            } else {
                printf("\tUnknown device type for this extension.\n");
            }
        } else {
            printf("\tThis device does not support cl_intel_device_attribute_query.\n");
        }

        if (checkDeviceForExtension(devices[i], "cl_nv_device_attribute_query")) {
            printf("\tFor: cl_nv_device_attribute_query:\n");
            printf("\tDevice Compute Capability Major: %u\n", devices[i].getInfo<CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV>());
            printf("\tDevice Compute Capability Minor: %u\n", devices[i].getInfo<CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV>());
            printf("\tDevice Registers Per Block:      %u\n", devices[i].getInfo<CL_DEVICE_REGISTERS_PER_BLOCK_NV>());
            printf("\tDevice Warp Size:                %u\n", devices[i].getInfo<CL_DEVICE_WARP_SIZE_NV>());
            printf("\tDevice GPU Overlap:              %s\n", devices[i].getInfo<CL_DEVICE_GPU_OVERLAP_NV>() ? "true" : "false");
            printf("\tDevice Kernel Exec Timeout:      %s\n", devices[i].getInfo<CL_DEVICE_KERNEL_EXEC_TIMEOUT_NV>() ? "true" : "false");
            printf("\tDevice Integrated Memory:        %s\n", devices[i].getInfo<CL_DEVICE_INTEGRATED_MEMORY_NV>() ? "true" : "false");
        } else {
            printf("\tThis device does not support cl_nv_device_attribute_query.\n");
        }

    }
}

int main(
    int argc,
    char** argv )
{
    {
        popl::OptionParser op("Supported Options");

        bool printUsage = false;
        try {
            op.parse(argc, argv);
        } catch (std::exception& e) {
            fprintf(stderr, "Error: %s\n\n", e.what());
            printUsage = true;
        }

        if (printUsage || !op.unknown_options().empty() || !op.non_option_args().empty()) {
            fprintf(stderr,
                "Usage: extendeddevicequeries [options]\n"
                "%s", op.help().c_str());
            return -1;
        }
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
