/*
// Copyright (c) 2025-2026 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <stdio.h>
#include <cinttypes>
#include <vector>
#include <popl/popl.hpp>

#include <CL/cl_ext.h>

#ifndef cl_khr_spirv_queries
#define cl_khr_spirv_queries 1
#define CL_KHR_SPIRV_QUERIES_EXTENSION_NAME "cl_khr_spirv_queries"
#define CL_DEVICE_SPIRV_EXTENDED_INSTRUCTION_SETS_KHR   0x12B9
#define CL_DEVICE_SPIRV_EXTENSIONS_KHR                  0x12BA
#define CL_DEVICE_SPIRV_CAPABILITIES_KHR                0x12BB
#endif

#define SPV_ENABLE_UTILITY_CODE
#include <spirv/unified1/spirv.hpp>

#include <CL/opencl.hpp>

#include "util.hpp"

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
                "Usage: spirvqueries [options]\n"
                "%s", op.help().c_str());
            return -1;
        }
    }

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    for (size_t p = 0; p < platforms.size(); p++) {
        const cl::Platform& platform = platforms[p];

        printf("Platform[%zu]:\n", p);
        printf("\tName:             %s\n", platform.getInfo<CL_PLATFORM_NAME>().c_str() );
        printf("\tVendor:           %s\n", platform.getInfo<CL_PLATFORM_VENDOR>().c_str() );
        printf("\tPlatform Version: %s\n", platform.getInfo<CL_PLATFORM_VERSION>().c_str() );

        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
        for (size_t d = 0; d < devices.size(); d++) {
            const cl::Device& device = devices[d];

            printf("\tDevice[%zu]:\n", d);
            printf("\tName:           %s\n", device.getInfo<CL_DEVICE_NAME>().c_str() );
            printf("\tVendor:         %s\n", device.getInfo<CL_DEVICE_VENDOR>().c_str() );
            printf("\tDevice Version: %s\n", device.getInfo<CL_DEVICE_VERSION>().c_str() );
            printf("\tDriver Version: %s\n", device.getInfo<CL_DRIVER_VERSION>().c_str() );

            auto spirvExtendedInstructionSets =
                device.getInfo<CL_DEVICE_SPIRV_EXTENDED_INSTRUCTION_SETS_KHR>();
            printf("\t\tSupported SPIR-V ExtendedInstructionSets:\n");
            for (auto s : spirvExtendedInstructionSets) {
                printf("\t\t\t%s\n", s);
            }

            auto spirvExtensions =
                device.getInfo<CL_DEVICE_SPIRV_EXTENSIONS_KHR>();
            printf("\t\tSupported SPIR-V Extensions:\n");
            for (auto s : spirvExtensions) {
                printf("\t\t\t%s\n", s);
            }

            auto spirvCapabilities =
                device.getInfo<CL_DEVICE_SPIRV_CAPABILITIES_KHR>();
            printf("\t\tSupported SPIR-V Capabilities:\n");
            for (auto c : spirvCapabilities) {
                printf("\t\t\t%s\n", spv::CapabilityToString(static_cast<spv::Capability>(c)));
            }
        }
        printf("\n");
    }

    printf( "Done.\n" );

    return 0;
}
