/*
// Copyright (c) 2020 Ben Ashbaugh
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
#include "libusm.h"

void PrintUSMCaps(
    const char* label,
    cl_device_unified_shared_memory_capabilities_intel usmcaps )
{
    printf("%s: %s%s%s%s\n",
        label,
        ( usmcaps & CL_UNIFIED_SHARED_MEMORY_ACCESS_INTEL                   ) ? "\n\t\tCL_UNIFIED_SHARED_MEMORY_ACCESS_INTEL"                   : "",
        ( usmcaps & CL_UNIFIED_SHARED_MEMORY_ATOMIC_ACCESS_INTEL            ) ? "\n\t\tCL_UNIFIED_SHARED_MEMORY_ATOMIC_ACCESS_INTEL"            : "",
        ( usmcaps & CL_UNIFIED_SHARED_MEMORY_CONCURRENT_ACCESS_INTEL        ) ? "\n\t\tCL_UNIFIED_SHARED_MEMORY_CONCURRENT_ACCESS_INTEL"        : "",
        ( usmcaps & CL_UNIFIED_SHARED_MEMORY_CONCURRENT_ATOMIC_ACCESS_INTEL ) ? "\n\t\tCL_UNIFIED_SHARED_MEMORY_CONCURRENT_ATOMIC_ACCESS_INTEL" : "" );
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
        bool printUsage = false;
        try {
            op.parse(argc, argv);
        } catch (std::exception& e) {
            fprintf(stderr, "Error: %s\n\n", e.what());
            printUsage = true;
        }

        if (printUsage || !op.unknown_options().empty() || !op.non_option_args().empty()) {
            fprintf(stderr,
                "Usage: usmqueries [options]\n"
                "%s", op.help().c_str());
            return -1;
        }
    }

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    printf("Running on platform: %s\n",
        platforms[platformIndex].getInfo<CL_PLATFORM_NAME>().c_str() );
    libusm::initialize(platforms[platformIndex]());

    std::vector<cl::Device> devices;
    platforms[platformIndex].getDevices(CL_DEVICE_TYPE_ALL, &devices);

    printf("Running on device: %s\n",
        devices[deviceIndex].getInfo<CL_DEVICE_NAME>().c_str() );

    cl_device_unified_shared_memory_capabilities_intel usmcaps = 0;

    clGetDeviceInfo(
        devices[deviceIndex](),
        CL_DEVICE_HOST_MEM_CAPABILITIES_INTEL,
        sizeof(usmcaps),
        &usmcaps,
        nullptr );
    PrintUSMCaps( "CL_DEVICE_HOST_MEM_CAPABILITIES_INTEL", usmcaps );

    clGetDeviceInfo(
        devices[deviceIndex](),
        CL_DEVICE_DEVICE_MEM_CAPABILITIES_INTEL,
        sizeof(usmcaps),
        &usmcaps,
        nullptr );
    PrintUSMCaps( "CL_DEVICE_DEVICE_MEM_CAPABILITIES_INTEL", usmcaps );

    clGetDeviceInfo(
        devices[deviceIndex](),
        CL_DEVICE_SINGLE_DEVICE_SHARED_MEM_CAPABILITIES_INTEL,
        sizeof(usmcaps),
        &usmcaps,
        nullptr );
    PrintUSMCaps( "CL_DEVICE_SINGLE_DEVICE_SHARED_MEM_CAPABILITIES_INTEL", usmcaps );

    clGetDeviceInfo(
        devices[deviceIndex](),
        CL_DEVICE_CROSS_DEVICE_SHARED_MEM_CAPABILITIES_INTEL,
        sizeof(usmcaps),
        &usmcaps,
        nullptr );
    PrintUSMCaps( "CL_DEVICE_CROSS_DEVICE_SHARED_MEM_CAPABILITIES_INTEL", usmcaps );

    clGetDeviceInfo(
        devices[deviceIndex](),
        CL_DEVICE_SHARED_SYSTEM_MEM_CAPABILITIES_INTEL,
        sizeof(usmcaps),
        &usmcaps,
        nullptr );
    PrintUSMCaps( "CL_DEVICE_SHARED_SYSTEM_MEM_CAPABILITIES_INTEL", usmcaps );

    printf("Cleaning up...\n");

    return 0;
}