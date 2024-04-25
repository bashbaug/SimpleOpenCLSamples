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
#include <popl/popl.hpp>

#include <CL/opencl.hpp>

#if defined(CL_VERSION_3_0)

static void PrintPlatformInfoSummary(
    cl::Platform platform )
{
    printf("\tName:           %s\n", platform.getInfo<CL_PLATFORM_NAME>().c_str() );
    printf("\tVendor:         %s\n", platform.getInfo<CL_PLATFORM_VENDOR>().c_str() );
    printf("\tDriver Version: %s\n", platform.getInfo<CL_PLATFORM_VERSION>().c_str() );

    // Use the query for the platform numeric version as a test for
    // OpenCL 3.0 support.  If this query fails then this probably
    // isn't an OpenCL 3.0 platform:

    cl_int test = CL_SUCCESS;
    cl_version platformVersion =
        platform.getInfo<CL_PLATFORM_NUMERIC_VERSION>(&test);
    if( test == CL_SUCCESS )
    {
        printf("\tPlatform Numeric Version: %u.%u.%u\n",
            CL_VERSION_MAJOR(platformVersion),
            CL_VERSION_MINOR(platformVersion),
            CL_VERSION_PATCH(platformVersion));
    }
    else
    {
        printf("Query for CL_PLATFORM_NUMERIC_VERSION failed.\n");
        return;
    }

    std::vector<cl_name_version>  extensions =
        platform.getInfo<CL_PLATFORM_EXTENSIONS_WITH_VERSION>();
    for( auto& extension : extensions )
    {
        printf("\t\tExtension (version): %s (%u.%u.%u)\n",
            extension.name,
            CL_VERSION_MAJOR(extension.version),
            CL_VERSION_MINOR(extension.version),
            CL_VERSION_PATCH(extension.version) );
    }
}

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

static void PrintDeviceAtomicCapabilities(
    const char* label,
    cl_device_atomic_capabilities caps )
{
    printf("%s: %s%s%s%s%s%s%s\n",
        label,
        ( caps & CL_DEVICE_ATOMIC_ORDER_RELAXED     ) ? "\n\t\tCL_DEVICE_ATOMIC_ORDER_RELAXED"      : "",
        ( caps & CL_DEVICE_ATOMIC_ORDER_ACQ_REL     ) ? "\n\t\tCL_DEVICE_ATOMIC_ORDER_ACQ_REL"      : "",
        ( caps & CL_DEVICE_ATOMIC_ORDER_SEQ_CST     ) ? "\n\t\tCL_DEVICE_ATOMIC_ORDER_SEQ_CST"      : "",
        ( caps & CL_DEVICE_ATOMIC_SCOPE_WORK_ITEM   ) ? "\n\t\tCL_DEVICE_ATOMIC_SCOPE_WORK_ITEM"    : "",
        ( caps & CL_DEVICE_ATOMIC_SCOPE_WORK_GROUP  ) ? "\n\t\tCL_DEVICE_ATOMIC_SCOPE_WORK_GROUP"   : "",
        ( caps & CL_DEVICE_ATOMIC_SCOPE_DEVICE      ) ? "\n\t\tCL_DEVICE_ATOMIC_SCOPE_DEVICE"       : "",
        ( caps & CL_DEVICE_ATOMIC_SCOPE_ALL_DEVICES ) ? "\n\t\tCL_DEVICE_ATOMIC_SCOPE_ALL_DEVICES"  : "");
}

#if defined(CL_DEVICE_DEVICE_ENQUEUE_CAPABILITIES)
static void PrintDeviceDeviceEnqueueCapabilities(
    const char* label,
    cl_device_device_enqueue_capabilities caps )
{
    printf("%s: %s%s\n",
        label,
        ( caps & CL_DEVICE_QUEUE_SUPPORTED          ) ? "\n\t\tCL_DEVICE_QUEUE_SUPPORTED"           : "",
        ( caps & CL_DEVICE_QUEUE_REPLACEABLE_DEFAULT) ? "\n\t\tCL_DEVICE_QUEUE_REPLACEABLE_DEFAULT" : "");
}
#endif

static void PrintDeviceInfoSummary(
    const std::vector<cl::Device>& devices )
{
    size_t  i = 0;
    for( i = 0; i < devices.size(); i++ )
    {
        printf("Device[%d]:\n", (int)i );

        cl_device_type  deviceType = devices[i].getInfo<CL_DEVICE_TYPE>();
        PrintDeviceType("\tType:           ", deviceType);

        printf("\tName:             %s\n", devices[i].getInfo<CL_DEVICE_NAME>().c_str() );
        printf("\tVendor:           %s\n", devices[i].getInfo<CL_DEVICE_VENDOR>().c_str() );
        printf("\tDevice Version:   %s\n", devices[i].getInfo<CL_DEVICE_VERSION>().c_str() );
        printf("\tOpenCL C Version: %s\n", devices[i].getInfo<CL_DEVICE_OPENCL_C_VERSION>().c_str() );
        printf("\tDriver Version:   %s\n", devices[i].getInfo<CL_DRIVER_VERSION>().c_str() );

        // Use the query for the device numeric version as a test for
        // OpenCL 3.0 support.  If this query fails then this probably
        // isn't an OpenCL 3.0 device:
        cl_int test = CL_SUCCESS;
        cl_version deviceVersion =
            devices[i].getInfo<CL_DEVICE_NUMERIC_VERSION>(&test);
        if( test == CL_SUCCESS )
        {
            printf("\tDevice Numeric Version:   %u.%u.%u\n",
                CL_VERSION_MAJOR(deviceVersion),
                CL_VERSION_MINOR(deviceVersion),
                CL_VERSION_PATCH(deviceVersion));
        }
        else
        {
            printf("Query for CL_DEVICE_NUMERIC_VERSION failed.\n");
            continue;
        }

        std::vector<cl_name_version> clcVersions =
            devices[i].getInfo<CL_DEVICE_OPENCL_C_ALL_VERSIONS>();
        for( auto& clcVersion : clcVersions )
        {
            printf("\t\tOpenCL C (version): %s (%u.%u.%u)\n",
                clcVersion.name,
                CL_VERSION_MAJOR(clcVersion.version),
                CL_VERSION_MINOR(clcVersion.version),
                CL_VERSION_PATCH(clcVersion.version));
        }

        std::vector<cl_name_version> clcFeatures =
            devices[i].getInfo<CL_DEVICE_OPENCL_C_FEATURES>();
        for( auto& clcFeature : clcFeatures )
        {
            printf("\t\tOpenCL C Feature (version): %s (%u.%u.%u)\n",
                clcFeature.name,
                CL_VERSION_MAJOR(clcFeature.version),
                CL_VERSION_MINOR(clcFeature.version),
                CL_VERSION_PATCH(clcFeature.version));
        }

        std::vector<cl_name_version> extensions =
            devices[i].getInfo<CL_DEVICE_EXTENSIONS_WITH_VERSION>();
        for( auto& extension : extensions )
        {
            printf("\t\tExtension (version): %s (%u.%u.%u)\n",
                extension.name,
                CL_VERSION_MAJOR(extension.version),
                CL_VERSION_MINOR(extension.version),
                CL_VERSION_PATCH(extension.version) );
        }

        std::vector<cl_name_version> ils =
            devices[i].getInfo<CL_DEVICE_ILS_WITH_VERSION>();
        for( auto& il : ils )
        {
            printf("\t\tIL (version): %s (%u.%u.%u)\n",
                il.name,
                CL_VERSION_MAJOR(il.version),
                CL_VERSION_MINOR(il.version),
                CL_VERSION_PATCH(il.version) );
        }

        std::vector<cl_name_version> builtInKernels =
            devices[i].getInfo<CL_DEVICE_BUILT_IN_KERNELS_WITH_VERSION>();
        for( auto& builtInFunction : builtInKernels )
        {
            printf("\t\tBuilt-in Function (version): %s (%u.%u.%u)\n",
                builtInFunction.name,
                CL_VERSION_MAJOR(builtInFunction.version),
                CL_VERSION_MINOR(builtInFunction.version),
                CL_VERSION_PATCH(builtInFunction.version) );
        }

        cl_device_atomic_capabilities   deviceAtomicMemoryCapabilities =
            devices[i].getInfo<CL_DEVICE_ATOMIC_MEMORY_CAPABILITIES>();
        PrintDeviceAtomicCapabilities("\tAtomic Memory Capabilities", deviceAtomicMemoryCapabilities);

        cl_device_atomic_capabilities   deviceAtomicFenceCapabilities =
            devices[i].getInfo<CL_DEVICE_ATOMIC_MEMORY_CAPABILITIES>();
        PrintDeviceAtomicCapabilities("\tAtomic Fence Capabilities", deviceAtomicFenceCapabilities);

        size_t  devicePreferredWorkGroupSizeMultiple =
            devices[i].getInfo<CL_DEVICE_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>();
        cl_bool deviceNonUniformWorkGroupSupport =
            devices[i].getInfo<CL_DEVICE_NON_UNIFORM_WORK_GROUP_SUPPORT>();
        cl_bool deviceWorkGroupCollectiveFunctionsSupport =
            devices[i].getInfo<CL_DEVICE_WORK_GROUP_COLLECTIVE_FUNCTIONS_SUPPORT>();
        cl_bool deviceGenericAddressSpaceSupport =
            devices[i].getInfo<CL_DEVICE_GENERIC_ADDRESS_SPACE_SUPPORT>();
// This is an older query that should eventually be removed.
// It was replaced by CL_DEVICE_DEVICE_ENQUEUE_CAPABILITIES.
#if defined(CL_DEVICE_DEVICE_ENQUEUE_SUPPORT)
        cl_bool deviceDeviceEnqueueSupport =
            devices[i].getInfo<CL_DEVICE_DEVICE_ENQUEUE_SUPPORT>();
#endif
// This is the newer query.
// The ifdefs are only needed until this enum is in the official headers.
#if defined(CL_DEVICE_DEVICE_ENQUEUE_CAPABILITIES)
        cl_device_device_enqueue_capabilities deviceDeviceEnqueueCapabilities =
            devices[i].getInfo<CL_DEVICE_DEVICE_ENQUEUE_CAPABILITIES>();
#endif
        cl_bool devicePipeSupport =
            devices[i].getInfo<CL_DEVICE_PIPE_SUPPORT>();

        printf("\tPreferred Work Group Size Multiple:       %u\n",
            (cl_uint)devicePreferredWorkGroupSizeMultiple);
        printf("\tNon-Uniform Work Group Support:           %s\n",
            deviceNonUniformWorkGroupSupport ? "CL_TRUE" : "CL_FALSE" );
        printf("\tWork Group Collective Functions Support:  %s\n",
            deviceWorkGroupCollectiveFunctionsSupport ? "CL_TRUE" : "CL_FALSE" );
        printf("\tGeneric Address Space Support:            %s\n",
            deviceGenericAddressSpaceSupport ? "CL_TRUE" : "CL_FALSE" );
#if defined(CL_DEVICE_DEVICE_ENQUEUE_SUPPORT)
        printf("\tDevice Enqueue Support:                   %s\n",
            deviceDeviceEnqueueSupport ? "CL_TRUE" : "CL_FALSE" );
#endif
#if defined(CL_DEVICE_DEVICE_ENQUEUE_CAPABILITIES)
        PrintDeviceDeviceEnqueueCapabilities("\tDevice Enqueue Capabilities",
            deviceDeviceEnqueueCapabilities);
#endif
        printf("\tPipe Support:                             %s\n",
            devicePipeSupport ? "CL_TRUE" : "CL_FALSE" );
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
                "Usage: newqueriespp [options]\n"
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

#else

#pragma message("newqueriespp: OpenCL 3.0 not found.  Please update your OpenCL headers.")

int main()
{
    printf("newqueriespp: OpenCL 3.0 not found.  Please update your OpenCL headers.\n");
    return 0;
};

#endif
