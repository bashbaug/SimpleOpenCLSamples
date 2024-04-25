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

#include <CL/cl.h>
#include <CL/cl_ext.h>

#if defined(CL_VERSION_3_0)

static cl_int AllocateAndGetPlatformInfoString(
    cl_platform_id platformId,
    cl_platform_info param_name,
    char*& param_value )
{
    cl_int  errorCode = CL_SUCCESS;
    size_t  size = 0;

    if( errorCode == CL_SUCCESS )
    {
        if( param_value != NULL )
        {
            delete [] param_value;
            param_value = NULL;
        }
    }

    if( errorCode == CL_SUCCESS )
    {
        errorCode = clGetPlatformInfo(
            platformId,
            param_name,
            0,
            NULL,
            &size );
    }

    if( errorCode == CL_SUCCESS )
    {
        if( size != 0 )
        {
            param_value = new char[ size ];
            if( param_value == NULL )
            {
                errorCode = CL_OUT_OF_HOST_MEMORY;
            }
        }
    }

    if( errorCode == CL_SUCCESS )
    {
        errorCode = clGetPlatformInfo(
            platformId,
            param_name,
            size,
            param_value,
            NULL );
    }

    if( errorCode != CL_SUCCESS )
    {
        delete [] param_value;
        param_value = NULL;
    }

    return errorCode;
}

static void PrintPlatformInfoSummary(
    cl_platform_id platformId )
{
    cl_int  errorCode = CL_SUCCESS;

    char*           platformName = NULL;
    char*           platformVendor = NULL;
    char*           platformVersion = NULL;

    errorCode |= AllocateAndGetPlatformInfoString(
        platformId,
        CL_PLATFORM_NAME,
        platformName );
    errorCode |= AllocateAndGetPlatformInfoString(
        platformId,
        CL_PLATFORM_VENDOR,
        platformVendor );
    errorCode |= AllocateAndGetPlatformInfoString(
        platformId,
        CL_PLATFORM_VERSION,
        platformVersion );

    printf("\tName:           %s\n", platformName );
    printf("\tVendor:         %s\n", platformVendor );
    printf("\tDriver Version: %s\n", platformVersion );

    // Use the query for the platform numeric version as a test for
    // OpenCL 3.0 support.  If this query fails then this probably
    // isn't an OpenCL 3.0 platform:
    {
        cl_version version = 0;
        cl_int test = clGetPlatformInfo(
            platformId,
            CL_PLATFORM_NUMERIC_VERSION,
            sizeof(version),
            &version,
            NULL );
        if( test == CL_SUCCESS )
        {
            printf("\tPlatform Numeric Version: %u.%u.%u\n",
                CL_VERSION_MAJOR(version),
                CL_VERSION_MINOR(version),
                CL_VERSION_PATCH(version));
        }
        else
        {
            printf("Query for CL_PLATFORM_NUMERIC_VERSION failed.\n");
            goto Cleanup;
        }
    }
    {
        size_t  sizeInBytes = 0;
        clGetPlatformInfo(
            platformId,
            CL_PLATFORM_EXTENSIONS_WITH_VERSION,
            0,
            NULL,
            &sizeInBytes );

        size_t  numExtensions = sizeInBytes / sizeof(cl_name_version);
        std::vector<cl_name_version>  extensions{ numExtensions };
        clGetPlatformInfo(
            platformId,
            CL_PLATFORM_EXTENSIONS_WITH_VERSION,
            sizeInBytes,
            extensions.data(),
            NULL );

        for( auto& extension : extensions )
        {
            printf("\t\tExtension (version): %s (%u.%u.%u)\n",
                extension.name,
                CL_VERSION_MAJOR(extension.version),
                CL_VERSION_MINOR(extension.version),
                CL_VERSION_PATCH(extension.version) );
        }
    }

Cleanup:
    delete [] platformName;
    delete [] platformVendor;
    delete [] platformVersion;

    platformName = NULL;
    platformVendor = NULL;
    platformVersion = NULL;
}

static cl_int AllocateAndGetDeviceInfoString(
    cl_device_id    device,
    cl_device_info  param_name,
    char*&          param_value )
{
    cl_int  errorCode = CL_SUCCESS;
    size_t  size = 0;

    if( errorCode == CL_SUCCESS )
    {
        if( param_value != NULL )
        {
            delete [] param_value;
            param_value = NULL;
        }
    }

    if( errorCode == CL_SUCCESS )
    {
        errorCode = clGetDeviceInfo(
            device,
            param_name,
            0,
            NULL,
            &size );
    }

    if( errorCode == CL_SUCCESS )
    {
        if( size != 0 )
        {
            param_value = new char[ size ];
            if( param_value == NULL )
            {
                errorCode = CL_OUT_OF_HOST_MEMORY;
            }
        }
    }

    if( errorCode == CL_SUCCESS )
    {
        errorCode = clGetDeviceInfo(
            device,
            param_name,
            size,
            param_value,
            NULL );
    }

    if( errorCode != CL_SUCCESS )
    {
        delete [] param_value;
        param_value = NULL;
    }

    return errorCode;
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
    cl_device_id* devices,
    size_t numDevices )
{
    cl_device_type  deviceType;
    char*           deviceName = NULL;
    char*           deviceVendor = NULL;
    char*           deviceVersion = NULL;
    char*           deviceOpenCLCVersion = NULL;
    char*           driverVersion = NULL;

    size_t  i = 0;
    for( i = 0; i < numDevices; i++ )
    {
        cl_int  errorCode = CL_SUCCESS;

        errorCode |= clGetDeviceInfo(
            devices[i],
            CL_DEVICE_TYPE,
            sizeof( deviceType ),
            &deviceType,
            NULL );
        errorCode |= AllocateAndGetDeviceInfoString(
            devices[i],
            CL_DEVICE_NAME,
            deviceName );
        errorCode |= AllocateAndGetDeviceInfoString(
            devices[i],
            CL_DEVICE_VENDOR,
            deviceVendor );
        errorCode |= AllocateAndGetDeviceInfoString(
            devices[i],
            CL_DEVICE_VERSION,
            deviceVersion );
        errorCode |= AllocateAndGetDeviceInfoString(
            devices[i],
            CL_DEVICE_OPENCL_C_VERSION,
            deviceOpenCLCVersion );
        errorCode |= AllocateAndGetDeviceInfoString(
            devices[i],
            CL_DRIVER_VERSION,
            driverVersion );

        if( errorCode == CL_SUCCESS )
        {
            printf("Device[%d]:\n", (int)i );

            PrintDeviceType("\tType:           ", deviceType);

            printf("\tName:             %s\n", deviceName );
            printf("\tVendor:           %s\n", deviceVendor );
            printf("\tDevice Version:   %s\n", deviceVersion );
            printf("\tOpenCL C Version: %s\n", deviceOpenCLCVersion );
            printf("\tDriver Version:   %s\n", driverVersion );

            // Use the query for the device numeric version as a test for
            // OpenCL 3.0 support.  If this query fails then this probably
            // isn't an OpenCL 3.0 device:
            {
                cl_version version = 0;
                cl_int test = clGetDeviceInfo(
                    devices[i],
                    CL_DEVICE_NUMERIC_VERSION,
                    sizeof(version),
                    &version,
                    NULL );
                if( test == CL_SUCCESS )
                {
                    printf("\tDevice Numeric Version:   %u.%u.%u\n",
                        CL_VERSION_MAJOR(version),
                        CL_VERSION_MINOR(version),
                        CL_VERSION_PATCH(version));
                }
                else
                {
                    printf("Query for CL_DEVICE_NUMERIC_VERSION failed.\n");
                    goto Cleanup;
                }
            }
            {
                size_t  sizeInBytes = 0;
                clGetDeviceInfo(
                    devices[i],
                    CL_DEVICE_OPENCL_C_ALL_VERSIONS,
                    0,
                    NULL,
                    &sizeInBytes );

                size_t  numCLCVersions = sizeInBytes / sizeof(cl_name_version);
                std::vector<cl_name_version>  clcVersions{ numCLCVersions };
                clGetDeviceInfo(
                    devices[i],
                    CL_DEVICE_OPENCL_C_ALL_VERSIONS,
                    sizeInBytes,
                    clcVersions.data(),
                    NULL );

                for( auto& clcVersion : clcVersions )
                {
                    printf("\t\tOpenCL C (version): %s (%u.%u.%u)\n",
                        clcVersion.name,
                        CL_VERSION_MAJOR(clcVersion.version),
                        CL_VERSION_MINOR(clcVersion.version),
                        CL_VERSION_PATCH(clcVersion.version) );
                }
            }
            {
                size_t  sizeInBytes = 0;
                clGetDeviceInfo(
                    devices[i],
                    CL_DEVICE_OPENCL_C_FEATURES,
                    0,
                    NULL,
                    &sizeInBytes );

                size_t  numCLCFeatures = sizeInBytes / sizeof(cl_name_version);
                std::vector<cl_name_version>  clcFeatures{ numCLCFeatures };
                clGetDeviceInfo(
                    devices[i],
                    CL_DEVICE_OPENCL_C_FEATURES,
                    sizeInBytes,
                    clcFeatures.data(),
                    NULL );

                for( auto& clcFeature : clcFeatures )
                {
                    printf("\t\tOpenCL C Feature (version): %s (%u.%u.%u)\n",
                        clcFeature.name,
                        CL_VERSION_MAJOR(clcFeature.version),
                        CL_VERSION_MINOR(clcFeature.version),
                        CL_VERSION_PATCH(clcFeature.version) );
                }
            }
            {
                size_t  sizeInBytes = 0;
                clGetDeviceInfo(
                    devices[i],
                    CL_DEVICE_EXTENSIONS_WITH_VERSION,
                    0,
                    NULL,
                    &sizeInBytes );

                size_t  numExtensions = sizeInBytes / sizeof(cl_name_version);
                std::vector<cl_name_version>  extensions{ numExtensions };
                clGetDeviceInfo(
                    devices[i],
                    CL_DEVICE_EXTENSIONS_WITH_VERSION,
                    sizeInBytes,
                    extensions.data(),
                    NULL );

                for( auto& extension : extensions )
                {
                    printf("\t\tExtension (version): %s (%u.%u.%u)\n",
                        extension.name,
                        CL_VERSION_MAJOR(extension.version),
                        CL_VERSION_MINOR(extension.version),
                        CL_VERSION_PATCH(extension.version) );
                }
            }
            {
                size_t  sizeInBytes = 0;
                clGetDeviceInfo(
                    devices[i],
                    CL_DEVICE_ILS_WITH_VERSION,
                    0,
                    NULL,
                    &sizeInBytes );

                size_t  numILs = sizeInBytes / sizeof(cl_name_version);
                std::vector<cl_name_version> ils{ numILs };
                clGetDeviceInfo(
                    devices[i],
                    CL_DEVICE_ILS_WITH_VERSION,
                    sizeInBytes,
                    ils.data(),
                    NULL );

                for( auto& il : ils )
                {
                    printf("\t\tIL (version): %s (%u.%u.%u)\n",
                        il.name,
                        CL_VERSION_MAJOR(il.version),
                        CL_VERSION_MINOR(il.version),
                        CL_VERSION_PATCH(il.version) );
                }
            }
            {
                size_t  sizeInBytes = 0;
                clGetDeviceInfo(
                    devices[i],
                    CL_DEVICE_BUILT_IN_KERNELS_WITH_VERSION,
                    0,
                    NULL,
                    &sizeInBytes );

                size_t  numBuiltInKernels = sizeInBytes / sizeof(cl_name_version);
                std::vector<cl_name_version> builtInKernels{ numBuiltInKernels };
                clGetDeviceInfo(
                    devices[i],
                    CL_DEVICE_BUILT_IN_KERNELS_WITH_VERSION,
                    sizeInBytes,
                    builtInKernels.data(),
                    NULL );

                for( auto& builtInFunction : builtInKernels )
                {
                    printf("\t\tBuilt-in Function (version): %s (%u.%u.%u)\n",
                        builtInFunction.name,
                        CL_VERSION_MAJOR(builtInFunction.version),
                        CL_VERSION_MINOR(builtInFunction.version),
                        CL_VERSION_PATCH(builtInFunction.version) );
                }
            }

            cl_device_atomic_capabilities   deviceAtomicMemoryCapabilities = 0;
            clGetDeviceInfo(
                devices[i],
                CL_DEVICE_ATOMIC_MEMORY_CAPABILITIES,
                sizeof(deviceAtomicMemoryCapabilities),
                &deviceAtomicMemoryCapabilities,
                NULL );
            PrintDeviceAtomicCapabilities(
                "\tAtomic Memory Capabilities",
                deviceAtomicMemoryCapabilities);

            cl_device_atomic_capabilities   deviceAtomicFenceCapabilities = 0;
            clGetDeviceInfo(
                devices[i],
                CL_DEVICE_ATOMIC_FENCE_CAPABILITIES,
                sizeof(deviceAtomicFenceCapabilities),
                &deviceAtomicFenceCapabilities,
                NULL );
            PrintDeviceAtomicCapabilities(
                "\tAtomic Fence Capabilities",
                deviceAtomicFenceCapabilities);

            size_t  devicePreferredWorkGroupSizeMultiple = 0;
            clGetDeviceInfo(
                devices[i],
                CL_DEVICE_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
                sizeof(devicePreferredWorkGroupSizeMultiple),
                &devicePreferredWorkGroupSizeMultiple,
                NULL );

            cl_bool deviceNonUniformWorkGroupSupport = CL_FALSE;
            clGetDeviceInfo(
                devices[i],
                CL_DEVICE_NON_UNIFORM_WORK_GROUP_SUPPORT,
                sizeof(deviceNonUniformWorkGroupSupport),
                &deviceNonUniformWorkGroupSupport,
                NULL );

            cl_bool deviceWorkGroupCollectiveFunctionsSupport = CL_FALSE;
            clGetDeviceInfo(
                devices[i],
                CL_DEVICE_WORK_GROUP_COLLECTIVE_FUNCTIONS_SUPPORT,
                sizeof(deviceWorkGroupCollectiveFunctionsSupport),
                &deviceWorkGroupCollectiveFunctionsSupport,
                NULL );

            cl_bool deviceGenericAddressSpaceSupport = CL_FALSE;
            clGetDeviceInfo(
                devices[i],
                CL_DEVICE_GENERIC_ADDRESS_SPACE_SUPPORT,
                sizeof(deviceGenericAddressSpaceSupport),
                &deviceGenericAddressSpaceSupport,
                NULL );

// This is an older query that should eventually be removed.
// It was replaced by CL_DEVICE_DEVICE_ENQUEUE_CAPABILITIES.
#if defined(CL_DEVICE_DEVICE_ENQUEUE_SUPPORT)
            cl_bool deviceDeviceEnqueueSupport = CL_FALSE;
            clGetDeviceInfo(
                devices[i],
                CL_DEVICE_DEVICE_ENQUEUE_SUPPORT,
                sizeof(deviceDeviceEnqueueSupport),
                &deviceDeviceEnqueueSupport,
                NULL );
#endif

// This is the newer query.
// The ifdefs are only needed until this enum is in the official headers.
#if defined(CL_DEVICE_DEVICE_ENQUEUE_CAPABILITIES)
            cl_device_device_enqueue_capabilities deviceDeviceEnqueueCapabilities = 0;
            clGetDeviceInfo(
                devices[i],
                CL_DEVICE_DEVICE_ENQUEUE_CAPABILITIES,
                sizeof(deviceDeviceEnqueueCapabilities),
                &deviceDeviceEnqueueCapabilities,
                NULL );
#endif

            cl_bool devicePipeSupport = CL_FALSE;
            clGetDeviceInfo(
                devices[i],
                CL_DEVICE_PIPE_SUPPORT,
                sizeof(devicePipeSupport),
                &devicePipeSupport,
                NULL );

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
        else
        {
            fprintf(stderr, "Error getting device info for device %d.\n", (int)i );
        }

Cleanup:
        delete [] deviceName;
        delete [] deviceVendor;
        delete [] deviceVersion;
        delete [] driverVersion;

        deviceName = NULL;
        deviceVendor = NULL;
        deviceVersion = NULL;
        driverVersion = NULL;
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
                "Usage: newqueries [options]\n"
                "%s", op.help().c_str());
            return -1;
        }
    }

    cl_uint numPlatforms = 0;
    clGetPlatformIDs( 0, NULL, &numPlatforms );
    printf( "Enumerated %u platforms.\n\n", numPlatforms );

    std::vector<cl_platform_id> platforms;
    platforms.resize( numPlatforms );
    clGetPlatformIDs( numPlatforms, platforms.data(), NULL );

    for( auto& platform : platforms )
    {
        printf( "Platform:\n" );
        PrintPlatformInfoSummary( platform );

        cl_uint numDevices = 0;
        clGetDeviceIDs( platform, CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices );

        std::vector<cl_device_id> devices;
        devices.resize( numDevices );
        clGetDeviceIDs( platform, CL_DEVICE_TYPE_ALL, numDevices, devices.data(), NULL );

        PrintDeviceInfoSummary( devices.data(), numDevices );
        printf( "\n" );
    }

    printf( "Done.\n" );

    return 0;
}

#else

#pragma message("newqueries: OpenCL 3.0 not found.  Please update your OpenCL headers.")

int main()
{
    printf("newqueries: OpenCL 3.0 not found.  Please update your OpenCL headers.\n");
    return 0;
};

#endif
