/*
// Copyright (c) 2023 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <popl/popl.hpp>

#include <CL/opencl.hpp>

void PrintDeviceSVMCaps(
    const char* label,
    cl_device_svm_capabilities svmcaps )
{
    printf("\t%s: %s%s%s%s%s%s%s\n",
        label,
        ( svmcaps & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER) ? "\n\t\tCL_DEVICE_SVM_COARSE_GRAIN_BUFFER" : "",
        ( svmcaps & CL_DEVICE_SVM_FINE_GRAIN_BUFFER  ) ? "\n\t\tCL_DEVICE_SVM_FINE_GRAIN_BUFFER"   : "",
        ( svmcaps & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM  ) ? "\n\t\tCL_DEVICE_SVM_FINE_GRAIN_SYSTEM"   : "",
        ( svmcaps & CL_DEVICE_SVM_ATOMICS            ) ? "\n\t\tCL_DEVICE_SVM_ATOMICS"             : "",
        ( svmcaps & CL_DEVICE_SVM_DEVICE_ALLOC_EXP   ) ? "\n\t\tCL_DEVICE_SVM_DEVICE_ALLOC_EXP"    : "",
        ( svmcaps & CL_DEVICE_SVM_HOST_ALLOC_EXP     ) ? "\n\t\tCL_DEVICE_SVM_HOST_ALLOC_EXP"      : "",
        ( svmcaps & CL_DEVICE_SVM_SHARED_ALLOC_EXP   ) ? "\n\t\tCL_DEVICE_SVM_SHARED_ALLOC_EXP"    : "" );
}

void PrintSVMMemFlags(
    const char* label,
    cl_svm_mem_flags flags)
{
    if (flags == 0) {
        printf("\t\t%s: (none)\n", label);
    } else {
        printf("\t\t%s: %s%s%s%s%s\n",
            label,
            (flags & CL_MEM_SVM_FINE_GRAIN_BUFFER) ? "\n\t\t\tCL_MEM_SVM_FINE_GRAIN_BUFFER" : "",
            (flags & CL_MEM_SVM_ATOMICS          ) ? "\n\t\t\tCL_MEM_SVM_ATOMICS" : "",
            (flags & CL_MEM_SVM_DEVICE_EXP       ) ? "\n\t\t\tCL_MEM_SVM_DEVICE_EXP" : "",
            (flags & CL_MEM_SVM_HOST_EXP         ) ? "\n\t\t\tCL_MEM_SVM_HOST_EXP" : "",
            (flags & CL_MEM_SVM_SHARED_EXP       ) ? "\n\t\t\tCL_MEM_SVM_SHARED_EXP" : "");
    }
}

void PrintUSVMCaps(
    const char* label,
    cl_device_unified_svm_capabilities_exp usvmcaps )
{
    printf("\t\t%s: %s%s%s%s%s%s%s%s%s%s%s\n",
        label,
        ( usvmcaps & CL_UNIFIED_SVM_SINGLE_ADDRESS_SPACE_EXP     ) ? "\n\t\t\tCL_UNIFIED_SVM_SINGLE_ADDRESS_SPACE_EXP"      : "",
        ( usvmcaps & CL_UNIFIED_SVM_SYSTEM_ALLOCATOR_EXP         ) ? "\n\t\t\tCL_UNIFIED_SVM_SYSTEM_ALLOCATOR_EXP"          : "",
        ( usvmcaps & CL_UNIFIED_SVM_DEVICE_OWNED_EXP             ) ? "\n\t\t\tCL_UNIFIED_SVM_DEVICE_OWNED_EXP"              : "",
        ( usvmcaps & CL_UNIFIED_SVM_HOST_OWNED_EXP               ) ? "\n\t\t\tCL_UNIFIED_SVM_HOST_OWNED_EXP"                : "",
        ( usvmcaps & CL_UNIFIED_SVM_HOST_ACCESSIBLE_EXP          ) ? "\n\t\t\tCL_UNIFIED_SVM_HOST_ACCESSIBLE_EXP"           : "",
        ( usvmcaps & CL_UNIFIED_SVM_HOST_ACCESSIBLE_WITH_MAP_EXP ) ? "\n\t\t\tCL_UNIFIED_SVM_HOST_ACCESSIBLE_WITH_MAP_EXP"  : "",
        ( usvmcaps & CL_UNIFIED_SVM_DEVICE_ACCESS_EXP            ) ? "\n\t\t\tCL_UNIFIED_SVM_DEVICE_ACCESS_EXP"             : "",
        ( usvmcaps & CL_UNIFIED_SVM_DEVICE_ATOMIC_ACCESS_EXP     ) ? "\n\t\t\tCL_UNIFIED_SVM_DEVICE_ATOMIC_ACCESS_EXP"      : "",
        ( usvmcaps & CL_UNIFIED_SVM_CONCURRENT_ACCESS_EXP        ) ? "\n\t\t\tCL_UNIFIED_SVM_CONCURRENT_ACCESS_EXP"         : "",
        ( usvmcaps & CL_UNIFIED_SVM_CONCURRENT_ATOMIC_ACCESS_EXP ) ? "\n\t\t\tCL_UNIFIED_SVM_CONCURRENT_ATOMIC_ACCESS_EXP"  : "",
        ( usvmcaps & CL_UNIFIED_SVM_INDIRECT_ACCESS_EXP          ) ? "\n\t\t\tCL_UNIFIED_SVM_INDIRECT_ACCESS_EXP"           : "" );

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
                "Usage: usvmqueries [options]\n"
                "%s", op.help().c_str());
            return -1;
        }
    }

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    for( size_t i = 0; i < platforms.size(); i++ )
    {
        printf( "Platform[%zu]: %s\n",
            i,
            platforms[i].getInfo<CL_PLATFORM_NAME>().c_str());

        std::vector<cl::Device> devices;
        platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &devices);

        for( size_t d = 0; d < devices.size(); d++ )
        {
            printf("\tDevice[%zu]: %s\n",
                d,
                devices[d].getInfo<CL_DEVICE_NAME>().c_str());

            cl_device_svm_capabilities svmcaps = 0;
            clGetDeviceInfo(
                devices[d](),
                CL_DEVICE_SVM_CAPABILITIES,
                sizeof(svmcaps),
                &svmcaps,
                nullptr );
            PrintDeviceSVMCaps( "CL_DEVICE_SVM_CAPABILITIES", svmcaps );

            std::vector<cl_device_unified_svm_type_exp> usmTypes =
                devices[d].getInfo<CL_DEVICE_UNIFIED_SVM_TYPES_EXP>();

            for (size_t t = 0; t < usmTypes.size(); t++)
            {
                printf("\tUSM Type[%zu]:\n", t);
                PrintSVMMemFlags( "mem_flags", usmTypes[t].mem_flags );
                PrintUSVMCaps("capabilities", usmTypes[t].capabilities);
            }

            printf( "\n" );
        }
    }

    printf("Cleaning up...\n");

    return 0;
}