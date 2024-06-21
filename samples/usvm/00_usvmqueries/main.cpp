/*
// Copyright (c) 2023 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <popl/popl.hpp>

#include <CL/opencl.hpp>

#define CASE_TO_STRING(_e) case _e: return #_e;

static const char* svm_type_to_string(cl_svm_type_exp type)
{
    switch (type) {
    CASE_TO_STRING(CL_SVM_TYPE_COARSE_GRAIN_BUFFER_EXP);
    CASE_TO_STRING(CL_SVM_TYPE_FINE_GRAIN_BUFFER_EXP);
    CASE_TO_STRING(CL_SVM_TYPE_FINE_GRAIN_BUFFER_WITH_ATOMICS_EXP);
    CASE_TO_STRING(CL_SVM_TYPE_FINE_GRAIN_SYSTEM_EXP);
    CASE_TO_STRING(CL_SVM_TYPE_HOST_EXP);
    CASE_TO_STRING(CL_SVM_TYPE_DEVICE_EXP);
    CASE_TO_STRING(CL_SVM_TYPE_SHARED_EXP);
    default: return "Unknown cl_svm_type_exp";
    }
}

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

void PrintUSVMType(
    const char* label,
    cl_svm_type_exp type)
{
    printf("\t\t%s: %s\n",
        label,
        svm_type_to_string(type));
}

void PrintUSVMCaps(
    const char* label,
    cl_svm_type_capabilities_exp svmcaps )
{
    printf("\t\t%s: %s%s%s%s%s%s%s%s%s%s%s\n",
        label,
        ( svmcaps & CL_SVM_CAPABILITY_SINGLE_ADDRESS_SPACE_EXP     ) ? "\n\t\t\tCL_SVM_CAPABILITY_SINGLE_ADDRESS_SPACE_EXP"      : "",
        ( svmcaps & CL_SVM_CAPABILITY_SYSTEM_ALLOCATOR_EXP         ) ? "\n\t\t\tCL_SVM_CAPABILITY_SYSTEM_ALLOCATOR_EXP"          : "",
        ( svmcaps & CL_SVM_CAPABILITY_DEVICE_OWNED_EXP             ) ? "\n\t\t\tCL_SVM_CAPABILITY_DEVICE_OWNED_EXP"              : "",
        ( svmcaps & CL_SVM_CAPABILITY_HOST_OWNED_EXP               ) ? "\n\t\t\tCL_SVM_CAPABILITY_HOST_OWNED_EXP"                : "",
        ( svmcaps & CL_SVM_CAPABILITY_HOST_ACCESSIBLE_EXP          ) ? "\n\t\t\tCL_SVM_CAPABILITY_HOST_ACCESSIBLE_EXP"           : "",
        ( svmcaps & CL_SVM_CAPABILITY_HOST_ACCESSIBLE_WITH_MAP_EXP ) ? "\n\t\t\tCL_SVM_CAPABILITY_HOST_ACCESSIBLE_WITH_MAP_EXP"  : "",
        ( svmcaps & CL_SVM_CAPABILITY_DEVICE_ACCESS_EXP            ) ? "\n\t\t\tCL_SVM_CAPABILITY_DEVICE_ACCESS_EXP"             : "",
        ( svmcaps & CL_SVM_CAPABILITY_DEVICE_ATOMIC_ACCESS_EXP     ) ? "\n\t\t\tCL_SVM_CAPABILITY_DEVICE_ATOMIC_ACCESS_EXP"      : "",
        ( svmcaps & CL_SVM_CAPABILITY_CONCURRENT_ACCESS_EXP        ) ? "\n\t\t\tCL_SVM_CAPABILITY_CONCURRENT_ACCESS_EXP"         : "",
        ( svmcaps & CL_SVM_CAPABILITY_CONCURRENT_ATOMIC_ACCESS_EXP ) ? "\n\t\t\tCL_SVM_CAPABILITY_CONCURRENT_ATOMIC_ACCESS_EXP"  : "",
        ( svmcaps & CL_SVM_CAPABILITY_INDIRECT_ACCESS_EXP          ) ? "\n\t\t\tCL_SVM_CAPABILITY_INDIRECT_ACCESS_EXP"           : "" );

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

            std::vector<cl_device_svm_type_capabilities_exp> usmTypes =
                devices[d].getInfo<CL_DEVICE_SVM_TYPE_CAPABILITIES_EXP>();

            for (size_t t = 0; t < usmTypes.size(); t++)
            {
                printf("\tUSM Type[%zu]:\n", t);
                PrintUSVMType( "type", usmTypes[t].type );
                PrintUSVMCaps("capabilities", usmTypes[t].capabilities);
            }

            cl::Context context{devices[d]};

            cl_svm_type_exp type = 0;
            clGetSuggestedSVMTypeEXP(
                context(),
                devices[d](),
                CL_SVM_CAPABILITY_DEVICE_ACCESS_EXP,
                &type);
            printf("\tSuggested SVM type with device handle is: %s\n",
                type == 0 ? "(none)" : svm_type_to_string(type));

            type = 0;
            clGetSuggestedSVMTypeEXP(
                context(),
                nullptr,
                CL_SVM_CAPABILITY_DEVICE_ACCESS_EXP,
                &type);
            printf("\tSuggested SVM type NULL device handle is: %s\n",
                type == 0 ? "(none)" : svm_type_to_string(type));

            printf( "\n" );
        }
    }

    printf("Cleaning up...\n");

    return 0;
}