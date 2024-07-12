/*
// Copyright (c) 2023 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <popl/popl.hpp>

#include <CL/opencl.hpp>

#include <cinttypes>

void PrintUSVMCaps(
    const char* label,
    cl_svm_capabilities_exp svmcaps )
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

            std::vector<cl_device_svm_type_capabilities_exp> usmTypes =
                devices[d].getInfo<CL_DEVICE_SVM_TYPE_CAPABILITIES_EXP>();

            for (size_t t = 0; t < usmTypes.size(); t++)
            {
                printf("\tUSM Type[%zu]:\n", t);
                PrintUSVMCaps("capabilities", usmTypes[t].capabilities);
                if (usmTypes[t].toggleable_capabilities) {
                    PrintUSVMCaps("toggleable_capabilities", usmTypes[t].toggleable_capabilities);
                }
            }

            cl::Context context{devices[d]};

            cl_svm_capabilities_exp caps = 0;
            clGetSuggestedSVMCapabilitiesEXP(
                context(),
                1,
                &devices[d](),
                CL_SVM_CAPABILITY_DEVICE_ACCESS_EXP,
                &caps);
            printf("\tSuggested SVM caps with device handle is: %016" PRIx64 "\n",
                caps);

            caps = 0;
            clGetSuggestedSVMCapabilitiesEXP(
                context(),
                0,
                nullptr,
                CL_SVM_CAPABILITY_DEVICE_ACCESS_EXP,
                &caps);
            printf("\tSuggested SVM type NULL device handle is: %016" PRIx64 "\n",
                caps);

            printf( "\n" );
        }
    }

    printf("Cleaning up...\n");

    return 0;
}