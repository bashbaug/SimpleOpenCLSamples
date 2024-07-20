/*
// Copyright (c) 2023 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <popl/popl.hpp>

#include <CL/opencl.hpp>

#include <cinttypes>

static inline bool has_all_svm_caps(
    cl_svm_capabilities_exp caps,
    cl_svm_capabilities_exp check)
{
    return (caps & check) == check;
}

static const char* get_svm_name(cl_svm_capabilities_exp svmcaps)
{
    if (has_all_svm_caps(svmcaps, CL_SVM_TYPE_MACRO_SYSTEM_EXP)) {
        return "SYSTEM";
    }
    if (has_all_svm_caps(svmcaps, CL_SVM_TYPE_MACRO_FINE_GRAIN_BUFFER_EXP)) {
        return "FINE_GRAIN_BUFFER";
    }
    if (has_all_svm_caps(svmcaps, CL_SVM_TYPE_MACRO_COARSE_GRAIN_BUFFER_EXP)) {
        return "COARSE_GRAIN_BUFFER";
    }
    if (has_all_svm_caps(svmcaps, CL_SVM_TYPE_MACRO_HOST_EXP)) {
        return "HOST";
    }
    if (has_all_svm_caps(svmcaps, CL_SVM_TYPE_MACRO_DEVICE_EXP)) {
        return "DEVICE";
    }
    if (has_all_svm_caps(svmcaps, CL_SVM_TYPE_MACRO_SINGLE_DEVICE_SHARED_EXP)) {
        return "SINGLE_DEVICE_SHARED";
    }
    return "*** UNKNOWN! ***";
}

void PrintUSVMCaps(
    const char* prefix,
    cl_svm_capabilities_exp svmcaps )
{
    printf("%s%s%s%s%s%s%s%s%s%s%s%s%s%s\n",
        prefix,
        ( svmcaps & CL_SVM_CAPABILITY_SINGLE_ADDRESS_SPACE_EXP     ) ? "\n\t\t\tCL_SVM_CAPABILITY_SINGLE_ADDRESS_SPACE_EXP"      : "",
        ( svmcaps & CL_SVM_CAPABILITY_SYSTEM_ALLOCATED_EXP         ) ? "\n\t\t\tCL_SVM_CAPABILITY_SYSTEM_ALLOCATED_EXP"          : "",
        ( svmcaps & CL_SVM_CAPABILITY_DEVICE_OWNED_EXP             ) ? "\n\t\t\tCL_SVM_CAPABILITY_DEVICE_OWNED_EXP"              : "",
        ( svmcaps & CL_SVM_CAPABILITY_HOST_OWNED_EXP               ) ? "\n\t\t\tCL_SVM_CAPABILITY_HOST_OWNED_EXP"                : "",
        ( svmcaps & CL_SVM_CAPABILITY_HOST_READ_EXP                ) ? "\n\t\t\tCL_SVM_CAPABILITY_HOST_READ_EXP"                 : "",
        ( svmcaps & CL_SVM_CAPABILITY_HOST_WRITE_EXP               ) ? "\n\t\t\tCL_SVM_CAPABILITY_HOST_WRITE_EXP"                : "",
        ( svmcaps & CL_SVM_CAPABILITY_HOST_MAP_EXP                 ) ? "\n\t\t\tCL_SVM_CAPABILITY_HOST_MAP_EXP"                  : "",
        ( svmcaps & CL_SVM_CAPABILITY_DEVICE_READ_EXP              ) ? "\n\t\t\tCL_SVM_CAPABILITY_DEVICE_READ_EXP"               : "",
        ( svmcaps & CL_SVM_CAPABILITY_DEVICE_WRITE_EXP             ) ? "\n\t\t\tCL_SVM_CAPABILITY_DEVICE_WRITE_EXP"              : "",
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
                printf("\t\tinferred name: %s\n", get_svm_name(usmTypes[t].capabilities));
                PrintUSVMCaps("\t\tcapabilities: ", usmTypes[t].capabilities);
                if (usmTypes[t].optional_capabilities) {
                    PrintUSVMCaps("\t\toptional_capabilities: ", usmTypes[t].optional_capabilities);
                }
            }

            cl::Context context{devices[d]};

            cl_svm_capabilities_exp caps = 0;
            clGetSuggestedSVMCapabilitiesEXP(
                context(),
                1,
                &devices[d](),
                CL_SVM_TYPE_MACRO_DEVICE_EXP,
                &caps);
            printf("\tSuggested SVM caps with device handle is: %016" PRIx64 "\n",
                caps);

            caps = 0;
            clGetSuggestedSVMCapabilitiesEXP(
                context(),
                0,
                nullptr,
                CL_SVM_TYPE_MACRO_DEVICE_EXP,
                &caps);
            printf("\tSuggested SVM caps NULL device handle is: %016" PRIx64 "\n",
                caps);

            printf( "\n" );
        }
    }

    printf("Cleaning up...\n");

    return 0;
}