/*
// Copyright (c) 2023 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <popl/popl.hpp>

#include <CL/opencl.hpp>

#include <cinttypes>

static inline bool has_all_svm_caps(
    cl_svm_capabilities_khr caps,
    cl_svm_capabilities_khr check)
{
    return (caps & check) == check;
}

static const char* get_svm_name(cl_svm_capabilities_khr svmcaps)
{
    if (has_all_svm_caps(svmcaps, CL_SVM_TYPE_MACRO_SYSTEM_KHR)) {
        return "SYSTEM";
    }
    if (has_all_svm_caps(svmcaps, CL_SVM_TYPE_MACRO_FINE_GRAIN_BUFFER_KHR)) {
        return "FINE_GRAIN_BUFFER";
    }
    if (has_all_svm_caps(svmcaps, CL_SVM_TYPE_MACRO_COARSE_GRAIN_BUFFER_KHR)) {
        return "COARSE_GRAIN_BUFFER";
    }
    if (has_all_svm_caps(svmcaps, CL_SVM_TYPE_MACRO_HOST_KHR)) {
        return "HOST";
    }
    if (has_all_svm_caps(svmcaps, CL_SVM_TYPE_MACRO_DEVICE_KHR)) {
        return "DEVICE";
    }
    if (has_all_svm_caps(svmcaps, CL_SVM_TYPE_MACRO_SINGLE_DEVICE_SHARED_KHR)) {
        return "SINGLE_DEVICE_SHARED";
    }
    if (svmcaps == 0) {
        return "UNSUPPORTED";
    }
    return "*** UNKNOWN! ***";
}

void PrintUSVMCaps(
    const char* prefix,
    cl_svm_capabilities_khr svmcaps )
{
    printf("%s%s%s%s%s%s%s%s%s%s%s%s%s%s\n",
        prefix,
        ( svmcaps & CL_SVM_CAPABILITY_SINGLE_ADDRESS_SPACE_KHR     ) ? "\n\t\t\tCL_SVM_CAPABILITY_SINGLE_ADDRESS_SPACE_KHR"      : "",
        ( svmcaps & CL_SVM_CAPABILITY_SYSTEM_ALLOCATED_KHR         ) ? "\n\t\t\tCL_SVM_CAPABILITY_SYSTEM_ALLOCATED_KHR"          : "",
        ( svmcaps & CL_SVM_CAPABILITY_DEVICE_OWNED_KHR             ) ? "\n\t\t\tCL_SVM_CAPABILITY_DEVICE_OWNED_KHR"              : "",
        ( svmcaps & CL_SVM_CAPABILITY_HOST_OWNED_KHR               ) ? "\n\t\t\tCL_SVM_CAPABILITY_HOST_OWNED_KHR"                : "",
        ( svmcaps & CL_SVM_CAPABILITY_HOST_READ_KHR                ) ? "\n\t\t\tCL_SVM_CAPABILITY_HOST_READ_KHR"                 : "",
        ( svmcaps & CL_SVM_CAPABILITY_HOST_WRITE_KHR               ) ? "\n\t\t\tCL_SVM_CAPABILITY_HOST_WRITE_KHR"                : "",
        ( svmcaps & CL_SVM_CAPABILITY_HOST_MAP_KHR                 ) ? "\n\t\t\tCL_SVM_CAPABILITY_HOST_MAP_KHR"                  : "",
        ( svmcaps & CL_SVM_CAPABILITY_DEVICE_READ_KHR              ) ? "\n\t\t\tCL_SVM_CAPABILITY_DEVICE_READ_KHR"               : "",
        ( svmcaps & CL_SVM_CAPABILITY_DEVICE_WRITE_KHR             ) ? "\n\t\t\tCL_SVM_CAPABILITY_DEVICE_WRITE_KHR"              : "",
        ( svmcaps & CL_SVM_CAPABILITY_DEVICE_ATOMIC_ACCESS_KHR     ) ? "\n\t\t\tCL_SVM_CAPABILITY_DEVICE_ATOMIC_ACCESS_KHR"      : "",
        ( svmcaps & CL_SVM_CAPABILITY_CONCURRENT_ACCESS_KHR        ) ? "\n\t\t\tCL_SVM_CAPABILITY_CONCURRENT_ACCESS_KHR"         : "",
        ( svmcaps & CL_SVM_CAPABILITY_CONCURRENT_ATOMIC_ACCESS_KHR ) ? "\n\t\t\tCL_SVM_CAPABILITY_CONCURRENT_ATOMIC_ACCESS_KHR"  : "",
        ( svmcaps & CL_SVM_CAPABILITY_INDIRECT_ACCESS_KHR          ) ? "\n\t\t\tCL_SVM_CAPABILITY_INDIRECT_ACCESS_KHR"           : "" );
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
        printf("\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n");
        printf("Platform[%zu]: %s\n",
            i,
            platforms[i].getInfo<CL_PLATFORM_NAME>().c_str());

        std::vector<cl_svm_capabilities_khr> typeCapsPlatform =
            platforms[i].getInfo<CL_PLATFORM_SVM_TYPE_CAPABILITIES_KHR>();
        for (size_t t = 0; t < typeCapsPlatform.size(); t++)
        {
            printf("USM Platform Type[%zu]:\n", t);
            printf("\tinferred name: %s\n", get_svm_name(typeCapsPlatform[t]));
            PrintUSVMCaps("\tcapabilities: ", typeCapsPlatform[t]);
        }

        std::vector<cl::Device> devices;
        platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &devices);

        for( size_t d = 0; d < devices.size(); d++ )
        {
            printf("\n================================================================\n");
            printf("Device[%zu]: %s\n",
                d,
                devices[d].getInfo<CL_DEVICE_NAME>().c_str());

            std::vector<cl_svm_capabilities_khr> typeCapsDevice =
                devices[d].getInfo<CL_DEVICE_SVM_TYPE_CAPABILITIES_KHR>();
            for (size_t t = 0; t < typeCapsDevice.size(); t++)
            {
                printf("USM Type[%zu]:\n", t);
                printf("\tinferred name: %s\n", get_svm_name(typeCapsDevice[t]));
                PrintUSVMCaps("\tcapabilities: ", typeCapsDevice[t]);
            }

            printf("\n----------------------------------------------------------------\n");
            cl::Context context{devices[d]};

            const cl_svm_alloc_properties_khr props[] = {
                CL_SVM_ALLOC_ASSOCIATED_DEVICE_HANDLE_KHR, (cl_svm_alloc_properties_khr)devices[d](),
                0,
            };

            cl_uint suggested = CL_UINT_MAX;
            clGetSVMSuggestedTypeIndexKHR(
                context(),
                CL_SVM_TYPE_MACRO_DEVICE_KHR,
                0,
                props,
                0,
                &suggested);
            printf("Suggested device SVM type index with associated device is: %u (%08X)\n",
                suggested, suggested);

            suggested = CL_UINT_MAX;
            clGetSVMSuggestedTypeIndexKHR(
                context(),
                CL_SVM_TYPE_MACRO_DEVICE_KHR,
                0,
                nullptr,
                0,
                &suggested);
            printf("Suggested device SVM type index with no associated device is: %u (%08X)\n",
                suggested, suggested);

            suggested = CL_UINT_MAX;
            clGetSVMSuggestedTypeIndexKHR(
                context(),
                CL_SVM_TYPE_MACRO_COARSE_GRAIN_BUFFER_KHR,
                0,
                nullptr,
                0,
                &suggested);
            printf("Suggested coarse grain SVM type index with no associated device is: %u (%08X)\n",
                suggested, suggested);
        }
    }

    printf("\nCleaning up...\n");

    return 0;
}