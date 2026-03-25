/*
// Copyright (c) 2024-2026 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <popl/popl.hpp>

#include <CL/opencl.hpp>

void PrintSVMCaps(
    const char* label,
    cl_device_svm_capabilities svmcaps )
{
    printf("\t%s: %s%s%s%s\n",
        label,
        ( svmcaps & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER   ) ? "\n\t\tCL_DEVICE_SVM_COARSE_GRAIN_BUFFER"   : "",
        ( svmcaps & CL_DEVICE_SVM_FINE_GRAIN_BUFFER     ) ? "\n\t\tCL_DEVICE_SVM_FINE_GRAIN_BUFFER"     : "",
        ( svmcaps & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM     ) ? "\n\t\tCL_DEVICE_SVM_FINE_GRAIN_SYSTEM"     : "",
        ( svmcaps & CL_DEVICE_SVM_ATOMICS               ) ? "\n\t\tCL_DEVICE_SVM_ATOMICS"               : "" );
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
                "Usage: svmqueries [options]\n"
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

            cl_device_svm_capabilities svmcaps =
                devices[d].getInfo<CL_DEVICE_SVM_CAPABILITIES>();
            PrintSVMCaps( "CL_DEVICE_SVM_CAPABILITIES", svmcaps );

            printf( "\n" );
        }
    }

    printf("Cleaning up...\n");

    return 0;
}