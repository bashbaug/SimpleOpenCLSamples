/*
// Copyright (c) 2023 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <stdio.h>
#include <vector>
#include <popl/popl.hpp>

#include <CL/opencl.hpp>

typedef cl_int (*pfn_clShutdownOCLICD)(void);
pfn_clShutdownOCLICD clShutdownOCLICD = NULL;

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
                "Usage: loadershutdown [options]\n"
                "%s", op.help().c_str());
            return -1;
        }
    }


    clShutdownOCLICD = (pfn_clShutdownOCLICD)
        clGetExtensionFunctionAddress("clShutdownOCLICD");

    if (clShutdownOCLICD == NULL) {
        printf("Couldn't get function pointer to clShutdownOCLICD!\n");
        printf("This is normal and some ICD loaders do not support this functionality.\n");
        printf("Exiting...\n");
        return 0;
    }

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    printf("Before shutting down: queried %zu platforms.\n", platforms.size());
    for (const auto& platform: platforms) {
        cl_int errorCode = CL_SUCCESS;
        auto name = platform.getInfo<CL_PLATFORM_NAME>(&errorCode);
        if (errorCode == CL_SUCCESS) {
            printf("\tPlatform: %s\n", name.c_str());
        } else {
            printf("\tQuery returned %d\n", errorCode);
        }
    }

    clShutdownOCLICD();

    printf("\nAfter shutting down:\n");
    for (const auto& platform: platforms) {
        cl_int errorCode = CL_SUCCESS;
        auto name = platform.getInfo<CL_PLATFORM_NAME>(&errorCode);
        if (errorCode == CL_SUCCESS) {
            printf("\tPlatform: %s\n", name.c_str());
        } else {
            printf("\tQuery returned %d\n", errorCode);
        }
    }

    return 0;
}
