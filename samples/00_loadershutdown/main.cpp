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

static void PrintPlatformList(const std::vector<cl::Platform>& platforms)
{
    for (const auto& platform: platforms) {
        cl_int errorCode = CL_SUCCESS;
        auto name = platform.getInfo<CL_PLATFORM_NAME>(&errorCode);
        if (errorCode == CL_SUCCESS) {
            printf("\tPlatform: %s\n", name.c_str());
        } else {
            printf("\tQuery returned %d\n", errorCode);
        }
    }
}

int main(
    int argc,
    char** argv )
{
    bool skipShutdown = false;

    {
        popl::OptionParser op("Supported Options");
        op.add<popl::Switch>("s", "skipshutdown", "Skip ICD Loader Shutdown", &skipShutdown);

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

    std::vector<cl::Platform> before;
    cl::Platform::get(&before);

    printf("Before shutting down: queried %zu platform(s).\n", before.size());
    PrintPlatformList(before);

    if (!skipShutdown) {
        printf("\nCalling clShutdownOCLICD()!\n");
        clShutdownOCLICD();
    }

    std::vector<cl::Platform> after;
    cl::Platform::get(&after);

    printf("\nAfter shutting down: queried %zu platform(s).\n", after.size());

    printf("\nPlatform information from old platform list:\n");
    PrintPlatformList(before);

    printf("\nPlatform information from new platform list:\n");
    PrintPlatformList(after);

    printf("\nAll done.\n");
    return 0;
}
