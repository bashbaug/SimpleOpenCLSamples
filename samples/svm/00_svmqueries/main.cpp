/*
// Copyright (c) 2024 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <popl/popl.hpp>

#include <CL/opencl.hpp>

void PrintSVMCaps(
    const char* label,
    cl_device_svm_capabilities svmcaps )
{
    printf("%s: %s%s%s%s\n",
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
    int platformIndex = 0;
    int deviceIndex = 0;

    {
        popl::OptionParser op("Supported Options");
        op.add<popl::Value<int>>("p", "platform", "Platform Index", platformIndex, &platformIndex);
        op.add<popl::Value<int>>("d", "device", "Device Index", deviceIndex, &deviceIndex);
        bool printUsage = false;
        try {
            op.parse(argc, argv);
        } catch (std::exception& e) {
            fprintf(stderr, "Error: %s\n\n", e.what());
            printUsage = true;
        }

        if (printUsage || !op.unknown_options().empty() || !op.non_option_args().empty()) {
            fprintf(stderr,
                "Usage: usmqueries [options]\n"
                "%s", op.help().c_str());
            return -1;
        }
    }

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    printf("Running on platform: %s\n",
        platforms[platformIndex].getInfo<CL_PLATFORM_NAME>().c_str() );

    std::vector<cl::Device> devices;
    platforms[platformIndex].getDevices(CL_DEVICE_TYPE_ALL, &devices);

    printf("Running on device: %s\n",
        devices[deviceIndex].getInfo<CL_DEVICE_NAME>().c_str() );

    cl_device_svm_capabilities svmcaps = devices[deviceIndex].getInfo<CL_DEVICE_SVM_CAPABILITIES>();
    PrintSVMCaps( "CL_DEVICE_SVM_CAPABILITIES", svmcaps );

    printf("Cleaning up...\n");

    return 0;
}