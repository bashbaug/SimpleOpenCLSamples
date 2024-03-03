/*
// Copyright (c) 2019-2024 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <popl/popl.hpp>

#include <CL/opencl.hpp>

#include <thread>

static const char kernelString[] = R"CLC(
int func(); // not implemented, produces link error

kernel void Test(global int* dst)
{
    uint id = get_global_id(0);
    dst[id] = func();
}
)CLC";

void CL_CALLBACK program_callback(cl_program program, void* user_data)
{
    printf("In program_callback: program = %p, user_data = %p\n", program, user_data);
    cl::Program query{program, true};
    if (query() != nullptr) {
        for (auto& device : query.getInfo<CL_PROGRAM_DEVICES>()) {
            printf("Program build status: %d\n", query.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device));
            printf("Program build log:\n%s\n", query.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device).c_str() );
        }
    }
    printf("End of program callback.\n\n");
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
                "Usage: copybufferkernel [options]\n"
                "%s", op.help().c_str());
            return -1;
        }
    }

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Platform& platform = platforms[platformIndex];

    printf("Running on platform: %s\n",
        platform.getInfo<CL_PLATFORM_NAME>().c_str() );

    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    cl::Device& device = devices[deviceIndex];

    printf("Running on device: %s\n",
        device.getInfo<CL_DEVICE_NAME>().c_str() );

    cl::Context context{device};
    cl::Program object{context, kernelString};

    cl_int errorCode = CL_SUCCESS;

    printf("\n\nCompiling program object %p...\n", object());
    errorCode = object.compile(nullptr, program_callback, nullptr);
    printf("clCompileProgram() returned %d\n", errorCode);
    if (object() != nullptr) {
        while (object.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device) == CL_BUILD_IN_PROGRESS) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        printf("Program compile log for device %s:\n",
            device.getInfo<CL_DEVICE_NAME>().c_str() );
        printf("%s\n",
            object.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device).c_str() );
    }

    printf("\n\nLinking program...\n");
    cl::Program program = cl::linkProgram({object}, nullptr, program_callback, nullptr, &errorCode);
    printf("clLinkProgram() returned %d\n", errorCode);
    if (program() != nullptr) {
        printf("clLinkProgram() created program object %p.\n", program());
        while (program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device) == CL_BUILD_IN_PROGRESS) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        printf("Program link log for device %s:\n",
            device.getInfo<CL_DEVICE_NAME>().c_str() );
        printf("%s\n",
            program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device).c_str() );
    } else {
        printf("clLinkProgram() returned nullptr.\n");
    }

    printf("All done.\n");

    return 0;
}
