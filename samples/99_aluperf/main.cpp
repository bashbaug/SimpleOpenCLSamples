/*
// Copyright (c) 2022 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <popl/popl.hpp>

#include <CL/opencl.hpp>

#include <algorithm>
#include <cinttypes>
#include <sstream>
#include <vector>

int main(
    int argc,
    char** argv )
{
    int platformIndex = 0;
    int deviceIndex = 0;

    size_t gws = 1024 * 1024;
    size_t lws = 0;
    size_t iterations = 16;
    size_t ops = 1024;
    uint32_t seed = 0;

    std::string datatype("float");
    std::string operation("z = z + x");
    std::string buildOptions;

    {
        popl::OptionParser op("Supported Options");
        op.add<popl::Value<int>>("p", "platform", "Platform Index", platformIndex, &platformIndex);
        op.add<popl::Value<int>>("d", "device", "Device Index", deviceIndex, &deviceIndex);
        op.add<popl::Value<size_t>>("g", "gws", "Global Work Size", gws, &gws);
        op.add<popl::Value<size_t>>("l", "lws", "Local Work Size (0 -> NULL)", lws, &lws);
        op.add<popl::Value<size_t>>("i", "iterations", "Iterations", iterations, &iterations);
        op.add<popl::Value<size_t>>("o", "ops", "Operations to Run Per Kernel", ops, &ops);
        op.add<popl::Value<uint32_t>>("s", "seed", "Seed Value For Computation", seed, &seed);
        op.add<popl::Value<std::string>>("", "options", "Program Build Options", buildOptions, &buildOptions);
        op.add<popl::Value<std::string>>("", "type", "Data Type for Computation", datatype, &datatype);
        op.add<popl::Value<std::string>>("", "operation", "Operation to Test", operation, &operation);

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
            fprintf(stderr,
                "\n"
                "Note: for best results, the operation should assign to z and be a function of z.\n"
                "    Other symbols that can be used are:\n"
                "      x: data read from a buffer, unique for each local id\n"
                "      y: this work-item's local id\n"
                "    Example: z = z + x\n");
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

    cl::Context context{devices[deviceIndex]};

    std::stringstream ss;

    if (datatype.rfind("half", 0) == 0) {
        ss << "#pragma OPENCL EXTENSION cl_khr_fp16 : enable" << std::endl;
    }
    else if (datatype.rfind("double", 0) == 0) {
        ss << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable" << std::endl;
    }

    ss << "__kernel void Bench(__global " << datatype << " * buffer)" << std::endl;
    ss << "{" << std::endl;
    ss << "    " << datatype << " x = buffer[get_local_id(0)];" << std::endl;
    ss << "    " << datatype << " y = (" << datatype << ")get_local_id(0);" << std::endl;
    ss << "    " << datatype << " z = x;" << std::endl;
    for (size_t i = 0; i < ops; i++) {
        ss << " " << operation << ";" << std::endl;
    }

    ss << "    buffer[get_local_id(0)] = z;" << std::endl;
    ss << "}" << std::endl;

    cl::Program program{ context, ss.str() };

    printf("Building program with build options: %s\n",
        buildOptions.empty() ? "(none)" : buildOptions.c_str());
    program.build(buildOptions.c_str());
    cl::Kernel kernel{ program, "Bench" };

    cl::CommandQueue commandQueue{context, devices[deviceIndex], CL_QUEUE_PROFILING_ENABLE};

    std::vector<uint32_t> data(1024 * 16 * 16 * 2, seed);
    cl::Buffer buf = cl::Buffer{
        context,
        CL_MEM_COPY_HOST_PTR,
        data.size() * sizeof(data[0]),
        data.data() };

    kernel.setArg(0, buf);

    double minTime = 1e9;
    double maxTime = 0;

    for (size_t i = 0; i < iterations; i++) {
        printf("."); fflush(stdout);

        cl::Event event;
        commandQueue.enqueueNDRangeKernel(
            kernel,
            cl::NullRange,
            cl::NDRange{gws},
            (lws == 0) ? cl::NullRange : cl::NDRange{lws},
            nullptr,
            &event);
        commandQueue.finish();

        cl_ulong start = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        cl_ulong end = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();

        double time = (end - start) / 1e9;

        minTime = std::min(time, minTime);
        maxTime = std::max(time, maxTime);
    }

    //double rate = (double)gws * ops / (minTime * 1024 * 1024 * 1024);
    double rate = (double)gws * ops / (minTime * 1e9);

    std::stringstream fnty;
    fnty << operation << " (" << datatype << ")";

    printf("\n");
    printf("%32s %10s %10s %12s\n", "Function", "Min Time", "Max Time", "Max GOps/s");
    printf("%32s %10f %10f %12f\n", fnty.str().c_str(), minTime, maxTime, rate);

    return 0;
}
