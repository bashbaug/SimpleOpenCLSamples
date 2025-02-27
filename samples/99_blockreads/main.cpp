/*
// Copyright (c) 2019-2025 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <popl/popl.hpp>

#include <CL/opencl.hpp>

#include <fstream>
#include <string>

#include "util.hpp"

static std::string readStringFromFile(
    const std::string& filename )
{
    std::ifstream is(filename, std::ios::binary);
    if (!is.good()) {
        printf("Couldn't open file '%s'!\n", filename.c_str());
        return "";
    }

    size_t filesize = 0;
    is.seekg(0, std::ios::end);
    filesize = (size_t)is.tellg();
    is.seekg(0, std::ios::beg);

    std::string source{
        std::istreambuf_iterator<char>(is),
        std::istreambuf_iterator<char>() };

    return source;
}

template <typename T>
void fill_matrix(std::vector<T>& M, size_t numRows, size_t numCols)
{
    for (size_t r = 0; r < numRows; r++) {
        for (size_t c = 0; c < numCols; c++) {
            T value = static_cast<T>(((r % 256) * 65536) + (c % 256));
            M.push_back(value);
        }
    }
}

template <>
void fill_matrix(std::vector<uint8_t>& M, size_t numRows, size_t numCols)
{
    uint8_t value = 0;
    for (size_t r = 0; r < numRows; r++) {
        for (size_t c = 0; c < numCols; c++) {
            M.push_back(value++);
        }
    }
}

int main(
    int argc,
    char** argv )
{
    int platformIndex = 0;
    int deviceIndex = 0;

    std::string fileName("block_read_kernel.cl");
    std::string kernelName("BlockReadTest");
    std::string buildOptions;
    size_t gwx = 512;

    {
        popl::OptionParser op("Supported Options");
        op.add<popl::Value<int>>("p", "platform", "Platform Index", platformIndex, &platformIndex);
        op.add<popl::Value<int>>("d", "device", "Device Index", deviceIndex, &deviceIndex);
        op.add<popl::Value<std::string>>("", "file", "Kernel File Name", fileName, &fileName);
        op.add<popl::Value<std::string>>("", "name", "Kernel Name", kernelName, &kernelName);
        op.add<popl::Value<std::string>>("", "options", "Program Build Options", buildOptions, &buildOptions);
        op.add<popl::Value<size_t>>("", "gwx", "Global Work Size", gwx, &gwx);
        bool printUsage = false;
        try {
            op.parse(argc, argv);
        } catch (std::exception& e) {
            fprintf(stderr, "Error: %s\n\n", e.what());
            printUsage = true;
        }

        if (printUsage || !op.unknown_options().empty() || !op.non_option_args().empty()) {
            fprintf(stderr,
                "Usage: blockreads [options]\n"
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

    bool has_cl_intel_subgroup_2d_block_io =
        checkDeviceForExtension(devices[deviceIndex], "cl_intel_subgroup_2d_block_io");
    if (has_cl_intel_subgroup_2d_block_io) {
        printf("Device supports cl_intel_subgroup_2d_block_io.\n");
    } else {
        printf("Device does not support cl_intel_subgroup_2d_block_io, exiting.\n");
        return -1;
    }
    cl::Context context{devices[deviceIndex]};
    cl::CommandQueue commandQueue{context, devices[deviceIndex]};

    printf("Reading program source from file: %s\n", fileName.c_str() );
    std::string kernelString = readStringFromFile(fileName.c_str());

    printf("Building program with build options: %s\n",
        buildOptions.empty() ? "(none)" : buildOptions.c_str() );
    cl::Program program{ context, kernelString };
    program.build(buildOptions.c_str());
    for( auto& device : program.getInfo<CL_PROGRAM_DEVICES>() )
    {
        printf("Program build log for device %s:\n",
            device.getInfo<CL_DEVICE_NAME>().c_str() );
        printf("%s\n",
            program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device).c_str() );
    }
    printf("Creating kernel: %s\n", kernelName.c_str() );
    cl::Kernel kernel = cl::Kernel{ program, kernelName.c_str() };

    constexpr size_t numRows = 64;
    constexpr size_t numCols = 64;

    //std::vector<uint32_t> matrix;
    std::vector<uint8_t> matrix;
    matrix.reserve(numRows * numCols);
    fill_matrix(matrix, numRows, numCols);

    cl::Buffer mem = cl::Buffer{
        context,
        CL_MEM_COPY_HOST_PTR,
        matrix.size() * sizeof(matrix[0]),
        matrix.data() };

    // execution
    kernel.setArg(0, mem);
    kernel.setArg(1, static_cast<int>(numCols * sizeof(matrix[0])));
    kernel.setArg(2, static_cast<int>(numRows));
    commandQueue.enqueueNDRangeKernel(
        kernel,
        cl::NullRange,
        cl::NDRange{16},
        cl::NDRange{16} );

    commandQueue.finish();

    printf("Done.\n");

    return 0;
}
