/*
// Copyright (c) 2019-2021 Ben Ashbaugh
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
*/

#include <popl/popl.hpp>

#include <CL/opencl.hpp>

#include <fstream>
#include <string>

static std::vector<cl_uchar> readSPIRVFromFile(
    const std::string& filename )
{
    std::ifstream is(filename, std::ios::binary);
    std::vector<cl_uchar> ret;
    if (!is.good()) {
        printf("Couldn't open file '%s'!\n", filename.c_str());
        return ret;
    }

    size_t filesize = 0;
    is.seekg(0, std::ios::end);
    filesize = (size_t)is.tellg();
    is.seekg(0, std::ios::beg);

    ret.reserve(filesize);
    ret.insert(
        ret.begin(),
        std::istreambuf_iterator<char>(is),
        std::istreambuf_iterator<char>() );

    return ret;
}

static cl_uint getPlatformVersion(const cl::Platform& platform)
{
    cl_uint major = 0;
    cl_uint minor = 0;

    std::string version = platform.getInfo<CL_PLATFORM_VERSION>();

    // The platform version string has the form:
    //   OpenCL <Major>.<Minor> <Vendor Specific Info>
    const std::string prefix{"OpenCL "};
    if (!version.compare(0, prefix.length(), prefix)) {
        const char* check = version.c_str() + prefix.length();
        while (isdigit(check[0])) {
            major *= 10;
            major += check[0] - '0';
            ++check;
        }
        if (check[0] == '.') {
            ++check;
        }
        while (isdigit(check[0])) {
            minor *= 10;
            minor += check[0] - '0';
            ++check;
        }
    }

    return (major << 16) | minor;
}

static cl::Program createProgramWithIL(
    const cl::Context& context,
    const std::vector<cl_uchar>& il )
{
    cl_program program = nullptr;

    std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
    if (!devices.empty()) {
        cl::Platform platform{ devices[0].getInfo<CL_DEVICE_PLATFORM>() };
#ifdef CL_VERSION_2_1
        if (getPlatformVersion(platform) >= 0x00020001) {
            std::string ilVersions = devices[0].getInfo<CL_DEVICE_IL_VERSION>();
            if (!ilVersions.empty()) {
                program = clCreateProgramWithIL(
                    context(),
                    il.data(),
                    il.size(),
                    nullptr);
            }
        }
        else
#endif
        {
            auto clCreateProgramWithILKHR_ = (clCreateProgramWithILKHR_fn)
                clGetExtensionFunctionAddressForPlatform(
                    platform(),
                    "clCreateProgramWithILKHR");

            if (clCreateProgramWithILKHR_) {
                program = clCreateProgramWithILKHR_(
                    context(),
                    il.data(),
                    il.size(),
                    nullptr);
            }
        }
    }

    return cl::Program{ program };
}

int main(
    int argc,
    char** argv )
{
    int platformIndex = 0;
    int deviceIndex = 0;

    std::string fileName(sizeof(void*) == 8  ? "sample_kernel64.spv" : "sample_kernel32.spv");
    std::string kernelName("Test");
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
                "Usage: spirvkernelfromfile [options]\n"
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
    printf("CL_DEVICE_ADDRESS_BITS is %d for this device.\n",
        devices[deviceIndex].getInfo<CL_DEVICE_ADDRESS_BITS>() );

    cl::Context context{devices[deviceIndex]};
    cl::CommandQueue commandQueue = cl::CommandQueue{context, devices[deviceIndex]};

    printf("Reading SPIR-V from file: %s\n", fileName.c_str());
    std::vector<cl_uchar> spirv = readSPIRVFromFile(fileName);

    printf("Building program with build options: %s\n",
        buildOptions.empty() ? "(none)" : buildOptions.c_str() );
    cl::Program program = createProgramWithIL( context, spirv );
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

    cl::Buffer deviceMemDst = cl::Buffer{
        context,
        CL_MEM_ALLOC_HOST_PTR,
        gwx * sizeof( cl_uint ) };

    // execution
    kernel.setArg(0, deviceMemDst);
    commandQueue.enqueueNDRangeKernel(
        kernel,
        cl::NullRange,
        cl::NDRange{gwx});

    // No results to check for this sample, but do verify that execution
    // has completed.
    commandQueue.finish();

    printf("Done.\n");

    return 0;
}
