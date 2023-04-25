/*
// Copyright (c) 2019-2023 Ben Ashbaugh
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

int main(
    int argc,
    char** argv )
{
    int platformIndex = 0;
    int deviceIndex = 0;

    std::string fileName("ndrange_sample_kernel.cl");
    std::string kernelName("Test");
    std::string buildOptions;
    std::string compileOptions;
    std::string linkOptions;
    bool compile = false;
    size_t gwx = 512;
    size_t gwy = 1;
    size_t lwx = 32;
    size_t lwy = 1;

    {
        bool advanced = false;

        popl::OptionParser op("Supported Options");
        op.add<popl::Value<int>>("p", "platform", "Platform Index", platformIndex, &platformIndex);
        op.add<popl::Value<int>>("d", "device", "Device Index", deviceIndex, &deviceIndex);
        op.add<popl::Value<std::string>>("", "file", "Kernel File Name", fileName, &fileName);
        op.add<popl::Value<std::string>>("", "name", "Kernel Name", kernelName, &kernelName);
        op.add<popl::Value<std::string>>("", "options", "Program Build Options", buildOptions, &buildOptions);
        op.add<popl::Value<size_t>>("", "gwx", "Global Work Size in the X-dimension", gwx, &gwx);
        op.add<popl::Value<size_t>>("", "gwy", "Global Work Size in the Y-dimension", gwy, &gwy);
        op.add<popl::Value<size_t>>("", "lwx", "Local Work Size in the X-dimension (0 -> NULL)", lwx, &lwx);
        op.add<popl::Value<size_t>>("", "lwy", "Local Work Size in the Y-dimension (0 -> NULL)", lwy, &lwy);
        op.add<popl::Switch>("a", "advanced", "Show advanced options", &advanced);
        op.add<popl::Switch, popl::Attribute::advanced>("c", "compilelink", "Use clCompileProgram and clLinkProgram", &compile);
        op.add<popl::Value<std::string>, popl::Attribute::advanced>("", "compileoptions", "Program Compile Options", compileOptions, &compileOptions);
        op.add<popl::Value<std::string>, popl::Attribute::advanced>("", "linkoptions", "Program Link Options", linkOptions, &linkOptions);
        bool printUsage = false;
        try {
            op.parse(argc, argv);
        } catch (std::exception& e) {
            fprintf(stderr, "Error: %s\n\n", e.what());
            printUsage = true;
        }

        if (printUsage || !op.unknown_options().empty() || !op.non_option_args().empty()) {
            fprintf(stderr,
                "Usage: ndrangekernelfromfile [options]\n"
                "%s", op.help(advanced ? popl::Attribute::advanced : popl::Attribute::optional).c_str());
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
    cl::CommandQueue commandQueue = cl::CommandQueue{context, devices[deviceIndex]};

    printf("Reading program source from file: %s\n", fileName.c_str() );
    std::string kernelString = readStringFromFile(fileName.c_str());

    cl::Program program;
    if (compile) {
        printf("Compiling program with compile options: %s\n",
            compileOptions.empty() ? "(none)" : compileOptions.c_str() );
        cl::Program object{ context, kernelString };
        object.compile(compileOptions.c_str());
        for( auto& device : program.getInfo<CL_PROGRAM_DEVICES>() )
        {
            printf("Program compile log for device %s:\n",
                device.getInfo<CL_DEVICE_NAME>().c_str() );
            printf("%s\n",
                object.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device).c_str() );
        }

        printf("Linking program with link options: %s\n",
            linkOptions.empty() ? "(none)" : linkOptions.c_str() );
        program = cl::linkProgram({object}, linkOptions.c_str());
        for( auto& device : program.getInfo<CL_PROGRAM_DEVICES>() )
        {
            printf("Program link log for device %s:\n",
                device.getInfo<CL_DEVICE_NAME>().c_str() );
            printf("%s\n",
                program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device).c_str() );
        }
    } else {
        printf("Building program with build options: %s\n",
            buildOptions.empty() ? "(none)" : buildOptions.c_str() );
        program = cl::Program{ context, kernelString };
        program.build(buildOptions.c_str());
        for( auto& device : program.getInfo<CL_PROGRAM_DEVICES>() )
        {
            printf("Program build log for device %s:\n",
                device.getInfo<CL_DEVICE_NAME>().c_str() );
            printf("%s\n",
                program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device).c_str() );
        }
    }
    printf("Creating kernel: %s\n", kernelName.c_str() );
    cl::Kernel kernel = cl::Kernel{ program, kernelName.c_str() };

    cl::Buffer deviceMemDst = cl::Buffer{
        context,
        CL_MEM_ALLOC_HOST_PTR,
        gwx * gwy * sizeof( cl_uint ) };

    // execution
    kernel.setArg(0, deviceMemDst);
    commandQueue.enqueueNDRangeKernel(
        kernel,
        cl::NullRange,
        cl::NDRange{gwx, gwy},
        (lwx == 0) || (lwy == 0) ?
            cl::NullRange :
            cl::NDRange{lwx, lwy});

    // verify results by printing the first few values
    if (gwx > 3) {
        auto ptr = (const cl_uint*)commandQueue.enqueueMapBuffer(
            deviceMemDst,
            CL_TRUE,
            CL_MAP_READ,
            0,
            gwx * gwy * sizeof( cl_uint ) );

        printf("First few values: [0] = %u, [1] = %u, [2] = %u\n", ptr[0], ptr[1], ptr[2]);

        commandQueue.enqueueUnmapMemObject(
            deviceMemDst,
            (void*)ptr );
    }

    commandQueue.finish();

    printf("Done.\n");

    return 0;
}
