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

static cl::Program createProgramWithIL(
    const cl::Context& context,
    const std::vector<cl_uchar>& il )
{
    cl_program program = nullptr;

    // Use the core clCreateProgramWithIL if a device supports OpenCL 2.1 or
    // newer and SPIR-V.
    bool useCore = false;

    // Use the extension clCreateProgramWithILKHR if a device supports
    // cl_khr_il_program.
    bool useExtension = false;

    std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
    for (auto device : devices) {
#ifdef CL_VERSION_2_1
        // Note: This could look for "SPIR-V" in CL_DEVICE_IL_VERSION.
        if (getDeviceOpenCLVersion(device) >= CL_MAKE_VERSION(2, 1, 0) &&
            !device.getInfo<CL_DEVICE_IL_VERSION>().empty()) {
            useCore = true;
        }
#endif
        if (checkDeviceForExtension(device, "cl_khr_il_program")) {
            useExtension = true;
        }
    }

#ifdef CL_VERSION_2_1
    if (useCore) {
        program = clCreateProgramWithIL(
            context(),
            il.data(),
            il.size(),
            nullptr);
    }
    else
#endif
    if (useExtension) {
        cl::Platform platform{ devices[0].getInfo<CL_DEVICE_PLATFORM>() };

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

    return cl::Program{ program };
}

int main(
    int argc,
    char** argv )
{
    int platformIndex = 0;
    int deviceIndex = 0;

    std::string fileName(sizeof(void*) == 8  ? "sample_kernel64.spv" : "sample_kernel32.spv");
    std::string buildOptions;

    {
        popl::OptionParser op("Supported Options");
        op.add<popl::Value<int>>("p", "platform", "Platform Index", platformIndex, &platformIndex);
        op.add<popl::Value<int>>("d", "device", "Device Index", deviceIndex, &deviceIndex);
        op.add<popl::Value<std::string>>("", "file", "Kernel File Name", fileName, &fileName);
        op.add<popl::Value<std::string>>("", "options", "Program Build Options", buildOptions, &buildOptions);
        bool printUsage = false;
        try {
            op.parse(argc, argv);
        } catch (std::exception& e) {
            fprintf(stderr, "Error: %s\n\n", e.what());
            printUsage = true;
        }

        if (printUsage || !op.unknown_options().empty() || !op.non_option_args().empty()) {
            fprintf(stderr,
                "Usage: globalvar [options]\n"
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
    printf("CL_DEVICE_ADDRESS_BITS is %d for this device.\n",
        device.getInfo<CL_DEVICE_ADDRESS_BITS>() );

    // Check for SPIR-V support.  If the device supports OpenCL 2.1 or newer
    // we can use the core clCreateProgramWithIL API.  Otherwise, if the device
    // the cl_khr_il_program extension we can use the clCreateProgramWithILKHR
    // extension API.  If neither is supported then we cannot run this sample.
#ifdef CL_VERSION_2_1
    // Note: This could look for "SPIR-V" in CL_DEVICE_IL_VERSION.
    if (getDeviceOpenCLVersion(device) >= CL_MAKE_VERSION(2, 1, 0) &&
        !device.getInfo<CL_DEVICE_IL_VERSION>().empty()) {
        printf("Device supports OpenCL 2.1 or newer, using clCreateProgramWithIL.\n");
    } else
#endif
    if (checkDeviceForExtension(device, "cl_khr_il_program")) {
        printf("Device supports cl_khr_il_program, using clCreateProgramWithILKHR.\n");
    } else {
        printf("Device does not support SPIR-V, exiting.\n");
        return -1;
    }

    cl::Context context{device};
    cl::CommandQueue commandQueue{context, device};

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

    typedef cl_int CL_API_CALL
    clGetDeviceGlobalVariablePointerINTEL_t(
        cl_device_id device,
        cl_program program,
        const char *globalVariableName,
        size_t *globalVariableSizeRet,
        void **globalVariablePointerRet);

    typedef clGetDeviceGlobalVariablePointerINTEL_t *
    clGetDeviceGlobalVariablePointerINTEL_fn;

    typedef cl_int CL_API_CALL
    clGetMemAllocInfoINTEL_t(
        cl_context context,
        const void* ptr,
        cl_mem_info_intel param_name,
        size_t param_value_size,
        void* param_value,
        size_t* param_value_size_ret);

    typedef clGetMemAllocInfoINTEL_t *
    clGetMemAllocInfoINTEL_fn ;

    auto clGetDeviceGlobalVariablePointerINTEL = (clGetDeviceGlobalVariablePointerINTEL_fn)
        clGetExtensionFunctionAddressForPlatform(platform(), "clGetDeviceGlobalVariablePointerINTEL");

    auto clGetMemAllocInfoINTEL = (clGetMemAllocInfoINTEL_fn)
        clGetExtensionFunctionAddressForPlatform(platform(), "clGetMemAllocInfoINTEL");

    if (clGetDeviceGlobalVariablePointerINTEL == nullptr) {
        printf("Couldn't get function pointer for clGetDeviceGlobalVariablePointerINTEL!\n");
    } else if (clGetMemAllocInfoINTEL == nullptr) {
        printf("Couldn't get function pointer for clGetMemAllocInfoINTEL!\n");
    } else {
        cl_int errorCode = CL_SUCCESS;
        size_t gvsize = 0; void* gvptr = nullptr;
        errorCode = clGetDeviceGlobalVariablePointerINTEL(
            device(), program(), "uid67f037ed289236e5____ZL2dg", &gvsize, &gvptr);
        printf("clGetDeviceGlobalVariablePointerINTEL with uid67f037ed289236e5____ZL2dg returned %d: %zu %p\n", errorCode, gvsize, gvptr);

        gvsize = 0; gvptr = nullptr;
        errorCode = clGetDeviceGlobalVariablePointerINTEL(
            device(), program(), "_ZL2dg", &gvsize, &gvptr);
        printf("clGetDeviceGlobalVariablePointerINTEL with _ZL2dg returned %d: %zu %p\n", errorCode, gvsize, gvptr);

        cl_unified_shared_memory_type_intel gvtype = 0;
        errorCode = clGetMemAllocInfoINTEL(
            context(), gvptr, CL_MEM_ALLOC_TYPE_INTEL, sizeof(gvtype), &gvtype, nullptr);

        printf("clGetMemAllocInfoINTEL returned %d: %04X\n", errorCode, gvtype);
    }

    printf("Done.\n");

    return 0;
}
