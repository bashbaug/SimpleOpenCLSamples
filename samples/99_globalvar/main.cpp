/*
// Copyright (c) 2025-2026 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <popl/popl.hpp>

#include <CL/opencl.hpp>

#include <fstream>
#include <string>

#include "util.hpp"

static std::vector<char> readSPIRVFromFile(
    const std::string& filename )
{
    std::ifstream is(filename, std::ios::binary);
    std::vector<char> ret;
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

int main(int argc, char** argv )
{
    int platformIndex = 0;
    int deviceIndex = 0;

    std::string fileName(sizeof(void*) == 8  ? "device_global64.spv" : "device_global32.spv");
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

    printf("Running on platform: %s\n", platform.getInfo<CL_PLATFORM_NAME>().c_str() );

    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    cl::Device& device = devices[deviceIndex];

    printf("Running on device: %s\n", device.getInfo<CL_DEVICE_NAME>().c_str() );
    printf("Running on drivers: %s\n", device.getInfo<CL_DRIVER_VERSION>().c_str() );
    printf("CL_DEVICE_ADDRESS_BITS is %d for this device.\n", device.getInfo<CL_DEVICE_ADDRESS_BITS>() );

    cl::Context context{device};
    cl::CommandQueue commandQueue{context, device};

    printf("Reading SPIR-V from file: %s\n", fileName.c_str());
    std::vector<char> spirv = readSPIRVFromFile(fileName);

    printf("Building program with build options: %s\n",
        buildOptions.empty() ? "(none)" : buildOptions.c_str() );
    cl::Program program{context, spirv};
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

    constexpr const char* HostAccessName = "HostAccessName";
    constexpr const char* ExportName = "ExportName";

    if (clGetDeviceGlobalVariablePointerINTEL == nullptr) {
        printf("Couldn't get function pointer for clGetDeviceGlobalVariablePointerINTEL!\n");
    } else if (clGetMemAllocInfoINTEL == nullptr) {
        printf("Couldn't get function pointer for clGetMemAllocInfoINTEL!\n");
    } else {
        cl_int errorCode = CL_SUCCESS;
        size_t gvsize = 0; void* gvptr = nullptr;
        errorCode = clGetDeviceGlobalVariablePointerINTEL(
            device(), program(), HostAccessName, &gvsize, &gvptr);
        printf("clGetDeviceGlobalVariablePointerINTEL with %s returned %d: %p (size %zu)\n", HostAccessName, errorCode, gvptr, gvsize);

        gvsize = 0; gvptr = nullptr;
        errorCode = clGetDeviceGlobalVariablePointerINTEL(
            device(), program(), ExportName, &gvsize, &gvptr);
        printf("clGetDeviceGlobalVariablePointerINTEL with %s returned %d: %p (size %zu)\n", ExportName, errorCode, gvptr, gvsize);

        cl_unified_shared_memory_type_intel gvtype = 0;
        errorCode = clGetMemAllocInfoINTEL(
            context(), gvptr, CL_MEM_ALLOC_TYPE_INTEL, sizeof(gvtype), &gvtype, nullptr);

        printf("clGetMemAllocInfoINTEL returned %d: %04X\n", errorCode, gvtype);
    }

    typedef cl_int CL_API_CALL
    clEnqueueWriteGlobalVariableINTEL_t(
        cl_command_queue command_queue,
        cl_program program,
        const char* name,
        cl_bool blocking_write,
        size_t size,
        size_t offset,
        const void* ptr,
        cl_uint num_events_in_wait_list,
        const cl_event* event_wait_list,
        cl_event* event);

    typedef clEnqueueWriteGlobalVariableINTEL_t *
    clEnqueueWriteGlobalVariableINTEL_fn;

    auto clEnqueueWriteGlobalVariableINTEL = (clEnqueueWriteGlobalVariableINTEL_fn)
        clGetExtensionFunctionAddressForPlatform(platform(), "clEnqueueWriteGlobalVariableINTEL");

    if (clEnqueueWriteGlobalVariableINTEL == nullptr) {
        printf("Couldn't get function pointer for clEnqueueWriteGlobalVariableINTEL!\n");
    } else {
        const int value = 0xDEADBEEF;
        cl_int errorCode;

        errorCode = clEnqueueWriteGlobalVariableINTEL(
            commandQueue(),
            program(),
            HostAccessName,
            CL_TRUE,
            sizeof(value), 0, &value,
            0, nullptr, nullptr);
        printf("clEnqueueWriteGlobalVariableINTEL with %s to write %08X returned %d\n", HostAccessName, value, errorCode);

        errorCode = clEnqueueWriteGlobalVariableINTEL(
            commandQueue(),
            program(),
            ExportName,
            CL_TRUE,
            sizeof(value), 0, &value,
            0, nullptr, nullptr);
        printf("clEnqueueWriteGlobalVariableINTEL with %s to write %08X returned %d\n", ExportName, value, errorCode);
    }

    typedef cl_int CL_API_CALL
    clEnqueueReadGlobalVariableINTEL_t(
        cl_command_queue command_queue,
        cl_program program,
        const char* name,
        cl_bool blocking_read,
        size_t size,
        size_t offset,
        void* ptr,
        cl_uint num_events_in_wait_list,
        const cl_event* event_wait_list,
        cl_event* event);

    typedef clEnqueueReadGlobalVariableINTEL_t *
    clEnqueueReadGlobalVariableINTEL_fn;

    auto clEnqueueReadGlobalVariableINTEL = (clEnqueueReadGlobalVariableINTEL_fn)
        clGetExtensionFunctionAddressForPlatform(platform(), "clEnqueueReadGlobalVariableINTEL");

    if (clEnqueueReadGlobalVariableINTEL == nullptr) {
        printf("Couldn't get function pointer for clEnqueueReadGlobalVariableINTEL!\n");
    } else {
        int value;
        cl_int errorCode;

        value = -1;
        errorCode = clEnqueueReadGlobalVariableINTEL(
            commandQueue(),
            program(),
            HostAccessName,
            CL_TRUE,
            sizeof(value), 0, &value,
            0, nullptr, nullptr);
        printf("clEnqueueReadGlobalVariableINTEL with %s returned %d, read %08X\n", HostAccessName, errorCode, value);

        value = -1;
        errorCode = clEnqueueReadGlobalVariableINTEL(
            commandQueue(),
            program(),
            ExportName,
            CL_TRUE,
            sizeof(value), 0, &value,
            0, nullptr, nullptr);
        printf("clEnqueueReadGlobalVariableINTEL with %s returned %d, read %08X\n", ExportName, errorCode, value);
    }

    printf("Done.\n");
    return 0;
}
