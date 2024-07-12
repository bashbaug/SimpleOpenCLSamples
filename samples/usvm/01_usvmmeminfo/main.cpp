/*
// Copyright (c) 2023 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <popl/popl.hpp>

#include <CL/opencl.hpp>

#include <cinttypes>

// Each of these functions should eventually move into opencl.hpp:

static cl_svm_capabilities_exp
getSVM_INFO_CAPABILITIES_EXP( cl::Context& context, const void* ptr )
{
    cl_svm_capabilities_exp caps = 0;
    clGetSVMInfoEXP(
        context(),
        ptr,
        CL_SVM_INFO_CAPABILITIES_EXP,
        sizeof(caps),
        &caps,
        nullptr );
    return caps;
}

static const void*
getSVM_INFO_BASE_PTR_EXP( cl::Context& context, const void* ptr )
{
    const void* base = nullptr;
    clGetSVMInfoEXP(
        context(),
        ptr,
        CL_SVM_INFO_BASE_PTR_EXP,
        sizeof(base),
        &base,
        nullptr );
    return base;
}

static size_t
getSVM_INFO_SIZE_EXP( cl::Context& context, const void* ptr )
{
    size_t size = 0;
    clGetSVMInfoEXP(
        context(),
        ptr,
        CL_SVM_INFO_SIZE_EXP,
        sizeof(size),
        &size,
        nullptr );
    return size;
}

static cl_device_id
getSVM_INFO_ASSOCIATED_DEVICE_HANDLE_EXP( cl::Context& context, const void* ptr )
{
    cl_device_id device = 0;
    clGetSVMInfoEXP(
        context(),
        ptr,
        CL_SVM_INFO_ASSOCIATED_DEVICE_HANDLE_EXP,
        sizeof(device),
        &device,
        nullptr );
    return device;
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
                "Usage: usmmeminfo [options]\n"
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
    printf("\nDevice handle: %p\n", devices[deviceIndex]());

    cl::Context context{devices[deviceIndex]};

    std::vector<cl_device_svm_type_capabilities_exp> usmTypes =
        devices[deviceIndex].getInfo<CL_DEVICE_SVM_TYPE_CAPABILITIES_EXP>();
    for (const auto& type : usmTypes) {
        printf("\nTesting Allocations with caps %016" PRIx64 ":\n", type.capabilities);
        const cl_svm_alloc_properties_exp associatedDeviceProps[] = {
            CL_SVM_ALLOC_ASSOCIATED_DEVICE_HANDLE_EXP, (cl_svm_alloc_properties_exp)devices[deviceIndex](),
            0,
        };
        const cl_svm_alloc_properties_exp* props = nullptr;
        if (type.capabilities & ~CL_SVM_CAPABILITY_DEVICE_UNASSOCIATED_EXP) {
            props = associatedDeviceProps;
        }
        char* ptr0 = (char*)clSVMAllocWithPropertiesEXP(
            context(),
            props,
            type.capabilities,
            CL_MEM_READ_WRITE,
            16,
            0,
            nullptr );
        printf("Allocated pointer 0: ptr = %p\n", ptr0);
        char* ptr1 = (char*)clSVMAllocWithPropertiesEXP(
            context(),
            props,
            type.capabilities,
            CL_MEM_READ_WRITE,
            16,
            0,
            nullptr );
        printf("Allocated pointer 1: ptr = %p\n", ptr1);

        cl_svm_capabilities_exp caps = 0;

        caps = getSVM_INFO_CAPABILITIES_EXP(context, ptr0);
        printf("Queried base pointer 0: caps = %016" PRIx64 "\n", caps);

        caps = getSVM_INFO_CAPABILITIES_EXP(context, ptr0 + 4);
        printf("Queried offset pointer 0: caps = %016" PRIx64 "\n", caps);

        caps = getSVM_INFO_CAPABILITIES_EXP(context, ptr0 + 64);
        printf("Queried out of range pointer 0: caps = %016" PRIx64 "\n", caps);

        caps = getSVM_INFO_CAPABILITIES_EXP(context, ptr1);
        printf("Queried base pointer 1: caps = %016" PRIx64 "\n", caps);

        caps = getSVM_INFO_CAPABILITIES_EXP(context, ptr1 + 4);
        printf("Queried offset pointer 1: caps = %016" PRIx64 "\n", caps);

        caps = getSVM_INFO_CAPABILITIES_EXP(context, ptr1 + 64);
        printf("Queried out of range pointer 1: caps = %016" PRIx64 "\n", caps);

        const void* base = getSVM_INFO_BASE_PTR_EXP(context, ptr0 + 4);
        printf("Queried offset pointer 0: base = %p\n", base);

        size_t size = getSVM_INFO_SIZE_EXP(context, ptr0 + 4);
        printf("Queried offset pointer 0: size = %u\n", (unsigned)size);

        cl_device_id device = getSVM_INFO_ASSOCIATED_DEVICE_HANDLE_EXP(context, ptr0 + 4);
        printf("Queried offset pointer 0: device = %p\n", device);

        clSVMFree(
            context(),
            ptr0 );
        clSVMFree(
            context(),
            ptr1 );
        printf("Freed pointers and done!\n");
    }

    return 0;
}
