/*
// Copyright (c) 2023 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <popl/popl.hpp>

#include <CL/opencl.hpp>

// Each of these functions should eventually move into opencl.hpp:

static cl_svm_type_exp
getSVM_INFO_TYPE_EXP( cl::Context& context, const void* ptr )
{
    cl_svm_type_exp type = 0;
    clGetSVMInfoEXP(
        context(),
        ptr,
        CL_SVM_INFO_TYPE_EXP,
        sizeof(type),
        &type,
        nullptr );
    return type;
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

#define CASE_TO_STRING(_e) case _e: return #_e;

static const char* svm_type_to_string(cl_svm_type_exp type)
{
    switch (type) {
        CASE_TO_STRING(CL_SVM_TYPE_COARSE_GRAIN_BUFFER_EXP);
        CASE_TO_STRING(CL_SVM_TYPE_FINE_GRAIN_BUFFER_EXP);
        CASE_TO_STRING(CL_SVM_TYPE_FINE_GRAIN_BUFFER_WITH_ATOMICS_EXP);
        CASE_TO_STRING(CL_SVM_TYPE_FINE_GRAIN_SYSTEM_EXP);
        CASE_TO_STRING(CL_SVM_TYPE_HOST_EXP);
        CASE_TO_STRING(CL_SVM_TYPE_DEVICE_EXP);
        CASE_TO_STRING(CL_SVM_TYPE_SHARED_EXP);
    default: return "Unknown cl_svm_type_exp";
    }
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

    cl::Context context{devices[deviceIndex]};

    cl_device_svm_capabilities svmCaps =
        devices[deviceIndex].getInfo<CL_DEVICE_SVM_CAPABILITIES>();
    if( svmCaps & CL_DEVICE_SVM_HOST_ALLOC_EXP )
    {
        printf("\nTesting Host Allocations:\n");
        char* ptr0 = (char*)clSVMAllocWithPropertiesEXP(
            context(),
            nullptr,
            CL_SVM_TYPE_HOST_EXP,
            CL_MEM_READ_WRITE,
            16,
            0,
            nullptr );
        printf("Allocated Host pointer 0: ptr = %p\n", ptr0);
        char* ptr1 = (char*)clSVMAllocWithPropertiesEXP(
            context(),
            nullptr,
            CL_SVM_TYPE_HOST_EXP,
            CL_MEM_READ_WRITE,
            16,
            0,
            nullptr );
        printf("Allocated Host pointer 1: ptr = %p\n", ptr1);

        cl_svm_type_exp type = 0;

        type = getSVM_INFO_TYPE_EXP(context, ptr1);
        printf("Queried base pointer 1: type = %s (%X)\n", svm_type_to_string(type), type);

        type = getSVM_INFO_TYPE_EXP(context, ptr1 + 4);
        printf("Queried offset pointer 1: type = %s (%X)\n", svm_type_to_string(type), type);

        type = getSVM_INFO_TYPE_EXP(context, ptr1 + 64);
        printf("Queried out of range pointer 1: type = %s (%X)\n", svm_type_to_string(type), type);

        type = getSVM_INFO_TYPE_EXP(context, ptr0);
        printf("Queried base pointer 0: type = %s (%X)\n", svm_type_to_string(type), type);

        type = getSVM_INFO_TYPE_EXP(context, ptr0 + 4);
        printf("Queried offset pointer 0: type = %s (%X)\n", svm_type_to_string(type), type);

        type = getSVM_INFO_TYPE_EXP(context, ptr0 + 64);
        printf("Queried out of range pointer 0: type = %s (%X)\n", svm_type_to_string(type), type);

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
    else
    {
        printf("\nThis device does not support HOST allocations.\n");
    }

    if( svmCaps & CL_DEVICE_SVM_DEVICE_ALLOC_EXP )
    {
        printf("\nTesting Device Allocations:\n");
        printf("Associated Device is: %p (%s)\n",
            devices[deviceIndex](),
            devices[deviceIndex].getInfo<CL_DEVICE_NAME>().c_str());
        const cl_svm_alloc_properties_exp props[] = {
            CL_SVM_ALLOC_ASSOCIATED_DEVICE_HANDLE_EXP, (cl_svm_alloc_properties_exp)devices[deviceIndex](),
            0,
        };
        char* ptr0 = (char*)clSVMAllocWithPropertiesEXP(
            context(),
            props,
            CL_SVM_TYPE_DEVICE_EXP,
            CL_MEM_READ_WRITE,
            16,
            0,
            nullptr );
        printf("Allocated Device pointer 0: ptr = %p\n", ptr0);
        char* ptr1 = (char*)clSVMAllocWithPropertiesEXP(
            context(),
            props,
            CL_SVM_TYPE_DEVICE_EXP,
            CL_MEM_READ_WRITE,
            16,
            0,
            nullptr );
        printf("Allocated Device pointer 1: ptr = %p\n", ptr1);

        cl_svm_type_exp type = 0;

        type = getSVM_INFO_TYPE_EXP(context, ptr1);
        printf("Queried base pointer 1: type = %s (%X)\n", svm_type_to_string(type), type);

        type = getSVM_INFO_TYPE_EXP(context, ptr1 + 4);
        printf("Queried offset pointer 1: type = %s (%X)\n", svm_type_to_string(type), type);

        type = getSVM_INFO_TYPE_EXP(context, ptr1 + 64);
        printf("Queried out of range pointer 1: type = %s (%X)\n", svm_type_to_string(type), type);

        type = getSVM_INFO_TYPE_EXP(context, ptr0);
        printf("Queried base pointer 0: type = %s (%X)\n", svm_type_to_string(type), type);

        type = getSVM_INFO_TYPE_EXP(context, ptr0 + 4);
        printf("Queried offset pointer 0: type = %s (%X)\n", svm_type_to_string(type), type);

        type = getSVM_INFO_TYPE_EXP(context, ptr0 + 64);
        printf("Queried out of range pointer 0: type = %s (%X)\n", svm_type_to_string(type), type);

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
    else
    {
        printf("\nThis device does not support DEVICE allocations.\n");
    }

    if( svmCaps & CL_DEVICE_SVM_SHARED_ALLOC_EXP )
    {
        printf("\nTesting Shared Allocations:\n");
        printf("Associated Device is: %p (%s)\n",
            devices[deviceIndex](),
            devices[deviceIndex].getInfo<CL_DEVICE_NAME>().c_str());
        const cl_svm_alloc_properties_exp props[] = {
            CL_SVM_ALLOC_ASSOCIATED_DEVICE_HANDLE_EXP, (cl_svm_alloc_properties_exp)devices[deviceIndex](),
            0,
        };
        char* ptr0 = (char*)clSVMAllocWithPropertiesEXP(
            context(),
            props,
            CL_SVM_TYPE_SHARED_EXP,
            CL_MEM_READ_WRITE,
            16,
            0,
            nullptr );
        printf("Allocated Shared pointer 0: ptr = %p\n", ptr0);
        char* ptr1 = (char*)clSVMAllocWithPropertiesEXP(
            context(),
            props,
            CL_SVM_TYPE_SHARED_EXP,
            CL_MEM_READ_WRITE,
            16,
            0,
            nullptr );
        printf("Allocated Shared pointer 1: ptr = %p\n", ptr1);

        cl_svm_type_exp type = 0;

        type = getSVM_INFO_TYPE_EXP(context, ptr1);
        printf("Queried base pointer 1: type = %s (%X)\n", svm_type_to_string(type), type);

        type = getSVM_INFO_TYPE_EXP(context, ptr1 + 4);
        printf("Queried offset pointer 1: type = %s (%X)\n", svm_type_to_string(type), type);

        type = getSVM_INFO_TYPE_EXP(context, ptr1 + 64);
        printf("Queried out of range pointer 1: type = %s (%X)\n", svm_type_to_string(type), type);

        type = getSVM_INFO_TYPE_EXP(context, ptr0);
        printf("Queried base pointer 0: type = %s (%X)\n", svm_type_to_string(type), type);

        type = getSVM_INFO_TYPE_EXP(context, ptr0 + 4);
        printf("Queried offset pointer 0: type = %s (%X)\n", svm_type_to_string(type), type);

        type = getSVM_INFO_TYPE_EXP(context, ptr0 + 64);
        printf("Queried out of range pointer 0: type = %s (%X)\n", svm_type_to_string(type), type);

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
    else
    {
        printf("\nThis device does not support SHARED allocations.\n");
    }

    return 0;
}
