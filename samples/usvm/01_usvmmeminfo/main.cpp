/*
// Copyright (c) 2023 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <popl/popl.hpp>

#include <CL/opencl.hpp>

#include <cinttypes>

static inline bool has_all_svm_caps(
    cl_svm_capabilities_khr caps,
    cl_svm_capabilities_khr check)
{
    return (caps & check) == check;
}

static const char* get_svm_name(cl_svm_capabilities_khr svmcaps)
{
    if (has_all_svm_caps(svmcaps, CL_SVM_TYPE_MACRO_SYSTEM_KHR)) {
        return "SYSTEM";
    }
    if (has_all_svm_caps(svmcaps, CL_SVM_TYPE_MACRO_FINE_GRAIN_BUFFER_KHR)) {
        return "FINE_GRAIN_BUFFER";
    }
    if (has_all_svm_caps(svmcaps, CL_SVM_TYPE_MACRO_COARSE_GRAIN_BUFFER_KHR)) {
        return "COARSE_GRAIN_BUFFER";
    }
    if (has_all_svm_caps(svmcaps, CL_SVM_TYPE_MACRO_HOST_KHR)) {
        return "HOST";
    }
    if (has_all_svm_caps(svmcaps, CL_SVM_TYPE_MACRO_DEVICE_KHR)) {
        return "DEVICE";
    }
    if (has_all_svm_caps(svmcaps, CL_SVM_TYPE_MACRO_SINGLE_DEVICE_SHARED_KHR)) {
        return "SINGLE_DEVICE_SHARED";
    }
    return "*** UNKNOWN! ***";
}

// Each of these functions should eventually move into opencl.hpp:

static cl_uint
getSVM_INFO_TYPE_INDEX_KHR( cl::Context& context, const void* ptr )
{
    cl_uint index = 0;
    clGetSVMPointerInfoKHR(
        context(),
        nullptr,
        ptr,
        CL_SVM_INFO_TYPE_INDEX_KHR,
        sizeof(index),
        &index,
        nullptr );
    return index;
}

static const void*
getSVM_INFO_BASE_PTR_KHR( cl::Context& context, const void* ptr )
{
    const void* base = nullptr;
    clGetSVMPointerInfoKHR(
        context(),
        nullptr,
        ptr,
        CL_SVM_INFO_BASE_PTR_KHR,
        sizeof(base),
        &base,
        nullptr );
    return base;
}

static size_t
getSVM_INFO_SIZE_KHR( cl::Context& context, const void* ptr )
{
    size_t size = 0;
    clGetSVMPointerInfoKHR(
        context(),
        nullptr,
        ptr,
        CL_SVM_INFO_SIZE_KHR,
        sizeof(size),
        &size,
        nullptr );
    return size;
}

static cl_device_id
getSVM_INFO_ASSOCIATED_DEVICE_HANDLE_KHR( cl::Context& context, const void* ptr )
{
    cl_device_id device = 0;
    clGetSVMPointerInfoKHR(
        context(),
        nullptr,
        ptr,
        CL_SVM_INFO_ASSOCIATED_DEVICE_HANDLE_KHR,
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

    std::vector<cl_svm_capabilities_khr> typeCaps =
        devices[deviceIndex].getInfo<CL_DEVICE_SVM_TYPE_CAPABILITIES_KHR>();
    for (size_t index = 0; index < typeCaps.size(); index++) {
        cl_svm_capabilities_khr caps = typeCaps[index];

        const size_t cAllocSize = 16;
        printf("\nTesting %s (%016" PRIx64 "), alloc size = %zu:\n",
            get_svm_name(caps),
            caps,
            cAllocSize);
        if (caps & CL_SVM_CAPABILITY_SYSTEM_ALLOCATED_KHR) {
            printf("SVM type is system allocated, skipping...\n");
            continue;
        }

        const cl_svm_alloc_properties_khr associatedDeviceProps[] = {
            CL_SVM_ALLOC_ASSOCIATED_DEVICE_HANDLE_KHR, (cl_svm_alloc_properties_khr)devices[deviceIndex](),
            0,
        };
        const cl_svm_alloc_properties_khr* props = nullptr;
        if (!(caps & CL_SVM_CAPABILITY_DEVICE_UNASSOCIATED_KHR)) {
            props = associatedDeviceProps;
        }
        char* ptr0 = (char*)clSVMAllocWithPropertiesKHR(
            context(),
            props,
            static_cast<cl_uint>(index),
            cAllocSize,
            nullptr );
        printf("Allocated pointer 0: ptr = %p\n", ptr0);
        char* ptr1 = (char*)clSVMAllocWithPropertiesKHR(
            context(),
            props,
            static_cast<cl_uint>(index),
            cAllocSize,
            nullptr );
        printf("Allocated pointer 1: ptr = %p\n", ptr1);

        cl_uint qi = 0;

        qi = getSVM_INFO_TYPE_INDEX_KHR(context, ptr0);
        printf("Queried index for base pointer 0: %u\n", qi);

        qi = getSVM_INFO_TYPE_INDEX_KHR(context, ptr0 + 4);
        printf("Queried index for offset pointer 0: %u\n", qi);

        qi = getSVM_INFO_TYPE_INDEX_KHR(context, ptr0 + 64);
        printf("Queried index for out of range pointer 0: %u\n", qi);

        qi = getSVM_INFO_TYPE_INDEX_KHR(context, ptr1);
        printf("Queried index for base pointer 1: %u\n", qi);

        qi = getSVM_INFO_TYPE_INDEX_KHR(context, ptr1 + 4);
        printf("Queried index for offset pointer 1: %u\n", qi);

        qi = getSVM_INFO_TYPE_INDEX_KHR(context, ptr1 + 64);
        printf("Queried index for out of range pointer 1: %u\n", qi);

        const void* base = getSVM_INFO_BASE_PTR_KHR(context, ptr0 + 4);
        printf("Queried base address for offset pointer 0: %p\n", base);

        size_t size = getSVM_INFO_SIZE_KHR(context, ptr0 + 4);
        printf("Queried size for offset pointer 0: %zu\n", size);

        cl_device_id device = getSVM_INFO_ASSOCIATED_DEVICE_HANDLE_KHR(context, ptr0 + 4);
        printf("Queried device id for offset pointer 0: %p\n", device);

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
