/*
// Copyright (c) 2020-2025 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <popl/popl.hpp>

#include <CL/opencl.hpp>

// Each of these functions should eventually move into opencl.hpp:

static cl_unified_shared_memory_type_intel
getMEM_ALLOC_TYPE_INTEL( cl::Context& context, const void* ptr )
{
    cl_unified_shared_memory_type_intel type = 0;
    clGetMemAllocInfoINTEL(
        context(),
        ptr,
        CL_MEM_ALLOC_TYPE_INTEL,
        sizeof(type),
        &type,
        nullptr );
    return type;
}

static const void*
getMEM_ALLOC_BASE_PTR_INTEL( cl::Context& context, const void* ptr )
{
    const void* base = nullptr;
    clGetMemAllocInfoINTEL(
        context(),
        ptr,
        CL_MEM_ALLOC_BASE_PTR_INTEL,
        sizeof(base),
        &base,
        nullptr );
    return base;
}

static size_t
getMEM_ALLOC_SIZE_INTEL( cl::Context& context, const void* ptr )
{
    size_t size = 0;
    clGetMemAllocInfoINTEL(
        context(),
        ptr,
        CL_MEM_ALLOC_SIZE_INTEL,
        sizeof(size),
        &size,
        nullptr );
    return size;
}

static cl_device_id
getMEM_ALLOC_DEVICE_INTEL( cl::Context& context, const void* ptr )
{
    cl_device_id device = 0;
    clGetMemAllocInfoINTEL(
        context(),
        ptr,
        CL_MEM_ALLOC_DEVICE_INTEL,
        sizeof(device),
        &device,
        nullptr );
    return device;
}

static const char*
usm_type_to_string( cl_unified_shared_memory_type_intel type )
{
    switch( type )
    {
    case CL_MEM_TYPE_UNKNOWN_INTEL: return "CL_MEM_TYPE_UNKNOWN_INTEL";
    case CL_MEM_TYPE_HOST_INTEL:    return "CL_MEM_TYPE_HOST_INTEL";
    case CL_MEM_TYPE_DEVICE_INTEL:  return "CL_MEM_TYPE_DEVICE_INTEL";
    case CL_MEM_TYPE_SHARED_INTEL:  return "CL_MEM_TYPE_SHARED_INTEL";
    default: break;
    }
    return "***Unknown USM Type***";
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

    cl_device_unified_shared_memory_capabilities_intel usmcaps = 0;
    cl_int errCode;

    errCode = clGetDeviceInfo(
        devices[deviceIndex](),
        CL_DEVICE_HOST_MEM_CAPABILITIES_INTEL,
        sizeof(usmcaps),
        &usmcaps,
        nullptr );
    if( errCode == CL_SUCCESS && usmcaps != 0 )
    {
        printf("\nTesting Host Allocations:\n");
        char* ptr0 = (char*)clHostMemAllocINTEL(
            context(),
            nullptr,
            16,
            0,
            nullptr );
        printf("Allocated Host pointer 0: ptr = %p\n", ptr0);
        char* ptr1 = (char*)clHostMemAllocINTEL(
            context(),
            nullptr,
            16,
            0,
            nullptr );
        printf("Allocated Host pointer 1: ptr = %p\n", ptr1);

        cl_unified_shared_memory_type_intel type = 0;

        type = getMEM_ALLOC_TYPE_INTEL(context, ptr1);
        printf("Queried base pointer 1: type = %s (%X)\n", usm_type_to_string(type), type);

        type = getMEM_ALLOC_TYPE_INTEL(context, ptr1 + 4);
        printf("Queried offset pointer 1: type = %s (%X)\n", usm_type_to_string(type), type);

        type = getMEM_ALLOC_TYPE_INTEL(context, ptr1 + 64);
        printf("Queried out of range pointer 1: type = %s (%X)\n", usm_type_to_string(type), type);

        type = getMEM_ALLOC_TYPE_INTEL(context, ptr0);
        printf("Queried base pointer 0: type = %s (%X)\n", usm_type_to_string(type), type);

        type = getMEM_ALLOC_TYPE_INTEL(context, ptr0 + 4);
        printf("Queried offset pointer 0: type = %s (%X)\n", usm_type_to_string(type), type);

        type = getMEM_ALLOC_TYPE_INTEL(context, ptr0 + 64);
        printf("Queried out of range pointer 0: type = %s (%X)\n", usm_type_to_string(type), type);

        const void* base = getMEM_ALLOC_BASE_PTR_INTEL(context, ptr0 + 4);
        printf("Queried offset pointer 0: base = %p\n", base);

        size_t size = getMEM_ALLOC_SIZE_INTEL(context, ptr0 + 4);
        printf("Queried offset pointer 0: size = %u\n", (unsigned)size);

        cl_device_id device = getMEM_ALLOC_DEVICE_INTEL(context, ptr0 + 4);
        printf("Queried offset pointer 0: device = %p\n", device);

        clMemFreeINTEL(
            context(),
            ptr0 );
        clMemFreeINTEL(
            context(),
            ptr1 );
        printf("Freed pointers and done!\n");
    }
    else
    {
        printf("\nThis device does not support HOST allocations.\n");
    }

    errCode = clGetDeviceInfo(
        devices[deviceIndex](),
        CL_DEVICE_DEVICE_MEM_CAPABILITIES_INTEL,
        sizeof(usmcaps),
        &usmcaps,
        nullptr );
    if( errCode == CL_SUCCESS && usmcaps != 0 )
    {
        printf("\nTesting Device Allocations:\n");
        printf("Associated Device is: %p (%s)\n",
            devices[deviceIndex](),
            devices[deviceIndex].getInfo<CL_DEVICE_NAME>().c_str());
        char* ptr0 = (char*)clDeviceMemAllocINTEL(
            context(),
            devices[deviceIndex](),
            nullptr,
            16,
            0,
            nullptr );
        printf("Allocated Device pointer 0: ptr = %p\n", ptr0);
        char* ptr1 = (char*)clDeviceMemAllocINTEL(
            context(),
            devices[deviceIndex](),
            nullptr,
            16,
            0,
            nullptr );
        printf("Allocated Device pointer 1: ptr = %p\n", ptr1);

        cl_unified_shared_memory_type_intel type = 0;

        type = getMEM_ALLOC_TYPE_INTEL(context, ptr1);
        printf("Queried base pointer 1: type = %s (%X)\n", usm_type_to_string(type), type);

        type = getMEM_ALLOC_TYPE_INTEL(context, ptr1 + 4);
        printf("Queried offset pointer 1: type = %s (%X)\n", usm_type_to_string(type), type);

        type = getMEM_ALLOC_TYPE_INTEL(context, ptr1 + 64);
        printf("Queried out of range pointer 1: type = %s (%X)\n", usm_type_to_string(type), type);

        type = getMEM_ALLOC_TYPE_INTEL(context, ptr0);
        printf("Queried base pointer 0: type = %s (%X)\n", usm_type_to_string(type), type);

        type = getMEM_ALLOC_TYPE_INTEL(context, ptr0 + 4);
        printf("Queried offset pointer 0: type = %s (%X)\n", usm_type_to_string(type), type);

        type = getMEM_ALLOC_TYPE_INTEL(context, ptr0 + 64);
        printf("Queried out of range pointer 0: type = %s (%X)\n", usm_type_to_string(type), type);

        const void* base = getMEM_ALLOC_BASE_PTR_INTEL(context, ptr0 + 4);
        printf("Queried offset pointer 0: base = %p\n", base);

        size_t size = getMEM_ALLOC_SIZE_INTEL(context, ptr0 + 4);
        printf("Queried offset pointer 0: size = %u\n", (unsigned)size);

        cl_device_id device = getMEM_ALLOC_DEVICE_INTEL(context, ptr0 + 4);
        printf("Queried offset pointer 0: device = %p\n", device);

        clMemFreeINTEL(
            context(),
            ptr0 );
        clMemFreeINTEL(
            context(),
            ptr1 );
        printf("Freed pointers and done!\n");
    }
    else
    {
        printf("\nThis device does not support DEVICE allocations.\n");
    }

    errCode = clGetDeviceInfo(
        devices[deviceIndex](),
        CL_DEVICE_SINGLE_DEVICE_SHARED_MEM_CAPABILITIES_INTEL,
        sizeof(usmcaps),
        &usmcaps,
        nullptr );
    if( errCode == CL_SUCCESS && usmcaps != 0 )
    {
        printf("\nTesting Shared Allocations:\n");
        printf("Associated Device is: %p (%s)\n",
            devices[deviceIndex](),
            devices[deviceIndex].getInfo<CL_DEVICE_NAME>().c_str());
        char* ptr0 = (char*)clSharedMemAllocINTEL(
            context(),
            devices[deviceIndex](),
            nullptr,
            16,
            0,
            nullptr );
        printf("Allocated Shared pointer 0: ptr = %p\n", ptr0);
        char* ptr1 = (char*)clSharedMemAllocINTEL(
            context(),
            devices[deviceIndex](),
            nullptr,
            16,
            0,
            nullptr );
        printf("Allocated Shared pointer 1: ptr = %p\n", ptr1);

        cl_unified_shared_memory_type_intel type = 0;

        type = getMEM_ALLOC_TYPE_INTEL(context, ptr1);
        printf("Queried base pointer 1: type = %s (%X)\n", usm_type_to_string(type), type);

        type = getMEM_ALLOC_TYPE_INTEL(context, ptr1 + 4);
        printf("Queried offset pointer 1: type = %s (%X)\n", usm_type_to_string(type), type);

        type = getMEM_ALLOC_TYPE_INTEL(context, ptr1 + 64);
        printf("Queried out of range pointer 1: type = %s (%X)\n", usm_type_to_string(type), type);

        type = getMEM_ALLOC_TYPE_INTEL(context, ptr0);
        printf("Queried base pointer 0: type = %s (%X)\n", usm_type_to_string(type), type);

        type = getMEM_ALLOC_TYPE_INTEL(context, ptr0 + 4);
        printf("Queried offset pointer 0: type = %s (%X)\n", usm_type_to_string(type), type);

        type = getMEM_ALLOC_TYPE_INTEL(context, ptr0 + 64);
        printf("Queried out of range pointer 0: type = %s (%X)\n", usm_type_to_string(type), type);

        const void* base = getMEM_ALLOC_BASE_PTR_INTEL(context, ptr0 + 4);
        printf("Queried offset pointer 0: base = %p\n", base);

        size_t size = getMEM_ALLOC_SIZE_INTEL(context, ptr0 + 4);
        printf("Queried offset pointer 0: size = %u\n", (unsigned)size);

        cl_device_id device = getMEM_ALLOC_DEVICE_INTEL(context, ptr0 + 4);
        printf("Queried offset pointer 0: device = %p\n", device);

        clMemFreeINTEL(
            context(),
            ptr0 );
        clMemFreeINTEL(
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