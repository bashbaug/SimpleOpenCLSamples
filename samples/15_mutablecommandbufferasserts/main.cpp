/*
// Copyright (c) 2022-2026 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <popl/popl.hpp>

#include <CL/opencl.hpp>

#include <cinttypes>

#include "util.hpp"

#if defined(cl_khr_command_buffer_mutable_dispatch)

#if !defined(CL_MUTABLE_DISPATCH_ASSERTS_KHR)
typedef cl_bitfield         cl_mutable_dispatch_asserts_khr;
#define CL_COMMAND_BUFFER_MUTABLE_DISPATCH_ASSERTS_KHR  0x12B7
#define CL_MUTABLE_DISPATCH_ASSERTS_KHR                 0x12B8
#define CL_MUTABLE_DISPATCH_ASSERT_NO_ADDITIONAL_WORK_GROUPS_KHR (1 << 0)
#endif // !defined(CL_MUTABLE_DISPATCH_ASSERTS_KHR)

const size_t    gwx = 1024;
const size_t    lwx = 16;

static const char kernelString[] = R"CLC(
kernel void CopyBuffer( global uint* dst, global uint* src )
{
    uint id = get_global_id(0);
    dst[id] = src[id];
}
)CLC";

int main(
    int argc,
    char** argv )
{
    int platformIndex = 0;
    int deviceIndex = 0;
    bool noCmdBufAssert = false;
    bool noCmdAssert = false;

    {
        popl::OptionParser op("Supported Options");
        op.add<popl::Value<int>>("p", "platform", "Platform Index", platformIndex, &platformIndex);
        op.add<popl::Value<int>>("d", "device", "Device Index", deviceIndex, &deviceIndex);
        op.add<popl::Switch>("", "noCmdBufAssert", "Skip Command Buffer Assert", &noCmdBufAssert);
        op.add<popl::Switch>("", "noCmdAssert", "Skip Command Assert", &noCmdAssert);

        bool printUsage = false;
        try {
            op.parse(argc, argv);
        } catch (std::exception& e) {
            fprintf(stderr, "Error: %s\n\n", e.what());
            printUsage = true;
        }

        if (printUsage || !op.unknown_options().empty() || !op.non_option_args().empty()) {
            fprintf(stderr,
                "Usage: mutablecommandbufferasserts [options]\n"
                "%s", op.help().c_str());
            return -1;
        }
    }

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    if (!checkPlatformIndex(platforms, platformIndex)) {
        return -1;
    }

    printf("Running on platform: %s\n",
        platforms[platformIndex].getInfo<CL_PLATFORM_NAME>().c_str() );

    std::vector<cl::Device> devices;
    platforms[platformIndex].getDevices(CL_DEVICE_TYPE_ALL, &devices);

    printf("Running on device: %s\n",
        devices[deviceIndex].getInfo<CL_DEVICE_NAME>().c_str() );

    // device queries:

    bool has_cl_khr_command_buffer =
        checkDeviceForExtension(devices[deviceIndex], CL_KHR_COMMAND_BUFFER_EXTENSION_NAME);
    if (has_cl_khr_command_buffer) {
        printf("Device supports " CL_KHR_COMMAND_BUFFER_EXTENSION_NAME ".\n");
    } else {
        printf("Device does not support " CL_KHR_COMMAND_BUFFER_EXTENSION_NAME ", exiting.\n");
        return -1;
    }
    bool has_cl_khr_command_buffer_mutable_dispatch =
        checkDeviceForExtension(devices[deviceIndex], CL_KHR_COMMAND_BUFFER_MUTABLE_DISPATCH_EXTENSION_NAME);
    if (has_cl_khr_command_buffer_mutable_dispatch) {
        printf("Device supports " CL_KHR_COMMAND_BUFFER_MUTABLE_DISPATCH_EXTENSION_NAME ".\n");
    } else {
        printf("Device does not support " CL_KHR_COMMAND_BUFFER_MUTABLE_DISPATCH_EXTENSION_NAME ", exiting.\n");
        return -1;
    }

    cl_mutable_dispatch_fields_khr mutableCaps = 0;
    clGetDeviceInfo(
        devices[deviceIndex](),
        CL_DEVICE_MUTABLE_DISPATCH_CAPABILITIES_KHR,
        sizeof(mutableCaps),
        &mutableCaps,
        NULL );
    if (!(mutableCaps & CL_MUTABLE_DISPATCH_GLOBAL_SIZE_KHR)) {
        printf("Device does not support modifying the global work size, exiting.\n");
        return -1;
    }

    printf("Adding Command Buffer Assert?  %s\n", noCmdBufAssert ? "No" : "Yes");
    printf("Adding Command Assert?  %s\n", noCmdAssert ? "No" : "Yes");

    cl::Context context{devices[deviceIndex]};
    cl::CommandQueue commandQueue{context, devices[deviceIndex]};

    cl::Program program{ context, kernelString };
    program.build();
    cl::Kernel kernel = cl::Kernel{ program, "CopyBuffer" };

    cl::Buffer deviceMemSrc = cl::Buffer{
        context,
        CL_MEM_ALLOC_HOST_PTR,
        gwx * sizeof( cl_uint ) };

    cl::Buffer deviceMemDst = cl::Buffer{
        context,
        CL_MEM_ALLOC_HOST_PTR,
        gwx * sizeof( cl_uint ) };

    // initialization
    {
        cl_uint*    pSrc = (cl_uint*)commandQueue.enqueueMapBuffer(
            deviceMemSrc,
            CL_TRUE,
            CL_MAP_WRITE_INVALIDATE_REGION,
            0,
            gwx * sizeof(cl_uint) );

        for( size_t i = 0; i < gwx; i++ )
        {
            pSrc[i] = (cl_uint)(i);
        }

        commandQueue.enqueueUnmapMemObject(
            deviceMemSrc,
            pSrc );
    }

    const cl_command_buffer_properties_khr cbprops[] = {
        CL_COMMAND_BUFFER_FLAGS_KHR,
        CL_COMMAND_BUFFER_MUTABLE_KHR,
        CL_COMMAND_BUFFER_MUTABLE_DISPATCH_ASSERTS_KHR,
        noCmdBufAssert
            ? 0
            : (cl_command_buffer_properties_khr)
                  CL_MUTABLE_DISPATCH_ASSERT_NO_ADDITIONAL_WORK_GROUPS_KHR,
        0,
    };
    cl_command_buffer_khr cmdbuf = clCreateCommandBufferKHR(
        1,
        &commandQueue(),
        cbprops,
        NULL);

    kernel.setArg(0, deviceMemDst);
    kernel.setArg(1, deviceMemSrc);
    const cl_command_properties_khr cmdprops[] = {
        CL_MUTABLE_DISPATCH_ASSERTS_KHR,
        noCmdAssert
            ? 0
            : (cl_command_properties_khr)
                  CL_MUTABLE_DISPATCH_ASSERT_NO_ADDITIONAL_WORK_GROUPS_KHR,
        0,
    };
    const size_t gwx_x2 = gwx * 2;
    cl_sync_point_khr sync_point;
    cl_mutable_command_khr command;
    clCommandNDRangeKernelKHR(
        cmdbuf,     // command_buffer
        NULL,       // command_queue - note NULL!
        cmdprops,   // properties
        kernel(),   // kernel
        1,          // work_dim
        NULL,       // global_work_offset
        &gwx_x2,    // global_work_size
        &lwx,       // local_work_size
        0,          // num_sync_points_in_wait_list
        NULL,       // sync_point_wait_list
        &sync_point,// sync_point
        &command);  // mutable_handle

    clFinalizeCommandBufferKHR(cmdbuf);

    // mutate the command buffer, adding work-groups.
    // This should generate an error with mutable dispatch asserts.
    {
        const size_t gwx_x4 = gwx * 4;
        cl_mutable_dispatch_config_khr dispatchConfig = {};
        dispatchConfig.command = command;
        dispatchConfig.global_work_size = &gwx_x4;

        const cl_uint updateCount = 1;
        const cl_command_buffer_update_type_khr updateTypes[updateCount] = {
            CL_STRUCTURE_TYPE_MUTABLE_DISPATCH_CONFIG_KHR,
        };
        const void* updateConfigs[updateCount] = {
            &dispatchConfig,
        };

        cl_int check = clUpdateMutableCommandsKHR(
            cmdbuf,
            updateCount,
            updateTypes,
            updateConfigs );
        printf("clUpdateMutableCommandsKHR() to increase work-groups returned %s.\n",
            check == CL_SUCCESS ? "SUCCESS" : "an ERROR");
    }

    // mutate the command buffer, reducing work-groups.
    // This should not generate an error even with mutable dispatch asserts.
    {
        cl_mutable_dispatch_config_khr dispatchConfig = {};
        dispatchConfig.command = command;
        dispatchConfig.global_work_size = &gwx;

        const cl_uint updateCount = 1;
        const cl_command_buffer_update_type_khr updateTypes[updateCount] = {
            CL_STRUCTURE_TYPE_MUTABLE_DISPATCH_CONFIG_KHR,
        };
        const void* updateConfigs[updateCount] = {
            &dispatchConfig,
        };

        cl_int check = clUpdateMutableCommandsKHR(
            cmdbuf,
            updateCount,
            updateTypes,
            updateConfigs );
        printf("clUpdateMutableCommandsKHR() to reduce work-groups returned %s.\n",
            check == CL_SUCCESS ? "SUCCESS" : "an ERROR");
    }

    clEnqueueCommandBufferKHR(
        0,
        NULL,
        cmdbuf,
        0,
        NULL,
        NULL);

    // verification
    {
        const cl_uint*  pDst = (const cl_uint*)commandQueue.enqueueMapBuffer(
            deviceMemDst,
            CL_TRUE,
            CL_MAP_READ,
            0,
            gwx * sizeof(cl_uint) );

        unsigned int    mismatches = 0;

        for( size_t i = 0; i < gwx; i++ )
        {
            if( pDst[i] != i )
            {
                if( mismatches < 16 )
                {
                    fprintf(stderr, "MisMatch!  dst[%d] == %08X, want %08X\n",
                        (unsigned int)i,
                        pDst[i],
                        (unsigned int)i );
                }
                mismatches++;
            }
        }

        if( mismatches )
        {
            fprintf(stderr, "Error: Found %d mismatches / %d values!!!\n",
                mismatches,
                (unsigned int)gwx );
        }
        else
        {
            printf("Success.\n");
        }

        commandQueue.enqueueUnmapMemObject(
            deviceMemDst,
            (void*)pDst );
    }

    clReleaseCommandBufferKHR(cmdbuf);

    return 0;
}

#else

#pragma message("mutablecommandbuffers: cl_khr_command_buffer_mutable_dispatch not found.  Please update your OpenCL headers.")

int main()
{
    printf("mutablecommandbuffers: cl_khr_command_buffer_mutable_dispatch not found.  Please update your OpenCL headers.\n");
    return 0;
};

#endif
