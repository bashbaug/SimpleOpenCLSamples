/*
// Copyright (c) 2019-2025 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <popl/popl.hpp>

#include <CL/opencl.hpp>

#include <cinttypes>

#include "util.hpp"

#if !defined(cl_ext_buffer_device_address)
#define cl_ext_buffer_device_address 1
#define CL_EXT_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME \
    "cl_ext_buffer_device_address"

typedef cl_ulong cl_mem_device_address_ext;

typedef cl_int CL_API_CALL
clSetKernelArgDevicePointerEXT_t(
    cl_kernel kernel,
    cl_uint arg_index,
    cl_mem_device_address_ext arg_value);

typedef clSetKernelArgDevicePointerEXT_t *
clSetKernelArgDevicePointerEXT_fn ;    

#define CL_MEM_DEVICE_PRIVATE_ADDRESS_EXT           0x5000
#define CL_MEM_DEVICE_ADDRESS_EXT                   0x5001
#define CL_KERNEL_EXEC_INFO_DEVICE_PTRS_EXT         0x5002
#endif

// temp
namespace cl {
namespace detail {
CL_HPP_DECLARE_PARAM_TRAITS_(cl_mem_info, CL_MEM_DEVICE_ADDRESS_EXT, cl_mem_device_address_ext);
}
}

const size_t    gwx = 1024*1024;

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
                "Usage: bufferdeviceaddress [options]\n"
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

    bool has_cl_ext_buffer_device_address =
        checkDeviceForExtension(devices[deviceIndex], CL_EXT_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
    if (has_cl_ext_buffer_device_address) {
        printf("Device supports " CL_EXT_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME ".\n");
    } else {
        printf("Device does not support " CL_EXT_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME ", exiting.\n");
        return -1;
    }

    auto clSetKernelArgDevicePointerEXT = (clSetKernelArgDevicePointerEXT_fn)
        clGetExtensionFunctionAddressForPlatform(
            platforms[platformIndex](), "clSetKernelArgDevicePointerEXT");

    cl::Context context{devices[deviceIndex]};
    cl::CommandQueue commandQueue{context, devices[deviceIndex]};

    cl::Program program{ context, kernelString };
    program.build();
    cl::Kernel kernel = cl::Kernel{ program, "CopyBuffer" };

    std::vector<cl_mem_properties> props = {
        CL_MEM_DEVICE_PRIVATE_ADDRESS_EXT, CL_TRUE,
        0,
    };
    cl::Buffer deviceMemSrc = cl::Buffer{
        context,
        props,
        CL_MEM_ALLOC_HOST_PTR,
        gwx * sizeof( cl_uint ) };

    cl::Buffer deviceMemDst = cl::Buffer{
        context,
        props,
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

    cl_ulong    deviceSrcAddr = deviceMemSrc.getInfo<CL_MEM_DEVICE_ADDRESS_EXT>();
    cl_ulong    deviceDstAddr = deviceMemDst.getInfo<CL_MEM_DEVICE_ADDRESS_EXT>();
    printf("Src buffer device address: %08" PRIx64 "\n", deviceSrcAddr);
    printf("Dst buffer device address: %08" PRIx64 "\n", deviceDstAddr);

    // execution
    clSetKernelArgDevicePointerEXT(kernel(), 0, deviceDstAddr);
    clSetKernelArgDevicePointerEXT(kernel(), 1, deviceSrcAddr);
    commandQueue.enqueueNDRangeKernel(
        kernel,
        cl::NullRange,
        cl::NDRange{gwx} );

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

    return 0;
}
