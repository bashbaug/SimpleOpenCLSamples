/*
// Copyright (c) 2019-2026 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <popl/popl.hpp>

#include <CL/opencl.hpp>

#include "util.hpp"

const size_t    gwx = 1024*1024;

int main(int argc, char** argv)
{
    bool verbose = false;
    int platformIndex = 0;
    int deviceIndex = 0;

    {
        popl::OptionParser op("Supported Options");
        op.add<popl::Switch, popl::Attribute::advanced>("v", "verbose", "Verbose Output", &verbose);
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
                "Usage: copybuffer [options]\n"
                "%s", op.help().c_str());
            return -1;
        }
    }

    cl::Device device;
    if (!setupDevice(device, platformIndex, deviceIndex, verbose)) {
        return -1;
    }

    cl::Context context{device};
    cl::CommandQueue commandQueue{context, device};

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

    // execution
    commandQueue.enqueueCopyBuffer(
        deviceMemSrc,
        deviceMemDst,
        0,
        0,
        gwx * sizeof(cl_uint) );

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
