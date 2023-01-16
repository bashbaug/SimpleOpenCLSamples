/*
// Copyright (c) 2019-2021 Ben Ashbaugh
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

#include "util.hpp"

#ifndef CL_KHR_SEMAPHORE_EXTENSION_NAME
#define CL_KHR_SEMAPHORE_EXTENSION_NAME "cl_khr_semaphore"
#endif

static const char kernelString[] = R"CLC(
kernel void Add1( global uint* dst, global uint* src )
{
    uint id = get_global_id(0);
    dst[id] = src[id] + 1;
}
)CLC";

static void PrintSemaphoreTypes(const std::vector<cl_semaphore_type_khr>& types)
{
    for (auto type : types) {
        switch(type) {
        case CL_SEMAPHORE_TYPE_BINARY_KHR:
            printf("\t\tCL_SEMAPHORE_TYPE_BINARY_KHR\n");
            break;
        default:
            printf("\t\t(Unknown semaphore type: %08X)\n", type);
            break;
        }
    }
}

int main(
    int argc,
    char** argv )
{
    int platformIndex = 0;
    int deviceIndex = 0;

    size_t gwx = 512;

    {
        bool advanced = false;

        popl::OptionParser op("Supported Options");
        op.add<popl::Value<int>>("p", "platform", "Platform Index", platformIndex, &platformIndex);
        op.add<popl::Value<int>>("d", "device", "Device Index", deviceIndex, &deviceIndex);
        op.add<popl::Value<size_t>>("", "gwx", "Global Work Size", gwx, &gwx);
        bool printUsage = false;
        try {
            op.parse(argc, argv);
        } catch (std::exception& e) {
            fprintf(stderr, "Error: %s\n\n", e.what());
            printUsage = true;
        }

        if (printUsage || !op.unknown_options().empty() || !op.non_option_args().empty()) {
            fprintf(stderr,
                "Usage: semaphores [options]\n"
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

    // device queries:

    bool has_cl_khr_command_buffer =
        checkDeviceForExtension(devices[deviceIndex], CL_KHR_SEMAPHORE_EXTENSION_NAME);
    if (has_cl_khr_command_buffer) {
        printf("Device supports " CL_KHR_SEMAPHORE_EXTENSION_NAME ".\n");
    } else {
        printf("Device does not support " CL_KHR_SEMAPHORE_EXTENSION_NAME ", exiting.\n");
        return -1;
    }

    std::vector<cl_semaphore_type_khr> platformSemaphoreTypes =
        platforms[platformIndex].getInfo<CL_PLATFORM_SEMAPHORE_TYPES_KHR>();
    printf("\tPlatform Semaphore Types:\n");
    PrintSemaphoreTypes(platformSemaphoreTypes);

    std::vector<cl_semaphore_type_khr> deviceSemaphoreTypes =
        devices[deviceIndex].getInfo<CL_DEVICE_SEMAPHORE_TYPES_KHR>();
    printf("\tDevice Semaphore Types:\n");
    PrintSemaphoreTypes(deviceSemaphoreTypes);

    cl::Context context{devices[deviceIndex]};
    cl::CommandQueue q0 = cl::CommandQueue{context, devices[deviceIndex]};
    cl::CommandQueue q1 = cl::CommandQueue{context, devices[deviceIndex]};

    cl::Program program{ context, kernelString };
    program.build();
    cl::Kernel kernel{ program, "Add1" };

    cl::Buffer b0 = cl::Buffer{
        context,
        CL_MEM_ALLOC_HOST_PTR,
        gwx * sizeof( cl_uint ) };
    cl::Buffer b1 = cl::Buffer{
        context,
        CL_MEM_ALLOC_HOST_PTR,
        gwx * sizeof( cl_uint ) };

    cl_semaphore_properties_khr semaphoreProperties[] = {
        CL_SEMAPHORE_TYPE_KHR,
        CL_SEMAPHORE_TYPE_BINARY_KHR,
        0,
    };
    cl_semaphore_khr semaphore = clCreateSemaphoreWithPropertiesKHR(
        context(),
        semaphoreProperties,
        NULL );

    cl_uint pattern = 0;
    q0.enqueueFillBuffer(b0, pattern, 0, gwx * sizeof(cl_uint));
    q0.finish();

    // execution

    kernel.setArg(0, b1);   // dst
    kernel.setArg(1, b0);   // src
    q0.enqueueNDRangeKernel( kernel, cl::NullRange, cl::NDRange{gwx} );
    clEnqueueSignalSemaphoresKHR(q0(), 1, &semaphore, NULL, 0, NULL, NULL);

    clEnqueueWaitSemaphoresKHR(q1(), 1, &semaphore, NULL, 0, NULL, NULL);
    kernel.setArg(0, b0);   // dst
    kernel.setArg(1, b1);   // src
    q1.enqueueNDRangeKernel( kernel, cl::NullRange, cl::NDRange{gwx} );
    clEnqueueSignalSemaphoresKHR(q1(), 1, &semaphore, NULL, 0, NULL, NULL);

    clEnqueueWaitSemaphoresKHR(q1(), 1, &semaphore, NULL, 0, NULL, NULL);
    kernel.setArg(0, b1);   // dst
    kernel.setArg(1, b0);   // src
    q1.enqueueNDRangeKernel( kernel, cl::NullRange, cl::NDRange{gwx} );

    q0.flush();
    q1.finish();

    // semaphore queries:
    {
        printf("\tSemaphore Info:\n");

        cl_semaphore_type_khr semaphoreType = 0;
        clGetSemaphoreInfoKHR(
            semaphore,
            CL_SEMAPHORE_TYPE_KHR,
            sizeof(semaphoreType),
            &semaphoreType,
            NULL );
        printf("\t\tCL_SEMAPHORE_TYPE_KHR: %u\n", semaphoreType);

        cl_context testContext = NULL;
        clGetSemaphoreInfoKHR(
            semaphore,
            CL_SEMAPHORE_CONTEXT_KHR,
            sizeof(testContext),
            &testContext,
            NULL );
        printf("\t\tCL_SEMAPHORE_CONTEXT_KHR: %p (%s)\n",
            testContext,
            testContext == context() ? "matches" : "MISMATCH!");

        cl_uint refCount = 0;
        clGetSemaphoreInfoKHR(
            semaphore,
            CL_SEMAPHORE_REFERENCE_COUNT_KHR,
            sizeof(refCount),
            &refCount,
            NULL );
        printf("\t\tCL_SEMAPHORE_REFERENCE_COUNT_KHR: %u\n", refCount);

        clReleaseSemaphoreKHR(semaphore);
        semaphore = NULL;
    }

    // verify results by printing the first few values
    if (gwx > 3) {
        auto ptr = (const cl_uint*)q1.enqueueMapBuffer(
            b1,
            CL_TRUE,
            CL_MAP_READ,
            0,
            gwx * sizeof( cl_uint ) );

        printf("First few values: [0] = %u, [1] = %u, [2] = %u\n", ptr[0], ptr[1], ptr[2]);

        q1.enqueueUnmapMemObject(
            b1,
            (void*)ptr );
        q1.finish();
    }

    printf("Done.\n");

    return 0;
}
