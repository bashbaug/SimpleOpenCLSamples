/*
// Copyright (c) 2020 Ben Ashbaugh
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

#include "libusm.h"

#if defined cl_intel_unified_shared_memory

static clHostMemAllocINTEL_fn pfn_clHostMemAllocINTEL = NULL;
CL_API_ENTRY void* CL_API_CALL
clHostMemAllocINTEL(
            cl_context context,
            const cl_mem_properties_intel* properties,
            size_t size,
            cl_uint alignment,
            cl_int* errcode_ret)
{
    if( pfn_clHostMemAllocINTEL )
    {
        return pfn_clHostMemAllocINTEL(
            context,
            properties,
            size,
            alignment,
            errcode_ret );
    }
    else
    {
        if( errcode_ret )
        {
            errcode_ret[0] = CL_INVALID_OPERATION;
        }
        return NULL;
    }
}

static clDeviceMemAllocINTEL_fn pfn_clDeviceMemAllocINTEL = NULL;
CL_API_ENTRY void* CL_API_CALL
clDeviceMemAllocINTEL(
            cl_context context,
            cl_device_id device,
			const cl_mem_properties_intel* properties,
            size_t size,
            cl_uint alignment,
            cl_int* errcode_ret)
{
    if( pfn_clDeviceMemAllocINTEL )
    {
        return pfn_clDeviceMemAllocINTEL(
            context,
            device,
            properties,
            size,
            alignment,
            errcode_ret );
    }
    else
    {
        if( errcode_ret )
        {
            errcode_ret[0] = CL_INVALID_OPERATION;
        }
        return NULL;
    }
}

static clSharedMemAllocINTEL_fn pfn_clSharedMemAllocINTEL = NULL;
CL_API_ENTRY void* CL_API_CALL
clSharedMemAllocINTEL(
            cl_context context,
            cl_device_id device,
            const cl_mem_properties_intel* properties,
            size_t size,
            cl_uint alignment,
            cl_int* errcode_ret)
{
    if( pfn_clSharedMemAllocINTEL )
    {
        return pfn_clSharedMemAllocINTEL(
            context,
            device,
            properties,
            size,
            alignment,
            errcode_ret );
    }
    else
    {
        if( errcode_ret )
        {
            errcode_ret[0] = CL_INVALID_OPERATION;
        }
        return NULL;
    }
}

static clMemFreeINTEL_fn pfn_clMemFreeINTEL = NULL;
CL_API_ENTRY cl_int CL_API_CALL
clMemFreeINTEL(
            cl_context context,
            void* ptr)
{
    if( pfn_clMemFreeINTEL )
    {
        return pfn_clMemFreeINTEL(
            context,
            ptr );
    }
    else
    {
        return CL_INVALID_OPERATION;
    }
}

static clMemBlockingFreeINTEL_fn pfn_clMemBlockingFreeINTEL  = NULL;
CL_API_ENTRY cl_int CL_API_CALL
clMemBlockingFreeINTEL(
            cl_context context,
            void* ptr)
{
    if( pfn_clMemBlockingFreeINTEL )
    {
        return pfn_clMemBlockingFreeINTEL(
            context,
            ptr );
    }
    else
    {
        return CL_INVALID_OPERATION;
    }
}

static clGetMemAllocInfoINTEL_fn pfn_clGetMemAllocInfoINTEL = NULL;
CL_API_ENTRY cl_int CL_API_CALL
clGetMemAllocInfoINTEL(
            cl_context context,
            const void* ptr,
            cl_mem_info_intel param_name,
            size_t param_value_size,
            void* param_value,
            size_t* param_value_size_ret)
{
    if( pfn_clGetMemAllocInfoINTEL )
    {
        return pfn_clGetMemAllocInfoINTEL(
            context,
            ptr,
            param_name,
            param_value_size,
            param_value,
            param_value_size_ret);
    }
    else
    {
        return CL_INVALID_OPERATION;
    }
}

static clSetKernelArgMemPointerINTEL_fn pfn_clSetKernelArgMemPointerINTEL = NULL;
CL_API_ENTRY cl_int CL_API_CALL
clSetKernelArgMemPointerINTEL(
            cl_kernel kernel,
            cl_uint arg_index,
            const void* arg_value)
{
    if( pfn_clSetKernelArgMemPointerINTEL )
    {
        return pfn_clSetKernelArgMemPointerINTEL(
            kernel,
            arg_index,
            arg_value );
    }
    else
    {
        return CL_INVALID_OPERATION;
    }
}

static clEnqueueMemsetINTEL_fn pfn_clEnqueueMemsetINTEL = NULL;
CL_API_ENTRY cl_int CL_API_CALL
clEnqueueMemsetINTEL(
            cl_command_queue queue,
            void* dst_ptr,
            cl_int value,
            size_t size,
            cl_uint num_events_in_wait_list,
            const cl_event* event_wait_list,
            cl_event* event)
{
    if( pfn_clEnqueueMemsetINTEL )
    {
        return pfn_clEnqueueMemsetINTEL(
            queue,
            dst_ptr,
            value,
            size,
            num_events_in_wait_list,
            event_wait_list,
            event );
    }
    else
    {
        return CL_INVALID_OPERATION;
    }
}

static clEnqueueMemFillINTEL_fn pfn_clEnqueueMemFillINTEL = NULL;
CL_API_ENTRY cl_int CL_API_CALL
clEnqueueMemFillINTEL(
            cl_command_queue queue,
            void* dst_ptr,
            const void* pattern,
            size_t pattern_size,
            size_t size,
            cl_uint num_events_in_wait_list,
            const cl_event* event_wait_list,
            cl_event* event)
{
    if( pfn_clEnqueueMemFillINTEL )
    {
        return pfn_clEnqueueMemFillINTEL(
            queue,
            dst_ptr,
            pattern,
            pattern_size,
            size,
            num_events_in_wait_list,
            event_wait_list,
            event );
    }
    else
    {
        return CL_INVALID_OPERATION;
    }
}

static clEnqueueMemcpyINTEL_fn pfn_clEnqueueMemcpyINTEL = NULL;
CL_API_ENTRY cl_int CL_API_CALL
clEnqueueMemcpyINTEL(
            cl_command_queue queue,
            cl_bool blocking,
            void* dst_ptr,
            const void* src_ptr,
            size_t size,
            cl_uint num_events_in_wait_list,
            const cl_event* event_wait_list,
            cl_event* event)
{
    if( pfn_clEnqueueMemcpyINTEL )
    {
        return pfn_clEnqueueMemcpyINTEL(
            queue,
            blocking,
            dst_ptr,
            src_ptr,
            size,
            num_events_in_wait_list,
            event_wait_list,
            event );
    }
    else
    {
        return CL_INVALID_OPERATION;
    }
}

static clEnqueueMigrateMemINTEL_fn pfn_clEnqueueMigrateMemINTEL = NULL;
CL_API_ENTRY cl_int CL_API_CALL
clEnqueueMigrateMemINTEL(
            cl_command_queue queue,
            const void* ptr,
            size_t size,
            cl_mem_migration_flags flags,
            cl_uint num_events_in_wait_list,
            const cl_event* event_wait_list,
            cl_event* event)
{
    if( pfn_clEnqueueMigrateMemINTEL )
    {
        return pfn_clEnqueueMigrateMemINTEL(
            queue,
            ptr,
            size,
            flags,
            num_events_in_wait_list,
            event_wait_list,
            event );
    }
    else
    {
        return CL_INVALID_OPERATION;
    }
}

static clEnqueueMemAdviseINTEL_fn pfn_clEnqueueMemAdviseINTEL = NULL;
CL_API_ENTRY cl_int CL_API_CALL
clEnqueueMemAdviseINTEL(
            cl_command_queue queue,
            const void* ptr,
            size_t size,
            cl_mem_advice_intel advice,
            cl_uint num_events_in_wait_list,
            const cl_event* event_wait_list,
            cl_event* event)
{
    if( pfn_clEnqueueMemAdviseINTEL )
    {
        return pfn_clEnqueueMemAdviseINTEL(
            queue,
            ptr,
            size,
            advice,
            num_events_in_wait_list,
            event_wait_list,
            event );
    }
    else
    {
        return CL_INVALID_OPERATION;
    }
}

#endif /* cl_intel_unified-shared_memory */

#ifdef __cplusplus
namespace libusm {
#endif

#define GET_EXTENSION( _funcname )                                      \
    pfn_ ## _funcname = ( _funcname ## _fn )                            \
        clGetExtensionFunctionAddressForPlatform(platform, #_funcname);

void initialize( cl_platform_id platform )
{
#if defined cl_intel_unified_shared_memory
    GET_EXTENSION( clHostMemAllocINTEL );
    GET_EXTENSION( clDeviceMemAllocINTEL );
    GET_EXTENSION( clSharedMemAllocINTEL );
    GET_EXTENSION( clMemFreeINTEL );
    GET_EXTENSION( clGetMemAllocInfoINTEL );
    GET_EXTENSION( clSetKernelArgMemPointerINTEL );
    GET_EXTENSION( clEnqueueMemsetINTEL );
    GET_EXTENSION( clEnqueueMemcpyINTEL );
    GET_EXTENSION( clEnqueueMigrateMemINTEL );
    GET_EXTENSION( clEnqueueMemAdviseINTEL );
#endif
}

#ifdef __cplusplus
}
#endif
