/*
// Copyright (c) 2023 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <CL/cl.h>

#include "scope_profiler.h"

extern const struct _cl_icd_dispatch* g_pNextDispatch;

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clGetPlatformIDs_layer(
    cl_uint num_entries,
    cl_platform_id* platforms,
    cl_uint* num_platforms)
{
    PROFILE_SCOPE("clGetPlatformIDs");
    return g_pNextDispatch->clGetPlatformIDs(
        num_entries,
        platforms,
        num_platforms);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clGetPlatformInfo_layer(
    cl_platform_id platform,
    cl_platform_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    PROFILE_SCOPE("clGetPlatformInfo");
    return g_pNextDispatch->clGetPlatformInfo(
        platform,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clGetDeviceIDs_layer(
    cl_platform_id platform,
    cl_device_type device_type,
    cl_uint num_entries,
    cl_device_id* devices,
    cl_uint* num_devices)
{
    PROFILE_SCOPE("clGetDeviceIDs");
    return g_pNextDispatch->clGetDeviceIDs(
        platform,
        device_type,
        num_entries,
        devices,
        num_devices);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clGetDeviceInfo_layer(
    cl_device_id device,
    cl_device_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    PROFILE_SCOPE("clGetDeviceInfo");
    return g_pNextDispatch->clGetDeviceInfo(
        device,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
}

///////////////////////////////////////////////////////////////////////////////

static cl_context CL_API_CALL clCreateContext_layer(
    const cl_context_properties* properties,
    cl_uint num_devices,
    const cl_device_id* devices,
    void (CL_CALLBACK* pfn_notify)(const char* errinfo, const void* private_info, size_t cb, void* user_data),
    void* user_data,
    cl_int* errcode_ret)
{
    PROFILE_SCOPE("clCreateContext");
    return g_pNextDispatch->clCreateContext(
        properties,
        num_devices,
        devices,
        pfn_notify,
        user_data,
        errcode_ret);
}

///////////////////////////////////////////////////////////////////////////////

static cl_context CL_API_CALL clCreateContextFromType_layer(
    const cl_context_properties* properties,
    cl_device_type device_type,
    void (CL_CALLBACK* pfn_notify)(const char* errinfo, const void* private_info, size_t cb, void* user_data),
    void* user_data,
    cl_int* errcode_ret)
{
    PROFILE_SCOPE("clCreateContextFromType");
    return g_pNextDispatch->clCreateContextFromType(
        properties,
        device_type,
        pfn_notify,
        user_data,
        errcode_ret);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clRetainContext_layer(
    cl_context context)
{
    PROFILE_SCOPE("clRetainContext");
    return g_pNextDispatch->clRetainContext(
        context);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clReleaseContext_layer(
    cl_context context)
{
    PROFILE_SCOPE("clReleaseContext");
    return g_pNextDispatch->clReleaseContext(
        context);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clGetContextInfo_layer(
    cl_context context,
    cl_context_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    PROFILE_SCOPE("clGetContextInfo");
    return g_pNextDispatch->clGetContextInfo(
        context,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clRetainCommandQueue_layer(
    cl_command_queue command_queue)
{
    PROFILE_SCOPE("clRetainCommandQueue");
    return g_pNextDispatch->clRetainCommandQueue(
        command_queue);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clReleaseCommandQueue_layer(
    cl_command_queue command_queue)
{
    PROFILE_SCOPE("clReleaseCommandQueue");
    return g_pNextDispatch->clReleaseCommandQueue(
        command_queue);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clGetCommandQueueInfo_layer(
    cl_command_queue command_queue,
    cl_command_queue_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    PROFILE_SCOPE("clGetCommandQueueInfo");
    return g_pNextDispatch->clGetCommandQueueInfo(
        command_queue,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
}

///////////////////////////////////////////////////////////////////////////////

static cl_mem CL_API_CALL clCreateBuffer_layer(
    cl_context context,
    cl_mem_flags flags,
    size_t size,
    void* host_ptr,
    cl_int* errcode_ret)
{
    PROFILE_SCOPE("clCreateBuffer");
    return g_pNextDispatch->clCreateBuffer(
        context,
        flags,
        size,
        host_ptr,
        errcode_ret);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clRetainMemObject_layer(
    cl_mem memobj)
{
    PROFILE_SCOPE("clRetainMemObject");
    return g_pNextDispatch->clRetainMemObject(
        memobj);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clReleaseMemObject_layer(
    cl_mem memobj)
{
    PROFILE_SCOPE("clReleaseMemObject");
    return g_pNextDispatch->clReleaseMemObject(
        memobj);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clGetSupportedImageFormats_layer(
    cl_context context,
    cl_mem_flags flags,
    cl_mem_object_type image_type,
    cl_uint num_entries,
    cl_image_format* image_formats,
    cl_uint* num_image_formats)
{
    PROFILE_SCOPE("clGetSupportedImageFormats");
    return g_pNextDispatch->clGetSupportedImageFormats(
        context,
        flags,
        image_type,
        num_entries,
        image_formats,
        num_image_formats);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clGetMemObjectInfo_layer(
    cl_mem memobj,
    cl_mem_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    PROFILE_SCOPE("clGetMemObjectInfo");
    return g_pNextDispatch->clGetMemObjectInfo(
        memobj,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clGetImageInfo_layer(
    cl_mem image,
    cl_image_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    PROFILE_SCOPE("clGetImageInfo");
    return g_pNextDispatch->clGetImageInfo(
        image,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clRetainSampler_layer(
    cl_sampler sampler)
{
    PROFILE_SCOPE("clRetainSampler");
    return g_pNextDispatch->clRetainSampler(
        sampler);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clReleaseSampler_layer(
    cl_sampler sampler)
{
    PROFILE_SCOPE("clReleaseSampler");
    return g_pNextDispatch->clReleaseSampler(
        sampler);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clGetSamplerInfo_layer(
    cl_sampler sampler,
    cl_sampler_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    PROFILE_SCOPE("clGetSamplerInfo");
    return g_pNextDispatch->clGetSamplerInfo(
        sampler,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
}

///////////////////////////////////////////////////////////////////////////////

static cl_program CL_API_CALL clCreateProgramWithSource_layer(
    cl_context context,
    cl_uint count,
    const char** strings,
    const size_t* lengths,
    cl_int* errcode_ret)
{
    PROFILE_SCOPE("clCreateProgramWithSource");
    return g_pNextDispatch->clCreateProgramWithSource(
        context,
        count,
        strings,
        lengths,
        errcode_ret);
}

///////////////////////////////////////////////////////////////////////////////

static cl_program CL_API_CALL clCreateProgramWithBinary_layer(
    cl_context context,
    cl_uint num_devices,
    const cl_device_id* device_list,
    const size_t* lengths,
    const unsigned char** binaries,
    cl_int* binary_status,
    cl_int* errcode_ret)
{
    PROFILE_SCOPE("clCreateProgramWithBinary");
    return g_pNextDispatch->clCreateProgramWithBinary(
        context,
        num_devices,
        device_list,
        lengths,
        binaries,
        binary_status,
        errcode_ret);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clRetainProgram_layer(
    cl_program program)
{
    PROFILE_SCOPE("clRetainProgram");
    return g_pNextDispatch->clRetainProgram(
        program);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clReleaseProgram_layer(
    cl_program program)
{
    PROFILE_SCOPE("clReleaseProgram");
    return g_pNextDispatch->clReleaseProgram(
        program);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clBuildProgram_layer(
    cl_program program,
    cl_uint num_devices,
    const cl_device_id* device_list,
    const char* options,
    void (CL_CALLBACK* pfn_notify)(cl_program program, void* user_data),
    void* user_data)
{
    PROFILE_SCOPE("clBuildProgram");
    return g_pNextDispatch->clBuildProgram(
        program,
        num_devices,
        device_list,
        options,
        pfn_notify,
        user_data);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clGetProgramInfo_layer(
    cl_program program,
    cl_program_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    PROFILE_SCOPE("clGetProgramInfo");
    return g_pNextDispatch->clGetProgramInfo(
        program,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clGetProgramBuildInfo_layer(
    cl_program program,
    cl_device_id device,
    cl_program_build_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    PROFILE_SCOPE("clGetProgramBuildInfo");
    return g_pNextDispatch->clGetProgramBuildInfo(
        program,
        device,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
}

///////////////////////////////////////////////////////////////////////////////

static cl_kernel CL_API_CALL clCreateKernel_layer(
    cl_program program,
    const char* kernel_name,
    cl_int* errcode_ret)
{
    PROFILE_SCOPE("clCreateKernel");
    return g_pNextDispatch->clCreateKernel(
        program,
        kernel_name,
        errcode_ret);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clCreateKernelsInProgram_layer(
    cl_program program,
    cl_uint num_kernels,
    cl_kernel* kernels,
    cl_uint* num_kernels_ret)
{
    PROFILE_SCOPE("clCreateKernelsInProgram");
    return g_pNextDispatch->clCreateKernelsInProgram(
        program,
        num_kernels,
        kernels,
        num_kernels_ret);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clRetainKernel_layer(
    cl_kernel kernel)
{
    PROFILE_SCOPE("clRetainKernel");
    return g_pNextDispatch->clRetainKernel(
        kernel);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clReleaseKernel_layer(
    cl_kernel kernel)
{
    PROFILE_SCOPE("clReleaseKernel");
    return g_pNextDispatch->clReleaseKernel(
        kernel);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clSetKernelArg_layer(
    cl_kernel kernel,
    cl_uint arg_index,
    size_t arg_size,
    const void* arg_value)
{
    PROFILE_SCOPE("clSetKernelArg");
    return g_pNextDispatch->clSetKernelArg(
        kernel,
        arg_index,
        arg_size,
        arg_value);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clGetKernelInfo_layer(
    cl_kernel kernel,
    cl_kernel_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    PROFILE_SCOPE("clGetKernelInfo");
    return g_pNextDispatch->clGetKernelInfo(
        kernel,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clGetKernelWorkGroupInfo_layer(
    cl_kernel kernel,
    cl_device_id device,
    cl_kernel_work_group_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    PROFILE_SCOPE("clGetKernelWorkGroupInfo");
    return g_pNextDispatch->clGetKernelWorkGroupInfo(
        kernel,
        device,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clWaitForEvents_layer(
    cl_uint num_events,
    const cl_event* event_list)
{
    PROFILE_SCOPE("clWaitForEvents");
    return g_pNextDispatch->clWaitForEvents(
        num_events,
        event_list);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clGetEventInfo_layer(
    cl_event event,
    cl_event_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    PROFILE_SCOPE("clGetEventInfo");
    return g_pNextDispatch->clGetEventInfo(
        event,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clRetainEvent_layer(
    cl_event event)
{
    PROFILE_SCOPE("clRetainEvent");
    return g_pNextDispatch->clRetainEvent(
        event);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clReleaseEvent_layer(
    cl_event event)
{
    PROFILE_SCOPE("clReleaseEvent");
    return g_pNextDispatch->clReleaseEvent(
        event);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clGetEventProfilingInfo_layer(
    cl_event event,
    cl_profiling_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    PROFILE_SCOPE("clGetEventProfilingInfo");
    return g_pNextDispatch->clGetEventProfilingInfo(
        event,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clFlush_layer(
    cl_command_queue command_queue)
{
    PROFILE_SCOPE("clFlush");
    return g_pNextDispatch->clFlush(
        command_queue);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clFinish_layer(
    cl_command_queue command_queue)
{
    PROFILE_SCOPE("clFinish");
    return g_pNextDispatch->clFinish(
        command_queue);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clEnqueueReadBuffer_layer(
    cl_command_queue command_queue,
    cl_mem buffer,
    cl_bool blocking_read,
    size_t offset,
    size_t size,
    void* ptr,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    PROFILE_SCOPE("clEnqueueReadBuffer");
    return g_pNextDispatch->clEnqueueReadBuffer(
        command_queue,
        buffer,
        blocking_read,
        offset,
        size,
        ptr,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clEnqueueWriteBuffer_layer(
    cl_command_queue command_queue,
    cl_mem buffer,
    cl_bool blocking_write,
    size_t offset,
    size_t size,
    const void* ptr,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    PROFILE_SCOPE("clEnqueueWriteBuffer");
    return g_pNextDispatch->clEnqueueWriteBuffer(
        command_queue,
        buffer,
        blocking_write,
        offset,
        size,
        ptr,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clEnqueueCopyBuffer_layer(
    cl_command_queue command_queue,
    cl_mem src_buffer,
    cl_mem dst_buffer,
    size_t src_offset,
    size_t dst_offset,
    size_t size,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    PROFILE_SCOPE("clEnqueueCopyBuffer");
    return g_pNextDispatch->clEnqueueCopyBuffer(
        command_queue,
        src_buffer,
        dst_buffer,
        src_offset,
        dst_offset,
        size,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clEnqueueReadImage_layer(
    cl_command_queue command_queue,
    cl_mem image,
    cl_bool blocking_read,
    const size_t* origin,
    const size_t* region,
    size_t row_pitch,
    size_t slice_pitch,
    void* ptr,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    PROFILE_SCOPE("clEnqueueReadImage");
    return g_pNextDispatch->clEnqueueReadImage(
        command_queue,
        image,
        blocking_read,
        origin,
        region,
        row_pitch,
        slice_pitch,
        ptr,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clEnqueueWriteImage_layer(
    cl_command_queue command_queue,
    cl_mem image,
    cl_bool blocking_write,
    const size_t* origin,
    const size_t* region,
    size_t input_row_pitch,
    size_t input_slice_pitch,
    const void* ptr,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    PROFILE_SCOPE("clEnqueueWriteImage");
    return g_pNextDispatch->clEnqueueWriteImage(
        command_queue,
        image,
        blocking_write,
        origin,
        region,
        input_row_pitch,
        input_slice_pitch,
        ptr,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clEnqueueCopyImage_layer(
    cl_command_queue command_queue,
    cl_mem src_image,
    cl_mem dst_image,
    const size_t* src_origin,
    const size_t* dst_origin,
    const size_t* region,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    PROFILE_SCOPE("clEnqueueCopyImage");
    return g_pNextDispatch->clEnqueueCopyImage(
        command_queue,
        src_image,
        dst_image,
        src_origin,
        dst_origin,
        region,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clEnqueueCopyImageToBuffer_layer(
    cl_command_queue command_queue,
    cl_mem src_image,
    cl_mem dst_buffer,
    const size_t* src_origin,
    const size_t* region,
    size_t dst_offset,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    PROFILE_SCOPE("clEnqueueCopyImageToBuffer");
    return g_pNextDispatch->clEnqueueCopyImageToBuffer(
        command_queue,
        src_image,
        dst_buffer,
        src_origin,
        region,
        dst_offset,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clEnqueueCopyBufferToImage_layer(
    cl_command_queue command_queue,
    cl_mem src_buffer,
    cl_mem dst_image,
    size_t src_offset,
    const size_t* dst_origin,
    const size_t* region,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    PROFILE_SCOPE("clEnqueueCopyBufferToImage");
    return g_pNextDispatch->clEnqueueCopyBufferToImage(
        command_queue,
        src_buffer,
        dst_image,
        src_offset,
        dst_origin,
        region,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

///////////////////////////////////////////////////////////////////////////////

static void* CL_API_CALL clEnqueueMapBuffer_layer(
    cl_command_queue command_queue,
    cl_mem buffer,
    cl_bool blocking_map,
    cl_map_flags map_flags,
    size_t offset,
    size_t size,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event,
    cl_int* errcode_ret)
{
    PROFILE_SCOPE("clEnqueueMapBuffer");
    return g_pNextDispatch->clEnqueueMapBuffer(
        command_queue,
        buffer,
        blocking_map,
        map_flags,
        offset,
        size,
        num_events_in_wait_list,
        event_wait_list,
        event,
        errcode_ret);
}

///////////////////////////////////////////////////////////////////////////////

static void* CL_API_CALL clEnqueueMapImage_layer(
    cl_command_queue command_queue,
    cl_mem image,
    cl_bool blocking_map,
    cl_map_flags map_flags,
    const size_t* origin,
    const size_t* region,
    size_t* image_row_pitch,
    size_t* image_slice_pitch,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event,
    cl_int* errcode_ret)
{
    PROFILE_SCOPE("clEnqueueMapImage");
    return g_pNextDispatch->clEnqueueMapImage(
        command_queue,
        image,
        blocking_map,
        map_flags,
        origin,
        region,
        image_row_pitch,
        image_slice_pitch,
        num_events_in_wait_list,
        event_wait_list,
        event,
        errcode_ret);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clEnqueueUnmapMemObject_layer(
    cl_command_queue command_queue,
    cl_mem memobj,
    void* mapped_ptr,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    PROFILE_SCOPE("clEnqueueUnmapMemObject");
    return g_pNextDispatch->clEnqueueUnmapMemObject(
        command_queue,
        memobj,
        mapped_ptr,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clEnqueueNDRangeKernel_layer(
    cl_command_queue command_queue,
    cl_kernel kernel,
    cl_uint work_dim,
    const size_t* global_work_offset,
    const size_t* global_work_size,
    const size_t* local_work_size,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    PROFILE_SCOPE("clEnqueueNDRangeKernel");
    return g_pNextDispatch->clEnqueueNDRangeKernel(
        command_queue,
        kernel,
        work_dim,
        global_work_offset,
        global_work_size,
        local_work_size,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clEnqueueNativeKernel_layer(
    cl_command_queue command_queue,
    void (CL_CALLBACK* user_func)(void*),
    void* args,
    size_t cb_args,
    cl_uint num_mem_objects,
    const cl_mem* mem_list,
    const void** args_mem_loc,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    PROFILE_SCOPE("clEnqueueNativeKernel");
    return g_pNextDispatch->clEnqueueNativeKernel(
        command_queue,
        user_func,
        args,
        cb_args,
        num_mem_objects,
        mem_list,
        args_mem_loc,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clSetCommandQueueProperty_layer(
    cl_command_queue command_queue,
    cl_command_queue_properties properties,
    cl_bool enable,
    cl_command_queue_properties* old_properties)
{
    PROFILE_SCOPE("clSetCommandQueueProperty");
    return g_pNextDispatch->clSetCommandQueueProperty(
        command_queue,
        properties,
        enable,
        old_properties);
}

///////////////////////////////////////////////////////////////////////////////

static cl_mem CL_API_CALL clCreateImage2D_layer(
    cl_context context,
    cl_mem_flags flags,
    const cl_image_format* image_format,
    size_t image_width,
    size_t image_height,
    size_t image_row_pitch,
    void* host_ptr,
    cl_int* errcode_ret)
{
    PROFILE_SCOPE("clCreateImage2D");
    return g_pNextDispatch->clCreateImage2D(
        context,
        flags,
        image_format,
        image_width,
        image_height,
        image_row_pitch,
        host_ptr,
        errcode_ret);
}

///////////////////////////////////////////////////////////////////////////////

static cl_mem CL_API_CALL clCreateImage3D_layer(
    cl_context context,
    cl_mem_flags flags,
    const cl_image_format* image_format,
    size_t image_width,
    size_t image_height,
    size_t image_depth,
    size_t image_row_pitch,
    size_t image_slice_pitch,
    void* host_ptr,
    cl_int* errcode_ret)
{
    PROFILE_SCOPE("clCreateImage3D");
    return g_pNextDispatch->clCreateImage3D(
        context,
        flags,
        image_format,
        image_width,
        image_height,
        image_depth,
        image_row_pitch,
        image_slice_pitch,
        host_ptr,
        errcode_ret);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clEnqueueMarker_layer(
    cl_command_queue command_queue,
    cl_event* event)
{
    PROFILE_SCOPE("clEnqueueMarker");
    return g_pNextDispatch->clEnqueueMarker(
        command_queue,
        event);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clEnqueueWaitForEvents_layer(
    cl_command_queue command_queue,
    cl_uint num_events,
    const cl_event* event_list)
{
    PROFILE_SCOPE("clEnqueueWaitForEvents");
    return g_pNextDispatch->clEnqueueWaitForEvents(
        command_queue,
        num_events,
        event_list);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clEnqueueBarrier_layer(
    cl_command_queue command_queue)
{
    PROFILE_SCOPE("clEnqueueBarrier");
    return g_pNextDispatch->clEnqueueBarrier(
        command_queue);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clUnloadCompiler_layer(
    void)
{
    PROFILE_SCOPE("clUnloadCompiler");
    return g_pNextDispatch->clUnloadCompiler();
}

///////////////////////////////////////////////////////////////////////////////

static void* CL_API_CALL clGetExtensionFunctionAddress_layer(
    const char* func_name)
{
    PROFILE_SCOPE("clGetExtensionFunctionAddress");
    return g_pNextDispatch->clGetExtensionFunctionAddress(
        func_name);
}

///////////////////////////////////////////////////////////////////////////////

static cl_command_queue CL_API_CALL clCreateCommandQueue_layer(
    cl_context context,
    cl_device_id device,
    cl_command_queue_properties properties,
    cl_int* errcode_ret)
{
    PROFILE_SCOPE("clCreateCommandQueue");
    return g_pNextDispatch->clCreateCommandQueue(
        context,
        device,
        properties,
        errcode_ret);
}

///////////////////////////////////////////////////////////////////////////////

static cl_sampler CL_API_CALL clCreateSampler_layer(
    cl_context context,
    cl_bool normalized_coords,
    cl_addressing_mode addressing_mode,
    cl_filter_mode filter_mode,
    cl_int* errcode_ret)
{
    PROFILE_SCOPE("clCreateSampler");
    return g_pNextDispatch->clCreateSampler(
        context,
        normalized_coords,
        addressing_mode,
        filter_mode,
        errcode_ret);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clEnqueueTask_layer(
    cl_command_queue command_queue,
    cl_kernel kernel,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    PROFILE_SCOPE("clEnqueueTask");
    return g_pNextDispatch->clEnqueueTask(
        command_queue,
        kernel,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

///////////////////////////////////////////////////////////////////////////////

static cl_mem CL_API_CALL clCreateSubBuffer_layer(
    cl_mem buffer,
    cl_mem_flags flags,
    cl_buffer_create_type buffer_create_type,
    const void* buffer_create_info,
    cl_int* errcode_ret)
{
    PROFILE_SCOPE("clCreateSubBuffer");
    return g_pNextDispatch->clCreateSubBuffer(
        buffer,
        flags,
        buffer_create_type,
        buffer_create_info,
        errcode_ret);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clSetMemObjectDestructorCallback_layer(
    cl_mem memobj,
    void (CL_CALLBACK* pfn_notify)(cl_mem memobj, void* user_data),
    void* user_data)
{
    PROFILE_SCOPE("clSetMemObjectDestructorCallback");
    return g_pNextDispatch->clSetMemObjectDestructorCallback(
        memobj,
        pfn_notify,
        user_data);
}

///////////////////////////////////////////////////////////////////////////////

static cl_event CL_API_CALL clCreateUserEvent_layer(
    cl_context context,
    cl_int* errcode_ret)
{
    PROFILE_SCOPE("clCreateUserEvent");
    return g_pNextDispatch->clCreateUserEvent(
        context,
        errcode_ret);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clSetUserEventStatus_layer(
    cl_event event,
    cl_int execution_status)
{
    PROFILE_SCOPE("clSetUserEventStatus");
    return g_pNextDispatch->clSetUserEventStatus(
        event,
        execution_status);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clSetEventCallback_layer(
    cl_event event,
    cl_int command_exec_callback_type,
    void (CL_CALLBACK* pfn_notify)(cl_event event, cl_int event_command_status, void *user_data),
    void* user_data)
{
    PROFILE_SCOPE("clSetEventCallback");
    return g_pNextDispatch->clSetEventCallback(
        event,
        command_exec_callback_type,
        pfn_notify,
        user_data);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clEnqueueReadBufferRect_layer(
    cl_command_queue command_queue,
    cl_mem buffer,
    cl_bool blocking_read,
    const size_t* buffer_origin,
    const size_t* host_origin,
    const size_t* region,
    size_t buffer_row_pitch,
    size_t buffer_slice_pitch,
    size_t host_row_pitch,
    size_t host_slice_pitch,
    void* ptr,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    PROFILE_SCOPE("clEnqueueReadBufferRect");
    return g_pNextDispatch->clEnqueueReadBufferRect(
        command_queue,
        buffer,
        blocking_read,
        buffer_origin,
        host_origin,
        region,
        buffer_row_pitch,
        buffer_slice_pitch,
        host_row_pitch,
        host_slice_pitch,
        ptr,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clEnqueueWriteBufferRect_layer(
    cl_command_queue command_queue,
    cl_mem buffer,
    cl_bool blocking_write,
    const size_t* buffer_origin,
    const size_t* host_origin,
    const size_t* region,
    size_t buffer_row_pitch,
    size_t buffer_slice_pitch,
    size_t host_row_pitch,
    size_t host_slice_pitch,
    const void* ptr,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    PROFILE_SCOPE("clEnqueueWriteBufferRect");
    return g_pNextDispatch->clEnqueueWriteBufferRect(
        command_queue,
        buffer,
        blocking_write,
        buffer_origin,
        host_origin,
        region,
        buffer_row_pitch,
        buffer_slice_pitch,
        host_row_pitch,
        host_slice_pitch,
        ptr,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clEnqueueCopyBufferRect_layer(
    cl_command_queue command_queue,
    cl_mem src_buffer,
    cl_mem dst_buffer,
    const size_t* src_origin,
    const size_t* dst_origin,
    const size_t* region,
    size_t src_row_pitch,
    size_t src_slice_pitch,
    size_t dst_row_pitch,
    size_t dst_slice_pitch,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    PROFILE_SCOPE("clEnqueueCopyBufferRect");
    return g_pNextDispatch->clEnqueueCopyBufferRect(
        command_queue,
        src_buffer,
        dst_buffer,
        src_origin,
        dst_origin,
        region,
        src_row_pitch,
        src_slice_pitch,
        dst_row_pitch,
        dst_slice_pitch,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clCreateSubDevices_layer(
    cl_device_id in_device,
    const cl_device_partition_property* properties,
    cl_uint num_devices,
    cl_device_id* out_devices,
    cl_uint* num_devices_ret)
{
    PROFILE_SCOPE("clCreateSubDevices");
    return g_pNextDispatch->clCreateSubDevices(
        in_device,
        properties,
        num_devices,
        out_devices,
        num_devices_ret);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clRetainDevice_layer(
    cl_device_id device)
{
    PROFILE_SCOPE("clRetainDevice");
    return g_pNextDispatch->clRetainDevice(
        device);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clReleaseDevice_layer(
    cl_device_id device)
{
    PROFILE_SCOPE("clReleaseDevice");
    return g_pNextDispatch->clReleaseDevice(
        device);
}

///////////////////////////////////////////////////////////////////////////////

static cl_mem CL_API_CALL clCreateImage_layer(
    cl_context context,
    cl_mem_flags flags,
    const cl_image_format* image_format,
    const cl_image_desc* image_desc,
    void* host_ptr,
    cl_int* errcode_ret)
{
    PROFILE_SCOPE("clCreateImage");
    return g_pNextDispatch->clCreateImage(
        context,
        flags,
        image_format,
        image_desc,
        host_ptr,
        errcode_ret);
}

///////////////////////////////////////////////////////////////////////////////

static cl_program CL_API_CALL clCreateProgramWithBuiltInKernels_layer(
    cl_context context,
    cl_uint num_devices,
    const cl_device_id* device_list,
    const char* kernel_names,
    cl_int* errcode_ret)
{
    PROFILE_SCOPE("clCreateProgramWithBuiltInKernels");
    return g_pNextDispatch->clCreateProgramWithBuiltInKernels(
        context,
        num_devices,
        device_list,
        kernel_names,
        errcode_ret);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clCompileProgram_layer(
    cl_program program,
    cl_uint num_devices,
    const cl_device_id* device_list,
    const char* options,
    cl_uint num_input_headers,
    const cl_program* input_headers,
    const char** header_include_names,
    void (CL_CALLBACK* pfn_notify)(cl_program program, void* user_data),
    void* user_data)
{
    PROFILE_SCOPE("clCompileProgram");
    return g_pNextDispatch->clCompileProgram(
        program,
        num_devices,
        device_list,
        options,
        num_input_headers,
        input_headers,
        header_include_names,
        pfn_notify,
        user_data);
}

///////////////////////////////////////////////////////////////////////////////

static cl_program CL_API_CALL clLinkProgram_layer(
    cl_context context,
    cl_uint num_devices,
    const cl_device_id* device_list,
    const char* options,
    cl_uint num_input_programs,
    const cl_program* input_programs,
    void (CL_CALLBACK* pfn_notify)(cl_program program, void* user_data),
    void* user_data,
    cl_int* errcode_ret)
{
    PROFILE_SCOPE("clLinkProgram");
    return g_pNextDispatch->clLinkProgram(
        context,
        num_devices,
        device_list,
        options,
        num_input_programs,
        input_programs,
        pfn_notify,
        user_data,
        errcode_ret);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clUnloadPlatformCompiler_layer(
    cl_platform_id platform)
{
    PROFILE_SCOPE("clUnloadPlatformCompiler");
    return g_pNextDispatch->clUnloadPlatformCompiler(
        platform);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clGetKernelArgInfo_layer(
    cl_kernel kernel,
    cl_uint arg_index,
    cl_kernel_arg_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    PROFILE_SCOPE("clGetKernelArgInfo");
    return g_pNextDispatch->clGetKernelArgInfo(
        kernel,
        arg_index,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clEnqueueFillBuffer_layer(
    cl_command_queue command_queue,
    cl_mem buffer,
    const void* pattern,
    size_t pattern_size,
    size_t offset,
    size_t size,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    PROFILE_SCOPE("clEnqueueFillBuffer");
    return g_pNextDispatch->clEnqueueFillBuffer(
        command_queue,
        buffer,
        pattern,
        pattern_size,
        offset,
        size,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clEnqueueFillImage_layer(
    cl_command_queue command_queue,
    cl_mem image,
    const void* fill_color,
    const size_t* origin,
    const size_t* region,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    PROFILE_SCOPE("clEnqueueFillImage");
    return g_pNextDispatch->clEnqueueFillImage(
        command_queue,
        image,
        fill_color,
        origin,
        region,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clEnqueueMigrateMemObjects_layer(
    cl_command_queue command_queue,
    cl_uint num_mem_objects,
    const cl_mem* mem_objects,
    cl_mem_migration_flags flags,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    PROFILE_SCOPE("clEnqueueMigrateMemObjects");
    return g_pNextDispatch->clEnqueueMigrateMemObjects(
        command_queue,
        num_mem_objects,
        mem_objects,
        flags,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clEnqueueMarkerWithWaitList_layer(
    cl_command_queue command_queue,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    PROFILE_SCOPE("clEnqueueMarkerWithWaitList");
    return g_pNextDispatch->clEnqueueMarkerWithWaitList(
        command_queue,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clEnqueueBarrierWithWaitList_layer(
    cl_command_queue command_queue,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    PROFILE_SCOPE("clEnqueueBarrierWithWaitList");
    return g_pNextDispatch->clEnqueueBarrierWithWaitList(
        command_queue,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

///////////////////////////////////////////////////////////////////////////////

static void* CL_API_CALL clGetExtensionFunctionAddressForPlatform_layer(
    cl_platform_id platform,
    const char* func_name)
{
    PROFILE_SCOPE("clGetExtensionFunctionAddressForPlatform");
    return g_pNextDispatch->clGetExtensionFunctionAddressForPlatform(
        platform,
        func_name);
}

///////////////////////////////////////////////////////////////////////////////

static cl_command_queue CL_API_CALL clCreateCommandQueueWithProperties_layer(
    cl_context context,
    cl_device_id device,
    const cl_queue_properties* properties,
    cl_int* errcode_ret)
{
    PROFILE_SCOPE("clCreateCommandQueueWithProperties");
    return g_pNextDispatch->clCreateCommandQueueWithProperties(
        context,
        device,
        properties,
        errcode_ret);
}

///////////////////////////////////////////////////////////////////////////////

static cl_mem CL_API_CALL clCreatePipe_layer(
    cl_context context,
    cl_mem_flags flags,
    cl_uint pipe_packet_size,
    cl_uint pipe_max_packets,
    const cl_pipe_properties* properties,
    cl_int* errcode_ret)
{
    PROFILE_SCOPE("clCreatePipe");
    return g_pNextDispatch->clCreatePipe(
        context,
        flags,
        pipe_packet_size,
        pipe_max_packets,
        properties,
        errcode_ret);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clGetPipeInfo_layer(
    cl_mem pipe,
    cl_pipe_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    PROFILE_SCOPE("clGetPipeInfo");
    return g_pNextDispatch->clGetPipeInfo(
        pipe,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
}

///////////////////////////////////////////////////////////////////////////////

static void* CL_API_CALL clSVMAlloc_layer(
    cl_context context,
    cl_svm_mem_flags flags,
    size_t size,
    cl_uint alignment)
{
    PROFILE_SCOPE("clSVMAlloc");
    return g_pNextDispatch->clSVMAlloc(
        context,
        flags,
        size,
        alignment);
}

///////////////////////////////////////////////////////////////////////////////

static void CL_API_CALL clSVMFree_layer(
    cl_context context,
    void* svm_pointer)
{
    PROFILE_SCOPE("clSVMFree");
    return g_pNextDispatch->clSVMFree(
        context,
        svm_pointer);
}

///////////////////////////////////////////////////////////////////////////////

static cl_sampler CL_API_CALL clCreateSamplerWithProperties_layer(
    cl_context context,
    const cl_sampler_properties* sampler_properties,
    cl_int* errcode_ret)
{
    PROFILE_SCOPE("clCreateSamplerWithProperties");
    return g_pNextDispatch->clCreateSamplerWithProperties(
        context,
        sampler_properties,
        errcode_ret);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clSetKernelArgSVMPointer_layer(
    cl_kernel kernel,
    cl_uint arg_index,
    const void* arg_value)
{
    PROFILE_SCOPE("clSetKernelArgSVMPointer");
    return g_pNextDispatch->clSetKernelArgSVMPointer(
        kernel,
        arg_index,
        arg_value);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clSetKernelExecInfo_layer(
    cl_kernel kernel,
    cl_kernel_exec_info param_name,
    size_t param_value_size,
    const void* param_value)
{
    PROFILE_SCOPE("clSetKernelExecInfo");
    return g_pNextDispatch->clSetKernelExecInfo(
        kernel,
        param_name,
        param_value_size,
        param_value);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clEnqueueSVMFree_layer(
    cl_command_queue command_queue,
    cl_uint num_svm_pointers,
    void* svm_pointers[],
    void (CL_CALLBACK* pfn_free_func)(cl_command_queue queue, cl_uint num_svm_pointers, void* svm_pointers[], void* user_data),
    void* user_data,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    PROFILE_SCOPE("clEnqueueSVMFree");
    return g_pNextDispatch->clEnqueueSVMFree(
        command_queue,
        num_svm_pointers,
        svm_pointers,
        pfn_free_func,
        user_data,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clEnqueueSVMMemcpy_layer(
    cl_command_queue command_queue,
    cl_bool blocking_copy,
    void* dst_ptr,
    const void* src_ptr,
    size_t size,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    PROFILE_SCOPE("clEnqueueSVMMemcpy");
    return g_pNextDispatch->clEnqueueSVMMemcpy(
        command_queue,
        blocking_copy,
        dst_ptr,
        src_ptr,
        size,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clEnqueueSVMMemFill_layer(
    cl_command_queue command_queue,
    void* svm_ptr,
    const void* pattern,
    size_t pattern_size,
    size_t size,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    PROFILE_SCOPE("clEnqueueSVMMemFill");
    return g_pNextDispatch->clEnqueueSVMMemFill(
        command_queue,
        svm_ptr,
        pattern,
        pattern_size,
        size,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clEnqueueSVMMap_layer(
    cl_command_queue command_queue,
    cl_bool blocking_map,
    cl_map_flags flags,
    void* svm_ptr,
    size_t size,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    PROFILE_SCOPE("clEnqueueSVMMap");
    return g_pNextDispatch->clEnqueueSVMMap(
        command_queue,
        blocking_map,
        flags,
        svm_ptr,
        size,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clEnqueueSVMUnmap_layer(
    cl_command_queue command_queue,
    void* svm_ptr,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    PROFILE_SCOPE("clEnqueueSVMUnmap");
    return g_pNextDispatch->clEnqueueSVMUnmap(
        command_queue,
        svm_ptr,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clSetDefaultDeviceCommandQueue_layer(
    cl_context context,
    cl_device_id device,
    cl_command_queue command_queue)
{
    PROFILE_SCOPE("clSetDefaultDeviceCommandQueue");
    return g_pNextDispatch->clSetDefaultDeviceCommandQueue(
        context,
        device,
        command_queue);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clGetDeviceAndHostTimer_layer(
    cl_device_id device,
    cl_ulong* device_timestamp,
    cl_ulong* host_timestamp)
{
    PROFILE_SCOPE("clGetDeviceAndHostTimer");
    return g_pNextDispatch->clGetDeviceAndHostTimer(
        device,
        device_timestamp,
        host_timestamp);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clGetHostTimer_layer(
    cl_device_id device,
    cl_ulong* host_timestamp)
{
    PROFILE_SCOPE("clGetHostTimer");
    return g_pNextDispatch->clGetHostTimer(
        device,
        host_timestamp);
}

///////////////////////////////////////////////////////////////////////////////

static cl_program CL_API_CALL clCreateProgramWithIL_layer(
    cl_context context,
    const void* il,
    size_t length,
    cl_int* errcode_ret)
{
    PROFILE_SCOPE("clCreateProgramWithIL");
    return g_pNextDispatch->clCreateProgramWithIL(
        context,
        il,
        length,
        errcode_ret);
}

///////////////////////////////////////////////////////////////////////////////

static cl_kernel CL_API_CALL clCloneKernel_layer(
    cl_kernel source_kernel,
    cl_int* errcode_ret)
{
    PROFILE_SCOPE("clCloneKernel");
    return g_pNextDispatch->clCloneKernel(
        source_kernel,
        errcode_ret);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clGetKernelSubGroupInfo_layer(
    cl_kernel kernel,
    cl_device_id device,
    cl_kernel_sub_group_info param_name,
    size_t input_value_size,
    const void* input_value,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    PROFILE_SCOPE("clGetKernelSubGroupInfo");
    return g_pNextDispatch->clGetKernelSubGroupInfo(
        kernel,
        device,
        param_name,
        input_value_size,
        input_value,
        param_value_size,
        param_value,
        param_value_size_ret);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clEnqueueSVMMigrateMem_layer(
    cl_command_queue command_queue,
    cl_uint num_svm_pointers,
    const void** svm_pointers,
    const size_t* sizes,
    cl_mem_migration_flags flags,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    PROFILE_SCOPE("clEnqueueSVMMigrateMem");
    return g_pNextDispatch->clEnqueueSVMMigrateMem(
        command_queue,
        num_svm_pointers,
        svm_pointers,
        sizes,
        flags,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clSetProgramSpecializationConstant_layer(
    cl_program program,
    cl_uint spec_id,
    size_t spec_size,
    const void* spec_value)
{
    PROFILE_SCOPE("clSetProgramSpecializationConstant");
    return g_pNextDispatch->clSetProgramSpecializationConstant(
        program,
        spec_id,
        spec_size,
        spec_value);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clSetProgramReleaseCallback_layer(
    cl_program program,
    void (CL_CALLBACK* pfn_notify)(cl_program program, void* user_data),
    void* user_data)
{
    PROFILE_SCOPE("clSetProgramReleaseCallback");
    return g_pNextDispatch->clSetProgramReleaseCallback(
        program,
        pfn_notify,
        user_data);
}

///////////////////////////////////////////////////////////////////////////////

static cl_int CL_API_CALL clSetContextDestructorCallback_layer(
    cl_context context,
    void (CL_CALLBACK* pfn_notify)(cl_context context, void* user_data),
    void* user_data)
{
    PROFILE_SCOPE("clSetContextDestructorCallback");
    return g_pNextDispatch->clSetContextDestructorCallback(
        context,
        pfn_notify,
        user_data);
}

///////////////////////////////////////////////////////////////////////////////

static cl_mem CL_API_CALL clCreateBufferWithProperties_layer(
    cl_context context,
    const cl_mem_properties* properties,
    cl_mem_flags flags,
    size_t size,
    void* host_ptr,
    cl_int* errcode_ret)
{
    PROFILE_SCOPE("clCreateBufferWithProperties");
    return g_pNextDispatch->clCreateBufferWithProperties(
        context,
        properties,
        flags,
        size,
        host_ptr,
        errcode_ret);
}

///////////////////////////////////////////////////////////////////////////////

static cl_mem CL_API_CALL clCreateImageWithProperties_layer(
    cl_context context,
    const cl_mem_properties* properties,
    cl_mem_flags flags,
    const cl_image_format* image_format,
    const cl_image_desc* image_desc,
    void* host_ptr,
    cl_int* errcode_ret)
{
    PROFILE_SCOPE("clCreateImageWithProperties");
    return g_pNextDispatch->clCreateImageWithProperties(
        context,
        properties,
        flags,
        image_format,
        image_desc,
        host_ptr,
        errcode_ret);
}
