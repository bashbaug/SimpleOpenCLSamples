/*
// Copyright (c) 20220 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <CL/cl.h>
#include <CL/cl_ext.h>

///////////////////////////////////////////////////////////////////////////////
// Emulated Functions

cl_command_buffer_khr CL_API_CALL clCreateCommandBufferKHR_EMU(
    cl_uint num_queues,
    const cl_command_queue* queues,
    const cl_command_buffer_properties_khr* properties,
    cl_int* errcode_ret);

cl_int CL_API_CALL clFinalizeCommandBufferKHR_EMU(
    cl_command_buffer_khr command_buffer);

cl_int CL_API_CALL clRetainCommandBufferKHR_EMU(
    cl_command_buffer_khr command_buffer);

cl_int CL_API_CALL clReleaseCommandBufferKHR_EMU(
    cl_command_buffer_khr command_buffer);

cl_int CL_API_CALL clEnqueueCommandBufferKHR_EMU(
    cl_uint num_queues,
    cl_command_queue* queues,
    cl_command_buffer_khr command_buffer,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event);

cl_int CL_API_CALL clCommandBarrierWithWaitListKHR_EMU(
    cl_command_buffer_khr command_buffer,
    cl_command_queue command_queue,
    cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr* sync_point_wait_list,
    cl_sync_point_khr* sync_point,
    cl_mutable_command_khr* mutable_handle);

cl_int CL_API_CALL clCommandCopyBufferKHR_EMU(
    cl_command_buffer_khr command_buffer,
    cl_command_queue command_queue,
    cl_mem src_buffer,
    cl_mem dst_buffer,
    size_t src_offset,
    size_t dst_offset,
    size_t size,
    cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr* sync_point_wait_list,
    cl_sync_point_khr* sync_point,
    cl_mutable_command_khr* mutable_handle);

cl_int CL_API_CALL clCommandCopyBufferRectKHR_EMU(
    cl_command_buffer_khr command_buffer,
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
    cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr* sync_point_wait_list,
    cl_sync_point_khr* sync_point,
    cl_mutable_command_khr* mutable_handle);

cl_int CL_API_CALL clCommandCopyBufferToImageKHR_EMU(
    cl_command_buffer_khr command_buffer,
    cl_command_queue command_queue,
    cl_mem src_buffer,
    cl_mem dst_image,
    size_t src_offset,
    const size_t* dst_origin,
    const size_t* region,
    cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr* sync_point_wait_list,
    cl_sync_point_khr* sync_point,
    cl_mutable_command_khr* mutable_handle);

cl_int CL_API_CALL clCommandCopyImageKHR_EMU(
    cl_command_buffer_khr command_buffer,
    cl_command_queue command_queue,
    cl_mem src_image,
    cl_mem dst_image,
    const size_t* src_origin,
    const size_t* dst_origin,
    const size_t* region,
    cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr* sync_point_wait_list,
    cl_sync_point_khr* sync_point,
    cl_mutable_command_khr* mutable_handle);

cl_int CL_API_CALL clCommandCopyImageToBufferKHR_EMU(
    cl_command_buffer_khr command_buffer,
    cl_command_queue command_queue,
    cl_mem src_image,
    cl_mem dst_buffer,
    const size_t* src_origin,
    const size_t* region,
    size_t dst_offset,
    cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr* sync_point_wait_list,
    cl_sync_point_khr* sync_point,
    cl_mutable_command_khr* mutable_handle);

cl_int CL_API_CALL clCommandFillBufferKHR_EMU(
    cl_command_buffer_khr command_buffer,
    cl_command_queue command_queue,
    cl_mem buffer,
    const void* pattern,
    size_t pattern_size,
    size_t offset,
    size_t size,
    cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr* sync_point_wait_list,
    cl_sync_point_khr* sync_point,
    cl_mutable_command_khr* mutable_handle);

cl_int CL_API_CALL clCommandFillImageKHR_EMU(
    cl_command_buffer_khr command_buffer,
    cl_command_queue command_queue,
    cl_mem image,
    const void* fill_color,
    const size_t* origin,
    const size_t* region,
    cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr* sync_point_wait_list,
    cl_sync_point_khr* sync_point,
    cl_mutable_command_khr* mutable_handle);

cl_int CL_API_CALL clCommandNDRangeKernelKHR_EMU(
    cl_command_buffer_khr command_buffer,
    cl_command_queue command_queue,
    const cl_ndrange_kernel_command_properties_khr* properties,
    cl_kernel kernel,
    cl_uint work_dim,
    const size_t* global_work_offset,
    const size_t* global_work_size,
    const size_t* local_work_size,
    cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr* sync_point_wait_list,
    cl_sync_point_khr* sync_point,
    cl_mutable_command_khr* mutable_handle);

cl_int CL_API_CALL clGetCommandBufferInfoKHR_EMU(
    cl_command_buffer_khr command_buffer,
    cl_command_buffer_info_khr param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret);

///////////////////////////////////////////////////////////////////////////////
// Override Functions

bool clGetDeviceInfo_override(
    cl_device_id device,
    cl_device_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret,
    cl_int* errcode_ret);

bool clGetPlatformInfo_override(
    cl_platform_id platform,
    cl_platform_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret,
    cl_int* errcode_ret);
