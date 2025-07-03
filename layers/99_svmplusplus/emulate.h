/*
// Copyright (c) 2023 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <CL/cl.h>
#include <CL/cl_ext.h>

extern const struct _cl_icd_dispatch* g_pNextDispatch;

void* CL_API_CALL clSVMAllocWithPropertiesKHR_EMU(
    cl_context context,
    const cl_svm_alloc_properties_khr* properties,
    cl_uint svm_type_index,
    size_t size,
    cl_int* errcode_ret);

cl_int CL_API_CALL clSVMFreeWithPropertiesKHR_EMU(
    cl_context context,
    const cl_svm_free_properties_khr* properties,
    cl_svm_free_flags_khr flags,
    void* ptr);

cl_int CL_API_CALL clGetSVMSuggestedTypeIndexKHR_EMU(
    cl_context context,
    cl_svm_capabilities_khr required_capabilities,
    cl_svm_capabilities_khr desired_capabilities,
    const cl_svm_alloc_properties_khr* properties,
    size_t size,
    cl_uint* suggested_svm_type_index);

cl_int CL_API_CALL clGetSVMPointerInfoKHR_EMU(
    cl_context context,
    cl_device_id device,
    const void* ptr,
    cl_svm_pointer_info_khr param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret);

cl_int CL_API_CALL clGetDeviceInfo_override(
    cl_device_id device,
    cl_device_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret);

cl_int CL_API_CALL clGetEventInfo_override(
    cl_event event,
    cl_event_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret);

cl_int CL_API_CALL clGetPlatformInfo_override(
    cl_platform_id platform,
    cl_platform_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret);

cl_int CL_API_CALL clSetKernelArgSVMPointer_override(
    cl_kernel kernel,
    cl_uint arg_index,
    const void* arg_value);

cl_int CL_API_CALL clSetKernelExecInfo_override(
    cl_kernel kernel,
    cl_kernel_exec_info param_name,
    size_t param_value_size,
    const void* param_value);

void CL_API_CALL clSVMFree_override(
    cl_context context,
    void* ptr);

cl_int CL_API_CALL clEnqueueSVMFree_override(
    cl_command_queue command_queue,
    cl_uint num_svm_pointers,
    void* svm_pointers[],
    void (CL_CALLBACK* pfn_free_func)(
        cl_command_queue queue,
        cl_uint num_svm_pointers,
        void* svm_pointers[],
        void* user_data),
    void* user_data,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event);

cl_int CL_API_CALL clEnqueueSVMMemcpy_override(
    cl_command_queue command_queue,
    cl_bool blocking_copy,
    void* dst_ptr,
    const void* src_ptr,
    size_t size,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event);

cl_int CL_API_CALL clEnqueueSVMMemFill_override(
    cl_command_queue command_queue,
    void* svm_ptr,
    const void* pattern,
    size_t pattern_size,
    size_t size,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event);

cl_int CL_API_CALL clEnqueueSVMMigrateMem_override(
    cl_command_queue command_queue,
    cl_uint num_svm_pointers,
    const void** svm_pointers,
    const size_t* sizes,
    cl_mem_migration_flags flags,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event);

cl_int CL_API_CALL clReleaseEvent_override(
    cl_event event);
