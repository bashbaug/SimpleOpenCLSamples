/*
// Copyright (c) 2023 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <CL/cl.h>
#include <CL/cl_ext.h>

extern const struct _cl_icd_dispatch* g_pNextDispatch;

void* CL_API_CALL clSVMAllocWithPropertiesEXP_EMU(
    cl_context context,
    const cl_svm_alloc_properties_exp* properties,
    cl_svm_capabilities_exp capabilities,
    cl_mem_flags flags,
    size_t size,
    cl_uint alignment,
    cl_int* errcode_ret);

cl_int CL_API_CALL clSVMFreeWithPropertiesEXP_EMU(
    cl_context context,
    const cl_svm_free_properties_exp* properties,
    cl_svm_free_flags_exp flags,
    void* ptr);

cl_int CL_API_CALL clGetSuggestedSVMCapabilitiesEXP_EMU(
    cl_context context,
    cl_uint num_devices,
    const cl_device_id* devices,
    cl_svm_capabilities_exp required_capabilities,
    cl_svm_capabilities_exp* suggested_capabilities);

cl_int CL_API_CALL clGetSVMInfoEXP_EMU(
    cl_context context,
    cl_device_id device,
    const void* ptr,
    cl_svm_info_exp param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret);

cl_int CL_API_CALL clGetDeviceInfo_override(
    cl_device_id device,
    cl_device_info param_name,
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

cl_int CL_API_CALL clEnqueueSVMMemAdviseEXP_EMU(
    cl_command_queue command_queue,
    const void* ptr,
    size_t size,
    cl_svm_advice_exp advice,
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
