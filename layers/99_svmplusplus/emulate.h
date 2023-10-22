/*
// Copyright (c) 2023 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <CL/cl.h>
#include <CL/cl_ext.h>

#include <map>

// TODO: Move this to a shared header.

#define CL_EXP_NEW_SVM_EXTENSION_NAME \
    "cl_exp_unified_svm"

// New-ish types and enums:

typedef cl_bitfield         cl_device_unified_shared_memory_capabilities_exp;   // analogous to cl_device_unified_shared_memory_capabilities_intel
typedef cl_properties       cl_svm_mem_properties_exp;                          // analogous to cl_mem_properties_intel
typedef cl_uint             cl_svm_mem_info_exp;                                // analogous to cl_mem_info_intel
typedef cl_uint             cl_svm_mem_type_exp;                                // analogous to cl_unified_shared_memory_type_intel
typedef cl_uint             cl_svm_mem_advice_exp;                              // analogous to cl_mem_advice_intel
typedef cl_bitfield         cl_svm_free_flags_exp;      // new
typedef cl_properties       cl_svm_free_properties_exp; // new

/* cl_svm_mem_flags */
#define CL_MEM_SVM_DEVICE_EXP               (1 << 16)
#define CL_MEM_SVM_HOST_EXP                 (1 << 17)
#define CL_MEM_SVM_SHARED_EXP               (1 << 18)

/* cl_device_svm_capabilities */
// These may not be needed - can be derived from specific device queries!
#define CL_DEVICE_SVM_DEVICE_ALLOC_EXP      (1 << 4)
#define CL_DEVICE_SVM_HOST_ALLOC_EXP        (1 << 5)
#define CL_DEVICE_SVM_SHARED_ALLOC_EXP      (1 << 6)

/* cl_svm_free_flags_exp */
#define CL_SVM_FREE_NON_BLOCKING_EXP        (1 << 0)
#define CL_SVM_FREE_BLOCKING_EXP            (1 << 1)

/* cl_svm_mem_properties_exp */
#define CL_SVM_MEM_ASSOCIATED_DEVICE_HANDLE_EXP             0x10100 // note: placeholder!
// consider: CL_SVM_MEM_DEVICE_HANDLE_LIST for cross-device allocations?

// Aliased types and enums:

/* cl_device_info - aliases for USM */
#define CL_DEVICE_HOST_MEM_CAPABILITIES_EXP                 0x4190
#define CL_DEVICE_DEVICE_MEM_CAPABILITIES_EXP               0x4191
#define CL_DEVICE_SINGLE_DEVICE_SHARED_MEM_CAPABILITIES_EXP 0x4192
#define CL_DEVICE_CROSS_DEVICE_SHARED_MEM_CAPABILITIES_EXP  0x4193
#define CL_DEVICE_SHARED_SYSTEM_MEM_CAPABILITIES_EXP        0x4194

/* cl_unified_shared_memory_capabilities_intel - bitfield - aliases for USM */
#define CL_UNIFIED_SHARED_MEMORY_ACCESS_EXP                 (1 << 0)
#define CL_UNIFIED_SHARED_MEMORY_ATOMIC_ACCESS_EXP          (1 << 1)
#define CL_UNIFIED_SHARED_MEMORY_CONCURRENT_ACCESS_EXP      (1 << 2)
#define CL_UNIFIED_SHARED_MEMORY_CONCURRENT_ATOMIC_ACCESS_EXP (1 << 3)

// TODO: should these be cl_mem_svm_flags?
// CL_MEM_ALLOC_WRITE_COMBINED_INTEL               (1 << 0)
// CL_MEM_ALLOC_INITIAL_PLACEMENT_DEVICE_INTEL     (1 << 1)
// CL_MEM_ALLOC_INITIAL_PLACEMENT_HOST_INTEL       (1 << 2)

/* cl_svm_mem_info_exp */
#define CL_SVM_MEM_TYPE_EXP                                 0x419A
#define CL_SVM_MEM_BASE_PTR_EXP                             0x419B
#define CL_SVM_MEM_SIZE_EXP                                 0x419C
#define CL_SVM_MEM_DEVICE_EXP                               0x419D

/* cl_svm_mem_type_exp */
#define CL_SVM_MEM_TYPE_UNKNOWN_EXP                         0x4196
#define CL_SVM_MEM_TYPE_HOST_EXP                            0x4197
#define CL_SVM_MEM_TYPE_DEVICE_EXP                          0x4198
#define CL_SVM_MEM_TYPE_SHARED_EXP                          0x4199
// TODO: do we need types for SVM buffer, SVM coarse grain buffer, SVM fine grain buffer, ... ?

/* cl_kernel_exec_info */
#define CL_KERNEL_EXEC_INFO_INDIRECT_HOST_ACCESS_EXP        0x4200
#define CL_KERNEL_EXEC_INFO_INDIRECT_DEVICE_ACCESS_EXP      0x4201
#define CL_KERNEL_EXEC_INFO_INDIRECT_SHARED_ACCESS_EXP      0x4202
// TODO: do we need indirect access flags for SVM buffer, ... ?

/* cl_command_type */
#define CL_COMMAND_MEMADVISE_EXP                            0x4207

// New functions:

typedef void* CL_API_CALL
clSVMAllocWithPropertiesEXP_t(
    cl_context context,
    const cl_svm_mem_properties_exp* properties,
    cl_svm_mem_flags flags,
    size_t size,
    cl_uint alignment,
    cl_int* errcode_ret);

typedef clSVMAllocWithPropertiesEXP_t *
clSVMAllocWithPropertiesEXP_fn ;

typedef cl_int CL_API_CALL
clSVMFreeWithPropertiesEXP_t(
    cl_context context,
    const cl_svm_free_properties_exp* properties,
    cl_svm_free_flags_exp flags,
    void* ptr);

typedef clSVMFreeWithPropertiesEXP_t *
clSVMFreeWithPropertiesEXP_fn ;

typedef cl_int CL_API_CALL
clGetSVMMemInfoEXP_t(
    const void* ptr,
    cl_svm_mem_info_exp param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret);

typedef clGetSVMMemInfoEXP_t *
clGetSVMMemInfoEXP_fn ;

typedef cl_int CL_API_CALL
clEnqueueSVMMemAdviseEXP_t(
    cl_command_queue command_queue,
    const void* ptr,
    size_t size,
    cl_svm_mem_advice_exp advice,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event);

typedef clEnqueueMemAdviseINTEL_t *
clEnqueueMemAdviseINTEL_fn ;

#if !defined(CL_NO_NON_ICD_DISPATCH_EXTENSION_PROTOTYPES)

extern CL_API_ENTRY void* CL_API_CALL
clSVMAllocWithPropertiesEXP(
    cl_context context,
    const cl_svm_mem_properties_exp* properties,
    cl_svm_mem_flags flags,
    size_t size,
    cl_uint alignment,
    cl_int* errcode_ret);

extern CL_API_ENTRY cl_int CL_API_CALL
clSVMFreeWithPropertiesEXP(
    cl_context context,
    const cl_svm_free_properties_exp* properties,
    cl_svm_free_flags_exp flags,
    void* ptr);

extern CL_API_ENTRY cl_int CL_API_CALL
clGetSVMMemInfoEXP(
    const void* ptr,
    cl_svm_mem_info_exp param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret);

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueSVMMemAdviseEXP(
    cl_command_queue command_queue,
    const void* ptr,
    size_t size,
    cl_svm_mem_advice_exp advice,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event);

#endif /* !defined(CL_NO_NON_ICD_DISPATCH_EXTENSION_PROTOTYPES) */

extern const struct _cl_icd_dispatch* g_pNextDispatch;

void* CL_API_CALL clSVMAllocWithPropertiesEXP_EMU(
    cl_context context,
    const cl_svm_mem_properties_exp* properties,
    cl_svm_mem_flags flags,
    size_t size,
    cl_uint alignment,
    cl_int* errcode_ret);

cl_int CL_API_CALL clSVMFreeWithPropertiesEXP_EMU(
    cl_context context,
    const cl_svm_free_properties_exp* properties,
    cl_svm_free_flags_exp flags,
    void* ptr);

cl_int CL_API_CALL clGetSVMMemInfoEXP_EMU(
    cl_context context,
    const void* ptr,
    cl_svm_mem_info_exp param_name,
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

cl_int CL_API_CALL clEnqueueSVMMemAdviseEXP_EMU(
    cl_command_queue command_queue,
    const void* ptr,
    size_t size,
    cl_svm_mem_advice_exp advice,
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
