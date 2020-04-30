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

#include <CL/cl.h>

#ifdef _WIN32
#include <malloc.h>
#include <windows.h>
#endif
#ifdef __linux__
#include <dlfcn.h>
#endif

#define _SCL_MAX_NUM_PLATFORMS 64

#define _SCL_VALIDATE_HANDLE_RETURN_ERROR(_handle, _error)              \
    if (_handle == NULL) return _error;

#define _SCL_VALIDATE_HANDLE_RETURN_HANDLE(_handle, _error)             \
    if (_handle == NULL) {                                              \
        if (errcode_ret) *errcode_ret = _error;                         \
        return NULL;                                                    \
    }

///////////////////////////////////////////////////////////////////////////////
// API Function Pointers:

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clGetPlatformIDs)(
    cl_uint num_entries,
    cl_platform_id* platforms,
    cl_uint* num_platforms);
#else
typedef void* _sclpfn_clGetPlatformIDs;
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clGetPlatformInfo)(
    cl_platform_id platform,
    cl_platform_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret);
#else
typedef void* _sclpfn_clGetPlatformInfo;
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clGetDeviceIDs)(
    cl_platform_id platform,
    cl_device_type device_type,
    cl_uint num_entries,
    cl_device_id* devices,
    cl_uint* num_devices);
#else
typedef void* _sclpfn_clGetDeviceIDs;
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clGetDeviceInfo)(
    cl_device_id device,
    cl_device_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret);
#else
typedef void* _sclpfn_clGetDeviceInfo;
#endif

#ifdef CL_VERSION_1_0
typedef cl_context (CL_API_CALL *_sclpfn_clCreateContext)(
    const cl_context_properties* properties,
    cl_uint num_devices,
    const cl_device_id* devices,
    void (CL_CALLBACK* pfn_notify)(const char* errinfo, const void* private_info, size_t cb, void* user_data),
    void* user_data,
    cl_int* errcode_ret);
#else
typedef void* _sclpfn_clCreateContext;
#endif

#ifdef CL_VERSION_1_0
typedef cl_context (CL_API_CALL *_sclpfn_clCreateContextFromType)(
    const cl_context_properties* properties,
    cl_device_type device_type,
    void (CL_CALLBACK* pfn_notify)(const char* errinfo, const void* private_info, size_t cb, void* user_data),
    void* user_data,
    cl_int* errcode_ret);
#else
typedef void* _sclpfn_clCreateContextFromType;
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clRetainContext)(
    cl_context context);
#else
typedef void* _sclpfn_clRetainContext;
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clReleaseContext)(
    cl_context context);
#else
typedef void* _sclpfn_clReleaseContext;
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clGetContextInfo)(
    cl_context context,
    cl_context_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret);
#else
typedef void* _sclpfn_clGetContextInfo;
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clRetainCommandQueue)(
    cl_command_queue command_queue);
#else
typedef void* _sclpfn_clRetainCommandQueue;
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clReleaseCommandQueue)(
    cl_command_queue command_queue);
#else
typedef void* _sclpfn_clReleaseCommandQueue;
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clGetCommandQueueInfo)(
    cl_command_queue command_queue,
    cl_command_queue_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret);
#else
typedef void* _sclpfn_clGetCommandQueueInfo;
#endif

#ifdef CL_VERSION_1_0
typedef cl_mem (CL_API_CALL *_sclpfn_clCreateBuffer)(
    cl_context context,
    cl_mem_flags flags,
    size_t size,
    void* host_ptr,
    cl_int* errcode_ret);
#else
typedef void* _sclpfn_clCreateBuffer;
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clRetainMemObject)(
    cl_mem memobj);
#else
typedef void* _sclpfn_clRetainMemObject;
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clReleaseMemObject)(
    cl_mem memobj);
#else
typedef void* _sclpfn_clReleaseMemObject;
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clGetSupportedImageFormats)(
    cl_context context,
    cl_mem_flags flags,
    cl_mem_object_type image_type,
    cl_uint num_entries,
    cl_image_format* image_formats,
    cl_uint* num_image_formats);
#else
typedef void* _sclpfn_clGetSupportedImageFormats;
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clGetMemObjectInfo)(
    cl_mem memobj,
    cl_mem_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret);
#else
typedef void* _sclpfn_clGetMemObjectInfo;
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clGetImageInfo)(
    cl_mem image,
    cl_image_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret);
#else
typedef void* _sclpfn_clGetImageInfo;
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clRetainSampler)(
    cl_sampler sampler);
#else
typedef void* _sclpfn_clRetainSampler;
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clReleaseSampler)(
    cl_sampler sampler);
#else
typedef void* _sclpfn_clReleaseSampler;
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clGetSamplerInfo)(
    cl_sampler sampler,
    cl_sampler_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret);
#else
typedef void* _sclpfn_clGetSamplerInfo;
#endif

#ifdef CL_VERSION_1_0
typedef cl_program (CL_API_CALL *_sclpfn_clCreateProgramWithSource)(
    cl_context context,
    cl_uint count,
    const char** strings,
    const size_t* lengths,
    cl_int* errcode_ret);
#else
typedef void* _sclpfn_clCreateProgramWithSource;
#endif

#ifdef CL_VERSION_1_0
typedef cl_program (CL_API_CALL *_sclpfn_clCreateProgramWithBinary)(
    cl_context context,
    cl_uint num_devices,
    const cl_device_id* device_list,
    const size_t* lengths,
    const unsigned char** binaries,
    cl_int* binary_status,
    cl_int* errcode_ret);
#else
typedef void* _sclpfn_clCreateProgramWithBinary;
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clRetainProgram)(
    cl_program program);
#else
typedef void* _sclpfn_clRetainProgram;
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clReleaseProgram)(
    cl_program program);
#else
typedef void* _sclpfn_clReleaseProgram;
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clBuildProgram)(
    cl_program program,
    cl_uint num_devices,
    const cl_device_id* device_list,
    const char* options,
    void (CL_CALLBACK* pfn_notify)(cl_program program, void* user_data),
    void* user_data);
#else
typedef void* _sclpfn_clBuildProgram;
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clGetProgramInfo)(
    cl_program program,
    cl_program_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret);
#else
typedef void* _sclpfn_clGetProgramInfo;
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clGetProgramBuildInfo)(
    cl_program program,
    cl_device_id device,
    cl_program_build_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret);
#else
typedef void* _sclpfn_clGetProgramBuildInfo;
#endif

#ifdef CL_VERSION_1_0
typedef cl_kernel (CL_API_CALL *_sclpfn_clCreateKernel)(
    cl_program program,
    const char* kernel_name,
    cl_int* errcode_ret);
#else
typedef void* _sclpfn_clCreateKernel;
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clCreateKernelsInProgram)(
    cl_program program,
    cl_uint num_kernels,
    cl_kernel* kernels,
    cl_uint* num_kernels_ret);
#else
typedef void* _sclpfn_clCreateKernelsInProgram;
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clRetainKernel)(
    cl_kernel kernel);
#else
typedef void* _sclpfn_clRetainKernel;
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clReleaseKernel)(
    cl_kernel kernel);
#else
typedef void* _sclpfn_clReleaseKernel;
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clSetKernelArg)(
    cl_kernel kernel,
    cl_uint arg_index,
    size_t arg_size,
    const void* arg_value);
#else
typedef void* _sclpfn_clSetKernelArg;
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clGetKernelInfo)(
    cl_kernel kernel,
    cl_kernel_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret);
#else
typedef void* _sclpfn_clGetKernelInfo;
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clGetKernelWorkGroupInfo)(
    cl_kernel kernel,
    cl_device_id device,
    cl_kernel_work_group_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret);
#else
typedef void* _sclpfn_clGetKernelWorkGroupInfo;
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clWaitForEvents)(
    cl_uint num_events,
    const cl_event* event_list);
#else
typedef void* _sclpfn_clWaitForEvents;
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clGetEventInfo)(
    cl_event event,
    cl_event_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret);
#else
typedef void* _sclpfn_clGetEventInfo;
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clRetainEvent)(
    cl_event event);
#else
typedef void* _sclpfn_clRetainEvent;
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clReleaseEvent)(
    cl_event event);
#else
typedef void* _sclpfn_clReleaseEvent;
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clGetEventProfilingInfo)(
    cl_event event,
    cl_profiling_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret);
#else
typedef void* _sclpfn_clGetEventProfilingInfo;
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clFlush)(
    cl_command_queue command_queue);
#else
typedef void* _sclpfn_clFlush;
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clFinish)(
    cl_command_queue command_queue);
#else
typedef void* _sclpfn_clFinish;
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clEnqueueReadBuffer)(
    cl_command_queue command_queue,
    cl_mem buffer,
    cl_bool blocking_read,
    size_t offset,
    size_t size,
    void* ptr,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event);
#else
typedef void* _sclpfn_clEnqueueReadBuffer;
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clEnqueueWriteBuffer)(
    cl_command_queue command_queue,
    cl_mem buffer,
    cl_bool blocking_write,
    size_t offset,
    size_t size,
    const void* ptr,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event);
#else
typedef void* _sclpfn_clEnqueueWriteBuffer;
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clEnqueueCopyBuffer)(
    cl_command_queue command_queue,
    cl_mem src_buffer,
    cl_mem dst_buffer,
    size_t src_offset,
    size_t dst_offset,
    size_t size,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event);
#else
typedef void* _sclpfn_clEnqueueCopyBuffer;
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clEnqueueReadImage)(
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
    cl_event* event);
#else
typedef void* _sclpfn_clEnqueueReadImage;
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clEnqueueWriteImage)(
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
    cl_event* event);
#else
typedef void* _sclpfn_clEnqueueWriteImage;
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clEnqueueCopyImage)(
    cl_command_queue command_queue,
    cl_mem src_image,
    cl_mem dst_image,
    const size_t* src_origin,
    const size_t* dst_origin,
    const size_t* region,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event);
#else
typedef void* _sclpfn_clEnqueueCopyImage;
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clEnqueueCopyImageToBuffer)(
    cl_command_queue command_queue,
    cl_mem src_image,
    cl_mem dst_buffer,
    const size_t* src_origin,
    const size_t* region,
    size_t dst_offset,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event);
#else
typedef void* _sclpfn_clEnqueueCopyImageToBuffer;
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clEnqueueCopyBufferToImage)(
    cl_command_queue command_queue,
    cl_mem src_buffer,
    cl_mem dst_image,
    size_t src_offset,
    const size_t* dst_origin,
    const size_t* region,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event);
#else
typedef void* _sclpfn_clEnqueueCopyBufferToImage;
#endif

#ifdef CL_VERSION_1_0
typedef void* (CL_API_CALL *_sclpfn_clEnqueueMapBuffer)(
    cl_command_queue command_queue,
    cl_mem buffer,
    cl_bool blocking_map,
    cl_map_flags map_flags,
    size_t offset,
    size_t size,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event,
    cl_int* errcode_ret);
#else
typedef void* _sclpfn_clEnqueueMapBuffer;
#endif

#ifdef CL_VERSION_1_0
typedef void* (CL_API_CALL *_sclpfn_clEnqueueMapImage)(
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
    cl_int* errcode_ret);
#else
typedef void* _sclpfn_clEnqueueMapImage;
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clEnqueueUnmapMemObject)(
    cl_command_queue command_queue,
    cl_mem memobj,
    void* mapped_ptr,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event);
#else
typedef void* _sclpfn_clEnqueueUnmapMemObject;
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clEnqueueNDRangeKernel)(
    cl_command_queue command_queue,
    cl_kernel kernel,
    cl_uint work_dim,
    const size_t* global_work_offset,
    const size_t* global_work_size,
    const size_t* local_work_size,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event);
#else
typedef void* _sclpfn_clEnqueueNDRangeKernel;
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clEnqueueNativeKernel)(
    cl_command_queue command_queue,
    void (CL_CALLBACK* user_func)(void*),
    void* args,
    size_t cb_args,
    cl_uint num_mem_objects,
    const cl_mem* mem_list,
    const void** args_mem_loc,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event);
#else
typedef void* _sclpfn_clEnqueueNativeKernel;
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clSetCommandQueueProperty)(
    cl_command_queue command_queue,
    cl_command_queue_properties properties,
    cl_bool enable,
    cl_command_queue_properties* old_properties);
#else
typedef void* _sclpfn_clSetCommandQueueProperty;
#endif

#ifdef CL_VERSION_1_0
typedef cl_mem (CL_API_CALL *_sclpfn_clCreateImage2D)(
    cl_context context,
    cl_mem_flags flags,
    const cl_image_format* image_format,
    size_t image_width,
    size_t image_height,
    size_t image_row_pitch,
    void* host_ptr,
    cl_int* errcode_ret);
#else
typedef void* _sclpfn_clCreateImage2D;
#endif

#ifdef CL_VERSION_1_0
typedef cl_mem (CL_API_CALL *_sclpfn_clCreateImage3D)(
    cl_context context,
    cl_mem_flags flags,
    const cl_image_format* image_format,
    size_t image_width,
    size_t image_height,
    size_t image_depth,
    size_t image_row_pitch,
    size_t image_slice_pitch,
    void* host_ptr,
    cl_int* errcode_ret);
#else
typedef void* _sclpfn_clCreateImage3D;
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clEnqueueMarker)(
    cl_command_queue command_queue,
    cl_event* event);
#else
typedef void* _sclpfn_clEnqueueMarker;
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clEnqueueWaitForEvents)(
    cl_command_queue command_queue,
    cl_uint num_events,
    const cl_event* event_list);
#else
typedef void* _sclpfn_clEnqueueWaitForEvents;
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clEnqueueBarrier)(
    cl_command_queue command_queue);
#else
typedef void* _sclpfn_clEnqueueBarrier;
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clUnloadCompiler)(
    void );
#else
typedef void* _sclpfn_clUnloadCompiler;
#endif

#ifdef CL_VERSION_1_0
typedef void* (CL_API_CALL *_sclpfn_clGetExtensionFunctionAddress)(
    const char* func_name);
#else
typedef void* _sclpfn_clGetExtensionFunctionAddress;
#endif

#ifdef CL_VERSION_1_0
typedef cl_command_queue (CL_API_CALL *_sclpfn_clCreateCommandQueue)(
    cl_context context,
    cl_device_id device,
    cl_command_queue_properties properties,
    cl_int* errcode_ret);
#else
typedef void* _sclpfn_clCreateCommandQueue;
#endif

#ifdef CL_VERSION_1_0
typedef cl_sampler (CL_API_CALL *_sclpfn_clCreateSampler)(
    cl_context context,
    cl_bool normalized_coords,
    cl_addressing_mode addressing_mode,
    cl_filter_mode filter_mode,
    cl_int* errcode_ret);
#else
typedef void* _sclpfn_clCreateSampler;
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clEnqueueTask)(
    cl_command_queue command_queue,
    cl_kernel kernel,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event);
#else
typedef void* _sclpfn_clEnqueueTask;
#endif

#ifdef CL_VERSION_1_1
typedef cl_mem (CL_API_CALL *_sclpfn_clCreateSubBuffer)(
    cl_mem buffer,
    cl_mem_flags flags,
    cl_buffer_create_type buffer_create_type,
    const void* buffer_create_info,
    cl_int* errcode_ret);
#else
typedef void* _sclpfn_clCreateSubBuffer;
#endif

#ifdef CL_VERSION_1_1
typedef cl_int (CL_API_CALL *_sclpfn_clSetMemObjectDestructorCallback)(
    cl_mem memobj,
    void (CL_CALLBACK* pfn_notify)(cl_mem memobj, void* user_data),
    void* user_data);
#else
typedef void* _sclpfn_clSetMemObjectDestructorCallback;
#endif

#ifdef CL_VERSION_1_1
typedef cl_event (CL_API_CALL *_sclpfn_clCreateUserEvent)(
    cl_context context,
    cl_int* errcode_ret);
#else
typedef void* _sclpfn_clCreateUserEvent;
#endif

#ifdef CL_VERSION_1_1
typedef cl_int (CL_API_CALL *_sclpfn_clSetUserEventStatus)(
    cl_event event,
    cl_int execution_status);
#else
typedef void* _sclpfn_clSetUserEventStatus;
#endif

#ifdef CL_VERSION_1_1
typedef cl_int (CL_API_CALL *_sclpfn_clSetEventCallback)(
    cl_event event,
    cl_int command_exec_callback_type,
    void (CL_CALLBACK* pfn_notify)(cl_event event, cl_int event_command_status, void *user_data),
    void* user_data);
#else
typedef void* _sclpfn_clSetEventCallback;
#endif

#ifdef CL_VERSION_1_1
typedef cl_int (CL_API_CALL *_sclpfn_clEnqueueReadBufferRect)(
    cl_command_queue command_queue,
    cl_mem buffer,
    cl_bool blocking_read,
    const size_t* buffer_offset,
    const size_t* host_offset,
    const size_t* region,
    size_t buffer_row_pitch,
    size_t buffer_slice_pitch,
    size_t host_row_pitch,
    size_t host_slice_pitch,
    void* ptr,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event);
#else
typedef void* _sclpfn_clEnqueueReadBufferRect;
#endif

#ifdef CL_VERSION_1_1
typedef cl_int (CL_API_CALL *_sclpfn_clEnqueueWriteBufferRect)(
    cl_command_queue command_queue,
    cl_mem buffer,
    cl_bool blocking_write,
    const size_t* buffer_offset,
    const size_t* host_offset,
    const size_t* region,
    size_t buffer_row_pitch,
    size_t buffer_slice_pitch,
    size_t host_row_pitch,
    size_t host_slice_pitch,
    const void* ptr,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event);
#else
typedef void* _sclpfn_clEnqueueWriteBufferRect;
#endif

#ifdef CL_VERSION_1_1
typedef cl_int (CL_API_CALL *_sclpfn_clEnqueueCopyBufferRect)(
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
    cl_event* event);
#else
typedef void* _sclpfn_clEnqueueCopyBufferRect;
#endif

#ifdef CL_VERSION_1_2
typedef cl_int (CL_API_CALL *_sclpfn_clCreateSubDevices)(
    cl_device_id in_device,
    const cl_device_partition_property* properties,
    cl_uint num_devices,
    cl_device_id* out_devices,
    cl_uint* num_devices_ret);
#else
typedef void* _sclpfn_clCreateSubDevices;
#endif

#ifdef CL_VERSION_1_2
typedef cl_int (CL_API_CALL *_sclpfn_clRetainDevice)(
    cl_device_id device);
#else
typedef void* _sclpfn_clRetainDevice;
#endif

#ifdef CL_VERSION_1_2
typedef cl_int (CL_API_CALL *_sclpfn_clReleaseDevice)(
    cl_device_id device);
#else
typedef void* _sclpfn_clReleaseDevice;
#endif

#ifdef CL_VERSION_1_2
typedef cl_mem (CL_API_CALL *_sclpfn_clCreateImage)(
    cl_context context,
    cl_mem_flags flags,
    const cl_image_format* image_format,
    const cl_image_desc* image_desc,
    void* host_ptr,
    cl_int* errcode_ret);
#else
typedef void* _sclpfn_clCreateImage;
#endif

#ifdef CL_VERSION_1_2
typedef cl_program (CL_API_CALL *_sclpfn_clCreateProgramWithBuiltInKernels)(
    cl_context context,
    cl_uint num_devices,
    const cl_device_id* device_list,
    const char* kernel_names,
    cl_int* errcode_ret);
#else
typedef void* _sclpfn_clCreateProgramWithBuiltInKernels;
#endif

#ifdef CL_VERSION_1_2
typedef cl_int (CL_API_CALL *_sclpfn_clCompileProgram)(
    cl_program program,
    cl_uint num_devices,
    const cl_device_id* device_list,
    const char* options,
    cl_uint num_input_headers,
    const cl_program* input_headers,
    const char** header_include_names,
    void (CL_CALLBACK* pfn_notify)(cl_program program, void* user_data),
    void* user_data);
#else
typedef void* _sclpfn_clCompileProgram;
#endif

#ifdef CL_VERSION_1_2
typedef cl_program (CL_API_CALL *_sclpfn_clLinkProgram)(
    cl_context context,
    cl_uint num_devices,
    const cl_device_id* device_list,
    const char* options,
    cl_uint num_input_programs,
    const cl_program* input_programs,
    void (CL_CALLBACK* pfn_notify)(cl_program program, void* user_data),
    void* user_data,
    cl_int* errcode_ret);
#else
typedef void* _sclpfn_clLinkProgram;
#endif

#ifdef CL_VERSION_1_2
typedef cl_int (CL_API_CALL *_sclpfn_clUnloadPlatformCompiler)(
    cl_platform_id platform);
#else
typedef void* _sclpfn_clUnloadPlatformCompiler;
#endif

#ifdef CL_VERSION_1_2
typedef cl_int (CL_API_CALL *_sclpfn_clGetKernelArgInfo)(
    cl_kernel kernel,
    cl_uint arg_indx,
    cl_kernel_arg_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret);
#else
typedef void* _sclpfn_clGetKernelArgInfo;
#endif

#ifdef CL_VERSION_1_2
typedef cl_int (CL_API_CALL *_sclpfn_clEnqueueFillBuffer)(
    cl_command_queue command_queue,
    cl_mem buffer,
    const void* pattern,
    size_t pattern_size,
    size_t offset,
    size_t size,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event);
#else
typedef void* _sclpfn_clEnqueueFillBuffer;
#endif

#ifdef CL_VERSION_1_2
typedef cl_int (CL_API_CALL *_sclpfn_clEnqueueFillImage)(
    cl_command_queue command_queue,
    cl_mem image,
    const void* fill_color,
    const size_t* origin,
    const size_t* region,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event);
#else
typedef void* _sclpfn_clEnqueueFillImage;
#endif

#ifdef CL_VERSION_1_2
typedef cl_int (CL_API_CALL *_sclpfn_clEnqueueMigrateMemObjects)(
    cl_command_queue command_queue,
    cl_uint num_mem_objects,
    const cl_mem* mem_objects,
    cl_mem_migration_flags flags,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event);
#else
typedef void* _sclpfn_clEnqueueMigrateMemObjects;
#endif

#ifdef CL_VERSION_1_2
typedef cl_int (CL_API_CALL *_sclpfn_clEnqueueMarkerWithWaitList)(
    cl_command_queue command_queue,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event);
#else
typedef void* _sclpfn_clEnqueueMarkerWithWaitList;
#endif

#ifdef CL_VERSION_1_2
typedef cl_int (CL_API_CALL *_sclpfn_clEnqueueBarrierWithWaitList)(
    cl_command_queue command_queue,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event);
#else
typedef void* _sclpfn_clEnqueueBarrierWithWaitList;
#endif

#ifdef CL_VERSION_1_2
typedef void* (CL_API_CALL *_sclpfn_clGetExtensionFunctionAddressForPlatform)(
    cl_platform_id platform,
    const char* func_name);
#else
typedef void* _sclpfn_clGetExtensionFunctionAddressForPlatform;
#endif

#ifdef CL_VERSION_2_0
typedef cl_command_queue (CL_API_CALL *_sclpfn_clCreateCommandQueueWithProperties)(
    cl_context context,
    cl_device_id device,
    const cl_queue_properties* properties,
    cl_int* errcode_ret);
#else
typedef void* _sclpfn_clCreateCommandQueueWithProperties;
#endif

#ifdef CL_VERSION_2_0
typedef cl_mem (CL_API_CALL *_sclpfn_clCreatePipe)(
    cl_context context,
    cl_mem_flags flags,
    cl_uint pipe_packet_size,
    cl_uint pipe_max_packets,
    const cl_pipe_properties* properties,
    cl_int* errcode_ret);
#else
typedef void* _sclpfn_clCreatePipe;
#endif

#ifdef CL_VERSION_2_0
typedef cl_int (CL_API_CALL *_sclpfn_clGetPipeInfo)(
    cl_mem pipe,
    cl_pipe_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret);
#else
typedef void* _sclpfn_clGetPipeInfo;
#endif

#ifdef CL_VERSION_2_0
typedef void* (CL_API_CALL *_sclpfn_clSVMAlloc)(
    cl_context context,
    cl_svm_mem_flags flags,
    size_t size,
    cl_uint alignment);
#else
typedef void* _sclpfn_clSVMAlloc;
#endif

#ifdef CL_VERSION_2_0
typedef void (CL_API_CALL *_sclpfn_clSVMFree)(
    cl_context context,
    void* svm_pointer);
#else
typedef void* _sclpfn_clSVMFree;
#endif

#ifdef CL_VERSION_2_0
typedef cl_sampler (CL_API_CALL *_sclpfn_clCreateSamplerWithProperties)(
    cl_context context,
    const cl_sampler_properties* sampler_properties,
    cl_int* errcode_ret);
#else
typedef void* _sclpfn_clCreateSamplerWithProperties;
#endif

#ifdef CL_VERSION_2_0
typedef cl_int (CL_API_CALL *_sclpfn_clSetKernelArgSVMPointer)(
    cl_kernel kernel,
    cl_uint arg_index,
    const void* arg_value);
#else
typedef void* _sclpfn_clSetKernelArgSVMPointer;
#endif

#ifdef CL_VERSION_2_0
typedef cl_int (CL_API_CALL *_sclpfn_clSetKernelExecInfo)(
    cl_kernel kernel,
    cl_kernel_exec_info param_name,
    size_t param_value_size,
    const void* param_value);
#else
typedef void* _sclpfn_clSetKernelExecInfo;
#endif

#ifdef CL_VERSION_2_0
typedef cl_int (CL_API_CALL *_sclpfn_clEnqueueSVMFree)(
    cl_command_queue command_queue,
    cl_uint num_svm_pointers,
    void* svm_pointers[],
    void (CL_CALLBACK* pfn_free_func)(cl_command_queue queue, cl_uint num_svm_pointers, void* svm_pointers[], void* user_data),
    void* user_data,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event);
#else
typedef void* _sclpfn_clEnqueueSVMFree;
#endif

#ifdef CL_VERSION_2_0
typedef cl_int (CL_API_CALL *_sclpfn_clEnqueueSVMMemcpy)(
    cl_command_queue command_queue,
    cl_bool blocking_copy,
    void* dst_ptr,
    const void* src_ptr,
    size_t size,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event);
#else
typedef void* _sclpfn_clEnqueueSVMMemcpy;
#endif

#ifdef CL_VERSION_2_0
typedef cl_int (CL_API_CALL *_sclpfn_clEnqueueSVMMemFill)(
    cl_command_queue command_queue,
    void* svm_ptr,
    const void* pattern,
    size_t pattern_size,
    size_t size,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event);
#else
typedef void* _sclpfn_clEnqueueSVMMemFill;
#endif

#ifdef CL_VERSION_2_0
typedef cl_int (CL_API_CALL *_sclpfn_clEnqueueSVMMap)(
    cl_command_queue command_queue,
    cl_bool blocking_map,
    cl_map_flags flags,
    void* svm_ptr,
    size_t size,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event);
#else
typedef void* _sclpfn_clEnqueueSVMMap;
#endif

#ifdef CL_VERSION_2_0
typedef cl_int (CL_API_CALL *_sclpfn_clEnqueueSVMUnmap)(
    cl_command_queue command_queue,
    void* svm_ptr,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event);
#else
typedef void* _sclpfn_clEnqueueSVMUnmap;
#endif

#ifdef CL_VERSION_2_1
typedef cl_int (CL_API_CALL *_sclpfn_clSetDefaultDeviceCommandQueue)(
    cl_context context,
    cl_device_id device,
    cl_command_queue command_queue);
#else
typedef void* _sclpfn_clSetDefaultDeviceCommandQueue;
#endif

#ifdef CL_VERSION_2_1
typedef cl_int (CL_API_CALL *_sclpfn_clGetDeviceAndHostTimer)(
    cl_device_id device,
    cl_ulong* device_timestamp,
    cl_ulong* host_timestamp);
#else
typedef void* _sclpfn_clGetDeviceAndHostTimer;
#endif

#ifdef CL_VERSION_2_1
typedef cl_int (CL_API_CALL *_sclpfn_clGetHostTimer)(
    cl_device_id device,
    cl_ulong* host_timestamp);
#else
typedef void* _sclpfn_clGetHostTimer;
#endif

#ifdef CL_VERSION_2_1
typedef cl_program (CL_API_CALL *_sclpfn_clCreateProgramWithIL)(
    cl_context context,
    const void* il,
    size_t length,
    cl_int* errcode_ret);
#else
typedef void* _sclpfn_clCreateProgramWithIL;
#endif

#ifdef CL_VERSION_2_1
typedef cl_kernel (CL_API_CALL *_sclpfn_clCloneKernel)(
    cl_kernel source_kernel,
    cl_int* errcode_ret);
#else
typedef void* _sclpfn_clCloneKernel;
#endif

#ifdef CL_VERSION_2_1
typedef cl_int (CL_API_CALL *_sclpfn_clGetKernelSubGroupInfo)(
    cl_kernel kernel,
    cl_device_id device,
    cl_kernel_sub_group_info param_name,
    size_t input_value_size,
    const void* input_value,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret);
#else
typedef void* _sclpfn_clGetKernelSubGroupInfo;
#endif

#ifdef CL_VERSION_2_1
typedef cl_int (CL_API_CALL *_sclpfn_clEnqueueSVMMigrateMem)(
    cl_command_queue command_queue,
    cl_uint num_svm_pointers,
    const void** svm_pointers,
    const size_t* sizes,
    cl_mem_migration_flags flags,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event);
#else
typedef void* _sclpfn_clEnqueueSVMMigrateMem;
#endif

#ifdef CL_VERSION_2_2
typedef cl_int (CL_API_CALL *_sclpfn_clSetProgramReleaseCallback)(
    cl_program program,
    void (CL_CALLBACK* pfn_notify)(cl_program program, void* user_data),
    void* user_data);
#else
typedef void* _sclpfn_clSetProgramReleaseCallback;
#endif

#ifdef CL_VERSION_2_2
typedef cl_int (CL_API_CALL *_sclpfn_clSetProgramSpecializationConstant)(
    cl_program program,
    cl_uint spec_id,
    size_t spec_size,
    const void* spec_value);
#else
typedef void* _sclpfn_clSetProgramSpecializationConstant;
#endif

#ifdef CL_VERSION_3_0
typedef cl_mem (CL_API_CALL *_sclpfn_clCreateBufferWithProperties)(
    cl_context context,
    const cl_mem_properties* properties,
    cl_mem_flags flags,
    size_t size,
    void* host_ptr,
    cl_int* errcode_ret);
#else
typedef void* _sclpfn_clCreateBufferWithProperties;
#endif

#ifdef CL_VERSION_3_0
typedef cl_mem (CL_API_CALL *_sclpfn_clCreateImageWithProperties)(
    cl_context context,
    const cl_mem_properties* properties,
    cl_mem_flags flags,
    const cl_image_format* image_format,
    const cl_image_desc* image_desc,
    void* host_ptr,
    cl_int* errcode_ret);
#else
typedef void* _sclpfn_clCreateImageWithProperties;
#endif

///////////////////////////////////////////////////////////////////////////////
// Dispatch Table - this must match the Khronos ICD loader!

struct _sclDispatchTable {
    /* OpenCL 1.0 */
    _sclpfn_clGetPlatformIDs                         clGetPlatformIDs;
    _sclpfn_clGetPlatformInfo                        clGetPlatformInfo;
    _sclpfn_clGetDeviceIDs                           clGetDeviceIDs;
    _sclpfn_clGetDeviceInfo                          clGetDeviceInfo;
    _sclpfn_clCreateContext                          clCreateContext;
    _sclpfn_clCreateContextFromType                  clCreateContextFromType;
    _sclpfn_clRetainContext                          clRetainContext;
    _sclpfn_clReleaseContext                         clReleaseContext;
    _sclpfn_clGetContextInfo                         clGetContextInfo;
    _sclpfn_clCreateCommandQueue                     clCreateCommandQueue;
    _sclpfn_clRetainCommandQueue                     clRetainCommandQueue;
    _sclpfn_clReleaseCommandQueue                    clReleaseCommandQueue;
    _sclpfn_clGetCommandQueueInfo                    clGetCommandQueueInfo;
    _sclpfn_clSetCommandQueueProperty                clSetCommandQueueProperty;
    _sclpfn_clCreateBuffer                           clCreateBuffer;
    _sclpfn_clCreateImage2D                          clCreateImage2D;
    _sclpfn_clCreateImage3D                          clCreateImage3D;
    _sclpfn_clRetainMemObject                        clRetainMemObject;
    _sclpfn_clReleaseMemObject                       clReleaseMemObject;
    _sclpfn_clGetSupportedImageFormats               clGetSupportedImageFormats;
    _sclpfn_clGetMemObjectInfo                       clGetMemObjectInfo;
    _sclpfn_clGetImageInfo                           clGetImageInfo;
    _sclpfn_clCreateSampler                          clCreateSampler;
    _sclpfn_clRetainSampler                          clRetainSampler;
    _sclpfn_clReleaseSampler                         clReleaseSampler;
    _sclpfn_clGetSamplerInfo                         clGetSamplerInfo;
    _sclpfn_clCreateProgramWithSource                clCreateProgramWithSource;
    _sclpfn_clCreateProgramWithBinary                clCreateProgramWithBinary;
    _sclpfn_clRetainProgram                          clRetainProgram;
    _sclpfn_clReleaseProgram                         clReleaseProgram;
    _sclpfn_clBuildProgram                           clBuildProgram;
    _sclpfn_clUnloadCompiler                         clUnloadCompiler;
    _sclpfn_clGetProgramInfo                         clGetProgramInfo;
    _sclpfn_clGetProgramBuildInfo                    clGetProgramBuildInfo;
    _sclpfn_clCreateKernel                           clCreateKernel;
    _sclpfn_clCreateKernelsInProgram                 clCreateKernelsInProgram;
    _sclpfn_clRetainKernel                           clRetainKernel;
    _sclpfn_clReleaseKernel                          clReleaseKernel;
    _sclpfn_clSetKernelArg                           clSetKernelArg;
    _sclpfn_clGetKernelInfo                          clGetKernelInfo;
    _sclpfn_clGetKernelWorkGroupInfo                 clGetKernelWorkGroupInfo;
    _sclpfn_clWaitForEvents                          clWaitForEvents;
    _sclpfn_clGetEventInfo                           clGetEventInfo;
    _sclpfn_clRetainEvent                            clRetainEvent;
    _sclpfn_clReleaseEvent                           clReleaseEvent;
    _sclpfn_clGetEventProfilingInfo                  clGetEventProfilingInfo;
    _sclpfn_clFlush                                  clFlush;
    _sclpfn_clFinish                                 clFinish;
    _sclpfn_clEnqueueReadBuffer                      clEnqueueReadBuffer;
    _sclpfn_clEnqueueWriteBuffer                     clEnqueueWriteBuffer;
    _sclpfn_clEnqueueCopyBuffer                      clEnqueueCopyBuffer;
    _sclpfn_clEnqueueReadImage                       clEnqueueReadImage;
    _sclpfn_clEnqueueWriteImage                      clEnqueueWriteImage;
    _sclpfn_clEnqueueCopyImage                       clEnqueueCopyImage;
    _sclpfn_clEnqueueCopyImageToBuffer               clEnqueueCopyImageToBuffer;
    _sclpfn_clEnqueueCopyBufferToImage               clEnqueueCopyBufferToImage;
    _sclpfn_clEnqueueMapBuffer                       clEnqueueMapBuffer;
    _sclpfn_clEnqueueMapImage                        clEnqueueMapImage;
    _sclpfn_clEnqueueUnmapMemObject                  clEnqueueUnmapMemObject;
    _sclpfn_clEnqueueNDRangeKernel                   clEnqueueNDRangeKernel;
    _sclpfn_clEnqueueTask                            clEnqueueTask;
    _sclpfn_clEnqueueNativeKernel                    clEnqueueNativeKernel;
    _sclpfn_clEnqueueMarker                          clEnqueueMarker;
    _sclpfn_clEnqueueWaitForEvents                   clEnqueueWaitForEvents;
    _sclpfn_clEnqueueBarrier                         clEnqueueBarrier;
    _sclpfn_clGetExtensionFunctionAddress            clGetExtensionFunctionAddress;
    void* /* _sclpfn_clCreateFromGLBuffer       */   clCreateFromGLBuffer;
    void* /* _sclpfn_clCreateFromGLTexture2D    */   clCreateFromGLTexture2D;
    void* /* _sclpfn_clCreateFromGLTexture3D    */   clCreateFromGLTexture3D;
    void* /* _sclpfn_clCreateFromGLRenderbuffer */   clCreateFromGLRenderbuffer;
    void* /* _sclpfn_clGetGLObjectInfo          */   clGetGLObjectInfo;
    void* /* _sclpfn_clGetGLTextureInfo         */   clGetGLTextureInfo;
    void* /* _sclpfn_clEnqueueAcquireGLObjects  */   clEnqueueAcquireGLObjects;
    void* /* _sclpfn_clEnqueueReleaseGLObjects  */   clEnqueueReleaseGLObjects;
    void* /* _sclpfn_clGetGLContextInfoKHR      */   clGetGLContextInfoKHR;

    /* cl_khr_d3d10_sharing */
    void* /* _sclpfn_clGetDeviceIDsFromD3D10KHR      */ clGetDeviceIDsFromD3D10KHR;
    void* /* _sclpfn_clCreateFromD3D10BufferKHR      */ clCreateFromD3D10BufferKHR;
    void* /* _sclpfn_clCreateFromD3D10Texture2DKHR   */ clCreateFromD3D10Texture2DKHR;
    void* /* _sclpfn_clCreateFromD3D10Texture3DKHR   */ clCreateFromD3D10Texture3DKHR;
    void* /* _sclpfn_clEnqueueAcquireD3D10ObjectsKHR */ clEnqueueAcquireD3D10ObjectsKHR;
    void* /* _sclpfn_clEnqueueReleaseD3D10ObjectsKHR */ clEnqueueReleaseD3D10ObjectsKHR;

    /* OpenCL 1.1 */
    _sclpfn_clSetEventCallback                       clSetEventCallback;
    _sclpfn_clCreateSubBuffer                        clCreateSubBuffer;
    _sclpfn_clSetMemObjectDestructorCallback         clSetMemObjectDestructorCallback;
    _sclpfn_clCreateUserEvent                        clCreateUserEvent;
    _sclpfn_clSetUserEventStatus                     clSetUserEventStatus;
    _sclpfn_clEnqueueReadBufferRect                  clEnqueueReadBufferRect;
    _sclpfn_clEnqueueWriteBufferRect                 clEnqueueWriteBufferRect;
    _sclpfn_clEnqueueCopyBufferRect                  clEnqueueCopyBufferRect;

    /* cl_ext_device_fission */
    void* /* _sclpfn_clCreateSubDevicesEXT */       clCreateSubDevicesEXT;
    void* /* _sclpfn_clRetainDeviceEXT     */       clRetainDeviceEXT;
    void* /* _sclpfn_clReleaseDeviceEXT    */       clReleaseDeviceEXT;

    /* cl_khr_gl_event */
    void* /* _sclpfn_clCreateEventFromGLsyncKHR */  clCreateEventFromGLsyncKHR;

    /* OpenCL 1.2 */
    _sclpfn_clCreateSubDevices                      clCreateSubDevices;
    _sclpfn_clRetainDevice                          clRetainDevice;
    _sclpfn_clReleaseDevice                         clReleaseDevice;
    _sclpfn_clCreateImage                           clCreateImage;
    _sclpfn_clCreateProgramWithBuiltInKernels       clCreateProgramWithBuiltInKernels;
    _sclpfn_clCompileProgram                        clCompileProgram;
    _sclpfn_clLinkProgram                           clLinkProgram;
    _sclpfn_clUnloadPlatformCompiler                clUnloadPlatformCompiler;
    _sclpfn_clGetKernelArgInfo                      clGetKernelArgInfo;
    _sclpfn_clEnqueueFillBuffer                     clEnqueueFillBuffer;
    _sclpfn_clEnqueueFillImage                      clEnqueueFillImage;
    _sclpfn_clEnqueueMigrateMemObjects              clEnqueueMigrateMemObjects;
    _sclpfn_clEnqueueMarkerWithWaitList             clEnqueueMarkerWithWaitList;
    _sclpfn_clEnqueueBarrierWithWaitList            clEnqueueBarrierWithWaitList;
    _sclpfn_clGetExtensionFunctionAddressForPlatform clGetExtensionFunctionAddressForPlatform;
    void* /* _sclpfn_clCreateFromGLTexture */       clCreateFromGLTexture;

    /* cl_khr_d3d11_sharing */
    void* /* _sclpfn_clGetDeviceIDsFromD3D11KHR      */ clGetDeviceIDsFromD3D11KHR;
    void* /* _sclpfn_clCreateFromD3D11BufferKHR      */ clCreateFromD3D11BufferKHR;
    void* /* _sclpfn_clCreateFromD3D11Texture2DKHR   */ clCreateFromD3D11Texture2DKHR;
    void* /* _sclpfn_clCreateFromD3D11Texture3DKHR   */ clCreateFromD3D11Texture3DKHR;
    void* /* _sclpfn_clCreateFromDX9MediaSurfaceKHR  */ clCreateFromDX9MediaSurfaceKHR;
    void* /* _sclpfn_clEnqueueAcquireD3D11ObjectsKHR */ clEnqueueAcquireD3D11ObjectsKHR;
    void* /* _sclpfn_clEnqueueReleaseD3D11ObjectsKHR */ clEnqueueReleaseD3D11ObjectsKHR;

    /* cl_khr_dx9_media_sharing */
    void* /* _sclpfn_clGetDeviceIDsFromDX9MediaAdapterKHR */    clGetDeviceIDsFromDX9MediaAdapterKHR;
    void* /* _sclpfn_clEnqueueAcquireDX9MediaSurfacesKHR  */    clEnqueueAcquireDX9MediaSurfacesKHR;
    void* /* _sclpfn_clEnqueueReleaseDX9MediaSurfacesKHR  */    clEnqueueReleaseDX9MediaSurfacesKHR;

    /* cl_khr_egl_image */
    void* /* _sclpfn_clCreateFromEGLImageKHR       */   clCreateFromEGLImageKHR;
    void* /* _sclpfn_clEnqueueAcquireEGLObjectsKHR */   clEnqueueAcquireEGLObjectsKHR;
    void* /* _sclpfn_clEnqueueReleaseEGLObjectsKHR */   clEnqueueReleaseEGLObjectsKHR;

    /* cl_khr_egl_event */
    void* /* _sclpfn_clCreateEventFromEGLSyncKHR  */    clCreateEventFromEGLSyncKHR;

    /* OpenCL 2.0 */
    _sclpfn_clCreateCommandQueueWithProperties      clCreateCommandQueueWithProperties;
    _sclpfn_clCreatePipe                            clCreatePipe;
    _sclpfn_clGetPipeInfo                           clGetPipeInfo;
    _sclpfn_clSVMAlloc                              clSVMAlloc;
    _sclpfn_clSVMFree                               clSVMFree;
    _sclpfn_clEnqueueSVMFree                        clEnqueueSVMFree;
    _sclpfn_clEnqueueSVMMemcpy                      clEnqueueSVMMemcpy;
    _sclpfn_clEnqueueSVMMemFill                     clEnqueueSVMMemFill;
    _sclpfn_clEnqueueSVMMap                         clEnqueueSVMMap;
    _sclpfn_clEnqueueSVMUnmap                       clEnqueueSVMUnmap;
    _sclpfn_clCreateSamplerWithProperties           clCreateSamplerWithProperties;
    _sclpfn_clSetKernelArgSVMPointer                clSetKernelArgSVMPointer;
    _sclpfn_clSetKernelExecInfo                     clSetKernelExecInfo;

    /* cl_khr_sub_groups */
    void* /* _sclpfn_clGetKernelSubGroupInfoKHR */  clGetKernelSubGroupInfoKHR;

    /* OpenCL 2.1 */
    _sclpfn_clCloneKernel                           clCloneKernel;
    _sclpfn_clCreateProgramWithIL                   clCreateProgramWithIL;
    _sclpfn_clEnqueueSVMMigrateMem                  clEnqueueSVMMigrateMem;
    _sclpfn_clGetDeviceAndHostTimer                 clGetDeviceAndHostTimer;
    _sclpfn_clGetHostTimer                          clGetHostTimer;
    _sclpfn_clGetKernelSubGroupInfo                 clGetKernelSubGroupInfo;
    _sclpfn_clSetDefaultDeviceCommandQueue          clSetDefaultDeviceCommandQueue;

    /* OpenCL 2.2 */
    _sclpfn_clSetProgramReleaseCallback             clSetProgramReleaseCallback;
    _sclpfn_clSetProgramSpecializationConstant      clSetProgramSpecializationConstant;
};

struct _cl_platform_id {
    _sclDispatchTable *dispatch;
};

struct _cl_device_id {
    _sclDispatchTable *dispatch;
};

struct _cl_context {
    _sclDispatchTable *dispatch;
};

struct _cl_command_queue {
    _sclDispatchTable *dispatch;
};

struct _cl_mem {
    _sclDispatchTable *dispatch;
};

struct _cl_program {
    _sclDispatchTable *dispatch;
};

struct _cl_kernel {
    _sclDispatchTable *dispatch;
};

struct _cl_event {
    _sclDispatchTable *dispatch;
};

struct _cl_sampler {
    _sclDispatchTable *dispatch;
};

///////////////////////////////////////////////////////////////////////////////
// Manually written API function definitions:

// This error code is defined by the ICD extension, but it may not have
// been included yet:
#ifdef CL_PLATFORM_NOT_FOUND_KHR
#define _SCL_PLATFORM_NOT_FOUND_KHR CL_PLATFORM_NOT_FOUND_KHR
#else
#define _SCL_PLATFORM_NOT_FOUND_KHR -1001
#endif

#ifdef _WIN32
typedef HMODULE _sclModuleHandle;
#define _sclOpenICDLoader()                     ::LoadLibraryA("OpenCL.dll")
#define _sclGetFunctionAddress(_module, _name)  ::GetProcAddress(_module, _name)
#endif
#ifdef __linux__
typedef void*   _sclModuleHandle;
#define _sclOpenICDLoader()                     ::dlopen("libOpenCL.so", RTLD_LAZY | RTLD_LOCAL)
#define _sclGetFunctionAddress(_module, _name)  ::dlsym(_module, _name)
#endif

// This is a helper function to find a platform from context properties:
static inline cl_platform_id _sclGetPlatfromFromContextProperties(
    const cl_context_properties* properties)
{
    if (properties != NULL) {
        while (properties[0] != 0 ) {
            if (CL_CONTEXT_PLATFORM == (cl_int)properties[0]) {
                cl_platform_id platform = (cl_platform_id)properties[1];
                return platform;
            }
            properties += 2;
        }
    }
    return NULL;
}

#ifdef __cplusplus
extern "C" {
#endif

CL_API_ENTRY cl_int CL_API_CALL clGetPlatformIDs(
    cl_uint num_entries,
    cl_platform_id* platforms,
    cl_uint* num_platforms)
{
    static _sclModuleHandle module = _sclOpenICDLoader();
    _sclpfn_clGetPlatformIDs _clGetPlatformIDs = 
        (_sclpfn_clGetPlatformIDs)_sclGetFunctionAddress(
            module, "clGetPlatformIDs");
    _sclpfn_clGetExtensionFunctionAddressForPlatform _clGetExtensionFunctionAddressForPlatform =
        (_sclpfn_clGetExtensionFunctionAddressForPlatform)_sclGetFunctionAddress(
            module, "clGetExtensionFunctionAddressForPlatform");

    // Basic error checks:
    if ((platforms == NULL && num_entries != 0) ||
        (platforms == NULL && num_platforms == NULL)) {
        return CL_INVALID_VALUE;
    }

    if (_clGetPlatformIDs) {
        // Only return platforms that support the ICD extension.
        cl_int errorCode = CL_SUCCESS;
        cl_platform_id* all_platforms = NULL;
        cl_uint total_num_platforms = 0;
        cl_uint num_icd_platforms = 0;
        cl_uint p = 0;

        // Get the total number of platforms:
        errorCode = _clGetPlatformIDs(0, NULL, &total_num_platforms);
        if (errorCode != CL_SUCCESS) {
            return errorCode;
        }
        if (total_num_platforms >= 0) {
            // Sanity check:
            if (total_num_platforms > _SCL_MAX_NUM_PLATFORMS) {
                total_num_platforms = _SCL_MAX_NUM_PLATFORMS;
            }

            all_platforms = (cl_platform_id*)alloca(
                total_num_platforms * sizeof(cl_platform_id));
            errorCode = _clGetPlatformIDs(total_num_platforms, all_platforms, NULL);
            if (errorCode != CL_SUCCESS) {
                return errorCode;
            }

            for (p = 0; p < total_num_platforms; p++) {
                if (_clGetExtensionFunctionAddressForPlatform(
                        all_platforms[p], "clIcdGetPlatformIDsKHR")) {
                    if (num_icd_platforms < num_entries && platforms != NULL) {
                        platforms[num_icd_platforms] = all_platforms[p];
                    }
                    num_icd_platforms++;
                }
            }

            if (num_platforms) {
                num_platforms[0] = num_icd_platforms;
            }

            return CL_SUCCESS;
        }
    }

    // The cl_khr_icd spec says that an error should be returned if no
    // platforms are found, but this is not an error condition in the OpenCL
    // spec.
#if 1
    return _SCL_PLATFORM_NOT_FOUND_KHR;
#else
    if (num_platforms) {
        num_platforms[0] = 0;
    }
    return CL_SUCCESS;
#endif
}

CL_API_ENTRY void* CL_API_CALL clGetExtensionFunctionAddress(
    const char* function_name)
{
#if 0
    static _sclModuleHandle module = _sclOpenICDLoader();
    _sclpfn_clGetExtensionFunctionAddress _clGetExtensionFunctionAddress =
        (_sclpfn_clGetExtensionFunctionAddress)::GetProcAddress(
            module, "clGetExtensionFunctionAddress");
    if (_clGetExtensionFunctionAddress) {
        return _clGetExtensionFunctionAddress(function_name);
    }
#endif
    return NULL;
}

CL_API_ENTRY cl_int CL_API_CALL clUnloadCompiler(void)
{
    return CL_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
// Generated API function definitions:

#ifdef CL_VERSION_1_0

CL_API_ENTRY cl_int CL_API_CALL clGetPlatformInfo(
    cl_platform_id platform,
    cl_platform_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(platform, CL_INVALID_PLATFORM);
    return platform->dispatch->clGetPlatformInfo(
        platform,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_0

CL_API_ENTRY cl_int CL_API_CALL clGetDeviceIDs(
    cl_platform_id platform,
    cl_device_type device_type,
    cl_uint num_entries,
    cl_device_id* devices,
    cl_uint* num_devices)
{
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(platform, CL_INVALID_PLATFORM);
    return platform->dispatch->clGetDeviceIDs(
        platform,
        device_type,
        num_entries,
        devices,
        num_devices);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_0

CL_API_ENTRY cl_int CL_API_CALL clGetDeviceInfo(
    cl_device_id device,
    cl_device_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(device, CL_INVALID_DEVICE);
    return device->dispatch->clGetDeviceInfo(
        device,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_0

CL_API_ENTRY cl_context CL_API_CALL clCreateContext(
    const cl_context_properties* properties,
    cl_uint num_devices,
    const cl_device_id* devices,
    void (CL_CALLBACK* pfn_notify)(const char* errinfo, const void* private_info, size_t cb, void* user_data),
    void* user_data,
    cl_int* errcode_ret)
{
    if (num_devices == 0 || devices == NULL) {
        _SCL_VALIDATE_HANDLE_RETURN_HANDLE(NULL, CL_INVALID_VALUE);
    }
    _SCL_VALIDATE_HANDLE_RETURN_HANDLE(devices[0], CL_INVALID_DEVICE);
    return devices[0]->dispatch->clCreateContext(
        properties,
        num_devices,
        devices,
        pfn_notify,
        user_data,
        errcode_ret);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_0

CL_API_ENTRY cl_context CL_API_CALL clCreateContextFromType(
    const cl_context_properties* properties,
    cl_device_type device_type,
    void (CL_CALLBACK* pfn_notify)(const char* errinfo, const void* private_info, size_t cb, void* user_data),
    void* user_data,
    cl_int* errcode_ret)
{
    cl_platform_id platform = _sclGetPlatfromFromContextProperties(properties);
    _SCL_VALIDATE_HANDLE_RETURN_HANDLE(platform, CL_INVALID_PLATFORM);
    return platform->dispatch->clCreateContextFromType(
        properties,
        device_type,
        pfn_notify,
        user_data,
        errcode_ret);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_0

CL_API_ENTRY cl_int CL_API_CALL clRetainContext(
    cl_context context)
{
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(context, CL_INVALID_CONTEXT);
    return context->dispatch->clRetainContext(
        context);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_0

CL_API_ENTRY cl_int CL_API_CALL clReleaseContext(
    cl_context context)
{
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(context, CL_INVALID_CONTEXT);
    return context->dispatch->clReleaseContext(
        context);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_0

CL_API_ENTRY cl_int CL_API_CALL clGetContextInfo(
    cl_context context,
    cl_context_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(context, CL_INVALID_CONTEXT);
    return context->dispatch->clGetContextInfo(
        context,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_0

CL_API_ENTRY cl_int CL_API_CALL clRetainCommandQueue(
    cl_command_queue command_queue)
{
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return command_queue->dispatch->clRetainCommandQueue(
        command_queue);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_0

CL_API_ENTRY cl_int CL_API_CALL clReleaseCommandQueue(
    cl_command_queue command_queue)
{
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return command_queue->dispatch->clReleaseCommandQueue(
        command_queue);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_0

CL_API_ENTRY cl_int CL_API_CALL clGetCommandQueueInfo(
    cl_command_queue command_queue,
    cl_command_queue_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return command_queue->dispatch->clGetCommandQueueInfo(
        command_queue,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_0

CL_API_ENTRY cl_mem CL_API_CALL clCreateBuffer(
    cl_context context,
    cl_mem_flags flags,
    size_t size,
    void* host_ptr,
    cl_int* errcode_ret)
{
    _SCL_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    return context->dispatch->clCreateBuffer(
        context,
        flags,
        size,
        host_ptr,
        errcode_ret);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_0

CL_API_ENTRY cl_int CL_API_CALL clRetainMemObject(
    cl_mem memobj)
{
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(memobj, CL_INVALID_MEM_OBJECT);
    return memobj->dispatch->clRetainMemObject(
        memobj);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_0

CL_API_ENTRY cl_int CL_API_CALL clReleaseMemObject(
    cl_mem memobj)
{
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(memobj, CL_INVALID_MEM_OBJECT);
    return memobj->dispatch->clReleaseMemObject(
        memobj);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_0

CL_API_ENTRY cl_int CL_API_CALL clGetSupportedImageFormats(
    cl_context context,
    cl_mem_flags flags,
    cl_mem_object_type image_type,
    cl_uint num_entries,
    cl_image_format* image_formats,
    cl_uint* num_image_formats)
{
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(context, CL_INVALID_CONTEXT);
    return context->dispatch->clGetSupportedImageFormats(
        context,
        flags,
        image_type,
        num_entries,
        image_formats,
        num_image_formats);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_0

CL_API_ENTRY cl_int CL_API_CALL clGetMemObjectInfo(
    cl_mem memobj,
    cl_mem_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(memobj, CL_INVALID_MEM_OBJECT);
    return memobj->dispatch->clGetMemObjectInfo(
        memobj,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_0

CL_API_ENTRY cl_int CL_API_CALL clGetImageInfo(
    cl_mem image,
    cl_image_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(image, CL_INVALID_MEM_OBJECT);
    return image->dispatch->clGetImageInfo(
        image,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_0

CL_API_ENTRY cl_int CL_API_CALL clRetainSampler(
    cl_sampler sampler)
{
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(sampler, CL_INVALID_SAMPLER);
    return sampler->dispatch->clRetainSampler(
        sampler);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_0

CL_API_ENTRY cl_int CL_API_CALL clReleaseSampler(
    cl_sampler sampler)
{
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(sampler, CL_INVALID_SAMPLER);
    return sampler->dispatch->clReleaseSampler(
        sampler);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_0

CL_API_ENTRY cl_int CL_API_CALL clGetSamplerInfo(
    cl_sampler sampler,
    cl_sampler_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(sampler, CL_INVALID_SAMPLER);
    return sampler->dispatch->clGetSamplerInfo(
        sampler,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_0

CL_API_ENTRY cl_program CL_API_CALL clCreateProgramWithSource(
    cl_context context,
    cl_uint count,
    const char** strings,
    const size_t* lengths,
    cl_int* errcode_ret)
{
    _SCL_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    return context->dispatch->clCreateProgramWithSource(
        context,
        count,
        strings,
        lengths,
        errcode_ret);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_0

CL_API_ENTRY cl_program CL_API_CALL clCreateProgramWithBinary(
    cl_context context,
    cl_uint num_devices,
    const cl_device_id* device_list,
    const size_t* lengths,
    const unsigned char** binaries,
    cl_int* binary_status,
    cl_int* errcode_ret)
{
    _SCL_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    return context->dispatch->clCreateProgramWithBinary(
        context,
        num_devices,
        device_list,
        lengths,
        binaries,
        binary_status,
        errcode_ret);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_0

CL_API_ENTRY cl_int CL_API_CALL clRetainProgram(
    cl_program program)
{
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(program, CL_INVALID_PROGRAM);
    return program->dispatch->clRetainProgram(
        program);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_0

CL_API_ENTRY cl_int CL_API_CALL clReleaseProgram(
    cl_program program)
{
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(program, CL_INVALID_PROGRAM);
    return program->dispatch->clReleaseProgram(
        program);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_0

CL_API_ENTRY cl_int CL_API_CALL clBuildProgram(
    cl_program program,
    cl_uint num_devices,
    const cl_device_id* device_list,
    const char* options,
    void (CL_CALLBACK* pfn_notify)(cl_program program, void* user_data),
    void* user_data)
{
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(program, CL_INVALID_PROGRAM);
    return program->dispatch->clBuildProgram(
        program,
        num_devices,
        device_list,
        options,
        pfn_notify,
        user_data);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_0

CL_API_ENTRY cl_int CL_API_CALL clGetProgramInfo(
    cl_program program,
    cl_program_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(program, CL_INVALID_PROGRAM);
    return program->dispatch->clGetProgramInfo(
        program,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_0

CL_API_ENTRY cl_int CL_API_CALL clGetProgramBuildInfo(
    cl_program program,
    cl_device_id device,
    cl_program_build_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(program, CL_INVALID_PROGRAM);
    return program->dispatch->clGetProgramBuildInfo(
        program,
        device,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_0

CL_API_ENTRY cl_kernel CL_API_CALL clCreateKernel(
    cl_program program,
    const char* kernel_name,
    cl_int* errcode_ret)
{
    _SCL_VALIDATE_HANDLE_RETURN_HANDLE(program, CL_INVALID_PROGRAM);
    return program->dispatch->clCreateKernel(
        program,
        kernel_name,
        errcode_ret);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_0

CL_API_ENTRY cl_int CL_API_CALL clCreateKernelsInProgram(
    cl_program program,
    cl_uint num_kernels,
    cl_kernel* kernels,
    cl_uint* num_kernels_ret)
{
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(program, CL_INVALID_PROGRAM);
    return program->dispatch->clCreateKernelsInProgram(
        program,
        num_kernels,
        kernels,
        num_kernels_ret);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_0

CL_API_ENTRY cl_int CL_API_CALL clRetainKernel(
    cl_kernel kernel)
{
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(kernel, CL_INVALID_KERNEL);
    return kernel->dispatch->clRetainKernel(
        kernel);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_0

CL_API_ENTRY cl_int CL_API_CALL clReleaseKernel(
    cl_kernel kernel)
{
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(kernel, CL_INVALID_KERNEL);
    return kernel->dispatch->clReleaseKernel(
        kernel);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_0

CL_API_ENTRY cl_int CL_API_CALL clSetKernelArg(
    cl_kernel kernel,
    cl_uint arg_index,
    size_t arg_size,
    const void* arg_value)
{
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(kernel, CL_INVALID_KERNEL);
    return kernel->dispatch->clSetKernelArg(
        kernel,
        arg_index,
        arg_size,
        arg_value);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_0

CL_API_ENTRY cl_int CL_API_CALL clGetKernelInfo(
    cl_kernel kernel,
    cl_kernel_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(kernel, CL_INVALID_KERNEL);
    return kernel->dispatch->clGetKernelInfo(
        kernel,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_0

CL_API_ENTRY cl_int CL_API_CALL clGetKernelWorkGroupInfo(
    cl_kernel kernel,
    cl_device_id device,
    cl_kernel_work_group_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(kernel, CL_INVALID_KERNEL);
    return kernel->dispatch->clGetKernelWorkGroupInfo(
        kernel,
        device,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_0

CL_API_ENTRY cl_int CL_API_CALL clWaitForEvents(
    cl_uint num_events,
    const cl_event* event_list)
{
    if (num_events == 0 || event_list == NULL) {
        return CL_INVALID_VALUE;
    }
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(event_list[0], CL_INVALID_EVENT);
    return event_list[0]->dispatch->clWaitForEvents(
        num_events,
        event_list);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_0

CL_API_ENTRY cl_int CL_API_CALL clGetEventInfo(
    cl_event event,
    cl_event_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(event, CL_INVALID_EVENT);
    return event->dispatch->clGetEventInfo(
        event,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_0

CL_API_ENTRY cl_int CL_API_CALL clRetainEvent(
    cl_event event)
{
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(event, CL_INVALID_EVENT);
    return event->dispatch->clRetainEvent(
        event);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_0

CL_API_ENTRY cl_int CL_API_CALL clReleaseEvent(
    cl_event event)
{
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(event, CL_INVALID_EVENT);
    return event->dispatch->clReleaseEvent(
        event);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_0

CL_API_ENTRY cl_int CL_API_CALL clGetEventProfilingInfo(
    cl_event event,
    cl_profiling_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(event, CL_INVALID_EVENT);
    return event->dispatch->clGetEventProfilingInfo(
        event,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_0

CL_API_ENTRY cl_int CL_API_CALL clFlush(
    cl_command_queue command_queue)
{
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return command_queue->dispatch->clFlush(
        command_queue);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_0

CL_API_ENTRY cl_int CL_API_CALL clFinish(
    cl_command_queue command_queue)
{
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return command_queue->dispatch->clFinish(
        command_queue);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_0

CL_API_ENTRY cl_int CL_API_CALL clEnqueueReadBuffer(
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
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return command_queue->dispatch->clEnqueueReadBuffer(
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

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_0

CL_API_ENTRY cl_int CL_API_CALL clEnqueueWriteBuffer(
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
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return command_queue->dispatch->clEnqueueWriteBuffer(
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

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_0

CL_API_ENTRY cl_int CL_API_CALL clEnqueueCopyBuffer(
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
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return command_queue->dispatch->clEnqueueCopyBuffer(
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

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_0

CL_API_ENTRY cl_int CL_API_CALL clEnqueueReadImage(
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
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return command_queue->dispatch->clEnqueueReadImage(
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

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_0

CL_API_ENTRY cl_int CL_API_CALL clEnqueueWriteImage(
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
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return command_queue->dispatch->clEnqueueWriteImage(
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

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_0

CL_API_ENTRY cl_int CL_API_CALL clEnqueueCopyImage(
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
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return command_queue->dispatch->clEnqueueCopyImage(
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

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_0

CL_API_ENTRY cl_int CL_API_CALL clEnqueueCopyImageToBuffer(
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
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return command_queue->dispatch->clEnqueueCopyImageToBuffer(
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

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_0

CL_API_ENTRY cl_int CL_API_CALL clEnqueueCopyBufferToImage(
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
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return command_queue->dispatch->clEnqueueCopyBufferToImage(
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

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_0

CL_API_ENTRY void* CL_API_CALL clEnqueueMapBuffer(
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
    _SCL_VALIDATE_HANDLE_RETURN_HANDLE(command_queue, CL_INVALID_COMMAND_QUEUE);
    return command_queue->dispatch->clEnqueueMapBuffer(
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

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_0

CL_API_ENTRY void* CL_API_CALL clEnqueueMapImage(
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
    _SCL_VALIDATE_HANDLE_RETURN_HANDLE(command_queue, CL_INVALID_COMMAND_QUEUE);
    return command_queue->dispatch->clEnqueueMapImage(
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

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_0

CL_API_ENTRY cl_int CL_API_CALL clEnqueueUnmapMemObject(
    cl_command_queue command_queue,
    cl_mem memobj,
    void* mapped_ptr,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return command_queue->dispatch->clEnqueueUnmapMemObject(
        command_queue,
        memobj,
        mapped_ptr,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_0

CL_API_ENTRY cl_int CL_API_CALL clEnqueueNDRangeKernel(
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
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return command_queue->dispatch->clEnqueueNDRangeKernel(
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

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_0

CL_API_ENTRY cl_int CL_API_CALL clEnqueueNativeKernel(
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
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return command_queue->dispatch->clEnqueueNativeKernel(
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

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_0

CL_API_ENTRY cl_int CL_API_CALL clSetCommandQueueProperty(
    cl_command_queue command_queue,
    cl_command_queue_properties properties,
    cl_bool enable,
    cl_command_queue_properties* old_properties)
{
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return command_queue->dispatch->clSetCommandQueueProperty(
        command_queue,
        properties,
        enable,
        old_properties);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_0

CL_API_ENTRY cl_mem CL_API_CALL clCreateImage2D(
    cl_context context,
    cl_mem_flags flags,
    const cl_image_format* image_format,
    size_t image_width,
    size_t image_height,
    size_t image_row_pitch,
    void* host_ptr,
    cl_int* errcode_ret)
{
    _SCL_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    return context->dispatch->clCreateImage2D(
        context,
        flags,
        image_format,
        image_width,
        image_height,
        image_row_pitch,
        host_ptr,
        errcode_ret);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_0

CL_API_ENTRY cl_mem CL_API_CALL clCreateImage3D(
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
    _SCL_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    return context->dispatch->clCreateImage3D(
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

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_0

CL_API_ENTRY cl_int CL_API_CALL clEnqueueMarker(
    cl_command_queue command_queue,
    cl_event* event)
{
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return command_queue->dispatch->clEnqueueMarker(
        command_queue,
        event);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_0

CL_API_ENTRY cl_int CL_API_CALL clEnqueueWaitForEvents(
    cl_command_queue command_queue,
    cl_uint num_events,
    const cl_event* event_list)
{
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return command_queue->dispatch->clEnqueueWaitForEvents(
        command_queue,
        num_events,
        event_list);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_0

CL_API_ENTRY cl_int CL_API_CALL clEnqueueBarrier(
    cl_command_queue command_queue)
{
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return command_queue->dispatch->clEnqueueBarrier(
        command_queue);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_0

CL_API_ENTRY cl_command_queue CL_API_CALL clCreateCommandQueue(
    cl_context context,
    cl_device_id device,
    cl_command_queue_properties properties,
    cl_int* errcode_ret)
{
    _SCL_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    return context->dispatch->clCreateCommandQueue(
        context,
        device,
        properties,
        errcode_ret);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_0

CL_API_ENTRY cl_sampler CL_API_CALL clCreateSampler(
    cl_context context,
    cl_bool normalized_coords,
    cl_addressing_mode addressing_mode,
    cl_filter_mode filter_mode,
    cl_int* errcode_ret)
{
    _SCL_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    return context->dispatch->clCreateSampler(
        context,
        normalized_coords,
        addressing_mode,
        filter_mode,
        errcode_ret);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_0

CL_API_ENTRY cl_int CL_API_CALL clEnqueueTask(
    cl_command_queue command_queue,
    cl_kernel kernel,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return command_queue->dispatch->clEnqueueTask(
        command_queue,
        kernel,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_1

CL_API_ENTRY cl_mem CL_API_CALL clCreateSubBuffer(
    cl_mem buffer,
    cl_mem_flags flags,
    cl_buffer_create_type buffer_create_type,
    const void* buffer_create_info,
    cl_int* errcode_ret)
{
    _SCL_VALIDATE_HANDLE_RETURN_HANDLE(buffer, CL_INVALID_MEM_OBJECT);
    return buffer->dispatch->clCreateSubBuffer(
        buffer,
        flags,
        buffer_create_type,
        buffer_create_info,
        errcode_ret);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_1

CL_API_ENTRY cl_int CL_API_CALL clSetMemObjectDestructorCallback(
    cl_mem memobj,
    void (CL_CALLBACK* pfn_notify)(cl_mem memobj, void* user_data),
    void* user_data)
{
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(memobj, CL_INVALID_MEM_OBJECT);
    return memobj->dispatch->clSetMemObjectDestructorCallback(
        memobj,
        pfn_notify,
        user_data);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_1

CL_API_ENTRY cl_event CL_API_CALL clCreateUserEvent(
    cl_context context,
    cl_int* errcode_ret)
{
    _SCL_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    return context->dispatch->clCreateUserEvent(
        context,
        errcode_ret);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_1

CL_API_ENTRY cl_int CL_API_CALL clSetUserEventStatus(
    cl_event event,
    cl_int execution_status)
{
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(event, CL_INVALID_EVENT);
    return event->dispatch->clSetUserEventStatus(
        event,
        execution_status);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_1

CL_API_ENTRY cl_int CL_API_CALL clSetEventCallback(
    cl_event event,
    cl_int command_exec_callback_type,
    void (CL_CALLBACK* pfn_notify)(cl_event event, cl_int event_command_status, void *user_data),
    void* user_data)
{
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(event, CL_INVALID_EVENT);
    return event->dispatch->clSetEventCallback(
        event,
        command_exec_callback_type,
        pfn_notify,
        user_data);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_1

CL_API_ENTRY cl_int CL_API_CALL clEnqueueReadBufferRect(
    cl_command_queue command_queue,
    cl_mem buffer,
    cl_bool blocking_read,
    const size_t* buffer_offset,
    const size_t* host_offset,
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
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return command_queue->dispatch->clEnqueueReadBufferRect(
        command_queue,
        buffer,
        blocking_read,
        buffer_offset,
        host_offset,
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

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_1

CL_API_ENTRY cl_int CL_API_CALL clEnqueueWriteBufferRect(
    cl_command_queue command_queue,
    cl_mem buffer,
    cl_bool blocking_write,
    const size_t* buffer_offset,
    const size_t* host_offset,
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
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return command_queue->dispatch->clEnqueueWriteBufferRect(
        command_queue,
        buffer,
        blocking_write,
        buffer_offset,
        host_offset,
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

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_1

CL_API_ENTRY cl_int CL_API_CALL clEnqueueCopyBufferRect(
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
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return command_queue->dispatch->clEnqueueCopyBufferRect(
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

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_2

CL_API_ENTRY cl_int CL_API_CALL clCreateSubDevices(
    cl_device_id in_device,
    const cl_device_partition_property* properties,
    cl_uint num_devices,
    cl_device_id* out_devices,
    cl_uint* num_devices_ret)
{
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(in_device, CL_INVALID_DEVICE);
    return in_device->dispatch->clCreateSubDevices(
        in_device,
        properties,
        num_devices,
        out_devices,
        num_devices_ret);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_2

CL_API_ENTRY cl_int CL_API_CALL clRetainDevice(
    cl_device_id device)
{
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(device, CL_INVALID_DEVICE);
    return device->dispatch->clRetainDevice(
        device);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_2

CL_API_ENTRY cl_int CL_API_CALL clReleaseDevice(
    cl_device_id device)
{
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(device, CL_INVALID_DEVICE);
    return device->dispatch->clReleaseDevice(
        device);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_2

CL_API_ENTRY cl_mem CL_API_CALL clCreateImage(
    cl_context context,
    cl_mem_flags flags,
    const cl_image_format* image_format,
    const cl_image_desc* image_desc,
    void* host_ptr,
    cl_int* errcode_ret)
{
    _SCL_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    return context->dispatch->clCreateImage(
        context,
        flags,
        image_format,
        image_desc,
        host_ptr,
        errcode_ret);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_2

CL_API_ENTRY cl_program CL_API_CALL clCreateProgramWithBuiltInKernels(
    cl_context context,
    cl_uint num_devices,
    const cl_device_id* device_list,
    const char* kernel_names,
    cl_int* errcode_ret)
{
    _SCL_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    return context->dispatch->clCreateProgramWithBuiltInKernels(
        context,
        num_devices,
        device_list,
        kernel_names,
        errcode_ret);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_2

CL_API_ENTRY cl_int CL_API_CALL clCompileProgram(
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
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(program, CL_INVALID_PROGRAM);
    return program->dispatch->clCompileProgram(
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

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_2

CL_API_ENTRY cl_program CL_API_CALL clLinkProgram(
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
    _SCL_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    return context->dispatch->clLinkProgram(
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

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_2

CL_API_ENTRY cl_int CL_API_CALL clUnloadPlatformCompiler(
    cl_platform_id platform)
{
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(platform, CL_INVALID_PLATFORM);
    return platform->dispatch->clUnloadPlatformCompiler(
        platform);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_2

CL_API_ENTRY cl_int CL_API_CALL clGetKernelArgInfo(
    cl_kernel kernel,
    cl_uint arg_indx,
    cl_kernel_arg_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(kernel, CL_INVALID_KERNEL);
    return kernel->dispatch->clGetKernelArgInfo(
        kernel,
        arg_indx,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_2

CL_API_ENTRY cl_int CL_API_CALL clEnqueueFillBuffer(
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
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return command_queue->dispatch->clEnqueueFillBuffer(
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

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_2

CL_API_ENTRY cl_int CL_API_CALL clEnqueueFillImage(
    cl_command_queue command_queue,
    cl_mem image,
    const void* fill_color,
    const size_t* origin,
    const size_t* region,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return command_queue->dispatch->clEnqueueFillImage(
        command_queue,
        image,
        fill_color,
        origin,
        region,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_2

CL_API_ENTRY cl_int CL_API_CALL clEnqueueMigrateMemObjects(
    cl_command_queue command_queue,
    cl_uint num_mem_objects,
    const cl_mem* mem_objects,
    cl_mem_migration_flags flags,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return command_queue->dispatch->clEnqueueMigrateMemObjects(
        command_queue,
        num_mem_objects,
        mem_objects,
        flags,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_2

CL_API_ENTRY cl_int CL_API_CALL clEnqueueMarkerWithWaitList(
    cl_command_queue command_queue,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return command_queue->dispatch->clEnqueueMarkerWithWaitList(
        command_queue,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_2

CL_API_ENTRY cl_int CL_API_CALL clEnqueueBarrierWithWaitList(
    cl_command_queue command_queue,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return command_queue->dispatch->clEnqueueBarrierWithWaitList(
        command_queue,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_1_2

CL_API_ENTRY void* CL_API_CALL clGetExtensionFunctionAddressForPlatform(
    cl_platform_id platform,
    const char* func_name)
{
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(platform, NULL);
    return platform->dispatch->clGetExtensionFunctionAddressForPlatform(
        platform,
        func_name);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_2_0

CL_API_ENTRY cl_command_queue CL_API_CALL clCreateCommandQueueWithProperties(
    cl_context context,
    cl_device_id device,
    const cl_queue_properties* properties,
    cl_int* errcode_ret)
{
    _SCL_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    return context->dispatch->clCreateCommandQueueWithProperties(
        context,
        device,
        properties,
        errcode_ret);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_2_0

CL_API_ENTRY cl_mem CL_API_CALL clCreatePipe(
    cl_context context,
    cl_mem_flags flags,
    cl_uint pipe_packet_size,
    cl_uint pipe_max_packets,
    const cl_pipe_properties* properties,
    cl_int* errcode_ret)
{
    _SCL_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    return context->dispatch->clCreatePipe(
        context,
        flags,
        pipe_packet_size,
        pipe_max_packets,
        properties,
        errcode_ret);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_2_0

CL_API_ENTRY cl_int CL_API_CALL clGetPipeInfo(
    cl_mem pipe,
    cl_pipe_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(pipe, CL_INVALID_MEM_OBJECT);
    return pipe->dispatch->clGetPipeInfo(
        pipe,
        param_name,
        param_value_size,
        param_value,
        param_value_size_ret);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_2_0

CL_API_ENTRY void* CL_API_CALL clSVMAlloc(
    cl_context context,
    cl_svm_mem_flags flags,
    size_t size,
    cl_uint alignment)
{
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(context, NULL);
    return context->dispatch->clSVMAlloc(
        context,
        flags,
        size,
        alignment);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_2_0

CL_API_ENTRY void CL_API_CALL clSVMFree(
    cl_context context,
    void* svm_pointer)
{
    if (context == NULL) return;
    context->dispatch->clSVMFree(
        context,
        svm_pointer);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_2_0

CL_API_ENTRY cl_sampler CL_API_CALL clCreateSamplerWithProperties(
    cl_context context,
    const cl_sampler_properties* sampler_properties,
    cl_int* errcode_ret)
{
    _SCL_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    return context->dispatch->clCreateSamplerWithProperties(
        context,
        sampler_properties,
        errcode_ret);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_2_0

CL_API_ENTRY cl_int CL_API_CALL clSetKernelArgSVMPointer(
    cl_kernel kernel,
    cl_uint arg_index,
    const void* arg_value)
{
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(kernel, CL_INVALID_KERNEL);
    return kernel->dispatch->clSetKernelArgSVMPointer(
        kernel,
        arg_index,
        arg_value);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_2_0

CL_API_ENTRY cl_int CL_API_CALL clSetKernelExecInfo(
    cl_kernel kernel,
    cl_kernel_exec_info param_name,
    size_t param_value_size,
    const void* param_value)
{
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(kernel, CL_INVALID_KERNEL);
    return kernel->dispatch->clSetKernelExecInfo(
        kernel,
        param_name,
        param_value_size,
        param_value);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_2_0

CL_API_ENTRY cl_int CL_API_CALL clEnqueueSVMFree(
    cl_command_queue command_queue,
    cl_uint num_svm_pointers,
    void* svm_pointers[],
    void (CL_CALLBACK* pfn_free_func)(cl_command_queue queue, cl_uint num_svm_pointers, void* svm_pointers[], void* user_data),
    void* user_data,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return command_queue->dispatch->clEnqueueSVMFree(
        command_queue,
        num_svm_pointers,
        svm_pointers,
        pfn_free_func,
        user_data,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_2_0

CL_API_ENTRY cl_int CL_API_CALL clEnqueueSVMMemcpy(
    cl_command_queue command_queue,
    cl_bool blocking_copy,
    void* dst_ptr,
    const void* src_ptr,
    size_t size,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return command_queue->dispatch->clEnqueueSVMMemcpy(
        command_queue,
        blocking_copy,
        dst_ptr,
        src_ptr,
        size,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_2_0

CL_API_ENTRY cl_int CL_API_CALL clEnqueueSVMMemFill(
    cl_command_queue command_queue,
    void* svm_ptr,
    const void* pattern,
    size_t pattern_size,
    size_t size,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return command_queue->dispatch->clEnqueueSVMMemFill(
        command_queue,
        svm_ptr,
        pattern,
        pattern_size,
        size,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_2_0

CL_API_ENTRY cl_int CL_API_CALL clEnqueueSVMMap(
    cl_command_queue command_queue,
    cl_bool blocking_map,
    cl_map_flags flags,
    void* svm_ptr,
    size_t size,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return command_queue->dispatch->clEnqueueSVMMap(
        command_queue,
        blocking_map,
        flags,
        svm_ptr,
        size,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_2_0

CL_API_ENTRY cl_int CL_API_CALL clEnqueueSVMUnmap(
    cl_command_queue command_queue,
    void* svm_ptr,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return command_queue->dispatch->clEnqueueSVMUnmap(
        command_queue,
        svm_ptr,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_2_1

CL_API_ENTRY cl_int CL_API_CALL clSetDefaultDeviceCommandQueue(
    cl_context context,
    cl_device_id device,
    cl_command_queue command_queue)
{
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(context, CL_INVALID_CONTEXT);
    return context->dispatch->clSetDefaultDeviceCommandQueue(
        context,
        device,
        command_queue);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_2_1

CL_API_ENTRY cl_int CL_API_CALL clGetDeviceAndHostTimer(
    cl_device_id device,
    cl_ulong* device_timestamp,
    cl_ulong* host_timestamp)
{
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(device, CL_INVALID_DEVICE);
    return device->dispatch->clGetDeviceAndHostTimer(
        device,
        device_timestamp,
        host_timestamp);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_2_1

CL_API_ENTRY cl_int CL_API_CALL clGetHostTimer(
    cl_device_id device,
    cl_ulong* host_timestamp)
{
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(device, CL_INVALID_DEVICE);
    return device->dispatch->clGetHostTimer(
        device,
        host_timestamp);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_2_1

CL_API_ENTRY cl_program CL_API_CALL clCreateProgramWithIL(
    cl_context context,
    const void* il,
    size_t length,
    cl_int* errcode_ret)
{
    _SCL_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    return context->dispatch->clCreateProgramWithIL(
        context,
        il,
        length,
        errcode_ret);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_2_1

CL_API_ENTRY cl_kernel CL_API_CALL clCloneKernel(
    cl_kernel source_kernel,
    cl_int* errcode_ret)
{
    _SCL_VALIDATE_HANDLE_RETURN_HANDLE(source_kernel, CL_INVALID_KERNEL);
    return source_kernel->dispatch->clCloneKernel(
        source_kernel,
        errcode_ret);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_2_1

CL_API_ENTRY cl_int CL_API_CALL clGetKernelSubGroupInfo(
    cl_kernel kernel,
    cl_device_id device,
    cl_kernel_sub_group_info param_name,
    size_t input_value_size,
    const void* input_value,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(kernel, CL_INVALID_KERNEL);
    return kernel->dispatch->clGetKernelSubGroupInfo(
        kernel,
        device,
        param_name,
        input_value_size,
        input_value,
        param_value_size,
        param_value,
        param_value_size_ret);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_2_1

CL_API_ENTRY cl_int CL_API_CALL clEnqueueSVMMigrateMem(
    cl_command_queue command_queue,
    cl_uint num_svm_pointers,
    const void** svm_pointers,
    const size_t* sizes,
    cl_mem_migration_flags flags,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(command_queue, CL_INVALID_COMMAND_QUEUE);
    return command_queue->dispatch->clEnqueueSVMMigrateMem(
        command_queue,
        num_svm_pointers,
        svm_pointers,
        sizes,
        flags,
        num_events_in_wait_list,
        event_wait_list,
        event);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_2_2

CL_API_ENTRY cl_int CL_API_CALL clSetProgramReleaseCallback(
    cl_program program,
    void (CL_CALLBACK* pfn_notify)(cl_program program, void* user_data),
    void* user_data)
{
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(program, CL_INVALID_PROGRAM);
    return program->dispatch->clSetProgramReleaseCallback(
        program,
        pfn_notify,
        user_data);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_2_2

CL_API_ENTRY cl_int CL_API_CALL clSetProgramSpecializationConstant(
    cl_program program,
    cl_uint spec_id,
    size_t spec_size,
    const void* spec_value)
{
    _SCL_VALIDATE_HANDLE_RETURN_ERROR(program, CL_INVALID_PROGRAM);
    return program->dispatch->clSetProgramSpecializationConstant(
        program,
        spec_id,
        spec_size,
        spec_value);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_3_0

CL_API_ENTRY cl_mem CL_API_CALL clCreateBufferWithProperties(
    cl_context context,
    const cl_mem_properties* properties,
    cl_mem_flags flags,
    size_t size,
    void* host_ptr,
    cl_int* errcode_ret)
{
    _SCL_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    return context->dispatch->clCreateBufferWithProperties(
        context,
        properties,
        flags,
        size,
        host_ptr,
        errcode_ret);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CL_VERSION_3_0

CL_API_ENTRY cl_mem CL_API_CALL clCreateImageWithProperties(
    cl_context context,
    const cl_mem_properties* properties,
    cl_mem_flags flags,
    const cl_image_format* image_format,
    const cl_image_desc* image_desc,
    void* host_ptr,
    cl_int* errcode_ret)
{
    _SCL_VALIDATE_HANDLE_RETURN_HANDLE(context, CL_INVALID_CONTEXT);
    return context->dispatch->clCreateImageWithProperties(
        context,
        properties,
        flags,
        image_format,
        image_desc,
        host_ptr,
        errcode_ret);
}

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef __cplusplus
}
#endif
