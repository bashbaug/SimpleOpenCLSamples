/*
// Copyright (c) 2019 Ben Ashbaugh
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

static void _sclInit(void);

///////////////////////////////////////////////////////////////////////////////



#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clGetPlatformIDs)(
    cl_uint num_entries,
    cl_platform_id* platforms,
    cl_uint* num_platforms);

static _sclpfn_clGetPlatformIDs _sclGetPlatformIDs = nullptr;

CL_API_ENTRY cl_int CL_API_CALL clGetPlatformIDs(
    cl_uint num_entries,
    cl_platform_id* platforms,
    cl_uint* num_platforms)
{
    _sclInit();
    if (_sclGetPlatformIDs) {
        return _sclGetPlatformIDs(
            num_entries,
            platforms,
            num_platforms);
    }
    return CL_INVALID_OPERATION;
}
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clGetPlatformInfo)(
    cl_platform_id platform,
    cl_platform_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret);

static _sclpfn_clGetPlatformInfo _sclGetPlatformInfo = nullptr;

CL_API_ENTRY cl_int CL_API_CALL clGetPlatformInfo(
    cl_platform_id platform,
    cl_platform_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    if (_sclGetPlatformInfo) {
        return _sclGetPlatformInfo(
            platform,
            param_name,
            param_value_size,
            param_value,
            param_value_size_ret);
    }
    return CL_INVALID_OPERATION;
}
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clGetDeviceIDs)(
    cl_platform_id platform,
    cl_device_type device_type,
    cl_uint num_entries,
    cl_device_id* devices,
    cl_uint* num_devices);

static _sclpfn_clGetDeviceIDs _sclGetDeviceIDs = nullptr;

CL_API_ENTRY cl_int CL_API_CALL clGetDeviceIDs(
    cl_platform_id platform,
    cl_device_type device_type,
    cl_uint num_entries,
    cl_device_id* devices,
    cl_uint* num_devices)
{
    if (_sclGetDeviceIDs) {
        return _sclGetDeviceIDs(
            platform,
            device_type,
            num_entries,
            devices,
            num_devices);
    }
    return CL_INVALID_OPERATION;
}
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clGetDeviceInfo)(
    cl_device_id device,
    cl_device_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret);

static _sclpfn_clGetDeviceInfo _sclGetDeviceInfo = nullptr;

CL_API_ENTRY cl_int CL_API_CALL clGetDeviceInfo(
    cl_device_id device,
    cl_device_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    if (_sclGetDeviceInfo) {
        return _sclGetDeviceInfo(
            device,
            param_name,
            param_value_size,
            param_value,
            param_value_size_ret);
    }
    return CL_INVALID_OPERATION;
}
#endif

#ifdef CL_VERSION_1_0
typedef cl_context (CL_API_CALL *_sclpfn_clCreateContext)(
    const cl_context_properties* properties,
    cl_uint num_devices,
    const cl_device_id* devices,
    void (CL_CALLBACK* pfn_notify)(const char*, const void*, size_t, void*),
    void* user_data,
    cl_int* errcode_ret);

static _sclpfn_clCreateContext _sclCreateContext = nullptr;

CL_API_ENTRY cl_context CL_API_CALL clCreateContext(
    const cl_context_properties* properties,
    cl_uint num_devices,
    const cl_device_id* devices,
    void (CL_CALLBACK* pfn_notify)(const char*, const void*, size_t, void*),
    void* user_data,
    cl_int* errcode_ret)
{
    if (_sclCreateContext) {
        return _sclCreateContext(
            properties,
            num_devices,
            devices,
            pfn_notify,
            user_data,
            errcode_ret);
    }
    if (errcode_ret) *errcode_ret = CL_INVALID_OPERATION;
    return NULL;
}
#endif

#ifdef CL_VERSION_1_0
typedef cl_context (CL_API_CALL *_sclpfn_clCreateContextFromType)(
    const cl_context_properties* properties,
    cl_device_type device_type,
    void (CL_CALLBACK* pfn_notify)(const char*, const void*, size_t, void*),
    void* user_data,
    cl_int* errcode_ret);

static _sclpfn_clCreateContextFromType _sclCreateContextFromType = nullptr;

CL_API_ENTRY cl_context CL_API_CALL clCreateContextFromType(
    const cl_context_properties* properties,
    cl_device_type device_type,
    void (CL_CALLBACK* pfn_notify)(const char*, const void*, size_t, void*),
    void* user_data,
    cl_int* errcode_ret)
{
    if (_sclCreateContextFromType) {
        return _sclCreateContextFromType(
            properties,
            device_type,
            pfn_notify,
            user_data,
            errcode_ret);
    }
    if (errcode_ret) *errcode_ret = CL_INVALID_OPERATION;
    return NULL;
}
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clRetainContext)(
    cl_context context);

static _sclpfn_clRetainContext _sclRetainContext = nullptr;

CL_API_ENTRY cl_int CL_API_CALL clRetainContext(
    cl_context context)
{
    if (_sclRetainContext) {
        return _sclRetainContext(
            context);
    }
    return CL_INVALID_OPERATION;
}
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clReleaseContext)(
    cl_context context);

static _sclpfn_clReleaseContext _sclReleaseContext = nullptr;

CL_API_ENTRY cl_int CL_API_CALL clReleaseContext(
    cl_context context)
{
    if (_sclReleaseContext) {
        return _sclReleaseContext(
            context);
    }
    return CL_INVALID_OPERATION;
}
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clGetContextInfo)(
    cl_context context,
    cl_context_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret);

static _sclpfn_clGetContextInfo _sclGetContextInfo = nullptr;

CL_API_ENTRY cl_int CL_API_CALL clGetContextInfo(
    cl_context context,
    cl_context_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    if (_sclGetContextInfo) {
        return _sclGetContextInfo(
            context,
            param_name,
            param_value_size,
            param_value,
            param_value_size_ret);
    }
    return CL_INVALID_OPERATION;
}
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clRetainCommandQueue)(
    cl_command_queue command_queue);

static _sclpfn_clRetainCommandQueue _sclRetainCommandQueue = nullptr;

CL_API_ENTRY cl_int CL_API_CALL clRetainCommandQueue(
    cl_command_queue command_queue)
{
    if (_sclRetainCommandQueue) {
        return _sclRetainCommandQueue(
            command_queue);
    }
    return CL_INVALID_OPERATION;
}
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clReleaseCommandQueue)(
    cl_command_queue command_queue);

static _sclpfn_clReleaseCommandQueue _sclReleaseCommandQueue = nullptr;

CL_API_ENTRY cl_int CL_API_CALL clReleaseCommandQueue(
    cl_command_queue command_queue)
{
    if (_sclReleaseCommandQueue) {
        return _sclReleaseCommandQueue(
            command_queue);
    }
    return CL_INVALID_OPERATION;
}
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clGetCommandQueueInfo)(
    cl_command_queue command_queue,
    cl_command_queue_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret);

static _sclpfn_clGetCommandQueueInfo _sclGetCommandQueueInfo = nullptr;

CL_API_ENTRY cl_int CL_API_CALL clGetCommandQueueInfo(
    cl_command_queue command_queue,
    cl_command_queue_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    if (_sclGetCommandQueueInfo) {
        return _sclGetCommandQueueInfo(
            command_queue,
            param_name,
            param_value_size,
            param_value,
            param_value_size_ret);
    }
    return CL_INVALID_OPERATION;
}
#endif

#ifdef CL_VERSION_1_0
typedef cl_mem (CL_API_CALL *_sclpfn_clCreateBuffer)(
    cl_context context,
    cl_mem_flags flags,
    size_t size,
    void* host_ptr,
    cl_int* errcode_ret);

static _sclpfn_clCreateBuffer _sclCreateBuffer = nullptr;

CL_API_ENTRY cl_mem CL_API_CALL clCreateBuffer(
    cl_context context,
    cl_mem_flags flags,
    size_t size,
    void* host_ptr,
    cl_int* errcode_ret)
{
    if (_sclCreateBuffer) {
        return _sclCreateBuffer(
            context,
            flags,
            size,
            host_ptr,
            errcode_ret);
    }
    if (errcode_ret) *errcode_ret = CL_INVALID_OPERATION;
    return NULL;
}
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clRetainMemObject)(
    cl_mem memobj);

static _sclpfn_clRetainMemObject _sclRetainMemObject = nullptr;

CL_API_ENTRY cl_int CL_API_CALL clRetainMemObject(
    cl_mem memobj)
{
    if (_sclRetainMemObject) {
        return _sclRetainMemObject(
            memobj);
    }
    return CL_INVALID_OPERATION;
}
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clReleaseMemObject)(
    cl_mem memobj);

static _sclpfn_clReleaseMemObject _sclReleaseMemObject = nullptr;

CL_API_ENTRY cl_int CL_API_CALL clReleaseMemObject(
    cl_mem memobj)
{
    if (_sclReleaseMemObject) {
        return _sclReleaseMemObject(
            memobj);
    }
    return CL_INVALID_OPERATION;
}
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clGetSupportedImageFormats)(
    cl_context context,
    cl_mem_flags flags,
    cl_mem_object_type image_type,
    cl_uint num_entries,
    cl_image_format* image_formats,
    cl_uint* num_image_formats);

static _sclpfn_clGetSupportedImageFormats _sclGetSupportedImageFormats = nullptr;

CL_API_ENTRY cl_int CL_API_CALL clGetSupportedImageFormats(
    cl_context context,
    cl_mem_flags flags,
    cl_mem_object_type image_type,
    cl_uint num_entries,
    cl_image_format* image_formats,
    cl_uint* num_image_formats)
{
    if (_sclGetSupportedImageFormats) {
        return _sclGetSupportedImageFormats(
            context,
            flags,
            image_type,
            num_entries,
            image_formats,
            num_image_formats);
    }
    return CL_INVALID_OPERATION;
}
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clGetMemObjectInfo)(
    cl_mem memobj,
    cl_mem_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret);

static _sclpfn_clGetMemObjectInfo _sclGetMemObjectInfo = nullptr;

CL_API_ENTRY cl_int CL_API_CALL clGetMemObjectInfo(
    cl_mem memobj,
    cl_mem_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    if (_sclGetMemObjectInfo) {
        return _sclGetMemObjectInfo(
            memobj,
            param_name,
            param_value_size,
            param_value,
            param_value_size_ret);
    }
    return CL_INVALID_OPERATION;
}
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clGetImageInfo)(
    cl_mem image,
    cl_image_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret);

static _sclpfn_clGetImageInfo _sclGetImageInfo = nullptr;

CL_API_ENTRY cl_int CL_API_CALL clGetImageInfo(
    cl_mem image,
    cl_image_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    if (_sclGetImageInfo) {
        return _sclGetImageInfo(
            image,
            param_name,
            param_value_size,
            param_value,
            param_value_size_ret);
    }
    return CL_INVALID_OPERATION;
}
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clRetainSampler)(
    cl_sampler sampler);

static _sclpfn_clRetainSampler _sclRetainSampler = nullptr;

CL_API_ENTRY cl_int CL_API_CALL clRetainSampler(
    cl_sampler sampler)
{
    if (_sclRetainSampler) {
        return _sclRetainSampler(
            sampler);
    }
    return CL_INVALID_OPERATION;
}
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clReleaseSampler)(
    cl_sampler sampler);

static _sclpfn_clReleaseSampler _sclReleaseSampler = nullptr;

CL_API_ENTRY cl_int CL_API_CALL clReleaseSampler(
    cl_sampler sampler)
{
    if (_sclReleaseSampler) {
        return _sclReleaseSampler(
            sampler);
    }
    return CL_INVALID_OPERATION;
}
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clGetSamplerInfo)(
    cl_sampler sampler,
    cl_sampler_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret);

static _sclpfn_clGetSamplerInfo _sclGetSamplerInfo = nullptr;

CL_API_ENTRY cl_int CL_API_CALL clGetSamplerInfo(
    cl_sampler sampler,
    cl_sampler_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    if (_sclGetSamplerInfo) {
        return _sclGetSamplerInfo(
            sampler,
            param_name,
            param_value_size,
            param_value,
            param_value_size_ret);
    }
    return CL_INVALID_OPERATION;
}
#endif

#ifdef CL_VERSION_1_0
typedef cl_program (CL_API_CALL *_sclpfn_clCreateProgramWithSource)(
    cl_context context,
    cl_uint count,
    const char** strings,
    const size_t* lengths,
    cl_int* errcode_ret);

static _sclpfn_clCreateProgramWithSource _sclCreateProgramWithSource = nullptr;

CL_API_ENTRY cl_program CL_API_CALL clCreateProgramWithSource(
    cl_context context,
    cl_uint count,
    const char** strings,
    const size_t* lengths,
    cl_int* errcode_ret)
{
    if (_sclCreateProgramWithSource) {
        return _sclCreateProgramWithSource(
            context,
            count,
            strings,
            lengths,
            errcode_ret);
    }
    if (errcode_ret) *errcode_ret = CL_INVALID_OPERATION;
    return NULL;
}
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

static _sclpfn_clCreateProgramWithBinary _sclCreateProgramWithBinary = nullptr;

CL_API_ENTRY cl_program CL_API_CALL clCreateProgramWithBinary(
    cl_context context,
    cl_uint num_devices,
    const cl_device_id* device_list,
    const size_t* lengths,
    const unsigned char** binaries,
    cl_int* binary_status,
    cl_int* errcode_ret)
{
    if (_sclCreateProgramWithBinary) {
        return _sclCreateProgramWithBinary(
            context,
            num_devices,
            device_list,
            lengths,
            binaries,
            binary_status,
            errcode_ret);
    }
    if (errcode_ret) *errcode_ret = CL_INVALID_OPERATION;
    return NULL;
}
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clRetainProgram)(
    cl_program program);

static _sclpfn_clRetainProgram _sclRetainProgram = nullptr;

CL_API_ENTRY cl_int CL_API_CALL clRetainProgram(
    cl_program program)
{
    if (_sclRetainProgram) {
        return _sclRetainProgram(
            program);
    }
    return CL_INVALID_OPERATION;
}
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clReleaseProgram)(
    cl_program program);

static _sclpfn_clReleaseProgram _sclReleaseProgram = nullptr;

CL_API_ENTRY cl_int CL_API_CALL clReleaseProgram(
    cl_program program)
{
    if (_sclReleaseProgram) {
        return _sclReleaseProgram(
            program);
    }
    return CL_INVALID_OPERATION;
}
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clBuildProgram)(
    cl_program program,
    cl_uint num_devices,
    const cl_device_id* device_list,
    const char* options,
    void (CL_CALLBACK* pfn_notify)(cl_program program, void* user_data),
    void* user_data);

static _sclpfn_clBuildProgram _sclBuildProgram = nullptr;

CL_API_ENTRY cl_int CL_API_CALL clBuildProgram(
    cl_program program,
    cl_uint num_devices,
    const cl_device_id* device_list,
    const char* options,
    void (CL_CALLBACK* pfn_notify)(cl_program program, void* user_data),
    void* user_data)
{
    if (_sclBuildProgram) {
        return _sclBuildProgram(
            program,
            num_devices,
            device_list,
            options,
            pfn_notify,
            user_data);
    }
    return CL_INVALID_OPERATION;
}
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clGetProgramInfo)(
    cl_program program,
    cl_program_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret);

static _sclpfn_clGetProgramInfo _sclGetProgramInfo = nullptr;

CL_API_ENTRY cl_int CL_API_CALL clGetProgramInfo(
    cl_program program,
    cl_program_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    if (_sclGetProgramInfo) {
        return _sclGetProgramInfo(
            program,
            param_name,
            param_value_size,
            param_value,
            param_value_size_ret);
    }
    return CL_INVALID_OPERATION;
}
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clGetProgramBuildInfo)(
    cl_program program,
    cl_device_id device,
    cl_program_build_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret);

static _sclpfn_clGetProgramBuildInfo _sclGetProgramBuildInfo = nullptr;

CL_API_ENTRY cl_int CL_API_CALL clGetProgramBuildInfo(
    cl_program program,
    cl_device_id device,
    cl_program_build_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    if (_sclGetProgramBuildInfo) {
        return _sclGetProgramBuildInfo(
            program,
            device,
            param_name,
            param_value_size,
            param_value,
            param_value_size_ret);
    }
    return CL_INVALID_OPERATION;
}
#endif

#ifdef CL_VERSION_1_0
typedef cl_kernel (CL_API_CALL *_sclpfn_clCreateKernel)(
    cl_program program,
    const char* kernel_name,
    cl_int* errcode_ret);

static _sclpfn_clCreateKernel _sclCreateKernel = nullptr;

CL_API_ENTRY cl_kernel CL_API_CALL clCreateKernel(
    cl_program program,
    const char* kernel_name,
    cl_int* errcode_ret)
{
    if (_sclCreateKernel) {
        return _sclCreateKernel(
            program,
            kernel_name,
            errcode_ret);
    }
    if (errcode_ret) *errcode_ret = CL_INVALID_OPERATION;
    return NULL;
}
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clCreateKernelsInProgram)(
    cl_program program,
    cl_uint num_kernels,
    cl_kernel* kernels,
    cl_uint* num_kernels_ret);

static _sclpfn_clCreateKernelsInProgram _sclCreateKernelsInProgram = nullptr;

CL_API_ENTRY cl_int CL_API_CALL clCreateKernelsInProgram(
    cl_program program,
    cl_uint num_kernels,
    cl_kernel* kernels,
    cl_uint* num_kernels_ret)
{
    if (_sclCreateKernelsInProgram) {
        return _sclCreateKernelsInProgram(
            program,
            num_kernels,
            kernels,
            num_kernels_ret);
    }
    return CL_INVALID_OPERATION;
}
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clRetainKernel)(
    cl_kernel kernel);

static _sclpfn_clRetainKernel _sclRetainKernel = nullptr;

CL_API_ENTRY cl_int CL_API_CALL clRetainKernel(
    cl_kernel kernel)
{
    if (_sclRetainKernel) {
        return _sclRetainKernel(
            kernel);
    }
    return CL_INVALID_OPERATION;
}
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clReleaseKernel)(
    cl_kernel kernel);

static _sclpfn_clReleaseKernel _sclReleaseKernel = nullptr;

CL_API_ENTRY cl_int CL_API_CALL clReleaseKernel(
    cl_kernel kernel)
{
    if (_sclReleaseKernel) {
        return _sclReleaseKernel(
            kernel);
    }
    return CL_INVALID_OPERATION;
}
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clSetKernelArg)(
    cl_kernel kernel,
    cl_uint arg_index,
    size_t arg_size,
    const void* arg_value);

static _sclpfn_clSetKernelArg _sclSetKernelArg = nullptr;

CL_API_ENTRY cl_int CL_API_CALL clSetKernelArg(
    cl_kernel kernel,
    cl_uint arg_index,
    size_t arg_size,
    const void* arg_value)
{
    if (_sclSetKernelArg) {
        return _sclSetKernelArg(
            kernel,
            arg_index,
            arg_size,
            arg_value);
    }
    return CL_INVALID_OPERATION;
}
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clGetKernelInfo)(
    cl_kernel kernel,
    cl_kernel_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret);

static _sclpfn_clGetKernelInfo _sclGetKernelInfo = nullptr;

CL_API_ENTRY cl_int CL_API_CALL clGetKernelInfo(
    cl_kernel kernel,
    cl_kernel_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    if (_sclGetKernelInfo) {
        return _sclGetKernelInfo(
            kernel,
            param_name,
            param_value_size,
            param_value,
            param_value_size_ret);
    }
    return CL_INVALID_OPERATION;
}
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clGetKernelWorkGroupInfo)(
    cl_kernel kernel,
    cl_device_id device,
    cl_kernel_work_group_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret);

static _sclpfn_clGetKernelWorkGroupInfo _sclGetKernelWorkGroupInfo = nullptr;

CL_API_ENTRY cl_int CL_API_CALL clGetKernelWorkGroupInfo(
    cl_kernel kernel,
    cl_device_id device,
    cl_kernel_work_group_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    if (_sclGetKernelWorkGroupInfo) {
        return _sclGetKernelWorkGroupInfo(
            kernel,
            device,
            param_name,
            param_value_size,
            param_value,
            param_value_size_ret);
    }
    return CL_INVALID_OPERATION;
}
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clWaitForEvents)(
    cl_uint num_events,
    const cl_event* event_list);

static _sclpfn_clWaitForEvents _sclWaitForEvents = nullptr;

CL_API_ENTRY cl_int CL_API_CALL clWaitForEvents(
    cl_uint num_events,
    const cl_event* event_list)
{
    if (_sclWaitForEvents) {
        return _sclWaitForEvents(
            num_events,
            event_list);
    }
    return CL_INVALID_OPERATION;
}
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clGetEventInfo)(
    cl_event event,
    cl_event_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret);

static _sclpfn_clGetEventInfo _sclGetEventInfo = nullptr;

CL_API_ENTRY cl_int CL_API_CALL clGetEventInfo(
    cl_event event,
    cl_event_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    if (_sclGetEventInfo) {
        return _sclGetEventInfo(
            event,
            param_name,
            param_value_size,
            param_value,
            param_value_size_ret);
    }
    return CL_INVALID_OPERATION;
}
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clRetainEvent)(
    cl_event event);

static _sclpfn_clRetainEvent _sclRetainEvent = nullptr;

CL_API_ENTRY cl_int CL_API_CALL clRetainEvent(
    cl_event event)
{
    if (_sclRetainEvent) {
        return _sclRetainEvent(
            event);
    }
    return CL_INVALID_OPERATION;
}
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clReleaseEvent)(
    cl_event event);

static _sclpfn_clReleaseEvent _sclReleaseEvent = nullptr;

CL_API_ENTRY cl_int CL_API_CALL clReleaseEvent(
    cl_event event)
{
    if (_sclReleaseEvent) {
        return _sclReleaseEvent(
            event);
    }
    return CL_INVALID_OPERATION;
}
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clGetEventProfilingInfo)(
    cl_event event,
    cl_profiling_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret);

static _sclpfn_clGetEventProfilingInfo _sclGetEventProfilingInfo = nullptr;

CL_API_ENTRY cl_int CL_API_CALL clGetEventProfilingInfo(
    cl_event event,
    cl_profiling_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    if (_sclGetEventProfilingInfo) {
        return _sclGetEventProfilingInfo(
            event,
            param_name,
            param_value_size,
            param_value,
            param_value_size_ret);
    }
    return CL_INVALID_OPERATION;
}
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clFlush)(
    cl_command_queue command_queue);

static _sclpfn_clFlush _sclFlush = nullptr;

CL_API_ENTRY cl_int CL_API_CALL clFlush(
    cl_command_queue command_queue)
{
    if (_sclFlush) {
        return _sclFlush(
            command_queue);
    }
    return CL_INVALID_OPERATION;
}
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clFinish)(
    cl_command_queue command_queue);

static _sclpfn_clFinish _sclFinish = nullptr;

CL_API_ENTRY cl_int CL_API_CALL clFinish(
    cl_command_queue command_queue)
{
    if (_sclFinish) {
        return _sclFinish(
            command_queue);
    }
    return CL_INVALID_OPERATION;
}
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

static _sclpfn_clEnqueueReadBuffer _sclEnqueueReadBuffer = nullptr;

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
    if (_sclEnqueueReadBuffer) {
        return _sclEnqueueReadBuffer(
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
    return CL_INVALID_OPERATION;
}
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

static _sclpfn_clEnqueueWriteBuffer _sclEnqueueWriteBuffer = nullptr;

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
    if (_sclEnqueueWriteBuffer) {
        return _sclEnqueueWriteBuffer(
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
    return CL_INVALID_OPERATION;
}
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

static _sclpfn_clEnqueueCopyBuffer _sclEnqueueCopyBuffer = nullptr;

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
    if (_sclEnqueueCopyBuffer) {
        return _sclEnqueueCopyBuffer(
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
    return CL_INVALID_OPERATION;
}
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

static _sclpfn_clEnqueueReadImage _sclEnqueueReadImage = nullptr;

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
    if (_sclEnqueueReadImage) {
        return _sclEnqueueReadImage(
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
    return CL_INVALID_OPERATION;
}
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

static _sclpfn_clEnqueueWriteImage _sclEnqueueWriteImage = nullptr;

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
    if (_sclEnqueueWriteImage) {
        return _sclEnqueueWriteImage(
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
    return CL_INVALID_OPERATION;
}
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

static _sclpfn_clEnqueueCopyImage _sclEnqueueCopyImage = nullptr;

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
    if (_sclEnqueueCopyImage) {
        return _sclEnqueueCopyImage(
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
    return CL_INVALID_OPERATION;
}
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

static _sclpfn_clEnqueueCopyImageToBuffer _sclEnqueueCopyImageToBuffer = nullptr;

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
    if (_sclEnqueueCopyImageToBuffer) {
        return _sclEnqueueCopyImageToBuffer(
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
    return CL_INVALID_OPERATION;
}
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

static _sclpfn_clEnqueueCopyBufferToImage _sclEnqueueCopyBufferToImage = nullptr;

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
    if (_sclEnqueueCopyBufferToImage) {
        return _sclEnqueueCopyBufferToImage(
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
    return CL_INVALID_OPERATION;
}
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

static _sclpfn_clEnqueueMapBuffer _sclEnqueueMapBuffer = nullptr;

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
    if (_sclEnqueueMapBuffer) {
        return _sclEnqueueMapBuffer(
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
    return nullptr;
}
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

static _sclpfn_clEnqueueMapImage _sclEnqueueMapImage = nullptr;

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
    if (_sclEnqueueMapImage) {
        return _sclEnqueueMapImage(
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
    return nullptr;
}
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clEnqueueUnmapMemObject)(
    cl_command_queue command_queue,
    cl_mem memobj,
    void* mapped_ptr,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event);

static _sclpfn_clEnqueueUnmapMemObject _sclEnqueueUnmapMemObject = nullptr;

CL_API_ENTRY cl_int CL_API_CALL clEnqueueUnmapMemObject(
    cl_command_queue command_queue,
    cl_mem memobj,
    void* mapped_ptr,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    if (_sclEnqueueUnmapMemObject) {
        return _sclEnqueueUnmapMemObject(
            command_queue,
            memobj,
            mapped_ptr,
            num_events_in_wait_list,
            event_wait_list,
            event);
    }
    return CL_INVALID_OPERATION;
}
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

static _sclpfn_clEnqueueNDRangeKernel _sclEnqueueNDRangeKernel = nullptr;

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
    if (_sclEnqueueNDRangeKernel) {
        return _sclEnqueueNDRangeKernel(
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
    return CL_INVALID_OPERATION;
}
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

static _sclpfn_clEnqueueNativeKernel _sclEnqueueNativeKernel = nullptr;

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
    if (_sclEnqueueNativeKernel) {
        return _sclEnqueueNativeKernel(
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
    return CL_INVALID_OPERATION;
}
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clSetCommandQueueProperty)(
    cl_command_queue command_queue,
    cl_command_queue_properties properties,
    cl_bool enable,
    cl_command_queue_properties* old_properties);

static _sclpfn_clSetCommandQueueProperty _sclSetCommandQueueProperty = nullptr;

CL_API_ENTRY cl_int CL_API_CALL clSetCommandQueueProperty(
    cl_command_queue command_queue,
    cl_command_queue_properties properties,
    cl_bool enable,
    cl_command_queue_properties* old_properties)
{
    if (_sclSetCommandQueueProperty) {
        return _sclSetCommandQueueProperty(
            command_queue,
            properties,
            enable,
            old_properties);
    }
    return CL_INVALID_OPERATION;
}
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

static _sclpfn_clCreateImage2D _sclCreateImage2D = nullptr;

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
    if (_sclCreateImage2D) {
        return _sclCreateImage2D(
            context,
            flags,
            image_format,
            image_width,
            image_height,
            image_row_pitch,
            host_ptr,
            errcode_ret);
    }
    if (errcode_ret) *errcode_ret = CL_INVALID_OPERATION;
    return NULL;
}
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

static _sclpfn_clCreateImage3D _sclCreateImage3D = nullptr;

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
    if (_sclCreateImage3D) {
        return _sclCreateImage3D(
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
    if (errcode_ret) *errcode_ret = CL_INVALID_OPERATION;
    return NULL;
}
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clEnqueueMarker)(
    cl_command_queue command_queue,
    cl_event* event);

static _sclpfn_clEnqueueMarker _sclEnqueueMarker = nullptr;

CL_API_ENTRY cl_int CL_API_CALL clEnqueueMarker(
    cl_command_queue command_queue,
    cl_event* event)
{
    if (_sclEnqueueMarker) {
        return _sclEnqueueMarker(
            command_queue,
            event);
    }
    return CL_INVALID_OPERATION;
}
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clEnqueueWaitForEvents)(
    cl_command_queue command_queue,
    cl_uint num_events,
    const cl_event* event_list);

static _sclpfn_clEnqueueWaitForEvents _sclEnqueueWaitForEvents = nullptr;

CL_API_ENTRY cl_int CL_API_CALL clEnqueueWaitForEvents(
    cl_command_queue command_queue,
    cl_uint num_events,
    const cl_event* event_list)
{
    if (_sclEnqueueWaitForEvents) {
        return _sclEnqueueWaitForEvents(
            command_queue,
            num_events,
            event_list);
    }
    return CL_INVALID_OPERATION;
}
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clEnqueueBarrier)(
    cl_command_queue command_queue);

static _sclpfn_clEnqueueBarrier _sclEnqueueBarrier = nullptr;

CL_API_ENTRY cl_int CL_API_CALL clEnqueueBarrier(
    cl_command_queue command_queue)
{
    if (_sclEnqueueBarrier) {
        return _sclEnqueueBarrier(
            command_queue);
    }
    return CL_INVALID_OPERATION;
}
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clUnloadCompiler)(
    void );

static _sclpfn_clUnloadCompiler _sclUnloadCompiler = nullptr;

CL_API_ENTRY cl_int CL_API_CALL clUnloadCompiler(
    void )
{
    if (_sclUnloadCompiler) {
        return _sclUnloadCompiler(
            );
    }
    return CL_INVALID_OPERATION;
}
#endif

#ifdef CL_VERSION_1_0
typedef void* (CL_API_CALL *_sclpfn_clGetExtensionFunctionAddress)(
    const char* func_name);

static _sclpfn_clGetExtensionFunctionAddress _sclGetExtensionFunctionAddress = nullptr;

CL_API_ENTRY void* CL_API_CALL clGetExtensionFunctionAddress(
    const char* func_name)
{
    _sclInit();
    if (_sclGetExtensionFunctionAddress) {
        return _sclGetExtensionFunctionAddress(
            func_name);
    }
    return nullptr;
}
#endif

#ifdef CL_VERSION_1_0
typedef cl_command_queue (CL_API_CALL *_sclpfn_clCreateCommandQueue)(
    cl_context context,
    cl_device_id device,
    cl_command_queue_properties properties,
    cl_int* errcode_ret);

static _sclpfn_clCreateCommandQueue _sclCreateCommandQueue = nullptr;

CL_API_ENTRY cl_command_queue CL_API_CALL clCreateCommandQueue(
    cl_context context,
    cl_device_id device,
    cl_command_queue_properties properties,
    cl_int* errcode_ret)
{
    if (_sclCreateCommandQueue) {
        return _sclCreateCommandQueue(
            context,
            device,
            properties,
            errcode_ret);
    }
    if (errcode_ret) *errcode_ret = CL_INVALID_OPERATION;
    return NULL;
}
#endif

#ifdef CL_VERSION_1_0
typedef cl_sampler (CL_API_CALL *_sclpfn_clCreateSampler)(
    cl_context context,
    cl_bool normalized_coords,
    cl_addressing_mode addressing_mode,
    cl_filter_mode filter_mode,
    cl_int* errcode_ret);

static _sclpfn_clCreateSampler _sclCreateSampler = nullptr;

CL_API_ENTRY cl_sampler CL_API_CALL clCreateSampler(
    cl_context context,
    cl_bool normalized_coords,
    cl_addressing_mode addressing_mode,
    cl_filter_mode filter_mode,
    cl_int* errcode_ret)
{
    if (_sclCreateSampler) {
        return _sclCreateSampler(
            context,
            normalized_coords,
            addressing_mode,
            filter_mode,
            errcode_ret);
    }
    if (errcode_ret) *errcode_ret = CL_INVALID_OPERATION;
    return NULL;
}
#endif

#ifdef CL_VERSION_1_0
typedef cl_int (CL_API_CALL *_sclpfn_clEnqueueTask)(
    cl_command_queue command_queue,
    cl_kernel kernel,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event);

static _sclpfn_clEnqueueTask _sclEnqueueTask = nullptr;

CL_API_ENTRY cl_int CL_API_CALL clEnqueueTask(
    cl_command_queue command_queue,
    cl_kernel kernel,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    if (_sclEnqueueTask) {
        return _sclEnqueueTask(
            command_queue,
            kernel,
            num_events_in_wait_list,
            event_wait_list,
            event);
    }
    return CL_INVALID_OPERATION;
}
#endif

#ifdef CL_VERSION_1_1
typedef cl_mem (CL_API_CALL *_sclpfn_clCreateSubBuffer)(
    cl_mem buffer,
    cl_mem_flags flags,
    cl_buffer_create_type buffer_create_type,
    const void* buffer_create_info,
    cl_int* errcode_ret);

static _sclpfn_clCreateSubBuffer _sclCreateSubBuffer = nullptr;

CL_API_ENTRY cl_mem CL_API_CALL clCreateSubBuffer(
    cl_mem buffer,
    cl_mem_flags flags,
    cl_buffer_create_type buffer_create_type,
    const void* buffer_create_info,
    cl_int* errcode_ret)
{
    if (_sclCreateSubBuffer) {
        return _sclCreateSubBuffer(
            buffer,
            flags,
            buffer_create_type,
            buffer_create_info,
            errcode_ret);
    }
    if (errcode_ret) *errcode_ret = CL_INVALID_OPERATION;
    return NULL;
}
#endif

#ifdef CL_VERSION_1_1
typedef cl_int (CL_API_CALL *_sclpfn_clSetMemObjectDestructorCallback)(
    cl_mem memobj,
    void (CL_CALLBACK* pfn_notify)(cl_mem memobj, void* user_data),
    void* user_data);

static _sclpfn_clSetMemObjectDestructorCallback _sclSetMemObjectDestructorCallback = nullptr;

CL_API_ENTRY cl_int CL_API_CALL clSetMemObjectDestructorCallback(
    cl_mem memobj,
    void (CL_CALLBACK* pfn_notify)(cl_mem memobj, void* user_data),
    void* user_data)
{
    if (_sclSetMemObjectDestructorCallback) {
        return _sclSetMemObjectDestructorCallback(
            memobj,
            pfn_notify,
            user_data);
    }
    return CL_INVALID_OPERATION;
}
#endif

#ifdef CL_VERSION_1_1
typedef cl_event (CL_API_CALL *_sclpfn_clCreateUserEvent)(
    cl_context context,
    cl_int* errcode_ret);

static _sclpfn_clCreateUserEvent _sclCreateUserEvent = nullptr;

CL_API_ENTRY cl_event CL_API_CALL clCreateUserEvent(
    cl_context context,
    cl_int* errcode_ret)
{
    if (_sclCreateUserEvent) {
        return _sclCreateUserEvent(
            context,
            errcode_ret);
    }
    if (errcode_ret) *errcode_ret = CL_INVALID_OPERATION;
    return NULL;
}
#endif

#ifdef CL_VERSION_1_1
typedef cl_int (CL_API_CALL *_sclpfn_clSetUserEventStatus)(
    cl_event event,
    cl_int execution_status);

static _sclpfn_clSetUserEventStatus _sclSetUserEventStatus = nullptr;

CL_API_ENTRY cl_int CL_API_CALL clSetUserEventStatus(
    cl_event event,
    cl_int execution_status)
{
    if (_sclSetUserEventStatus) {
        return _sclSetUserEventStatus(
            event,
            execution_status);
    }
    return CL_INVALID_OPERATION;
}
#endif

#ifdef CL_VERSION_1_1
typedef cl_int (CL_API_CALL *_sclpfn_clSetEventCallback)(
    cl_event event,
    cl_int command_exec_callback_type,
    void (CL_CALLBACK* pfn_notify)(cl_event event, cl_int type, void *user_data),
    void* user_data);

static _sclpfn_clSetEventCallback _sclSetEventCallback = nullptr;

CL_API_ENTRY cl_int CL_API_CALL clSetEventCallback(
    cl_event event,
    cl_int command_exec_callback_type,
    void (CL_CALLBACK* pfn_notify)(cl_event event, cl_int type, void *user_data),
    void* user_data)
{
    if (_sclSetEventCallback) {
        return _sclSetEventCallback(
            event,
            command_exec_callback_type,
            pfn_notify,
            user_data);
    }
    return CL_INVALID_OPERATION;
}
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

static _sclpfn_clEnqueueReadBufferRect _sclEnqueueReadBufferRect = nullptr;

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
    if (_sclEnqueueReadBufferRect) {
        return _sclEnqueueReadBufferRect(
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
    return CL_INVALID_OPERATION;
}
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

static _sclpfn_clEnqueueWriteBufferRect _sclEnqueueWriteBufferRect = nullptr;

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
    if (_sclEnqueueWriteBufferRect) {
        return _sclEnqueueWriteBufferRect(
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
    return CL_INVALID_OPERATION;
}
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

static _sclpfn_clEnqueueCopyBufferRect _sclEnqueueCopyBufferRect = nullptr;

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
    if (_sclEnqueueCopyBufferRect) {
        return _sclEnqueueCopyBufferRect(
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
    return CL_INVALID_OPERATION;
}
#endif

#ifdef CL_VERSION_1_2
typedef cl_int (CL_API_CALL *_sclpfn_clCreateSubDevices)(
    cl_device_id in_device,
    const cl_device_partition_property* properties,
    cl_uint num_devices,
    cl_device_id* out_devices,
    cl_uint* num_devices_ret);

static _sclpfn_clCreateSubDevices _sclCreateSubDevices = nullptr;

CL_API_ENTRY cl_int CL_API_CALL clCreateSubDevices(
    cl_device_id in_device,
    const cl_device_partition_property* properties,
    cl_uint num_devices,
    cl_device_id* out_devices,
    cl_uint* num_devices_ret)
{
    if (_sclCreateSubDevices) {
        return _sclCreateSubDevices(
            in_device,
            properties,
            num_devices,
            out_devices,
            num_devices_ret);
    }
    return CL_INVALID_OPERATION;
}
#endif

#ifdef CL_VERSION_1_2
typedef cl_int (CL_API_CALL *_sclpfn_clRetainDevice)(
    cl_device_id device);

static _sclpfn_clRetainDevice _sclRetainDevice = nullptr;

CL_API_ENTRY cl_int CL_API_CALL clRetainDevice(
    cl_device_id device)
{
    if (_sclRetainDevice) {
        return _sclRetainDevice(
            device);
    }
    return CL_INVALID_OPERATION;
}
#endif

#ifdef CL_VERSION_1_2
typedef cl_int (CL_API_CALL *_sclpfn_clReleaseDevice)(
    cl_device_id device);

static _sclpfn_clReleaseDevice _sclReleaseDevice = nullptr;

CL_API_ENTRY cl_int CL_API_CALL clReleaseDevice(
    cl_device_id device)
{
    if (_sclReleaseDevice) {
        return _sclReleaseDevice(
            device);
    }
    return CL_INVALID_OPERATION;
}
#endif

#ifdef CL_VERSION_1_2
typedef cl_mem (CL_API_CALL *_sclpfn_clCreateImage)(
    cl_context context,
    cl_mem_flags flags,
    const cl_image_format* image_format,
    const cl_image_desc* image_desc,
    void* host_ptr,
    cl_int* errcode_ret);

static _sclpfn_clCreateImage _sclCreateImage = nullptr;

CL_API_ENTRY cl_mem CL_API_CALL clCreateImage(
    cl_context context,
    cl_mem_flags flags,
    const cl_image_format* image_format,
    const cl_image_desc* image_desc,
    void* host_ptr,
    cl_int* errcode_ret)
{
    if (_sclCreateImage) {
        return _sclCreateImage(
            context,
            flags,
            image_format,
            image_desc,
            host_ptr,
            errcode_ret);
    }
    if (errcode_ret) *errcode_ret = CL_INVALID_OPERATION;
    return NULL;
}
#endif

#ifdef CL_VERSION_1_2
typedef cl_program (CL_API_CALL *_sclpfn_clCreateProgramWithBuiltInKernels)(
    cl_context context,
    cl_uint num_devices,
    const cl_device_id* device_list,
    const char* kernel_names,
    cl_int* errcode_ret);

static _sclpfn_clCreateProgramWithBuiltInKernels _sclCreateProgramWithBuiltInKernels = nullptr;

CL_API_ENTRY cl_program CL_API_CALL clCreateProgramWithBuiltInKernels(
    cl_context context,
    cl_uint num_devices,
    const cl_device_id* device_list,
    const char* kernel_names,
    cl_int* errcode_ret)
{
    if (_sclCreateProgramWithBuiltInKernels) {
        return _sclCreateProgramWithBuiltInKernels(
            context,
            num_devices,
            device_list,
            kernel_names,
            errcode_ret);
    }
    if (errcode_ret) *errcode_ret = CL_INVALID_OPERATION;
    return NULL;
}
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

static _sclpfn_clCompileProgram _sclCompileProgram = nullptr;

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
    if (_sclCompileProgram) {
        return _sclCompileProgram(
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
    return CL_INVALID_OPERATION;
}
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

static _sclpfn_clLinkProgram _sclLinkProgram = nullptr;

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
    if (_sclLinkProgram) {
        return _sclLinkProgram(
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
    if (errcode_ret) *errcode_ret = CL_INVALID_OPERATION;
    return NULL;
}
#endif

#ifdef CL_VERSION_1_2
typedef cl_int (CL_API_CALL *_sclpfn_clUnloadPlatformCompiler)(
    cl_platform_id platform);

static _sclpfn_clUnloadPlatformCompiler _sclUnloadPlatformCompiler = nullptr;

CL_API_ENTRY cl_int CL_API_CALL clUnloadPlatformCompiler(
    cl_platform_id platform)
{
    if (_sclUnloadPlatformCompiler) {
        return _sclUnloadPlatformCompiler(
            platform);
    }
    return CL_INVALID_OPERATION;
}
#endif

#ifdef CL_VERSION_1_2
typedef cl_int (CL_API_CALL *_sclpfn_clGetKernelArgInfo)(
    cl_kernel kernel,
    cl_uint arg_indx,
    cl_kernel_arg_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret);

static _sclpfn_clGetKernelArgInfo _sclGetKernelArgInfo = nullptr;

CL_API_ENTRY cl_int CL_API_CALL clGetKernelArgInfo(
    cl_kernel kernel,
    cl_uint arg_indx,
    cl_kernel_arg_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    if (_sclGetKernelArgInfo) {
        return _sclGetKernelArgInfo(
            kernel,
            arg_indx,
            param_name,
            param_value_size,
            param_value,
            param_value_size_ret);
    }
    return CL_INVALID_OPERATION;
}
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

static _sclpfn_clEnqueueFillBuffer _sclEnqueueFillBuffer = nullptr;

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
    if (_sclEnqueueFillBuffer) {
        return _sclEnqueueFillBuffer(
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
    return CL_INVALID_OPERATION;
}
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

static _sclpfn_clEnqueueFillImage _sclEnqueueFillImage = nullptr;

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
    if (_sclEnqueueFillImage) {
        return _sclEnqueueFillImage(
            command_queue,
            image,
            fill_color,
            origin,
            region,
            num_events_in_wait_list,
            event_wait_list,
            event);
    }
    return CL_INVALID_OPERATION;
}
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

static _sclpfn_clEnqueueMigrateMemObjects _sclEnqueueMigrateMemObjects = nullptr;

CL_API_ENTRY cl_int CL_API_CALL clEnqueueMigrateMemObjects(
    cl_command_queue command_queue,
    cl_uint num_mem_objects,
    const cl_mem* mem_objects,
    cl_mem_migration_flags flags,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    if (_sclEnqueueMigrateMemObjects) {
        return _sclEnqueueMigrateMemObjects(
            command_queue,
            num_mem_objects,
            mem_objects,
            flags,
            num_events_in_wait_list,
            event_wait_list,
            event);
    }
    return CL_INVALID_OPERATION;
}
#endif

#ifdef CL_VERSION_1_2
typedef cl_int (CL_API_CALL *_sclpfn_clEnqueueMarkerWithWaitList)(
    cl_command_queue command_queue,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event);

static _sclpfn_clEnqueueMarkerWithWaitList _sclEnqueueMarkerWithWaitList = nullptr;

CL_API_ENTRY cl_int CL_API_CALL clEnqueueMarkerWithWaitList(
    cl_command_queue command_queue,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    if (_sclEnqueueMarkerWithWaitList) {
        return _sclEnqueueMarkerWithWaitList(
            command_queue,
            num_events_in_wait_list,
            event_wait_list,
            event);
    }
    return CL_INVALID_OPERATION;
}
#endif

#ifdef CL_VERSION_1_2
typedef cl_int (CL_API_CALL *_sclpfn_clEnqueueBarrierWithWaitList)(
    cl_command_queue command_queue,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event);

static _sclpfn_clEnqueueBarrierWithWaitList _sclEnqueueBarrierWithWaitList = nullptr;

CL_API_ENTRY cl_int CL_API_CALL clEnqueueBarrierWithWaitList(
    cl_command_queue command_queue,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    if (_sclEnqueueBarrierWithWaitList) {
        return _sclEnqueueBarrierWithWaitList(
            command_queue,
            num_events_in_wait_list,
            event_wait_list,
            event);
    }
    return CL_INVALID_OPERATION;
}
#endif

#ifdef CL_VERSION_1_2
typedef void* (CL_API_CALL *_sclpfn_clGetExtensionFunctionAddressForPlatform)(
    cl_platform_id platform,
    const char* func_name);

static _sclpfn_clGetExtensionFunctionAddressForPlatform _sclGetExtensionFunctionAddressForPlatform = nullptr;

CL_API_ENTRY void* CL_API_CALL clGetExtensionFunctionAddressForPlatform(
    cl_platform_id platform,
    const char* func_name)
{
    if (_sclGetExtensionFunctionAddressForPlatform) {
        return _sclGetExtensionFunctionAddressForPlatform(
            platform,
            func_name);
    }
    return nullptr;
}
#endif

#ifdef CL_VERSION_2_0
typedef cl_command_queue (CL_API_CALL *_sclpfn_clCreateCommandQueueWithProperties)(
    cl_context context,
    cl_device_id device,
    const cl_queue_properties* properties,
    cl_int* errcode_ret);

static _sclpfn_clCreateCommandQueueWithProperties _sclCreateCommandQueueWithProperties = nullptr;

CL_API_ENTRY cl_command_queue CL_API_CALL clCreateCommandQueueWithProperties(
    cl_context context,
    cl_device_id device,
    const cl_queue_properties* properties,
    cl_int* errcode_ret)
{
    if (_sclCreateCommandQueueWithProperties) {
        return _sclCreateCommandQueueWithProperties(
            context,
            device,
            properties,
            errcode_ret);
    }
    if (errcode_ret) *errcode_ret = CL_INVALID_OPERATION;
    return NULL;
}
#endif

#ifdef CL_VERSION_2_0
typedef cl_mem (CL_API_CALL *_sclpfn_clCreatePipe)(
    cl_context context,
    cl_mem_flags flags,
    cl_uint pipe_packet_size,
    cl_uint pipe_max_packets,
    const cl_pipe_properties* properties,
    cl_int* errcode_ret);

static _sclpfn_clCreatePipe _sclCreatePipe = nullptr;

CL_API_ENTRY cl_mem CL_API_CALL clCreatePipe(
    cl_context context,
    cl_mem_flags flags,
    cl_uint pipe_packet_size,
    cl_uint pipe_max_packets,
    const cl_pipe_properties* properties,
    cl_int* errcode_ret)
{
    if (_sclCreatePipe) {
        return _sclCreatePipe(
            context,
            flags,
            pipe_packet_size,
            pipe_max_packets,
            properties,
            errcode_ret);
    }
    if (errcode_ret) *errcode_ret = CL_INVALID_OPERATION;
    return NULL;
}
#endif

#ifdef CL_VERSION_2_0
typedef cl_int (CL_API_CALL *_sclpfn_clGetPipeInfo)(
    cl_mem pipe,
    cl_pipe_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret);

static _sclpfn_clGetPipeInfo _sclGetPipeInfo = nullptr;

CL_API_ENTRY cl_int CL_API_CALL clGetPipeInfo(
    cl_mem pipe,
    cl_pipe_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    if (_sclGetPipeInfo) {
        return _sclGetPipeInfo(
            pipe,
            param_name,
            param_value_size,
            param_value,
            param_value_size_ret);
    }
    return CL_INVALID_OPERATION;
}
#endif

#ifdef CL_VERSION_2_0
typedef void* (CL_API_CALL *_sclpfn_clSVMAlloc)(
    cl_context context,
    cl_svm_mem_flags flags,
    size_t size,
    cl_uint alignment);

static _sclpfn_clSVMAlloc _sclSVMAlloc = nullptr;

CL_API_ENTRY void* CL_API_CALL clSVMAlloc(
    cl_context context,
    cl_svm_mem_flags flags,
    size_t size,
    cl_uint alignment)
{
    if (_sclSVMAlloc) {
        return _sclSVMAlloc(
            context,
            flags,
            size,
            alignment);
    }
    return nullptr;
}
#endif

#ifdef CL_VERSION_2_0
typedef void (CL_API_CALL *_sclpfn_clSVMFree)(
    cl_context context,
    void* svm_pointer);

static _sclpfn_clSVMFree _sclSVMFree = nullptr;

CL_API_ENTRY void CL_API_CALL clSVMFree(
    cl_context context,
    void* svm_pointer)
{
    if (_sclSVMFree) {
        return _sclSVMFree(
            context,
            svm_pointer);
    }
}
#endif

#ifdef CL_VERSION_2_0
typedef cl_sampler (CL_API_CALL *_sclpfn_clCreateSamplerWithProperties)(
    cl_context context,
    const cl_sampler_properties* sampler_properties,
    cl_int* errcode_ret);

static _sclpfn_clCreateSamplerWithProperties _sclCreateSamplerWithProperties = nullptr;

CL_API_ENTRY cl_sampler CL_API_CALL clCreateSamplerWithProperties(
    cl_context context,
    const cl_sampler_properties* sampler_properties,
    cl_int* errcode_ret)
{
    if (_sclCreateSamplerWithProperties) {
        return _sclCreateSamplerWithProperties(
            context,
            sampler_properties,
            errcode_ret);
    }
    if (errcode_ret) *errcode_ret = CL_INVALID_OPERATION;
    return NULL;
}
#endif

#ifdef CL_VERSION_2_0
typedef cl_int (CL_API_CALL *_sclpfn_clSetKernelArgSVMPointer)(
    cl_kernel kernel,
    cl_uint arg_index,
    const void* arg_value);

static _sclpfn_clSetKernelArgSVMPointer _sclSetKernelArgSVMPointer = nullptr;

CL_API_ENTRY cl_int CL_API_CALL clSetKernelArgSVMPointer(
    cl_kernel kernel,
    cl_uint arg_index,
    const void* arg_value)
{
    if (_sclSetKernelArgSVMPointer) {
        return _sclSetKernelArgSVMPointer(
            kernel,
            arg_index,
            arg_value);
    }
    return CL_INVALID_OPERATION;
}
#endif

#ifdef CL_VERSION_2_0
typedef cl_int (CL_API_CALL *_sclpfn_clSetKernelExecInfo)(
    cl_kernel kernel,
    cl_kernel_exec_info param_name,
    size_t param_value_size,
    const void* param_value);

static _sclpfn_clSetKernelExecInfo _sclSetKernelExecInfo = nullptr;

CL_API_ENTRY cl_int CL_API_CALL clSetKernelExecInfo(
    cl_kernel kernel,
    cl_kernel_exec_info param_name,
    size_t param_value_size,
    const void* param_value)
{
    if (_sclSetKernelExecInfo) {
        return _sclSetKernelExecInfo(
            kernel,
            param_name,
            param_value_size,
            param_value);
    }
    return CL_INVALID_OPERATION;
}
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

static _sclpfn_clEnqueueSVMFree _sclEnqueueSVMFree = nullptr;

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
    if (_sclEnqueueSVMFree) {
        return _sclEnqueueSVMFree(
            command_queue,
            num_svm_pointers,
            svm_pointers,
            pfn_free_func,
            user_data,
            num_events_in_wait_list,
            event_wait_list,
            event);
    }
    return CL_INVALID_OPERATION;
}
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

static _sclpfn_clEnqueueSVMMemcpy _sclEnqueueSVMMemcpy = nullptr;

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
    if (_sclEnqueueSVMMemcpy) {
        return _sclEnqueueSVMMemcpy(
            command_queue,
            blocking_copy,
            dst_ptr,
            src_ptr,
            size,
            num_events_in_wait_list,
            event_wait_list,
            event);
    }
    return CL_INVALID_OPERATION;
}
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

static _sclpfn_clEnqueueSVMMemFill _sclEnqueueSVMMemFill = nullptr;

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
    if (_sclEnqueueSVMMemFill) {
        return _sclEnqueueSVMMemFill(
            command_queue,
            svm_ptr,
            pattern,
            pattern_size,
            size,
            num_events_in_wait_list,
            event_wait_list,
            event);
    }
    return CL_INVALID_OPERATION;
}
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

static _sclpfn_clEnqueueSVMMap _sclEnqueueSVMMap = nullptr;

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
    if (_sclEnqueueSVMMap) {
        return _sclEnqueueSVMMap(
            command_queue,
            blocking_map,
            flags,
            svm_ptr,
            size,
            num_events_in_wait_list,
            event_wait_list,
            event);
    }
    return CL_INVALID_OPERATION;
}
#endif

#ifdef CL_VERSION_2_0
typedef cl_int (CL_API_CALL *_sclpfn_clEnqueueSVMUnmap)(
    cl_command_queue command_queue,
    void* svm_ptr,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event);

static _sclpfn_clEnqueueSVMUnmap _sclEnqueueSVMUnmap = nullptr;

CL_API_ENTRY cl_int CL_API_CALL clEnqueueSVMUnmap(
    cl_command_queue command_queue,
    void* svm_ptr,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
    if (_sclEnqueueSVMUnmap) {
        return _sclEnqueueSVMUnmap(
            command_queue,
            svm_ptr,
            num_events_in_wait_list,
            event_wait_list,
            event);
    }
    return CL_INVALID_OPERATION;
}
#endif

#ifdef CL_VERSION_2_1
typedef cl_int (CL_API_CALL *_sclpfn_clSetDefaultDeviceCommandQueue)(
    cl_context context,
    cl_device_id device,
    cl_command_queue command_queue);

static _sclpfn_clSetDefaultDeviceCommandQueue _sclSetDefaultDeviceCommandQueue = nullptr;

CL_API_ENTRY cl_int CL_API_CALL clSetDefaultDeviceCommandQueue(
    cl_context context,
    cl_device_id device,
    cl_command_queue command_queue)
{
    if (_sclSetDefaultDeviceCommandQueue) {
        return _sclSetDefaultDeviceCommandQueue(
            context,
            device,
            command_queue);
    }
    return CL_INVALID_OPERATION;
}
#endif

#ifdef CL_VERSION_2_1
typedef cl_int (CL_API_CALL *_sclpfn_clGetDeviceAndHostTimer)(
    cl_device_id device,
    cl_ulong* device_timestamp,
    cl_ulong* host_timestamp);

static _sclpfn_clGetDeviceAndHostTimer _sclGetDeviceAndHostTimer = nullptr;

CL_API_ENTRY cl_int CL_API_CALL clGetDeviceAndHostTimer(
    cl_device_id device,
    cl_ulong* device_timestamp,
    cl_ulong* host_timestamp)
{
    if (_sclGetDeviceAndHostTimer) {
        return _sclGetDeviceAndHostTimer(
            device,
            device_timestamp,
            host_timestamp);
    }
    return CL_INVALID_OPERATION;
}
#endif

#ifdef CL_VERSION_2_1
typedef cl_int (CL_API_CALL *_sclpfn_clGetHostTimer)(
    cl_device_id device,
    cl_ulong* host_timestamp);

static _sclpfn_clGetHostTimer _sclGetHostTimer = nullptr;

CL_API_ENTRY cl_int CL_API_CALL clGetHostTimer(
    cl_device_id device,
    cl_ulong* host_timestamp)
{
    if (_sclGetHostTimer) {
        return _sclGetHostTimer(
            device,
            host_timestamp);
    }
    return CL_INVALID_OPERATION;
}
#endif

#ifdef CL_VERSION_2_1
typedef cl_program (CL_API_CALL *_sclpfn_clCreateProgramWithIL)(
    cl_context context,
    const void* il,
    size_t length,
    cl_int* errcode_ret);

static _sclpfn_clCreateProgramWithIL _sclCreateProgramWithIL = nullptr;

CL_API_ENTRY cl_program CL_API_CALL clCreateProgramWithIL(
    cl_context context,
    const void* il,
    size_t length,
    cl_int* errcode_ret)
{
    if (_sclCreateProgramWithIL) {
        return _sclCreateProgramWithIL(
            context,
            il,
            length,
            errcode_ret);
    }
    if (errcode_ret) *errcode_ret = CL_INVALID_OPERATION;
    return NULL;
}
#endif

#ifdef CL_VERSION_2_1
typedef cl_kernel (CL_API_CALL *_sclpfn_clCloneKernel)(
    cl_kernel source_kernel,
    cl_int* errcode_ret);

static _sclpfn_clCloneKernel _sclCloneKernel = nullptr;

CL_API_ENTRY cl_kernel CL_API_CALL clCloneKernel(
    cl_kernel source_kernel,
    cl_int* errcode_ret)
{
    if (_sclCloneKernel) {
        return _sclCloneKernel(
            source_kernel,
            errcode_ret);
    }
    if (errcode_ret) *errcode_ret = CL_INVALID_OPERATION;
    return NULL;
}
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

static _sclpfn_clGetKernelSubGroupInfo _sclGetKernelSubGroupInfo = nullptr;

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
    if (_sclGetKernelSubGroupInfo) {
        return _sclGetKernelSubGroupInfo(
            kernel,
            device,
            param_name,
            input_value_size,
            input_value,
            param_value_size,
            param_value,
            param_value_size_ret);
    }
    return CL_INVALID_OPERATION;
}
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

static _sclpfn_clEnqueueSVMMigrateMem _sclEnqueueSVMMigrateMem = nullptr;

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
    if (_sclEnqueueSVMMigrateMem) {
        return _sclEnqueueSVMMigrateMem(
            command_queue,
            num_svm_pointers,
            svm_pointers,
            sizes,
            flags,
            num_events_in_wait_list,
            event_wait_list,
            event);
    }
    return CL_INVALID_OPERATION;
}
#endif

#ifdef CL_VERSION_2_2
typedef cl_int (CL_API_CALL *_sclpfn_clSetProgramReleaseCallback)(
    cl_program program,
    void (CL_CALLBACK* pfn_notify)(cl_program program, void* user_data),
    void* user_data);

static _sclpfn_clSetProgramReleaseCallback _sclSetProgramReleaseCallback = nullptr;

CL_API_ENTRY cl_int CL_API_CALL clSetProgramReleaseCallback(
    cl_program program,
    void (CL_CALLBACK* pfn_notify)(cl_program program, void* user_data),
    void* user_data)
{
    if (_sclSetProgramReleaseCallback) {
        return _sclSetProgramReleaseCallback(
            program,
            pfn_notify,
            user_data);
    }
    return CL_INVALID_OPERATION;
}
#endif

#ifdef CL_VERSION_2_2
typedef cl_int (CL_API_CALL *_sclpfn_clSetProgramSpecializationConstant)(
    cl_program program,
    cl_uint spec_id,
    size_t spec_size,
    const void* spec_value);

static _sclpfn_clSetProgramSpecializationConstant _sclSetProgramSpecializationConstant = nullptr;

CL_API_ENTRY cl_int CL_API_CALL clSetProgramSpecializationConstant(
    cl_program program,
    cl_uint spec_id,
    size_t spec_size,
    const void* spec_value)
{
    if (_sclSetProgramSpecializationConstant) {
        return _sclSetProgramSpecializationConstant(
            program,
            spec_id,
            spec_size,
            spec_value);
    }
    return CL_INVALID_OPERATION;
}
#endif

///////////////////////////////////////////////////////////////////////////////

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

#define GET_FUNCTION(_funcname)                                         \
    _s ## _funcname = ( _sclpfn_ ## _funcname )                         \
        _sclGetFunctionAddress(module, #_funcname);

static void _sclInit(void)
{
    static _sclModuleHandle module = nullptr;
    
    if (module == nullptr) {
        module = _sclOpenICDLoader();

#ifdef CL_VERSION_1_0
        GET_FUNCTION(clGetPlatformIDs);
        GET_FUNCTION(clGetPlatformInfo);
        GET_FUNCTION(clGetDeviceIDs);
        GET_FUNCTION(clGetDeviceInfo);
        GET_FUNCTION(clCreateContext);
        GET_FUNCTION(clCreateContextFromType);
        GET_FUNCTION(clRetainContext);
        GET_FUNCTION(clReleaseContext);
        GET_FUNCTION(clGetContextInfo);
        GET_FUNCTION(clCreateCommandQueue);
        GET_FUNCTION(clRetainCommandQueue);
        GET_FUNCTION(clReleaseCommandQueue);
        GET_FUNCTION(clGetCommandQueueInfo);
        GET_FUNCTION(clSetCommandQueueProperty);
        GET_FUNCTION(clCreateBuffer);
        GET_FUNCTION(clCreateImage2D);
        GET_FUNCTION(clCreateImage3D);
        GET_FUNCTION(clRetainMemObject);
        GET_FUNCTION(clReleaseMemObject);
        GET_FUNCTION(clGetSupportedImageFormats);
        GET_FUNCTION(clGetMemObjectInfo);
        GET_FUNCTION(clGetImageInfo);
        GET_FUNCTION(clCreateSampler);
        GET_FUNCTION(clRetainSampler);
        GET_FUNCTION(clReleaseSampler);
        GET_FUNCTION(clGetSamplerInfo);
        GET_FUNCTION(clCreateProgramWithSource);
        GET_FUNCTION(clCreateProgramWithBinary);
        GET_FUNCTION(clRetainProgram);
        GET_FUNCTION(clReleaseProgram);
        GET_FUNCTION(clBuildProgram);
        GET_FUNCTION(clUnloadCompiler);
        GET_FUNCTION(clGetProgramInfo);
        GET_FUNCTION(clGetProgramBuildInfo);
        GET_FUNCTION(clCreateKernel);
        GET_FUNCTION(clCreateKernelsInProgram);
        GET_FUNCTION(clRetainKernel);
        GET_FUNCTION(clReleaseKernel);
        GET_FUNCTION(clSetKernelArg);
        GET_FUNCTION(clGetKernelInfo);
        GET_FUNCTION(clGetKernelWorkGroupInfo);
        GET_FUNCTION(clWaitForEvents);
        GET_FUNCTION(clGetEventInfo);
        GET_FUNCTION(clRetainEvent);
        GET_FUNCTION(clReleaseEvent);
        GET_FUNCTION(clGetEventProfilingInfo);
        GET_FUNCTION(clFlush);
        GET_FUNCTION(clFinish);
        GET_FUNCTION(clEnqueueReadBuffer);
        GET_FUNCTION(clEnqueueWriteBuffer);
        GET_FUNCTION(clEnqueueCopyBuffer);
        GET_FUNCTION(clEnqueueReadImage);
        GET_FUNCTION(clEnqueueWriteImage);
        GET_FUNCTION(clEnqueueCopyImage);
        GET_FUNCTION(clEnqueueCopyImageToBuffer);
        GET_FUNCTION(clEnqueueCopyBufferToImage);
        GET_FUNCTION(clEnqueueMapBuffer);
        GET_FUNCTION(clEnqueueMapImage);
        GET_FUNCTION(clEnqueueUnmapMemObject);
        GET_FUNCTION(clEnqueueNDRangeKernel);
        GET_FUNCTION(clEnqueueTask);
        GET_FUNCTION(clEnqueueNativeKernel);
        GET_FUNCTION(clEnqueueMarker);
        GET_FUNCTION(clEnqueueWaitForEvents);
        GET_FUNCTION(clEnqueueBarrier);
        GET_FUNCTION(clGetExtensionFunctionAddress);
#endif

#ifdef CL_VERSION_1_1
        GET_FUNCTION(clSetEventCallback);
        GET_FUNCTION(clCreateSubBuffer);
        GET_FUNCTION(clSetMemObjectDestructorCallback);
        GET_FUNCTION(clCreateUserEvent);
        GET_FUNCTION(clSetUserEventStatus);
        GET_FUNCTION(clEnqueueReadBufferRect);
        GET_FUNCTION(clEnqueueWriteBufferRect);
        GET_FUNCTION(clEnqueueCopyBufferRect);
#endif

#ifdef CL_VERSION_1_2
        GET_FUNCTION(clCreateSubDevices);
        GET_FUNCTION(clRetainDevice);
        GET_FUNCTION(clReleaseDevice);
        GET_FUNCTION(clCreateImage);
        GET_FUNCTION(clCreateProgramWithBuiltInKernels);
        GET_FUNCTION(clCompileProgram);
        GET_FUNCTION(clLinkProgram);
        GET_FUNCTION(clUnloadPlatformCompiler);
        GET_FUNCTION(clGetKernelArgInfo);
        GET_FUNCTION(clEnqueueFillBuffer);
        GET_FUNCTION(clEnqueueFillImage);
        GET_FUNCTION(clEnqueueMigrateMemObjects);
        GET_FUNCTION(clEnqueueMarkerWithWaitList);
        GET_FUNCTION(clEnqueueBarrierWithWaitList);
        GET_FUNCTION(clGetExtensionFunctionAddressForPlatform);
#endif

#ifdef CL_VERSION_2_0
        GET_FUNCTION(clCreateCommandQueueWithProperties);
        GET_FUNCTION(clCreatePipe);
        GET_FUNCTION(clGetPipeInfo);
        GET_FUNCTION(clSVMAlloc);
        GET_FUNCTION(clSVMFree);
        GET_FUNCTION(clEnqueueSVMFree);
        GET_FUNCTION(clEnqueueSVMMemcpy);
        GET_FUNCTION(clEnqueueSVMMemFill);
        GET_FUNCTION(clEnqueueSVMMap);
        GET_FUNCTION(clEnqueueSVMUnmap);
        GET_FUNCTION(clCreateSamplerWithProperties);
        GET_FUNCTION(clSetKernelArgSVMPointer);
        GET_FUNCTION(clSetKernelExecInfo);
#endif

#ifdef CL_VERSION_2_1
        GET_FUNCTION(clCloneKernel);
        GET_FUNCTION(clCreateProgramWithIL);
        GET_FUNCTION(clEnqueueSVMMigrateMem);
        GET_FUNCTION(clGetDeviceAndHostTimer);
        GET_FUNCTION(clGetHostTimer);
        GET_FUNCTION(clGetKernelSubGroupInfo);
        GET_FUNCTION(clSetDefaultDeviceCommandQueue);
#endif

#ifdef CL_VERSION_2_2
        GET_FUNCTION(clSetProgramReleaseCallback);
        GET_FUNCTION(clSetProgramSpecializationConstant);
#endif
    }
}
