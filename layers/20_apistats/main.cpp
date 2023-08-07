/*
// Copyright (c) 2022 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#if defined _WIN32 || defined __CYGWIN__
#ifdef __GNUC__
#define CL_API_ENTRY __attribute__((dllexport))
#else
#define CL_API_ENTRY __declspec(dllexport)
#endif
#else
#if __GNUC__ >= 4
#define CL_API_ENTRY __attribute__((visibility("default")))
#else
#define CL_API_ENTRY
#endif
#endif

#include <CL/cl_layer.h>

#include "layer_util.hpp"

#include "dispatch.h"

const struct _cl_icd_dispatch* g_pNextDispatch = NULL;

static struct _cl_icd_dispatch dispatch = {
    // OpenCL 1.0
    clGetPlatformIDs_layer,
    clGetPlatformInfo_layer,
    clGetDeviceIDs_layer,
    clGetDeviceInfo_layer,
    clCreateContext_layer,
    clCreateContextFromType_layer,
    clRetainContext_layer,
    clReleaseContext_layer,
    clGetContextInfo_layer,
    clCreateCommandQueue_layer,
    clRetainCommandQueue_layer,
    clReleaseCommandQueue_layer,
    clGetCommandQueueInfo_layer,
    clSetCommandQueueProperty_layer,
    clCreateBuffer_layer,
    clCreateImage2D_layer,
    clCreateImage3D_layer,
    clRetainMemObject_layer,
    clReleaseMemObject_layer,
    clGetSupportedImageFormats_layer,
    clGetMemObjectInfo_layer,
    clGetImageInfo_layer,
    clCreateSampler_layer,
    clRetainSampler_layer,
    clReleaseSampler_layer,
    clGetSamplerInfo_layer,
    clCreateProgramWithSource_layer,
    clCreateProgramWithBinary_layer,
    clRetainProgram_layer,
    clReleaseProgram_layer,
    clBuildProgram_layer,
    clUnloadCompiler_layer,
    clGetProgramInfo_layer,
    clGetProgramBuildInfo_layer,
    clCreateKernel_layer,
    clCreateKernelsInProgram_layer,
    clRetainKernel_layer,
    clReleaseKernel_layer,
    clSetKernelArg_layer,
    clGetKernelInfo_layer,
    clGetKernelWorkGroupInfo_layer,
    clWaitForEvents_layer,
    clGetEventInfo_layer,
    clRetainEvent_layer,
    clReleaseEvent_layer,
    clGetEventProfilingInfo_layer,
    clFlush_layer,
    clFinish_layer,
    clEnqueueReadBuffer_layer,
    clEnqueueWriteBuffer_layer,
    clEnqueueCopyBuffer_layer,
    clEnqueueReadImage_layer,
    clEnqueueWriteImage_layer,
    clEnqueueCopyImage_layer,
    clEnqueueCopyImageToBuffer_layer,
    clEnqueueCopyBufferToImage_layer,
    clEnqueueMapBuffer_layer,
    clEnqueueMapImage_layer,
    clEnqueueUnmapMemObject_layer,
    clEnqueueNDRangeKernel_layer,
    clEnqueueTask_layer,
    clEnqueueNativeKernel_layer,
    clEnqueueMarker_layer,
    clEnqueueWaitForEvents_layer,
    clEnqueueBarrier_layer,
    clGetExtensionFunctionAddress_layer,

    nullptr, // clCreateFromGLBuffer
    nullptr, // clCreateFromGLTexture2D
    nullptr, // clCreateFromGLTexture3D
    nullptr, // clCreateFromGLRenderbuffer
    nullptr, // clGetGLObjectInfo
    nullptr, // clGetGLTextureInfo
    nullptr, // clEnqueueAcquireGLObjects
    nullptr, // clEnqueueReleaseGLObjects
    nullptr, // clGetGLContextInfoKHR

    // cl_khr_d3d10_sharing
    nullptr, // clGetDeviceIDsFromD3D10KHR
    nullptr, // clCreateFromD3D10BufferKHR
    nullptr, // clCreateFromD3D10Texture2DKHR
    nullptr, // clCreateFromD3D10Texture3DKHR
    nullptr, // clEnqueueAcquireD3D10ObjectsKHR
    nullptr, // clEnqueueReleaseD3D10ObjectsKHR

    // OpenCL 1.1
    clSetEventCallback_layer,
    clCreateSubBuffer_layer,
    clSetMemObjectDestructorCallback_layer,
    clCreateUserEvent_layer,
    clSetUserEventStatus_layer,
    clEnqueueReadBufferRect_layer,
    clEnqueueWriteBufferRect_layer,
    clEnqueueCopyBufferRect_layer,

    // cl_ext_device_fission
    nullptr, // clCreateSubDevicesEXT
    nullptr, // clRetainDeviceEXT
    nullptr, // clReleaseDeviceEXT

    // cl_khr_gl_event
    nullptr, // clCreateEventFromGLsyncKHR

    // OpenCL 1.2
    clCreateSubDevices_layer,
    clRetainDevice_layer,
    clReleaseDevice_layer,
    clCreateImage_layer,
    clCreateProgramWithBuiltInKernels_layer,
    clCompileProgram_layer,
    clLinkProgram_layer,
    clUnloadPlatformCompiler_layer,
    clGetKernelArgInfo_layer,
    clEnqueueFillBuffer_layer,
    clEnqueueFillImage_layer,
    clEnqueueMigrateMemObjects_layer,
    clEnqueueMarkerWithWaitList_layer,
    clEnqueueBarrierWithWaitList_layer,
    clGetExtensionFunctionAddressForPlatform_layer,

    nullptr, // clCreateFromGLTexture

    // cl_khr_d3d11_sharing
    nullptr, // clGetDeviceIDsFromD3D11KHR
    nullptr, // clCreateFromD3D11BufferKHR
    nullptr, // clCreateFromD3D11Texture2DKHR
    nullptr, // clCreateFromD3D11Texture3DKHR
    nullptr, // clCreateFromDX9MediaSurfaceKHR
    nullptr, // clEnqueueAcquireD3D11ObjectsKHR
    nullptr, // clEnqueueReleaseD3D11ObjectsKHR
    
    // cl_khr_dx9_media_sharing
    nullptr, // clGetDeviceIDsFromDX9MediaAdapterKHR
    nullptr, // clEnqueueAcquireDX9MediaSurfacesKHR
    nullptr, // clEnqueueReleaseDX9MediaSurfacesKHR

    // cl_khr_egl_image
    nullptr, // clCreateFromEGLImageKHR
    nullptr, // clEnqueueAcquireEGLObjectsKHR
    nullptr, // clEnqueueReleaseEGLObjectsKHR

    // cl_khr_egl_event
    nullptr, // clCreateEventFromEGLSyncKHR

    // OpenCL 2.0
    clCreateCommandQueueWithProperties_layer,
    clCreatePipe_layer,
    clGetPipeInfo_layer,
    clSVMAlloc_layer,
    clSVMFree_layer,
    clEnqueueSVMFree_layer,
    clEnqueueSVMMemcpy_layer,
    clEnqueueSVMMemFill_layer,
    clEnqueueSVMMap_layer,
    clEnqueueSVMUnmap_layer,
    clCreateSamplerWithProperties_layer,
    clSetKernelArgSVMPointer_layer,
    clSetKernelExecInfo_layer,

    // cl_khr_sub_groups
    nullptr, // clGetKernelSubGroupInfoKHR

    // OpenCL 2.1
    clCloneKernel_layer,
    clCreateProgramWithIL_layer,
    clEnqueueSVMMigrateMem_layer,
    clGetDeviceAndHostTimer_layer,
    clGetHostTimer_layer,
    clGetKernelSubGroupInfo_layer,
    clSetDefaultDeviceCommandQueue_layer,

    // OpenCL 2.2
    clSetProgramReleaseCallback_layer,
    clSetProgramSpecializationConstant_layer,

    // OpenCL 3.0
    clCreateBufferWithProperties_layer,
    clCreateImageWithProperties_layer,
    clSetContextDestructorCallback_layer,
};

CL_API_ENTRY cl_int CL_API_CALL clGetLayerInfo(
    cl_layer_info  param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
    switch (param_name) {
    case CL_LAYER_API_VERSION:
        {
            auto ptr = (cl_layer_api_version*)param_value;
            auto value = cl_layer_api_version{CL_LAYER_API_VERSION_100};
            return writeParamToMemory(
                param_value_size,
                value,
                param_value_size_ret,
                ptr);
        }
        break;
#if defined(CL_LAYER_NAME)
    case CL_LAYER_NAME:
        {
            auto ptr = (char*)param_value;
            return writeStringToMemory(
                param_value_size,
                "API Stats Collection and Reporting Layer",
                param_value_size_ret,
                ptr);
        }
        break;
#endif
    default:
        return CL_INVALID_VALUE;
    }
    return CL_SUCCESS;
}

CL_API_ENTRY cl_int CL_API_CALL clInitLayer(
    cl_uint num_entries,
    const struct _cl_icd_dispatch* target_dispatch,
    cl_uint* num_entries_out,
    const struct _cl_icd_dispatch** layer_dispatch_ret)
{
    const size_t dispatchTableSize =
        sizeof(dispatch) / sizeof(dispatch.clGetPlatformIDs);

    if (target_dispatch == nullptr || 
        num_entries_out == nullptr ||
        layer_dispatch_ret == nullptr) {
        return CL_INVALID_VALUE;
    }

    if (num_entries < dispatchTableSize) {
        return CL_INVALID_VALUE;
    }

    g_pNextDispatch = target_dispatch;

    *layer_dispatch_ret = &dispatch;
    *num_entries_out = dispatchTableSize;

    return CL_SUCCESS;
}
