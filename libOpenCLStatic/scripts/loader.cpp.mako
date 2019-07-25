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

<%
# These APIs initialize the function pointers:
apiinit = {
    'clGetPlatformIDs',                 # calls into the ICD loader
    'clGetExtensionFunctionAddress',    # could call into the ICD loader or return NULL
    }
%>
%for function in spec.findall('feature/require/command'):
<%
      api = apisigs[function.get('name')]
%>
#ifdef ${apivers[api.Name]}
typedef ${api.RetType} (CL_API_CALL *_sclpfn_${api.Name})(
%for i, param in enumerate(api.Params):
%  if i < len(api.Params)-1:
    ${param.Type} ${param.Name}${param.TypeEnd},
%  else:
    ${param.Type} ${param.Name}${param.TypeEnd});
%  endif
%endfor

static _sclpfn_${api.Name} _s${api.Name} = nullptr;

CL_API_ENTRY ${api.RetType} CL_API_CALL ${api.Name}(
%for i, param in enumerate(api.Params):
%  if i < len(api.Params)-1:
    ${param.Type} ${param.Name}${param.TypeEnd},
%  else:
    ${param.Type} ${param.Name}${param.TypeEnd})
%  endif
%endfor
{
%if api.Name in apiinit:
    _sclInit();
%endif
    if (_s${api.Name}) {
        return _s${api.Name}(
%for i, param in enumerate(api.Params):
%  if i < len(api.Params)-1:
            ${param.Name},
%  else:
            ${param.Name});
%  endif
%endfor
    }
%if api.RetType in apihandles:
    if (errcode_ret) *errcode_ret = CL_INVALID_OPERATION;
    return NULL;
%elif api.RetType == "void*":
    return nullptr;
%elif api.RetType != "void":
    return CL_INVALID_OPERATION;
%endif
}
#endif
%endfor

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

#define GET_FUNCTION(_funcname)                                         ${"\\"}
    _s ## _funcname = ( _sclpfn_ ## _funcname )                         ${"\\"}
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
