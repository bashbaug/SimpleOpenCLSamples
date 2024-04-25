# N-Body Simulation with Vulkan

## Sample Purpose

This sample uses OpenCL to compute an [N-body simulation](https://en.wikipedia.org/wiki/N-body_simulation), which is then rendered with Vulkan.

This sample can create an OpenCL buffer directly from the Vulkan vertex buffer when supported.
In order to share the Vulkan vertex buffer with OpenCL, the OpenCL device must support [cl_khr_external_memory](https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_Ext.html#cl_khr_external_memory).
Additionally, the Vulkan device must support exporting the Vulkan buffer as an OS-specific external memory handle, and the OpenCL device must support importing external memory handles of that type.
Creating the OpenCL buffer directly from the Vulkan vertex buffer avoids a memory copy and can improve performance.

For Windows, the external memory handle types that are currently supported are:

* `CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_WIN32_KHR`

For Linux, the external memory handle types that are currently supported are:

* `CL_EXTERNAL_MEMORY_HANDLE_DMA_BUF_KHR`
* `CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_FD_KHR`

This sample can also share semaphores between Vulkan and OpenCL when supported.
In order to share a Vulkan semaphore with OpenCL, the OpenCL device must support [cl_khr_external_semaphore](https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_Ext.html#cl_khr_external_semaphore).
Additionally, the Vulkan device must support exporting the Vulkan semaphore as an OS-specific external semaphore handle, and the OpenCL device must support importing external semaphore handles of that type.
Sharing a semaphore object between Vulkan and OpenCL avoids synchronization on the host and can also improve performance.

For Windows, the external semaphore handle types that are currently supported are:

* `CL_SEMAPHORE_HANDLE_OPAQUE_WIN32_KHR`

For Linux, the external semaphore handle types that are currently supported are:

* `CL_SEMAPHORE_HANDLE_OPAQUE_FD_KHR`

For more information about these extensions, please see [this blog post](https://www.khronos.org/blog/khronos-releases-opencl-3.0-extensions-for-neural-network-inferencing-and-opencl-vulkan-interop).
When the extensions are not supported the sample will still run, although perhaps with lower performance.

Important note: The OpenCL extensions used in this sample are very new!
If the sample does not run correctly please ensure you are using the latest OpenCL drivers for your device, or use the command line option to force copying on the host.

## Key APIs and Concepts

This example shows how to share an Vulkan texture and semaphore with OpenCL.

```c
clEnqueueAcquireExternalMemObjectsKHR
clEnqueueReleaseExternalMemObjectsKHR

clCreateSemaphoreWithPropertiesKHR
clEnqueueSignalSemaphoresKHR
clReleaseSemaphoreKHR
```

## Command Line Options

Note: Many of these command line arguments are identical to the earlier Julia set sample.

| Option | Default Value | Description |
|:--|:-:|:--|
| `-d <index>` | 0 | Specify the index of the OpenCL device in the platform to execute on the sample on.
| `-p <index>` | 0 | Specify the index of the OpenCL platform to execute the sample on.
| `--hostcopy` | n/a | Do not use the `cl_khr_external_memory` extension and unconditionally copy on the host.
| `--hostsync` | n/a | Do not use the `cl_khr_external_semaphore` extension and exclusively synchronize on the host.
| `--nodevicelocal` | n/a | Do not use device local images (`VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT`).  May be useful for debugging.
| `-n` | 1024 | Specify the number of bodies to simulate.
| `-g` | 0| Specify the local work size.  If the local work size is zero a `NULL` local work size is used.
| `-w` | 1024 | Specify the render width in pixels.
| `-h` | 1024 | Specify the render height in pixels.
| `--paused` | n/a | Start with the animation paused.

## Controls While Running

| Control | Description |
|:--|:--|
| `Escape` | Exits from the sample.
| `Space` | Toggle animation.
| `S` | Single-step the simulation.

## How to Generate Vulkan SPIR-V Files

The SPIR-V files for the Vulkan vertex shader and fragment shader were compiled with `glslang`, which is included in the Vulkan SDK.
The command lines used to compile the SPIR-V files were:

```sh
/path/to/glslangvalidator --target-env vulkan1.0 nbodyvk.vert -o nbodyvk.vert.spv
/path/to/glslangvalidator --target-env vulkan1.0 nbodyvk.frag -o nbodyvk.frag.spv
```
