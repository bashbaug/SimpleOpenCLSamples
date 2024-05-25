# dmemhelloworld

## Sample Purpose

This is the first Unified Shared Memory sample that meaningfully stores and uses data in a Unified Shared Memory allocation.

This sample demonstrates usage of device memory allocations.
Other similar samples demonstrate usage of host memory and shared memory allocations.
Device memory allocations are owned by a specific device, and generally trade off high performance for limited access.
Kernels operating on device memory should perform just as well, if not better, than OpenCL buffers or Shared Virtual Memory allocations.

The sample initializes a source USM allocation, copies it to a destination USM allocation using a kernel, then checks on the host that the copy was performed correctly.

## Key APIs and Concepts

This sample allocates device memory using `clDeviceMemAllocINTEL` and frees it using `clMemFreeINTEL`.

Since device memory cannot be directly accessed by the host, this sample initializes the source buffer by copying into it using `clEnqueueMemcpyINTEL`.
This sample also uses `clEnqueueMemcpyINTEL` to copy out of the destination buffer to verify that the copy was performed correctly.

Within a kernel, a Unified Shared Memory allocation can be accessed similar to an OpenCL buffer (a `cl_mem`), or a Shared Virtual Memory allocation.
Unified Shared Memory allocations are set as an argument to a kernel using `clSetKernelArgMemPointerINTEL`.

Since Unified Shared Memory is an OpenCL extension, this sample uses the `OpenCLExt` extension loader library to query the extension APIs.
Please see the OpenCL Extension Loader [README](https://github.com/bashbaug/opencl-extension-loader) for more detail.

This sample currently uses c APIs because the C++ bindings do not support Unified Shared Memory (yet).
When support for Unified Shared Memory is added to the C++ bindings the samples will be updated to use the C++ bindings instead, which should simplify the sample slightly.

## Command Line Options

| Option | Default Value | Description |
|:--|:-:|:--|
| `-d <index>` | 0 | Specify the index of the OpenCL device in the platform to execute on the sample on.
| `-p <index>` | 0 | Specify the index of the OpenCL platform to execute the sample on.
