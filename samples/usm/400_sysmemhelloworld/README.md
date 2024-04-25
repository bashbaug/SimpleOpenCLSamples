# smemhelloworld

## Sample Purpose

This sample demonstrates usage of shared system memory allocations.
Shared system memory is allocated using a system allocator, such as `malloc` or `new`.
Other similar samples demonstrate usage of device memory, host memory, and shared memory that is allocated using special unified shared memory APIs.

Just like the shared memory that is allocated using special unified shared memory APIs, shared system memory allocations share ownership and are intended to implicitly migrate between the host and one or more devices.
Shared system memory allocations are the easiest way to enable applications to use Unified Shared Memory, but implementing shared system memory requires support from the OpenCL device, the OpenCL implementation, and the operating system, and its usage is not widespread (yet!).

The sample initializes a source USM allocation, copies it to a destination USM allocation using a kernel, then checks on the host that the copy was performed correctly.

## Key APIs and Concepts

This sample allocates shared system memory using the standard system `malloc` and frees it using `free`.

Since shared system memory may be directly accessed and manipulated on the host, this sample does not need to use any special Unified Shared Memory APIs to copy to or from a shared system allocation, or to map or unmap a shared system allocation.
Instead, this sample simply ensures that copy kernel is complete before verifying that the copy was performed correctly.
For simplicity, this sample ensures all commands in the command queue are complete using `clFinish`, but other completion mechanisms could be used instead that may be more efficient.

Within a kernel, a Unified Shared Memory allocation can be accessed similar to an OpenCL buffer (a `cl_mem`), or a Shared Virtual Memory allocation.
Unified Shared Memory allocations are set as an argument to a kernel using `clSetKernelArgMemPointerINTEL`.

When profiling an application using shared memory allocations, be aware that migrations between the host and the device may be occurring implicitly.
These implicit transfers may cause additional apparent latency when launching a kernel (for transfers to the device) or completion latency (for transfers to the host) versus device memory or host memory allocations.

Since Unified Shared Memory is an OpenCL extension, this sample uses the `OpenCLExt` extension loader library to query the extension APIs.
Please see the OpenCL Extension Loader [README](https://github.com/bashbaug/opencl-extension-loader) for more detail.

This sample currently uses c APIs because the C++ bindings do not support Unified Shared Memory (yet).
When support for Unified Shared Memory is added to the C++ bindings the samples will be updated to use the C++ bindings instead, which should simplify the sample slightly.

## Command Line Options

| Option | Default Value | Description |
|:--|:-:|:--|
| `-d <index>` | 0 | Specify the index of the OpenCL device in the platform to execute on the sample on.
| `-p <index>` | 0 | Specify the index of the OpenCL platform to execute the sample on.
