# hmemhelloworld

## Sample Purpose

This sample demonstrates usage of a host memory allocation.
Other similar samples demonstrate usage of device memory and shared memory allocations.

Host memory allocations are owned by the host, and generally trade wide access for potentially lower performance.
Because of its wide access, using host memory is one of the easiest ways to enable an application to use Unified Shared Memory, albeit at a potential performance cost.

The sample initializes a source USM allocation, copies it to a destination USM allocation using a kernel, then checks that the copy occurred correctly.

## Key APIs and Concepts

This sample allocates host memory using `clHostMemAllocINTEL` and frees it using `clMemFreeINTEL`.

Since host memory may be directly accessed and manipulated on the host, this sample does not need to use any special Unified Shared Memory APIs to copy to or from a host allocation, or to map or unmap a host allocation.
Instead, this sample may simply ensure that copy kernel is complete before verifying that the copy occurred correctly.
For simplicity, this sample ensures all commands in the command queue are complete using `clFinish`, but other completion mechanisms could be used instead that may be more efficient.

Within a kernel, a Unified Shared Memory allocation can be accessed similar to an OpenCL buffer (a `cl_mem`), or a Shared Virtual Memory allocation.
A Unified Shared Memory allocation may be set as an argument to a kernel using `clSetKernelArgMemPointerINTEL`.

Since Unified Shared Memory is an OpenCL extension, this sample uses the `libusm` library to query the extension APIs.
Please see the `libusm` [README](../libusm/README.md) for more detail.

This sample currently uses c APIs because the C++ bindings do not support Unified Shared Memory (yet).
When support for Unified Shared Memory is added to the C++ bindings the samples will be updated to use the C++ bindings instead, which should simplify the sample slightly.

## Command Line Options

| Option | Default Value | Description |
|:--|:-:|:--|
| `-d <index>` | 0 | Specify the index of the OpenCL device in the platform to execute on the sample on.
| `-p <index>` | 0 | Specify the index of the OpenCL platform to execute the sample on.
