# hmemlinkedlist

## Sample Purpose

This sample demonstrates how to build a linked list on the host using host Unified Shared Memory, access and modify the linked list in a kernel, then access and check the contents of the linked list on the host.

Because host Unified Shared Memory can be directly read from and written to on the host, this samples is much more straightforward than the equivalent sample that builds a linked list in device memory.

## Key APIs and Concepts

This sample demonstrates how to indicate that a kernel may access any host Unified Shared Memory allocation using `clSetKernelExecInfo` and `CL_KERNEL_EXEC_INFO_INDIRECT_HOST_ACCESS_INTEL`, without specifying all allocations explicitly.
For kernels that operate on complex data structures consisting of many Unified Shared Memory allocations, this can considerably improve API efficiency.

Since Unified Shared Memory is an OpenCL extension, this sample uses the `OpenCLExt` extension loader library to query the extension APIs.
Please see the OpenCL Extension Loader [README](https://github.com/bashbaug/opencl-extension-loader) for more detail.

This sample currently uses c APIs because the C++ bindings do not support Unified Shared Memory (yet).
When support for Unified Shared Memory is added to the C++ bindings the samples will be updated to use the C++ bindings instead, which should simplify the sample slightly.

## Command Line Options

| Option | Default Value | Description |
|:--|:-:|:--|
| `-d <index>` | 0 | Specify the index of the OpenCL device in the platform to execute on the sample on.
| `-p <index>` | 0 | Specify the index of the OpenCL platform to execute the sample on.
| `-n <number>` | 4 | Specify the number of linked list nodes to create.
