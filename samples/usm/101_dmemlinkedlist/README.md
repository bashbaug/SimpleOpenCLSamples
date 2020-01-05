# dmemlinkedlist

## Sample Purpose

This sample demonstrates how to build and initialize a linked list using device Unified Shared Memory, access and modify the linked list in a kernel, then access and check the contents of the linked list on the host.

Because device Unified Shared Memory cannot be directly read from or written to on the host, the linked list must be constructed and verified using explicit memory copies.

## Key APIs and Concepts

This sample demonstrates how indicate that a kernel may access any device Unified Shared Memory allocation using `clSetKernelExecInfo` and `CL_KERNEL_EXEC_INFO_INDIRECT_DEVICE_ACCESS_INTEL`, without specifying all allocations explicitly.
For kernels that operate on complex data structures consisting of many Unified Shared Memory allocations,, this can considerably improve API efficiency.

Since Unified Shared Memory is an OpenCL extension, this sample uses the `libusm` library to query the extension APIs.
Please see the `libusm` [README](../libusm/README.md) for more detail.

This sample currently uses c APIs because the C++ bindings do not support Unified Shared Memory (yet).
When support for Unified Shared Memory is added to the C++ bindings the samples will be updated to use the C++ bindings instead, which should simplify the sample slightly.

## Command Line Options

| Option | Default Value | Description |
|:--|:-:|:--|
| `-d <index>` | 0 | Specify the index of the OpenCL device in the platform to execute on the sample on.
| `-p <index>` | 0 | Specify the index of the OpenCL platform to execute the sample on.
| `-n <number>` | 4 | Specify the number of linked list nodes to create.
