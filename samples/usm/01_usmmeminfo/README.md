# usmmeminfo

## Sample Purpose

This sample allocates Unified Shared Memory of each supported type and queries the properties of each allocation.
These properties can be used to determine if a pointer points to a USM allocation, and if so, how the USM allocation may be used on a device.

All properties qre queried for a pointer into the middle of an allocation.
The "type" property of the allocation is queried for the base address of the allocation, a pointer into the middle of the allocation, and an out-of-range pointer.

## Key APIs and Concepts

This sample primarily demonstrates the `clGetMemAllocInfoINTEL` API to query properties of a Unified Shared Memory allocation.
This is also the first sample that allocates (and frees) Unified Shared Memory.
This sample allocates host memory (using `clHostMemAllocINTEL`), device memory (using `clDeviceMemAllocINTEL`), and shared memory (using `clSharedMemAllocINTEL`).
When all queries are complete and the USM allocation is no longer required, the allocation is freed using `clMemFreeINTEL`.

Since Unified Shared Memory is an OpenCL extension, this sample uses the `libusm` library to query the extension APIs.
Please see the `libusm` [README](../libusm/README.md) for more detail.

This sample currently uses c APIs because the C++ bindings do not support Unified Shared Memory (yet).
When support for Unified Shared Memory is added to the C++ bindings the samples will be updated to use the C++ bindings instead, which should simplify the sample slightly.

## Command Line Options

| Option | Default Value | Description |
|:--|:-:|:--|
| `-d <index>` | 0 | Specify the index of the OpenCL device in the platform to execute on the sample on.
| `-p <index>` | 0 | Specify the index of the OpenCL platform to execute the sample on.
