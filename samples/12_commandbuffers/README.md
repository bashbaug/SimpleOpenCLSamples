# commandbuffers

## Sample Purpose

This is a modified version of the copybufferkernel sample that demonstrates how to use the OpenCL extension [cl_khr_command_buffer](https://www.khronos.org/registry/OpenCL/specs/3.0-unified/html/OpenCL_Ext.html#cl_khr_command_buffer).
As of this writing, `cl_khr_command_buffer` is a provisional extension.
This sample uses the functionality described in v0.9.0 of the extension.

This is an optional extension and some devices may not support `cl_khr_command_buffer`, but the sample may still run using the [cl_khr_command_buffer emulation layer](../../layers/10_cmdbufemu).

This sample requires the OpenCL Extension Loader to get the extension APIs for command buffers.

## Key APIs and Concepts

This sample demonstrates how to query the command buffer properties supported by a device, and the properties of a command buffer.

This sample also demonstrates how to create, finalize, and execute a command buffer.

```c
clCreateCommandBufferKHR
clGetCommandBufferInfoKHR
clCommandNDRangeKernelKHR
clFinalizeCommandBufferKHR
clEnqueueCommandBufferKHR
```

## Things to Try

Here are some suggested ways to modify this sample to learn more:

1. Change the kernel arguments after recording the ND-range kernel command command into the command buffer.
Does this affect the command in the command buffer?
2. Try timing the same commands with and without a command buffer.
Is it faster or slower to execute commands from a command buffer?

## Command Line Options

| Option | Default Value | Description |
|:--|:-:|:--|
| `-d <index>` | 0 | Specify the index of the OpenCL device in the platform to execute on the sample on.
| `-p <index>` | 0 | Specify the index of the OpenCL platform to execute the sample on.
