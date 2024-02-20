# mutablecommandbufferasserts

## Sample Purpose

This is an intermediate-level sample that demonstrates how to pass assertions guaranteeing certain behavior when modifying command buffers using the OpenCL extension [cl_khr_command_buffer_mutable_dispatch](https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_Ext.html#cl_khr_command_buffer_mutable_dispatch).
As of this writing, `cl_khr_command_buffer_mutable_dispatch` is a provisional extension.
This sample uses the functionality described in v0.9.1 of the extension.

This is an optional extension and some devices may not support `cl_khr_command_buffer_mutable_dispatch`, but the sample may still run using the [cl_khr_command_buffer emulation layer](../../layers/10_cmdbufemu).

This sample requires the OpenCL Extension Loader to get the extension APIs for command buffers.

## Key APIs and Concepts

This sample demonstrates how to pass mutable dispatch assertions when command buffer is created or when an ND-range kernel command is recorded into a command buffer.

```c
CL_COMMAND_BUFFER_MUTABLE_DISPATCH_ASSERTS_KHR
CL_MUTABLE_DISPATCH_ASSERTS_KHR
CL_MUTABLE_DISPATCH_ASSERT_NO_ADDITIONAL_WORK_GROUPS_KHR
```

## Command Line Options

| Option | Default Value | Description |
|:--|:-:|:--|
| `-d <index>` | 0 | Specify the index of the OpenCL device in the platform to execute on the sample on.
| `-p <index>` | 0 | Specify the index of the OpenCL platform to execute the sample on.
| `--noCmdBufAssert` | N/A | Do not pass an assertion when the command buffer is created (`CL_COMMAND_BUFFER_MUTABLE_DISPATCH_ASSERTS_KHR`)
| `--noCmdAssert` | N/A | Do not pass an assertion when the command is recorded into the command buffer (`CL_MUTABLE_DISPATCH_ASSERTS_KHR`)
