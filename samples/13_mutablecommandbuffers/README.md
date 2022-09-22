# mutablecommandbuffers

## Sample Purpose

This is an intermediate-level sample that demonstrates how to use the OpenCL extension [cl_khr_command_buffer_mutable_dispatch](https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_Ext.html#cl_khr_command_buffer_mutable_dispatch) to modify a command buffer after it has been finalized.
As of this writing, `cl_khr_command_buffer_mutable_dispatch` is a provisional extension.
This sample uses the functionality described in v0.9.0 of the extension.

This is an optional extension and some devices may not support `cl_khr_command_buffer_mutable_dispatch`, but the sample may still run using the [cl_khr_command_buffer emulation layer](../../layers/10_cmdbufemu).

This sample requires the OpenCL Extension Loader to get the extension APIs for command buffers.

## Key APIs and Concepts

This sample demonstrates how to query the mutable dispatch capabilities supported by a device, how to create a mutable command buffer, and how to query the properties of a mutable command.

This sample also demonstrates how to mutate (modify) a command buffer after it has been finalized.

```c
clGetMutableCommandInfoKHR
clUpdateMutableCommandsKHR
```

## Command Line Options

| Option | Default Value | Description |
|:--|:-:|:--|
| `-d <index>` | 0 | Specify the index of the OpenCL device in the platform to execute on the sample on.
| `-p <index>` | 0 | Specify the index of the OpenCL platform to execute the sample on.
