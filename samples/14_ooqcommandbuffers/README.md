# commandbuffers

## Sample Purpose

This is a modified version of the commandbuffers sample that demonstrates how to use the OpenCL extension [cl_khr_command_buffer](https://www.khronos.org/registry/OpenCL/specs/3.0-unified/html/OpenCL_Ext.html#cl_khr_command_buffer) using proof-of-concept C++ bindings for command buffers.

This sample requires the OpenCL Extension Loader to get the extension APIs for command buffers.

## Command Line Options

| Option | Default Value | Description |
|:--|:-:|:--|
| `-d <index>` | 0 | Specify the index of the OpenCL device in the platform to execute on the sample on.
| `-p <index>` | 0 | Specify the index of the OpenCL platform to execute the sample on.
