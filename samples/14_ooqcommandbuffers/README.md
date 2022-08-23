# ooqcommandbuffers

## Sample Purpose

This is another modified version of the commandbuffers sample that demonstrates how to use the OpenCL extension [cl_khr_command_buffer](https://www.khronos.org/registry/OpenCL/specs/3.0-unified/html/OpenCL_Ext.html#cl_khr_command_buffer) to create and submit work to an out-of-order queue.
This sample works by submitting two relatively slow kernels to initialize two buffers, then a faster data parallel kernel to add the two buffers together.
There are no dependencies between the two relatively slow kernels so they may run in parallel, but they must complete before the data parallel kernel can execute.

This sample requires the OpenCL Extension Loader to get the extension APIs for command buffers.

## Command Line Options

| Option | Default Value | Description |
|:--|:-:|:--|
| `-d <index>` | 0 | Specify the index of the OpenCL device in the platform to execute on the sample on.
| `-p <index>` | 0 | Specify the index of the OpenCL platform to execute the sample on.
