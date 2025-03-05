# semaphores

## Sample Purpose

This sample demonstrates how to use the OpenCL extension [cl_khr_semaphore](https://www.khronos.org/registry/OpenCL/specs/3.0-unified/html/OpenCL_Ext.html#cl_khr_semaphore) to enforce dependencies between command queues.
As of this writing, `cl_khr_semaphore` is a provisional extension.
This sample uses the functionality described in v0.9.0 of the extension.

This is an optional extension and some devices may not support `cl_khr_semaphore`, but the sample may still run using the [cl_khr_semaphore emulation layer](../../layers/11_semaemu).

This sample requires the OpenCL Extension Loader to get the extension APIs for semaphores.

## Key APIs and Concepts

This sample demonstrates how to query the semaphore properties supported by a device, and the properties of a semaphore.

This sample also demonstrates how to create, signal, and wait on a semaphore.

```c
clCreateSemaphoreWithPropertiesKHR
clEnqueueSignalSemaphoresKHR
clEnqueueWaitSemaphoresKHR
clGetSemaphoreInfoKHR
```

## Command Line Options

| Option | Default Value | Description |
|:--|:-:|:--|
| `-d <index>` | 0 | Specify the index of the OpenCL device in the platform to execute on the sample on.
| `-p <index>` | 0 | Specify the index of the OpenCL platform to execute the sample on.
| `--gwx <number>` | 512 | Specify the global work size to execute.
