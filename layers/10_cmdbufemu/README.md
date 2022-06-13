# Command Buffer Emulation

## Layer Purpose

This is a layer that demonstrates how to emulate functionality - in this case, [cl_khr_command_buffer](https://www.khronos.org/registry/OpenCL/specs/3.0-unified/html/OpenCL_Ext.html#cl_khr_command_buffer) - using a layer.
It works by intercepting calls to `clGetExtensionFunctionAddressForPlatform` to query function pointers for the `cl_khr_command_buffer` extension APIs.
If a query succeeds by default then the layer does nothing and simply returned the queried function pointer as-is.
If the query is unsuccessful however, then the layer returns its own function pointer, which will record the contents of the command buffer for later playback.

This command buffer emulation layer currently implements v0.9.0 of the `cl_khr_command_buffer` extension.
The functionality in this emulation layer is sufficient to run the command buffer samples in this repository.

Please note that the emulated command buffers are intended to be functional, but unlike a native implementation of `cl_khr_command_buffer`, they may not provide any performance benefit over similar code without using command buffers.

## Key APIs and Concepts

The most important concepts to understand from this sample are how to intercept `clGetExtensionFunctionAddressForPlatform` to return emulated functions for an extension.

```c
clGetExtensionFunctionAddressForPlatform
clInitLayer
```

## Known Limitations

This section describes some of the limitations of the emulated `cl_khr_command_buffer` functionality:

* The event associated with `clEnqueueCommandBufferKHR` will not have the proper event command type `CL_COMMAND_COMMAND_BUFFER_KHR`.
* Event profiling for `clEnqueueCommandBufferKHR` will not be correct for the `QUEUED`, `SUBMITTED`, and `START` times.
