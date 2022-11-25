# Semaphore Emulation

## Layer Purpose

This is a layer that demonstrates how to emulate functionality - in this case, the [cl_khr_semaphore](https://www.khronos.org/registry/OpenCL/specs/3.0-unified/html/OpenCL_Ext.html#cl_khr_semaphore) extension - using a layer.
It works by intercepting calls to `clGetExtensionFunctionAddressForPlatform` to query function pointers for the `cl_khr_semaphore` extension APIs.
If a query succeeds by default then the layer does nothing and simply returns the queried function pointer as-is.
If the query is unsuccessful however, then the layer returns its own function pointer, which will record the contents of the command buffer for later playback.

This command buffer emulation layer currently implements v0.9.0 of the `cl_khr_semaphore` extension.
The functionality in this emulation layer is sufficient to run the semaphore samples in this repository.

Please note that the emulated semaphores are intended to be functional, but unlike a native implementation, they may not provide any performance benefit over similar code without using semaphores.

## Layer Requirement

This layer calls `clEnqueueMarkerWithWaitList` and therefore requires OpenCL 1.2.

## Key APIs and Concepts

The most important concepts to understand from this sample are how to intercept `clGetExtensionFunctionAddressForPlatform` to return emulated functions for an extension.

```c
clGetExtensionFunctionAddressForPlatform
clInitLayer
```

## Known Limitations

This section describes some of the limitations of the emulated `cl_khr_semaphore` functionality:

* The layer does not support waiting on a semaphore (blocked by a user event) before signaling the semaphore.
* Many error conditions are not properly checked for and returned.
