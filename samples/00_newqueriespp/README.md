# newqueriespp

## Sample Purpose

This is another sample that demonstrates the new platform and device queries that were added in OpenCL 3.0.

This sample uses the OpenCL C++ API bindings instead of the OpenCL C APIs.
As before, by using the OpenCL C++ API bindings this sample is much shorter than the equivalent sample using the OpenCL C APIs!

This sample will only execute the new queries for platforms or devices that support OpenCL 3.0, and will skip any platforms or devices that do not support OpenCL 3.0.

## Key APIs and Concepts

This sample demonstrates how to use the OpenCL C++ API bindings to perform the new platform and device queries that were added in OpenCL 3.0.

```c
clGetPlatformInfo
clGetDeviceInfo
```
