# newqueries

## Sample Purpose

This is a sample that demonstrates the new platform and device queries that were added in OpenCL 3.0.
It builds on the previous sample that simply enumerates the OpenCL platforms that are installed on a machine, and the OpenCL devices that these platforms expose.

This is one of the few samples that uses the OpenCL C APIs, as described in the OpenCL specification.
Most of other samples use the OpenCL C++ API bindings, since they make it a lot easier to write and understand OpenCL code!

This sample will only execute the new queries for platforms or devices that support OpenCL 3.0, and will skip any platforms or devices that do not support OpenCL 3.0.

## Key APIs and Concepts

The new OpenCL 3.0 queries make it much easier to identify platform and device capabilities such as the supported OpenCL version, OpenCL C versions, intermediate language versions, and extensions.
In many cases the same information can be queried using queries from earlier versions of OpenCL, but extracting the same information frequently required parsing string queries and hence was complicated and error-prone.

```c
clGetPlatformInfo
clGetDeviceInfo
```
