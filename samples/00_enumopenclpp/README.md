# enumopenclpp

## Sample Purpose

This is another very simple sample that demonstrates how to enumerate the OpenCL platforms that are installed on a machine, and the OpenCL devices that these platforms expose.

This sample uses the OpenCL C++ API bindings instead of the OpenCL C APIs.
By using the OpenCL C++ API bindings, this sample can be written using a little more than 100 lines of code, versus a little more than 300 lines of code for the version using the OpenCL C APIs!

This sample should produce the same output as the sample that uses the OpenCL C APIs, making it another good first sample to run to verify that OpenCL is correctly installed on your machine, and that your build environment is correctly setup.

## Key APIs and Concepts

The most important concept to understand from this sample are how the OpenCL C++ API bindings can be used to write code that is more concise and (at least, in this author's opinion) easier to author and understand, compared to OpenCL code written using the OpenCL C APIs.

This isn't the only way to write code with a higher level model that eventually generates OpenCL API calls, but it is one that is well-supported and documented.
Most of the samples in this repo will use the OpenCL C++ API bindings.

```c
clGetPlatformIDs
clGetDeviceIDs
clGetPlatformInfo
clGetDeviceInfo
```
