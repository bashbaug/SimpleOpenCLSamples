# Simple OpenCL<sup>TM</sup> Samples

[![build](https://github.com/bashbaug/SimpleOpenCLSamples/workflows/build/badge.svg)](https://github.com/bashbaug/SimpleOpenCLSamples/actions?query=workflow%3Abuild)

This repo contains simple OpenCL samples that demonstrate how to build
OpenCL applications using only the Khronos-provided headers and libs.
All samples have been tested on Windows and Linux.


## Code Structure

```
README.md               This file
LICENSE                 License information
CMakeLists.txt          Top-level CMakefile
external/               External Projects (headers and libs)
include/                Include Files (OpenCL C++ bindings)
samples/                Samples
tutorials/              Tutorials
```

## How to Build the Samples

The samples require the following external dependencies:

OpenCL Headers:

    git clone https://github.com/KhronosGroup/OpenCL-Headers external/OpenCL-Headers

OpenCL ICD Loader:

    git clone https://github.com/KhronosGroup/opencl-icd-loader external/opencl-icd-loader

After satisfying the external dependencies create build files using CMake.  For example:

    mkdir build && cd build
    cmake ..

Then, build with the generated build files.

## How to Run the Samples

To run the samples, you will need to obtain and install an ICD loader and an 
OpenCL implementation (ICD) that supports the `cl_khr_icd` extension.

The ICD loader is likely provided by your operating system or an OpenCL
implementation.  If desired, you may use the ICD loader that is built along 
with these OpenCL samples.  The OpenCL implementation will likely be provided 
by your OpenCL device vendor.  There are several open source OpenCL
implementations as well.

## Further Reading

* [Environment Setup for Ubuntu 18.04](docs/env/ubuntu/18.04.md)
* [OpenCLPapers](https://github.com/bashbaug/OpenCLPapers)
* [OpenCL Specs](https://www.khronos.org/registry/OpenCL/specs/)
* [OpenCL Return Codes](https://streamhpc.com/blog/2013-04-28/opencl-error-codes/)

## License

These samples are licensed under the [MIT License](LICENSE).

Notes:
* The OpenCL C++ bindings are built from the
[Khronos OpenCL-CLHPP Repo](https://github.com/KhronosGroup/OpenCL-CLHPP),
and is licensed under the
[Khronos(tm) License](https://github.com/KhronosGroup/OpenCL-CLHPP/blob/master/LICENSE.txt).

---
OpenCL and the OpenCL logo are trademarks of Apple Inc. used by permission by Khronos.

\* Other names and brands may be claimed as the property of others.