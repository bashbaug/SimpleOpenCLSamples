# Simple OpenCL<sup>TM</sup> Samples

[![build](https://github.com/bashbaug/SimpleOpenCLSamples/workflows/build/badge.svg?branch=main)](https://github.com/bashbaug/SimpleOpenCLSamples/actions?query=workflow%3Abuild+branch%3Amain)

This repo contains simple OpenCL samples that demonstrate how to build
OpenCL applications using only the Khronos-provided headers and libs.
All samples have been tested on Windows and Linux.

Most of the samples are written in C and C++ using the OpenCL C++ bindings.
A few of the samples have been ported to Python using [PyOpenCL](https://pypi.org/project/pyopencl/).


## Code Structure

```
README.md               This file
LICENSE                 License information
CMakeLists.txt          Top-level CMakefile
external/               External Projects (headers and libs)
include/                Include Files (OpenCL C++ bindings)
layers/                 Sample Layers
samples/                Sample Applications
tutorials/              Tutorials
```

## How to Build the Samples

The samples require the following external dependencies:

OpenCL Headers:

    git clone https://github.com/KhronosGroup/OpenCL-Headers external/OpenCL-Headers

OpenCL ICD Loader:

    git clone https://github.com/KhronosGroup/opencl-icd-loader external/opencl-icd-loader

Many samples that use extensions additionally require the OpenCL Extension Loader:

    git clone https://github.com/bashbaug/opencl-extension-loader external/opencl-extension-loader

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

## A Note About Error Checking

For brevity, most samples do not include error checking. This means that a
sample may crash or incorrectly report success if an OpenCL error occurs. By
defining the CMake variable `SAMPLES_ENABLE_EXCEPTIONS` many samples can instead
throw an exception if an OpenCL error occurs.

Tools like the [OpenCL Intercept Layer](https://github.com/intel/opencl-intercept-layer)
can also be useful to detect when an OpenCL error occurs and to identify the
cause of the error.

## License

These samples are licensed under the [MIT License](LICENSE).

Notes:
* The OpenCL C++ bindings are built from the
[Khronos OpenCL-CLHPP Repo](https://github.com/KhronosGroup/OpenCL-CLHPP),
and is licensed under the
[Khronos(tm) License](https://github.com/KhronosGroup/OpenCL-CLHPP/blob/master/LICENSE.txt).
* The samples use [popl](https://github.com/badaix/popl) for its options
parsing, which is licensed under the MIT License.

---
OpenCL and the OpenCL logo are trademarks of Apple Inc. used by permission by Khronos.

\* Other names and brands may be claimed as the property of others.