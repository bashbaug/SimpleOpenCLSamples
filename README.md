# Simple OpenCL<sup>TM</sup> Samples

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
```

## Environment Setup

Building an OpenCL application requires the OpenCL API and ICD Loader headers. Running an OpenCL application requires
the associated ICD and runtime objects.

* [Generic Linux (build from source)](docs/env/linux.md)
* [Ubuntu 18.04](docs/env/ubuntu/18.04.md)

## Building the Samples

Follow the environment setup instructions for your environment of choice, then create the build files using CMake:

```
$ mkdir build && cd build
$ cmake ..
$ make -j
```

## Running the Samples

Follow the environment setup and build instructions, then run the sample of interest from the build directory. For example:

```
$ samples/00_enumopenclpp/enumopenclpp
<platform information shown here>
Done.
$
```

## Further Reading

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