# spirvkernelfromfile

## Sample Purpose

This sample is very similar to the previous sample that reads an OpenCL kernel from a file, except in this sample the kernel is passed as [SPIR-V](https://www.khronos.org/spir/) intermediate representation rather than as OpenCL C source.

SPIR-V is an intermediate representation, so it is an excellent target for other source languages targeting execution on an OpenCL device.
There are also compiler front-ends, also referred to as SPIR-V "generators", that accept standard OpenCL C as input and emit SPIR-V as output.
Some of these compiler front-ends support additional language features beyond standard OpenCL C, for example, the community-driven [C++ for OpenCL in Clang](https://clang.llvm.org/docs/UsersManual.html#c-for-opencl) initiative enables many C++ features in kernels.
Many [SYCL](https://www.khronos.org/sycl/) implementations also support compilation to SPIR-V.

There are currently two mechanisms an OpenCL application can use to pass SPIR-V intermediate representation to an OpenCL implementation:

1. SPIR-V is a core feature of OpenCL 2.1, supported via the [`clCreateProgramWithIL`](https://www.khronos.org/registry/OpenCL/specs/2.2/html/OpenCL_API.html#clCreateProgramWithIL) API.
Using this mechanism requires enabling OpenCL 2.1 in the OpenCL headers, linking with an OpenCL 2.1 or newer ICD loader, deploying to a system with an OpenCL 2.1 or newer ICD loader, and executing on an OpenCL platform and device that supports OpenCL 2.1.
2. SPIR-V may also be supported via the [`cl_khr_il_program`](https://www.khronos.org/registry/OpenCL/specs/2.2/html/OpenCL_Ext.html#cl_khr_il_program) extension, which adds the `clCreateProgramWithILKHR` API.
Using this mechanism requires extension headers and querying for the extension API, which should only succeed when the platform and device supports the `cl_khr_il_program` extension.

These mechanisms are not mutually exclusive; an implementation may support one or both mechanisms, and an application may check for one or both mechanisms.
By default, this sample tries to use the core OpenCL 2.1 API first, and tries to use the `cl_khr_il_program` extension API as a fallback.
If desired, the sample can be modified to use the extension API exclusively, which adds deployment flexibility since it does not require an OpenCL 2.1 or newer ICD loader, but does require the OpenCL implementation to support `cl_khr_il_program`.

Notes:

1. Like the previous sample that read an OpenCL C kernel from a file, this sample builds and runs the SPIR-V kernel in the file, but does not check for specific results.
2. To run successfully, the SPIR-V kernel should accept a single global memory kernel argument, and should write fewer than `gwx` 32-bit values to the kernel argument buffer.
3. SPIR-V files are compiled for a specific pointer size, which much match the pointer size for the OpenCL device.
The pointer size for the OpenCL device is frequently 32-bits for 32-bit a application executable and 64-bits for a 64-bit application executable, but not always.
4. The `install` target (e.g. `make install` on Linux, or right-click on `INSTALL` and build in Visual Studio, for example) will automatically copy the SPIR-V kernel files to the install directory with the application directory.

## How to Generate SPIR-V Files:

There are many ways to generate SPIR-V files.
This sample generated its SPIR-V files using the [Clang](https://clang.llvm.org) frontend compiler to compile from OpenCL C to LLVM IR, followed by the Khronos [SPIR-V LLVM Translator](https://github.com/KhronosGroup/SPIRV-LLVM-Translator) to translate LLVM IR to SPIR-V.
At the time of this writing, here were the command lines used to perform these steps:

For the 32-bit SPIR-V file:

```sh
$ clang -c -cl-std=CL1.2 -target spir -emit-llvm -Xclang -finclude-default-header -O3 sample_kernel.cl -o sample_kernel32.ll
$ llvm-spirv sample_kernel32.ll -o sample_kernel32.spv
```

And, for the 64-bit SPIR-V file:

```sh
$ clang -c -cl-std=CL1.2 -target spir64 -emit-llvm -Xclang -finclude-default-header -O3 sample_kernel.cl -o sample_kernel64.ll
$ llvm-spirv sample_kernel64.ll -o sample_kernel64.spv
```

The generated SPIR-V files can be disassembled using `spirv-dis`, which is part of the [SPIR-V Tools](https://github.com/KhronosGroup/SPIRV-Tools).

## Key APIs and Concepts

This sample demonstrates the `clCreateProgramWithIL` API.
This sample is also the first sample that demonstrates how to use an OpenCL extension and the `clGetExtensionFunctionAddressForPlatform` API.
Like the previous sample that read the OpenCL C source from a file, this sample supports optional program "build options", and queries and prints the program "build log" after compilation.

```c
clCreateProgramWithIL
clGetExtensionFunctionAddressForPlatform
```

## Things to Try

Here are some suggested ways to modify this sample to learn more:

1. Download and build the SPIR-V tools and use `spirv-dis` to examine the contents of the SPIR-V files.
What are the differences between the 32-bit and 64-bit SPIR-V files?
2. Download and build clang and the SPIR-V LLVM Translator.
Can you generate your own SPIR-V files from the sample kernel?
3. Add a template function to your kernel and compile it using "C++ for OpenCL in Clang".
Hint: At the time of this writing, you should pass `-cl-std=clc++` to Clang to compile your file for C++ for OpenCL in Clang. vs. standard OpenCL C.
See example [here](https://godbolt.org/z/hEogpZ).

## Command Line Options

| Option | Default Value | Description |
|:--|:-:|:--|
| `-d <index>` | 0 | Specify the index of the OpenCL device in the platform to execute on the sample on.
| `-p <index>` | 0 | Specify the index of the OpenCL platform to execute the sample on.
| `--file <string>` | See Description | Specify the name of the file with the SPIR-V kernel intermediate representation.  The default value for a 32-bit executable is `sample_kernel32.spv` and the default value for a 64-bit executable is `sample_kernel64.spv`.
| `--name <string>` | `Test` | Specify the name of the OpenCL kernel in the source file.
| `--options <string>` | None | Specify optional program build options.
| `--gwx <number>` | 512 | Specify the global work size to execute.
