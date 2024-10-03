# Floating-point Atomic Adds

## Sample Purpose

This is an advanced sample that demonstrates how to do atomic floating-point addition in a kernel.
The most standard way to perform atomic floating-point addition uses the [cl_ext_float_atomics](https://registry.khronos.org/OpenCL/extensions/ext/cl_ext_float_atomics.html) extension.
This extension adds device queries and built-in functions to optionally support floating-point atomic add, min, max, load, and store on 16-bit, 32-bit, and 64-bit floating-point types.
When the `cl_ext_float_atomics` extension is supported, and 32-bit floating point atomic adds are supported, this sample will use the built-in functions added by this extension.

This sample also includes fallback implementations when the `cl_ext_float_atomics` extension is not supported:

* For NVIDIA GPUs, this sample includes a fallback that does the floating-point atomic add using inline PTX assembly language.
* For AMD GPUs, this sample includes a fallback that calls a compiler intrinsic to do the floating-point atomic add.
* For other devices, this sample includes two fallback implementations:
    * The first emulates the floating-point atomic add using 32-bit `atomic_xchg` functions.
      This fallback implementation cannot reliably return the "old" value that was in memory before performing the atomic add, so it is unsuitable for all usages, but it does work for some important uses-cases, such as reductions.
    * The second emulates the floating-point atomic add using 32-bit `atomic_cmpxchg` functions.
      This is a slower emulation, but it is able to reliably return the "old" value that was in memory before performing the atomic add.

This sample was inspired by the blog post: https://pipinspace.github.io/blog/atomic-float-addition-in-opencl.html

## Key APIs and Concepts

```
CL_DEVICE_SINGLE_FP_ATOMIC_CAPABILITIES_EXT
__opencl_c_ext_fp32_global_atomic_add
atomic_fetch_add_explicit
atomic_xchg
atomic_cmpxchg
```

## Command Line Options

| Option | Default Value | Description |
|:--|:-:|:--|
| `-d <index>` | 0 | Specify the index of the OpenCL device in the platform to execute on the sample on.
| `-p <index>` | 0 | Specify the index of the OpenCL platform to execute the sample on.
| `-i <number>` | 16 | Specify the number of iterations to execute.
| `--gwx <number>` | 16384 | Specify the global work size, which is also the number of floating-point atomics to perform.
| `-e` | N/A | Unconditionally use the emulated floating-point atomic add.
| `-s` | N/A | Unconditionally use the slower and safer emulated floating-point atomic add.
| `-e` | N/A | Check intermediate results for correctness, unsupported for the faster emulated atomics, requires adding a positive value.
