# ndrangekernelfromfile

## Sample Purpose

This is a modified version of a previous sample that loads a kernel from a file.
The most important difference between the previous sample and this sample is that this sample specifies a "local work size" when enqueueing the kernel for execution.

When enqueueing an OpenCL kernel for execution, we can specify only a "global work size", describing the range of work-items that will execute the kernel, or we can specify both a "global work size" and a "local work size".
If we specify a "local work size" then the local work size this will describe how work-items are grouped into work-groups for execution.

Work-items in a work-group are special: they're guaranteed to execute concurrently, they can exchange results via "local memory", and they can synchronize execution using barriers.
The default kernel for this sample does not take advantage of any of these features, however future samples will!

Many devices support only "uniform" work-groups, where the local work size evenly divides the global work size.
To check if "non-uniform" work-groups are supported, query the OpenCL version supported by the device: only "uniform" work-groups are supported for OpenCL 1.x devices, support for "non-uniform" work-groups are required for OpenCL 2.x devices, and support for non-uniform work-groups is optional for OpenCL 3.0 devices and can be queried using `CL_DEVICE_NON_UNIFORM_WORK_GROUP_SUPPORT`.
If non-uniform work-groups are supported, they will need to be enabled by compiling with the `-cl-std=CL2.0` or `-cl-std=CL3.0` compile, link, or build options.

Additional notes:

1. This sample builds and runs the kernel in the file, but does not check for specific results.
2. To run successfully, the kernel should accept a single global memory kernel argument, and should write fewer than `gwx` 32-bit values to the kernel argument buffer.
3. The `install` target (`make install` on Linux, or right-click on `INSTALL` and build in Visual Studio, for example) will automatically copy the kernel file to the install directory with the application directory.
4. If the local work size `lwx` is set to zero this sample will use a `NULL` local work size, just like the previous sample.

## Key APIs and Concepts

This sample demonstrates how to specify groupings of work-items when enqueueing an OpenCL kernel for execution by passing a non-`NULL` local work size to `clEnqueueNDRangeKernel`.

This sample also supports compiling and linking the program object separately using `clCompileProgram` and `clLinkProgram` rather than performing both steps using `clBuildProgram`.

```
clEnqueueNDRangeKernel with a non-NULL local work size
clCompileProgram
clLinkProgram
```

## Things to Try

Here are some suggested ways to modify this sample to learn more:

1. How does performance change with different local work sizes?
What happens with a local work size equal to `1`?
Does performance get better or worse with larger local work sizes?
Can you find a local work size that performs better than the `NULL` local work size?
2. Does your device support non-uniform work-groups?
If not, what happens if you pass a local work size that does not evenly divide the global work size?
If so, can you pass the right build options to enable non-uniform work-group support?
3. Do you see any difference compiling and linking the program separately vs. building it in one step?
4. Can you load and compile a program from a different file and link with it?

## Command Line Options

| Option | Default Value | Description |
|:--|:-:|:--|
| `-d <index>` | 0 | Specify the index of the OpenCL device in the platform to execute on the sample on.
| `-p <index>` | 0 | Specify the index of the OpenCL platform to execute the sample on.
| `--file <string>` | `ndrange_sample_kernel.cl` | Specify the name of the file with the OpenCL kernel source.
| `--name <string>` | `Test` | Specify the name of the OpenCL kernel in the source file.
| `--options <string>` | None | Specify optional program build options.
| `--gwx <number>` | 512 | Specify the global work size to execute, in the X-dimension (first dimension, fastest moving)
| `--gwy <number>` | 1 | Specify the global work size to execute, in the Y-dimension (second dimension, slowest moving)
| `--lwx <number>` | 32 | Specify the local work size grouping, in the X-dimension (first dimension, fastest moving).
| `--lwy <number>` | 1 | Specify the local work size grouping, in the Y-dimension (second dimension, slowest moving).
| `-a` | `false` | Show advanced options.
| `-c` | `false` | (advanced) Use `clCompileProgram` and `clLinkProgram` instead of `clBuildProgram`.
| `--compileoptions <string>` | None | (advanced) Specify optional program compile options.
| `--linkoptions <string>` | None | (advanced) Specify optional program link options.
