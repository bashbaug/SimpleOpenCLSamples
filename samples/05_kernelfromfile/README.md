# kernelfromfile

## Sample Purpose

In all of the samples so far the OpenCL C kernel source has been defined in the host code as a C++ raw string literal.
This is convenient for samples because the kernel source gets embedded into the compiled application executable.
In some cases, though, it is convenient to keep the kernel source separate from the application instead.
For example, if the kernel source is in a separate file, the kernel may be modified without rebuilding the application or requiring host application source code.
This sample demonstrates how to read kernel code from a separate file.

Notes:

1. This sample builds and runs the kernel in the file, but does not check for specific results.
2. To run successfully, the kernel should accept a single global memory kernel argument, and should write fewer than `gwx` 32-bit values to the kernel argument buffer.
3. The `install` target (`make install` on Linux, or right-click on `INSTALL` and build in Visual Studio, for example) will automatically copy the kernel file to the install directory with the application directory.

## Key APIs and Concepts

This sample also demonstrates additional API features that are often useful when building OpenCL programs:

* This sample supports optional program "build options".
* This sample queries and prints the program "build log" after compilation.
The program build log contains compiler diagnostics, such as build errors or warnings.

```c
clBuildProgram with build options
clGetProgramBuildInfo with CL_PROGRAM_BUILD_LOG
```

## Things to Try

Here are some suggested ways to modify this sample to learn more:

1. Change the kernel source file to write a different value to the result buffer.
If you make a mistake and the kernel is syntactically incorrect, what gets printed in the program build log?
2. Pass in a program build option and observe how it modifies how the kernel is compiled or the behavior of the kernel.
The easiest way to do this is to define a preprocessor symbol with `-D`.
Or, compile for a specific OpenCL C version using `-cl-std`.
3. Print a value from the kernel using `printf`.
Do you see the value printed that you expect?
4. Modify the host code to print the first few values in the result buffer, or to validate that the results are what you expect.
Can you read the expected result buffer from a file?
5. Modify the host code to pass an additional buffer to the OpenCL kernel.
Can you initialize the contents of the buffer from a file?

## Command Line Options

| Option | Default Value | Description |
|:--|:-:|:--|
| `-d <index>` | 0 | Specify the index of the OpenCL device in the platform to execute on the sample on.
| `-p <index>` | 0 | Specify the index of the OpenCL platform to execute the sample on.
| `--file <string>` | `sample_kernel.cl` | Specify the name of the file with the OpenCL kernel source.
| `--name <string>` | `Test` | Specify the name of the OpenCL kernel in the source file.
| `--options <string>` | None | Specify optional program build options.
| `--gwx <number>` | 512 | Specify the global work size to execute.
