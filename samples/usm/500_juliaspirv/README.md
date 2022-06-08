# Julia Set with SPIR-V and USM

## Sample Purpose

This is a more modern version of the Julia Set sample.
Unlike the previous version that built the Julia Set program from OpenCL C source and operated on an OpenCL buffer, this version builds the Julia Set program from SPIR-V intermediate representation and writes to host USM.
Because host USM is directly accessible by the host, no mapping or un-mapping is required.

## Key APIs and Concepts

```c
clCreateProgramWithIL
clHostMemAllocINTEL
```

## Command Line Options

| Option | Default Value | Description |
|:--|:-:|:--|
| `-d <index>` | 0 | Specify the index of the OpenCL device in the platform to execute on the sample on.
| `-p <index>` | 0 | Specify the index of the OpenCL platform to execute the sample on.
| `--options <string>` | None | Specify optional program build options.
| `-i <number>` | 16 | Specify the number of iterations to execute.
| `--gwx <number>` | 512 | Specify the global work size to execute, in the X direction.  This also determines the width of the generated image.
| `--gwy <number>` | 512 | Specify the global work size to execute, in the Y direction.  This also determines the height of the generated image.
| `--lwx <number>` | 0 | Specify the local work size in the X direction.  If either local works size dimension is zero a `NULL` local work size is used.
| `--lwy <number>` | 0 | Specify the local work size in the Y direction.  If either local works size dimension is zero a `NULL` local work size is used.
