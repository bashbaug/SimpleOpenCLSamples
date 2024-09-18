# Floating-point Atomic Adds

## Sample Purpose

TODO

Inspired by: https://pipinspace.github.io/blog/atomic-float-addition-in-opencl.html

## Key APIs and Concepts

TODO

## Command Line Options

| Option | Default Value | Description |
|:--|:-:|:--|
| `-d <index>` | 0 | Specify the index of the OpenCL device in the platform to execute on the sample on.
| `-p <index>` | 0 | Specify the index of the OpenCL platform to execute the sample on.
| `-i <number>` | 16 | Specify the number of iterations to execute.
| `--gwx <number>` | 1024 | Specify the global work size to execute, which is also the number of floating-point atomics to perform.
| `-e` | N/A | Unconditionally use the emulated floating-point atomic add.
