# Mandelbrot Set

## Sample Purpose

This is a port of the [ISPC Mandelbrot](https://github.com/ispc/ispc/tree/master/examples/mandelbrot) sample.
It uses an OpenCL kernel to compute a [Mandelbrot set](https://en.wikipedia.org/wiki/Mandelbrot_set) image, which is then written to a BMP file.
Each OpenCL work item computes one element of the set.

This assuredly is not the fastest Mandelbrot kernel on any OpenCL implementation, but it should perform reasonably well - much better than an equivalent serial implementation!

![Mandelbrot Image](mandelbrot.png)

As with prior samples, the source code for the OpenCL kernel is embedded into the host code as a raw string, and by default, this sample will run in the first enumerated OpenCL device on the first enumerated OpenCL platform.
To run on a different OpenCL device or platform, please use the provided command line options.

## Key APIs and Concepts

This example shows how to create an OpenCL program from a source string and enqueue an ND range for the kernel into an OpenCL command queue.

```c
clCreateProgramWithSource
clBuildProgram
clCreateKernel
clSetKernelArg
clEnqueueNDRangeKernel
```

## Command Line Options

| Option | Default Value | Description |
|:--|:-:|:--|
| `-d <index>` | 0 | Specify the index of the OpenCL device in the platform to execute on the sample on.
| `-p <index>` | 0 | Specify the index of the OpenCL platform to execute the sample on.
