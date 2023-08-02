# Sobel Filter with OpenGL

## Sample Purpose

This is an enhanced version of the earlier Julia set sample.
Unlike the previous sample that simply displayed the generated Julia set image, this sample executes a [Sobel edge detection filter](https://en.wikipedia.org/wiki/Sobel_operator) on the generated Julia set before displaying it.

This sample can share the OpenGL texture with OpenCL when supported.
In order to share the OpenGL texture with OpenCL, the OpenCL device must support the [cl_khr_gl_sharing](https://www.khronos.org/registry/OpenCL/specs/3.0-unified/html/OpenCL_Ext.html#cl_khr_gl_sharing) extension, and the OpenCL device must support sharing with the OpenGL context.
If sharing is not supported then the application will still run, but the output image will be copied from OpenCL to OpenGL on the host.

Additionally, this sample can use implicit synchronization between OpenGL and OpenCL when supported.
Implicit synchronization requires support for the [cl_khr_gl_event](https://www.khronos.org/registry/OpenCL/specs/3.0-unified/html/OpenCL_Ext.html#cl_khr_gl_event) extension.
If implicit synchronization is not supported then the application will still run, but synchronization will be done manually.

## Key APIs and Concepts

This example shows how create a sampler, pass it to a kernel, and use it to read data from an image.

```c
clCreateSampler
```

## Things to Try

Here are some suggested ways to modify this sample to learn more:

1. Instead of creating a sampler and passing it to the Sobel kernel, instead define the sampler in the kernel itself.
Does this change the performance of the kernel?
2. Instead of using a sampler to access the image, use the sampler-less `read_imagef` functions instead.
Be careful doing this because the sampler-less functions to not support reading out-of-bounds (the sampler-less address mode is implicitly `CL_ADDRESS_NONE`).
Does this change the performance of the kernel?

## Command Line Options

Note: Many of these command line arguments are identical to the earlier Julia set sample.

| Option | Default Value | Description |
|:--|:-:|:--|
| `-d <index>` | 0 | Specify the index of the OpenCL device in the platform to execute on the sample on.
| `-p <index>` | 0 | Specify the index of the OpenCL platform to execute the sample on.
| `--hostcopy` | n/a | Do not use the `cl_khr_gl_sharing` extension and unconditionally copy on the host.
| `--hostsync` | n/a | Do not use the `cl_khr_gl_event` extension and exclusively synchronize on the host.
| `--gwx <number>` | 512 | Specify the global work size to execute, in the X direction.  This also determines the width of the generated image.
| `--gwy <number>` | 512 | Specify the global work size to execute, in the Y direction.  This also determines the height of the generated image.
| `--lwx <number>` | 0 | Specify the local work size in the X direction.  If either local works size dimension is zero a `NULL` local work size is used.
| `--lwy <number>` | 0 | Specify the local work size in the Y direction.  If either local works size dimension is zero a `NULL` local work size is used.
| `--paused` | n/a | Start with the animation paused.

## Controls While Running

| Control | Description |
|:--|:--|
| `Escape` | Exits from the sample.
| `Space` | Toggle animation.
| `V` | Toggle vsync (default: `true`). Disabling vsync may increase framerate but may cause [screen tearing](https://en.wikipedia.org/wiki/Screen_tearing).
| `A` | Increase the real part of the complex constant `C`.
| `Z` | Decrease the real part of the complex constant `C`.
| `S` | Increase the imaginary part of the complex constant `C`.
| `X` | Decrease the imaginary part of the complex constant `C`.
