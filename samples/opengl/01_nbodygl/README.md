# N-Body Simulation with OpenGL

## Sample Purpose

Write me!

## Key APIs and Concepts

This example shows how to share an OpenGL buffer with OpenCL.

```c
clGetGLContextInfoKHR
clCreateFromGLBuffer
clEnqueueAcquireGLObjects
clEnqueueReleaseGLObjects
```

## Command Line Options

Note: Many of these command line arguments are identical to the earlier Julia set sample.

| Option | Default Value | Description |
|:--|:-:|:--|
| `-d <index>` | 0 | Specify the index of the OpenCL device in the platform to execute on the sample on.
| `-p <index>` | 0 | Specify the index of the OpenCL platform to execute the sample on.
| `--hostcopy` | n/a | Do not use the `cl_khr_gl_sharing` extension and unconditionally copy on the host.
| `--hostsync` | n/a | Do not use the `cl_khr_gl_event` extension and exclusively synchronize on the host.
| `-n` | 1024 | Specify the number of bodies to simulate.
| `-g` | 0| Specify the local work size.  If the local works size is zero a `NULL` local work size is used.
| `-w` | 1600 | Specify the render width in pixels.
| `-h` | 900 | Specify the render height in pixels.

## Controls While Running

| Control | Description |
|:--|:--|
| `Escape` | Exits from the sample.
| `Space` | Toggle animation (default: `false`).
| `V` | Toggle vsync (default: `true`). Disabling vsync may increase framerate but may cause [screen tearing](https://en.wikipedia.org/wiki/Screen_tearing).
| `R` | Re-initialize the simulation.
