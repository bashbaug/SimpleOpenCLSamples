# N-Body Simulation with OpenGL

## Sample Purpose

This sample uses OpenCL to compute an [N-body simulation](https://en.wikipedia.org/wiki/N-body_simulation), which is then rendered with OpenGL.

This sample currently does not share the OpenCL buffer with OpenGL and will unconditionally copy from OpenCL to OpenGL on the host.
It is most useful as a reference for the similar Vulkan sample.

## Key APIs and Concepts

This example shows how to copy from an OpenCL buffer to OpenGL.

## Command Line Options

| Option | Default Value | Description |
|:--|:-:|:--|
| `-d <index>` | 0 | Specify the index of the OpenCL device in the platform to execute on the sample on.
| `-p <index>` | 0 | Specify the index of the OpenCL platform to execute the sample on.
| `-n` | 1024 | Specify the number of bodies to simulate.
| `-g` | 0| Specify the local work size.  If the local works size is zero a `NULL` local work size is used.
| `-w` | 1024 | Specify the render width in pixels.
| `-h` | 1024 | Specify the render height in pixels.

## Controls While Running

| Control | Description |
|:--|:--|
| `Escape` | Exits from the sample.
| `Space` | Toggle animation (default: `false`).
| `S` | Single-step the simulation.
| `R` | Re-initialize the simulation.
| `V` | Toggle vsync (default: `true`). Disabling vsync may increase framerate but may cause [screen tearing](https://en.wikipedia.org/wiki/Screen_tearing).
