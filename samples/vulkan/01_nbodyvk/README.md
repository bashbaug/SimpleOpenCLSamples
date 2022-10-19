# N-Body Simulation with OpenGL

## Sample Purpose

Write me!

## Key APIs and Concepts

This example shows how to share an Vulkan buffer with OpenCL.

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
| `S` | Single-step the simulation.
| `R` | Re-initialize the simulation.

## How to Generate Vulkan SPIR-V Files

The SPIR-V files for the Vulkan vertex shader and fragment shader were compiled with `glslang`, which is included in the Vulkan SDK.
The command lines used to compile the SPIR-V files were:

```sh
/path/to/glslangvalidator --target-env vulkan1.0 nbodyvk.vert -o nbodyvk.vert.spv
/path/to/glslangvalidator --target-env vulkan1.0 nbodyvk.frag -o nbodyvk.frag.spv
```
