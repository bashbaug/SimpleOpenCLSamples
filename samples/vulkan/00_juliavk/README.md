# Julia Set with Vulkan

## Sample Purpose

This is a modified version of the earlier Julia set sample.
Similar to the earlier Julia Set sample, an OpenCL kernel is used to generate a Julia set image.
The main difference between this sample and the earlier sample is that in this sample the Julia set image is used as an Vulkan texture and rendered to the screen instead of writing it to a BMP file.

This sample currently copies the result of the OpenCL kernel to the Vulkan texture on the host, however several new extensions were recently provisionally released to enable [interop between OpenCL and Vulkan devices](https://www.khronos.org/blog/khronos-releases-opencl-3.0-extensions-for-neural-network-inferencing-and-opencl-vulkan-interop).
When drivers are available that support these extensions, this sample will be updated to use them!

## Key APIs and Concepts

This example shows how to share an Vulkan texture with OpenCL.

## Command Line Options

Note: Many of these command line arguments are identical to the earlier Julia set sample.

| Option | Default Value | Description |
|:--|:-:|:--|
| `-d <index>` | 0 | Specify the index of the OpenCL device in the platform to execute on the sample on.
| `-p <index>` | 0 | Specify the index of the OpenCL platform to execute the sample on.
| `--gwx <number>` | 512 | Specify the global work size to execute, in the X direction.  This also determines the width of the generated image.
| `--gwy <number>` | 512 | Specify the global work size to execute, in the Y direction.  This also determines the height of the generated image.
| `--lwx <number>` | 0 | Specify the local work size in the X direction.  If either local works size dimension is zero a `NULL` local work size is used.
| `--lwy <number>` | 0 | Specify the local work size in the Y direction.  If either local works size dimension is zero a `NULL` local work size is used.

## Controls While Running

| Control | Description |
|:--|:--|
| `Escape` | Exits from the sample.
| `Space` | Toggle animation (default: `false`).
| `A` | Increase the real part of the complex constant `C`.
| `Z` | Decrease the real part of the complex constant `C`.
| `S` | Increase the imaginary part of the complex constant `C`.
| `X` | Decrease the imaginary part of the complex constant `C`.

## How to Generate Vulkan SPIR-V Files

The SPIR-V files for the Vulkan vertex shader and fragment shader were compiled with `glslang`, which is included in the Vulkan SDK.
The command lines used to compile the SPIR-V files were:

```sh
/path/to/glslangvalidator --target-env vulkan1.0 juliavk.vert -o juliavk.vert.spv
/path/to/glslangvalidator --target-env vulkan1.0 juliavk.frag -o juliavk.frag.spv
```
