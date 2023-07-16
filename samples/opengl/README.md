# OpenGL Samples

This directory contains samples that use [OpenGL](https://www.khronos.org/opengl/) to visualize results.
OpenGL is a widely supported industry standard for graphics and rendering and can interoperate well with many OpenCL devices.

## Dependencies

These samples require OpenGL headers and libraries.

Additionally, these samples use [GLFW](https://www.glfw.org/) to abstract many of the operating system specific parts of OpenGL.
GLFW supports Windows, macOS, and Linux.
Pre-built packages are available for many platforms, or GLFW may be built from source.

If these dependencies are not found then the OpenGL samples will not be built.

### Using Pre-build GLFW Packages

This is the preferred method for using GLFW.
Please see the [GLFW Download Page](https://www.glfw.org/download.html) for details.

### Building GLFW from Source

The following steps are recommended when GLFW is built from source.
Please refer to the [Compiling GLFW](https://www.glfw.org/docs/latest/compile_guide.html) reference page for details.

1. Build GLFW separately from these OpenCL samples.
The GLFW source code may be cloned into a completely separate directory or into the `external` directory for these samples.
2. Build GLFW as a static library.
3. On Linux, build GLFW for X11.
3. Build a `Release` or `RelWithDebInfo` GLFW.
4. Install GLFW either into the `external` directory for the samples (recommended), or into a system directory.
If GLFW is installed into the `external` directory it _should_ be detected automatically by these samples.

Sample build instructions:

1. Clone the GLFW source code.
In these instructions we will clone into the `external` directory.

    ```sh
    $ git clone https://github.com/glfw/glfw.git external/glfw-src
    ```

2. Create build files.

    ```sh
    $ cd external/glfw-src && mkdir build && cd build
    $ cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/path/to/your/SimpleOpenCLSamples/external/glfw \
        -DGLFW_BUILD_DOCS=0 -DGLFW_BUILD_EXAMPLES=0 -DGLFW_BUILD_TESTS=0
    ```

3. Build and install GLFW into the `external` directory.

    ```sh
    $ cmake --build . --target install --config Release
    ```

After installing, GLFW should be found by the OpenCL samples.

## Summary of OpenGL Samples

* [juliagl](./00_juliagl): Demonstrates sharing an OpenGL texture with OpenCL.
* [nbodygl](./01_nbodygl): Demonstrates rendering an OpenCL vertex buffer with OpenGL.
* [sobelgl](./01_sobelgl): Displays the result of a Sobel edge detection filter with OpenGL.
