# OpenCLUtils

This is my library of OpenCL utility functions.
There are many like it, but this one is mine.

This work is currently Intel Confidential, for Internal Use Only!

## Dependencies

This project requires a C++ compiler and OpenCL headers.
There are no other known dependencies.

The OpenCL headers are discovered via the CMake `find_package(OpenCL)`, which will usually automatically find OpenCL headers that are installed on the system.
If they don't, or if other headers should be used instead, please modify the CMake variable `OpenCL_INCLUDE_DIRS`.

## Build

The project uses [CMake][cmake] to generate platform-specific build files.
It is designed to easily integrate into other projects that are built using CMake.
The easiest way to use it is to clone this repo into its own directory.
For example:

```sh
cd <app-dir>
git clone http://github.intel.com/bashbaug/OpenCLUtils.git external/openclutils
```

Alternatively, this project may be managed as a git submodule.

Then, in the application CMake file, add this project, and its include directories:

```
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/openclutils)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/openclutils/include)
```

Finally, link your application with the OpenCLUtils lib:

```
target_link_libraries(<your_app_name> OpenCLUtils)
````

This should generate build files that automatically build and link with this utility library.

## License

OpenCLUtils may eventually be licensed under the [MIT License](LICENSE).

This work is currently Intel Confidential, for Internal Use Only.

[cmake]: https://cmake.org/
