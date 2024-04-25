# Using the Intercept Layer for OpenCL Applications

## Part 1: Fixing an OpenCL Error

In this part of the tutorial we will fix a bug so the program will no longer crash.
The Intercept Layer for OpenCL Applications can easily identify OpenCL errors in a program and provide clues why the OpenCL error might be occurring.

To find where this OpenCL error is occurring, enable the controls `ErrorLogging` and `CallLogging`.
Please refer to the [controls](https://github.com/intel/opencl-intercept-layer/blob/master/docs/controls.md#controls) page for instructions how to enable these controls and other controls that will be used throughout the tutorial.
Note that `CallLogging` can be enabled by passing the `-c` option to `cliloader` and does not necessarily need to be enabled manually.

After enabling these controls, re-run the tutorial application.

```
$ cliloader ./sinjulia 
-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
CLIntercept (64-bit) is loading...
CLIntercept file location: /home/bashbaug/bin/../lib/libOpenCL.so
CLIntercept URL: https://github.com/intel/opencl-intercept-layer
CLIntercept git description: v3.0.0-11-gd73caba
CLIntercept git refspec: refs/heads/master
CLIntercept git hash: d73caba0273207c47d3094865c1a9e145acf2018
CLIntercept optional features:
    cliloader(supported)
    cliprof(supported)
    kernel overrides(supported)
    ITT tracing(NOT supported)
    MDAPI(supported)
    clock(steady_clock)
CLIntercept environment variable prefix: CLI_
CLIntercept config file: clintercept.conf
Trying to load dispatch from: ./real_libOpenCL.so
Couldn't load library: ./real_libOpenCL.so
Trying to load dispatch from: /usr/lib/x86_64-linux-gnu/libOpenCL.so.1
Couldn't get exported function pointer to: clGetGLContextInfoKHR
... success!
Control CallLogging is set to non-default value: true
Control ErrorLogging is set to non-default value: true
Control ReportToStderr is set to non-default value: true
Timer Started!
... loading complete.
>>>> clGetPlatformIDs
<<<< clGetPlatformIDs -> CL_SUCCESS
>>>> clGetPlatformIDs
<<<< clGetPlatformIDs -> CL_SUCCESS
*** Important Note! ***
This is the Intercept Layer Tutorial application.
It will crash initially!  Please see the tutorial README for details.
>>>> clGetPlatformInfo: platform = Intel(R) OpenCL HD Graphics (0x55637a388e20), param_name = CL_PLATFORM_NAME (00000902)
<<<< clGetPlatformInfo -> CL_SUCCESS
>>>> clGetPlatformInfo: platform = Intel(R) OpenCL HD Graphics (0x55637a388e20), param_name = CL_PLATFORM_NAME (00000902)
<<<< clGetPlatformInfo -> CL_SUCCESS
Running on platform: Intel(R) OpenCL HD Graphics
>>>> clGetDeviceIDs: platform = Intel(R) OpenCL HD Graphics (0x55637a388e20), device_type = <unknown> (1000)
ERROR! clGetDeviceIDs returned CL_INVALID_DEVICE_TYPE (-31)
<<<< clGetDeviceIDs -> CL_INVALID_DEVICE_TYPE
Segmentation fault (core dumped)
```

There are two parts of this output to pay attention to.
First, ensure that the controls have been properly enabled:

```
Control CallLogging is set to non-default value: true
Control ErrorLogging is set to non-default value: true
```

If `ErrorLogging` and `CallLogging` are not "set to non-default values" then the controls are not enabled properly.

Second, observe that that the call to `clGetDeviceIDs` is failing and is returning `CL_INVALID_DEVICE_TYPE`.
Furthermore, observe that the passed-in `device_type` is unrecognized.
Because the call to `clGetDeviceIDs` is failing, we don't have any OpenCL devices to choose from, and since the tutorial application does not gracefully handle this case it is crashing.
Let's pass in a proper device type so we have OpenCL devices to choose from.

```c++
// Part 1: Query the devices in this platform.
// When querying for OpenCL devices we pass the types of devices we want to
// query. This will either be one or more specific device types, e.g.
// CL_DEVICE_TYPE_CPU, or we can pass in CL_DEVICE_TYPE_ALL, which will get
// all devices. Passing CL_DEVICE_TYPE isn't a valid device type and will
// result in an OpenCL error. What should we pass instead?
platforms[platformIndex].getDevices(CL_DEVICE_TYPE, &devices);
```

After fixing this bug the tutorial application should be able to query devices in the platform and should no longer crash.

Move on to part 2!

## Next Step

* Part 2: [Fixing an OpenCL Program Build Error](part2.md)
