# Using the Intercept Layer for OpenCL Applications

## Part 0: Building and Running the Tutorial

This part of the tutorial is to ensure everything is setup correctly.
We will need to build the tutorial application itself and the Intercept Layer for OpenCL Applications.

### Building and Running the Tutorial Application

First, ensure that the tutorial application itself builds and runs.
It will crash initially - that's fine!
As part of the tutorial we will fix bugs that are preventing the tutorial application from running and running well.

```
$ ./sinjulia 
*** Important Note! ***
This is the Intercept Layer Tutorial application.
It will crash initially!  Please see the tutorial README for details.
Running on platform: Intel(R) OpenCL HD Graphics
Segmentation fault (core dumped)
```

If your system has multiple OpenCL platforms installed and you want to run on a different platform, choose it by passing the `-p` command line option.
If your OpenCL platform supports multiple OpenCL devices and you want to run on a different device, choose it by passing the `-d` command line option.
You can view the installed platforms and devices by running the `enumopencl` sample.

### Building and Installing the Intercept Layer

After the tutorial application is building and running, next build the [Intercept Layer for OpenCL Applications](https://github.com/intel/opencl-intercept-layer) by following the provided [build instructions](https://github.com/intel/opencl-intercept-layer/blob/master/docs/build.md).
This tutorial is written to use the [cliloader](https://github.com/intel/opencl-intercept-layer/blob/master/docs/cliloader.md) utility, but if you prefer you may follow the [installation instructions](https://github.com/intel/opencl-intercept-layer/blob/master/docs/install.md) instead.
After building and installing the Intercept Layer and `cliloader`, you should be able to use it to execute the tutorial application.

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
Control ReportToStderr is set to non-default value: true
Timer Started!
... loading complete.
*** Important Note! ***
This is the Intercept Layer Tutorial application.
It will crash initially!  Please see the tutorial README for details.
Running on platform: Intel(R) OpenCL HD Graphics
Segmentation fault (core dumped)
```

The tutorial application will still crash, but you should see output from the Intercept Layer as it is loading.
The output from the Intercept Layer as it is running is referred to as the "log".
By default, the log is emitted to `stderr`, but the `LogToFile` control can emit the log to a file and the `LogToDebugger` control can emit the log to a debugger instead, which is convenient for GUI applications or if the application generates a lot of log data.

If the Intercept Layer isn't working, please check the [Troubleshooting and Frequently Asked Questions](https://github.com/intel/opencl-intercept-layer/blob/master/docs/FAQ.md) page.

If it is working, move on to part 1!

## Next Step

* Part 1: [Fixing an OpenCL Error](part1.md)
