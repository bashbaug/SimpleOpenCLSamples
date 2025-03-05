# Using the Intercept Layer for OpenCL Applications

## Part 4: Profiling and Improving Performance

The tutorial application is running correctly now and generating a nice output bitmap - that's great!
Unfortunately, it still isn't running very well.
Let's see if we can figure out why.

The Intercept Layer for OpenCL Applications can profile both the OpenCL host API calls and the OpenCL commands executing on the device.
Enabling both types of profiling is a good way to discover where an application is spending its time.

To profile OpenCL host API calls, enable the control `HostPerformanceTiming`, and to profile OpenCL device commands, enable the control `DevicePerformanceTiming`.
For some programs, such as the tutorial application, it is helpful to set the control `HostPerformanceTimingMinEnqueue` to bypass host API calls made when setting up the context and compiling kernels.
Additionally, it may be helpful to enable the controls `DevicePerformanceTimeGWSTracking`, `DevicePerformanceTimeLWSTracking`, and `DevicePerformanceTimeKernelInfoTracking`, which will include additional information for OpenCL kernel execution on the device.
Note that `HostPerformanceTiming` can be enabled by passing the `-h` option to `cliloader` and does not necessarily need to be enabled manually.
Also note that `DevicePerformanceTiming` can be enabled by passing the `-d` option to `cliloader`, and the additional device timing controls can be enabled by passing the `-dv` option to `cliloader`.

After enabling these controls, re-run the tutorial application.
In addition to the usual loading log and application output:

```
$ cliloader -h -dv ./sinjulia
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
Control BuildLogging is set to non-default value: true
Control ErrorLogging is set to non-default value: true
Control ReportToStderr is set to non-default value: true
Control HostPerformanceTiming is set to non-default value: true
Control DevicePerformanceTiming is set to non-default value: true
Control DevicePerformanceTimeKernelInfoTracking is set to non-default value: true
Control DevicePerformanceTimeGWSTracking is set to non-default value: true
Control DevicePerformanceTimeLWSTracking is set to non-default value: true
Control HostPerformanceTimingMinEnqueue is set to non-default value: 2
Timer Started!
... loading complete.
Running on platform: Intel(R) OpenCL HD Graphics
Running on device: Intel(R) Graphics [0x5916]
Build Info for program 0x55af00e8bb10 (0000_F05F194C_0000_00000000) for 1 device(s):
    Build finished in 283.75 ms.
Build Status for device 0 = Intel(R) Graphics [0x5916] (OpenCL C 3.0 ): CL_BUILD_SUCCESS
-------> Start of Build Log:
<------- End of Build Log

Executing the kernel 16 times
Global Work Size = ( 3847, 2161 )
Local work size = NULL
Finished in 11.520581 seconds
Wrote image file sinjulia.bmp
```

The Intercept Layer will also display a "report" showing the OpenCL host API calls and how long they took to execute:

```
Host Performance Timing Results:

Total Time (ns): 11520204571

                     Function Name,  Calls,     Time (ns), Time (%),  Average (ns),      Min (ns),      Max (ns)
                clEnqueueMapBuffer,      1,         25902,    0.00%,         25902,         25902,         25902
clEnqueueNDRangeKernel( SinJulia ),     15,        349393,    0.00%,         23292,         20403,         45464
           clEnqueueUnmapMemObject,      1,         13578,    0.00%,         13578,         13578,         13578
                          clFinish,      2,   11518251007,   99.98%,    5759125503,          2258,   11518248749
                           clFlush,     16,          4125,    0.00%,           257,           188,          1022
             clReleaseCommandQueue,      1,          5412,    0.00%,          5412,          5412,          5412
                  clReleaseContext,      1,           431,    0.00%,           431,           431,           431
                   clReleaseDevice,      1,           758,    0.00%,           758,           758,           758
                   clReleaseKernel,      1,         21333,    0.00%,         21333,         21333,         21333
                clReleaseMemObject,      1,       1532215,    0.01%,       1532215,       1532215,       1532215
                  clReleaseProgram,      1,           417,    0.00%,           417,           417,           417
```

Also, the Intercept Layer will display a "report" showing the OpenCL commands that executed code on each OpenCL device and how long they took to execute:

```
Device Performance Timing Results for Intel(R) Graphics [0x5916] (24CUs, 1100MHz):

Total Time (ns): 11454451416

                                 Function Name,  Calls,     Time (ns), Time (%),  Average (ns),      Min (ns),      Max (ns)
SinJulia SIMD16 GWS[ 3847 x 2161 ] LWS[ NULL ],     16,   11454450744,  100.00%,     715903171,     714320583,     734597750
                            clEnqueueMapBuffer,      1,           482,    0.00%,           482,           482,           482
                       clEnqueueUnmapMemObject,      1,           190,    0.00%,           190,           190,           190
```

There is a lot to analyze here!
Let's look at the host performance timing report first.
Almost all of the host time is being spent in `clFinish`.
This generally means that the host is spending its time waiting for the OpenCL device to finish executing.
For the tutorial application this is not a surprise since it is fairly simple and only executes an OpenCL kernel in a loop.

The device performance timing result confirms that most of the time is spent executing the OpenCL kernel.
Notice that the time spent executing our SinJulia kernel is almost identical to the time spent in the host API call `clFinish`.
We can also observe the ND-range specified for the SinJulia kernel, such as the global work size and the local work-group size.
The tutorial application does not specify a local work-group size, so it shows up as `NULL` in the report, but if a local work-group size were specified it would also be included in the report.

Since we know most of the execution time of our application is spent executing the OpenCL kernel we should focus our optimization efforts there.
We can generally improve OpenCL execution time on the device by optimizing the ND-range for our kernel or the kernel code itself.
In this case, the global work size looks peculiar.
In fact, a closer examination indicates that the global work size is prime in both dimensions!
This is not ideal, since our kernel is compiled to require uniform work groups, meaning that the kernel is executing with a local work size equal to one.
For most data-parallel kernels we want a larger local work-group size.
If we enable the control `KernelInfoLogging` we can print information about each kernel when it is created, including the query for `CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE`, which will give a hint about the desired local work-group size for the device:

```
Kernel Info for: SinJulia
    For device: Intel(R) Graphics [0x5916]
        Num Args: 3
        Preferred Work Group Size Multiple: 16
        Work Group Size: 256
        Private Mem Size: 0
        Local Mem Size: 0
        Spill Mem Size: 0
```

Let's choose a different global work size instead, such as one that will output a 4K bitmap - 3840 x 2160.
This can be done by passing `--gwx 3840 --gwy 2160` to the tutorial application, or by modifying the default values in the application source code:

```
$ cliloader -h -dv ./sinjulia --gwx 3840 --gwy 2160
-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
CLIntercept (64-bit) is loading...
<snip>

Executing the kernel 16 times
Global Work Size = ( 3840, 2160 )
Local work size = NULL
Finished in 1.417555 seconds
Wrote image file sinjulia.bmp
Total Enqueues: 18


Host Performance Timing Results:

Total Time (ns): 1416862062

                     Function Name,  Calls,     Time (ns), Time (%),  Average (ns),      Min (ns),      Max (ns)
                clEnqueueMapBuffer,      1,         26399,    0.00%,         26399,         26399,         26399
clEnqueueNDRangeKernel( SinJulia ),     15,        295244,    0.02%,         19682,         17738,         41870
           clEnqueueUnmapMemObject,      1,         11975,    0.00%,         11975,         11975,         11975
                          clFinish,      2,    1415208192,   99.88%,     707604096,          2426,    1415205766
                           clFlush,     16,          4807,    0.00%,           300,           222,          1078
             clReleaseCommandQueue,      1,          5979,    0.00%,          5979,          5979,          5979
                  clReleaseContext,      1,           430,    0.00%,           430,           430,           430
                   clReleaseDevice,      1,           877,    0.00%,           877,           877,           877
                   clReleaseKernel,      1,         24206,    0.00%,         24206,         24206,         24206
                clReleaseMemObject,      1,       1283568,    0.09%,       1283568,       1283568,       1283568
                  clReleaseProgram,      1,           385,    0.00%,           385,           385,           385

Device Performance Timing Results for Intel(R) Graphics [0x5916] (24CUs, 1100MHz):

Total Time (ns): 1364981027

                                 Function Name,  Calls,     Time (ns), Time (%),  Average (ns),      Min (ns),      Max (ns)
SinJulia SIMD16 GWS[ 3840 x 2160 ] LWS[ NULL ],     16,    1364980328,  100.00%,      85311270,      85195000,      85634333
                            clEnqueueMapBuffer,      1,           508,    0.00%,           508,           508,           508
                       clEnqueueUnmapMemObject,      1,           191,    0.00%,           191,           191,           191
CLIntercept is shutting down...
... shutdown complete.

```

The magnitude of the improvement by choosing a non-prime global work size will be device-specific, but on this OpenCL device choosing the different global work size produced an **8x** improvement in performance - not bad!
Many other devices will see a similar level of improvement.

Move on to part 5!

## Next Step

* Part 5: [Improving Performance More](part5.md)
