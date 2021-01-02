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
Note that `HostPerformanceTiming` can be enabled by passing the `-h` option to `cliloader` and does not necessarily need to be set manually.
Also note that `DevicePerformanceTiming` can be enabled by passing the `-d` option to `cliloader`, and the additional device timing controls can be enabled by passing the `-dv` option to `cliloader`.

After setting these controls, re-run the tutorial application.  In addition to the usual loading log and application output:

```sh
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
Build Info for program 0x56116cf85b10 (0000_3DC4555B_0000_00000000) for 1 device(s):
    Build finished in 290.12 ms.
Build Status for device 0 = Intel(R) Graphics [0x5916] (OpenCL C 3.0 ): CL_BUILD_SUCCESS
-------> Start of Build Log:
<------- End of Build Log

Executing the kernel 16 times
Global Work Size = ( 3847, 2161 )
Local work size = NULL
Finished in 11.516864 seconds
Wrote image file sinjulia.bmp
```

The Intercept Layer for OpenCL Applications will also print a "report" showing the OpenCL host API calls and how long they took to execute:

```sh
Host Performance Timing Results:

Total Time (ns): 11663415031

                     Function Name,  Calls,     Time (ns), Time (%),  Average (ns),      Min (ns),      Max (ns)
                clEnqueueMapBuffer,      1,         88023,    0.00%,         88023,         88023,         88023
clEnqueueNDRangeKernel( SinJulia ),     15,        350098,    0.00%,         23339,         21470,         45724
           clEnqueueUnmapMemObject,      1,         11876,    0.00%,         11876,         11876,         11876
                          clFinish,      2,   11661543982,   99.98%,    5830771991,          2640,   11661541342
                           clFlush,     16,          4211,    0.00%,           263,           185,          1258
             clReleaseCommandQueue,      1,          5276,    0.00%,          5276,          5276,          5276
                  clReleaseContext,      1,           429,    0.00%,           429,           429,           429
                   clReleaseDevice,      1,           803,    0.00%,           803,           803,           803
                   clReleaseKernel,      1,         21528,    0.00%,         21528,         21528,         21528
                clReleaseMemObject,      1,       1388346,    0.01%,       1388346,       1388346,       1388346
                  clReleaseProgram,      1,           459,    0.00%,           459,           459,           459
```

Also, the Intercept Layer for OpenCL Applications will print a "report" showing the OpenCL commands that executed code on each OpenCL device and how long they took to execute:

```sh
Device Performance Timing Results for Intel(R) Graphics [0x5916] (24CUs, 1100MHz):

Total Time (ns): 11466683589

                                 Function Name,  Calls,     Time (ns), Time (%),  Average (ns),      Min (ns),      Max (ns)
SinJulia SIMD16 GWS[ 3847 x 2161 ] LWS[ NULL ],     16,   11466682328,  100.00%,     716667645,     714626750,     738733333
                            clEnqueueMapBuffer,      1,           815,    0.00%,           815,           815,           815
                       clEnqueueUnmapMemObject,      1,           446,    0.00%,           446,           446,           446
```

There is a lot to analyze here!
Let's look at the host performance timing report first.
Almost all of the host time is being spent in `clFinish`.
This generally means that the host is spending its time waiting for the OpenCL device to finish executing.
For the tutorial application this is not a surprise since it is fairly simple and only executes an OpenCL kernel in a loop.

The device performance timing result confirms that most of the time is spent executing the OpenCL kernel.
Notice that the time spent executing our SinJulia kernel is almost identical to the time spent in the host API call `clFinish`.
We can also observe the global work size for the OpenCL kernel.
The tutorial application does not specify a local work-group size, so it shows up as `NULL` in the report, but if a local work-group size were specified it would be included in the report also.

Since we know most of the execution time of our application is spent executing the OpenCL kernel we know where to focus our optimization efforts.
We can generally improve OpenCL execution time on the device by optimizing the ND-range when we execute our kernel or the kernel code itself.
In this case, the global work size looks peculiar.
In fact, a closer examination indicates that the global work size is prime in both dimensions!
This is not ideal, since our kernel is compiled to require uniform work groups, meaning that the kernel is executing with a local work size equal to one.
For most data-parallel kernels we want a larger local work-group size.
If we enable the control `KernelInfoLogging` we can print information about each kernel when it is created, including the query for `CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE`, which will give a hint about the desired local work-group size for the device:

```sh
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
This can be done initially by passing `-gwx 3840 -gwy 2160` to the tutorial application, or by modifying the default values in the application source code:

```sh
$ cliloader -h -dv ./sinjulia -gwx 3840 -gwy 2160
-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
CLIntercept (64-bit) is loading...
<snip>

Executing the kernel 16 times
Global Work Size = ( 3840, 2160 )
Local work size = NULL
Finished in 1.422414 seconds
Wrote image file sinjulia.bmp
Total Enqueues: 18


Host Performance Timing Results:

Total Time (ns): 1421782871

                     Function Name,  Calls,     Time (ns), Time (%),  Average (ns),      Min (ns),      Max (ns)
                clEnqueueMapBuffer,      1,         24587,    0.00%,         24587,         24587,         24587
clEnqueueNDRangeKernel( SinJulia ),     15,        315471,    0.02%,         21031,         17276,         42736
           clEnqueueUnmapMemObject,      1,         13731,    0.00%,         13731,         13731,         13731
                          clFinish,      2,    1419975979,   99.87%,     709987989,          3286,    1419972693
                           clFlush,     16,          4900,    0.00%,           306,           196,          1173
             clReleaseCommandQueue,      1,          4900,    0.00%,          4900,          4900,          4900
                  clReleaseContext,      1,           608,    0.00%,           608,           608,           608
                   clReleaseDevice,      1,         75593,    0.01%,         75593,         75593,         75593
                   clReleaseKernel,      1,         20689,    0.00%,         20689,         20689,         20689
                clReleaseMemObject,      1,       1345821,    0.09%,       1345821,       1345821,       1345821
                  clReleaseProgram,      1,           592,    0.00%,           592,           592,           592

Device Performance Timing Results for Intel(R) Graphics [0x5916] (24CUs, 1100MHz):

Total Time (ns): 1368212401

                                 Function Name,  Calls,     Time (ns), Time (%),  Average (ns),      Min (ns),      Max (ns)
SinJulia SIMD16 GWS[ 3840 x 2160 ] LWS[ NULL ],     16,    1368211662,  100.00%,      85513228,      85312000,      86285416
                            clEnqueueMapBuffer,      1,           480,    0.00%,           480,           480,           480
                       clEnqueueUnmapMemObject,      1,           259,    0.00%,           259,           259,           259
CLIntercept is shutting down...
... shutdown complete.

```

The magnitude of the improvement by choosing a non-prime global work size will be device-specific, but on this OpenCL device choosing a different global work size produced an **8x** improvement in performance - not bad!
Many other devices will see a similar level of improvement.

Move on to part 5!

## Next Step

* Part 5: [Improving Performance More](part5.md)
