# Using the Intercept Layer for OpenCL Applications

## Part 2: Fixing an OpenCL Program Build Error

In part 1 of the tutorial we fixed an OpenCL error so the program no longer crashes but it still does not run correctly.
In this part of the tutorial we will fix the next error in the OpenCL kernel code.
The Intercept Layer for OpenCL Applications can pinpoint errors in OpenCL kernel code as well as OpenCL API calls and has tools for quickly iterating to find a fix.

After fixing the bug in part 1, and with `ErrorLogging` and `CallLogging` still set, the next OpenCL error is this one:

```
$ cliloader ./sinjulia 
-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
CLIntercept (64-bit) is loading...
<snip>
>>>> clBuildProgram: program = 0x55e07f164b10, pfn_notify = (nil)
ERROR! clBuildProgram returned CL_BUILD_PROGRAM_FAILURE (-11)
<<<< clBuildProgram -> CL_BUILD_PROGRAM_FAILURE
<snip>
```

A `clBuildProgram` error usually means that the OpenCL kernel code is incorrect.
To identify the error in the OpenCL kernel code, enable the control `BuildLogging`.

After enabling this control, re-run the tutorial application.

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
Control BuildLogging is set to non-default value: true
Control CallLogging is set to non-default value: true
Control ErrorLogging is set to non-default value: true
Control ReportToStderr is set to non-default value: true
Timer Started!
... loading complete.
<snip>
>>>> clBuildProgram: program = 0x56131b0feb10, pfn_notify = (nil)
ERROR! clBuildProgram returned CL_BUILD_PROGRAM_FAILURE (-11)
Build Info for program 0x56131b0feb10 (0000_66C650B0_0000_00000000) for 1 device(s):
    Build finished in 139.45 ms.
Build Status for device 0 = Intel(R) Graphics [0x5916] (OpenCL C 3.0 ): CL_BUILD_ERROR
-------> Start of Build Log:
1:13:27: error: use of undeclared identifier 'xMx'; did you mean 'xMax'?
    float zr = (float)x / xMx  * (zMax - zMin) + zMin;
                          ^~~
                          xMax
1:4:15: note: 'xMax' declared here
    const int xMax = get_global_size(0);
              ^
<------- End of Build Log

<<<< clBuildProgram -> CL_BUILD_PROGRAM_FAILURE
<snip>
```

As before, there are two parts of this output to pay attention to.
First, ensure that the control has been properly enabled:

```
Control BuildLogging is set to non-default value: true
```

If `BuildLogging` is not "set to a non-default value" then the control is not enabled properly.

Second, observe the error in the OpenCL kernel code.
Looks like there is a typo and `xMax` was incorrectly written as `xMx` - uh oh!

For the tutorial application, we have the application source code and it is easy to modify the OpenCL kernel code and rebuild, but this isn't always the case.
If it is not easy to modify the kernel and rebuild, we can [Dump and Inject](https://github.com/intel/opencl-intercept-layer/blob/master/docs/injecting_programs.md) the modified OpenCL kernel code instead.

To do this, we will first enable the control `DumpProgramSource` and re-run the tutorial application.
After enabling this control we should see output like the following in our log:

```
Dumping program to file (inject): /home/bashbaug/CLIntercept_Dump/sinjulia/CLI_0000_66C650B0_source.cl
```

"Dumps" are recordings of OpenCL objects emitted to files.
If the default "dump" directory is undesirable or unavailable, an alternate dump directory can be specified with the `DumpDir` control.

If we open the dumped source file we should see our OpenCL kernel source, with the error shown above in the build log.
Note that the program hash in the file name `66C650B0` matches the program hash in the build log.
Let's fix the typo in the OpenCL kernel source in this file, by changing `xMx` to `xMax`.
After fixing the typo, save it in an `Inject` sub-directory in the dump directory, enable the control `InjectProgramSource` and re-run the tutorial application.
Now we should see output like the following in our log:

```
Injecting source file: /home/bashbaug/CLIntercept_Dump/sinjulia/Inject/CLI_0000_66C650B0_source.cl
>>>> clCreateProgramWithSource: context = 0x55d0baaa0510, count = 1
<<<< clCreateProgramWithSource: returned 0x55d0baadcb10, program number = 0000 -> CL_SUCCESS
Injection options file doesn't exist: /home/bashbaug/CLIntercept_Dump/sinjulia/Inject/CLI_0000_66C650B0_0000_00000000_options.txt
Injection options file doesn't exist: /home/bashbaug/CLIntercept_Dump/sinjulia/Inject/CLI_66C650B0_0000_00000000_options.txt
Injection options file doesn't exist: /home/bashbaug/CLIntercept_Dump/sinjulia/Inject/CLI_66C650B0_options.txt
Injection options file doesn't exist: /home/bashbaug/CLIntercept_Dump/sinjulia/Inject/CLI_options.txt
>>>> clBuildProgram: program = 0x55d0baadcb10, pfn_notify = (nil)
Build Info for program 0x55d0baadcb10 (0000_66C650B0_0000_00000000) for 1 device(s):
    Build finished in 282.97 ms.
Build Status for device 0 = Intel(R) Graphics [0x5916] (OpenCL C 3.0 ): CL_BUILD_SUCCESS
-------> Start of Build Log:
<------- End of Build Log

<<<< clBuildProgram -> CL_SUCCESS
```

The lines about the injection options not existing are OK and expected, since we didn't include any options to inject.
Notice that the `clBuildProgram` OpenCL error is fixed - excellent!

Since this was the only error in our OpenCL kernel code, let's modify it in the application source code itself.
Modifying the kernel code in the application source code will change the program hash and the modified OpenCL kernel source should no longer be injected.

```
Injection source file doesn't exist: /home/bashbaug/CLIntercept_Dump/sinjulia/Inject/CLI_0000_F05F194C_source.cl
Injection source file doesn't exist: /home/bashbaug/CLIntercept_Dump/sinjulia/Inject/CLI_F05F194C_source.cl
>>>> clCreateProgramWithSource: context = 0x55b2bd586510, count = 1
<<<< clCreateProgramWithSource: returned 0x55b2bd5c2b10, program number = 0000 -> CL_SUCCESS
Dumping program to file (inject): /home/bashbaug/CLIntercept_Dump/sinjulia/CLI_0000_F05F194C_source.cl
Injection options file doesn't exist: /home/bashbaug/CLIntercept_Dump/sinjulia/Inject/CLI_0000_F05F194C_0000_00000000_options.txt
Injection options file doesn't exist: /home/bashbaug/CLIntercept_Dump/sinjulia/Inject/CLI_F05F194C_0000_00000000_options.txt
Injection options file doesn't exist: /home/bashbaug/CLIntercept_Dump/sinjulia/Inject/CLI_F05F194C_options.txt
Injection options file doesn't exist: /home/bashbaug/CLIntercept_Dump/sinjulia/Inject/CLI_options.txt
>>>> clBuildProgram: program = 0x55b2bd5c2b10, pfn_notify = (nil)
Build Info for program 0x55b2bd5c2b10 (0000_F05F194C_0000_00000000) for 1 device(s):
    Build finished in 282.76 ms.
Build Status for device 0 = Intel(R) Graphics [0x5916] (OpenCL C 3.0 ): CL_BUILD_SUCCESS
-------> Start of Build Log:
<------- End of Build Log

<<<< clBuildProgram -> CL_SUCCESS
```

After fixing the bug in the OpenCL kernel code, move on to part 3!

## Next Step

* Part 3: [Fixing an OpenCL Map Error](part3.md)
