# Using the Intercept Layer for OpenCL Applications

## Part 5: Improving Performance More

In part 4 we improved the performance of the tutorial application significantly by choosing a more efficient ND-range when we executed our kernel, but our performance is still limited by execution on the device.
Let's see if we can improve the performance of the kernel code itself.

The kernel code is pretty straightforward, consisting of just one main loop:

```c
float result = 0.0f;
for( int i = 0; i < cIterations; i++ ) {
    if(fabs(zi) > cThreshold) {
        break;
    }

    // zn = sin(z)
    float zrn = sin(zr) * cosh(zi);
    float zin = cos(zr) * sinh(zi);

    // z = c * zn = c * sin(z)
    zr = cr * zrn - ci * zin;
    zi = cr * zin + ci * zrn;

    result += 1.0f / cIterations;
}
```

One area to consider though are the calls to `sin`, `cos`, `cosh`, and `sinh`.
By default, the OpenCL C built-in math functions are precise across a wide range of inputs.
Since our goal is only to produce a nice picture, we're probably fine trading some of this precision for improved performance.
One way to trade off precision for performance is by passing the `-cl-fast-relaxed-math` program build option.
The Intercept Layer supports two different methods of specifying program build options for fast iteration and without modifying application source code.

The first method uses the same [Dump and Inject](https://github.com/intel/opencl-intercept-layer/blob/master/docs/injecting_programs.md) capabilities from part 2 but to inject modified program build options instead of modified OpenCL kernel code.
To use this method, enable the `InjectProgramSource` control from before, and put a program options file in the `Inject` directory similar to the way the OpenCL kernel code was injected in part 2.
This method allows for selectively specifying program build options for some OpenCL kernels without modifying program build options for all kernels.

Since the tutorial application only executes a single kernel, though, we can instead set `AppendBuildOptions` to append `-cl-fast-relaxed-math` to the build options for all OpenCL kernels.
Let's give this a try and run the tutorial application once more:

```
$ cliloader -h -dv ./sinjulia
-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
CLIntercept (64-bit) is loading...
<snip>
Control AppendBuildOptions is set to non-default value: -cl-fast-relaxed-math
<snip>

Executing the kernel 16 times
Global Work Size = ( 3840, 2160 )
Local work size = NULL
Finished in 0.491725 seconds
Wrote image file sinjulia.bmp
Total Enqueues: 18


Host Performance Timing Results:

Total Time (ns): 491035134

                     Function Name,  Calls,     Time (ns), Time (%),  Average (ns),      Min (ns),      Max (ns)
                clEnqueueMapBuffer,      1,         27610,    0.01%,         27610,         27610,         27610
clEnqueueNDRangeKernel( SinJulia ),     15,        286985,    0.06%,         19132,         16722,         45023
           clEnqueueUnmapMemObject,      1,         12864,    0.00%,         12864,         12864,         12864
                          clFinish,      2,     489182142,   99.62%,     244591071,          2555,     489179587
                           clFlush,     16,          4383,    0.00%,           273,           196,          1290
             clReleaseCommandQueue,      1,          5382,    0.00%,          5382,          5382,          5382
                  clReleaseContext,      1,           489,    0.00%,           489,           489,           489
                   clReleaseDevice,      1,           646,    0.00%,           646,           646,           646
                   clReleaseKernel,      1,         23465,    0.00%,         23465,         23465,         23465
                clReleaseMemObject,      1,       1490724,    0.30%,       1490724,       1490724,       1490724
                  clReleaseProgram,      1,           444,    0.00%,           444,           444,           444

Device Performance Timing Results for Intel(R) Graphics [0x5916] (24CUs, 1100MHz):

Total Time (ns): 457092230

                                 Function Name,  Calls,     Time (ns), Time (%),  Average (ns),      Min (ns),      Max (ns)
SinJulia SIMD16 GWS[ 3840 x 2160 ] LWS[ NULL ],     16,     457091912,  100.00%,      28568244,      28467166,      29449083
                            clEnqueueMapBuffer,      1,           126,    0.00%,           126,           126,           126
                       clEnqueueUnmapMemObject,      1,           192,    0.00%,           192,           192,           192
CLIntercept is shutting down...
... shutdown complete.
```

The magnitude of this improvement will also be device-specific, but on this OpenCL device using `-cl-fast-relaxed-math` produced another **2-3x** improvement in performance, and the output bitmap still looks good.
Many other devices will see a similar level of improvement.

Move on to the final part 6!

## Next Step

* Part 6: [Final Words and Additional Things to Try](part6.md)
