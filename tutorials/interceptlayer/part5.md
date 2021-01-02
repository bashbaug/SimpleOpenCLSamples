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
By default, most of the OpenCL C built-in math functions are precise across a wide range of inputs.
Since our goal is only to produce a nice picture, we're probably fine trading some of this precision for improved performance.
One way to trade off precision for performance is by passing the `-cl-fast-relaxed-math` program build option.
The Intercept Layer for OpenCL Applications supports two different methods of specifying program build options for fast iteration and without modifying application source code.

The first method uses the same [Dump and Inject](https://github.com/intel/opencl-intercept-layer/blob/master/docs/injecting_programs.md) capabilities from part 2 but to inject modified program build options instead of modified OpenCL kernel code.
To use this method, set the `InjectProgramSource` control from before, and put a program options file in the `Inject` directory similar to the way the OpenCL kernel code was injected in part 2.
This method allows for selectively specifying program build options for some OpenCL kernels without modifying program build options for all kernels.

Since the tutorial application only executes a single kernel, though, we can instead set `AppendBuildOptions` to append `-cl-fast-relaxed-math` to the build options for all OpenCL kernels.
Let's give this a try and run the tutorial application once more:

```sh
$ cliloader -h -dv ./sinjulia -gwx 3840 -gwy 2160
-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
CLIntercept (64-bit) is loading...
<snip>

Executing the kernel 16 times
Global Work Size = ( 3840, 2160 )
Local work size = NULL
Finished in 0.496394 seconds
Wrote image file sinjulia.bmp
Total Enqueues: 18


Host Performance Timing Results:

Total Time (ns): 496000074

                     Function Name,  Calls,     Time (ns), Time (%),  Average (ns),      Min (ns),      Max (ns)
                clEnqueueMapBuffer,      1,         14336,    0.00%,         14336,         14336,         14336
clEnqueueNDRangeKernel( SinJulia ),     15,        279617,    0.06%,         18641,         15896,         42094
           clEnqueueUnmapMemObject,      1,         41412,    0.01%,         41412,         41412,         41412
                          clFinish,      2,     494257608,   99.65%,     247128804,          8727,     494248881
                           clFlush,     16,          4065,    0.00%,           254,           188,          1053
             clReleaseCommandQueue,      1,          4925,    0.00%,          4925,          4925,          4925
                  clReleaseContext,      1,           646,    0.00%,           646,           646,           646
                   clReleaseDevice,      1,          1387,    0.00%,          1387,          1387,          1387
                   clReleaseKernel,      1,         22826,    0.00%,         22826,         22826,         22826
                clReleaseMemObject,      1,       1372348,    0.28%,       1372348,       1372348,       1372348
                  clReleaseProgram,      1,           904,    0.00%,           904,           904,           904

Device Performance Timing Results for Intel(R) Graphics [0x5916] (24CUs, 1100MHz):

Total Time (ns): 457021798

                                 Function Name,  Calls,     Time (ns), Time (%),  Average (ns),      Min (ns),      Max (ns)
SinJulia SIMD16 GWS[ 3840 x 2160 ] LWS[ NULL ],     16,     457021079,  100.00%,      28563817,      28467416,      28898416
                            clEnqueueMapBuffer,      1,           117,    0.00%,           117,           117,           117
                       clEnqueueUnmapMemObject,      1,           602,    0.00%,           602,           602,           602
```

The magnitude of this improvement will also be device-specific, but on this OpenCL device using `-cl-fast-relaxed-math` produced another **2-3x** improvement in performance, and the output bitmap still looks good.
Many other devices will see a similar level of improvement.

Move on to the final part 6!

## Next Step

* Part 6: [Final Words and Additional Things to Try](part6.md)
