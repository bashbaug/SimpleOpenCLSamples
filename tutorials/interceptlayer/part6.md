# Using the Intercept Layer for OpenCL Applications

## Part 6: Final Words and Additional Things to Try

If you have made it this far, great work!
We started with an OpenCL application that crashed and used the Intercept Layer for OpenCL Applications identify and fix the bugs that were preventing it from running correctly, then to profile the application to significantly improve its performance.

After fixing all of the bugs and making the suggested performance improvements, we should be able to execute the tutorial application without the Intercept Layer now and it will still run well:

```
$ ./sinjulia
Running on platform: Intel(R) OpenCL HD Graphics
Running on device: Intel(R) Graphics [0x5916]
Executing the kernel 16 times
Global Work Size = ( 3840, 2160 )
Local work size = NULL
Finished in 0.513813 seconds
Wrote image file sinjulia.bmp
```

This is the "official" end of the tutorial, but if you are looking for some additional things to try, either to explore the capabilities of the Intercept Layer for OpenCL Applications or to experiment with fractal images, here are a few suggestions:

* Visually trace the tutorial application using [Chrome Tracing](https://github.com/intel/opencl-intercept-layer/blob/master/docs/chrome_tracing.md).
    * How does the trace change if you enable the `FinishAfterEnqueue` control?
* Modify the tutorial application to output to an OpenCL image instead of to an OpenCL buffer.
    * Dump the output image using `DumpImagesBeforeEnqueue` and `DumpImagesAfterEnqueue`.
    * Does the version that outputs to an OpenCL image perform better or worse than the version that outputs to an OpenCL buffer?
* Write the tutorial application in [SYCL](https://www.khronos.org/sycl/) instead of using OpenCL directly.
    * Does the SYCL version generate similar OpenCL calls as the direct OpenCL version?
    * Does the SYCL version perform similarly to the direct OpenCL version?  If not, can you determine why it doesn't?
* Modify the tutorial application to generate different fractal images.
    * Choose a different complex constant `c` by changing the values of `cr` and `ci`.
    * Choose a different iteration function.
      Note that the tutorial application currently uses `f(z) = c * sin(z)` as its iteration function.
      Other common iteration functions can be found on Paul Bourke's site [here](http://paulbourke.net/fractals/juliaset/).
      Remember that the inputs to these functions are complex numbers!
    * Use a different range of inputs.
      Note that the tutorial application currently goes from `-pi/2` to `+pi/2` for both the real and imaginary axes.
    * Map the result value onto a different color or onto multiple colors.
