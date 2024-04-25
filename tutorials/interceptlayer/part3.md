# Using the Intercept Layer for OpenCL Applications

## Part 3: Fixing an OpenCL Map Error

In parts 1 and 2 of the tutorial we fixed OpenCL errors and now the tutorial application should run cleanly, without any OpenCL errors.
To be sure, we can un-set the `CallLogging` control, but keep `ErrorLogging` enabled, and verify that no OpenCL errors are reported.
Because `ErrorLogging` is fairly lightweight and does not produce a lot of output for a well-behaved program it is a good control to leave enabled by default.

If the tutorial application is running without any OpenCL errors we can check the output bitmap `sinjulia.bmp`.
If the tutorial application is running on some OpenCL devices such as a CPU or integrated GPU that does not have dedicated device memory then the output bitmap may look OK, but on other OpenCL devices the output bitmap may either be empty or may contain garbage.
In this part of the tutorial we will fix a subtle OpenCL error so the tutorial application generates the correct bitmap for all OpenCL devices.

When debugging errors such as these we can check the output of our OpenCL kernels by enabling `DumpBuffersBeforeEnqueue` and `DumpBuffersAfterEnqueue`.
Be warned though, this can consume a lot of disk space!
To reduce the amount of disk space required, dumping can be constrained to a specific region of the program using `DumpBuffersMinEnqueue` and `DumpBuffersMaxEnqueue`.
For this part of the tutorial, we will enable `DumpBuffersBeforeEnqueue`, `DumpBuffersAfterEnqueue`, and we will set `DumpBuffersMaxEnqueue` to `2`.
After setting these controls and re-running the tutorial application, we won't see any additional output in our log, but we should see files with the contents of our buffers in the dump directory:

```
$ ls -R ~/CLIntercept_Dump/sinjulia/
/home/bashbaug/CLIntercept_Dump/sinjulia/:
CLI_0000_66C650B0_source.cl  CLI_0000_F05F194C_source.cl  clintercept_report.txt  Inject  memDumpPostEnqueue  memDumpPreEnqueue  Modified

/home/bashbaug/CLIntercept_Dump/sinjulia/Inject:
CLI_0000_66C650B0_source.cl

/home/bashbaug/CLIntercept_Dump/sinjulia/memDumpPostEnqueue:
Enqueue_0000_Kernel_SinJulia_Arg_0_Buffer_0000.bin  Enqueue_0001_Kernel_SinJulia_Arg_0_Buffer_0000.bin  Enqueue_0002_Kernel_SinJulia_Arg_0_Buffer_0000.bin

/home/bashbaug/CLIntercept_Dump/sinjulia/memDumpPreEnqueue:
Enqueue_0000_Kernel_SinJulia_Arg_0_Buffer_0000.bin  Enqueue_0001_Kernel_SinJulia_Arg_0_Buffer_0000.bin  Enqueue_0002_Kernel_SinJulia_Arg_0_Buffer_0000.bin

/home/bashbaug/CLIntercept_Dump/sinjulia/Modified:
CLI_0000_66C650B0_source.cl
```

Note that the contents of the OpenCL buffers are dumped as binary files, so they will need to be viewed in a hex editor or some other tool capable of decoding binary files.
If you have a tool that can decode raw image files, note that the images are currently 3847 x 2161 x 4 bytes.
Even without a raw image decoder, though, a quick glance at these files indicates that they have regular and nonzero content, so why might the output bitmap be empty or garbage?

Let's re-enable `CallLogging` and check where the buffer is transferred from the device to the host before writing to the output bitmap:

```
>>>> clEnqueueMapBuffer: [ map count = 0 ] queue = 0x560f5febe240, buffer = 0x560f5fe1e7b0, blocking, map_flags = CL_MAP_WRITE_INVALIDATE_REGION (4), offset = 0, cb = 33253468
<<<< clEnqueueMapBuffer: [ map count = 1 ] returned 0x7f8d00a67000 -> CL_SUCCESS
Wrote image file sinjulia.bmp
>>>> clEnqueueUnmapMemObject: [ map count = 1 ] queue = 0x560f5febe240, memobj = 0x560f5fe1e7b0, mapped_ptr = 0x7f8d00a67000
<<<< clEnqueueUnmapMemObject: [ map count = 0 ] -> CL_SUCCESS
```

While this is syntactically correct and does not generate any OpenCL errors, we want to map the buffer to read its contents, not to write to it, right?
Let's fix that bug.

```c++
// Part 3: Fix the map flags.
// We want to read the results of our kernel and save them to a bitmap. The
// map flags below are more typically used to initialize a buffer. What map
// flag should we use instead?
auto buf = reinterpret_cast<const uint32_t*>(
    commandQueue.enqueueMapBuffer(
        deviceMemDst,
        CL_TRUE,
        CL_MAP_WRITE_INVALIDATE_REGION,
        0,
        gwx * gwy * sizeof(cl_uchar4) ) );
```

After mapping the buffer for reading instead of writing, the output bitmap should be correct on all devices.

Move on to part 4!

## Next Step

* Part 4: [Profiling and Improving Performance](part4.md)
