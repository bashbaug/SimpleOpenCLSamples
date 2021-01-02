# Using the Intercept Layer for OpenCL Applications

## Part 3: Fixing an OpenCL Map Error

In parts 1 and 2 of the tutorial we fixed OpenCL errors and now the tutorial application _should_ run cleanly, without any OpenCL errors.
To be sure, we can un-set the `CallLogging` control, but keep `ErrorLogging` enabled, and verify that no OpenCL errors are reported.
Because `ErrorLogging` is fairly lightweight and does not produce a lot of output for a well-behaved program it is a good control to leave enabled by default.

If the tutorial application is running without any OpenCL errors we can check the output bitmap `sinjulia.bmp`.
If you are running the tutorial application on a CPU or integrated GPU OpenCL device that does not have dedicated device memory then the output bitmap may look OK, but on some devices the output bitmap may either be empty or may contain garbage.
In this part of the tutorial we will fix a subtle OpenCL error so the tutorial application runs on all OpenCL devices.

When debugging errors such as these, good controls to set are `DumpBuffersBeforeEnqueue` and `DumpBuffersAfterEnqueue`.
Be warned though, this can consume a lot of disk space!
To reduce the amount of disk space required, dumping can be constrained to a specific region of the program using `DumpBuffersMinEnqueue` and `DumpBuffersMaxEnqueue`.
For this part of the tutorial, we will enable `DumpBuffersBeforeEnqueue`, `DumpBuffersAfterEnqueue`, and we will set `DumpBuffersMaxEnqueue` to `2`.
After setting these controls and re-running the tutorial application, we won't see any additional output in our log, but we should see files with the contents of our buffers in the dump directory:

```sh
$ ls -R ~/CLIntercept_Dump/sinjulia/
/home/bashbaug/CLIntercept_Dump/sinjulia/:
CLI_0000_3DC4555B_source.cl  clintercept_report.txt  memDumpPostEnqueue  memDumpPreEnqueue

/home/bashbaug/CLIntercept_Dump/sinjulia/memDumpPostEnqueue:
Enqueue_0001_Kernel_SinJulia_Arg_0_Buffer_0000.bin  Enqueue_0002_Kernel_SinJulia_Arg_0_Buffer_0000.bin

/home/bashbaug/CLIntercept_Dump/sinjulia/memDumpPreEnqueue:
Enqueue_0001_Kernel_SinJulia_Arg_0_Buffer_0000.bin  Enqueue_0002_Kernel_SinJulia_Arg_0_Buffer_0000.bin
```

Note that the contents of the OpenCL buffers are dumped as binary files, so they will need to be viewed in a hex editor or some other tool capable of decoding binary files.
Even a quick glance at these files indicates that they have regular and nonzero content though, so why might the output bitmap be empty or garbage?

Let's re-enable `CallLogging` and check where the buffer is transferred from the device to the host before writing to the output bitmap.

```sh
>>>> clEnqueueMapBuffer: [ map count = 0 ] queue = 0x55e31e172240, buffer = 0x55e31de8a290, blocking, map_flags = CL_MAP_WRITE_INVALIDATE_REGION (4), offset = 0, cb = 33253468
<<<< clEnqueueMapBuffer: [ map count = 1 ] returned 0x7f251fa6f000 -> CL_SUCCESS
Wrote image file sinjulia.bmp
>>>> clEnqueueUnmapMemObject: [ map count = 1 ] queue = 0x55e31e172240, memobj = 0x55e31de8a290, mapped_ptr = 0x7f251fa6f000
<<<< clEnqueueUnmapMemObject: [ map count = 0 ] -> CL_SUCCESS
```

While this is syntactically correct and does not generate any OpenCL errors, we want to map the buffer to read its contents, not to write to it, right?
Let's fix that bug.

```c++
// Part 3: Fix the map flags.
// We want to read the results of our kernel and save them to a bitmap. The
// map flags below are more typically used to initialize a buffer. What map
// flag should be used instead?
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
