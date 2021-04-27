# Queue Experiments

## Sample Purpose

This is an intermediate-level sample that performs several experiments using different command-queue properties or numbers of command-queues to execute independent ND-range kernels as efficiently as possible.
The ND-range kernel used in this sample is contrived to maximize opportunity for concurrent execution - only a single work-item is launched and the kernel itself simply wastes time in a tight loop - but the patterns used in this sample can be used to improve the performance of some real-world applications with opportunities for concurrent execution.

Several patterns are tested in this sample:

1. The first pattern executes one, two, and four independent ND-range kernels in an in-order command-queue.
Since an in-order command-queue requires the previous ND-range kernel to complete before the next ND-range kernel can start, two identical but independent ND-range kernels should take twice as long to execute as one ND-range kernel, and four ND-range kernels should take four times as long to execute.
This pattern requires no special OpenCL capabilities and should execute on all OpenCL devices.
2. The second pattern executes one, two, and four independent ND-range kernels in an out-of-order command-queue.
Since commands in an out-of-order command-queue can execute in any order and may overlap execution, when independent commands have no dependencies between them using an out-of-order command-queue may allow the commands to execute concurrently.
If the independent ND-range kernels execute concurrently on different physical hardware then multiple ND-range kernels may execute in the same time that it takes to execute one ND-range kernel.
This pattern requires the out-of-order command-queue capability and may not execute on all OpenCL devices.
3. The third pattern is similar to the second pattern in that it executes independent ND-range kernels in an out-of-order command-queue, but it uses events to synchronize execution.
4. The fourth pattern executes independent ND-range kernels using multiple in-order command-queues.
This pattern requires no special OpenCL capabilities and should execute on all OpenCL devices.
Some devices that do not support out-of-order command-queues can still execute commands concurrently using this pattern.

## Key APIs and Concepts

This sample demonstrates how to specify different command-queue properties and how to synchronize within and across command-queues.

```c
CL_DEVICE_QUEUE_PROPERTIES
CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE
clCreateCommandQueue / clCreateCommandQueueWithProperties
clFinish
clWaitForEvents
```

## Hints, Tips, and Other Comments

* Some devices do not support concurrent execution using any of these patterns.
This is not a bug!
Many parts of OpenCL _allow_ for concurrent execution, but very few parts of OpenCL _require_ concurrent execution.
This is not OpenCL-specific and is also true for other data-parallel programming models.

* If results are inconsistent, try to ensure no other processes are using the OpenCL device.
For GPU OpenCL devices this may require running the application in a console or terminal window rather from a desktop GUI.

## Command Line Options

| Option | Default Value | Description |
|:--|:-:|:--|
| `-d <index>` | 0 | Specify the index of the OpenCL device in the platform to execute on the sample on.
| `-p <index>` | 0 | Specify the index of the OpenCL platform to execute the sample on.
| `-k <number>` | 8 | Specify the number of kernels to execute for the variable execution.
| `-i <number>` | 1 | Specify the number of kernel iterations to execute.
| `-e <number>` | 1 | Specify the number of ND-range elements to execute (the global work size).
