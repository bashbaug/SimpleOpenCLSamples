# Multithreading

## Sample Purpose

This is an intermediate-level sample that demonstrates and tests OpenCL behavior when multiple host threads enqueue work to independent command-queues simultaneously.

The main thread sets up the sample by creating an OpenCL context and building an OpenCL program.
The main thread then spawns a configurable number of worker threads.

Each worker thread creates its own OpenCL command-queue, OpenCL buffer, and OpenCL kernel.
Each thread then repeatedly performs one of the following operations, chosen at random:

* Enqueue the kernel (70% probability).
* Flush the command-queue (10% probability).
* Finish the command-queue (10% probability).
* Read the result out of the buffer and check results (10% probability).

After the threads have run for a configurable amount of time, the main thread signals the worker threads to stop, waits for them to exit, and the application exits.

## Key APIs and Concepts

This sample demonstrates that OpenCL API calls are thread-safe and that multiple host threads may share the same OpenCL context and OpenCL program, so long as each thread uses its own command-queue and its own kernel object.

## Command Line Options

| Option | Default Value | Description |
|:--|:-:|:--|
| `-p <platform>` | 0 | Specify the index of the OpenCL platform to use. |
| `-d <device>` | 0 | Specify the index of the OpenCL device in the platform to use. |
| `-t <threads>` | 8 | Specify the number of worker threads to spawn. |
| `-s <seconds>` | 1 | Specify the number of seconds to run the worker threads. |
| `-e` | N/A | Create (and immediately destroy) an OpenCL event for each kernel enqueue. |
