# usmmigratemem

## Sample Purpose

This sample demonstrates how to explicitly migrate shared memory allocations to control when and how shared memory migrations occur.

Functionally, this sample is identical to [smemhelloworld](../300_smemhelloworld/README.md), but with explicit calls to migrate the source and destination shared allocations to the device before executing the copy kernel.

## Key APIs and Concepts

This sample explicitly migrates the source and destination shared memory allocations using `clEnqueueMigrateMemINTEL`.
The source allocation preserves its contents during migration, but the device allocation may be migrated without preserving its contents using `CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED`.

The costs of the explicit migrations may be profiled using standard event profiling mechanisms.

Since Unified Shared Memory is an OpenCL extension, this sample uses the `libusm` library to query the extension APIs.
Please see the `libusm` [README](../libusm/README.md) for more detail.

This sample currently uses c APIs because the C++ bindings do not support Unified Shared Memory (yet).
When support for Unified Shared Memory is added to the C++ bindings the samples will be updated to use the C++ bindings instead, which should simplify the sample slightly.

## Command Line Options

| Option | Default Value | Description |
|:--|:-:|:--|
| `-d <index>` | 0 | Specify the index of the OpenCL device in the platform to execute on the sample on.
| `-p <index>` | 0 | Specify the index of the OpenCL platform to execute the sample on.
