# usmqueries

## Sample Purpose

This sample queries and prints the Unified Shared Memory capabilities of a device.
Many USM samples require specific USM capabilities and this sample can be used to verify if it will or will not run on a device.

## Key APIs and Concepts

This sample demonstrates the new device queries for Unified Shared Memory capabilities.
This sample currently uses c APIs to perform the device queries because the C++ bindings do not support Unified Shared Memory (yet).
When support for Unified Shared Memory is added to the C++ bindings the samples will be updated to use the C++ bindings instead, which should simplify the sample slightly.

## Command Line Options

| Option | Default Value | Description |
|:--|:-:|:--|
| `-d <index>` | 0 | Specify the index of the OpenCL device in the platform to execute on the sample on.
| `-p <index>` | 0 | Specify the index of the OpenCL platform to execute the sample on.
