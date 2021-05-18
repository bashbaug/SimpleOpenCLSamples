# enumqueuefamilies

## Sample Purpose

This is a simple sample that queries and prints the command queue families that are supported by all OpenCL devices that support the command queue families extension.

Command queue families are described in the OpenCL extension [cl_intel_command_queue_families](https://www.khronos.org/registry/OpenCL/extensions/intel/cl_intel_command_queue_families.html).
This is an optional extension and some devices may not support command queue families.

For each command queue family, the sample prints:

* The name of the command queue family.
* The number of command queues in the command queue family.
* The command queue properties supported by command queues in the command queue family.
* The supported command queue capabilities for command queues in the command queue family.

## Key APIs and Concepts

This sample demonstrates the new device query `CL_DEVICE_QUEUE_FAMILY_PROPERTIES_INTEL`.
