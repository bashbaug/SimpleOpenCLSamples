# Unified Shared Memory Samples

This directory contains samples demonstrating Unified Shared Memory (USM).
Unified Shared Memory is intended to bring pointer-based programming to OpenCL, and is an alternative to OpenCL 2.0 Shared Virtual Memory (SVM).

## Unified Shared Memory Extension Status

The `cl_intel_unified_shared_memory` extension is a widely supported vendor extension.
The latest Unified Shared Memory extension specification can be found [here](https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_unified_shared_memory.html).

These samples require the OpenCL Extension Loader to get the extension APIs for Unified Shared Memory.

## Unified Shared Memory Advantages

Unified Shared Memory (USM) provides:

* Easier integration into existing code bases by representing OpenCL memory allocations as pointers rather than handles (`cl_mems`), with full support for pointer arithmetic into allocations.

* Fine-grain control over ownership and accessibility of OpenCL memory allocations, to optimally choose between performance and programmer convenience.

* A simpler programming model, by automatically migrating some memory allocations between OpenCL devices and the host.

Compared to Shared Virtual Memory (SVM), Unified Shared Memory provides:

* A similar pointer-based representation of memory allocations.

* A similar address equivalence for pointers to allocations on the host and the device.

* No need to map or unmap any USM allocations, similar to fine grain SVM allocations.

* No need to specify all of the allocations used by an OpenCL kernel, similar to fine grain system SVM allocations.

* More control over the initial placement of a memory allocation, and where a memory allocation is able to migrate.

* The ability to pass other implementation-specific properties during allocation.

* The ability to provide implementation-specific memory advice for some or all of a memory allocation, after allocation.

* The ability to query information about a memory allocation.

## Summary of Unified Shared Memory Samples

* [usmqueries](./00_usmqueries): Queries and prints the USM capabilities of a device.
* [usmmeminfo](./01_usmmeminfo): Allocates and queries properties of a USM allocation.
* [dmemhelloworld](./100_dmemhelloworld): Copy one "device" memory allocation to another.
* [dmemlinkedlist](./101_dmemlinkedlist): Create and modify a linked list in "device" memory.
* [hmemhelloworld](./200_hmemhelloworld): Copy one "host" memory allocation to another.
* [hmemlinkedlist](./201_hmemlinkedlist): Create and modify a linked list in "host" memory.
* [smemhelloworld](./300_smemhelloworld): Copy one "shared" memory allocation to another.
* [smemlinkedlist](./301_smemlinkedlist): Create and modify a linked list in "shared" memory.
* [usmmigratemem](./310_usmmigratemem): Copy one "shared" memory allocation to another, with explicit calls to migrate memory.