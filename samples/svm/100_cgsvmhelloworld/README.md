# cgsvmhelloworld

## Sample Purpose

This is the first Shared Virtual Memory (SVM) sample that meaningfully stores and uses data in a Shared Virtual Memory allocation.
This sample demonstrates usage of coarse-grained SVM allocations.
Other similar samples demonstrate usage of fine-grained SVM allocations.
This sample may not run on all OpenCL devices because SVM is an optional feature, though many devices do support coarse-grained SVM.

The sample initializes a coarse-grained SVM allocation, copies it to a destination coarse-grained SVM allocation using a kernel, then checks on the host that the copy was performed correctly.

## Key APIs and Concepts

This sample allocates coarse-grained SVM memory using `clSVMAlloc` and frees it using `clSVMFree`.

Since coarse-grained SVM cannot be directly accessed by the host, this sample initializes the source allocation by mapping it using `clEnqueueSVMMap`.
This sample also uses `clEnqueueSVMMap` to map the destination buffer to verify that the copy was performed correctly.

Within a kernel, a Shared Virtual Memory allocation can be accessed similar to an OpenCL buffer (a `cl_mem`).
Shared Virtual Memory allocations are set as an argument to a kernel using `clSetKernelArgSVMPointer`.

## Command Line Options

| Option | Default Value | Description |
|:--|:-:|:--|
| `-d <index>` | 0 | Specify the index of the OpenCL device in the platform to execute on the sample on.
| `-p <index>` | 0 | Specify the index of the OpenCL platform to execute the sample on.
