# cgsvmlinkedlist

## Sample Purpose

This sample demonstrates how to build a linked list on the host using coarse-grained Shared Virtual Memory (SVM) allocations, how to access and modify the linked list in a kernel, then how to access and check the contents of the linked list on the host.

Because device coarse-grained SVM cannot be directly read from or written to on the host, this example constructs and verifies the linked list using explicit memory copies.

## Key APIs and Concepts

This sample demonstrates how to use `clEnqueueSVMMemcpy` to explicitly copy between a Shared Virtual Memory allocation and an allocation on the host.

This sample also demonstrates how to specifying a set of indirectly accessed SVM pointers using `clSetKernelExecInfo` and `CL_KERNEL_EXEC_INFO_SVM_PTRS`.
This is required for kernels that operate on complex data structures consisting of Shared Virtual Memory allocations that are not directly passed as kernel arguments.

## Command Line Options

| Option | Default Value | Description |
|:--|:-:|:--|
| `-d <index>` | 0 | Specify the index of the OpenCL device in the platform to execute on the sample on.
| `-p <index>` | 0 | Specify the index of the OpenCL platform to execute the sample on.
| `-n <number>` | 4 | Specify the number of linked list nodes to create.
