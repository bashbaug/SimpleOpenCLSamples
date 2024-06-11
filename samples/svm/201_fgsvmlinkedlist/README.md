# fgsvmlinkedlist

## Sample Purpose

This sample demonstrates how to build a linked list on the host using fine-grained Shared Virtual Memory (SVM) allocations, how to access and modify the linked list in a kernel, then how to access and check the contents of the linked list on the host.

Because fine-grained SVM does not require any API calls to access the contents of an allocation on the host, this sample is much simpler than the coarse-grained SVM sample.

## Key APIs and Concepts

This sample only needs to ensure the device is not accessing the fine-grained SVM allocation before initializing the contents of the source allocation or verifying that the copy was performed correctly.
For simplicity, this sample calls `clFinish` to ensure all execution is complete on the device.

This sample also demonstrates how to specifying a set of indirectly accessed SVM pointers using `clSetKernelExecInfo` and `CL_KERNEL_EXEC_INFO_SVM_PTRS`.
This is still required for kernels that operate on complex data structures consisting of fine-grained Shared Virtual Memory allocations that are not directly passed as kernel arguments.

## Command Line Options

| Option | Default Value | Description |
|:--|:-:|:--|
| `-d <index>` | 0 | Specify the index of the OpenCL device in the platform to execute on the sample on.
| `-p <index>` | 0 | Specify the index of the OpenCL platform to execute the sample on.
| `-n <number>` | 4 | Specify the number of linked list nodes to create.
