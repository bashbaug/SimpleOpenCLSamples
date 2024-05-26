# fgsvmhelloworld

## Sample Purpose

This sample demonstrates usage of fine-grained Shared Virtual Memory (SVM) allocations.
This sample may not run on all OpenCL devices because many devices do not support fine-grained SVM.

The sample initializes a fine-grained SVM allocation, copies it to a destination coarse-grained SVM allocation using a kernel, then checks on the host that the copy was performed correctly.
Because fine-grained SVM does not require any API calls to access the contents of an allocation on the host, this sample is much simpler than the coarse-grained SVM sample.

## Key APIs and Concepts

This sample allocates fine-grained SVM memory using `clSVMAlloc` and frees it using `clSVMFree`.

This sample only needs to ensure the device is not accessing the fine-grained SVM allocation before initializing the contents of the source allocation or verifying that the copy was performed correctly.
For simplicity, this sample calls `clFinish` to ensure all execution is complete on the device.

Within a kernel, a Shared Virtual Memory allocation can be accessed similar to an OpenCL buffer (a `cl_mem`).
Shared Virtual Memory allocations are set as an argument to a kernel using `clSetKernelArgSVMPointer`.

## Command Line Options

| Option | Default Value | Description |
|:--|:-:|:--|
| `-d <index>` | 0 | Specify the index of the OpenCL device in the platform to execute on the sample on.
| `-p <index>` | 0 | Specify the index of the OpenCL platform to execute the sample on.
