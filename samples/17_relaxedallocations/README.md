# Relaxed Allocation Limits

## Sample Purpose

TODO

## Key APIs and Concepts

```
CL_MEM_ALLOW_UNRESTRICTED_SIZE_INTEL
-cl-intel-greater-than-4GB-buffer-required
```

## Command Line Options

| Option | Default Value | Description |
|:--|:-:|:--|
| `-d <index>` | 0 | Specify the index of the OpenCL device in the platform to execute on the sample on.
| `-p <index>` | 0 | Specify the index of the OpenCL platform to execute the sample on.
| `-s <number>` | 2 | Size to allocate in GB.
| `--svm` | N/A | Test USM allocations.
| `--usm` | N/A | Test SVM allocations.
