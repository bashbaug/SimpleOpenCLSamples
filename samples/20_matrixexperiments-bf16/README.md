# matrixexperiments-bf16

## Sample Purpose

This sample demonstrates various techniques to perform a large matrix multiplcation where the matrix elements contain 16-bit `bfloat16` data.
The sample includes many different implementations:

1. The "naive" implementation is a very simple implementation.
It is not very fast, but it is easy to understand, and it has no extension dependencies so it will run on many devices.
2. The "dpas" kernels use sub-group extensions to improve performance.
On some devices, they will also use specialized matrix multiplication extensions to further improve performance.
Because these kernels require certain extensions or a specific sub-group size, they may not run on all devices.
3. The "dpas blockread" kernels use additional sub-group extensions to further improve performance.

Most of the optimized kernels operate on fixed size tiles of matrix data.
For some of these kernels, parameters such as the number of matrix tiles per-sub-group or the number of sub-groups per work-group may be modified via program build options.
Experiment with different options to see what performs the best!

A good place to start for some devices is:

```sh
./matrixexperiments-bf16 -m4096 --options="-DSGS_PER_WG_X=4 -DSGS_PER_WG_Y=8 -DKK=2 -cl-intel-256-GRF-per-thread" --zero
```

## Key APIs and Concepts

This sample will optionally use the following OpenCL extensions:

* cl_intel_bfloat16_conversions
* cl_intel_required_subgroup_size
* cl_intel_split_work_group_barrier
* cl_intel_subgroup_2d_block_io
* cl_intel_subgroup_matrix_multiply_accumulate
* cl_intel_subgroups
* cl_intel_subgroups_short

## Command Line Options

| Option | Default Value | Description |
|:--|:-:|:--|
| `-p <index>` | 0 | Specify the index of the OpenCL platform to execute the sample on.
| `-d <index>` | 0 | Specify the index of the OpenCL device in the platform to execute on the sample on.
| `--file <string>` | `matrix_kernels_bf16.cl` | Specify the name of the file with the OpenCL kernel source.
| `--options <string>` | None | Specify optional program build options.
| `--matrixsize <int>` | 512 | Specify the dimensions of the matrix.
| `--iterations <int>` | 16 | Specify the number of iterations for performance testing.
| `--validate` | n/a | Validate results for correctness.
| `--zero` | n/a | Initialize all matrices to zero.
| `--identity` | n/a | Initialize all matrices to to one.
| `--fixed` | n/a | Initialize all matrices to values computed from the matrix row and column.
| `--emulate` | n/a | Do not use specialized matrix multiplication extensions.
| `--wallclock` | n/a | Measure performance using wallclock time instead of event profiling.
| `--skipinit` | n/a | Skip initialization of source matrices.
| `--roundrobin` | n/a | Use round robin thread scheduling.
| `--threshold <float>` | 0.01 | Set the threshold used when validating results.
| `--mask <int>` | ~0 | Set a mask to only run a subset of tests.

By default, the source matrices are populated with random data.
When validating results, it is recommended to use either "fixed" or "identity" data.
For best performance, use "zero" data".
