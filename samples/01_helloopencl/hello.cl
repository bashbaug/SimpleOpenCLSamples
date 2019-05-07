#if defined(cl_khr_fp64)
#  pragma OPENCL EXTENSION cl_khr_fp64: enable
#elif defined(cl_amd_fp64)
#  pragma OPENCL EXTENSION cl_amd_fp64: enable
#else
#  error double precision is not supported
#endif

kernel void hello_kernel(global const double *a, global const double *b, global double *c) {
	size_t gid = get_global_id(0);

	// do computation
	// in this trivial kernel, the runtime is dominated by address offset computations;
	// each add incurs the cost of three additional adds for the pointer math
	// that underlies each array access
	c[gid] = a[gid] + b[gid];
};