#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable

kernel void Test( global ulong* dst )
{
    ulong value = get_global_id(0) << 32 | get_global_id(0);
    atom_max(dst, value);
}
