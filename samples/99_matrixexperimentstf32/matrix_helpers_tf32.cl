#if defined(cl_intel_subgroups)

inline int compute_m(const int num_sgs, const int tM, const int MM)
{
    const int m_start = get_group_id(1) * num_sgs;
    const int m_index = num_sgs > 1 ? m_start + get_sub_group_id() : m_start;
    return m_index * tM * MM;
}

// Emulated dpas:
__attribute__((overloadable))
float emu_sub_group_tf32_tf32_matrix_mad_k8(float  a, float8 b, float  acc)
{
    float res = acc;

#if 1
    res = fma(sub_group_broadcast(a, 0), b.s0, res);
    res = fma(sub_group_broadcast(a, 1), b.s1, res);
    res = fma(sub_group_broadcast(a, 2), b.s2, res);
    res = fma(sub_group_broadcast(a, 3), b.s3, res);
    res = fma(sub_group_broadcast(a, 4), b.s4, res);
    res = fma(sub_group_broadcast(a, 5), b.s5, res);
    res = fma(sub_group_broadcast(a, 6), b.s6, res);
    res = fma(sub_group_broadcast(a, 7), b.s7, res);
#else
float  __attribute__((overloadable)) intel_sub_group_tf32_tf32_matrix_mad_k8_f32(short  a, int8 b, float  acc);
    uint    a_ui = as_uint(sub_group_shuffle(a, get_sub_group_local_id() / 2));
    short   aData = get_sub_group_local_id() % 2 ? as_short2(a_ui).hi : as_short2(a_ui).lo;
    res = intel_sub_group_tf32_tf32_matrix_mad_k8_f32(aData, as_int8(b), res);
#endif

    return res;
}

__attribute__((overloadable))
float2 emu_sub_group_tf32_tf32_matrix_mad_k8(float a, float8 b, float2 acc)
{
    float2 res = acc;

    res.s0 = fma(sub_group_broadcast(a,  0), b.s0, res.s0);
    res.s0 = fma(sub_group_broadcast(a,  1), b.s1, res.s0);
    res.s0 = fma(sub_group_broadcast(a,  2), b.s2, res.s0);
    res.s0 = fma(sub_group_broadcast(a,  3), b.s3, res.s0);
    res.s0 = fma(sub_group_broadcast(a,  4), b.s4, res.s0);
    res.s0 = fma(sub_group_broadcast(a,  5), b.s5, res.s0);
    res.s0 = fma(sub_group_broadcast(a,  6), b.s6, res.s0);
    res.s0 = fma(sub_group_broadcast(a,  7), b.s7, res.s0);

    res.s1 = fma(sub_group_broadcast(a,  8), b.s0, res.s1);
    res.s1 = fma(sub_group_broadcast(a,  9), b.s1, res.s1);
    res.s1 = fma(sub_group_broadcast(a, 10), b.s2, res.s1);
    res.s1 = fma(sub_group_broadcast(a, 11), b.s3, res.s1);
    res.s1 = fma(sub_group_broadcast(a, 12), b.s4, res.s1);
    res.s1 = fma(sub_group_broadcast(a, 13), b.s5, res.s1);
    res.s1 = fma(sub_group_broadcast(a, 14), b.s6, res.s1);
    res.s1 = fma(sub_group_broadcast(a, 15), b.s7, res.s1);

    return res;
}

__attribute__((overloadable))
float4 emu_sub_group_tf32_tf32_matrix_mad_k8(float2 a, float8 b, float4 acc)
{
    float4 res;

    res.s01 = emu_sub_group_tf32_tf32_matrix_mad_k8(a.s0, b, acc.s01);
    res.s23 = emu_sub_group_tf32_tf32_matrix_mad_k8(a.s1, b, acc.s23);

    return res;
}

__attribute__((overloadable))
float8 emu_sub_group_tf32_tf32_matrix_mad_k8(float4 a, float8 b, float8 acc)
{
    float8 res;

    res.s01 = emu_sub_group_tf32_tf32_matrix_mad_k8(a.s0, b, acc.s01);
    res.s23 = emu_sub_group_tf32_tf32_matrix_mad_k8(a.s1, b, acc.s23);
    res.s45 = emu_sub_group_tf32_tf32_matrix_mad_k8(a.s2, b, acc.s45);
    res.s67 = emu_sub_group_tf32_tf32_matrix_mad_k8(a.s3, b, acc.s67);

    return res;
}

// M rows x K columns
float load_a_rowmajor_d32_m1_k8_sg16(global float* A, int rowStart, int colStart, int stride)
{
    float ret;

    uint offset = rowStart * stride + colStart;
    offset += (get_sub_group_local_id() < 8) ? 0 : stride;
    offset += (get_sub_group_local_id() % 8);

    ret = A[offset];

    return ret;
}

// M rows x K columns
float load_a_rowmajor_d32_m2_k8_sg16(global float* A, int rowStart, int colStart, int stride)
{
    float ret;

    uint offset = rowStart * stride + colStart;
    offset += (get_sub_group_local_id() < 8) ? 0 : stride;
    offset += (get_sub_group_local_id() % 8);

    ret = A[offset];

    return ret;
}

// M rows x K columns
float2 load_a_rowmajor_d32_m4_k8_sg16(global float* A, int rowStart, int colStart, int stride)
{
    float2 ret;

    uint offset = rowStart * stride + colStart;
    offset += (get_sub_group_local_id() < 8) ? 0 : stride;
    offset += (get_sub_group_local_id() % 8);

    ret.s0 = A[offset]; offset += stride * 2;
    ret.s1 = A[offset]; offset += stride * 2;

    return ret;
}

// M rows x K columns
float4 load_a_rowmajor_d32_m8_k8_sg16(global float* A, int rowStart, int colStart, int stride)
{
    float4 ret;

    uint offset = rowStart * stride + colStart;
    offset += (get_sub_group_local_id() < 8) ? 0 : stride;
    offset += (get_sub_group_local_id() % 8);

    ret.s0 = A[offset]; offset += stride * 2;
    ret.s1 = A[offset]; offset += stride * 2;
    ret.s2 = A[offset]; offset += stride * 2;
    ret.s3 = A[offset]; offset += stride * 2;

    return ret;
}

// M rows x K columns x V tiles (in the M and K dimensions)
void prefetch_a_rowmajor_d32_m8v2_k8v2_sg16(global float* A, int rowStart, int colStart, int stride)
{
#if defined(PREFETCH_DEFAULT)
    uint offset = colStart + (rowStart + get_sub_group_local_id()) * stride;
    prefetch(A + offset, 1);
#endif // defined(PREFETCH_DEFAULT)
}

// K rows x N columns:
// Each work-item loads K values.
// Stride is in units of elements.
float8 load_b_rowmajor_d32_k8_nx(global float* B, int rowStart, int colStart, int stride)
{
    float8 ret;

    uint offset = rowStart * stride + colStart;

    ret.s0 = as_float(intel_sub_group_block_read((global uint*)B + offset)); offset += stride;
    ret.s1 = as_float(intel_sub_group_block_read((global uint*)B + offset)); offset += stride;
    ret.s2 = as_float(intel_sub_group_block_read((global uint*)B + offset)); offset += stride;
    ret.s3 = as_float(intel_sub_group_block_read((global uint*)B + offset)); offset += stride;
    ret.s4 = as_float(intel_sub_group_block_read((global uint*)B + offset)); offset += stride;
    ret.s5 = as_float(intel_sub_group_block_read((global uint*)B + offset)); offset += stride;
    ret.s6 = as_float(intel_sub_group_block_read((global uint*)B + offset)); offset += stride;
    ret.s7 = as_float(intel_sub_group_block_read((global uint*)B + offset)); offset += stride;

    return ret;
}

// K rows x N columns x V tiles (in the K and N dimensions)
void prefetch_b_rowmajor_d32_k8v2_n8v2_sg16(global float* B, int rowStart, int colStart, int stride)
{
#if defined(PREFETCH_DEFAULT)
    uint offset = colStart + (rowStart + get_sub_group_local_id()) * stride;
    prefetch(B + offset, 1);
#endif // defined(PREFETCH_DEFAULT)
}

void store_c_rowmajor_fp32_m1_nx(global float* C, float v, int rowStart, int colStart, int stride)
{
    global uint* C_ui = (global uint*)C;
    uint v_ui = as_uint(v);

    uint offset = rowStart * stride + colStart;

    intel_sub_group_block_write(C_ui + offset, v_ui); offset += stride;
}

void store_c_rowmajor_fp32_m2_nx(global float* C, float2 v, int rowStart, int colStart, int stride)
{
    global uint* C_ui = (global uint*)C;
    uint2 v_ui = as_uint2(v);

    uint offset = rowStart * stride + colStart;

    intel_sub_group_block_write(C_ui + offset, v_ui.s0); offset += stride;
    intel_sub_group_block_write(C_ui + offset, v_ui.s1); offset += stride;
}

void store_c_rowmajor_fp32_m4_nx(global float* C, float4 v, int rowStart, int colStart, int stride)
{
    global uint* C_ui = (global uint*)C;
    uint4 v_ui = as_uint4(v);

    uint offset = rowStart * stride + colStart;

    intel_sub_group_block_write(C_ui + offset, v_ui.s0); offset += stride;
    intel_sub_group_block_write(C_ui + offset, v_ui.s1); offset += stride;
    intel_sub_group_block_write(C_ui + offset, v_ui.s2); offset += stride;
    intel_sub_group_block_write(C_ui + offset, v_ui.s3); offset += stride;
}

void store_c_rowmajor_fp32_m8_nx(global float* C, float8 v, int rowStart, int colStart, int stride)
{
    global uint* C_ui = (global uint*)C;
    uint8 v_ui = as_uint8(v);

    uint offset = rowStart * stride + colStart;

    intel_sub_group_block_write(C_ui + offset, v_ui.s0); offset += stride;
    intel_sub_group_block_write(C_ui + offset, v_ui.s1); offset += stride;
    intel_sub_group_block_write(C_ui + offset, v_ui.s2); offset += stride;
    intel_sub_group_block_write(C_ui + offset, v_ui.s3); offset += stride;
    intel_sub_group_block_write(C_ui + offset, v_ui.s4); offset += stride;
    intel_sub_group_block_write(C_ui + offset, v_ui.s5); offset += stride;
    intel_sub_group_block_write(C_ui + offset, v_ui.s6); offset += stride;
    intel_sub_group_block_write(C_ui + offset, v_ui.s7); offset += stride;
}

#endif // defined(cl_intel_subgroups)

#ifdef cl_intel_subgroup_extended_block_read

// Note for 2D block reads:
//  - the tile width and height is encoded into the function name.
//  - base_address is the byte address.  Must be 64B aligned.
//  - width is the width of the entire matrix, in bytes.  Must be >= 64B.  Must be 4B aligned.
//  - height is the height of the entire matrix, or equivalently the number of rows.
//  - pitch is the number of bytes between rows of the entire matrix.  Must be >= 64B.  Must be a multiple of 8 bytes.
//  - coord is the number of elements (x coord) and row (y coord) to read from.  X coord must be multiple 4 for for 1B data and 2 for 2B data.

// Built-in functions are:

// #ifdef cl_intel_subgroup_extended_block_read
// ushort2  intel_subgroup_block_read_u8_m1k32v2(__global void *base_address, int width, int height, int pitch, int2 coord);
// ushort4  intel_subgroup_block_read_u8_m2k32v2(__global void *base_address, int width, int height, int pitch, int2 coord);
// ushort8  intel_subgroup_block_read_u8_m4k32v2(__global void *base_address, int width, int height, int pitch, int2 coord);
// ushort16 intel_subgroup_block_read_u8_m8k32v2(__global void *base_address, int width, int height, int pitch, int2 coord);
// ushort2  intel_subgroup_block_read_u16_m1k16v2(__global void *base_address, int width, int height, int pitch, int2 coord);
// ushort4  intel_subgroup_block_read_u16_m2k16v2(__global void *base_address, int width, int height, int pitch, int2 coord);
// ushort8  intel_subgroup_block_read_u16_m4k16v2(__global void *base_address, int width, int height, int pitch, int2 coord);
// ushort16 intel_subgroup_block_read_u16_m8k16v2(__global void *base_address, int width, int height, int pitch, int2 coord);
// uint8    intel_subgroup_block_read_transform_u8_k32(__global void *base_address, int width, int height, int pitch, int2 coord);
// uint8    intel_subgroup_block_read_transform_u16_k16(__global void *base_address, int width, int height, int pitch, int2 coord);
// uint8    intel_subgroup_block_read_transpose_u32_k8(__global void *base_address, int width, int height, int pitch, int2 coord);
// ulong4   intel_subgroup_block_read_transpose_u64_k4(__global void *base_address, int width, int height, int pitch, int2 coord);
// #endif //defined(cl_intel_subgroup_extended_block_read)


// For intrinsics, the pattern is:
//  - prefix: __builtin_IB_subgroup_block_read_flat or __builtin_IB_subgroup_block_write_flat
//  - operation (optional): _transpose or _transform
//  - for no transpose or transform:
//      - type / elements size: _u8 or _u16 or _u32 or _u64
//      - number of tile rows: _m32 or _m16 or _m8 or _m4 or _m2 or _m1
//      - tile width: _k64 or _k32 or _k16 or _k8
//      - number of tiles: _v2 or _v1
//  - for transpose:
//      - type / element size: _u64 or _u32
//      - number of tile rows: subgroup size (16)
//      - tile width: _k4 (for _u64) or _k8 (for _u32)
//      - number of tiles: 1
//  - for transform:
//      - type / element size: _u16 or _u8
//      - number of tile rows: _k32 (for _u8) or _k16 (for _u16)
//      - tile width: subgroup size (16)
//      - number of tiles: 1

// Define additional "non-vector" block read and writes.  These are supported by the hardware but are not in the headers:

uint   __builtin_IB_subgroup_block_read_flat_u32_m1k8v1(long baseoffset, int width_minus_one, int height_minus_one, int pitch_minus_one, int2 coord);
uint2  __builtin_IB_subgroup_block_read_flat_u32_m2k8v1(long baseoffset, int width_minus_one, int height_minus_one, int pitch_minus_one, int2 coord);
uint4  __builtin_IB_subgroup_block_read_flat_u32_m4k8v1(long baseoffset, int width_minus_one, int height_minus_one, int pitch_minus_one, int2 coord);
uint8  __builtin_IB_subgroup_block_read_flat_u32_m8k8v1(long baseoffset, int width_minus_one, int height_minus_one, int pitch_minus_one, int2 coord);

uint   __builtin_IB_subgroup_block_read_flat_u32_m1k16v1(long baseoffset, int width_minus_one, int height_minus_one, int pitch_minus_one, int2 coord);
uint2  __builtin_IB_subgroup_block_read_flat_u32_m2k16v1(long baseoffset, int width_minus_one, int height_minus_one, int pitch_minus_one, int2 coord);
uint4  __builtin_IB_subgroup_block_read_flat_u32_m4k16v1(long baseoffset, int width_minus_one, int height_minus_one, int pitch_minus_one, int2 coord);
uint8  __builtin_IB_subgroup_block_read_flat_u32_m8k16v1(long baseoffset, int width_minus_one, int height_minus_one, int pitch_minus_one, int2 coord);

uint   intel_subgroup_block_read_u32_m1k8(const __global void *base_address, int width, int height, int pitch, int2 coord)
{
    return __builtin_IB_subgroup_block_read_flat_u32_m1k8v1(as_long(base_address), width - 1, height - 1, pitch - 1, coord);
}
uint  intel_subgroup_block_read_u32_m2k8(const __global void *base_address, int width, int height, int pitch, int2 coord)
{
    return __builtin_IB_subgroup_block_read_flat_u32_m2k8v1(as_long(base_address), width - 1, height - 1, pitch - 1, coord).lo;
}
uint2  intel_subgroup_block_read_u32_m4k8(const __global void *base_address, int width, int height, int pitch, int2 coord)
{
    return __builtin_IB_subgroup_block_read_flat_u32_m4k8v1(as_long(base_address), width - 1, height - 1, pitch - 1, coord).lo;
}
uint4  intel_subgroup_block_read_u32_m8k8(const __global void *base_address, int width, int height, int pitch, int2 coord)
{
    return __builtin_IB_subgroup_block_read_flat_u32_m8k8v1(as_long(base_address), width - 1, height - 1, pitch - 1, coord).lo;
}

uint   intel_subgroup_block_read_u32_m1k16(const __global void *base_address, int width, int height, int pitch, int2 coord)
{
    return __builtin_IB_subgroup_block_read_flat_u32_m1k16v1(as_long(base_address), width - 1, height - 1, pitch - 1, coord);
}
uint2  intel_subgroup_block_read_u32_m2k16(const __global void *base_address, int width, int height, int pitch, int2 coord)
{
    return __builtin_IB_subgroup_block_read_flat_u32_m2k16v1(as_long(base_address), width - 1, height - 1, pitch - 1, coord);
}
uint4  intel_subgroup_block_read_u32_m4k16(const __global void *base_address, int width, int height, int pitch, int2 coord)
{
    return __builtin_IB_subgroup_block_read_flat_u32_m4k16v1(as_long(base_address), width - 1, height - 1, pitch - 1, coord);
}
uint8  intel_subgroup_block_read_u32_m8k16(const __global void *base_address, int width, int height, int pitch, int2 coord)
{
    return __builtin_IB_subgroup_block_read_flat_u32_m8k16v1(as_long(base_address), width - 1, height - 1, pitch - 1, coord);
}

uint8  __builtin_IB_subgroup_block_read_flat_u32_m8k8v2(long baseoffset, int width_minus_one, int height_minus_one, int pitch_minus_one, int2 coord);

uint8 intel_subgroup_block_read_u32_m8k8v2(const __global void* base_address, int width, int height, int pitch, int2 coord)
{
    return __builtin_IB_subgroup_block_read_flat_u32_m8k8v2(as_long(base_address), width - 1, height - 1, pitch - 1, coord);
}

void __builtin_IB_subgroup_block_write_flat_u32_m1k16v1(long baseoffset, int width_minus_one, int height_minus_one, int pitch_minus_one, int2 coord, uint  data);
void __builtin_IB_subgroup_block_write_flat_u32_m2k16v1(long baseoffset, int width_minus_one, int height_minus_one, int pitch_minus_one, int2 coord, uint2 data);
void __builtin_IB_subgroup_block_write_flat_u32_m4k16v1(long baseoffset, int width_minus_one, int height_minus_one, int pitch_minus_one, int2 coord, uint4 data);
void __builtin_IB_subgroup_block_write_flat_u32_m8k16v1(long baseoffset, int width_minus_one, int height_minus_one, int pitch_minus_one, int2 coord, uint8 data);

void intel_subgroup_block_write_u32_m1k16(__global void* base_address, int width, int height, int pitch, int2 coord, uint data)
{
    __builtin_IB_subgroup_block_write_flat_u32_m1k16v1(as_long(base_address), width - 1, height - 1, pitch - 1, coord, data);
}
void intel_subgroup_block_write_u32_m2k16(__global void* base_address, int width, int height, int pitch, int2 coord, uint2 data)
{
    __builtin_IB_subgroup_block_write_flat_u32_m2k16v1(as_long(base_address), width - 1, height - 1, pitch - 1, coord, data);
}
void intel_subgroup_block_write_u32_m4k16(__global void* base_address, int width, int height, int pitch, int2 coord, uint4 data)
{
    __builtin_IB_subgroup_block_write_flat_u32_m4k16v1(as_long(base_address), width - 1, height - 1, pitch - 1, coord, data);
}
void intel_subgroup_block_write_u32_m8k16(__global void* base_address, int width, int height, int pitch, int2 coord, uint8 data)
{
    __builtin_IB_subgroup_block_write_flat_u32_m8k16v1(as_long(base_address), width - 1, height - 1, pitch - 1, coord, data);
}

#endif // cl_intel_subgroup_extended_block_read
