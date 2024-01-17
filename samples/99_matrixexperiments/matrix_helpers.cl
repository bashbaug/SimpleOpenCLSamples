float bf16_to_fp32(ushort u)
{
#if defined(cl_intel_bfloat16_conversions)
    return intel_convert_as_bfloat16_float(u);
#else
    return as_float(u << 16);
#endif
}

#if defined(cl_intel_subgroups) && defined(cl_intel_subgroups_short)

// Emulated SIMD8 dpas:
__attribute__((overloadable))
float  emu_sub_group_bf16_bf16_matrix_mad_k16(int  a, int8 b, float  acc)
{
    float res = acc;

    res = fma(bf16_to_fp32(as_ushort2(sub_group_broadcast(a, 0)).x), bf16_to_fp32(as_ushort2(b.s0).x), res);
    res = fma(bf16_to_fp32(as_ushort2(sub_group_broadcast(a, 0)).y), bf16_to_fp32(as_ushort2(b.s0).y), res);
    res = fma(bf16_to_fp32(as_ushort2(sub_group_broadcast(a, 1)).x), bf16_to_fp32(as_ushort2(b.s1).x), res);
    res = fma(bf16_to_fp32(as_ushort2(sub_group_broadcast(a, 1)).y), bf16_to_fp32(as_ushort2(b.s1).y), res);
    res = fma(bf16_to_fp32(as_ushort2(sub_group_broadcast(a, 2)).x), bf16_to_fp32(as_ushort2(b.s2).x), res);
    res = fma(bf16_to_fp32(as_ushort2(sub_group_broadcast(a, 2)).y), bf16_to_fp32(as_ushort2(b.s2).y), res);
    res = fma(bf16_to_fp32(as_ushort2(sub_group_broadcast(a, 3)).x), bf16_to_fp32(as_ushort2(b.s3).x), res);
    res = fma(bf16_to_fp32(as_ushort2(sub_group_broadcast(a, 3)).y), bf16_to_fp32(as_ushort2(b.s3).y), res);
    res = fma(bf16_to_fp32(as_ushort2(sub_group_broadcast(a, 4)).x), bf16_to_fp32(as_ushort2(b.s4).x), res);
    res = fma(bf16_to_fp32(as_ushort2(sub_group_broadcast(a, 4)).y), bf16_to_fp32(as_ushort2(b.s4).y), res);
    res = fma(bf16_to_fp32(as_ushort2(sub_group_broadcast(a, 5)).x), bf16_to_fp32(as_ushort2(b.s5).x), res);
    res = fma(bf16_to_fp32(as_ushort2(sub_group_broadcast(a, 5)).y), bf16_to_fp32(as_ushort2(b.s5).y), res);
    res = fma(bf16_to_fp32(as_ushort2(sub_group_broadcast(a, 6)).x), bf16_to_fp32(as_ushort2(b.s6).x), res);
    res = fma(bf16_to_fp32(as_ushort2(sub_group_broadcast(a, 6)).y), bf16_to_fp32(as_ushort2(b.s6).y), res);
    res = fma(bf16_to_fp32(as_ushort2(sub_group_broadcast(a, 7)).x), bf16_to_fp32(as_ushort2(b.s7).x), res);
    res = fma(bf16_to_fp32(as_ushort2(sub_group_broadcast(a, 7)).y), bf16_to_fp32(as_ushort2(b.s7).y), res);

    return res;
}

__attribute__((overloadable))
float2 emu_sub_group_bf16_bf16_matrix_mad_k16(int2 a, int8 b, float2 acc)
{
    float2 res;

    res.s0 = emu_sub_group_bf16_bf16_matrix_mad_k16(a.s0, b, acc.s0);
    res.s1 = emu_sub_group_bf16_bf16_matrix_mad_k16(a.s1, b, acc.s1);

    return res;
}

__attribute__((overloadable))
float4 emu_sub_group_bf16_bf16_matrix_mad_k16(int4 a, int8 b, float4 acc)
{
    float4 res;

    res.s0 = emu_sub_group_bf16_bf16_matrix_mad_k16(a.s0, b, acc.s0);
    res.s1 = emu_sub_group_bf16_bf16_matrix_mad_k16(a.s1, b, acc.s1);
    res.s2 = emu_sub_group_bf16_bf16_matrix_mad_k16(a.s2, b, acc.s2);
    res.s3 = emu_sub_group_bf16_bf16_matrix_mad_k16(a.s3, b, acc.s3);

    return res;
}

__attribute__((overloadable))
float8 emu_sub_group_bf16_bf16_matrix_mad_k16(int8 a, int8 b, float8 acc)
{
    float8 res;

    res.s0 = emu_sub_group_bf16_bf16_matrix_mad_k16(a.s0, b, acc.s0);
    res.s1 = emu_sub_group_bf16_bf16_matrix_mad_k16(a.s1, b, acc.s1);
    res.s2 = emu_sub_group_bf16_bf16_matrix_mad_k16(a.s2, b, acc.s2);
    res.s3 = emu_sub_group_bf16_bf16_matrix_mad_k16(a.s3, b, acc.s3);
    res.s4 = emu_sub_group_bf16_bf16_matrix_mad_k16(a.s4, b, acc.s4);
    res.s5 = emu_sub_group_bf16_bf16_matrix_mad_k16(a.s5, b, acc.s5);
    res.s6 = emu_sub_group_bf16_bf16_matrix_mad_k16(a.s6, b, acc.s6);
    res.s7 = emu_sub_group_bf16_bf16_matrix_mad_k16(a.s7, b, acc.s7);

    return res;
}

// Emulated SIMD16 dpas:
__attribute__((overloadable))
float  emu_sub_group_bf16_bf16_matrix_mad_k16(short  a, int8 b, float  acc)
{
    float res = acc;

    res = fma(bf16_to_fp32(sub_group_broadcast(a,  0)), bf16_to_fp32(as_ushort2(b.s0).x), res);
    res = fma(bf16_to_fp32(sub_group_broadcast(a,  1)), bf16_to_fp32(as_ushort2(b.s0).y), res);
    res = fma(bf16_to_fp32(sub_group_broadcast(a,  2)), bf16_to_fp32(as_ushort2(b.s1).x), res);
    res = fma(bf16_to_fp32(sub_group_broadcast(a,  3)), bf16_to_fp32(as_ushort2(b.s1).y), res);
    res = fma(bf16_to_fp32(sub_group_broadcast(a,  4)), bf16_to_fp32(as_ushort2(b.s2).x), res);
    res = fma(bf16_to_fp32(sub_group_broadcast(a,  5)), bf16_to_fp32(as_ushort2(b.s2).y), res);
    res = fma(bf16_to_fp32(sub_group_broadcast(a,  6)), bf16_to_fp32(as_ushort2(b.s3).x), res);
    res = fma(bf16_to_fp32(sub_group_broadcast(a,  7)), bf16_to_fp32(as_ushort2(b.s3).y), res);
    res = fma(bf16_to_fp32(sub_group_broadcast(a,  8)), bf16_to_fp32(as_ushort2(b.s4).x), res);
    res = fma(bf16_to_fp32(sub_group_broadcast(a,  9)), bf16_to_fp32(as_ushort2(b.s4).y), res);
    res = fma(bf16_to_fp32(sub_group_broadcast(a, 10)), bf16_to_fp32(as_ushort2(b.s5).x), res);
    res = fma(bf16_to_fp32(sub_group_broadcast(a, 11)), bf16_to_fp32(as_ushort2(b.s5).y), res);
    res = fma(bf16_to_fp32(sub_group_broadcast(a, 12)), bf16_to_fp32(as_ushort2(b.s6).x), res);
    res = fma(bf16_to_fp32(sub_group_broadcast(a, 13)), bf16_to_fp32(as_ushort2(b.s6).y), res);
    res = fma(bf16_to_fp32(sub_group_broadcast(a, 14)), bf16_to_fp32(as_ushort2(b.s7).x), res);
    res = fma(bf16_to_fp32(sub_group_broadcast(a, 15)), bf16_to_fp32(as_ushort2(b.s7).y), res);

    return res;
}

__attribute__((overloadable))
float2 emu_sub_group_bf16_bf16_matrix_mad_k16(short2 a, int8 b, float2 acc)
{
    float2 res;

    res.s0 = emu_sub_group_bf16_bf16_matrix_mad_k16(a.s0, b, acc.s0);
    res.s1 = emu_sub_group_bf16_bf16_matrix_mad_k16(a.s1, b, acc.s1);

    return res;
}

__attribute__((overloadable))
float4 emu_sub_group_bf16_bf16_matrix_mad_k16(short4 a, int8 b, float4 acc)
{
    float4 res;

    res.s0 = emu_sub_group_bf16_bf16_matrix_mad_k16(a.s0, b, acc.s0);
    res.s1 = emu_sub_group_bf16_bf16_matrix_mad_k16(a.s1, b, acc.s1);
    res.s2 = emu_sub_group_bf16_bf16_matrix_mad_k16(a.s2, b, acc.s2);
    res.s3 = emu_sub_group_bf16_bf16_matrix_mad_k16(a.s3, b, acc.s3);

    return res;
}

__attribute__((overloadable))
float8 emu_sub_group_bf16_bf16_matrix_mad_k16(short8 a, int8 b, float8 acc)
{
    float8 res;

    res.s0 = emu_sub_group_bf16_bf16_matrix_mad_k16(a.s0, b, acc.s0);
    res.s1 = emu_sub_group_bf16_bf16_matrix_mad_k16(a.s1, b, acc.s1);
    res.s2 = emu_sub_group_bf16_bf16_matrix_mad_k16(a.s2, b, acc.s2);
    res.s3 = emu_sub_group_bf16_bf16_matrix_mad_k16(a.s3, b, acc.s3);
    res.s4 = emu_sub_group_bf16_bf16_matrix_mad_k16(a.s4, b, acc.s4);
    res.s5 = emu_sub_group_bf16_bf16_matrix_mad_k16(a.s5, b, acc.s5);
    res.s6 = emu_sub_group_bf16_bf16_matrix_mad_k16(a.s6, b, acc.s6);
    res.s7 = emu_sub_group_bf16_bf16_matrix_mad_k16(a.s7, b, acc.s7);

    return res;
}

// M rows x K columns
// This is the SIMD8 version, where each work-item loads two values.
int  load_a_rowmajor_d16_m1_k16_sg8(global ushort* A, int rowStart, int colStart, int stride)
{
    int ret;

    global uint* A_ui = (global uint*)A;
    int offset_ui = rowStart * stride / 2 + colStart / 2;
    ret = intel_sub_group_block_read(A_ui + offset_ui);

    return ret;
}

// M rows x K columns
// This is the SIMD8 version, where each work-item loads two values.
int2 load_a_rowmajor_d16_m2_k16_sg8(global ushort* A, int rowStart, int colStart, int stride)
{
    int2 ret;

    global uint* A_ui = (global uint*)A;
    int offset_ui = rowStart * stride / 2 + colStart / 2;

    ret.s0 = intel_sub_group_block_read(A_ui + offset_ui); offset_ui += stride / 2;
    ret.s1 = intel_sub_group_block_read(A_ui + offset_ui); offset_ui += stride / 2;

    return ret;
}

// M rows x K columns
// This is the SIMD8 version, where each work-item loads two values.
int4 load_a_rowmajor_d16_m4_k16_sg8(global ushort* A, int rowStart, int colStart, int stride)
{
    int4 ret;

    global uint* A_ui = (global uint*)A;
    int offset_ui = rowStart * stride / 2 + colStart / 2;

    ret.s0 = intel_sub_group_block_read(A_ui + offset_ui); offset_ui += stride / 2;
    ret.s1 = intel_sub_group_block_read(A_ui + offset_ui); offset_ui += stride / 2;
    ret.s2 = intel_sub_group_block_read(A_ui + offset_ui); offset_ui += stride / 2;
    ret.s3 = intel_sub_group_block_read(A_ui + offset_ui); offset_ui += stride / 2;

    return ret;
}

// M rows x K columns
// This is the SIMD8 version, where each work-item loads two values.
int8 load_a_rowmajor_d16_m8_k16_sg8(global ushort* A, int rowStart, int colStart, int stride)
{
    int8 ret;

    global uint* A_ui = (global uint*)A;
    int offset_ui = rowStart * stride / 2 + colStart / 2;

    ret.s0 = intel_sub_group_block_read(A_ui + offset_ui); offset_ui += stride / 2;
    ret.s1 = intel_sub_group_block_read(A_ui + offset_ui); offset_ui += stride / 2;
    ret.s2 = intel_sub_group_block_read(A_ui + offset_ui); offset_ui += stride / 2;
    ret.s3 = intel_sub_group_block_read(A_ui + offset_ui); offset_ui += stride / 2;
    ret.s4 = intel_sub_group_block_read(A_ui + offset_ui); offset_ui += stride / 2;
    ret.s5 = intel_sub_group_block_read(A_ui + offset_ui); offset_ui += stride / 2;
    ret.s6 = intel_sub_group_block_read(A_ui + offset_ui); offset_ui += stride / 2;
    ret.s7 = intel_sub_group_block_read(A_ui + offset_ui); offset_ui += stride / 2;

    return ret;
}

// M rows x K columns
// This is the SIMD16 version, where each work-item loads one values.
short load_a_rowmajor_d16_m1_k16_sg16(global ushort* A, int rowStart, int colStart, int stride)
{
    ushort ret;

    int offset = rowStart * stride + colStart;
    ret = intel_sub_group_block_read_us(A + offset);

    return as_short(ret);
}

// M rows x K columns
// This is the SIMD16 version, where each work-item loads one values.
short2 load_a_rowmajor_d16_m2_k16_sg16(global ushort* A, int rowStart, int colStart, int stride)
{
    ushort2 ret;

    int offset = rowStart * stride + colStart;
    ret.s0 = intel_sub_group_block_read_us(A + offset); offset += stride;
    ret.s1 = intel_sub_group_block_read_us(A + offset); offset += stride;

    return as_short2(ret);
}

// M rows x K columns
// This is the SIMD16 version, where each work-item loads one values.
short4 load_a_rowmajor_d16_m4_k16_sg16(global ushort* A, int rowStart, int colStart, int stride)
{
    ushort4 ret;

    int offset = rowStart * stride + colStart;
    ret.s0 = intel_sub_group_block_read_us(A + offset); offset += stride;
    ret.s1 = intel_sub_group_block_read_us(A + offset); offset += stride;
    ret.s2 = intel_sub_group_block_read_us(A + offset); offset += stride;
    ret.s3 = intel_sub_group_block_read_us(A + offset); offset += stride;

    return as_short4(ret);
}

// M rows x K columns
// This is the SIMD16 version, where each work-item loads one values.
short8 load_a_rowmajor_d16_m8_k16_sg16(global ushort* A, int rowStart, int colStart, int stride)
{
    ushort8 ret;

    int offset = rowStart * stride + colStart;
    ret.s0 = intel_sub_group_block_read_us(A + offset); offset += stride;
    ret.s1 = intel_sub_group_block_read_us(A + offset); offset += stride;
    ret.s2 = intel_sub_group_block_read_us(A + offset); offset += stride;
    ret.s3 = intel_sub_group_block_read_us(A + offset); offset += stride;
    ret.s4 = intel_sub_group_block_read_us(A + offset); offset += stride;
    ret.s5 = intel_sub_group_block_read_us(A + offset); offset += stride;
    ret.s6 = intel_sub_group_block_read_us(A + offset); offset += stride;
    ret.s7 = intel_sub_group_block_read_us(A + offset); offset += stride;

    return as_short8(ret);
}

// K rows x N columns:
// Each work-item loads K values and converts to VNNI.
// Stride is in units of elements.
int8 load_b_rowmajor_d16_k16_nx(global ushort* B, int rowStart, int colStart, int stride)
{
    int8 ret;

    int offset = rowStart * stride + colStart;

    ushort row0  = intel_sub_group_block_read_us(B + offset); offset += stride;
    ushort row1  = intel_sub_group_block_read_us(B + offset); offset += stride;
    ushort row2  = intel_sub_group_block_read_us(B + offset); offset += stride;
    ushort row3  = intel_sub_group_block_read_us(B + offset); offset += stride;
    ushort row4  = intel_sub_group_block_read_us(B + offset); offset += stride;
    ushort row5  = intel_sub_group_block_read_us(B + offset); offset += stride;
    ushort row6  = intel_sub_group_block_read_us(B + offset); offset += stride;
    ushort row7  = intel_sub_group_block_read_us(B + offset); offset += stride;
    ushort row8  = intel_sub_group_block_read_us(B + offset); offset += stride;
    ushort row9  = intel_sub_group_block_read_us(B + offset); offset += stride;
    ushort row10 = intel_sub_group_block_read_us(B + offset); offset += stride;
    ushort row11 = intel_sub_group_block_read_us(B + offset); offset += stride;
    ushort row12 = intel_sub_group_block_read_us(B + offset); offset += stride;
    ushort row13 = intel_sub_group_block_read_us(B + offset); offset += stride;
    ushort row14 = intel_sub_group_block_read_us(B + offset); offset += stride;
    ushort row15 = intel_sub_group_block_read_us(B + offset); offset += stride;

    ret.s0 = as_int((ushort2)(row0,  row1 ));
    ret.s1 = as_int((ushort2)(row2,  row3 ));
    ret.s2 = as_int((ushort2)(row4,  row5 ));
    ret.s3 = as_int((ushort2)(row6,  row7 ));
    ret.s4 = as_int((ushort2)(row8,  row9 ));
    ret.s5 = as_int((ushort2)(row10, row11));
    ret.s6 = as_int((ushort2)(row12, row13));
    ret.s7 = as_int((ushort2)(row14, row15));

    return ret;
}

// K rows x N columns:
// Each work-item loads K values that has already been converted to VNNI.
// Stride is in units of elements.
int8 load_b_vnni_d16_k16_nx(global ushort* B, int rowStart, int colStart, int stride)
{
    int8 ret;

    global uint* B_ui = (global uint*)B;
    int offset_ui = rowStart / 2 * stride + colStart;

    ret.s0 = intel_sub_group_block_read(B_ui + offset_ui); offset_ui += stride;
    ret.s1 = intel_sub_group_block_read(B_ui + offset_ui); offset_ui += stride;
    ret.s2 = intel_sub_group_block_read(B_ui + offset_ui); offset_ui += stride;
    ret.s3 = intel_sub_group_block_read(B_ui + offset_ui); offset_ui += stride;
    ret.s4 = intel_sub_group_block_read(B_ui + offset_ui); offset_ui += stride;
    ret.s5 = intel_sub_group_block_read(B_ui + offset_ui); offset_ui += stride;
    ret.s6 = intel_sub_group_block_read(B_ui + offset_ui); offset_ui += stride;
    ret.s7 = intel_sub_group_block_read(B_ui + offset_ui); offset_ui += stride;

    return ret;
}

void store_c_rowmajor_fp32_m1_nx(global float* C, float v, int rowStart, int colStart, int stride)
{
    global uint* C_ui = (global uint*)C;
    uint v_ui = as_uint(v);

    int offset = rowStart * stride + colStart;

    intel_sub_group_block_write(C_ui + offset, v_ui); offset += stride;
}

void store_c_rowmajor_fp32_m2_nx(global float* C, float2 v, int rowStart, int colStart, int stride)
{
    global uint* C_ui = (global uint*)C;
    uint2 v_ui = as_uint2(v);

    int offset = rowStart * stride + colStart;

    intel_sub_group_block_write(C_ui + offset, v_ui.s0); offset += stride;
    intel_sub_group_block_write(C_ui + offset, v_ui.s1); offset += stride;
}

void store_c_rowmajor_fp32_m4_nx(global float* C, float4 v, int rowStart, int colStart, int stride)
{
    global uint* C_ui = (global uint*)C;
    uint4 v_ui = as_uint4(v);

    int offset = rowStart * stride + colStart;

    intel_sub_group_block_write(C_ui + offset, v_ui.s0); offset += stride;
    intel_sub_group_block_write(C_ui + offset, v_ui.s1); offset += stride;
    intel_sub_group_block_write(C_ui + offset, v_ui.s2); offset += stride;
    intel_sub_group_block_write(C_ui + offset, v_ui.s3); offset += stride;
}

void store_c_rowmajor_fp32_m8_nx(global float* C, float8 v, int rowStart, int colStart, int stride)
{
    global uint* C_ui = (global uint*)C;
    uint8 v_ui = as_uint8(v);

    int offset = rowStart * stride + colStart;

    intel_sub_group_block_write(C_ui + offset, v_ui.s0); offset += stride;
    intel_sub_group_block_write(C_ui + offset, v_ui.s1); offset += stride;
    intel_sub_group_block_write(C_ui + offset, v_ui.s2); offset += stride;
    intel_sub_group_block_write(C_ui + offset, v_ui.s3); offset += stride;
    intel_sub_group_block_write(C_ui + offset, v_ui.s4); offset += stride;
    intel_sub_group_block_write(C_ui + offset, v_ui.s5); offset += stride;
    intel_sub_group_block_write(C_ui + offset, v_ui.s6); offset += stride;
    intel_sub_group_block_write(C_ui + offset, v_ui.s7); offset += stride;
}

#endif // defined(cl_intel_subgroups) && defined(cl_intel_subgroups_short)

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

ushort  __builtin_IB_subgroup_block_read_flat_u16_m1k16v1(long baseoffset, int width_minus_one, int height_minus_one, int pitch_minus_one, int2 coord);
ushort2 __builtin_IB_subgroup_block_read_flat_u16_m2k16v1(long baseoffset, int width_minus_one, int height_minus_one, int pitch_minus_one, int2 coord);
ushort4 __builtin_IB_subgroup_block_read_flat_u16_m4k16v1(long baseoffset, int width_minus_one, int height_minus_one, int pitch_minus_one, int2 coord);
ushort8 __builtin_IB_subgroup_block_read_flat_u16_m8k16v1(long baseoffset, int width_minus_one, int height_minus_one, int pitch_minus_one, int2 coord);

uint8 __builtin_IB_subgroup_block_read_flat_u32_m8k16v1(long baseoffset, int width_minus_one, int height_minus_one, int pitch_minus_one, int2 coord);

void __builtin_IB_subgroup_block_write_flat_u32_m1k16v1(long baseoffset, int width_minus_one, int height_minus_one, int pitch_minus_one, int2 coord, uint  data);
void __builtin_IB_subgroup_block_write_flat_u32_m2k16v1(long baseoffset, int width_minus_one, int height_minus_one, int pitch_minus_one, int2 coord, uint2 data);
void __builtin_IB_subgroup_block_write_flat_u32_m4k16v1(long baseoffset, int width_minus_one, int height_minus_one, int pitch_minus_one, int2 coord, uint4 data);
void __builtin_IB_subgroup_block_write_flat_u32_m8k16v1(long baseoffset, int width_minus_one, int height_minus_one, int pitch_minus_one, int2 coord, uint8 data);

ushort  intel_subgroup_block_read_u16_m1k16(const __global void *base_address, int width, int height, int pitch, int2 coord)
{
    return __builtin_IB_subgroup_block_read_flat_u16_m1k16v1(as_long(base_address), width - 1, height - 1, pitch - 1, coord);
}
ushort2 intel_subgroup_block_read_u16_m2k16(const __global void *base_address, int width, int height, int pitch, int2 coord)
{
    return __builtin_IB_subgroup_block_read_flat_u16_m2k16v1(as_long(base_address), width - 1, height - 1, pitch - 1, coord);
}
ushort4 intel_subgroup_block_read_u16_m4k16(const __global void *base_address, int width, int height, int pitch, int2 coord)
{
    return __builtin_IB_subgroup_block_read_flat_u16_m4k16v1(as_long(base_address), width - 1, height - 1, pitch - 1, coord);
}
ushort8 intel_subgroup_block_read_u16_m8k16(const __global void *base_address, int width, int height, int pitch, int2 coord)
{
    return __builtin_IB_subgroup_block_read_flat_u16_m8k16v1(as_long(base_address), width - 1, height - 1, pitch - 1, coord);
}

uint8 intel_subgroup_block_read_u32_m8k16(const __global void* base_address, int width, int height, int pitch, int2 coord)
{
    return __builtin_IB_subgroup_block_read_flat_u32_m8k16v1(as_long(base_address), width - 1, height - 1, pitch - 1, coord);
}

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
