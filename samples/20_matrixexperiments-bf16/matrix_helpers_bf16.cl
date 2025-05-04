/*
// Copyright (c) 2024-2025 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

float bf16_to_fp32(ushort u)
{
#if defined(cl_intel_bfloat16_conversions)
    return intel_convert_as_bfloat16_float(u);
#else
    return as_float(u << 16);
#endif
}

__attribute__((overloadable))
float activation(float f)
{
#if defined(ACTIVATION_RELU)
    return fmax(f, 0);
#else   // identity
    return f;
#endif
}

__attribute__((overloadable))
float2 activation(float2 f)
{
    float2 res;
    res.s0 = activation(f.s0);
    res.s1 = activation(f.s1);
    return res;
}

__attribute__((overloadable))
float4 activation(float4 f)
{
    float4 res;
    res.s0 = activation(f.s0);
    res.s1 = activation(f.s1);
    res.s2 = activation(f.s2);
    res.s3 = activation(f.s3);
    return res;
}

float8 activation(float8 f)
{
    float8 res;
    res.s0 = activation(f.s0);
    res.s1 = activation(f.s1);
    res.s2 = activation(f.s2);
    res.s3 = activation(f.s3);
    res.s4 = activation(f.s4);
    res.s5 = activation(f.s5);
    res.s6 = activation(f.s6);
    res.s7 = activation(f.s7);
    return res;
}

#ifndef __has_builtin
#define __has_builtin(x) 0
#endif
#if __has_builtin(__builtin_expect) == 0
#define __builtin_expect(x)
#endif

#if defined(cl_intel_subgroups) && defined(cl_intel_subgroups_short)

typedef global ushort* global_aligned_ushort_ptr __attribute__((align_value(4)));

inline int compute_m(const int num_sgs_x, const int num_sgs_y, const int tM, const int MM)
{
    const int m_start = get_group_id(1) * num_sgs_y;
    const int m_index = num_sgs_y > 1 ? m_start + get_sub_group_id() / num_sgs_x : m_start;
    return m_index * tM * MM;
}

inline int compute_n(const int num_sgs_x, const int num_sgs_y, const int tN, const int NN)
{
    const int n_start = get_group_id(0) * num_sgs_x;
    const int n_index = num_sgs_x > 1 ? n_start + get_sub_group_id() % num_sgs_x : n_start;
    return n_index * tN * NN;
}

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
int  load_a_rowmajor_16b_1r16c_sg8(global ushort* A, int rowStart, int colStart, int stride)
{
    int ret;

    global uint* A_ui = (global uint*)A;
    uint offset_ui = rowStart * stride / 2 + colStart / 2;
    ret = intel_sub_group_block_read(A_ui + offset_ui);

    return ret;
}

// M rows x K columns
// This is the SIMD8 version, where each work-item loads two values.
int2 load_a_rowmajor_16b_2r16c_sg8(global ushort* A, int rowStart, int colStart, int stride)
{
    int2 ret;

    global uint* A_ui = (global uint*)A;
    uint offset_ui = rowStart * stride / 2 + colStart / 2;

    ret.s0 = intel_sub_group_block_read(A_ui + offset_ui); offset_ui += stride / 2;
    ret.s1 = intel_sub_group_block_read(A_ui + offset_ui); offset_ui += stride / 2;

    return ret;
}

// M rows x K columns
// This is the SIMD8 version, where each work-item loads two values.
int4 load_a_rowmajor_16b_4r16c_sg8(global ushort* A, int rowStart, int colStart, int stride)
{
    int4 ret;

    global uint* A_ui = (global uint*)A;
    uint offset_ui = rowStart * stride / 2 + colStart / 2;

    ret.s0 = intel_sub_group_block_read(A_ui + offset_ui); offset_ui += stride / 2;
    ret.s1 = intel_sub_group_block_read(A_ui + offset_ui); offset_ui += stride / 2;
    ret.s2 = intel_sub_group_block_read(A_ui + offset_ui); offset_ui += stride / 2;
    ret.s3 = intel_sub_group_block_read(A_ui + offset_ui); offset_ui += stride / 2;

    return ret;
}

// M rows x K columns
// This is the SIMD8 version, where each work-item loads two values.
int8 load_a_rowmajor_16b_8r16c_sg8(global ushort* A, int rowStart, int colStart, int stride)
{
    int8 ret;

    global uint* A_ui = (global uint*)A;
    uint offset_ui = rowStart * stride / 2 + colStart / 2;

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

// M rows x K columns x V tiles (in the K dimension)
// This is the SIMD8 version, where each work-item loads two values.
// The first tile is returned the first components of the return value, the the next tile, etc.
int16 load_a_rowmajor_16b_8r16x2c_sg8(global ushort* A, int rowStart, int colStart, int stride)
{
    uint16 ret;

    global uint* A_ui = (global uint*)A;
    uint offset_ui = rowStart * stride / 2 + colStart / 2;

    ret.s08 = intel_sub_group_block_read2(A_ui + offset_ui); offset_ui += stride / 2;
    ret.s19 = intel_sub_group_block_read2(A_ui + offset_ui); offset_ui += stride / 2;
    ret.s2a = intel_sub_group_block_read2(A_ui + offset_ui); offset_ui += stride / 2;
    ret.s3b = intel_sub_group_block_read2(A_ui + offset_ui); offset_ui += stride / 2;
    ret.s4c = intel_sub_group_block_read2(A_ui + offset_ui); offset_ui += stride / 2;
    ret.s5d = intel_sub_group_block_read2(A_ui + offset_ui); offset_ui += stride / 2;
    ret.s6e = intel_sub_group_block_read2(A_ui + offset_ui); offset_ui += stride / 2;
    ret.s7f = intel_sub_group_block_read2(A_ui + offset_ui); offset_ui += stride / 2;

    return as_int16(ret);
}

// M rows x K columns x V tiles (in the K dimension)
void prefetch_a_rowmajor_16b_8r16x2c_sg8(global ushort* A, int rowStart, int colStart, int stride)
{
#if defined(PREFETCH_DEFAULT)
    uint offset = colStart + (rowStart + get_sub_group_local_id()) * stride;
    __builtin_assume((ulong)(A + offset) % 4 == 0);
    prefetch(A + offset, 2);
#endif // defined(PREFETCH_DEFAULT)
}

// M rows x K columns
// This is the SIMD16 version, where each work-item loads one value.
short load_a_rowmajor_16b_1r16c_sg16(global ushort* A, int rowStart, int colStart, int stride)
{
    ushort ret;

    uint offset = rowStart * stride + colStart;
    ret = intel_sub_group_block_read_us(A + offset);

    return as_short(ret);
}

// M rows x K columns
// This is the SIMD16 version, where each work-item loads one value.
short2 load_a_rowmajor_16b_2r16c_sg16(global ushort* A, int rowStart, int colStart, int stride)
{
    ushort2 ret;

    uint offset = rowStart * stride + colStart;
    ret.s0 = intel_sub_group_block_read_us(A + offset); offset += stride;
    ret.s1 = intel_sub_group_block_read_us(A + offset); offset += stride;

    return as_short2(ret);
}

// M rows x K columns
// This is the SIMD16 version, where each work-item loads one value.
short4 load_a_rowmajor_16b_4r16c_sg16(global ushort* A, int rowStart, int colStart, int stride)
{
    ushort4 ret;

    uint offset = rowStart * stride + colStart;
    ret.s0 = intel_sub_group_block_read_us(A + offset); offset += stride;
    ret.s1 = intel_sub_group_block_read_us(A + offset); offset += stride;
    ret.s2 = intel_sub_group_block_read_us(A + offset); offset += stride;
    ret.s3 = intel_sub_group_block_read_us(A + offset); offset += stride;

    return as_short4(ret);
}

// M rows x K columns
// This is the SIMD16 version, where each work-item loads one value.
short8 load_a_rowmajor_16b_8r16c_sg16(global ushort* A, int rowStart, int colStart, int stride)
{
    ushort8 ret;

    uint offset = rowStart * stride + colStart;
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

// M rows x K columns x V tiles (in the K dimension)
// This is the SIMD16 version, where each work-item loads one value.
// The first tile is returned the first components of the return value, the the next tile, etc.
short16 load_a_rowmajor_16b_8r16x2c_sg16(global ushort* A, int rowStart, int colStart, int stride)
{
    ushort16 ret;

    uint offset = rowStart * stride + colStart;
    ret.s08 = intel_sub_group_block_read_us2(A + offset); offset += stride;
    ret.s19 = intel_sub_group_block_read_us2(A + offset); offset += stride;
    ret.s2a = intel_sub_group_block_read_us2(A + offset); offset += stride;
    ret.s3b = intel_sub_group_block_read_us2(A + offset); offset += stride;
    ret.s4c = intel_sub_group_block_read_us2(A + offset); offset += stride;
    ret.s5d = intel_sub_group_block_read_us2(A + offset); offset += stride;
    ret.s6e = intel_sub_group_block_read_us2(A + offset); offset += stride;
    ret.s7f = intel_sub_group_block_read_us2(A + offset); offset += stride;

    return as_short16(ret);
}

// M rows x K columns x V tiles (in the M and K dimensions)
void prefetch_a_rowmajor_16b_8x2r16x2c_sg16(global ushort* A, int rowStart, int colStart, int stride)
{
#if defined(PREFETCH_DEFAULT)
    uint offset = colStart + (rowStart + get_sub_group_local_id()) * stride;
    __builtin_assume((ulong)(A + offset) % 4 == 0);
    prefetch(A + offset, 2);
#endif // defined(PREFETCH_DEFAULT)
}

// K rows x N columns:
// Each work-item loads K values and packs into 32-bits.
// Stride is in units of elements.
int8 load_b_rowmajor_16b_16rNc(global ushort* B, int rowStart, int colStart, int stride)
{
    int8 ret;

    uint offset = rowStart * stride + colStart;

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
// Each work-item loads K values that have already been packed into 32-bits.
// Stride is in units of elements.
int8 load_b_packed_16b_16rNc(global ushort* B, int rowStart, int colStart, int stride)
{
    int8 ret;

    global uint* B_ui = (global uint*)B;
    uint offset_ui = rowStart / 2 * stride + colStart;

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

// K rows x N columns x V tiles (in the N dimension)
void prefetch_b_rowmajor_16b_16r8x4c_sg8(global ushort* B, int rowStart, int colStart, int stride)
{
#if defined(PREFETCH_DEFAULT)
    uint offset = colStart + (rowStart + get_sub_group_local_id()) * stride;
    __builtin_assume((ulong)(B + offset) % 4 == 0);
    prefetch(B + offset, 2);    offset += 8 * stride;
    __builtin_assume((ulong)(B + offset) % 4 == 0);
    prefetch(B + offset, 2);    offset += 8 * stride;
#endif // defined(PREFETCH_DEFAULT)
}

// K rows x N columns x V tiles (in the N dimension)
void prefetch_b_rowmajor_16b_16r16x2c_sg16(global ushort* B, int rowStart, int colStart, int stride)
{
#if defined(PREFETCH_DEFAULT)
    uint offset = colStart + (rowStart + get_sub_group_local_id()) * stride;
    __builtin_assume((ulong)(B + offset) % 4 == 0);
    prefetch(B + offset, 2);
#endif // defined(PREFETCH_DEFAULT)
}

// K rows x N columns x V tiles (in the N dimension)
void prefetch_b_packed_16b_16r8x2c_sg8(global ushort* B, int rowStart, int colStart, int stride)
{
#if defined(PREFETCH_DEFAULT)
    global uint* B_ui = (global uint*)B;
    uint offset_ui = colStart + (rowStart / 2 + get_sub_group_local_id()) * stride;
    __builtin_assume((ulong)(B_ui + offset_ui) % 4 == 0);
    prefetch(B_ui + offset_ui, 1);
#endif // defined(PREFETCH_DEFAULT)
}

// K rows x N columns x V tiles (in the K dimension)
void prefetch_b_packed_16b_16x2r16c_sg16(global ushort* B, int rowStart, int colStart, int stride)
{
#if defined(PREFETCH_DEFAULT)
    global uint* B_ui = (global uint*)B;
    uint offset_ui = colStart + (rowStart / 2 + get_sub_group_local_id()) * stride;
    __builtin_assume((ulong)(B_ui + offset_ui) % 4 == 0);
    prefetch(B_ui + offset_ui, 1);
#endif // defined(PREFETCH_DEFAULT)
}

void store_c_rowmajor_fp32_1rNc(global float* C, float v, int rowStart, int colStart, int stride)
{
    global uint* C_ui = (global uint*)C;
    uint v_ui = as_uint(v);

    uint offset = rowStart * stride + colStart;

    intel_sub_group_block_write(C_ui + offset, v_ui); offset += stride;
}

void store_c_rowmajor_fp32_2rNc(global float* C, float2 v, int rowStart, int colStart, int stride)
{
    global uint* C_ui = (global uint*)C;
    uint2 v_ui = as_uint2(v);

    uint offset = rowStart * stride + colStart;

    intel_sub_group_block_write(C_ui + offset, v_ui.s0); offset += stride;
    intel_sub_group_block_write(C_ui + offset, v_ui.s1); offset += stride;
}

void store_c_rowmajor_fp32_4rNc(global float* C, float4 v, int rowStart, int colStart, int stride)
{
    global uint* C_ui = (global uint*)C;
    uint4 v_ui = as_uint4(v);

    uint offset = rowStart * stride + colStart;

    intel_sub_group_block_write(C_ui + offset, v_ui.s0); offset += stride;
    intel_sub_group_block_write(C_ui + offset, v_ui.s1); offset += stride;
    intel_sub_group_block_write(C_ui + offset, v_ui.s2); offset += stride;
    intel_sub_group_block_write(C_ui + offset, v_ui.s3); offset += stride;
}

void store_c_rowmajor_fp32_8rNc(global float* C, float8 v, int rowStart, int colStart, int stride)
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

#endif // defined(cl_intel_subgroups) && defined(cl_intel_subgroups_short)
