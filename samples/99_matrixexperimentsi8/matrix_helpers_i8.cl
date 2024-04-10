__attribute__((overloadable))
int activation(int i)
{
#if defined(ACTIVATION_RELU)
    return max(i, 0);
#else   // identity
    return i;
#endif
}

__attribute__((overloadable))
int2 activation(int2 i)
{
    int2 res;
    res.s0 = activation(i.s0);
    res.s1 = activation(i.s1);
    return res;
}

__attribute__((overloadable))
int4 activation(int4 i)
{
    int4 res;
    res.s0 = activation(i.s0);
    res.s1 = activation(i.s1);
    res.s2 = activation(i.s2);
    res.s3 = activation(i.s3);
    return res;
}

int8 activation(int8 i)
{
    int8 res;
    res.s0 = activation(i.s0);
    res.s1 = activation(i.s1);
    res.s2 = activation(i.s2);
    res.s3 = activation(i.s3);
    res.s4 = activation(i.s4);
    res.s5 = activation(i.s5);
    res.s6 = activation(i.s6);
    res.s7 = activation(i.s7);
    return res;
}

#ifndef __has_builtin
#define __has_builtin(x) 0
#endif
#if __has_builtin(__builtin_expect) == 0
#define __builtin_expect(x)
#endif

#if defined(cl_intel_subgroups) && defined(cl_intel_subgroups_char)

typedef global char* global_aligned_char_ptr __attribute__((align_value(4)));

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
int  emu_sub_group_i8_i8_matrix_mad_k32(int  a, int8 b, int  acc)
{
    int res = acc;

    // TODO: this could use integer dot products instead?

    res = as_char4(sub_group_broadcast(a, 0)).x * as_char4(b.s0).x + res;
    res = as_char4(sub_group_broadcast(a, 0)).y * as_char4(b.s0).y + res;
    res = as_char4(sub_group_broadcast(a, 0)).z * as_char4(b.s0).z + res;
    res = as_char4(sub_group_broadcast(a, 0)).w * as_char4(b.s0).w + res;

    res = as_char4(sub_group_broadcast(a, 1)).x * as_char4(b.s1).x + res;
    res = as_char4(sub_group_broadcast(a, 1)).y * as_char4(b.s1).y + res;
    res = as_char4(sub_group_broadcast(a, 1)).z * as_char4(b.s1).z + res;
    res = as_char4(sub_group_broadcast(a, 1)).w * as_char4(b.s1).w + res;

    res = as_char4(sub_group_broadcast(a, 2)).x * as_char4(b.s2).x + res;
    res = as_char4(sub_group_broadcast(a, 2)).y * as_char4(b.s2).y + res;
    res = as_char4(sub_group_broadcast(a, 2)).z * as_char4(b.s2).z + res;
    res = as_char4(sub_group_broadcast(a, 2)).w * as_char4(b.s2).w + res;

    res = as_char4(sub_group_broadcast(a, 3)).x * as_char4(b.s3).x + res;
    res = as_char4(sub_group_broadcast(a, 3)).y * as_char4(b.s3).y + res;
    res = as_char4(sub_group_broadcast(a, 3)).z * as_char4(b.s3).z + res;
    res = as_char4(sub_group_broadcast(a, 3)).w * as_char4(b.s3).w + res;

    res = as_char4(sub_group_broadcast(a, 4)).x * as_char4(b.s4).x + res;
    res = as_char4(sub_group_broadcast(a, 4)).y * as_char4(b.s4).y + res;
    res = as_char4(sub_group_broadcast(a, 4)).z * as_char4(b.s4).z + res;
    res = as_char4(sub_group_broadcast(a, 4)).w * as_char4(b.s4).w + res;

    res = as_char4(sub_group_broadcast(a, 5)).x * as_char4(b.s5).x + res;
    res = as_char4(sub_group_broadcast(a, 5)).y * as_char4(b.s5).y + res;
    res = as_char4(sub_group_broadcast(a, 5)).z * as_char4(b.s5).z + res;
    res = as_char4(sub_group_broadcast(a, 5)).w * as_char4(b.s5).w + res;

    res = as_char4(sub_group_broadcast(a, 6)).x * as_char4(b.s6).x + res;
    res = as_char4(sub_group_broadcast(a, 6)).y * as_char4(b.s6).y + res;
    res = as_char4(sub_group_broadcast(a, 6)).z * as_char4(b.s6).z + res;
    res = as_char4(sub_group_broadcast(a, 6)).w * as_char4(b.s6).w + res;

    res = as_char4(sub_group_broadcast(a, 7)).x * as_char4(b.s7).x + res;
    res = as_char4(sub_group_broadcast(a, 7)).y * as_char4(b.s7).y + res;
    res = as_char4(sub_group_broadcast(a, 7)).z * as_char4(b.s7).z + res;
    res = as_char4(sub_group_broadcast(a, 7)).w * as_char4(b.s7).w + res;

    return res;
}

__attribute__((overloadable))
int2 emu_sub_group_i8_i8_matrix_mad_k32(int2 a, int8 b, int2 acc)
{
    int2 res;

    res.s0 = emu_sub_group_i8_i8_matrix_mad_k32(a.s0, b, acc.s0);
    res.s1 = emu_sub_group_i8_i8_matrix_mad_k32(a.s1, b, acc.s1);

    return res;
}

__attribute__((overloadable))
int4 emu_sub_group_i8_i8_matrix_mad_k32(int4 a, int8 b, int4 acc)
{
    int4 res;

    res.s0 = emu_sub_group_i8_i8_matrix_mad_k32(a.s0, b, acc.s0);
    res.s1 = emu_sub_group_i8_i8_matrix_mad_k32(a.s1, b, acc.s1);
    res.s2 = emu_sub_group_i8_i8_matrix_mad_k32(a.s2, b, acc.s2);
    res.s3 = emu_sub_group_i8_i8_matrix_mad_k32(a.s3, b, acc.s3);

    return res;
}

__attribute__((overloadable))
int8 emu_sub_group_i8_i8_matrix_mad_k32(int8 a, int8 b, int8 acc)
{
    int8 res;

    res.s0 = emu_sub_group_i8_i8_matrix_mad_k32(a.s0, b, acc.s0);
    res.s1 = emu_sub_group_i8_i8_matrix_mad_k32(a.s1, b, acc.s1);
    res.s2 = emu_sub_group_i8_i8_matrix_mad_k32(a.s2, b, acc.s2);
    res.s3 = emu_sub_group_i8_i8_matrix_mad_k32(a.s3, b, acc.s3);
    res.s4 = emu_sub_group_i8_i8_matrix_mad_k32(a.s4, b, acc.s4);
    res.s5 = emu_sub_group_i8_i8_matrix_mad_k32(a.s5, b, acc.s5);
    res.s6 = emu_sub_group_i8_i8_matrix_mad_k32(a.s6, b, acc.s6);
    res.s7 = emu_sub_group_i8_i8_matrix_mad_k32(a.s7, b, acc.s7);

    return res;
}

// Emulated SIMD16 dpas:
__attribute__((overloadable))
int  emu_sub_group_i8_i8_matrix_mad_k32(short  a, int8 b, int  acc)
{
    float res = acc;

    res = as_char2(sub_group_broadcast(a,  0)).x * as_char4(b.s0).x + res;
    res = as_char2(sub_group_broadcast(a,  0)).y * as_char4(b.s0).y + res;
    res = as_char2(sub_group_broadcast(a,  1)).x * as_char4(b.s0).z + res;
    res = as_char2(sub_group_broadcast(a,  1)).y * as_char4(b.s0).w + res;

    res = as_char2(sub_group_broadcast(a,  2)).x * as_char4(b.s1).x + res;
    res = as_char2(sub_group_broadcast(a,  2)).y * as_char4(b.s1).y + res;
    res = as_char2(sub_group_broadcast(a,  3)).x * as_char4(b.s1).z + res;
    res = as_char2(sub_group_broadcast(a,  3)).y * as_char4(b.s1).w + res;

    res = as_char2(sub_group_broadcast(a,  4)).x * as_char4(b.s2).x + res;
    res = as_char2(sub_group_broadcast(a,  4)).y * as_char4(b.s2).y + res;
    res = as_char2(sub_group_broadcast(a,  5)).x * as_char4(b.s2).z + res;
    res = as_char2(sub_group_broadcast(a,  5)).y * as_char4(b.s2).w + res;

    res = as_char2(sub_group_broadcast(a,  6)).x * as_char4(b.s3).x + res;
    res = as_char2(sub_group_broadcast(a,  6)).y * as_char4(b.s3).y + res;
    res = as_char2(sub_group_broadcast(a,  7)).x * as_char4(b.s3).z + res;
    res = as_char2(sub_group_broadcast(a,  7)).y * as_char4(b.s3).w + res;

    res = as_char2(sub_group_broadcast(a,  8)).x * as_char4(b.s4).x + res;
    res = as_char2(sub_group_broadcast(a,  8)).y * as_char4(b.s4).y + res;
    res = as_char2(sub_group_broadcast(a,  9)).x * as_char4(b.s4).z + res;
    res = as_char2(sub_group_broadcast(a,  9)).y * as_char4(b.s4).w + res;

    res = as_char2(sub_group_broadcast(a, 10)).x * as_char4(b.s5).x + res;
    res = as_char2(sub_group_broadcast(a, 10)).y * as_char4(b.s5).y + res;
    res = as_char2(sub_group_broadcast(a, 11)).x * as_char4(b.s5).z + res;
    res = as_char2(sub_group_broadcast(a, 11)).y * as_char4(b.s5).w + res;

    res = as_char2(sub_group_broadcast(a, 12)).x * as_char4(b.s6).x + res;
    res = as_char2(sub_group_broadcast(a, 12)).y * as_char4(b.s6).y + res;
    res = as_char2(sub_group_broadcast(a, 13)).x * as_char4(b.s6).z + res;
    res = as_char2(sub_group_broadcast(a, 13)).y * as_char4(b.s6).w + res;

    res = as_char2(sub_group_broadcast(a, 14)).x * as_char4(b.s7).x + res;
    res = as_char2(sub_group_broadcast(a, 14)).y * as_char4(b.s7).y + res;
    res = as_char2(sub_group_broadcast(a, 15)).x * as_char4(b.s7).z + res;
    res = as_char2(sub_group_broadcast(a, 15)).y * as_char4(b.s7).w + res;

    return res;
}

__attribute__((overloadable))
int2 emu_sub_group_i8_i8_matrix_mad_k32(short2 a, int8 b, int2 acc)
{
    int2 res;

    res.s0 = emu_sub_group_i8_i8_matrix_mad_k32(a.s0, b, acc.s0);
    res.s1 = emu_sub_group_i8_i8_matrix_mad_k32(a.s1, b, acc.s1);

    return res;
}

__attribute__((overloadable))
int4 emu_sub_group_i8_i8_matrix_mad_k32(short4 a, int8 b, int4 acc)
{
    int4 res;

    res.s0 = emu_sub_group_i8_i8_matrix_mad_k32(a.s0, b, acc.s0);
    res.s1 = emu_sub_group_i8_i8_matrix_mad_k32(a.s1, b, acc.s1);
    res.s2 = emu_sub_group_i8_i8_matrix_mad_k32(a.s2, b, acc.s2);
    res.s3 = emu_sub_group_i8_i8_matrix_mad_k32(a.s3, b, acc.s3);

    return res;
}

__attribute__((overloadable))
int8 emu_sub_group_i8_i8_matrix_mad_k32(short8 a, int8 b, int8 acc)
{
    int8 res;

    res.s0 = emu_sub_group_i8_i8_matrix_mad_k32(a.s0, b, acc.s0);
    res.s1 = emu_sub_group_i8_i8_matrix_mad_k32(a.s1, b, acc.s1);
    res.s2 = emu_sub_group_i8_i8_matrix_mad_k32(a.s2, b, acc.s2);
    res.s3 = emu_sub_group_i8_i8_matrix_mad_k32(a.s3, b, acc.s3);
    res.s4 = emu_sub_group_i8_i8_matrix_mad_k32(a.s4, b, acc.s4);
    res.s5 = emu_sub_group_i8_i8_matrix_mad_k32(a.s5, b, acc.s5);
    res.s6 = emu_sub_group_i8_i8_matrix_mad_k32(a.s6, b, acc.s6);
    res.s7 = emu_sub_group_i8_i8_matrix_mad_k32(a.s7, b, acc.s7);

    return res;
}

// M rows x K columns
// This is the SIMD8 version, where each work-item loads four values.
int  load_a_rowmajor_d8_m1_k32_sg8(global char* A, int rowStart, int colStart, int stride)
{
    int ret;

    global uint* A_ui = (global uint*)A;
    uint offset_ui = rowStart * stride / 4 + colStart / 4;
    ret = intel_sub_group_block_read(A_ui + offset_ui);

    return ret;
}

// M rows x K columns
// This is the SIMD8 version, where each work-item loads four values.
int2 load_a_rowmajor_d8_m2_k32_sg8(global char* A, int rowStart, int colStart, int stride)
{
    int2 ret;

    global uint* A_ui = (global uint*)A;
    uint offset_ui = rowStart * stride / 4 + colStart / 4;

    ret.s0 = intel_sub_group_block_read(A_ui + offset_ui); offset_ui += stride / 4;
    ret.s1 = intel_sub_group_block_read(A_ui + offset_ui); offset_ui += stride / 4;

    return ret;
}

// M rows x K columns
// This is the SIMD8 version, where each work-item loads four values.
int4 load_a_rowmajor_d8_m4_k32_sg8(global char* A, int rowStart, int colStart, int stride)
{
    int4 ret;

    global uint* A_ui = (global uint*)A;
    uint offset_ui = rowStart * stride / 4 + colStart / 4;

    ret.s0 = intel_sub_group_block_read(A_ui + offset_ui); offset_ui += stride / 4;
    ret.s1 = intel_sub_group_block_read(A_ui + offset_ui); offset_ui += stride / 4;
    ret.s2 = intel_sub_group_block_read(A_ui + offset_ui); offset_ui += stride / 4;
    ret.s3 = intel_sub_group_block_read(A_ui + offset_ui); offset_ui += stride / 4;

    return ret;
}

// M rows x K columns
// This is the SIMD8 version, where each work-item loads four values.
int8 load_a_rowmajor_d8_m8_k32_sg8(global char* A, int rowStart, int colStart, int stride)
{
    int8 ret;

    global uint* A_ui = (global uint*)A;
    uint offset_ui = rowStart * stride / 4 + colStart / 4;

    ret.s0 = intel_sub_group_block_read(A_ui + offset_ui); offset_ui += stride / 4;
    ret.s1 = intel_sub_group_block_read(A_ui + offset_ui); offset_ui += stride / 4;
    ret.s2 = intel_sub_group_block_read(A_ui + offset_ui); offset_ui += stride / 4;
    ret.s3 = intel_sub_group_block_read(A_ui + offset_ui); offset_ui += stride / 4;
    ret.s4 = intel_sub_group_block_read(A_ui + offset_ui); offset_ui += stride / 4;
    ret.s5 = intel_sub_group_block_read(A_ui + offset_ui); offset_ui += stride / 4;
    ret.s6 = intel_sub_group_block_read(A_ui + offset_ui); offset_ui += stride / 4;
    ret.s7 = intel_sub_group_block_read(A_ui + offset_ui); offset_ui += stride / 4;

    return ret;
}

#if 0

// M rows x K columns x V tiles (in the K dimension)
// This is the SIMD8 version, where each work-item loads two values.
// The first tile is returned the first components of the return value, the the next tile, etc.
int16 load_a_rowmajor_d16_m8_k16v2_sg8(global ushort* A, int rowStart, int colStart, int stride)
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
void prefetch_a_rowmajor_d16_m8_k16v2_sg8(global ushort* A, int rowStart, int colStart, int stride)
{
#if defined(PREFETCH_DEFAULT)
    uint offset = colStart + (rowStart + get_sub_group_local_id()) * stride;
    __builtin_assume((ulong)(A + offset) % 4 == 0);
    prefetch(A + offset, 2);
#endif // defined(PREFETCH_DEFAULT)
}

#endif

// M rows x K columns
// This is the SIMD16 version, where each work-item loads two values.
short load_a_rowmajor_d8_m1_k32_sg16(global char* A, int rowStart, int colStart, int stride)
{
    ushort ret;

    global ushort* A_us = (global ushort*)A;
    uint offset_us = rowStart * stride / 2 + colStart / 2;

    ret = intel_sub_group_block_read_us(A_us + offset_us);

    return as_short(ret);
}

// M rows x K columns
// This is the SIMD16 version, where each work-item loads two values.
short2 load_a_rowmajor_d8_m2_k32_sg16(global char* A, int rowStart, int colStart, int stride)
{
    ushort2 ret;

    global ushort* A_us = (global ushort*)A;
    uint offset_us = rowStart * stride / 2 + colStart / 2;

    ret.s0 = intel_sub_group_block_read_us(A_us + offset_us); offset_us += stride / 2;
    ret.s1 = intel_sub_group_block_read_us(A_us + offset_us); offset_us += stride / 2;

    return as_short2(ret);
}

// M rows x K columns
// This is the SIMD16 version, where each work-item loads two values.
short4 load_a_rowmajor_d8_m4_k32_sg16(global char* A, int rowStart, int colStart, int stride)
{
    ushort4 ret;

    global ushort* A_us = (global ushort*)A;
    uint offset_us = rowStart * stride / 2 + colStart / 2;

    ret.s0 = intel_sub_group_block_read_us(A_us + offset_us); offset_us += stride / 2;
    ret.s1 = intel_sub_group_block_read_us(A_us + offset_us); offset_us += stride / 2;
    ret.s2 = intel_sub_group_block_read_us(A_us + offset_us); offset_us += stride / 2;
    ret.s3 = intel_sub_group_block_read_us(A_us + offset_us); offset_us += stride / 2;

    return as_short4(ret);
}

// M rows x K columns
// This is the SIMD16 version, where each work-item loads two values.
short8 load_a_rowmajor_d8_m8_k32_sg16(global char* A, int rowStart, int colStart, int stride)
{
    ushort8 ret;

    global ushort* A_us = (global ushort*)A;
    uint offset_us = rowStart * stride / 2 + colStart / 2;

    ret.s0 = intel_sub_group_block_read_us(A_us + offset_us); offset_us += stride / 2;
    ret.s1 = intel_sub_group_block_read_us(A_us + offset_us); offset_us += stride / 2;
    ret.s2 = intel_sub_group_block_read_us(A_us + offset_us); offset_us += stride / 2;
    ret.s3 = intel_sub_group_block_read_us(A_us + offset_us); offset_us += stride / 2;
    ret.s4 = intel_sub_group_block_read_us(A_us + offset_us); offset_us += stride / 2;
    ret.s5 = intel_sub_group_block_read_us(A_us + offset_us); offset_us += stride / 2;
    ret.s6 = intel_sub_group_block_read_us(A_us + offset_us); offset_us += stride / 2;
    ret.s7 = intel_sub_group_block_read_us(A_us + offset_us); offset_us += stride / 2;

    return as_short8(ret);
}

#if 0

// M rows x K columns x V tiles (in the K dimension)
// This is the SIMD16 version, where each work-item loads one value.
// The first tile is returned the first components of the return value, the the next tile, etc.
short16 load_a_rowmajor_d16_m8_k16v2_sg16(global ushort* A, int rowStart, int colStart, int stride)
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
void prefetch_a_rowmajor_d16_m8v2_k16v2_sg16(global ushort* A, int rowStart, int colStart, int stride)
{
#if defined(PREFETCH_DEFAULT)
    uint offset = colStart + (rowStart + get_sub_group_local_id()) * stride;
    __builtin_assume((ulong)(A + offset) % 4 == 0);
    prefetch(A + offset, 2);
#endif // defined(PREFETCH_DEFAULT)
}

#endif

// K rows x N columns:
// Each work-item loads K values and converts to VNNI.
// Stride is in units of elements.
int8 load_b_rowmajor_d8_k32_nx(global char* B, int rowStart, int colStart, int stride)
{
    int8 ret;

    global uchar* B_uc = (global uchar*)B;
    uint offset = rowStart * stride + colStart;

    uchar row0  = intel_sub_group_block_read_uc(B_uc + offset); offset += stride;
    uchar row1  = intel_sub_group_block_read_uc(B_uc + offset); offset += stride;
    uchar row2  = intel_sub_group_block_read_uc(B_uc + offset); offset += stride;
    uchar row3  = intel_sub_group_block_read_uc(B_uc + offset); offset += stride;
    uchar row4  = intel_sub_group_block_read_uc(B_uc + offset); offset += stride;
    uchar row5  = intel_sub_group_block_read_uc(B_uc + offset); offset += stride;
    uchar row6  = intel_sub_group_block_read_uc(B_uc + offset); offset += stride;
    uchar row7  = intel_sub_group_block_read_uc(B_uc + offset); offset += stride;
    uchar row8  = intel_sub_group_block_read_uc(B_uc + offset); offset += stride;
    uchar row9  = intel_sub_group_block_read_uc(B_uc + offset); offset += stride;
    uchar row10 = intel_sub_group_block_read_uc(B_uc + offset); offset += stride;
    uchar row11 = intel_sub_group_block_read_uc(B_uc + offset); offset += stride;
    uchar row12 = intel_sub_group_block_read_uc(B_uc + offset); offset += stride;
    uchar row13 = intel_sub_group_block_read_uc(B_uc + offset); offset += stride;
    uchar row14 = intel_sub_group_block_read_uc(B_uc + offset); offset += stride;
    uchar row15 = intel_sub_group_block_read_uc(B_uc + offset); offset += stride;
    uchar row16 = intel_sub_group_block_read_uc(B_uc + offset); offset += stride;
    uchar row17 = intel_sub_group_block_read_uc(B_uc + offset); offset += stride;
    uchar row18 = intel_sub_group_block_read_uc(B_uc + offset); offset += stride;
    uchar row19 = intel_sub_group_block_read_uc(B_uc + offset); offset += stride;
    uchar row20 = intel_sub_group_block_read_uc(B_uc + offset); offset += stride;
    uchar row21 = intel_sub_group_block_read_uc(B_uc + offset); offset += stride;
    uchar row22 = intel_sub_group_block_read_uc(B_uc + offset); offset += stride;
    uchar row23 = intel_sub_group_block_read_uc(B_uc + offset); offset += stride;
    uchar row24 = intel_sub_group_block_read_uc(B_uc + offset); offset += stride;
    uchar row25 = intel_sub_group_block_read_uc(B_uc + offset); offset += stride;
    uchar row26 = intel_sub_group_block_read_uc(B_uc + offset); offset += stride;
    uchar row27 = intel_sub_group_block_read_uc(B_uc + offset); offset += stride;
    uchar row28 = intel_sub_group_block_read_uc(B_uc + offset); offset += stride;
    uchar row29 = intel_sub_group_block_read_uc(B_uc + offset); offset += stride;
    uchar row30 = intel_sub_group_block_read_uc(B_uc + offset); offset += stride;
    uchar row31 = intel_sub_group_block_read_uc(B_uc + offset); offset += stride;

    ret.s0 = as_int((uchar4)(row0,  row1,  row2,  row3));
    ret.s1 = as_int((uchar4)(row4,  row5,  row6,  row7));
    ret.s2 = as_int((uchar4)(row8,  row9,  row10, row11));
    ret.s3 = as_int((uchar4)(row12, row13, row14, row15));
    ret.s4 = as_int((uchar4)(row16, row17, row18, row19));
    ret.s5 = as_int((uchar4)(row20, row21, row22, row23));
    ret.s6 = as_int((uchar4)(row24, row25, row26, row27));
    ret.s7 = as_int((uchar4)(row28, row29, row30, row31));

    return ret;
}

// K rows x N columns:
// Each work-item loads K values that has already been converted to VNNI.
// Stride is in units of elements.
int8 load_b_vnni_d8_k32_nx(global char* B, int rowStart, int colStart, int stride)
{
    int8 ret;

    global uint* B_ui = (global uint*)B;
    uint offset_ui = rowStart / 4 * stride + colStart;

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

#if 0

// K rows x N columns x V tiles (in the N dimension)
void prefetch_b_rowmajor_d16_k16_n8v4_sg8(global ushort* B, int rowStart, int colStart, int stride)
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
void prefetch_b_rowmajor_d16_k16_n16v2_sg16(global ushort* B, int rowStart, int colStart, int stride)
{
#if defined(PREFETCH_DEFAULT)
    uint offset = colStart + (rowStart + get_sub_group_local_id()) * stride;
    __builtin_assume((ulong)(B + offset) % 4 == 0);
    prefetch(B + offset, 2);
#endif // defined(PREFETCH_DEFAULT)
}

// K rows x N columns x V tiles (in the N dimension)
void prefetch_b_vnni_d16_k16_n8v2_sg8(global ushort* B, int rowStart, int colStart, int stride)
{
#if defined(PREFETCH_DEFAULT)
    global uint* B_ui = (global uint*)B;
    uint offset_ui = colStart + (rowStart / 2 + get_sub_group_local_id()) * stride;
    __builtin_assume((ulong)(B_ui + offset_ui) % 4 == 0);
    prefetch(B_ui + offset_ui, 1);
#endif // defined(PREFETCH_DEFAULT)
}

// K rows x N columns x V tiles (in the K dimension)
void prefetch_b_vnni_d16_k16v2_n16_sg16(global ushort* B, int rowStart, int colStart, int stride)
{
#if defined(PREFETCH_DEFAULT)
    global uint* B_ui = (global uint*)B;
    uint offset_ui = colStart + (rowStart / 2 + get_sub_group_local_id()) * stride;
    __builtin_assume((ulong)(B_ui + offset_ui) % 4 == 0);
    prefetch(B_ui + offset_ui, 1);
#endif // defined(PREFETCH_DEFAULT)
}

#endif

void store_c_rowmajor_int32_m1_nx(global int* C, int v, int rowStart, int colStart, int stride)
{
    global uint* C_ui = (global uint*)C;
    uint v_ui = as_uint(v);

    uint offset = rowStart * stride + colStart;

    intel_sub_group_block_write(C_ui + offset, v_ui); offset += stride;
}

void store_c_rowmajor_int32_m2_nx(global int* C, int2 v, int rowStart, int colStart, int stride)
{
    global uint* C_ui = (global uint*)C;
    uint2 v_ui = as_uint2(v);

    uint offset = rowStart * stride + colStart;

    intel_sub_group_block_write(C_ui + offset, v_ui.s0); offset += stride;
    intel_sub_group_block_write(C_ui + offset, v_ui.s1); offset += stride;
}

void store_c_rowmajor_int32_m4_nx(global int* C, int4 v, int rowStart, int colStart, int stride)
{
    global uint* C_ui = (global uint*)C;
    uint4 v_ui = as_uint4(v);

    uint offset = rowStart * stride + colStart;

    intel_sub_group_block_write(C_ui + offset, v_ui.s0); offset += stride;
    intel_sub_group_block_write(C_ui + offset, v_ui.s1); offset += stride;
    intel_sub_group_block_write(C_ui + offset, v_ui.s2); offset += stride;
    intel_sub_group_block_write(C_ui + offset, v_ui.s3); offset += stride;
}

void store_c_rowmajor_int32_m8_nx(global int* C, int8 v, int rowStart, int colStart, int stride)
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

#if 0
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

enum LSC_LDCC {
    LSC_LDCC_DEFAULT      = 0,
    LSC_LDCC_L1UC_L3UC    = 1,   // Override to L1 uncached and L3 uncached
    LSC_LDCC_L1UC_L3C     = 2,   // Override to L1 uncached and L3 cached
    LSC_LDCC_L1C_L3UC     = 3,   // Override to L1 cached and L3 uncached
    LSC_LDCC_L1C_L3C      = 4,   // Override to L1 cached and L3 cached
    LSC_LDCC_L1S_L3UC     = 5,   // Override to L1 streaming load and L3 uncached
    LSC_LDCC_L1S_L3C      = 6,   // Override to L1 streaming load and L3 cached
    LSC_LDCC_L1IAR_L3C    = 7,   // Override to L1 invalidate-after-read, and L3 cached
};

typedef ushort __attribute__((ext_vector_type(32))) ushort32;
typedef ushort __attribute__((ext_vector_type(64))) ushort64;

typedef uint __attribute__((ext_vector_type(32))) uint32;

// Define block reads, prefetches, and writes.  These are supported by the hardware but are not in the headers:

ushort   __builtin_IB_subgroup_block_read_flat_u16_m1k16v1(long baseoffset, int width_minus_one, int height_minus_one, int pitch_minus_one, int2 coord);
ushort2  __builtin_IB_subgroup_block_read_flat_u16_m2k16v1(long baseoffset, int width_minus_one, int height_minus_one, int pitch_minus_one, int2 coord);
ushort4  __builtin_IB_subgroup_block_read_flat_u16_m4k16v1(long baseoffset, int width_minus_one, int height_minus_one, int pitch_minus_one, int2 coord);
ushort8  __builtin_IB_subgroup_block_read_flat_u16_m8k16v1(long baseoffset, int width_minus_one, int height_minus_one, int pitch_minus_one, int2 coord);
ushort16 __builtin_IB_subgroup_block_read_flat_u16_m16k16v1(long baseoffset, int width_minus_one, int height_minus_one, int pitch_minus_one, int2 coord);
ushort32 __builtin_IB_subgroup_block_read_flat_u16_m32k16v1(long baseoffset, int width_minus_one, int height_minus_one, int pitch_minus_one, int2 coord);

ushort32 __builtin_IB_subgroup_block_read_flat_u16_m16k16v2(long baseoffset, int width_minus_one, int height_minus_one, int pitch_minus_one, int2 coord);
ushort64 __builtin_IB_subgroup_block_read_flat_u16_m32k16v2(long baseoffset, int width_minus_one, int height_minus_one, int pitch_minus_one, int2 coord);

uint8  __builtin_IB_subgroup_block_read_flat_u32_m8k16v1(long baseoffset, int width_minus_one, int height_minus_one, int pitch_minus_one, int2 coord);
uint16 __builtin_IB_subgroup_block_read_flat_u32_m16k16v1(long baseoffset, int width_minus_one, int height_minus_one, int pitch_minus_one, int2 coord);

uint16  __builtin_IB_subgroup_block_read_flat_transform_u16_k32(long baseoffset, int width_minus_one, int height_minus_one, int pitch_minus_one, int2 coord);

uint16  __builtin_IB_subgroup_block_read_flat_transform_u16_k16v2(long baseoffset, int width_minus_one, int height_minus_one, int pitch_minus_one, int2 coord);
uint32  __builtin_IB_subgroup_block_read_flat_transform_u16_k32v2(long baseoffset, int width_minus_one, int height_minus_one, int pitch_minus_one, int2 coord);


void __builtin_IB_subgroup_block_read_prefetch_u16_m1k16v1(long baseoffset, int width_minus_one, int height_minus_one, int pitch_minus_one, int2 coord, enum LSC_LDCC cache_control);
void __builtin_IB_subgroup_block_read_prefetch_u16_m2k16v1(long baseoffset, int width_minus_one, int height_minus_one, int pitch_minus_one, int2 coord, enum LSC_LDCC cache_control);
void __builtin_IB_subgroup_block_read_prefetch_u16_m4k16v1(long baseoffset, int width_minus_one, int height_minus_one, int pitch_minus_one, int2 coord, enum LSC_LDCC cache_control);
void __builtin_IB_subgroup_block_read_prefetch_u16_m8k16v1(long baseoffset, int width_minus_one, int height_minus_one, int pitch_minus_one, int2 coord, enum LSC_LDCC cache_control);
void __builtin_IB_subgroup_block_read_prefetch_u16_m8k16v2(long baseoffset, int width_minus_one, int height_minus_one, int pitch_minus_one, int2 coord, enum LSC_LDCC cache_control);
void __builtin_IB_subgroup_block_read_prefetch_u16_m16k16v1(long baseoffset, int width_minus_one, int height_minus_one, int pitch_minus_one, int2 coord, enum LSC_LDCC cache_control);
void __builtin_IB_subgroup_block_read_prefetch_u16_m32k16v1(long baseoffset, int width_minus_one, int height_minus_one, int pitch_minus_one, int2 coord, enum LSC_LDCC cache_control);

void __builtin_IB_subgroup_block_read_prefetch_u16_m16k16v2(long baseoffset, int width_minus_one, int height_minus_one, int pitch_minus_one, int2 coord, enum LSC_LDCC cache_control);
void __builtin_IB_subgroup_block_read_prefetch_u16_m32k16v2(long baseoffset, int width_minus_one, int height_minus_one, int pitch_minus_one, int2 coord, enum LSC_LDCC cache_control);

void __builtin_IB_subgroup_block_read_prefetch_u32_m8k16v1(long baseoffset, int width_minus_one, int height_minus_one, int pitch_minus_one, int2 coord, enum LSC_LDCC cache_control);
void __builtin_IB_subgroup_block_read_prefetch_u32_m16k16v1(long baseoffset, int width_minus_one, int height_minus_one, int pitch_minus_one, int2 coord, enum LSC_LDCC cache_control);


void __builtin_IB_subgroup_block_write_flat_u32_m1k16v1(long baseoffset, int width_minus_one, int height_minus_one, int pitch_minus_one, int2 coord, uint  data);
void __builtin_IB_subgroup_block_write_flat_u32_m2k16v1(long baseoffset, int width_minus_one, int height_minus_one, int pitch_minus_one, int2 coord, uint2 data);
void __builtin_IB_subgroup_block_write_flat_u32_m4k16v1(long baseoffset, int width_minus_one, int height_minus_one, int pitch_minus_one, int2 coord, uint4 data);
void __builtin_IB_subgroup_block_write_flat_u32_m8k16v1(long baseoffset, int width_minus_one, int height_minus_one, int pitch_minus_one, int2 coord, uint8 data);
void __builtin_IB_subgroup_block_write_flat_u32_m16k16v1(long baseoffset, int width_minus_one, int height_minus_one, int pitch_minus_one, int2 coord, uint16 data);

ushort   intel_subgroup_block_read_u16_m1k16(const __global void *base_address, int width, int height, int pitch, int2 coord)
{
    return __builtin_IB_subgroup_block_read_flat_u16_m1k16v1(as_long(base_address), width - 1, height - 1, pitch - 1, coord);
}
ushort2  intel_subgroup_block_read_u16_m2k16(const __global void *base_address, int width, int height, int pitch, int2 coord)
{
    return __builtin_IB_subgroup_block_read_flat_u16_m2k16v1(as_long(base_address), width - 1, height - 1, pitch - 1, coord);
}
ushort4  intel_subgroup_block_read_u16_m4k16(const __global void *base_address, int width, int height, int pitch, int2 coord)
{
    return __builtin_IB_subgroup_block_read_flat_u16_m4k16v1(as_long(base_address), width - 1, height - 1, pitch - 1, coord);
}
ushort8  intel_subgroup_block_read_u16_m8k16(const __global void *base_address, int width, int height, int pitch, int2 coord)
{
    return __builtin_IB_subgroup_block_read_flat_u16_m8k16v1(as_long(base_address), width - 1, height - 1, pitch - 1, coord);
}
ushort16 intel_subgroup_block_read_u16_m16k16(const __global void *base_address, int width, int height, int pitch, int2 coord)
{
    return __builtin_IB_subgroup_block_read_flat_u16_m16k16v1(as_long(base_address), width - 1, height - 1, pitch - 1, coord);
}
void intel_subgroup_block_read_u16_m32k16(const __global void *base_address, int width, int height, int pitch, int2 coord, ushort8 dst[4])
{
    ushort32 tmp = __builtin_IB_subgroup_block_read_flat_u16_m32k16v1(as_long(base_address), width - 1, height - 1, pitch - 1, coord);
    dst[0] = tmp.lo.lo;
    dst[1] = tmp.lo.hi;
    dst[2] = tmp.hi.lo;
    dst[3] = tmp.hi.hi;
}

void intel_subgroup_block_read_u16_m16k16v2(const __global void *base_address, int width, int height, int pitch, int2 coord, ushort8 dst[2][2])
{
    ushort32 tmp = __builtin_IB_subgroup_block_read_flat_u16_m16k16v2(as_long(base_address), width - 1, height - 1, pitch - 1, coord);
    dst[0][0] = tmp.lo.lo;
    dst[0][1] = tmp.lo.hi;
    dst[1][0] = tmp.hi.lo;
    dst[1][1] = tmp.hi.hi;
}
void intel_subgroup_block_read_u16_m32k16v2(const __global void *base_address, int width, int height, int pitch, int2 coord, ushort8 dst[2][4])
{
    ushort64 tmp = __builtin_IB_subgroup_block_read_flat_u16_m32k16v2(as_long(base_address), width - 1, height - 1, pitch - 1, coord);
    dst[0][0] = tmp.lo.lo.lo;
    dst[0][1] = tmp.lo.lo.hi;
    dst[0][2] = tmp.lo.hi.lo;
    dst[0][3] = tmp.lo.hi.hi;
    dst[1][0] = tmp.hi.lo.lo;
    dst[1][1] = tmp.hi.lo.hi;
    dst[1][2] = tmp.hi.hi.lo;
    dst[1][3] = tmp.hi.hi.hi;
}

uint8 intel_subgroup_block_read_u32_m8k16(const __global void* base_address, int width, int height, int pitch, int2 coord)
{
    return __builtin_IB_subgroup_block_read_flat_u32_m8k16v1(as_long(base_address), width - 1, height - 1, pitch - 1, coord);
}
uint16 intel_subgroup_block_read_u32_m16k16(const __global void* base_address, int width, int height, int pitch, int2 coord)
{
    return __builtin_IB_subgroup_block_read_flat_u32_m16k16v1(as_long(base_address), width - 1, height - 1, pitch - 1, coord);
}

// Each block is K rows x N columns, where the K rows have been VNNI transformed.
int8 intel_subgroup_block_read_transform_u16_k16n16(__global void *base_address, int width, int height, int pitch, int2 coord)
{
    // Note: this function is in the headers, but is named confusingly and returns unsigned integers rather than signed integers:
    return as_int8(intel_subgroup_block_read_transform_u16_k16(base_address, width, height, pitch, coord));
}
int16 intel_subgroup_block_read_transform_u16_k32n16(__global void *base_address, int width, int height, int pitch, int2 coord)
{
    return as_int16(__builtin_IB_subgroup_block_read_flat_transform_u16_k32(as_long(base_address), width - 1, height - 1, pitch - 1, coord));
}
int16 intel_subgroup_block_read_transform_u16_k16n16v2(__global void *base_address, int width, int height, int pitch, int2 coord)
{
    return as_int16(__builtin_IB_subgroup_block_read_flat_transform_u16_k16v2(as_long(base_address), width - 1, height - 1, pitch - 1, coord));
}
void intel_subgroup_block_read_transform_u16_k32n16v2(__global void *base_address, int width, int height, int pitch, int2 coord, int8 dst[2][2])
{
    uint32 tmp = __builtin_IB_subgroup_block_read_flat_transform_u16_k32v2(as_long(base_address), width - 1, height - 1, pitch - 1, coord);
    dst[0][0] = as_int8(tmp.lo.lo);
    dst[0][1] = as_int8(tmp.lo.hi);
    dst[1][0] = as_int8(tmp.hi.lo);
    dst[1][1] = as_int8(tmp.hi.hi);
}


#define BLOCK_PREFETCH_CACHE_TYPE LSC_LDCC_L1C_L3C

void intel_subgroup_block_prefetch_u16_m1k16(const __global void *base_address, int width, int height, int pitch, int2 coord)
{
#if defined(PREFETCH_DEFAULT)
    __builtin_IB_subgroup_block_read_prefetch_u16_m1k16v1(as_long(base_address), width - 1, height - 1, pitch - 1, coord, BLOCK_PREFETCH_CACHE_TYPE);
#endif // defined(PREFETCH_DEFAULT)
}
void intel_subgroup_block_prefetch_u16_m2k16(const __global void *base_address, int width, int height, int pitch, int2 coord)
{
#if defined(PREFETCH_DEFAULT)
    __builtin_IB_subgroup_block_read_prefetch_u16_m2k16v1(as_long(base_address), width - 1, height - 1, pitch - 1, coord, BLOCK_PREFETCH_CACHE_TYPE);
#endif // defined(PREFETCH_DEFAULT)
}
void intel_subgroup_block_prefetch_u16_m4k16(const __global void *base_address, int width, int height, int pitch, int2 coord)
{
#if defined(PREFETCH_DEFAULT)
    __builtin_IB_subgroup_block_read_prefetch_u16_m4k16v1(as_long(base_address), width - 1, height - 1, pitch - 1, coord, BLOCK_PREFETCH_CACHE_TYPE);
#endif // defined(PREFETCH_DEFAULT)
}
void intel_subgroup_block_prefetch_u16_m8k16(const __global void *base_address, int width, int height, int pitch, int2 coord)
{
#if defined(PREFETCH_DEFAULT)
    __builtin_IB_subgroup_block_read_prefetch_u16_m8k16v1(as_long(base_address), width - 1, height - 1, pitch - 1, coord, BLOCK_PREFETCH_CACHE_TYPE);
#endif // defined(PREFETCH_DEFAULT)
}
void intel_subgroup_block_prefetch_u16_m8k16v2(__global void *base_address, int width, int height, int pitch, int2 coord)
{
#if defined(PREFETCH_DEFAULT)
    __builtin_IB_subgroup_block_read_prefetch_u16_m8k16v2(as_long(base_address), width - 1, height - 1, pitch - 1, coord, BLOCK_PREFETCH_CACHE_TYPE);
#endif // defined(PREFETCH_DEFAULT)
}
void intel_subgroup_block_prefetch_u16_m16k16(const __global void *base_address, int width, int height, int pitch, int2 coord)
{
#if defined(PREFETCH_DEFAULT)
    __builtin_IB_subgroup_block_read_prefetch_u16_m16k16v1(as_long(base_address), width - 1, height - 1, pitch - 1, coord, BLOCK_PREFETCH_CACHE_TYPE);
#endif // defined(PREFETCH_DEFAULT)
}
void intel_subgroup_block_prefetch_u16_m32k16(const __global void *base_address, int width, int height, int pitch, int2 coord)
{
#if defined(PREFETCH_DEFAULT)
    __builtin_IB_subgroup_block_read_prefetch_u16_m32k16v1(as_long(base_address), width - 1, height - 1, pitch - 1, coord, BLOCK_PREFETCH_CACHE_TYPE);
#endif // defined(PREFETCH_DEFAULT)
}
void intel_subgroup_block_prefetch_u16_m16k16v2(const __global void *base_address, int width, int height, int pitch, int2 coord)
{
#if defined(PREFETCH_DEFAULT)
    __builtin_IB_subgroup_block_read_prefetch_u16_m16k16v2(as_long(base_address), width - 1, height - 1, pitch - 1, coord, BLOCK_PREFETCH_CACHE_TYPE);
#endif // defined(PREFETCH_DEFAULT)
}
void intel_subgroup_block_prefetch_u16_m32k16v2(const __global void *base_address, int width, int height, int pitch, int2 coord)
{
#if defined(PREFETCH_DEFAULT)
    __builtin_IB_subgroup_block_read_prefetch_u16_m32k16v2(as_long(base_address), width - 1, height - 1, pitch - 1, coord, BLOCK_PREFETCH_CACHE_TYPE);
#endif // defined(PREFETCH_DEFAULT)
}
void intel_subgroup_block_prefetch_u32_m8k16(const __global void *base_address, int width, int height, int pitch, int2 coord)
{
#if defined(PREFETCH_DEFAULT)
    __builtin_IB_subgroup_block_read_prefetch_u32_m8k16v1(as_long(base_address), width - 1, height - 1, pitch - 1, coord, BLOCK_PREFETCH_CACHE_TYPE);
#endif // defined(PREFETCH_DEFAULT)
}
void intel_subgroup_block_prefetch_u32_m16k16(const __global void *base_address, int width, int height, int pitch, int2 coord)
{
#if defined(PREFETCH_DEFAULT)
    __builtin_IB_subgroup_block_read_prefetch_u32_m16k16v1(as_long(base_address), width - 1, height - 1, pitch - 1, coord, BLOCK_PREFETCH_CACHE_TYPE);
#endif // defined(PREFETCH_DEFAULT)
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
void intel_subgroup_block_write_u32_m16k16(__global void* base_address, int width, int height, int pitch, int2 coord, uint16 data)
{
    __builtin_IB_subgroup_block_write_flat_u32_m16k16v1(as_long(base_address), width - 1, height - 1, pitch - 1, coord, data);
}

#endif // cl_intel_subgroup_extended_block_read
#endif
