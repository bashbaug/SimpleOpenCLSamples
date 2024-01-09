#define OVLD __attribute__((overloadable))

#if EMULATE_tN8
#define mat_mul_x8  my_sub_group_bf16_bf16_matrix_mad_k16
#else
#define mat_mul_x8  intel_sub_group_bf16_bf16_matrix_mad_k16
#endif

#if EMULATE_tN16
#define mat_mul_x16 my_sub_group_bf16_bf16_matrix_mad_k16
#else
#define mat_mul_x16 intel_sub_group_bf16_bf16_matrix_mad_k16
#endif

float bf16_to_fp32(ushort u)
{
#if defined(cl_intel_bfloat16_conversions)
    return intel_convert_as_bfloat16_float(u);
#else
    return as_float(u << 16);
#endif
}

kernel void bfloat16_naive(global float* C, global ushort* A, global ushort* B, int K)
{
    const int N = get_global_size(0);
    int m = get_global_id(1);
    int n = get_global_id(0);

    float sum = 0;
    for (int k = 0; k < K; k++) {
        sum = fma(bf16_to_fp32(A[m * K + k]), bf16_to_fp32(B[k * N + n]), sum);
    }

    C[m * N + n] = sum;
}

#if defined(cl_intel_subgroups) && defined(cl_intel_subgroups_short) && defined(cl_intel_required_subgroup_size)

// These are non-block read versions.
// They work on DG2 and PVC, and on other devices when emulated.

// SIMD8 versions:
static float  OVLD my_sub_group_bf16_bf16_matrix_mad_k16(int  a, int8 b, float  acc)
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

static float2 OVLD my_sub_group_bf16_bf16_matrix_mad_k16(int2 a, int8 b, float2 acc)
{
    float2 res;

    res.s0 = my_sub_group_bf16_bf16_matrix_mad_k16(a.s0, b, acc.s0);
    res.s1 = my_sub_group_bf16_bf16_matrix_mad_k16(a.s1, b, acc.s1);

    return res;
}

static float4 OVLD my_sub_group_bf16_bf16_matrix_mad_k16(int4 a, int8 b, float4 acc)
{
    float4 res;

    res.s0 = my_sub_group_bf16_bf16_matrix_mad_k16(a.s0, b, acc.s0);
    res.s1 = my_sub_group_bf16_bf16_matrix_mad_k16(a.s1, b, acc.s1);
    res.s2 = my_sub_group_bf16_bf16_matrix_mad_k16(a.s2, b, acc.s2);
    res.s3 = my_sub_group_bf16_bf16_matrix_mad_k16(a.s3, b, acc.s3);

    return res;
}

static float8 OVLD my_sub_group_bf16_bf16_matrix_mad_k16(int8 a, int8 b, float8 acc)
{
    float8 res;

    res.s0 = my_sub_group_bf16_bf16_matrix_mad_k16(a.s0, b, acc.s0);
    res.s1 = my_sub_group_bf16_bf16_matrix_mad_k16(a.s1, b, acc.s1);
    res.s2 = my_sub_group_bf16_bf16_matrix_mad_k16(a.s2, b, acc.s2);
    res.s3 = my_sub_group_bf16_bf16_matrix_mad_k16(a.s3, b, acc.s3);
    res.s4 = my_sub_group_bf16_bf16_matrix_mad_k16(a.s4, b, acc.s4);
    res.s5 = my_sub_group_bf16_bf16_matrix_mad_k16(a.s5, b, acc.s5);
    res.s6 = my_sub_group_bf16_bf16_matrix_mad_k16(a.s6, b, acc.s6);
    res.s7 = my_sub_group_bf16_bf16_matrix_mad_k16(a.s7, b, acc.s7);

    return res;
}

// SIMD16 versions:
static float  OVLD my_sub_group_bf16_bf16_matrix_mad_k16(short  a, int8 b, float  acc)
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

static float2 OVLD my_sub_group_bf16_bf16_matrix_mad_k16(short2 a, int8 b, float2 acc)
{
    float2 res;

    res.s0 = my_sub_group_bf16_bf16_matrix_mad_k16(a.s0, b, acc.s0);
    res.s1 = my_sub_group_bf16_bf16_matrix_mad_k16(a.s1, b, acc.s1);

    return res;
}

static float4 OVLD my_sub_group_bf16_bf16_matrix_mad_k16(short4 a, int8 b, float4 acc)
{
    float4 res;

    res.s0 = my_sub_group_bf16_bf16_matrix_mad_k16(a.s0, b, acc.s0);
    res.s1 = my_sub_group_bf16_bf16_matrix_mad_k16(a.s1, b, acc.s1);
    res.s2 = my_sub_group_bf16_bf16_matrix_mad_k16(a.s2, b, acc.s2);
    res.s3 = my_sub_group_bf16_bf16_matrix_mad_k16(a.s3, b, acc.s3);

    return res;
}

static float8 OVLD my_sub_group_bf16_bf16_matrix_mad_k16(short8 a, int8 b, float8 acc)
{
    float8 res;

    res.s0 = my_sub_group_bf16_bf16_matrix_mad_k16(a.s0, b, acc.s0);
    res.s1 = my_sub_group_bf16_bf16_matrix_mad_k16(a.s1, b, acc.s1);
    res.s2 = my_sub_group_bf16_bf16_matrix_mad_k16(a.s2, b, acc.s2);
    res.s3 = my_sub_group_bf16_bf16_matrix_mad_k16(a.s3, b, acc.s3);
    res.s4 = my_sub_group_bf16_bf16_matrix_mad_k16(a.s4, b, acc.s4);
    res.s5 = my_sub_group_bf16_bf16_matrix_mad_k16(a.s5, b, acc.s5);
    res.s6 = my_sub_group_bf16_bf16_matrix_mad_k16(a.s6, b, acc.s6);
    res.s7 = my_sub_group_bf16_bf16_matrix_mad_k16(a.s7, b, acc.s7);

    return res;
}

// M rows x K columns
// This is the SIMD8 version, where each work-item loads two values.
static int __load_a_row_major_bf16_k16_m1_x8(global ushort* A, int rowStart, int colStart, int stride)
{
    int ret;

    global uint* A_ui = (global uint*)A;
    int offset_ui = rowStart * stride / 2 + colStart / 2;
    ret = intel_sub_group_block_read(A_ui + offset_ui);

    return ret;
}

// M rows x K columns
// This is the SIMD8 version, where each work-item loads two values.
static int2 __load_a_row_major_bf16_k16_m2_x8(global ushort* A, int rowStart, int colStart, int stride)
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
static int4 __load_a_row_major_bf16_k16_m4_x8(global ushort* A, int rowStart, int colStart, int stride)
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
static int8 __load_a_row_major_bf16_k16_m8_x8(global ushort* A, int rowStart, int colStart, int stride)
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
static short __load_a_row_major_bf16_k16_m1_x16(global ushort* A, int rowStart, int colStart, int stride)
{
    ushort ret;

    int offset = rowStart * stride + colStart;
    ret = intel_sub_group_block_read_us(A + offset);

    return as_short(ret);
}

// M rows x K columns
// This is the SIMD16 version, where each work-item loads one values.
static short2 __load_a_row_major_bf16_k16_m2_x16(global ushort* A, int rowStart, int colStart, int stride)
{
    ushort2 ret;

    int offset = rowStart * stride + colStart;
    ret.s0 = intel_sub_group_block_read_us(A + offset); offset += stride;
    ret.s1 = intel_sub_group_block_read_us(A + offset); offset += stride;

    return as_short2(ret);
}

// M rows x K columns
// This is the SIMD16 version, where each work-item loads one values.
static short4 __load_a_row_major_bf16_k16_m4_x16(global ushort* A, int rowStart, int colStart, int stride)
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
static short8 __load_a_row_major_bf16_k16_m8_x16(global ushort* A, int rowStart, int colStart, int stride)
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
static int8 __load_b_row_major_bf16_k16(global ushort* B, int rowStart, int colStart, int stride)
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
static int8 __load_b_vnni_bf16_k16(global ushort* B, int rowStart, int colStart, int stride)
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

static void __store_c_row_major_fp32_m1(global float* C, float v, int rowStart, int colStart, int stride)
{
    global uint* C_ui = (global uint*)C;
    uint v_ui = as_uint(v);

    int offset = rowStart * stride + colStart;

    intel_sub_group_block_write(C_ui + offset, v_ui); offset += stride;
}

static void __store_c_row_major_fp32_m2(global float* C, float2 v, int rowStart, int colStart, int stride)
{
    global uint* C_ui = (global uint*)C;
    uint2 v_ui = as_uint2(v);

    int offset = rowStart * stride + colStart;

    intel_sub_group_block_write(C_ui + offset, v_ui.s0); offset += stride;
    intel_sub_group_block_write(C_ui + offset, v_ui.s1); offset += stride;
}

static void __store_c_row_major_fp32_m4(global float* C, float4 v, int rowStart, int colStart, int stride)
{
    global uint* C_ui = (global uint*)C;
    uint4 v_ui = as_uint4(v);

    int offset = rowStart * stride + colStart;

    intel_sub_group_block_write(C_ui + offset, v_ui.s0); offset += stride;
    intel_sub_group_block_write(C_ui + offset, v_ui.s1); offset += stride;
    intel_sub_group_block_write(C_ui + offset, v_ui.s2); offset += stride;
    intel_sub_group_block_write(C_ui + offset, v_ui.s3); offset += stride;
}

static void __store_c_row_major_fp32_m8(global float* C, float8 v, int rowStart, int colStart, int stride)
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

#if HAS_SIMD8

__attribute__((intel_reqd_sub_group_size(8)))
__attribute__((reqd_work_group_size(8, 1, 1)))
kernel void bfloat16_dpas_rowmajor_m1_n8(global float* C, global ushort* A, global ushort* B, int K)
{
    const int N = get_global_size(0);
    int m = get_group_id(1);
    int n = get_group_id(0) * get_local_size(0);

    float sum = 0;
    for (int k = 0; k < K; k += 16) {
        int     aData = __load_a_row_major_bf16_k16_m1_x8(A, m, k, K);
        int8    bData = __load_b_row_major_bf16_k16(B, k, n, N);
        sum = mat_mul_x8(aData, bData, sum);
    }

    __store_c_row_major_fp32_m1(C, sum, m, n, N);
}

__attribute__((intel_reqd_sub_group_size(8)))
__attribute__((reqd_work_group_size(8, 1, 1)))
kernel void bfloat16_dpas_rowmajor_m2_n8(global float* C, global ushort* A, global ushort* B, int K)
{
    const int N = get_global_size(0);
    int m = get_group_id(1) * 2;
    int n = get_group_id(0) * get_local_size(0);

    float2 sum = 0;
    for (int k = 0; k < K; k += 16) {
        int2    aData = __load_a_row_major_bf16_k16_m2_x8(A, m, k, K);
        int8    bData = __load_b_row_major_bf16_k16(B, k, n, N);
        sum = mat_mul_x8(aData, bData, sum);
    }

    __store_c_row_major_fp32_m2(C, sum, m, n, N);
}

__attribute__((intel_reqd_sub_group_size(8)))
__attribute__((reqd_work_group_size(8, 1, 1)))
kernel void bfloat16_dpas_rowmajor_m4_n8(global float* C, global ushort* A, global ushort* B, int K)
{
    const int N = get_global_size(0);
    int m = get_group_id(1) * 4;
    int n = get_group_id(0) * get_local_size(0);

    float4 sum = 0;
    for (int k = 0; k < K; k += 16) {
        int4    aData = __load_a_row_major_bf16_k16_m4_x8(A, m, k, K);
        int8    bData = __load_b_row_major_bf16_k16(B, k, n, N);
        sum = mat_mul_x8(aData, bData, sum);
    }

    __store_c_row_major_fp32_m4(C, sum, m, n, N);
}

__attribute__((intel_reqd_sub_group_size(8)))
__attribute__((reqd_work_group_size(8, 1, 1)))
kernel void bfloat16_dpas_rowmajor_m8_n8(global float* C, global ushort* A, global ushort* B, int K)
{
    const int N = get_global_size(0);
    int m = get_group_id(1) * 8;
    int n = get_group_id(0) * get_local_size(0);

    float8 sum = 0;
    for (int k = 0; k < K; k += 16) {
        int8    aData = __load_a_row_major_bf16_k16_m8_x8(A, m, k, K);
        int8    bData = __load_b_row_major_bf16_k16(B, k, n, N);
        sum = mat_mul_x8(aData, bData, sum);
    }

    __store_c_row_major_fp32_m8(C, sum, m, n, N);
}

#endif // HAS_SIMD8

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16, 1, 1)))
kernel void bfloat16_dpas_rowmajor_m1_n16(global float* C, global ushort* A, global ushort* B, int K)
{
    const int N = get_global_size(0);
    int m = get_group_id(1);
    int n = get_group_id(0) * get_local_size(0);

    float sum = 0;
    for (int k = 0; k < K; k += 16) {
        short   aData = __load_a_row_major_bf16_k16_m1_x16(A, m, k, K);
        int8    bData = __load_b_row_major_bf16_k16(B, k, n, N);
        sum = mat_mul_x16(aData, bData, sum);
    }

    __store_c_row_major_fp32_m1(C, sum, m, n, N);
}

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16, 1, 1)))
kernel void bfloat16_dpas_rowmajor_m2_n16(global float* C, global ushort* A, global ushort* B, int K)
{
    const int N = get_global_size(0);
    int m = get_group_id(1) * 2;
    int n = get_group_id(0) * get_local_size(0);

    float2 sum = 0;
    for (int k = 0; k < K; k += 16) {
        short2  aData = __load_a_row_major_bf16_k16_m2_x16(A, m, k, K);
        int8    bData = __load_b_row_major_bf16_k16(B, k, n, N);
        sum = mat_mul_x16(aData, bData, sum);
    }

    __store_c_row_major_fp32_m2(C, sum, m, n, N);
}

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16, 1, 1)))
kernel void bfloat16_dpas_rowmajor_m4_n16(global float* C, global ushort* A, global ushort* B, int K)
{
    const int N = get_global_size(0);
    int m = get_group_id(1) * 4;
    int n = get_group_id(0) * get_local_size(0);

    float4 sum = 0;
    for (int k = 0; k < K; k += 16) {
        short4  aData = __load_a_row_major_bf16_k16_m4_x16(A, m, k, K);
        int8    bData = __load_b_row_major_bf16_k16(B, k, n, N);
        sum = mat_mul_x16(aData, bData, sum);
    }

    __store_c_row_major_fp32_m4(C, sum, m, n, N);
}

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16, 1, 1)))
kernel void bfloat16_dpas_rowmajor_m8_n16(global float* C, global ushort* A, global ushort* B, int K)
{
    const int N = get_global_size(0);
    int m = get_group_id(1) * 8;
    int n = get_group_id(0) * get_local_size(0);

    float8 sum = 0;
    for (int k = 0; k < K; k += 16) {
        short8  aData = __load_a_row_major_bf16_k16_m8_x16(A, m, k, K);
        int8    bData = __load_b_row_major_bf16_k16(B, k, n, N);
        sum = mat_mul_x16(aData, bData, sum);
    }

    __store_c_row_major_fp32_m8(C, sum, m, n, N);
}

#if HAS_SIMD8

__attribute__((intel_reqd_sub_group_size(8)))
__attribute__((reqd_work_group_size(8, 1, 1)))
kernel void bfloat16_dpas_vnni_m1_n8(global float* C, global ushort* A, global ushort* B, int K)
{
    const int N = get_global_size(0);
    int m = get_group_id(1);
    int n = get_group_id(0) * get_local_size(0);

    float sum = 0;
    for (int k = 0; k < K; k += 16) {
        int     aData = __load_a_row_major_bf16_k16_m1_x8(A, m, k, K);
        int8    bData = __load_b_vnni_bf16_k16(B, k, n, N);
        sum = mat_mul_x8(aData, bData, sum);
    }

    __store_c_row_major_fp32_m1(C, sum, m, n, N);
}

__attribute__((intel_reqd_sub_group_size(8)))
__attribute__((reqd_work_group_size(8, 1, 1)))
kernel void bfloat16_dpas_vnni_m2_n8(global float* C, global ushort* A, global ushort* B, int K)
{
    const int N = get_global_size(0);
    int m = get_group_id(1) * 2;
    int n = get_group_id(0) * get_local_size(0);

    float2 sum = 0;
    for (int k = 0; k < K; k += 16) {
        int2    aData = __load_a_row_major_bf16_k16_m2_x8(A, m, k, K);
        int8    bData = __load_b_vnni_bf16_k16(B, k, n, N);
        sum = mat_mul_x8(aData, bData, sum);
    }

    __store_c_row_major_fp32_m2(C, sum, m, n, N);
}

__attribute__((intel_reqd_sub_group_size(8)))
__attribute__((reqd_work_group_size(8, 1, 1)))
kernel void bfloat16_dpas_vnni_m4_n8(global float* C, global ushort* A, global ushort* B, int K)
{
    const int N = get_global_size(0);
    int m = get_group_id(1) * 4;
    int n = get_group_id(0) * get_local_size(0);

    float4 sum = 0;
    for (int k = 0; k < K; k += 16) {
        int4    aData = __load_a_row_major_bf16_k16_m4_x8(A, m, k, K);
        int8    bData = __load_b_vnni_bf16_k16(B, k, n, N);
        sum = mat_mul_x8(aData, bData, sum);
    }

    __store_c_row_major_fp32_m4(C, sum, m, n, N);
}

__attribute__((intel_reqd_sub_group_size(8)))
__attribute__((reqd_work_group_size(8, 1, 1)))
kernel void bfloat16_dpas_vnni_m8_n8(global float* C, global ushort* A, global ushort* B, int K)
{
    const int N = get_global_size(0);
    int m = get_group_id(1) * 8;
    int n = get_group_id(0) * get_local_size(0);

    float8 sum = 0;
    for (int k = 0; k < K; k += 16) {
        int8    aData = __load_a_row_major_bf16_k16_m8_x8(A, m, k, K);
        int8    bData = __load_b_vnni_bf16_k16(B, k, n, N);
        sum = mat_mul_x8(aData, bData, sum);
    }

    __store_c_row_major_fp32_m8(C, sum, m, n, N);
}

#endif // HAS_SIMD8

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16, 1, 1)))
kernel void bfloat16_dpas_vnni_m1_n16(global float* C, global ushort* A, global ushort* B, int K)
{
    const int N = get_global_size(0);
    int m = get_group_id(1);
    int n = get_group_id(0) * get_local_size(0);

    float sum = 0;
    for (int k = 0; k < K; k += 16) {
        short   aData = __load_a_row_major_bf16_k16_m1_x16(A, m, k, K);
        int8    bData = __load_b_vnni_bf16_k16(B, k, n, N);
        sum = mat_mul_x16(aData, bData, sum);
    }

    __store_c_row_major_fp32_m1(C, sum, m, n, N);
}

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16, 1, 1)))
kernel void bfloat16_dpas_vnni_m2_n16(global float* C, global ushort* A, global ushort* B, int K)
{
    const int N = get_global_size(0);
    int m = get_group_id(1) * 2;
    int n = get_group_id(0) * get_local_size(0);

    float2 sum = 0;
    for (int k = 0; k < K; k += 16) {
        short2  aData = __load_a_row_major_bf16_k16_m2_x16(A, m, k, K);
        int8    bData = __load_b_vnni_bf16_k16(B, k, n, N);
        sum = mat_mul_x16(aData, bData, sum);
    }

    __store_c_row_major_fp32_m2(C, sum, m, n, N);
}

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16, 1, 1)))
kernel void bfloat16_dpas_vnni_m4_n16(global float* C, global ushort* A, global ushort* B, int K)
{
    const int N = get_global_size(0);
    int m = get_group_id(1) * 4;
    int n = get_group_id(0) * get_local_size(0);

    float4 sum = 0;
    for (int k = 0; k < K; k += 16) {
        short4  aData = __load_a_row_major_bf16_k16_m4_x16(A, m, k, K);
        int8    bData = __load_b_vnni_bf16_k16(B, k, n, N);
        sum = mat_mul_x16(aData, bData, sum);
    }

    __store_c_row_major_fp32_m4(C, sum, m, n, N);
}

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16, 1, 1)))
kernel void bfloat16_dpas_vnni_m8_n16(global float* C, global ushort* A, global ushort* B, int K)
{
    const int N = get_global_size(0);
    int m = get_group_id(1) * 8;
    int n = get_group_id(0) * get_local_size(0);

    float8 sum = 0;
    for (int k = 0; k < K; k += 16) {
        short8  aData = __load_a_row_major_bf16_k16_m8_x16(A, m, k, K);
        int8    bData = __load_b_vnni_bf16_k16(B, k, n, N);
        sum = mat_mul_x16(aData, bData, sum);
    }

    __store_c_row_major_fp32_m8(C, sum, m, n, N);
}

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

void intel_subgroup_block_write_u32_m1k16v1(__global void* base_address, int width, int height, int pitch, int2 coord, uint data)
{
    __builtin_IB_subgroup_block_write_flat_u32_m1k16v1(as_long(base_address), width - 1, height - 1, pitch - 1, coord, data);
}
void intel_subgroup_block_write_u32_m2k16v1(__global void* base_address, int width, int height, int pitch, int2 coord, uint2 data)
{
    __builtin_IB_subgroup_block_write_flat_u32_m2k16v1(as_long(base_address), width - 1, height - 1, pitch - 1, coord, data);
}
void intel_subgroup_block_write_u32_m4k16v1(__global void* base_address, int width, int height, int pitch, int2 coord, uint4 data)
{
    __builtin_IB_subgroup_block_write_flat_u32_m4k16v1(as_long(base_address), width - 1, height - 1, pitch - 1, coord, data);
}
void intel_subgroup_block_write_u32_m8k16v1(__global void* base_address, int width, int height, int pitch, int2 coord, uint8 data)
{
    __builtin_IB_subgroup_block_write_flat_u32_m8k16v1(as_long(base_address), width - 1, height - 1, pitch - 1, coord, data);
}

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16, 1, 1)))
kernel void bfloat16_dpas_blockread_rowmajor_m1_n16(global float* C, global ushort* A, global ushort* B, int K)
{
    const int M = get_global_size(1);
    const int N = get_global_size(0);
    int m = get_group_id(1);
    int n = get_group_id(0) * get_local_size(0);

    float sum = 0;
    for (int k = 0; k < K; k += 16) {
        short   aData = as_short(intel_subgroup_block_read_u16_m1k16(A, K * sizeof(ushort), M, K * sizeof(ushort), (int2)(k, m)));
        int8    bData = as_int8(intel_subgroup_block_read_transform_u16_k16(B, N * sizeof(ushort), K, N * sizeof(ushort), (int2)(n, k)));
        sum = mat_mul_x16(aData, bData, sum);
    }

    intel_subgroup_block_write_u32_m1k16v1(C, N * sizeof(float), M, N * sizeof(float), (int2)(n, m), as_uint(sum));
}

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16, 1, 1)))
kernel void bfloat16_dpas_blockread_rowmajor_m2_n16(global float* C, global ushort* A, global ushort* B, int K)
{
    const int M = get_global_size(1) * 2;
    const int N = get_global_size(0);
    int m = get_group_id(1) * 2;
    int n = get_group_id(0) * get_local_size(0);

    float2 sum = 0;
    for (int k = 0; k < K; k += 16) {
        short2  aData = as_short2(intel_subgroup_block_read_u16_m2k16(A, K * sizeof(ushort), M, K * sizeof(ushort), (int2)(k, m)));
        int8    bData = as_int8(intel_subgroup_block_read_transform_u16_k16(B, N * sizeof(ushort), K, N * sizeof(ushort), (int2)(n, k)));
        sum = mat_mul_x16(aData, bData, sum);
    }

    intel_subgroup_block_write_u32_m2k16v1(C, N * sizeof(float), M, N * sizeof(float), (int2)(n, m), as_uint2(sum));
}

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16, 1, 1)))
kernel void bfloat16_dpas_blockread_rowmajor_m4_n16(global float* C, global ushort* A, global ushort* B, int K)
{
    const int M = get_global_size(1) * 4;
    const int N = get_global_size(0);
    int m = get_group_id(1) * 4;
    int n = get_group_id(0) * get_local_size(0);

    float4 sum = 0;
    for (int k = 0; k < K; k += 16) {
        short4  aData = as_short4(intel_subgroup_block_read_u16_m4k16(A, K * sizeof(ushort), M, K * sizeof(ushort), (int2)(k, m)));
        int8    bData = as_int8(intel_subgroup_block_read_transform_u16_k16(B, N * sizeof(ushort), K, N * sizeof(ushort), (int2)(n, k)));
        sum = mat_mul_x16(aData, bData, sum);
    }

    intel_subgroup_block_write_u32_m4k16v1(C, N * sizeof(float), M, N * sizeof(float), (int2)(n, m), as_uint4(sum));
}

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16, 1, 1)))
kernel void bfloat16_dpas_blockread_rowmajor_m8_n16(global float* C, global ushort* A, global ushort* B, int K)
{
    const int M = get_global_size(1) * 8;
    const int N = get_global_size(0);
    int m = get_group_id(1) * 8;
    int n = get_group_id(0) * get_local_size(0);

    float8 sum = 0;
    for (int k = 0; k < K; k += 16) {
        short8  aData = as_short8(intel_subgroup_block_read_u16_m8k16(A, K * sizeof(ushort), M, K * sizeof(ushort), (int2)(k, m)));
        int8    bData = as_int8(intel_subgroup_block_read_transform_u16_k16(B, N * sizeof(ushort), K, N * sizeof(ushort), (int2)(n, k)));
        sum = mat_mul_x16(aData, bData, sum);
    }

    intel_subgroup_block_write_u32_m8k16v1(C, N * sizeof(float), M, N * sizeof(float), (int2)(n, m), as_uint8(sum));
}

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16, 1, 1)))
kernel void bfloat16_dpas_blockread_vnni_m1_n16(global float* C, global ushort* A, global ushort* B, int K)
{
    const int M = get_global_size(1);
    const int N = get_global_size(0);
    int m = get_group_id(1);
    int n = get_group_id(0) * get_local_size(0);

    float sum = 0;
    for (int k = 0; k < K; k += 16) {
        short   aData = as_short(intel_subgroup_block_read_u16_m1k16(A, K * sizeof(ushort), M, K * sizeof(ushort), (int2)(k, m)));
        int8    bData = as_int8(intel_subgroup_block_read_u32_m8k16(B, N * sizeof(uint), K, N * sizeof(uint), (int2)(n, k / 2)));
        sum = mat_mul_x16(aData, bData, sum);
    }

    intel_subgroup_block_write_u32_m1k16v1(C, N * sizeof(float), M, N * sizeof(float), (int2)(n, m), as_uint(sum));
}

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16, 1, 1)))
kernel void bfloat16_dpas_blockread_vnni_m2_n16(global float* C, global ushort* A, global ushort* B, int K)
{
    const int M = get_global_size(1) * 2;
    const int N = get_global_size(0);
    int m = get_group_id(1) * 2;
    int n = get_group_id(0) * get_local_size(0);

    float2 sum = 0;
    for (int k = 0; k < K; k += 16) {
        short2  aData = as_short2(intel_subgroup_block_read_u16_m2k16(A, K * sizeof(ushort), M, K * sizeof(ushort), (int2)(k, m)));
        int8    bData = as_int8(intel_subgroup_block_read_u32_m8k16(B, N * sizeof(uint), K, N * sizeof(uint), (int2)(n, k / 2)));
        sum = mat_mul_x16(aData, bData, sum);
    }

    intel_subgroup_block_write_u32_m2k16v1(C, N * sizeof(float), M, N * sizeof(float), (int2)(n, m), as_uint2(sum));
}

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16, 1, 1)))
kernel void bfloat16_dpas_blockread_vnni_m4_n16(global float* C, global ushort* A, global ushort* B, int K)
{
    const int M = get_global_size(1) * 4;
    const int N = get_global_size(0);
    int m = get_group_id(1) * 4;
    int n = get_group_id(0) * get_local_size(0);

    float4 sum = 0;
    for (int k = 0; k < K; k += 16) {
        short4  aData = as_short4(intel_subgroup_block_read_u16_m4k16(A, K * sizeof(ushort), M, K * sizeof(ushort), (int2)(k, m)));
        int8    bData = as_int8(intel_subgroup_block_read_u32_m8k16(B, N * sizeof(uint), K, N * sizeof(uint), (int2)(n, k / 2)));
        sum = mat_mul_x16(aData, bData, sum);
    }

    intel_subgroup_block_write_u32_m4k16v1(C, N * sizeof(float), M, N * sizeof(float), (int2)(n, m), as_uint4(sum));
}

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16, 1, 1)))
kernel void bfloat16_dpas_blockread_vnni_m8_n16(global float* C, global ushort* A, global ushort* B, int K)
{
    const int M = get_global_size(1) * 8;
    const int N = get_global_size(0);
    int m = get_group_id(1) * 8;
    int n = get_group_id(0) * get_local_size(0);

    float8 sum = 0;
    for (int k = 0; k < K; k += 16) {
        short8  aData = as_short8(intel_subgroup_block_read_u16_m8k16(A, K * sizeof(ushort), M, K * sizeof(ushort), (int2)(k, m)));
        int8    bData = as_int8(intel_subgroup_block_read_u32_m8k16(B, N * sizeof(uint), K, N * sizeof(uint), (int2)(n, k / 2)));
        sum = mat_mul_x16(aData, bData, sum);
    }

    intel_subgroup_block_write_u32_m8k16v1(C, N * sizeof(float), M, N * sizeof(float), (int2)(n, m), as_uint8(sum));
}

#endif // cl_intel_subgroup_extended_block_read

#endif // defined(cl_intel_subgroups) && defined(cl_intel_subgroups_short) && defined(cl_intel_required_subgroup_size)

#undef OVLD
