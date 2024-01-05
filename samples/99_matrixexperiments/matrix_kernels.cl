float bfloat16_to_float(ushort u)
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
        sum = fma(bfloat16_to_float(A[m * K + k]), bfloat16_to_float(B[k * N + n]), sum);
    }

    C[m * N + n] = sum;
}

#if defined(cl_intel_subgroup_matrix_multiply_accumulate)

// M rows x K columns
static int __load_a_row_major_bf16_k16_m1_x8(global ushort* A, int rowStart, int colStart, int stride)
{
    int ret;

    global uint* A_ui = (global uint*)A;
    int offset_ui = rowStart * stride / 2 + colStart / 2;
    ret = intel_sub_group_block_read(A_ui + offset_ui);

    return ret; 
}

// M rows x K columns
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

__attribute__((intel_reqd_sub_group_size(8)))
__attribute__((reqd_work_group_size(8, 1, 1)))
kernel void bfloat16_dpas_rowmajor_m1(global float* C, global ushort* A, global ushort* B, int K)
{
    const int N = get_global_size(0);
    int m = get_group_id(1);
    int n = get_group_id(0) * get_local_size(0);

    float sum = 0;
    for (int k = 0; k < K; k += 16) {
        int     aData = __load_a_row_major_bf16_k16_m1_x8(A, m, k, K);
        int8    bData = __load_b_row_major_bf16_k16(B, k, n, N);
        sum = intel_sub_group_bf16_bf16_matrix_mad_k16(aData, bData, sum);
    }

    __store_c_row_major_fp32_m1(C, sum, m, n, N);
}

__attribute__((intel_reqd_sub_group_size(8)))
__attribute__((reqd_work_group_size(8, 1, 1)))
kernel void bfloat16_dpas_rowmajor_m2(global float* C, global ushort* A, global ushort* B, int K)
{
    const int N = get_global_size(0);
    int m = get_group_id(1) * 2;
    int n = get_group_id(0) * get_local_size(0);

    float2 sum = 0;
    for (int k = 0; k < K; k += 16) {
        int2    aData = __load_a_row_major_bf16_k16_m2_x8(A, m, k, K);
        int8    bData = __load_b_row_major_bf16_k16(B, k, n, N);
        sum = intel_sub_group_bf16_bf16_matrix_mad_k16(aData, bData, sum);
    }

    __store_c_row_major_fp32_m2(C, sum, m, n, N);
}

__attribute__((intel_reqd_sub_group_size(8)))
__attribute__((reqd_work_group_size(8, 1, 1)))
kernel void bfloat16_dpas_rowmajor_m4(global float* C, global ushort* A, global ushort* B, int K)
{
    const int N = get_global_size(0);
    int m = get_group_id(1) * 4;
    int n = get_group_id(0) * get_local_size(0);

    float4 sum = 0;
    for (int k = 0; k < K; k += 16) {
        int4    aData = __load_a_row_major_bf16_k16_m4_x8(A, m, k, K);
        int8    bData = __load_b_row_major_bf16_k16(B, k, n, N);
        sum = intel_sub_group_bf16_bf16_matrix_mad_k16(aData, bData, sum);
    }

    __store_c_row_major_fp32_m4(C, sum, m, n, N);
}

__attribute__((intel_reqd_sub_group_size(8)))
__attribute__((reqd_work_group_size(8, 1, 1)))
kernel void bfloat16_dpas_rowmajor_m8(global float* C, global ushort* A, global ushort* B, int K)
{
    const int N = get_global_size(0);
    int m = get_group_id(1) * 8;
    int n = get_group_id(0) * get_local_size(0);

    float8 sum = 0;
    for (int k = 0; k < K; k += 16) {
        int8    aData = __load_a_row_major_bf16_k16_m8_x8(A, m, k, K);
        int8    bData = __load_b_row_major_bf16_k16(B, k, n, N);
        sum = intel_sub_group_bf16_bf16_matrix_mad_k16(aData, bData, sum);
    }

    __store_c_row_major_fp32_m8(C, sum, m, n, N);
}

__attribute__((intel_reqd_sub_group_size(8)))
__attribute__((reqd_work_group_size(8, 1, 1)))
kernel void bfloat16_dpas_vnni_m1(global float* C, global ushort* A, global ushort* B, int K)
{
    const int N = get_global_size(0);
    int m = get_group_id(1);
    int n = get_group_id(0) * get_local_size(0);

    float sum = 0;
    for (int k = 0; k < K; k += 16) {
        int     aData = __load_a_row_major_bf16_k16_m1_x8(A, m, k, K);
        int8    bData = __load_b_vnni_bf16_k16(B, k, n, N);
        sum = intel_sub_group_bf16_bf16_matrix_mad_k16(aData, bData, sum);
    }

    __store_c_row_major_fp32_m1(C, sum, m, n, N);
}

__attribute__((intel_reqd_sub_group_size(8)))
__attribute__((reqd_work_group_size(8, 1, 1)))
kernel void bfloat16_dpas_vnni_m2(global float* C, global ushort* A, global ushort* B, int K)
{
    const int N = get_global_size(0);
    int m = get_group_id(1) * 2;
    int n = get_group_id(0) * get_local_size(0);

    float2 sum = 0;
    for (int k = 0; k < K; k += 16) {
        int2    aData = __load_a_row_major_bf16_k16_m2_x8(A, m, k, K);
        int8    bData = __load_b_vnni_bf16_k16(B, k, n, N);
        sum = intel_sub_group_bf16_bf16_matrix_mad_k16(aData, bData, sum);
    }

    __store_c_row_major_fp32_m2(C, sum, m, n, N);
}

__attribute__((intel_reqd_sub_group_size(8)))
__attribute__((reqd_work_group_size(8, 1, 1)))
kernel void bfloat16_dpas_vnni_m4(global float* C, global ushort* A, global ushort* B, int K)
{
    const int N = get_global_size(0);
    int m = get_group_id(1) * 4;
    int n = get_group_id(0) * get_local_size(0);

    float4 sum = 0;
    for (int k = 0; k < K; k += 16) {
        int4    aData = __load_a_row_major_bf16_k16_m4_x8(A, m, k, K);
        int8    bData = __load_b_vnni_bf16_k16(B, k, n, N);
        sum = intel_sub_group_bf16_bf16_matrix_mad_k16(aData, bData, sum);
    }

    __store_c_row_major_fp32_m4(C, sum, m, n, N);
}

__attribute__((intel_reqd_sub_group_size(8)))
__attribute__((reqd_work_group_size(8, 1, 1)))
kernel void bfloat16_dpas_vnni_m8(global float* C, global ushort* A, global ushort* B, int K)
{
    const int N = get_global_size(0);
    int m = get_group_id(1) * 8;
    int n = get_group_id(0) * get_local_size(0);

    float8 sum = 0;
    for (int k = 0; k < K; k += 16) {
        int8    aData = __load_a_row_major_bf16_k16_m8_x8(A, m, k, K);
        int8    bData = __load_b_vnni_bf16_k16(B, k, n, N);
        sum = intel_sub_group_bf16_bf16_matrix_mad_k16(aData, bData, sum);
    }

    __store_c_row_major_fp32_m8(C, sum, m, n, N);
}

#endif // defined(cl_intel_subgroup_matrix_multiply_accumulate)
