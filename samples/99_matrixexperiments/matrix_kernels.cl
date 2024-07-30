#include "matrix_helpers.cl"

#if EMULATE_tN8
#define mat_mul_sg8  emu_sub_group_bf16_bf16_matrix_mad_k16
#else
#define mat_mul_sg8  intel_sub_group_bf16_bf16_matrix_mad_k16
#endif

#if EMULATE_tN16
#define mat_mul_sg16 emu_sub_group_bf16_bf16_matrix_mad_k16
#else
#define mat_mul_sg16 intel_sub_group_bf16_bf16_matrix_mad_k16
#endif

kernel void bfloat16_naive(global float* C, global ushort* A, global ushort* B, int K)
{
    const int N = get_global_size(0);
    const int m = get_global_id(1);
    const int n = get_global_id(0);

    float sum = 0;
    for (int k = 0; k < K; k++) {
        sum = fma(bf16_to_fp32(A[m * K + k]), bf16_to_fp32(B[k * N + n]), sum);
    }

    sum = activation(sum);
    C[m * N + n] = sum;
}

// For all bfloat16 kernels tK == 16:
#define tK 16

#if defined(cl_intel_subgroups) && defined(cl_intel_subgroups_short) && defined(cl_intel_required_subgroup_size)

#if HAS_SIMD8

// rowmajor kernels:

__attribute__((intel_reqd_sub_group_size(8))) __attribute__((reqd_work_group_size(8, 1, 1)))
kernel void bfloat16_dpas_rowmajor_m1_n8(global float* C, global ushort* A, global ushort* B, int K)
{
    __builtin_assume(K > 0);    // Always at least one K iteration.
    const int tM = 1;
    const int tN = 8;
    const int N = get_global_size(0);
    const int m = get_group_id(1) * tM;
    const int n = get_group_id(0) * tN;

    float sum = 0;
    for (int k = 0; k < K; k += tK) {
        int     aData = load_a_rowmajor_16b_1r16c_sg8(A, m, k, K);
        int8    bData = load_b_rowmajor_16b_16rNc(B, k, n, N);
        sum = mat_mul_sg8(aData, bData, sum);
    }

    sum = activation(sum);
    store_c_rowmajor_fp32_1rNc(C, sum, m, n, N);
}

__attribute__((intel_reqd_sub_group_size(8))) __attribute__((reqd_work_group_size(8, 1, 1)))
kernel void bfloat16_dpas_rowmajor_m2_n8(global float* C, global ushort* A, global ushort* B, int K)
{
    __builtin_assume(K > 0);    // Always at least one K iteration.
    const int tM = 2;
    const int tN = 8;
    const int N = get_global_size(0);
    const int m = get_group_id(1) * tM;
    const int n = get_group_id(0) * tN;

    float2 sum = 0;
    for (int k = 0; k < K; k += tK) {
        int2    aData = load_a_rowmajor_16b_2r16c_sg8(A, m, k, K);
        int8    bData = load_b_rowmajor_16b_16rNc(B, k, n, N);
        sum = mat_mul_sg8(aData, bData, sum);
    }

    sum = activation(sum);
    store_c_rowmajor_fp32_2rNc(C, sum, m, n, N);
}

__attribute__((intel_reqd_sub_group_size(8))) __attribute__((reqd_work_group_size(8, 1, 1)))
kernel void bfloat16_dpas_rowmajor_m4_n8(global float* C, global ushort* A, global ushort* B, int K)
{
    __builtin_assume(K > 0);    // Always at least one K iteration.
    const int tM = 4;
    const int tN = 8;
    const int N = get_global_size(0);
    const int m = get_group_id(1) * tM;
    const int n = get_group_id(0) * tN;

    float4 sum = 0;
    for (int k = 0; k < K; k += tK) {
        int4    aData = load_a_rowmajor_16b_4r16c_sg8(A, m, k, K);
        int8    bData = load_b_rowmajor_16b_16rNc(B, k, n, N);
        sum = mat_mul_sg8(aData, bData, sum);
    }

    sum = activation(sum);
    store_c_rowmajor_fp32_4rNc(C, sum, m, n, N);
}

__attribute__((intel_reqd_sub_group_size(8))) __attribute__((reqd_work_group_size(8, 1, 1)))
kernel void bfloat16_dpas_rowmajor_m8_n8(global float* C, global ushort* A, global ushort* B, int K)
{
    __builtin_assume(K > 0);    // Always at least one K iteration.
    const int tM = 8;
    const int tN = 8;
    const int N = get_global_size(0);
    const int m = get_group_id(1) * tM;
    const int n = get_group_id(0) * tN;

    float8 sum = 0;
    for (int k = 0; k < K; k += tK) {
        int8    aData = load_a_rowmajor_16b_8r16c_sg8(A, m, k, K);
        int8    bData = load_b_rowmajor_16b_16rNc(B, k, n, N);
        sum = mat_mul_sg8(aData, bData, sum);
    }

    sum = activation(sum);
    store_c_rowmajor_fp32_8rNc(C, sum, m, n, N);
}

// pre-packed kernels:

__attribute__((intel_reqd_sub_group_size(8))) __attribute__((reqd_work_group_size(8, 1, 1)))
kernel void bfloat16_dpas_vnni_m1_n8(global float* C, global ushort* A, global ushort* B, int K)
{
    __builtin_assume(K > 0);    // Always at least one K iteration.
    const int tM = 1;
    const int tN = 8;
    const int N = get_global_size(0);
    const int m = get_group_id(1) * tM;
    const int n = get_group_id(0) * tN;

    float sum = 0;
    for (int k = 0; k < K; k += tK) {
        int     aData = load_a_rowmajor_16b_1r16c_sg8(A, m, k, K);
        int8    bData = load_b_packed_16b_16rNc(B, k, n, N);
        sum = mat_mul_sg8(aData, bData, sum);
    }

    sum = activation(sum);
    store_c_rowmajor_fp32_1rNc(C, sum, m, n, N);
}

__attribute__((intel_reqd_sub_group_size(8))) __attribute__((reqd_work_group_size(8, 1, 1)))
kernel void bfloat16_dpas_vnni_m2_n8(global float* C, global ushort* A, global ushort* B, int K)
{
    __builtin_assume(K > 0);    // Always at least one K iteration.
    const int tM = 2;
    const int tN = 8;
    const int N = get_global_size(0);
    const int m = get_group_id(1) * tM;
    const int n = get_group_id(0) * tN;

    float2 sum = 0;
    for (int k = 0; k < K; k += tK) {
        int2    aData = load_a_rowmajor_16b_2r16c_sg8(A, m, k, K);
        int8    bData = load_b_packed_16b_16rNc(B, k, n, N);
        sum = mat_mul_sg8(aData, bData, sum);
    }

    sum = activation(sum);
    store_c_rowmajor_fp32_2rNc(C, sum, m, n, N);
}

__attribute__((intel_reqd_sub_group_size(8))) __attribute__((reqd_work_group_size(8, 1, 1)))
kernel void bfloat16_dpas_vnni_m4_n8(global float* C, global ushort* A, global ushort* B, int K)
{
    __builtin_assume(K > 0);    // Always at least one K iteration.
    const int tM = 4;
    const int tN = 8;
    const int N = get_global_size(0);
    const int m = get_group_id(1) * tM;
    const int n = get_group_id(0) * tN;

    float4 sum = 0;
    for (int k = 0; k < K; k += tK) {
        int4    aData = load_a_rowmajor_16b_4r16c_sg8(A, m, k, K);
        int8    bData = load_b_packed_16b_16rNc(B, k, n, N);
        sum = mat_mul_sg8(aData, bData, sum);
    }

    sum = activation(sum);
    store_c_rowmajor_fp32_4rNc(C, sum, m, n, N);
}

__attribute__((intel_reqd_sub_group_size(8))) __attribute__((reqd_work_group_size(8, 1, 1)))
kernel void bfloat16_dpas_vnni_m8_n8(global float* C, global ushort* A, global ushort* B, int K)
{
    __builtin_assume(K > 0);    // Always at least one K iteration.
    const int tM = 8;
    const int tN = 8;
    const int N = get_global_size(0);
    const int m = get_group_id(1) * tM;
    const int n = get_group_id(0) * tN;

    float8 sum = 0;
    for (int k = 0; k < K; k += tK) {
        int8    aData = load_a_rowmajor_16b_8r16c_sg8(A, m, k, K);
        int8    bData = load_b_packed_16b_16rNc(B, k, n, N);
        sum = mat_mul_sg8(aData, bData, sum);
    }

    sum = activation(sum);
    store_c_rowmajor_fp32_8rNc(C, sum, m, n, N);
}

#endif // HAS_SIMD8

// rowmajor krenels:

__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16, 1, 1)))
kernel void bfloat16_dpas_rowmajor_m1_n16(global float* C, global ushort* A, global ushort* B, int K)
{
    __builtin_assume(K > 0);    // Always at least one K iteration.
    const int tM = 1;
    const int tN = 16;
    const int N = get_global_size(0);
    const int m = get_group_id(1) * tM;
    const int n = get_group_id(0) * get_local_size(0);

    float sum = 0;
    for (int k = 0; k < K; k += tK) {
        short   aData = load_a_rowmajor_16b_1r16c_sg16(A, m, k, K);
        int8    bData = load_b_rowmajor_16b_16rNc(B, k, n, N);
        sum = mat_mul_sg16(aData, bData, sum);
    }

    sum = activation(sum);
    store_c_rowmajor_fp32_1rNc(C, sum, m, n, N);
}

__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16, 1, 1)))
kernel void bfloat16_dpas_rowmajor_m2_n16(global float* C, global ushort* A, global ushort* B, int K)
{
    __builtin_assume(K > 0);    // Always at least one K iteration.
    const int tM = 2;
    const int tN = 16;
    const int N = get_global_size(0);
    const int m = get_group_id(1) * tM;
    const int n = get_group_id(0) * get_local_size(0);

    float2 sum = 0;
    for (int k = 0; k < K; k += tK) {
        short2  aData = load_a_rowmajor_16b_2r16c_sg16(A, m, k, K);
        int8    bData = load_b_rowmajor_16b_16rNc(B, k, n, N);
        sum = mat_mul_sg16(aData, bData, sum);
    }

    sum = activation(sum);
    store_c_rowmajor_fp32_2rNc(C, sum, m, n, N);
}

__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16, 1, 1)))
kernel void bfloat16_dpas_rowmajor_m4_n16(global float* C, global ushort* A, global ushort* B, int K)
{
    __builtin_assume(K > 0);    // Always at least one K iteration.
    const int tM = 4;
    const int tN = 16;
    const int N = get_global_size(0);
    const int m = get_group_id(1) * tM;
    const int n = get_group_id(0) * get_local_size(0);

    float4 sum = 0;
    for (int k = 0; k < K; k += tK) {
        short4  aData = load_a_rowmajor_16b_4r16c_sg16(A, m, k, K);
        int8    bData = load_b_rowmajor_16b_16rNc(B, k, n, N);
        sum = mat_mul_sg16(aData, bData, sum);
    }

    sum = activation(sum);
    store_c_rowmajor_fp32_4rNc(C, sum, m, n, N);
}

__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16, 1, 1)))
kernel void bfloat16_dpas_rowmajor_m8_n16(global float* C, global ushort* A, global ushort* B, int K)
{
    __builtin_assume(K > 0);    // Always at least one K iteration.
    const int tM = 8;
    const int tN = 16;
    const int N = get_global_size(0);
    const int m = get_group_id(1) * tM;
    const int n = get_group_id(0) * get_local_size(0);

    float8 sum = 0;
    for (int k = 0; k < K; k += tK) {
        short8  aData = load_a_rowmajor_16b_8r16c_sg16(A, m, k, K);
        int8    bData = load_b_rowmajor_16b_16rNc(B, k, n, N);
        sum = mat_mul_sg16(aData, bData, sum);
    }

    sum = activation(sum);
    store_c_rowmajor_fp32_8rNc(C, sum, m, n, N);
}

// pre-packed kernels:

__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16, 1, 1)))
kernel void bfloat16_dpas_vnni_m1_n16(global float* C, global ushort* A, global ushort* B, int K)
{
    __builtin_assume(K > 0);    // Always at least one K iteration.
    const int tM = 1;
    const int tN = 16;
    const int N = get_global_size(0);
    const int m = get_group_id(1) * tM;
    const int n = get_group_id(0) * tN;

    float sum = 0;
    for (int k = 0; k < K; k += tK) {
        short   aData = load_a_rowmajor_16b_1r16c_sg16(A, m, k, K);
        int8    bData = load_b_packed_16b_16rNc(B, k, n, N);
        sum = mat_mul_sg16(aData, bData, sum);
    }

    sum = activation(sum);
    store_c_rowmajor_fp32_1rNc(C, sum, m, n, N);
}

__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16, 1, 1)))
kernel void bfloat16_dpas_vnni_m2_n16(global float* C, global ushort* A, global ushort* B, int K)
{
    __builtin_assume(K > 0);    // Always at least one K iteration.
    const int tM = 2;
    const int tN = 16;
    const int N = get_global_size(0);
    const int m = get_group_id(1) * tM;
    const int n = get_group_id(0) * tN;

    float2 sum = 0;
    for (int k = 0; k < K; k += tK) {
        short2  aData = load_a_rowmajor_16b_2r16c_sg16(A, m, k, K);
        int8    bData = load_b_packed_16b_16rNc(B, k, n, N);
        sum = mat_mul_sg16(aData, bData, sum);
    }

    sum = activation(sum);
    store_c_rowmajor_fp32_2rNc(C, sum, m, n, N);
}

__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16, 1, 1)))
kernel void bfloat16_dpas_vnni_m4_n16(global float* C, global ushort* A, global ushort* B, int K)
{
    __builtin_assume(K > 0);    // Always at least one K iteration.
    const int tM = 4;
    const int tN = 16;
    const int N = get_global_size(0);
    const int m = get_group_id(1) * tM;
    const int n = get_group_id(0) * tN;

    float4 sum = 0;
    for (int k = 0; k < K; k += tK) {
        short4  aData = load_a_rowmajor_16b_4r16c_sg16(A, m, k, K);
        int8    bData = load_b_packed_16b_16rNc(B, k, n, N);
        sum = mat_mul_sg16(aData, bData, sum);
    }

    sum = activation(sum);
    store_c_rowmajor_fp32_4rNc(C, sum, m, n, N);
}

__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16, 1, 1)))
kernel void bfloat16_dpas_vnni_m8_n16(global float* C, global ushort* A, global ushort* B, int K)
{
    __builtin_assume(K > 0);    // Always at least one K iteration.
    const int tM = 8;
    const int tN = 16;
    const int N = get_global_size(0);
    const int m = get_group_id(1) * tM;
    const int n = get_group_id(0) * tN;

    float8 sum = 0;
    for (int k = 0; k < K; k += tK) {
        short8  aData = load_a_rowmajor_16b_8r16c_sg16(A, m, k, K);
        int8    bData = load_b_packed_16b_16rNc(B, k, n, N);
        sum = mat_mul_sg16(aData, bData, sum);
    }

    sum = activation(sum);
    store_c_rowmajor_fp32_8rNc(C, sum, m, n, N);
}

#ifdef cl_intel_subgroup_extended_block_read

__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16, 1, 1)))
kernel void bfloat16_dpas_blockread_rowmajor_m1_n16(global float* C, global ushort* A, global ushort* B, int K)
{
    __builtin_assume(K > 0);    // Always at least one K iteration.
    const int tM = 1;
    const int tN = 16;
    const int M = get_global_size(1);
    const int N = get_global_size(0);
    const int m = get_group_id(1) * tM;
    const int n = get_group_id(0) * tN;

    float sum = 0;
    for (int k = 0; k < K; k += tK) {
        short   aData = as_short(intel_sub_group_block_read_16b_1r16c(A, K * sizeof(ushort), M, K * sizeof(ushort), (int2)(k, m)));
        int8    bData = as_int8(intel_sub_group_block_read_transform_16b_16r16c(B, N * sizeof(ushort), K, N * sizeof(ushort), (int2)(n, k)));
        sum = mat_mul_sg16(aData, bData, sum);
    }

    sum = activation(sum);
    intel_sub_group_block_write_32b_1r16c(C, N * sizeof(float), M, N * sizeof(float), (int2)(n, m), as_uint(sum));
}

__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16, 1, 1)))
kernel void bfloat16_dpas_blockread_rowmajor_m2_n16(global float* C, global ushort* A, global ushort* B, int K)
{
    __builtin_assume(K > 0);    // Always at least one K iteration.
    const int tM = 2;
    const int tN = 16;
    const int M = get_global_size(1) * tM;
    const int N = get_global_size(0);
    const int m = get_group_id(1) * tM;
    const int n = get_group_id(0) * tN;

    float2 sum = 0;
    for (int k = 0; k < K; k += tK) {
        short2  aData = as_short2(intel_sub_group_block_read_16b_2r16c(A, K * sizeof(ushort), M, K * sizeof(ushort), (int2)(k, m)));
        int8    bData = as_int8(intel_sub_group_block_read_transform_16b_16r16c(B, N * sizeof(ushort), K, N * sizeof(ushort), (int2)(n, k)));
        sum = mat_mul_sg16(aData, bData, sum);
    }

    sum = activation(sum);
    intel_sub_group_block_write_32b_2r16c(C, N * sizeof(float), M, N * sizeof(float), (int2)(n, m), as_uint2(sum));
}

__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16, 1, 1)))
kernel void bfloat16_dpas_blockread_rowmajor_m4_n16(global float* C, global ushort* A, global ushort* B, int K)
{
    __builtin_assume(K > 0);    // Always at least one K iteration.
    const int tM = 4;
    const int tN = 16;
    const int M = get_global_size(1) * tM;
    const int N = get_global_size(0);
    const int m = get_group_id(1) * tM;
    const int n = get_group_id(0) * tN;

    float4 sum = 0;
    for (int k = 0; k < K; k += tK) {
        short4  aData = as_short4(intel_sub_group_block_read_16b_4r16c(A, K * sizeof(ushort), M, K * sizeof(ushort), (int2)(k, m)));
        int8    bData = as_int8(intel_sub_group_block_read_transform_16b_16r16c(B, N * sizeof(ushort), K, N * sizeof(ushort), (int2)(n, k)));
        sum = mat_mul_sg16(aData, bData, sum);
    }

    sum = activation(sum);
    intel_sub_group_block_write_32b_4r16c(C, N * sizeof(float), M, N * sizeof(float), (int2)(n, m), as_uint4(sum));
}

__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16, 1, 1)))
kernel void bfloat16_dpas_blockread_rowmajor_m8_n16(global float* C, global ushort* A, global ushort* B, int K)
{
    __builtin_assume(K > 0);    // Always at least one K iteration.
    const int tM = 8;
    const int tN = 16;
    const int M = get_global_size(1) * tM;
    const int N = get_global_size(0);
    const int m = get_group_id(1) * tM;
    const int n = get_group_id(0) * tN;

    float8 sum = 0;
    for (int k = 0; k < K; k += tK) {
        short8  aData = as_short8(intel_sub_group_block_read_16b_8r16c(A, K * sizeof(ushort), M, K * sizeof(ushort), (int2)(k, m)));
        int8    bData = as_int8(intel_sub_group_block_read_transform_16b_16r16c(B, N * sizeof(ushort), K, N * sizeof(ushort), (int2)(n, k)));
        sum = mat_mul_sg16(aData, bData, sum);
    }

    sum = activation(sum);
    intel_sub_group_block_write_32b_8r16c(C, N * sizeof(float), M, N * sizeof(float), (int2)(n, m), as_uint8(sum));
}

__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16, 1, 1)))
kernel void bfloat16_dpas_blockread_vnni_m1_n16(global float* C, global ushort* A, global ushort* B, int K)
{
    __builtin_assume(K > 0);    // Always at least one K iteration.
    const int tM = 1;
    const int tN = 16;
    const int M = get_global_size(1) * tM;
    const int N = get_global_size(0);
    const int m = get_group_id(1) * tM;
    const int n = get_group_id(0) * tN;

    float sum = 0;
    for (int k = 0; k < K; k += tK) {
        short   aData = as_short(intel_sub_group_block_read_16b_1r16c(A, K * sizeof(ushort), M, K * sizeof(ushort), (int2)(k, m)));
        int8    bData = as_int8(intel_sub_group_block_read_32b_8r16c(B, N * sizeof(uint), K, N * sizeof(uint), (int2)(n, k / 2)));
        sum = mat_mul_sg16(aData, bData, sum);
    }

    sum = activation(sum);
    intel_sub_group_block_write_32b_1r16c(C, N * sizeof(float), M, N * sizeof(float), (int2)(n, m), as_uint(sum));
}

__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16, 1, 1)))
kernel void bfloat16_dpas_blockread_vnni_m2_n16(global float* C, global ushort* A, global ushort* B, int K)
{
    __builtin_assume(K > 0);    // Always at least one K iteration.
    const int tM = 2;
    const int tN = 16;
    const int M = get_global_size(1) * tM;
    const int N = get_global_size(0);
    const int m = get_group_id(1) * tM;
    const int n = get_group_id(0) * tN;

    float2 sum = 0;
    for (int k = 0; k < K; k += tK) {
        short2  aData = as_short2(intel_sub_group_block_read_16b_2r16c(A, K * sizeof(ushort), M, K * sizeof(ushort), (int2)(k, m)));
        int8    bData = as_int8(intel_sub_group_block_read_32b_8r16c(B, N * sizeof(uint), K, N * sizeof(uint), (int2)(n, k / 2)));
        sum = mat_mul_sg16(aData, bData, sum);
    }

    sum = activation(sum);
    intel_sub_group_block_write_32b_2r16c(C, N * sizeof(float), M, N * sizeof(float), (int2)(n, m), as_uint2(sum));
}

__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16, 1, 1)))
kernel void bfloat16_dpas_blockread_vnni_m4_n16(global float* C, global ushort* A, global ushort* B, int K)
{
    __builtin_assume(K > 0);    // Always at least one K iteration.
    const int tM = 4;
    const int tN = 16;
    const int M = get_global_size(1) * tM;
    const int N = get_global_size(0);
    const int m = get_group_id(1) * tM;
    const int n = get_group_id(0) * tN;

    float4 sum = 0;
    for (int k = 0; k < K; k += tK) {
        short4  aData = as_short4(intel_sub_group_block_read_16b_4r16c(A, K * sizeof(ushort), M, K * sizeof(ushort), (int2)(k, m)));
        int8    bData = as_int8(intel_sub_group_block_read_32b_8r16c(B, N * sizeof(uint), K, N * sizeof(uint), (int2)(n, k / 2)));
        sum = mat_mul_sg16(aData, bData, sum);
    }

    sum = activation(sum);
    intel_sub_group_block_write_32b_4r16c(C, N * sizeof(float), M, N * sizeof(float), (int2)(n, m), as_uint4(sum));
}

__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16, 1, 1)))
kernel void bfloat16_dpas_blockread_vnni_m8_n16(global float* C, global ushort* A, global ushort* B, int K)
{
    __builtin_assume(K > 0);    // Always at least one K iteration.
    const int tM = 8;
    const int tN = 16;
    const int M = get_global_size(1) * tM;
    const int N = get_global_size(0);
    const int m = get_group_id(1) * tM;
    const int n = get_group_id(0) * tN;

    float8 sum = 0;
    for (int k = 0; k < K; k += tK) {
        short8  aData = as_short8(intel_sub_group_block_read_16b_8r16c(A, K * sizeof(ushort), M, K * sizeof(ushort), (int2)(k, m)));
        int8    bData = as_int8(intel_sub_group_block_read_32b_8r16c(B, N * sizeof(uint), K, N * sizeof(uint), (int2)(n, k / 2)));
        sum = mat_mul_sg16(aData, bData, sum);
    }

    sum = activation(sum);
    intel_sub_group_block_write_32b_8r16c(C, N * sizeof(float), M, N * sizeof(float), (int2)(n, m), as_uint8(sum));
}

#endif // cl_intel_subgroup_extended_block_read

// Tiled matrix multiplication kernels, generated from a template:

#define MM 1
#define NN 1
#include "matrix_kernel_tiled.cl"
#undef MM
#undef NN

#define MM 2
#define NN 1
#include "matrix_kernel_tiled.cl"
#undef MM
#undef NN

#define MM 1
#define NN 2
#include "matrix_kernel_tiled.cl"
#undef MM
#undef NN

#define MM 2
#define NN 2
#include "matrix_kernel_tiled.cl"
#undef MM
#undef NN

#define MM 4
#define NN 2
#include "matrix_kernel_tiled.cl"
#undef MM
#undef NN

#define MM 2
#define NN 4
#include "matrix_kernel_tiled.cl"
#undef MM
#undef NN

#define MM 4
#define NN 4
#include "matrix_kernel_tiled.cl"
#undef MM
#undef NN

#endif // defined(cl_intel_subgroups) && defined(cl_intel_subgroups_short) && defined(cl_intel_required_subgroup_size)

#undef tK
