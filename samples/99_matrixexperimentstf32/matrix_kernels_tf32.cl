#include "matrix_helpers_tf32.cl"

#if EMULATE_tN16
#define mat_mul_sg16 emu_sub_group_tf32_tf32_matrix_mad_k8
#else
#define mat_mul_sg16 intel_sub_group_tf32_tf32_matrix_mad_k8
#endif

kernel void tf32_naive(global float* C, global float* A, global float* B, int K)
{
    const int N = get_global_size(0);
    const int m = get_global_id(1);
    const int n = get_global_id(0);

    float sum = 0;
    for (int k = 0; k < K; k++) {
        sum = fma(A[m * K + k], B[k * N + n], sum);
    }

    C[m * N + n] = sum;
}

// For all tf32 kernels tK == 8:
#define tK 8

#if defined(cl_intel_subgroups) && defined(cl_intel_required_subgroup_size)

// rowmajor krenels:

__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16, 1, 1)))
kernel void tf32_dpas_rowmajor_m1_n16(global float* C, global float* A, global float* B, int K)
{
    const int tM = 1;
    const int tN = 16;
    const int N = get_global_size(0);
    const int m = get_group_id(1) * tM;
    const int n = get_group_id(0) * get_local_size(0);

    float sum = 0;
    for (int k = 0; k < K; k += tK) {
        float   aData = load_a_rowmajor_d32_m1_k8_sg16(A, m, k, K);
        float8  bData = load_b_rowmajor_d32_k8_nx(B, k, n, N);
        sum = mat_mul_sg16(aData, bData, sum);
    }

    store_c_rowmajor_fp32_m1_nx(C, sum, m, n, N);
}

__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16, 1, 1)))
kernel void tf32_dpas_rowmajor_m2_n16(global float* C, global float* A, global float* B, int K)
{
    const int tM = 2;
    const int tN = 16;
    const int N = get_global_size(0);
    const int m = get_group_id(1) * tM;
    const int n = get_group_id(0) * get_local_size(0);

    float2 sum = 0;
    for (int k = 0; k < K; k += tK) {
        float   aData = load_a_rowmajor_d32_m2_k8_sg16(A, m, k, K);
        float8  bData = load_b_rowmajor_d32_k8_nx(B, k, n, N);
        sum = mat_mul_sg16(aData, bData, sum);
    }

    store_c_rowmajor_fp32_m2_nx(C, sum, m, n, N);
}

__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16, 1, 1)))
kernel void tf32_dpas_rowmajor_m4_n16(global float* C, global float* A, global float* B, int K)
{
    const int tM = 4;
    const int tN = 16;
    const int N = get_global_size(0);
    const int m = get_group_id(1) * tM;
    const int n = get_group_id(0) * get_local_size(0);

    float4 sum = 0;
    for (int k = 0; k < K; k += tK) {
        float2  aData = load_a_rowmajor_d32_m4_k8_sg16(A, m, k, K);
        float8  bData = load_b_rowmajor_d32_k8_nx(B, k, n, N);
        sum = mat_mul_sg16(aData, bData, sum);
    }

    store_c_rowmajor_fp32_m4_nx(C, sum, m, n, N);
}

__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16, 1, 1)))
kernel void tf32_dpas_rowmajor_m8_n16(global float* C, global float* A, global float* B, int K)
{
    const int tM = 8;
    const int tN = 16;
    const int N = get_global_size(0);
    const int m = get_group_id(1) * tM;
    const int n = get_group_id(0) * get_local_size(0);

    float8 sum = 0;
    for (int k = 0; k < K; k += tK) {
        float4  aData = load_a_rowmajor_d32_m8_k8_sg16(A, m, k, K);
        float8  bData = load_b_rowmajor_d32_k8_nx(B, k, n, N);
        sum = mat_mul_sg16(aData, bData, sum);
    }

    store_c_rowmajor_fp32_m8_nx(C, sum, m, n, N);
}

#ifdef cl_intel_subgroup_extended_block_read

__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16, 1, 1)))
kernel void tf32_dpas_blockread_rowmajor_m1_n16(global float* C, global float* A, global float* B, int K)
{
    const int tM = 1;
    const int tN = 16;
    const int M = get_global_size(1);
    const int N = get_global_size(0);
    const int m = get_group_id(1) * tM;
    const int n = get_group_id(0) * tN;

    float sum = 0;
    for (int k = 0; k < K; k += tK) {
        float   aData = as_float(intel_subgroup_block_read_u32_m1k8(A, K * sizeof(float), M, K * sizeof(float), (int2)(k, m)));
        float8  bData = as_float8(intel_subgroup_block_read_u32_m8k16(B, N * sizeof(float), K, N * sizeof(float), (int2)(n, k)));
        sum = mat_mul_sg16(aData, bData, sum);
    }

    intel_subgroup_block_write_u32_m1k16(C, N * sizeof(float), M, N * sizeof(float), (int2)(n, m), as_uint(sum));
}

__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16, 1, 1)))
kernel void tf32_dpas_blockread_rowmajor_m2_n16(global float* C, global float* A, global float* B, int K)
{
    const int tM = 2;
    const int tN = 16;
    const int M = get_global_size(1) * tM;
    const int N = get_global_size(0);
    const int m = get_group_id(1) * tM;
    const int n = get_group_id(0) * tN;

    float2 sum = 0;
    for (int k = 0; k < K; k += tK) {
        float   aData = as_float(intel_subgroup_block_read_u32_m2k8(A, K * sizeof(float), M, K * sizeof(float), (int2)(k, m)));
        float8  bData = as_float8(intel_subgroup_block_read_u32_m8k16(B, N * sizeof(float), K, N * sizeof(float), (int2)(n, k)));
        sum = mat_mul_sg16(aData, bData, sum);
    }

    intel_subgroup_block_write_u32_m2k16(C, N * sizeof(float), M, N * sizeof(float), (int2)(n, m), as_uint2(sum));
}

__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16, 1, 1)))
kernel void tf32_dpas_blockread_rowmajor_m4_n16(global float* C, global float* A, global float* B, int K)
{
    const int tM = 4;
    const int tN = 16;
    const int M = get_global_size(1) * tM;
    const int N = get_global_size(0);
    const int m = get_group_id(1) * tM;
    const int n = get_group_id(0) * tN;

    float4 sum = 0;
    for (int k = 0; k < K; k += tK) {
        float2  aData = as_float2(intel_subgroup_block_read_u32_m4k8(A, K * sizeof(float), M, K * sizeof(float), (int2)(k, m)));
        float8  bData = as_float8(intel_subgroup_block_read_u32_m8k16(B, N * sizeof(float), K, N * sizeof(float), (int2)(n, k)));
        sum = mat_mul_sg16(aData, bData, sum);
    }

    intel_subgroup_block_write_u32_m4k16(C, N * sizeof(float), M, N * sizeof(float), (int2)(n, m), as_uint4(sum));
}

__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16, 1, 1)))
kernel void tf32_dpas_blockread_rowmajor_m8_n16(global float* C, global float* A, global float* B, int K)
{
    const int tM = 8;
    const int tN = 16;
    const int M = get_global_size(1) * tM;
    const int N = get_global_size(0);
    const int m = get_group_id(1) * tM;
    const int n = get_group_id(0) * tN;

    float8 sum = 0;
    for (int k = 0; k < K; k += tK) {
        float4  aData = as_float4(intel_subgroup_block_read_u32_m8k8(A, K * sizeof(float), M, K * sizeof(float), (int2)(k, m)));
        float8  bData = as_float8(intel_subgroup_block_read_u32_m8k16(B, N * sizeof(float), K, N * sizeof(float), (int2)(n, k)));
        sum = mat_mul_sg16(aData, bData, sum);
    }

    intel_subgroup_block_write_u32_m8k16(C, N * sizeof(float), M, N * sizeof(float), (int2)(n, m), as_uint8(sum));
}

#endif // cl_intel_subgroup_extended_block_read

// Tiled matrix multiplication kernels, generated from a template:

#define MM 1
#define NN 1
#include "matrix_kernel_tiled_tf32.cl"
#undef MM
#undef NN

#define MM 2
#define NN 1
#include "matrix_kernel_tiled_tf32.cl"
#undef MM
#undef NN

#define MM 1
#define NN 2
#include "matrix_kernel_tiled_tf32.cl"
#undef MM
#undef NN

#define MM 2
#define NN 2
#include "matrix_kernel_tiled_tf32.cl"
#undef MM
#undef NN

#define MM 4
#define NN 2
#include "matrix_kernel_tiled_tf32.cl"
#undef MM
#undef NN

#define MM 2
#define NN 4
#include "matrix_kernel_tiled_tf32.cl"
#undef MM
#undef NN

#define MM 4
#define NN 4
#include "matrix_kernel_tiled_tf32.cl"
#undef MM
#undef NN

#endif // defined(cl_intel_subgroups) && defined(cl_intel_required_subgroup_size)

#undef tK
