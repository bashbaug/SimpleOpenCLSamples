#include "matrix_helpers_i8.cl"

#if EMULATE_tN8
#define mat_mul_sg8  emu_sub_group_i8_i8_matrix_mad_k32
#else
#define mat_mul_sg8  intel_sub_group_i8_i8_matrix_mad_k32
#endif

#if EMULATE_tN16
#define mat_mul_sg16 emu_sub_group_i8_i8_matrix_mad_k32
#else
#define mat_mul_sg16 intel_sub_group_i8_i8_matrix_mad_k32
#endif

kernel void i8_naive(global int* C, global char* A, global char* B, int K)
{
    const int N = get_global_size(0);
    const int m = get_global_id(1);
    const int n = get_global_id(0);

    int sum = 0;
    for (int k = 0; k < K; k++) {
        sum = A[m * K + k] * B[k * N + n] + sum;
    }

    sum = activation(sum);
    C[m * N + n] = sum;
}

kernel void i8_naive_TN(global int* C, global char* A, global char* B, int K)
{
    const int N = get_global_size(0);
    const int m = get_global_id(1);
    const int n = get_global_id(0);

    int sum = 0;
    for (int k = 0; k < K; k++) {
        sum = A[k * K + m] * B[k * N + n] + sum;
        if (get_global_id(0) == 1 && get_global_id(1) == 0) {
            printf("after iteration %d: sum is %d\n", k, sum);
        }
    }

    sum = activation(sum);
    C[m * N + n] = sum;
}

// For all i8 kernels tK == 32:
#define tK 32

#if defined(cl_intel_subgroups) && defined(cl_intel_subgroups_char) && defined(cl_intel_required_subgroup_size)

#if HAS_SIMD8

// rowmajor kernels:

__attribute__((intel_reqd_sub_group_size(8))) __attribute__((reqd_work_group_size(8, 1, 1)))
kernel void i8_dpas_rowmajor_m1_n8(global int* C, global char* A, global char* B, int K)
{
    __builtin_assume(K > 0);    // Always at least one K iteration.
    const int tM = 1;
    const int tN = 8;
    const int N = get_global_size(0);
    const int m = get_group_id(1) * tM;
    const int n = get_group_id(0) * tN;

    int sum = 0;
    for (int k = 0; k < K; k += tK) {
        int     aData = load_a_rowmajor_d8_m1_k32_sg8(A, m, k, K);
        int8    bData = load_b_rowmajor_d8_k32_nx(B, k, n, N);
        sum = mat_mul_sg8(aData, bData, sum);
    }

    sum = activation(sum);
    store_c_rowmajor_int32_m1_nx(C, sum, m, n, N);
}

__attribute__((intel_reqd_sub_group_size(8))) __attribute__((reqd_work_group_size(8, 1, 1)))
kernel void i8_dpas_rowmajor_m2_n8(global int* C, global char* A, global char* B, int K)
{
    __builtin_assume(K > 0);    // Always at least one K iteration.
    const int tM = 2;
    const int tN = 8;
    const int N = get_global_size(0);
    const int m = get_group_id(1) * tM;
    const int n = get_group_id(0) * tN;

    int2 sum = 0;
    for (int k = 0; k < K; k += tK) {
        int2    aData = load_a_rowmajor_d8_m2_k32_sg8(A, m, k, K);
        int8    bData = load_b_rowmajor_d8_k32_nx(B, k, n, N);
        sum = mat_mul_sg8(aData, bData, sum);
    }

    sum = activation(sum);
    store_c_rowmajor_int32_m2_nx(C, sum, m, n, N);
}

__attribute__((intel_reqd_sub_group_size(8))) __attribute__((reqd_work_group_size(8, 1, 1)))
kernel void i8_dpas_rowmajor_m4_n8(global int* C, global char* A, global char* B, int K)
{
    __builtin_assume(K > 0);    // Always at least one K iteration.
    const int tM = 4;
    const int tN = 8;
    const int N = get_global_size(0);
    const int m = get_group_id(1) * tM;
    const int n = get_group_id(0) * tN;

    int4 sum = 0;
    for (int k = 0; k < K; k += tK) {
        int4    aData = load_a_rowmajor_d8_m4_k32_sg8(A, m, k, K);
        int8    bData = load_b_rowmajor_d8_k32_nx(B, k, n, N);
        sum = mat_mul_sg8(aData, bData, sum);
    }

    sum = activation(sum);
    store_c_rowmajor_int32_m4_nx(C, sum, m, n, N);
}

__attribute__((intel_reqd_sub_group_size(8))) __attribute__((reqd_work_group_size(8, 1, 1)))
kernel void i8_dpas_rowmajor_m8_n8(global int* C, global char* A, global char* B, int K)
{
    __builtin_assume(K > 0);    // Always at least one K iteration.
    const int tM = 8;
    const int tN = 8;
    const int N = get_global_size(0);
    const int m = get_group_id(1) * tM;
    const int n = get_group_id(0) * tN;

    int8 sum = 0;
    for (int k = 0; k < K; k += tK) {
        int8    aData = load_a_rowmajor_d8_m8_k32_sg8(A, m, k, K);
        int8    bData = load_b_rowmajor_d8_k32_nx(B, k, n, N);
        sum = mat_mul_sg8(aData, bData, sum);
    }

    sum = activation(sum);
    store_c_rowmajor_int32_m8_nx(C, sum, m, n, N);
}

// vnni kernels:

__attribute__((intel_reqd_sub_group_size(8))) __attribute__((reqd_work_group_size(8, 1, 1)))
kernel void i8_dpas_vnni_m1_n8(global int* C, global char* A, global char* B, int K)
{
    __builtin_assume(K > 0);    // Always at least one K iteration.
    const int tM = 1;
    const int tN = 8;
    const int N = get_global_size(0);
    const int m = get_group_id(1) * tM;
    const int n = get_group_id(0) * tN;

    int sum = 0;
    for (int k = 0; k < K; k += tK) {
        int     aData = load_a_rowmajor_d8_m1_k32_sg8(A, m, k, K);
        int8    bData = load_b_vnni_d8_k32_nx(B, k, n, N);
        sum = mat_mul_sg8(aData, bData, sum);
    }

    sum = activation(sum);
    store_c_rowmajor_int32_m1_nx(C, sum, m, n, N);
}

__attribute__((intel_reqd_sub_group_size(8))) __attribute__((reqd_work_group_size(8, 1, 1)))
kernel void i8_dpas_vnni_m2_n8(global int* C, global char* A, global char* B, int K)
{
    __builtin_assume(K > 0);    // Always at least one K iteration.
    const int tM = 2;
    const int tN = 8;
    const int N = get_global_size(0);
    const int m = get_group_id(1) * tM;
    const int n = get_group_id(0) * tN;

    int2 sum = 0;
    for (int k = 0; k < K; k += tK) {
        int2    aData = load_a_rowmajor_d8_m2_k32_sg8(A, m, k, K);
        int8    bData = load_b_vnni_d8_k32_nx(B, k, n, N);
        sum = mat_mul_sg8(aData, bData, sum);
    }

    sum = activation(sum);
    store_c_rowmajor_int32_m2_nx(C, sum, m, n, N);
}

__attribute__((intel_reqd_sub_group_size(8))) __attribute__((reqd_work_group_size(8, 1, 1)))
kernel void i8_dpas_vnni_m4_n8(global int* C, global char* A, global char* B, int K)
{
    __builtin_assume(K > 0);    // Always at least one K iteration.
    const int tM = 4;
    const int tN = 8;
    const int N = get_global_size(0);
    const int m = get_group_id(1) * tM;
    const int n = get_group_id(0) * tN;

    int4 sum = 0;
    for (int k = 0; k < K; k += tK) {
        int4    aData = load_a_rowmajor_d8_m4_k32_sg8(A, m, k, K);
        int8    bData = load_b_vnni_d8_k32_nx(B, k, n, N);
        sum = mat_mul_sg8(aData, bData, sum);
    }

    sum = activation(sum);
    store_c_rowmajor_int32_m4_nx(C, sum, m, n, N);
}

__attribute__((intel_reqd_sub_group_size(8))) __attribute__((reqd_work_group_size(8, 1, 1)))
kernel void i8_dpas_vnni_m8_n8(global int* C, global char* A, global char* B, int K)
{
    __builtin_assume(K > 0);    // Always at least one K iteration.
    const int tM = 8;
    const int tN = 8;
    const int N = get_global_size(0);
    const int m = get_group_id(1) * tM;
    const int n = get_group_id(0) * tN;

    int8 sum = 0;
    for (int k = 0; k < K; k += tK) {
        int8    aData = load_a_rowmajor_d8_m8_k32_sg8(A, m, k, K);
        int8    bData = load_b_vnni_d8_k32_nx(B, k, n, N);
        sum = mat_mul_sg8(aData, bData, sum);
    }

    sum = activation(sum);
    store_c_rowmajor_int32_m8_nx(C, sum, m, n, N);
}

#endif // HAS_SIMD8

// rowmajor krenels:

__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16, 1, 1)))
kernel void i8_dpas_rowmajor_m1_n16(global int* C, global char* A, global char* B, int K)
{
    __builtin_assume(K > 0);    // Always at least one K iteration.
    const int tM = 1;
    const int tN = 16;
    const int N = get_global_size(0);
    const int m = get_group_id(1) * tM;
    const int n = get_group_id(0) * get_local_size(0);

    int sum = 0;
    for (int k = 0; k < K; k += tK) {
        short   aData = load_a_rowmajor_d8_m1_k32_sg16(A, m, k, K);
        int8    bData = load_b_rowmajor_d8_k32_nx(B, k, n, N);
        sum = mat_mul_sg16(aData, bData, sum);
    }

    sum = activation(sum);
    store_c_rowmajor_int32_m1_nx(C, sum, m, n, N);
}

__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16, 1, 1)))
kernel void i8_dpas_rowmajor_m2_n16(global int* C, global char* A, global char* B, int K)
{
    __builtin_assume(K > 0);    // Always at least one K iteration.
    const int tM = 2;
    const int tN = 16;
    const int N = get_global_size(0);
    const int m = get_group_id(1) * tM;
    const int n = get_group_id(0) * get_local_size(0);

    int2 sum = 0;
    for (int k = 0; k < K; k += tK) {
        short2  aData = load_a_rowmajor_d8_m2_k32_sg16(A, m, k, K);
        int8    bData = load_b_rowmajor_d8_k32_nx(B, k, n, N);
        sum = mat_mul_sg16(aData, bData, sum);
    }

    sum = activation(sum);
    store_c_rowmajor_int32_m2_nx(C, sum, m, n, N);
}

__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16, 1, 1)))
kernel void i8_dpas_rowmajor_m4_n16(global int* C, global char* A, global char* B, int K)
{
    __builtin_assume(K > 0);    // Always at least one K iteration.
    const int tM = 4;
    const int tN = 16;
    const int N = get_global_size(0);
    const int m = get_group_id(1) * tM;
    const int n = get_group_id(0) * get_local_size(0);

    int4 sum = 0;
    for (int k = 0; k < K; k += tK) {
        short4  aData = load_a_rowmajor_d8_m4_k32_sg16(A, m, k, K);
        int8    bData = load_b_rowmajor_d8_k32_nx(B, k, n, N);
        sum = mat_mul_sg16(aData, bData, sum);
    }

    sum = activation(sum);
    store_c_rowmajor_int32_m4_nx(C, sum, m, n, N);
}

__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16, 1, 1)))
kernel void i8_dpas_rowmajor_m8_n16(global int* C, global char* A, global char* B, int K)
{
    __builtin_assume(K > 0);    // Always at least one K iteration.
    const int tM = 8;
    const int tN = 16;
    const int N = get_global_size(0);
    const int m = get_group_id(1) * tM;
    const int n = get_group_id(0) * get_local_size(0);

    int8 sum = 0;
    for (int k = 0; k < K; k += tK) {
        short8  aData = load_a_rowmajor_d8_m8_k32_sg16(A, m, k, K);
        int8    bData = load_b_rowmajor_d8_k32_nx(B, k, n, N);
        sum = mat_mul_sg16(aData, bData, sum);
    }

    sum = activation(sum);
    store_c_rowmajor_int32_m8_nx(C, sum, m, n, N);
}

// vnni kernels:

__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16, 1, 1)))
kernel void i8_dpas_vnni_m1_n16(global int* C, global char* A, global char* B, int K)
{
    __builtin_assume(K > 0);    // Always at least one K iteration.
    const int tM = 1;
    const int tN = 16;
    const int N = get_global_size(0);
    const int m = get_group_id(1) * tM;
    const int n = get_group_id(0) * tN;

    int sum = 0;
    for (int k = 0; k < K; k += tK) {
        short   aData = load_a_rowmajor_d8_m1_k32_sg16(A, m, k, K);
        int8    bData = load_b_vnni_d8_k32_nx(B, k, n, N);
        sum = mat_mul_sg16(aData, bData, sum);
    }

    sum = activation(sum);
    store_c_rowmajor_int32_m1_nx(C, sum, m, n, N);
}

__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16, 1, 1)))
kernel void i8_dpas_vnni_m2_n16(global int* C, global char* A, global char* B, int K)
{
    __builtin_assume(K > 0);    // Always at least one K iteration.
    const int tM = 2;
    const int tN = 16;
    const int N = get_global_size(0);
    const int m = get_group_id(1) * tM;
    const int n = get_group_id(0) * tN;

    int2 sum = 0;
    for (int k = 0; k < K; k += tK) {
        short2  aData = load_a_rowmajor_d8_m2_k32_sg16(A, m, k, K);
        int8    bData = load_b_vnni_d8_k32_nx(B, k, n, N);
        sum = mat_mul_sg16(aData, bData, sum);
    }

    sum = activation(sum);
    store_c_rowmajor_int32_m2_nx(C, sum, m, n, N);
}

__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16, 1, 1)))
kernel void i8_dpas_vnni_m4_n16(global int* C, global char* A, global char* B, int K)
{
    __builtin_assume(K > 0);    // Always at least one K iteration.
    const int tM = 4;
    const int tN = 16;
    const int N = get_global_size(0);
    const int m = get_group_id(1) * tM;
    const int n = get_group_id(0) * tN;

    int4 sum = 0;
    for (int k = 0; k < K; k += tK) {
        short4  aData = load_a_rowmajor_d8_m4_k32_sg16(A, m, k, K);
        int8    bData = load_b_vnni_d8_k32_nx(B, k, n, N);
        sum = mat_mul_sg16(aData, bData, sum);
    }

    sum = activation(sum);
    store_c_rowmajor_int32_m4_nx(C, sum, m, n, N);
}

__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16, 1, 1)))
kernel void i8_dpas_vnni_m8_n16(global int* C, global char* A, global char* B, int K)
{
    __builtin_assume(K > 0);    // Always at least one K iteration.
    const int tM = 8;
    const int tN = 16;
    const int N = get_global_size(0);
    const int m = get_group_id(1) * tM;
    const int n = get_group_id(0) * tN;

    int8 sum = 0;
    for (int k = 0; k < K; k += tK) {
        short8  aData = load_a_rowmajor_d8_m8_k32_sg16(A, m, k, K);
        int8    bData = load_b_vnni_d8_k32_nx(B, k, n, N);
        sum = mat_mul_sg16(aData, bData, sum);
    }

    sum = activation(sum);
    store_c_rowmajor_int32_m8_nx(C, sum, m, n, N);
}

#ifdef cl_intel_subgroup_2d_block_io

__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16, 1, 1)))
kernel void i8_dpas_blockread_rowmajor_m1_n16(global int* C, global char* A, global char* B, int K)
{
    __builtin_assume(K > 0);    // Always at least one K iteration.
    const int tM = 1;
    const int tN = 16;
    const int M = get_global_size(1);
    const int N = get_global_size(0);
    const int m = get_group_id(1) * tM;
    const int n = get_group_id(0) * tN;

    int sum = 0;
    for (int k = 0; k < K; k += tK) {
        short   aData;
        intel_sub_group_2d_block_read_8b_1r32x1c(A, K * sizeof(char), M, K * sizeof(char), (int2)(k, m), (ushort*)&aData);
        int8    bData;
        intel_sub_group_2d_block_read_transform_8b_32r16x1c(B, N * sizeof(char), K, N * sizeof(char), (int2)(n, k), (uint*)&bData);
        sum = mat_mul_sg16(aData, bData, sum);
    }

    sum = activation(sum);
    intel_sub_group_2d_block_write_32b_1r16x1c(C, N * sizeof(float), M, N * sizeof(float), (int2)(n, m), (uint*)&sum);
}

__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16, 1, 1)))
kernel void i8_dpas_blockread_rowmajor_m2_n16(global int* C, global char* A, global char* B, int K)
{
    __builtin_assume(K > 0);    // Always at least one K iteration.
    const int tM = 2;
    const int tN = 16;
    const int M = get_global_size(1) * tM;
    const int N = get_global_size(0);
    const int m = get_group_id(1) * tM;
    const int n = get_group_id(0) * tN;

    int2 sum = 0;
    for (int k = 0; k < K; k += tK) {
        short2  aData;
        intel_sub_group_2d_block_read_8b_2r32x1c(A, K * sizeof(char), M, K * sizeof(char), (int2)(k, m), (ushort*)&aData);
        int8    bData;
        intel_sub_group_2d_block_read_transform_8b_32r16x1c(B, N * sizeof(char), K, N * sizeof(char), (int2)(n, k), (uint*)&bData);
        sum = mat_mul_sg16(aData, bData, sum);
    }

    sum = activation(sum);
    intel_sub_group_2d_block_write_32b_2r16x1c(C, N * sizeof(float), M, N * sizeof(float), (int2)(n, m), (uint*)&sum);
}

__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16, 1, 1)))
kernel void i8_dpas_blockread_rowmajor_m4_n16(global int* C, global char* A, global char* B, int K)
{
    __builtin_assume(K > 0);    // Always at least one K iteration.
    const int tM = 4;
    const int tN = 16;
    const int M = get_global_size(1) * tM;
    const int N = get_global_size(0);
    const int m = get_group_id(1) * tM;
    const int n = get_group_id(0) * tN;

    int4 sum = 0;
    for (int k = 0; k < K; k += tK) {
        short4  aData;
        intel_sub_group_2d_block_read_8b_4r32x1c(A, K * sizeof(char), M, K * sizeof(char), (int2)(k, m), (ushort*)&aData);
        int8    bData;
        intel_sub_group_2d_block_read_transform_8b_32r16x1c(B, N * sizeof(char), K, N * sizeof(char), (int2)(n, k), (uint*)&bData);
        sum = mat_mul_sg16(aData, bData, sum);
    }

    sum = activation(sum);
    intel_sub_group_2d_block_write_32b_4r16x1c(C, N * sizeof(float), M, N * sizeof(float), (int2)(n, m), (uint*)&sum);
}

__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16, 1, 1)))
kernel void i8_dpas_blockread_rowmajor_m8_n16(global int* C, global char* A, global char* B, int K)
{
    __builtin_assume(K > 0);    // Always at least one K iteration.
    const int tM = 8;
    const int tN = 16;
    const int M = get_global_size(1) * tM;
    const int N = get_global_size(0);
    const int m = get_group_id(1) * tM;
    const int n = get_group_id(0) * tN;

    int8 sum = 0;
    for (int k = 0; k < K; k += tK) {
        short8  aData;
        intel_sub_group_2d_block_read_8b_8r32x1c(A, K * sizeof(char), M, K * sizeof(char), (int2)(k, m), (ushort*)&aData);
        int8    bData;
        intel_sub_group_2d_block_read_transform_8b_32r16x1c(B, N * sizeof(char), K, N * sizeof(char), (int2)(n, k), (uint*)&bData);
        sum = mat_mul_sg16(aData, bData, sum);
    }

    sum = activation(sum);
    intel_sub_group_2d_block_write_32b_8r16x1c(C, N * sizeof(float), M, N * sizeof(float), (int2)(n, m), (uint*)&sum);
}

__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16, 1, 1)))
kernel void i8_dpas_blockread_vnni_m1_n16(global int* C, global char* A, global char* B, int K)
{
    __builtin_assume(K > 0);    // Always at least one K iteration.
    const int tM = 1;
    const int tN = 16;
    const int M = get_global_size(1) * tM;
    const int N = get_global_size(0);
    const int m = get_group_id(1) * tM;
    const int n = get_group_id(0) * tN;

    int sum = 0;
    for (int k = 0; k < K; k += tK) {
        short   aData;
        intel_sub_group_2d_block_read_8b_1r32x1c(A, K * sizeof(char), M, K * sizeof(char), (int2)(k, m), (ushort*)&aData);
        int8    bData;
        intel_sub_group_2d_block_read_32b_8r16x1c(B, N * sizeof(uint), K, N * sizeof(uint), (int2)(n, k / 4), (uint*)&bData);
        sum = mat_mul_sg16(aData, bData, sum);
    }

    sum = activation(sum);
    intel_sub_group_2d_block_write_32b_1r16x1c(C, N * sizeof(float), M, N * sizeof(float), (int2)(n, m), (uint*)&sum);
}

__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16, 1, 1)))
kernel void i8_dpas_blockread_vnni_m2_n16(global int* C, global char* A, global char* B, int K)
{
    __builtin_assume(K > 0);    // Always at least one K iteration.
    const int tM = 2;
    const int tN = 16;
    const int M = get_global_size(1) * tM;
    const int N = get_global_size(0);
    const int m = get_group_id(1) * tM;
    const int n = get_group_id(0) * tN;

    int2 sum = 0;
    for (int k = 0; k < K; k += tK) {
        short2  aData;
        intel_sub_group_2d_block_read_8b_2r32x1c(A, K * sizeof(char), M, K * sizeof(char), (int2)(k, m), (ushort*)&aData);
        int8    bData;
        intel_sub_group_2d_block_read_32b_8r16x1c(B, N * sizeof(uint), K, N * sizeof(uint), (int2)(n, k / 4), (uint*)&bData);
        sum = mat_mul_sg16(aData, bData, sum);
    }

    sum = activation(sum);
    intel_sub_group_2d_block_write_32b_2r16x1c(C, N * sizeof(float), M, N * sizeof(float), (int2)(n, m), (uint*)&sum);
}

__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16, 1, 1)))
kernel void i8_dpas_blockread_vnni_m4_n16(global int* C, global char* A, global char* B, int K)
{
    __builtin_assume(K > 0);    // Always at least one K iteration.
    const int tM = 4;
    const int tN = 16;
    const int M = get_global_size(1) * tM;
    const int N = get_global_size(0);
    const int m = get_group_id(1) * tM;
    const int n = get_group_id(0) * tN;

    int4 sum = 0;
    for (int k = 0; k < K; k += tK) {
        short4  aData;
        intel_sub_group_2d_block_read_8b_4r32x1c(A, K * sizeof(char), M, K * sizeof(char), (int2)(k, m), (ushort*)&aData);
        int8    bData;
        intel_sub_group_2d_block_read_32b_8r16x1c(B, N * sizeof(uint), K, N * sizeof(uint), (int2)(n, k / 4), (uint*)&bData);
        sum = mat_mul_sg16(aData, bData, sum);
    }

    sum = activation(sum);
    intel_sub_group_2d_block_write_32b_4r16x1c(C, N * sizeof(float), M, N * sizeof(float), (int2)(n, m), (uint*)&sum);
}

__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16, 1, 1)))
kernel void i8_dpas_blockread_vnni_m8_n16(global int* C, global char* A, global char* B, int K)
{
    __builtin_assume(K > 0);    // Always at least one K iteration.
    const int tM = 8;
    const int tN = 16;
    const int M = get_global_size(1) * tM;
    const int N = get_global_size(0);
    const int m = get_group_id(1) * tM;
    const int n = get_group_id(0) * tN;

    int8 sum = 0;
    for (int k = 0; k < K; k += tK) {
        short8  aData;
        intel_sub_group_2d_block_read_8b_8r32x1c(A, K * sizeof(char), M, K * sizeof(char), (int2)(k, m), (ushort*)&aData);
        int8    bData;
        intel_sub_group_2d_block_read_32b_8r16x1c(B, N * sizeof(uint), K, N * sizeof(uint), (int2)(n, k / 4), (uint*)&bData);
        sum = mat_mul_sg16(aData, bData, sum);
    }

    sum = activation(sum);
    intel_sub_group_2d_block_write_32b_8r16x1c(C, N * sizeof(float), M, N * sizeof(float), (int2)(n, m), (uint*)&sum);
}

__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16, 1, 1)))
kernel void i8_dpas_blockread_rowmajor_TN_m4_n16(global int* C, global char* A, global char* B, int K)
{
    __builtin_assume(K > 0);    // Always at least one K iteration.
    const int tM = 4;
    const int tN = 16;
    const int M = get_global_size(1) * tM;
    const int N = get_global_size(0);
    const int m = get_group_id(1) * tM;
    const int n = get_group_id(0) * tN;
    const int sglid = get_sub_group_local_id();

    int4 sum = 0;
    for (int k = 0; k < K; k += tK) {
        int2    readData;
        intel_sub_group_2d_block_read_transpose_32b_32r1x1c(A, M * sizeof(char), K, M * sizeof(char), (int2)(m / 4, k), (uint*)&readData);

        // Note: after the transpose block read:
        //      readData.s0 contains rows  0-15
        //      readData.s1 contains rows 16-31
        // So, WI0 has rows 0 and 16, WI1 has rows 1 and 17, etc.
        // We want WI0 to have rows 0 and 1, WI1 to have rows 2 and 3, etc.
        int shuffleIndex = sglid * 2 % 16;
        int loData0 = sub_group_shuffle(readData.s0, shuffleIndex);
        int hiData0 = sub_group_shuffle(readData.s1, shuffleIndex);
        int shuffledData0 = (sglid < 8) ? loData0 : hiData0;
        int loData1 = sub_group_shuffle(readData.s0, shuffleIndex + 1);
        int hiData1 = sub_group_shuffle(readData.s1, shuffleIndex + 1);
        int shuffledData1 = (sglid < 8) ? loData1 : hiData1;

        short4  aData;
        aData.s0 = as_short((char2)(as_char4(shuffledData0).s0, as_char4(shuffledData1).s0));
        aData.s1 = as_short((char2)(as_char4(shuffledData0).s1, as_char4(shuffledData1).s1));
        aData.s2 = as_short((char2)(as_char4(shuffledData0).s2, as_char4(shuffledData1).s2));
        aData.s3 = as_short((char2)(as_char4(shuffledData0).s3, as_char4(shuffledData1).s3));

        int8    bData;
        intel_sub_group_2d_block_read_transform_8b_32r16x1c(B, N * sizeof(char), K, N * sizeof(char), (int2)(n, k), (uint*)&bData);
        sum = mat_mul_sg16(aData, bData, sum);
    }

    sum = activation(sum);
    intel_sub_group_2d_block_write_32b_4r16x1c(C, N * sizeof(float), M, N * sizeof(float), (int2)(n, m), (uint*)&sum);
}

#endif // cl_intel_subgroup_2d_block_io

#if 0 // disable the tiled cases for now

// Tiled matrix multiplication kernels, generated from a template:

#define MM 1
#define NN 1
#include "matrix_kernel_tiled_i8.cl"
#undef MM
#undef NN

#define MM 2
#define NN 1
#include "matrix_kernel_tiled_i8.cl"
#undef MM
#undef NN

#define MM 1
#define NN 2
#include "matrix_kernel_tiled_i8.cl"
#undef MM
#undef NN

#define MM 2
#define NN 2
#include "matrix_kernel_tiled_i8.cl"
#undef MM
#undef NN

#define MM 4
#define NN 2
#include "matrix_kernel_tiled_i8.cl"
#undef MM
#undef NN

#define MM 2
#define NN 4
#include "matrix_kernel_tiled_i8.cl"
#undef MM
#undef NN

#define MM 4
#define NN 4
#include "matrix_kernel_tiled_i8.cl"
#undef MM
#undef NN

#endif // disable the tiled cases for now

#endif // defined(cl_intel_subgroups) && defined(cl_intel_subgroups_short) && defined(cl_intel_required_subgroup_size)

#undef tK
