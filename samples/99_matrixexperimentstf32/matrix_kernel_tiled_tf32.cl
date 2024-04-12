#if !defined(tK)
#error "tK is undefined!  This should be defined as the K dimension of the matrix tiles, which is dependent on the elemement type, likely 16 or 32."
#endif

#if !defined(MM)
#error "MM is undefined!  This should be defined as the number of matrix tiles in the M dimension."
#endif

#if !defined(NN)
#error "NN is undefined!  This should be defined as the number of matrix tiles in the N dimension."
#endif

#if !defined(KK)
#define KK 1
#endif

#if !defined(cl_intel_split_work_group_barrier) || defined(NO_SPLIT_BARRIERS)
#if !defined(cl_intel_split_work_group_barrier)
#warning "Unexpected: cl_intel_split_work_group_barrier is not supported?"
#endif
#define split_barrier_arrive()
#define split_barrier_wait()
#else
#define split_barrier_arrive()  intel_work_group_barrier_arrive(0)
#define split_barrier_wait()    intel_work_group_barrier_wait(0)
#endif

#define MM_KERNEL_NAMEX(PREFIX, tM, tN, MM, NN) PREFIX ## _m ## tM ## _n ## tN ## _ ## MM ## x ## NN
#define MM_KERNEL_NAME(PREFIX, tM, tN, MM, NN)  MM_KERNEL_NAMEX(PREFIX, tM, tN, MM, NN)

#define HELPER_NAMEX(PREFIX, MM, NN) PREFIX ## _m ## MM ## _n ## NN
#define HELPER_NAME(PREFIX, MM, NN)  HELPER_NAMEX(PREFIX, MM, NN)

#if !defined(SGS_PER_WG_X)
#define SGS_PER_WG_X 1
#endif

#if !defined(SGS_PER_WG_Y)
#define SGS_PER_WG_Y 4
#endif

#if !defined(PREFETCH_DISTANCE)
#define PREFETCH_DISTANCE 1
#endif

void HELPER_NAME(btile_load_rowmajor, MM, NN)(global float* B, int tN, int N, int k, int n, float8 bData[NN][KK])
{
    for (int kk = 0; kk < KK; kk++) {
        for (int nn = 0; nn < NN; nn++) {
            bData[nn][kk] = load_b_rowmajor_d32_k8_nx(B, k + kk * tK, n + nn * tN, N);
        }
    }
}

void HELPER_NAME(atile_prefetch_rowmajor_sg16, MM, NN)(global float* A, int tM, int K, int m, int prefetch_k)
{
    for (int kk = 0; kk < KK; kk+=2) {
        for (int mm = 0; mm < MM; mm+=2) {
            prefetch_a_rowmajor_d32_m8v2_k8v2_sg16(A, m + mm * tM, prefetch_k + kk * tK, K);
        }
    }
}

void HELPER_NAME(btile_prefetch_rowmajor_sg16, MM, NN)(global float* B, int tN, int N, int prefetch_k, int n)
{
    for (int kk = 0; kk < KK; kk+=2) {
        for (int nn = 0; nn < NN; nn+=2) {
            prefetch_b_rowmajor_d32_k8v2_n8v2_sg16(B, prefetch_k + kk * tK, n + nn * tN, N);
        }
    }
}

void HELPER_NAME(atile_load_rowmajor_sg16, MM, NN)(global float* A, int tM, int K, int m, int k, float4 aData[KK][MM])
{
    for (int kk = 0; kk < KK; kk++) {
        for (int mm = 0; mm < MM; mm++) {
            aData[kk][mm] = load_a_rowmajor_d32_m8_k8_sg16(A, m + mm * tM, k + kk * tK, K);
        }
    }
}

__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16 * SGS_PER_WG_X, SGS_PER_WG_Y, 1)))
kernel void MM_KERNEL_NAME(tf32_dpas_rowmajor_tiled, 8, 16, MM, NN)(global float* C, global float* A, global float* B, int K)
{
    __builtin_assume(K > 0);    // Always at least one K iteration.
    const int tM = 8;
    const int tN = 16;
    const int N = get_global_size(0) * NN;
    const int m = compute_m(SGS_PER_WG_X, SGS_PER_WG_Y, tM, MM);
    const int n = compute_n(SGS_PER_WG_X, SGS_PER_WG_Y, tN, NN);

    // Initial prefetch:
    int prefetch_k = 0;
    for (int p = 0; p < PREFETCH_DISTANCE; p++) {
        HELPER_NAME(atile_prefetch_rowmajor_sg16, MM, NN)(A, tM, K, m, prefetch_k);
        HELPER_NAME(btile_prefetch_rowmajor_sg16, MM, NN)(B, tN, N, prefetch_k, n);
        prefetch_k += tK * KK;
    }

    float8 sum[NN][MM];
    for (int mm = 0; mm < MM; mm++) {
        for (int nn = 0; nn < NN; nn++) {
            sum[nn][mm] = 0;
        }
    }

    split_barrier_arrive();

    for (int k = 0; k < K; k += tK * KK) {
        // Next prefetch:
        // TODO: skip prefetch on the last iterations.
        HELPER_NAME(atile_prefetch_rowmajor_sg16, MM, NN)(A, tM, K, m, prefetch_k);
        HELPER_NAME(btile_prefetch_rowmajor_sg16, MM, NN)(B, tN, N, prefetch_k, n);
        prefetch_k += tK * KK;

        float4  aData[KK][MM];
        HELPER_NAME(atile_load_rowmajor_sg16, MM, NN)(A, tM, K, m, k, aData);

        float8  bData[NN][KK];
        HELPER_NAME(btile_load_rowmajor, MM, NN)(B, tN, N, k, n, bData);

        for (int kk = 0; kk < KK; kk++) {
            for (int mm = 0; mm < MM; mm++) {
                for (int nn = 0; nn < NN; nn++) {
                    sum[nn][mm] = mat_mul_sg16(aData[kk][mm], bData[nn][kk], sum[nn][mm]);
                }
            }
        }

        split_barrier_wait();
        split_barrier_arrive();
    }

    split_barrier_wait();

    for (int nn = 0; nn < NN; nn++) {
        for (int mm = 0; mm < MM; mm++) {
            sum[nn][mm] = activation(sum[nn][mm]);
            store_c_rowmajor_fp32_m8_nx(C, sum[nn][mm], m + mm * tM, n + nn * tN, N);
        }
    }
}

#ifdef cl_intel_subgroup_extended_block_read

void HELPER_NAME(atile_block_load_rowmajor, MM, NN)(global float* A, int tM, int M, int K, int m, int k, float4 aData[KK][MM])
{
    if (KK % 2 == 0) {
        for (int kk = 0; kk < KK; kk+=2) {
            for (int mm = 0; mm < MM; mm++) {
                //if (get_sub_group_local_id() == 0) {
                //    printf("atile block load    : %d, %d, %2d:           m = %3d, k = %3d, mm = %2d, kk = %2d, coord = %3d, %3d\n", (int)get_group_id(1), (int)get_group_id(0), get_sub_group_id(), m, k, mm, kk, k + kk * tK, m + mm * tM);
                //}
                float8 aTemp = as_float8(intel_subgroup_block_read_u32_m8k8v2(A, K * sizeof(float), M, K * sizeof(float), (int2)(k + kk * tK, m + mm * tM)));
                aData[kk + 0][mm] = aTemp.lo;
                aData[kk + 1][mm] = aTemp.hi;
            }
        }
    } else {
        for (int kk = 0; kk < KK; kk++) {
            for (int mm = 0; mm < MM; mm++) {
                aData[kk][mm] = as_float4(intel_subgroup_block_read_u32_m8k8(A, K * sizeof(float), M, K * sizeof(float), (int2)(k + kk * tK, m + mm * tM)));
            }
        }
    }
}

void HELPER_NAME(btile_block_load_rowmajor, MM, NN)(global float* B, int tN, int K, int N, int k, int n, float8 bData[NN][KK])
{
    for (int kk = 0; kk < KK; kk++) {
        for (int nn = 0; nn < NN; nn++) {
            bData[nn][kk] = as_float8(intel_subgroup_block_read_u32_m8k16(B, N * sizeof(float), K, N * sizeof(float), (int2)(n + nn * tN, k + kk * tK)));
        }
    }
}

__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16 * SGS_PER_WG_X, SGS_PER_WG_Y, 1)))
kernel void MM_KERNEL_NAME(tf32_dpas_blockread_rowmajor_tiled, 8, 16, MM, NN)(global float* C, global float* A, global float* B, int K)
{
    __builtin_assume(K > 0);    // Always at least one K iteration.
    const int tM = 8;
    const int tN = 16;
    const int M = get_global_size(1) * tM * MM;
    const int N = get_global_size(0) * NN;
    const int m = compute_m(SGS_PER_WG_X, SGS_PER_WG_Y, tM, MM);
    const int n = compute_n(SGS_PER_WG_X, SGS_PER_WG_Y, tN, NN);

    int prefetch_k = 0;
    for (int p = 0; p < PREFETCH_DISTANCE; p++) {
        HELPER_NAME(atile_prefetch_rowmajor_sg16, MM, NN)(A, tM, K, m, prefetch_k);
        HELPER_NAME(btile_prefetch_rowmajor_sg16, MM, NN)(B, tN, N, prefetch_k, n);
        prefetch_k += tK * KK;
    }

    float8 sum[NN][MM];
    for (int mm = 0; mm < MM; mm++) {
        for (int nn = 0; nn < NN; nn++) {
            sum[nn][mm] = 0;
        }
    }

    split_barrier_arrive();

    for (int k = 0; k < K; k += tK * KK) {
        // TODO: skip prefetch on the last iterations.
        HELPER_NAME(atile_prefetch_rowmajor_sg16, MM, NN)(A, tM, K, m, prefetch_k);
        HELPER_NAME(btile_prefetch_rowmajor_sg16, MM, NN)(B, tN, N, prefetch_k, n);
        prefetch_k += tK * KK;

        float4  aData[KK][MM];
        HELPER_NAME(atile_block_load_rowmajor, MM, NN)(A, tM, M, K, m, k, aData);

        float8  bData[NN][KK];
        HELPER_NAME(btile_block_load_rowmajor, MM, NN)(B, tN, K, N, k, n, bData);

        for (int kk = 0; kk < KK; kk++) {
            for (int nn = 0; nn < NN; nn++) {
                for (int mm = 0; mm < MM; mm++) {
                    sum[nn][mm] = mat_mul_sg16(aData[kk][mm], bData[nn][kk], sum[nn][mm]);
                }
            }
        }

        split_barrier_wait();
        split_barrier_arrive();
    }

    split_barrier_wait();

    for (int mm = 0; mm < MM; mm++) {
        for (int nn = 0; nn < NN; nn++) {
            sum[nn][mm] = activation(sum[nn][mm]);
            intel_subgroup_block_write_u32_m8k16(C, N * sizeof(float), M, N * sizeof(float), (int2)(n + nn * tN, m + mm * tM), as_uint8(sum[nn][mm]));
        }
    }
}

#endif // cl_intel_subgroup_extended_block_read
