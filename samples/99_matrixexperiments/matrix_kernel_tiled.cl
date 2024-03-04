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

#if !defined(SGS_PER_WG)
// Launch four subgroups per work-group, to maximize cache reuse.
#define SGS_PER_WG 4
#endif

#if !defined(PREFETCH_DISTANCE)
#define PREFETCH_DISTANCE 1
#endif

void HELPER_NAME(btile_load_rowmajor, MM, NN)(global ushort* B, int tN, int N, int k, int n, int8 bData[KK][NN])
{
    for (int kk = 0; kk < KK; kk++) {
        for (int nn = 0; nn < NN; nn++) {
            bData[kk][nn] = load_b_rowmajor_d16_k16_nx(B, k + kk * tK, n + nn * tN, N);
        }
    }
}

void HELPER_NAME(btile_load_vnni, MM, NN)(global ushort* B, int tN, int N, int k, int n, int8 bData[KK][NN])
{
    for (int kk = 0; kk < KK; kk++) {
        for (int nn = 0; nn < NN; nn++) {
            bData[kk][nn] = load_b_vnni_d16_k16_nx(B, k + kk * tK, n + nn * tN, N);
        }
    }
}

#if HAS_SIMD8

void HELPER_NAME(atile_prefetch_rowmajor_sg8, MM, NN)(global ushort* A, int tM, int K, int m, int prefetch_k)
{
    for (int kk = 0; kk < KK; kk+=2) {
        for (int mm = 0; mm < MM; mm++) {
            prefetch_a_rowmajor_d16_m8_k16v2_sg8(A, m + mm * tM, prefetch_k + kk * tK, K);
        }
    }
}

void HELPER_NAME(btile_prefetch_rowmajor_sg8, MM, NN)(global ushort* B, int tN, int N, int prefetch_k, int n)
{
    for (int kk = 0; kk < KK; kk++) {
        for (int nn = 0; nn < NN; nn+=4) {
            prefetch_b_rowmajor_d16_k16_n8v4_sg8(B, prefetch_k + kk * tK, n + nn * tN, N);
        }
    }
}

void HELPER_NAME(btile_prefetch_vnni_sg8, MM, NN)(global ushort* B, int tN, int N, int prefetch_k, int n)
{
    for (int kk = 0; kk < KK; kk++) {
        for (int nn = 0; nn < NN; nn+=2) {
            prefetch_b_vnni_d16_k16_n8v2_sg8(B, prefetch_k + kk * tK, n + nn * tN, N);
        }
    }
}

void HELPER_NAME(atile_load_rowmajor_sg8, MM, NN)(global ushort* A, int tM, int K, int m, int k, int8 aData[KK][MM])
{
    if (KK % 2 == 0) {
        for (int kk = 0; kk < KK; kk+=2) {
            for (int mm = 0; mm < MM; mm++) {
                int16   aTemp = load_a_rowmajor_d16_m8_k16v2_sg8(A, m + mm * tM, k + kk * tK, K);
                aData[kk + 0][mm] = aTemp.lo;
                aData[kk + 1][mm] = aTemp.hi;
            }
        }
    } else {
        for (int kk = 0; kk < KK; kk++) {
            for (int mm = 0; mm < MM; mm++) {
                aData[kk][mm] = load_a_rowmajor_d16_m8_k16_sg8(A, m + mm * tM, k + kk * tK, K);
            }
        }
    }
}

__attribute__((intel_reqd_sub_group_size(8))) __attribute__((reqd_work_group_size(8, SGS_PER_WG, 1)))
kernel void MM_KERNEL_NAME(bfloat16_dpas_rowmajor_tiled, 8, 8, MM, NN)(global float* C, global_aligned_ushort_ptr A, global_aligned_ushort_ptr B, int K)
{
    const int tM = 8;
    const int tN = 8;
    const int N = get_global_size(0) * NN;
    const int m = compute_m(SGS_PER_WG, tM, MM);
    const int n = get_group_id(0) * tN * NN;

    // Initial prefetch:
    int prefetch_k = 0;
    for (int p = 0; p < PREFETCH_DISTANCE; p++) {
        HELPER_NAME(atile_prefetch_rowmajor_sg8, MM, NN)(A, tM, K, m, prefetch_k);
        HELPER_NAME(btile_prefetch_rowmajor_sg8, MM, NN)(B, tN, N, prefetch_k, n);
        prefetch_k += tK * KK;
    }

    float8 sum[MM][NN];
    for (int mm = 0; mm < MM; mm++) {
        for (int nn = 0; nn < NN; nn++) {
            sum[mm][nn] = 0;
        }
    }

    split_barrier_arrive();

    for (int k = 0; k < K; k += tK * KK) {
        // Next prefetch:
        // TODO: skip prefetch on the last iterations.
        HELPER_NAME(atile_prefetch_rowmajor_sg8, MM, NN)(A, tM, K, m, prefetch_k);
        HELPER_NAME(btile_prefetch_rowmajor_sg8, MM, NN)(B, tN, N, prefetch_k, n);
        prefetch_k += tK * KK;

        int8    aData[KK][MM];
        HELPER_NAME(atile_load_rowmajor_sg8, MM, NN)(A, tM, K, m, k, aData);

        int8    bData[KK][NN];
        HELPER_NAME(btile_load_rowmajor, MM, NN)(B, tN, N, k, n, bData);

        for (int kk = 0; kk < KK; kk++) {
            for (int mm = 0; mm < MM; mm++) {
                for (int nn = 0; nn < NN; nn++) {
                    sum[mm][nn] = mat_mul_sg8(aData[kk][mm], bData[kk][nn], sum[mm][nn]);
                }
            }
        }

        split_barrier_wait();
        split_barrier_arrive();
    }

    split_barrier_wait();

    for (int nn = 0; nn < NN; nn++) {
        for (int mm = 0; mm < MM; mm++) {
            sum[mm][nn] = activation(sum[mm][nn]);
            store_c_rowmajor_fp32_m8_nx(C, sum[mm][nn], m + mm * tM, n + nn * tN, N);
        }
    }
}

__attribute__((intel_reqd_sub_group_size(8))) __attribute__((reqd_work_group_size(8, SGS_PER_WG, 1)))
kernel void MM_KERNEL_NAME(bfloat16_dpas_vnni_tiled, 8, 8, MM, NN)(global float* C, global_aligned_ushort_ptr A, global_aligned_ushort_ptr B, int K)
{
    const int tM = 8;
    const int tN = 8;
    const int N = get_global_size(0) * NN;
    const int m = compute_m(SGS_PER_WG, tM, MM);
    const int n = get_group_id(0) * tN * NN;

    // Initial prefetch:
    int prefetch_k = 0;
    for (int p = 0; p < PREFETCH_DISTANCE; p++) {
        HELPER_NAME(atile_prefetch_rowmajor_sg8, MM, NN)(A, tM, K, m, prefetch_k);
        HELPER_NAME(btile_prefetch_vnni_sg8, MM, NN)(B, tN, N, prefetch_k, n);
        prefetch_k += tK * KK;
    }

    float8 sum[MM][NN];
    for (int mm = 0; mm < MM; mm++) {
        for (int nn = 0; nn < NN; nn++) {
            sum[mm][nn] = 0;
        }
    }

    split_barrier_arrive();

    for (int k = 0; k < K; k += tK * KK) {
        // Next prefetch:
        // TODO: skip prefetch on the last iterations.
        HELPER_NAME(atile_prefetch_rowmajor_sg8, MM, NN)(A, tM, K, m, prefetch_k);
        HELPER_NAME(btile_prefetch_rowmajor_sg8, MM, NN)(B, tN, N, prefetch_k, n);
        prefetch_k += tK * KK;

        int8    aData[KK][MM];
        HELPER_NAME(atile_load_rowmajor_sg8, MM, NN)(A, tM, K, m, k, aData);

        int8    bData[KK][NN];
        HELPER_NAME(btile_load_vnni, MM, NN)(B, tN, N, k, n, bData);

        for (int kk = 0; kk < KK; kk++) {
            for (int nn = 0; nn < NN; nn++) {
                for (int mm = 0; mm < MM; mm++) {
                    sum[mm][nn] = mat_mul_sg8(aData[kk][mm], bData[kk][nn], sum[mm][nn]);
                }
            }
        }

        split_barrier_wait();
        split_barrier_arrive();
    }

    split_barrier_wait();

    for (int mm = 0; mm < MM; mm++) {
        for (int nn = 0; nn < NN; nn++) {
            sum[mm][nn] = activation(sum[mm][nn]);
            store_c_rowmajor_fp32_m8_nx(C, sum[mm][nn], m + mm * tM, n + nn * tN, N);
        }
    }
}

#endif // HAS_SIMD8

void HELPER_NAME(atile_prefetch_rowmajor, MM, NN)(global ushort* A, int tM, int K, int m, int prefetch_k)
{
    for (int kk = 0; kk < KK; kk+=2) {
        for (int mm = 0; mm < MM; mm+=2) {
            prefetch_a_rowmajor_d16_m8v2_k16v2_sg16(A, m + mm * tM, prefetch_k + kk * tK, K);
        }
    }
}

void HELPER_NAME(btile_prefetch_rowmajor, MM, NN)(global ushort* B, int tN, int N, int prefetch_k, int n)
{
    for (int kk = 0; kk < KK; kk++) {
        for (int nn = 0; nn < NN; nn+=2) {
            prefetch_b_rowmajor_d16_k16_n16v2_sg16(B, prefetch_k + kk * tK, n + nn * tN, N);
        }
    }
}

void HELPER_NAME(btile_prefetch_vnni, MM, NN)(global ushort* B, int tN, int N, int prefetch_k, int n)
{
    for (int kk = 0; kk < KK; kk+=2) {
        for (int nn = 0; nn < NN; nn++) {
            prefetch_b_vnni_d16_k16v2_n16_sg16(B, prefetch_k + kk * tK, n + nn * tN, N);
        }
    }
}

void HELPER_NAME(atile_load_rowmajor, MM, NN)(global ushort* A, int tM, int K, int m, int k, short8 aData[KK][MM])
{
    if (KK % 2 == 0) {
        for (int kk = 0; kk < KK; kk+=2) {
            for (int mm = 0; mm < MM; mm++) {
                short16 aTemp = load_a_rowmajor_d16_m8_k16v2_sg16(A, m + mm * tM, k + kk * tK, K);
                aData[kk + 0][mm] = aTemp.lo;
                aData[kk + 1][mm] = aTemp.hi;
            }
        }
    } else {
        for (int kk = 0; kk < KK; kk++) {
            for (int mm = 0; mm < MM; mm++) {
                aData[kk][mm] = load_a_rowmajor_d16_m8_k16_sg16(A, m + mm * tM, k + kk * tK, K);
            }
        }
    }
}

__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16, SGS_PER_WG, 1)))
kernel void MM_KERNEL_NAME(bfloat16_dpas_rowmajor_tiled, 8, 16, MM, NN)(global float* C, global_aligned_ushort_ptr A, global_aligned_ushort_ptr B, int K)
{
    const int tM = 8;
    const int tN = 16;
    const int N = get_global_size(0) * NN;
    const int m = compute_m(SGS_PER_WG, tM, MM);
    const int n = get_group_id(0) * tN * NN;

    // Initial prefetch:
    int prefetch_k = 0;
    for (int p = 0; p < PREFETCH_DISTANCE; p++) {
        HELPER_NAME(atile_prefetch_rowmajor, MM, NN)(A, tM, K, m, prefetch_k);
        HELPER_NAME(btile_prefetch_rowmajor, MM, NN)(B, tN, N, prefetch_k, n);
        prefetch_k += tK * KK;
    }

    float8 sum[MM][NN];
    for (int mm = 0; mm < MM; mm++) {
        for (int nn = 0; nn < NN; nn++) {
            sum[mm][nn] = 0;
        }
    }

    split_barrier_arrive();

    for (int k = 0; k < K; k += tK * KK) {
        // Next prefetch:
        // TODO: skip prefetch on the last iterations.
        HELPER_NAME(atile_prefetch_rowmajor, MM, NN)(A, tM, K, m, prefetch_k);
        HELPER_NAME(btile_prefetch_rowmajor, MM, NN)(B, tN, N, prefetch_k, n);
        prefetch_k += tK * KK;

        short8  aData[KK][MM];
        HELPER_NAME(atile_load_rowmajor, MM, NN)(A, tM, K, m, k, aData);

        int8    bData[KK][NN];
        HELPER_NAME(btile_load_rowmajor, MM, NN)(B, tN, N, k, n, bData);

        for (int kk = 0; kk < KK; kk++) {
            for (int nn = 0; nn < NN; nn++) {
                for (int mm = 0; mm < MM; mm++) {
                    sum[mm][nn] = mat_mul_sg16(aData[kk][mm], bData[kk][nn], sum[mm][nn]);
                }
            }
        }

        split_barrier_wait();
        split_barrier_arrive();
    }

    split_barrier_wait();

    for (int mm = 0; mm < MM; mm++) {
        for (int nn = 0; nn < NN; nn++) {
            sum[mm][nn] = activation(sum[mm][nn]);
            store_c_rowmajor_fp32_m8_nx(C, sum[mm][nn], m + mm * tM, n + nn * tN, N);
        }
    }
}

__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16, SGS_PER_WG, 1)))
kernel void MM_KERNEL_NAME(bfloat16_dpas_vnni_tiled, 8, 16, MM, NN)(global float* C, global_aligned_ushort_ptr A, global_aligned_ushort_ptr B, int K)
{
    const int tM = 8;
    const int tN = 16;
    const int N = get_global_size(0) * NN;
    const int m = compute_m(SGS_PER_WG, tM, MM);
    const int n = get_group_id(0) * tN * NN;

    // Initial prefetch:
    int prefetch_k = 0;
    for (int p = 0; p < PREFETCH_DISTANCE; p++) {
        HELPER_NAME(atile_prefetch_rowmajor, MM, NN)(A, tM, K, m, prefetch_k);
        HELPER_NAME(btile_prefetch_vnni, MM, NN)(B, tN, N, prefetch_k, n);
        prefetch_k += tK * KK;
    }

    float8 sum[MM][NN];
    for (int mm = 0; mm < MM; mm++) {
        for (int nn = 0; nn < NN; nn++) {
            sum[mm][nn] = 0;
        }
    }

    split_barrier_arrive();

    for (int k = 0; k < K; k += tK * KK) {
        // Next prefetch:
        // TODO: skip prefetch on the last iterations.
        HELPER_NAME(atile_prefetch_rowmajor, MM, NN)(A, tM, K, m, prefetch_k);
        HELPER_NAME(btile_prefetch_vnni, MM, NN)(B, tN, N, prefetch_k, n);
        prefetch_k += tK * KK;

        short8  aData[KK][MM];
        HELPER_NAME(atile_load_rowmajor, MM, NN)(A, tM, K, m, k, aData);

        int8    bData[KK][NN];
        HELPER_NAME(btile_load_vnni, MM, NN)(B, tN, N, k, n, bData);

        for (int kk = 0; kk < KK; kk++) {
            for (int nn = 0; nn < NN; nn++) {
                for (int mm = 0; mm < MM; mm++) {
                    sum[mm][nn] = mat_mul_sg16(aData[kk][mm], bData[kk][nn], sum[mm][nn]);
                }
            }
        }

        split_barrier_wait();
        split_barrier_arrive();
    }

    split_barrier_wait();

    for (int mm = 0; mm < MM; mm++) {
        for (int nn = 0; nn < NN; nn++) {
            sum[mm][nn] = activation(sum[mm][nn]);
            store_c_rowmajor_fp32_m8_nx(C, sum[mm][nn], m + mm * tM, n + nn * tN, N);
        }
    }
}

#ifdef cl_intel_subgroup_extended_block_read

void HELPER_NAME(atile_load_blockread_rowmajor, MM, NN)(global ushort* A, int tM, int M, int K, int m, int k, short8 aData[KK][MM])
{
    if (KK % 2 == 0 & MM % 4 == 0) {
        for (int kk = 0; kk < KK; kk+=2) {
            for (int mm = 0; mm < MM; mm+=4) {
                ushort8 tmp[2][4];
                intel_subgroup_block_read_u16_m32k16v2(A, K * sizeof(ushort), M, K * sizeof(ushort), (int2)(k + kk * tK, m + mm * tM), tmp);
                for (int tkk = 0; tkk < 2; tkk++) {
                    for (int tmm = 0; tmm < 4; tmm++) {
                        aData[kk + tkk][mm + tmm] = as_short8(tmp[tkk][tmm]);
                    }
                }
            }
        }
    } else if (KK % 2 == 0 & MM % 2 == 0) {
        for (int kk = 0; kk < KK; kk+=2) {
            for (int mm = 0; mm < MM; mm+=2) {
                ushort8 tmp[2][2];
                intel_subgroup_block_read_u16_m16k16v2(A, K * sizeof(ushort), M, K * sizeof(ushort), (int2)(k + kk * tK, m + mm * tM), tmp);
                for (int tkk = 0; tkk < 2; tkk++) {
                    for (int tmm = 0; tmm < 2; tmm++) {
                        aData[kk + tkk][mm + tmm] = as_short8(tmp[tkk][tmm]);
                    }
                }
            }
        }
    } else if (KK % 2 == 0) {
        for (int kk = 0; kk < KK; kk+=2) {
            for (int mm = 0; mm < MM; mm++) {
                short16 aTemp = as_short16(intel_subgroup_block_read_u16_m8k16v2(A, K * sizeof(ushort), M, K * sizeof(ushort), (int2)(k + kk * tK, m + mm * tM)));
                aData[kk + 0][mm] = aTemp.lo;
                aData[kk + 1][mm] = aTemp.hi;
            }
        }
    } else if (MM % 4 == 0) {
        for (int kk = 0; kk < KK; kk++) {
            for (int mm = 0; mm < MM; mm+=4) {
                ushort8 tmp[4];
                intel_subgroup_block_read_u16_m32k16(A, K * sizeof(ushort), M, K * sizeof(ushort), (int2)(k + kk * tK, m + mm * tM), tmp);
                for (int tmm = 0; tmm < 4; tmm++) {
                    aData[kk][mm + tmm] = as_short8(tmp[tmm]);
                }
            }
        }
    } else {
        for (int kk = 0; kk < KK; kk++) {
            for (int mm = 0; mm < MM; mm++) {
                aData[kk][mm] = as_short8(intel_subgroup_block_read_u16_m8k16(A, K * sizeof(ushort), M, K * sizeof(ushort), (int2)(k + kk * tK, m + mm * tM)));
            }
        }
    }
}

// TODO: consider swapping KK and NN order!
void HELPER_NAME(btile_load_blockread_rowmajor, MM, NN)(global ushort* B, int tN, int K, int N, int k, int n, int8 bData[KK][NN])
{
    if (KK % 2 == 0 & NN % 2 == 0) {
        for (int kk = 0; kk < KK; kk+=2) {
            for (int nn = 0; nn < NN; nn+=2) {
                int8 tmp[2][2];
                intel_subgroup_block_read_transform_u16_k32n16v2(B, N * sizeof(ushort), K, N * sizeof(ushort), (int2)(n + nn * tN, k + kk * tK), tmp);
                for (int tnn = 0; tnn < 2; tnn++) {
                    for (int tkk = 0; tkk < 2; tkk++) {
                        bData[kk + tkk][nn + tnn] = tmp[tnn][tkk];
                    }
                }
            }
        }
    } else if (NN % 2 == 0) {
        for (int kk = 0; kk < KK; kk++) {
            for (int nn = 0; nn < NN; nn+=2) {
                int16 bTemp = intel_subgroup_block_read_transform_u16_k16n16v2(B, N * sizeof(ushort), K, N * sizeof(ushort), (int2)(n + nn * tN, k + kk * tK));
                bData[kk][nn + 0] = bTemp.lo;
                bData[kk][nn + 1] = bTemp.hi;
            }
        }
    } else if (KK % 2 == 0) {
        for (int kk = 0; kk < KK; kk+=2) {
            for (int nn = 0; nn < NN; nn++) {
                int16 bTemp = intel_subgroup_block_read_transform_u16_k32n16(B, N * sizeof(ushort), K, N * sizeof(ushort), (int2)(n + nn * tN, k + kk * tK));
                bData[kk + 0][nn] = bTemp.lo;
                bData[kk + 1][nn] = bTemp.hi;
            }
        }
    } else {
        for (int kk = 0; kk < KK; kk++) {
            for (int nn = 0; nn < NN; nn++) {
                bData[kk][nn] = intel_subgroup_block_read_transform_u16_k16n16(B, N * sizeof(ushort), K, N * sizeof(ushort), (int2)(n + nn * tN, k + kk * tK));
            }
        }
    }
}

void HELPER_NAME(btile_load_blockread_vnni, MM, NN)(global ushort* B, int tN, int K, int N, int k, int n, int8 bData[KK][NN])
{
    if (KK % 2 == 0) {
        for (int kk = 0; kk < KK; kk+=2) {
            for (int nn = 0; nn < NN; nn++) {
                int16 bTemp = as_int16(intel_subgroup_block_read_u32_m16k16(B, N * sizeof(uint), K, N * sizeof(uint), (int2)(n + nn * tN, (k + kk * tK) / 2)));
                bData[kk + 0][nn] = bTemp.lo;
                bData[kk + 1][nn] = bTemp.hi;
            }
        }
    } else {
        for (int kk = 0; kk < KK; kk++) {
            for (int nn = 0; nn < NN; nn++) {
                bData[kk][nn] = as_int8(intel_subgroup_block_read_u32_m8k16(B, N * sizeof(uint), K, N * sizeof(uint), (int2)(n + nn * tN, (k + kk * tK) / 2)));
            }
        }
    }
}

void HELPER_NAME(atile_block_prefetch_rowmajor, MM, NN)(global ushort* A, int tM, int M, int K, int m, int k)
{
    if (KK % 2 == 0 & MM % 4 == 0) {
        for (int kk = 0; kk < KK; kk+=2) {
            for (int mm = 0; mm < MM; mm+=4) {
                intel_subgroup_block_prefetch_u16_m32k16v2(A, K * sizeof(ushort), M, K * sizeof(ushort), (int2)(k + kk * tK, m + mm * tM));
            }
        }
    } else if (KK % 2 == 0 & MM % 2 == 0) {
        for (int kk = 0; kk < KK; kk+=2) {
            for (int mm = 0; mm < MM; mm+=2) {
                intel_subgroup_block_prefetch_u16_m16k16v2(A, K * sizeof(ushort), M, K * sizeof(ushort), (int2)(k + kk * tK, m + mm * tM));
            }
        }
    } else if (KK % 2 == 0) {
        for (int kk = 0; kk < KK; kk+=2) {
            for (int mm = 0; mm < MM; mm++) {
                intel_subgroup_block_prefetch_u16_m8k16v2(A, K * sizeof(ushort), M, K * sizeof(ushort), (int2)(k + kk * tK, m + mm * tM));
            }
        }
    } else if (MM % 4 == 0) {
        for (int kk = 0; kk < KK; kk++) {
            for (int mm = 0; mm < MM; mm+=4) {
                intel_subgroup_block_prefetch_u16_m32k16(A, K * sizeof(ushort), M, K * sizeof(ushort), (int2)(k + kk * tK, m + mm * tM));
            }
        }
    } else {
        for (int kk = 0; kk < KK; kk++) {
            for (int mm = 0; mm < MM; mm++) {
                intel_subgroup_block_prefetch_u16_m8k16(A, K * sizeof(ushort), M, K * sizeof(ushort), (int2)(k + kk * tK, m + mm * tM));
            }
        }
    }
}

__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16, SGS_PER_WG, 1)))
kernel void MM_KERNEL_NAME(bfloat16_dpas_blockread_rowmajor_tiled, 8, 16, MM, NN)(global float* C, global ushort* A, global ushort* B, int K)

{
    const int tM = 8;
    const int tN = 16;
    const int M = get_global_size(1) * tM * MM;
    const int N = get_global_size(0) * NN;
    const int m = compute_m(SGS_PER_WG, tM, MM);
    const int n = get_group_id(0) * tN * NN;

    int prefetch_k = 0;
    for (int p = 0; p < PREFETCH_DISTANCE; p++) {
        HELPER_NAME(btile_prefetch_rowmajor, MM, NN)(B, tN, N, prefetch_k, n);
        HELPER_NAME(atile_block_prefetch_rowmajor, MM, NN)(A, tM, M, K, m, prefetch_k);
        prefetch_k += tK * KK;
    }

    float8 sum[MM][NN];
    for (int mm = 0; mm < MM; mm++) {
        for (int nn = 0; nn < NN; nn++) {
            sum[mm][nn] = 0;
        }
    }

    split_barrier_arrive();

    for (int k = 0; k < K; k += tK * KK) {
        // TODO: skip prefetch on the last iterations.
        HELPER_NAME(btile_prefetch_rowmajor, MM, NN)(B, tN, N, prefetch_k, n);

        short8  aData[KK][MM];
        HELPER_NAME(atile_load_blockread_rowmajor, MM, NN)(A, tM, M, K, m, k, aData);

        int8    bData[KK][NN];
        HELPER_NAME(btile_load_blockread_rowmajor, MM, NN)(B, tN, K, N, k, n, bData);

        HELPER_NAME(atile_block_prefetch_rowmajor, MM, NN)(A, tM, M, K, m, prefetch_k);
        prefetch_k += tK * KK;

        for (int kk = 0; kk < KK; kk++) {
            for (int nn = 0; nn < NN; nn++) {
                for (int mm = 0; mm < MM; mm++) {
                    sum[mm][nn] = mat_mul_sg16(aData[kk][mm], bData[kk][nn], sum[mm][nn]);
                }
            }
        }

        split_barrier_wait();
        split_barrier_arrive();
    }

    split_barrier_wait();

    for (int mm = 0; mm < MM; mm++) {
        for (int nn = 0; nn < NN; nn++) {
            sum[mm][nn] = activation(sum[mm][nn]);
            intel_subgroup_block_write_u32_m8k16(C, N * sizeof(float), M, N * sizeof(float), (int2)(n + nn * tN, m + mm * tM), as_uint8(sum[mm][nn]));
        }
    }
}

__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16, SGS_PER_WG, 1)))
kernel void MM_KERNEL_NAME(bfloat16_dpas_blockread_vnni_tiled, 8, 16, MM, NN)(global float* C, global ushort* A, global ushort* B, int K)
{
    const int tM = 8;
    const int tN = 16;
    const int M = get_global_size(1) * tM * MM;
    const int N = get_global_size(0) * NN;
    const int m = compute_m(SGS_PER_WG, tM, MM);
    const int n = get_group_id(0) * tN * NN;

    int prefetch_k = 0;
    for (int p = 0; p < PREFETCH_DISTANCE; p++) {
        HELPER_NAME(btile_prefetch_vnni, MM, NN)(B, tN, N, prefetch_k, n);
        HELPER_NAME(atile_block_prefetch_rowmajor, MM, NN)(A, tM, M, K, m, prefetch_k);
        prefetch_k += tK * KK;
    }

    float8 sum[MM][NN];
    for (int mm = 0; mm < MM; mm++) {
        for (int nn = 0; nn < NN; nn++) {
            sum[mm][nn] = 0;
        }
    }

    split_barrier_arrive();

    for (int k = 0; k < K; k += tK * KK) {
        // TODO: skip prefetch on the last iterations.
        HELPER_NAME(btile_prefetch_vnni, MM, NN)(B, tN, N, prefetch_k, n);

        short8  aData[KK][MM];
        HELPER_NAME(atile_load_blockread_rowmajor, MM, NN)(A, tM, M, K, m, k, aData);

        int8    bData[KK][NN];
        HELPER_NAME(btile_load_blockread_vnni, MM, NN)(B, tN, K, N, k, n, bData);

        HELPER_NAME(atile_block_prefetch_rowmajor, MM, NN)(A, tM, M, K, m, prefetch_k);
        prefetch_k += tK * KK;

        for (int kk = 0; kk < KK; kk++) {
            for (int nn = 0; nn < NN; nn++) {
                for (int mm = 0; mm < MM; mm++) {
                    sum[mm][nn] = mat_mul_sg16(aData[kk][mm], bData[kk][nn], sum[mm][nn]);
                }
            }
        }

        split_barrier_wait();
        split_barrier_arrive();
    }

    split_barrier_wait();

    for (int mm = 0; mm < MM; mm++) {
        for (int nn = 0; nn < NN; nn++) {
            sum[mm][nn] = activation(sum[mm][nn]);
            intel_subgroup_block_write_u32_m8k16(C, N * sizeof(float), M, N * sizeof(float), (int2)(n + nn * tN, m + mm * tM), as_uint8(sum[mm][nn]));
        }
    }
}

#endif // cl_intel_subgroup_extended_block_read
