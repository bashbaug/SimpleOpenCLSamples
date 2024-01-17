#if !defined(tK)
#error "tK is undefined!  This should be defined as the K dimension of the matrix tiles, which is dependent on the elemement type, likely 16 or 32."
#endif

#if !defined(MM)
#error "MM is undefined!  This should be defined as the number of matrix tiles in the M dimension."
#endif

#if !defined(NN)
#error "NN is undefined!  This should be defined as the number of matrix tiles in the N dimension."
#endif

#define MM_KERNEL_NAMEX(PREFIX, tM, tN, MM, NN) PREFIX ## _m ## tM ## _n ## tN ## _ ## MM ## x ## NN
#define MM_KERNEL_NAME(PREFIX, tM, tN, MM, NN)  MM_KERNEL_NAMEX(PREFIX, tM, tN, MM, NN)

#if HAS_SIMD8

__attribute__((intel_reqd_sub_group_size(8))) __attribute__((reqd_work_group_size(8, 1, 1)))
kernel void MM_KERNEL_NAME(bfloat16_dpas_rowmajor_tiled, 8, 8, MM, NN)(global float* C, global ushort* A, global ushort* B, int K)
{
    const int tM = 8;
    const int tN = 8;
    const int N = get_global_size(0) * NN;
    const int m = get_group_id(1) * tM * MM;
    const int n = get_group_id(0) * tN * NN;

    float8 sum[MM][NN];
    for (int mm = 0; mm < MM; mm++) {
        for (int nn = 0; nn < NN; nn++) {
            sum[mm][nn] = 0;
        }
    }

    for (int k = 0; k < K; k += tK) {
        int8    aData[MM];
        for (int mm = 0; mm < MM; mm++) {
            aData[mm] = load_a_rowmajor_d16_m8_k16_sg8(A, m + mm * tM, k, K);
        }

        int8    bData[NN];
        for (int nn = 0; nn < NN; nn++) {
            bData[nn] = load_b_rowmajor_d16_k16_nx(B, k, n + nn * tN, N);
        }

        for (int mm = 0; mm < MM; mm++) {
            for (int nn = 0; nn < NN; nn++) {
                sum[mm][nn] = mat_mul_sg8(aData[mm], bData[nn], sum[mm][nn]);
            }
        }
    }

    for (int mm = 0; mm < MM; mm++) {
        for (int nn = 0; nn < NN; nn++) {
            store_c_rowmajor_fp32_m8_nx(C, sum[mm][nn], m + mm * tM, n + nn * tN, N);
        }
    }
}

__attribute__((intel_reqd_sub_group_size(8))) __attribute__((reqd_work_group_size(8, 1, 1)))
kernel void MM_KERNEL_NAME(bfloat16_dpas_vnni_tiled, 8, 8, MM, NN)(global float* C, global ushort* A, global ushort* B, int K)
{
    const int tM = 8;
    const int tN = 8;
    const int N = get_global_size(0) * NN;
    const int m = get_group_id(1) * tM * MM;
    const int n = get_group_id(0) * tN * NN;

    float8 sum[MM][NN];
    for (int mm = 0; mm < MM; mm++) {
        for (int nn = 0; nn < NN; nn++) {
            sum[mm][nn] = 0;
        }
    }

    for (int k = 0; k < K; k += tK) {
        int8    aData[MM];
        for (int mm = 0; mm < MM; mm++) {
            aData[mm] = load_a_rowmajor_d16_m8_k16_sg8(A, m + mm * tM, k, K);
        }

        int8    bData[NN];
        for (int nn = 0; nn < NN; nn++) {
            bData[nn] = load_b_vnni_d16_k16_nx(B, k, n + nn * tN, N);
        }

        for (int mm = 0; mm < MM; mm++) {
            for (int nn = 0; nn < NN; nn++) {
                sum[mm][nn] = mat_mul_sg8(aData[mm], bData[nn], sum[mm][nn]);
            }
        }
    }

    for (int mm = 0; mm < MM; mm++) {
        for (int nn = 0; nn < NN; nn++) {
            store_c_rowmajor_fp32_m8_nx(C, sum[mm][nn], m + mm * tM, n + nn * tN, N);
        }
    }
}

#endif // HAS_SIMD8

__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16, 1, 1)))
kernel void MM_KERNEL_NAME(bfloat16_dpas_rowmajor_tiled, 8, 16, MM, NN)(global float* C, global ushort* A, global ushort* B, int K)
{
    const int tM = 8;
    const int tN = 16;
    const int N = get_global_size(0) * NN;
    const int m = get_group_id(1) * tM * MM;
    const int n = get_group_id(0) * tN * NN;

    float8 sum[MM][NN];
    for (int mm = 0; mm < MM; mm++) {
        for (int nn = 0; nn < NN; nn++) {
            sum[mm][nn] = 0;
        }
    }

    for (int k = 0; k < K; k += tK) {
        short8  aData[MM];
        for (int mm = 0; mm < MM; mm++) {
            aData[mm] = load_a_rowmajor_d16_m8_k16_sg16(A, m + mm * tM, k, K);
        }

        int8    bData[NN];
        for (int nn = 0; nn < NN; nn++) {
            bData[nn] = load_b_rowmajor_d16_k16_nx(B, k, n + nn * tN, N);
        }

        for (int mm = 0; mm < MM; mm++) {
            for (int nn = 0; nn < NN; nn++) {
                sum[mm][nn] = mat_mul_sg16(aData[mm], bData[nn], sum[mm][nn]);
            }
        }
    }

    for (int mm = 0; mm < MM; mm++) {
        for (int nn = 0; nn < NN; nn++) {
            store_c_rowmajor_fp32_m8_nx(C, sum[mm][nn], m + mm * tM, n + nn * tN, N);
        }
    }
}

__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16, 1, 1)))
kernel void MM_KERNEL_NAME(bfloat16_dpas_vnni_tiled, 8, 16, MM, NN)(global float* C, global ushort* A, global ushort* B, int K)
{
    const int tM = 8;
    const int tN = 16;
    const int N = get_global_size(0) * NN;
    const int m = get_group_id(1) * tM * MM;
    const int n = get_group_id(0) * tN * NN;

    float8 sum[MM][NN];
    for (int mm = 0; mm < MM; mm++) {
        for (int nn = 0; nn < NN; nn++) {
            sum[mm][nn] = 0;
        }
    }

    for (int k = 0; k < K; k += tK) {
        short8  aData[MM];
        for (int mm = 0; mm < MM; mm++) {
            aData[mm] = load_a_rowmajor_d16_m8_k16_sg16(A, m + mm * tM, k, K);
        }

        int8    bData[NN];
        for (int nn = 0; nn < NN; nn++) {
            bData[nn] = load_b_vnni_d16_k16_nx(B, k, n + nn * tN, N);
        }

        for (int mm = 0; mm < MM; mm++) {
            for (int nn = 0; nn < NN; nn++) {
                sum[mm][nn] = mat_mul_sg16(aData[mm], bData[nn], sum[mm][nn]);
            }
        }
    }

    for (int mm = 0; mm < MM; mm++) {
        for (int nn = 0; nn < NN; nn++) {
            store_c_rowmajor_fp32_m8_nx(C, sum[mm][nn], m + mm * tM, n + nn * tN, N);
        }
    }
}

#ifdef cl_intel_subgroup_extended_block_read

__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16, 1, 1)))
kernel void MM_KERNEL_NAME(bfloat16_dpas_blockread_rowmajor_tiled, 8, 16, MM, NN)(global float* C, global ushort* A, global ushort* B, int K)
{
    const int tM = 8;
    const int tN = 16;
    const int M = get_global_size(1) * tM * MM;
    const int N = get_global_size(0) * NN;
    const int m = get_group_id(1) * tM * MM;
    const int n = get_group_id(0) * tN * NN;

    float8 sum[MM][NN];
    for (int mm = 0; mm < MM; mm++) {
        for (int nn = 0; nn < NN; nn++) {
            sum[mm][nn] = 0;
        }
    }

    for (int k = 0; k < K; k += tK) {
        short8  aData[MM];
        if (MM % 2 == 0) {
            for (int mm = 0; mm < MM; mm += 2) {
                short16 aTemp = as_short16(intel_subgroup_block_read_u16_m16k16(A, K * sizeof(ushort), M, K * sizeof(ushort), (int2)(k, m + mm * tM)));
                aData[mm + 0] = aTemp.lo;
                aData[mm + 1] = aTemp.hi;
            }
        } else {
            for (int mm = 0; mm < MM; mm++) {
                aData[mm] = as_short8(intel_subgroup_block_read_u16_m8k16(A, K * sizeof(ushort), M, K * sizeof(ushort), (int2)(k, m + mm * tM)));
            }
        }

        int8    bData[NN];
        for (int nn = 0; nn < NN; nn++) {
            bData[nn] = as_int8(intel_subgroup_block_read_transform_u16_k16(B, N * sizeof(ushort), K, N * sizeof(ushort), (int2)(n + nn * tN, k)));
        }

        for (int mm = 0; mm < MM; mm++) {
            for (int nn = 0; nn < NN; nn++) {
                sum[mm][nn] = mat_mul_sg16(aData[mm], bData[nn], sum[mm][nn]);
            }
        }
    }

    for (int mm = 0; mm < MM; mm++) {
        for (int nn = 0; nn < NN; nn++) {
            intel_subgroup_block_write_u32_m8k16(C, N * sizeof(float), M, N * sizeof(float), (int2)(n + nn * tN, m + mm * tM), as_uint8(sum[mm][nn]));
        }
    }
}

__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16, 1, 1)))
kernel void MM_KERNEL_NAME(bfloat16_dpas_blockread_vnni_tiled, 8, 16, MM, NN)(global float* C, global ushort* A, global ushort* B, int K)
{
    const int tM = 8;
    const int tN = 16;
    const int M = get_global_size(1) * tM * MM;
    const int N = get_global_size(0) * NN;
    const int m = get_group_id(1) * tM * MM;
    const int n = get_group_id(0) * tN * NN;

    float8 sum[MM][NN];
    for (int mm = 0; mm < MM; mm++) {
        for (int nn = 0; nn < NN; nn++) {
            sum[mm][nn] = 0;
        }
    }

    for (int k = 0; k < K; k += tK) {
        short8  aData[MM];
        if (MM % 2 == 0) {
            for (int mm = 0; mm < MM; mm += 2) {
                short16 aTemp = as_short16(intel_subgroup_block_read_u16_m16k16(A, K * sizeof(ushort), M, K * sizeof(ushort), (int2)(k, m + mm * tM)));
                aData[mm + 0] = aTemp.lo;
                aData[mm + 1] = aTemp.hi;
            }
        } else {
            for (int mm = 0; mm < MM; mm++) {
                aData[mm] = as_short8(intel_subgroup_block_read_u16_m8k16(A, K * sizeof(ushort), M, K * sizeof(ushort), (int2)(k, m + mm * tM)));
            }
        }

        int8    bData[NN];
        for (int nn = 0; nn < NN; nn++) {
            bData[nn] = as_int8(intel_subgroup_block_read_u32_m8k16(B, N * sizeof(uint), K, N * sizeof(uint), (int2)(n + nn * tN, k / 2)));
        }

        for (int mm = 0; mm < MM; mm++) {
            for (int nn = 0; nn < NN; nn++) {
                sum[mm][nn] = mat_mul_sg16(aData[mm], bData[nn], sum[mm][nn]);
            }
        }
    }

    for (int mm = 0; mm < MM; mm++) {
        for (int nn = 0; nn < NN; nn++) {
            intel_subgroup_block_write_u32_m8k16(C, N * sizeof(float), M, N * sizeof(float), (int2)(n + nn * tN, m + mm * tM), as_uint8(sum[mm][nn]));
        }
    }
}

#endif // cl_intel_subgroup_extended_block_read
