/*
// Copyright (c) 2024 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <popl/popl.hpp>

#include <CL/opencl.hpp>

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <sstream>
#include <string>
#include <random>
#include <vector>

#include "util.hpp"

using test_clock = std::chrono::high_resolution_clock;

int B = 8;      // Batch Size
int T = 1024;   // Sequence Length
int NH = 12;    // Number of Heads
int D = 64;     // Head Dimension (AKA Head Size)

int C = 0;      // Channels

bool causal = false;

// Note:
//  C  = NH * D
//  C  = D * NH
//  D  = C / NH
//  NH = C / D

// From:
// https://github.com/Dao-AILab/flash-attention/blob/main/benchmarks/benchmark_flash_attention.py
// B, T = bs_seqlen_vals = [(32, 512), (16, 1024), (8, 2048), (4, 4096), (2, 8192), (1, 16384)]
// D = headdim_vals = [64, 128]
// C = dim = 2048
// NH = nheads = dim // headdim = [32, 16]

bool zeroData = false;
bool identityData = false;
bool fixedData = false;
bool validate = false;
bool emulate = false;
bool wallclock = false;
bool skipinit = false;
bool roundRobin = false;
int testIterations = 16;
float threshold = 0.01f;

static size_t findMinSubGroupSize(cl::Device& device)
{
    auto s = device.getInfo<CL_DEVICE_SUB_GROUP_SIZES_INTEL>();
    auto it = std::min_element(std::begin(s), std::end(s));
    if (it != std::end(s)) {
        return *it;
    }
    return 0;
}

static void setRoundRobin(cl::Kernel& kernel)
{
    constexpr cl_kernel_exec_info CL_KERNEL_EXEC_INFO_THREAD_ARBITRATION_POLICY_INTEL = 0x10025;
    constexpr cl_uint CL_KERNEL_EXEC_INFO_THREAD_ARBITRATION_POLICY_ROUND_ROBIN_INTEL = 0x10023;
    const cl_uint policy = CL_KERNEL_EXEC_INFO_THREAD_ARBITRATION_POLICY_ROUND_ROBIN_INTEL;
    clSetKernelExecInfo(
        kernel(),
        CL_KERNEL_EXEC_INFO_THREAD_ARBITRATION_POLICY_INTEL,
        sizeof(policy),
        &policy);
}

template <typename TT>
static void fill_input(std::vector<TT>& M, size_t count)
{
    if (zeroData) {
        std::generate(std::begin(M), std::end(M), [&]{ return 0.0f; });
    }
    else if (identityData) {
        std::generate(std::begin(M), std::end(M), [&]{ return 1.0f; });
    } else if (fixedData) {
        for (size_t i = 0; i < count; i++) {
            M[i] = static_cast<float>(i);
        }
    } else {
        //std::random_device dev;
        //std::mt19937 rng(dev());
        std::default_random_engine rng;
        std::uniform_real_distribution<float> dist(0.0, 1.0);
        std::generate(std::begin(M), std::end(M), [&]{ return dist(rng); });
    }
}

// ----------------------------------------------------------------------------
// CPU code reference - adapted from llm.c
// This implements the naive three-pass algorithm.
template <typename TT>
void attention_forward_3p_cpu(
    std::vector<TT>& out,
    std::vector<TT>& preatt,
    std::vector<TT>& att,
    const std::vector<TT>& q_base,
    const std::vector<TT>& k_base,
    const std::vector<TT>& v_base)
{
    // q, k, v are (B, NH, T, D)
    // preatt, att are (B, NH, T, T)
    // output is (B, NH, T, D)
    TT scale = (TT)(1.0 / std::sqrt(D));

    for (int b = 0; b < B; b++) {
        for (int nh = 0; nh < NH; nh++) {
            for (int to = 0; to < T; to++) {    // outer t, row in the attention matrix
                const int tCheck = causal ? to + 1 : T;

                // pass 1: calculate query dot key
                const TT* q = q_base.data() + b * NH * T * D + nh * T * D + to * D;
                TT* preatt_bth = preatt.data() + b * NH * T * T + nh * T * T + to * T;
                for (int ti = 0; ti < T; ti++) {// inner t, col in the attention matrix
                    if (causal && ti > to) {
                        preatt_bth[ti] = -INFINITY; // causal mask
                    }
                    else {
                        const TT* k = k_base.data() + b * NH * T * D + nh * T * D + ti * D;

                        TT val = 0.0;
                        for (int d = 0; d < D; d++) {
                            val += q[d] * k[d];
                        }
                        val *= scale;
                        preatt_bth[ti] = val;
                    }
                }

                // pass 2: softmax
                TT* att_bth = att.data() + b * NH * T * T + nh * T * T + to * T;
                TT maxval = -10000.0; // TODO something better
                for (int ti = 0; ti < tCheck; ti++) {
                    TT val = preatt_bth[ti];
                    maxval = std::fmax(val, maxval);
                }

                TT expsum = 0.0;
                for (int ti = 0; ti < tCheck; ti++) {
                    TT expv = std::exp(preatt_bth[ti] - maxval);
                    expsum += expv;
                    att_bth[ti] = expv;
                }
                TT expsum_inv = (TT)(expsum == 0.0 ? 0.0 : 1.0 / expsum);

                for (int ti = 0; ti < T; ti++) {
                    if (ti < tCheck) {
                        att_bth[ti] *= expsum_inv;
                    } else {
                        // causal attention mask. not strictly necessary to set to zero here
                        // only doing this explicitly for debugging and checking to PyTorch
                        att_bth[ti] = 0.0f;
                    }
                }

                // pass 3: accumulate weighted values into the output of attention
                TT* out_bth = out.data() + b * NH * T * D + nh * T * D + to * D;
                for (int d = 0; d < D; d++) {
                    out_bth[d] = 0.0;
                }
                for (int ti = 0; ti < tCheck; ti++) {
                    const TT* v = v_base.data() + b * NH * T * D + nh * T * D + ti * D;
                    TT att_btht2 = att_bth[ti];
                    for (int d = 0; d < D; d++) {
                        out_bth[d] += att_btht2 * v[d];
                    }
                }
            }
        }
    }
}

// This implements a two-pass algorithm.
template <typename TT>
void attention_forward_2p_cpu(
    std::vector<TT>& out,
    std::vector<TT>& preatt,
    std::vector<TT>& att,
    const std::vector<TT>& q_base,
    const std::vector<TT>& k_base,
    const std::vector<TT>& v_base)
{
    // q, k, v are (B, NH, T, D)
    // preatt, att are (B, NH, T, T)
    // output is (B, NH, T, D)
    TT scale = (TT)(1.0 / std::sqrt(D));

    for (int b = 0; b < B; b++) {
        for (int nh = 0; nh < NH; nh++) {
            for (int to = 0; to < T; to++) {
                // pass 1: calculate query dot key / max / surrogate
                const TT* q = q_base.data() + b * NH * T * D + nh * T * D + to * D;
                TT* p = preatt.data() + b * NH * T * T + nh * T * T + to * T;
                TT mi = -INFINITY;
                TT di = 0.0;
                for (int ti = 0; ti < T; ti++) {
                    const TT* k = k_base.data() + b * NH * T * D + nh * T * D + ti * D;
                    TT xi = 0.0;
                    if (causal && ti > to) {
                        xi = -INFINITY; // causal mask
                    }
                    else {
                        for (int d = 0; d < D; d++) {
                            xi += q[d] * k[d];
                        }
                        xi *= scale;

                        TT mim1 = mi;
                        mi = fmax(mim1, xi);

                        TT smxi = std::exp(xi - mi);
                        TT exp_dmim1mi = std::exp(mim1 - mi);
                        di = di * exp_dmim1mi + smxi;
                    }

                    p[ti] = xi;
                }

                // pass 2: softmax / output
                TT* o = out.data() + b * NH * T * D + nh * T * D + to * D;
                TT* a = att.data() + b * NH * T * T + nh * T * T + to * T;
                for (int d = 0; d < D; d++) {
                    o[d] = 0.0;
                }
                for (int ti = 0; ti < T; ti++) {
                    const TT* v = v_base.data() + b * NH * T * D + nh * T * D + ti * D;
                    if (causal && ti > to) {
                        a[ti] = 0.0f;
                    }
                    else {
                        TT att = std::exp(p[ti] - mi) / di;
                        a[ti] = att;
                        for (int d = 0; d < D; d++) {
                            o[d] += att * v[d];
                        }
                    }
                }
            }
        }
    }
}

// This implements the one-pass flash attention algorithm without any blocking.
template <typename TT>
void attention_forward_1p_cpu(
    std::vector<TT>& out,
    const std::vector<TT>& q_base,
    const std::vector<TT>& k_base,
    const std::vector<TT>& v_base)
{
    // q, k, v are (B, NH, T, D)
    // output is (B, NH, T, D)
    const TT scale = (TT)(1.0 / std::sqrt(D));

    for (int b = 0; b < B; b++) {
        for (int nh = 0; nh < NH; nh++) {
            for (int to = 0; to < T; to++) {
                // Note: we multiply the initial output value by zero, so we do not
                // need to initialize it.

                TT* o = out.data() + b * NH * T * D + nh * T * D + to * D;

                const TT* q = q_base.data() + b * NH * T * D + nh * T * D + to * D;
                TT mi = -INFINITY;
                TT di = 0.0;

                for (int ti = 0; ti < T; ti++) {
                    const TT* k = k_base.data() + b * NH * T * D + nh * T * D + ti * D;
                    const TT* v = v_base.data() + b * NH * T * D + nh * T * D + ti * D;

                    // Compute xi = QK^T
                    TT xi = 0.0;
                    if (causal && to < ti) {
                        xi = -INFINITY;
                    }
                    else {
                        for (int d = 0; d < D; d++) {
                            xi += q[d] * k[d];
                        }
                        xi *= scale;
                    }

                    // Update the running maximum
                    TT mim1 = mi;
                    mi = std::fmax(mim1, xi);

                    // softmax(xi)
                    TT smxi = std::exp(xi - mi);

                    // Update di
                    TT alpha = std::exp(mim1 - mi);
                    di = di * alpha + smxi;

                    // Update the un-scaled output from softmax(xi) and V
                    for (int d = 0; d < D; d++) {
                        o[d] = o[d] * alpha + smxi * v[d];
                    }
                }

                // Epilog scaling (flash attention 2)
                for (int d = 0; d < D; d++) {
                    o[d] = o[d] / di;
                }
            }
        }
    }
}

// This implements the one-pass flash attention algorithm with column blocking.
template <typename TT>
void attention_forward_1p_colblock_cpu(
    std::vector<TT>& out,
    const std::vector<TT>& q_base,
    const std::vector<TT>& k_base,
    const std::vector<TT>& v_base)
{
    const int BC = 32;

    // q, k, v are (B, NH, T, D)
    // output is (B, NH, T, D)
    const TT scale = (TT)(1.0 / std::sqrt(D));

    std::vector<TT> xi(BC);     // BC columns
    std::vector<TT> smxi(BC);   // BC columns

    // This currently only works if T is evenly divisible by BR and BC
    if (T % BC != 0) {
        printf("currently requires sequence length to evenly divide block columns!\n");
        return;
    }

    for (int b = 0; b < B; b++) {
        for (int nh = 0; nh < NH; nh++) {
            for (int to = 0; to < T; to++) {
                // Note: we multiply the initial output value by zero, so we do not
                // need to initialize it.

                TT* o = out.data() + b * NH * T * D + nh * T * D + to * D;

                const TT* q = q_base.data() + b * NH * T * D + nh * T * D + to * D;
                TT mi = -INFINITY;
                TT di = 0.0;

                for (int ti = 0; ti < T; ti+=BC) {
                    const TT* k = k_base.data() + b * NH * T * D + nh * T * D + ti * D;
                    const TT* v = v_base.data() + b * NH * T * D + nh * T * D + ti * D;

                    // Compute xi = QK^T and rowmax
                    TT rowmax = -INFINITY;
                    for (int c = 0; c < BC; c++) {
                        TT val = 0.0;
                        if (causal && to < ti + c) {
                            val = -INFINITY;
                        }
                        else {
                            for (int d = 0; d < D; d++) {
                                val += q[d] * k[c * D + d];
                            }
                            val *= scale;
                        }
                        xi[c] = val;
                        rowmax = std::fmax(rowmax, xi[c]);
                    }

                    // Update the running maximum from the row maximum
                    TT mim1 = mi;
                    mi = std::fmax(mim1, rowmax);

                    // softmax and rowsum
                    TT rowsum = 0.0;
                    for (int c = 0; c < BC; c++) {
                        smxi[c] = std::exp(xi[c] - mi);
                        rowsum += smxi[c];
                    }

                    // Update di
                    TT alpha = std::exp(mim1 - mi);
                    di = di * alpha + rowsum;

                    // Update the un-scaled output from softmax(xi) and V
                    for (int d = 0; d < D; d++) {
                        TT val = 0.0;
                        for (int c = 0; c < BC; c++) {
                            val += smxi[c] * v[c * D + d];
                        }
                        o[d] = o[d] * alpha + val;
                    }
                }

                // Epilog scaling (flash attention 2)
                for (int d = 0; d < D; d++) {
                    o[d] = o[d] / di;
                }
            }
        }
    }
}

// This implements the one-pass flash attention algorithm with row and column blocking.
template <typename TT>
void attention_forward_1p_rcblock_cpu(
    std::vector<TT>& out,
    const std::vector<TT>& q_base,
    const std::vector<TT>& k_base,
    const std::vector<TT>& v_base)
{
    const int BR = 32;
    const int BC = 32;

    // q, k, v are (B, NH, T, D)
    // output is (B, NH, T, D)
    const TT scale = (TT)(1.0 / std::sqrt(D));

    // All of these caches can go into registers or SLM
    std::vector<TT> Qcache(BR * D); // BR rows and D columns
    std::vector<TT> Kcache(BC * D); // BC rows and D columns
    std::vector<TT> Vcache(BC * D); // BC rows and D columns
    std::vector<TT> Ocache(BR * D); // BR rows and D columns

    std::vector<TT> xi(BR * BC);    // BR rows and BC columns
    std::vector<TT> rowmax(BR);     // BR x local maximum
    std::vector<TT> rowsum(BR);     // BR x local sum
    std::vector<TT> mim1(BR);       // BR x previous maximum
    std::vector<TT> mi(BR);         // BR x running maximum
    std::vector<TT> alpha(BR);      // BR x exponential
    std::vector<TT> di(BR);         // BR x surrogate

    // This currently only works if T is evenly divisible by BR and BC
    if (T % BR != 0 || T % BC != 0) {
        printf("currently requires sequence length to evenly divide block rows and columns!\n");
        return;
    }

    for (int b = 0; b < B; b++) {
        for (int nh = 0; nh < NH; nh++) {
            for (int to = 0; to < T; to+=BR) {
                // Note: we multiply the initial output value by zero, so we do not
                // need to initialize it.

                // Load Q data into the Qcache
                {
                    const TT* q = q_base.data() + b * NH * T * D + nh * T * D + to * D;
                    for (int r = 0; r < BR; r++) {
                        for (int d = 0; d < D; d++) {
                            Qcache[r * D + d] = q[r * D + d];
                        }
                    }
                }

                // Initialize the maximum and surrogate
                for (int r = 0; r < BR; r++) {
                    mi[r] = -INFINITY;
                    di[r] = 0.0;
                }

                for (int ti = 0; ti < T; ti+=BC) {
                    // Load K and V data into the Kcache and Vcache
                    {
                        const TT* k = k_base.data() + b * NH * T * D + nh * T * D + ti * D;
                        const TT* v = v_base.data() + b * NH * T * D + nh * T * D + ti * D;
                        for (int c = 0; c < BC; c++) {
                            for (int d = 0; d < D; d++) {
                                Kcache[c * D + d] = k[c * D + d];
                                Vcache[c * D + d] = v[c * D + d];
                            }
                        }
                    }

                    // Compute xi = QK^T and rowmax
                    for (int r = 0; r < BR; r++) {
                        rowmax[r] = -INFINITY;
                        for (int c = 0; c < BC; c++) {
                            TT val = 0.0;
                            if (causal && to + r < ti + c) {
                                val = -INFINITY;
                            }
                            else {
                                for (int d = 0; d < D; d++) {
                                    val += Qcache[r * D + d] * Kcache[c * D + d];
                                }
                                val *= scale;
                            }
                            xi[r * BC + c] = val;
                            rowmax[r] = std::fmax(rowmax[r], val);
                        }
                    }

                    // Update the running maximum from the row maximum
                    for (int r = 0; r < BR; r++) {
                        mim1[r] = mi[r];
                        mi[r] = std::fmax(mim1[r], rowmax[r]);
                    }

                    // softmax and rowsum
                    for (int r = 0; r < BR; r++) {
                        rowsum[r] = 0.0;
                        for (int c = 0; c < BC; c++) {
                            TT val = xi[r * BC + c];
                            val = std::exp(val - mi[r]);
                            xi[r * BC + c] = val;
                            rowsum[r] += val;
                        }
                    }

                    // Update di
                    for (int r = 0; r < BR; r++) {
                        alpha[r] = std::exp(mim1[r] - mi[r]);
                        di[r] = di[r] * alpha[r] + rowsum[r];
                    }

                    // Update the un-scaled output from softmax(xi) and V
                    for (int r = 0; r < BR; r++) {
                        for (int d = 0; d < D; d++) {
                            TT val = 0.0;
                            for (int c = 0; c < BC; c++) {
                                val += xi[r * BC + c] * Vcache[c * D + d];
                            }
                            Ocache[r * D + d] = Ocache[r * D + d] * alpha[r] + val;
                        }
                    }
                }

                // Epilog scaling and store
                {
                    TT* o = out.data() + b * NH * T * D + nh * T * D + to * D;
                    for (int r = 0; r < BR; r++) {
                        for (int d = 0; d < D; d++) {
                            o[r * D + d] = Ocache[r * D + d] / di[r];
                        }
                    }
                }
            }
        }
    }
}

template <typename TT>
void check_results(
    const std::vector<TT>& check,
    const std::vector<TT>& reference,
    TT tolerance = 1e-5)
{
    if (check.size() != reference.size()) {
        printf("Size mismatch?  %zu vs %zu\n", check.size(), reference.size());
    }
    for (size_t i = 0; i < check.size(); i++) {
        // if (i < 5) {
        //     printf("%f %f\n", reference[i], check[i]);
        // }
        if (fabs(reference[i] - check[i]) > tolerance) {
            printf("Mismatch at %zu: %f vs %f\n", i, reference[i], check[i]);
            return;
        }
    }
}

template<class TT>
TT ceil_div(TT dividend, TT divisor) {
    return (dividend + divisor - 1) / divisor;
}

static float hw_time(cl::Event& event)
{
    auto ns = event.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
              event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    return ns / 1e9f;
}

// This implements the naive three pass algorithm.
// It has separate kernels for each of the three steps.
static void naive_3p_attention_forward(
    cl::Context& context, cl::Program& program, cl::CommandQueue& queue,
    cl::Buffer& out, cl::Buffer& preatt, cl::Buffer& att,
    cl::Buffer& q, cl::Buffer& k, cl::Buffer& v,
    size_t wgSize,
    const std::vector<float>& out_ref,
    const std::vector<float>& preatt_ref,
    const std::vector<float>& att_ref)
{
    printf("%80s: ", __FUNCTION__); fflush(stdout);

    cl::Kernel naive_query_key{program, "naive_3p_query_key"};
    cl::Kernel naive_softmax{program, "naive_3p_softmax"};
    cl::Kernel naive_value{program, "naive_3p_value"};
    if (naive_query_key() == nullptr ||
        naive_softmax() == nullptr ||
        naive_value() == nullptr) {
        printf("unsupported.\n");
    } else {
        size_t native_query_key_gws = B * NH * T * T;
        naive_query_key.setArg(0, preatt);
        naive_query_key.setArg(1, q);
        naive_query_key.setArg(2, k);

        size_t naive_softmax_gws = B * T * NH;
        naive_softmax.setArg(0, att);
        naive_softmax.setArg(1, preatt);

        size_t naive_value_gws = B * T * NH;
        naive_value.setArg(0, out);
        naive_value.setArg(1, att);
        naive_value.setArg(2, v);

        if (!skipinit) {
            queue.enqueueFillBuffer(out, 0, 0, out_ref.size() * sizeof(out_ref[0]));
        }

        float best = 999.0f;
        for (int test = 0; test < testIterations; test++) {
            auto start = test_clock::now();
            queue.enqueueNDRangeKernel(naive_query_key, cl::NullRange,
                cl::NDRange{native_query_key_gws});
            queue.enqueueNDRangeKernel(naive_softmax, cl::NullRange,
                cl::NDRange{naive_softmax_gws});
            queue.enqueueNDRangeKernel(naive_value, cl::NullRange,
                cl::NDRange{naive_value_gws});
            queue.finish();
            auto end = test_clock::now();
            std::chrono::duration<float> sw_time = end - start;
            auto elapsed = sw_time.count();
            best = std::min(best, elapsed);
        }
        printf("Best in %f seconds\n", best);

        if (validate) {
            printf("Checking results: preatt... "); fflush(stdout);
            std::vector<float> preatt_check(preatt_ref.size());
            queue.enqueueReadBuffer(preatt, CL_TRUE, 0, preatt_check.size() * sizeof(preatt_check[0]), preatt_check.data());
            check_results(preatt_check, preatt_ref);

            printf("Checking results: att... "); fflush(stdout);
            std::vector<float> att_check(att_ref.size());
            queue.enqueueReadBuffer(att, CL_TRUE, 0, att_check.size() * sizeof(att_check[0]), att_check.data());
            check_results(preatt_check, preatt_ref);

            printf("Checking results: out... "); fflush(stdout);
            std::vector<float> out_check(out_ref.size());
            queue.enqueueReadBuffer(out, CL_TRUE, 0, out_check.size() * sizeof(out_check[0]), out_check.data());
            check_results(out_check, out_ref);

            printf(" done!\n");
        }
    }
}

// This is a port of the minimal flash attention, see:
// https://github.com/tspeterkim/flash-attention-minimal
static void flash_attention_minimal_forward(
    cl::Context& context, cl::Program& program, cl::CommandQueue& queue,
    cl::Buffer& out,
    cl::Buffer& q, cl::Buffer& k, cl::Buffer& v,
    size_t wgSize,
    const std::vector<float>& out_ref)
{
    printf("%80s: ", __FUNCTION__); fflush(stdout);

    cl::Kernel flash_attention{program, "flash_attention_minimal"};
    if (flash_attention() == nullptr) {
        printf("unsupported.\n");
    } else {
        // Start of the port is here:
        const int Bc = 32;
        const int Br = 32;

        const int Tc = (int)std::ceil((float) T / Bc);
        const int Tr = (int)std::ceil((float) T / Br);
        const float softmax_scale = 1.0f / (float)std::sqrt(D);

        cl::Buffer l{context, CL_MEM_READ_WRITE, B * NH * T * sizeof(float)};
        cl::Buffer m{context, CL_MEM_READ_WRITE, B * NH * T * sizeof(float)};
        queue.enqueueFillBuffer(l,         0, 0, B * NH * T * sizeof(float));
        queue.enqueueFillBuffer(m, -10000.0f, 0, B * NH * T * sizeof(float));

        const int Q_LMSize  = Br * D * sizeof(float);
        const int KV_LMSize = Bc * D * sizeof(float);
        const int S_LMSize  = Br * Bc * sizeof(float);
        const int requiredLocalMemSize =
            Q_LMSize + KV_LMSize * 2 + S_LMSize;
        cl::Device device = queue.getInfo<CL_QUEUE_DEVICE>();
        cl_ulong maxLocalMemSize = device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();
        if (requiredLocalMemSize > maxLocalMemSize) {
            printf("Device max local memory size: %" PRIu64 ", required local memory size: %d\n",
                maxLocalMemSize, requiredLocalMemSize);
        }

        // TODO: Should this be Bc or Br?
        // Doesn't matter in practice right now, but could in the future.
        cl::NDRange flash_attention_gws(B * Bc, NH);  // batch_size x num_heads
        cl::NDRange flash_attention_lws(Bc);  // Bc threads per block
        flash_attention.setArg(0, q);
        flash_attention.setArg(1, k);
        flash_attention.setArg(2, v);
        flash_attention.setArg(3, Bc);
        flash_attention.setArg(4, Br);
        flash_attention.setArg(5, softmax_scale);
        flash_attention.setArg(6, l);
        flash_attention.setArg(7, m);
        flash_attention.setArg(8, out);
        flash_attention.setArg(9, cl::Local(Q_LMSize));
        flash_attention.setArg(10, cl::Local(KV_LMSize));
        flash_attention.setArg(11, cl::Local(KV_LMSize));
        flash_attention.setArg(12, cl::Local(S_LMSize));

        if (!skipinit) {
            queue.enqueueFillBuffer(out, 0, 0, out_ref.size() * sizeof(out_ref[0]));
        }

        float best = 999.0f;
        for (int test = 0; test < testIterations; test++) {
            auto start = test_clock::now();
            queue.enqueueNDRangeKernel(flash_attention, cl::NullRange,
                cl::NDRange{flash_attention_gws}, cl::NDRange{flash_attention_lws});
            queue.finish();
            auto end = test_clock::now();
            std::chrono::duration<float> sw_time = end - start;
            auto elapsed = sw_time.count();
            best = std::min(best, elapsed);
        }
        printf("Best in %f seconds\n", best);

        if (validate) {
            printf("Checking results: out... "); fflush(stdout);
            std::vector<float> out_check(out_ref.size());
            queue.enqueueReadBuffer(out, CL_TRUE, 0, out_check.size() * sizeof(out_check[0]), out_check.data());
            check_results(out_check, out_ref);
            printf(" done!\n");
        }
    }
}

static void flash_attention_forward(
    const std::string& kernelName,
    cl::Context& context, cl::Program& program, cl::CommandQueue& queue,
    cl::Buffer& out,
    cl::Buffer& q, cl::Buffer& k, cl::Buffer& v,
    size_t wgSize,
    const std::vector<float>& out_ref)
{
    std::string label(__FUNCTION__);
    label += "(";
    label += kernelName;
    label += ")";

    printf("%80s: ", label.c_str()); fflush(stdout);

    cl::Kernel flash_attention{program, kernelName.c_str()};
    if (flash_attention() == nullptr) {
        printf("unsupported.\n");
    } else {
        const float softmax_scale = 1.0f / (float)std::sqrt(D);

        cl::NDRange flash_attention_gws(T, NH, B);
        flash_attention.setArg(0, q);
        flash_attention.setArg(1, k);
        flash_attention.setArg(2, v);
        flash_attention.setArg(3, out);
        flash_attention.setArg(4, softmax_scale);

        if (!skipinit) {
            queue.enqueueFillBuffer(out, 0, 0, out_ref.size() * sizeof(out_ref[0]));
        }

        float best = 999.0f;
        for (int test = 0; test < testIterations; test++) {
            auto start = test_clock::now();
            queue.enqueueNDRangeKernel(flash_attention, cl::NullRange, flash_attention_gws);
            queue.finish();
            auto end = test_clock::now();
            std::chrono::duration<float> sw_time = end - start;
            auto elapsed = sw_time.count();
            best = std::min(best, elapsed);
        }
        printf("Best in %f seconds\n", best);

        if (validate) {
            printf("Checking results: out... "); fflush(stdout);
            std::vector<float> out_check(out_ref.size());
            queue.enqueueReadBuffer(out, CL_TRUE, 0, out_check.size() * sizeof(out_check[0]), out_check.data());
            check_results(out_check, out_ref);
            printf(" done!\n");
        }
    }
}

static void flash_attention_forward_wg(
    const std::string& kernelName,
    cl::Context& context, cl::Program& program, cl::CommandQueue& queue,
    cl::Buffer& out,
    cl::Buffer& q, cl::Buffer& k, cl::Buffer& v,
    size_t wgSize,
    const std::vector<float>& out_ref)
{
    std::string label(__FUNCTION__);
    label += "(";
    label += kernelName;
    label += ")";

    printf("%80s: ", label.c_str()); fflush(stdout);

    cl::Kernel flash_attention{program, kernelName.c_str()};
    if (flash_attention() == nullptr) {
        printf("unsupported.\n");
    } else {
        const float softmax_scale = 1.0f / (float)std::sqrt(D);

        cl::NDRange flash_attention_gws(T * D, NH, B);
        cl::NDRange flash_attention_lws(D, 1, 1);
        flash_attention.setArg(0, q);
        flash_attention.setArg(1, k);
        flash_attention.setArg(2, v);
        flash_attention.setArg(3, out);
        flash_attention.setArg(4, softmax_scale);

        if (!skipinit) {
            queue.enqueueFillBuffer(out, 0, 0, out_ref.size() * sizeof(out_ref[0]));
        }

        float best = 999.0f;
        for (int test = 0; test < testIterations; test++) {
            auto start = test_clock::now();
            queue.enqueueNDRangeKernel(flash_attention, cl::NullRange, flash_attention_gws, flash_attention_lws);
            queue.finish();
            auto end = test_clock::now();
            std::chrono::duration<float> sw_time = end - start;
            auto elapsed = sw_time.count();
            best = std::min(best, elapsed);
        }
        printf("Best in %f seconds\n", best);

        if (validate) {
            printf("Checking results: out... "); fflush(stdout);
            std::vector<float> out_check(out_ref.size());
            queue.enqueueReadBuffer(out, CL_TRUE, 0, out_check.size() * sizeof(out_check[0]), out_check.data());
            check_results(out_check, out_ref);
            printf(" done!\n");
        }
    }
}

int main(int argc, char** argv)
{
    int platformIndex = 0;
    int deviceIndex = 0;

    std::string fileName("attention_kernels.cl");
    std::string buildOptions;
    size_t wgSize = 32;

    size_t mask = ~0;

    {
        popl::OptionParser op("Supported Options");
        op.add<popl::Value<int>>("p", "platform", "Platform Index", platformIndex, &platformIndex);
        op.add<popl::Value<int>>("d", "device", "Device Index", deviceIndex, &deviceIndex);
        op.add<popl::Value<int>>("B", "batchsize", "Batch Size", B, &B);
        op.add<popl::Value<int>>("T", "seqlen", "Sequence Length", T, &T);
        op.add<popl::Value<int>>("H", "numheads", "Number of Heads", NH, &NH);
        op.add<popl::Value<int>>("D", "headdim", "Head Dimension", D, &D);
        op.add<popl::Switch>("", "causal", "Compute Causal Attention", &causal);
        op.add<popl::Value<std::string>>("", "file", "Kernel File Name", fileName, &fileName);
        op.add<popl::Value<std::string>>("", "options", "Program Build Options", buildOptions, &buildOptions);
        op.add<popl::Value<size_t>>("w", "wgsize", "Work-Group Size", wgSize, &wgSize);
        op.add<popl::Value<int>>("i", "iterations", "Test Iterations", testIterations, &testIterations);
        op.add<popl::Switch>("", "validate", "Validate Results", &validate);
        op.add<popl::Switch>("", "zero", "Use Zero Data", &zeroData);
        op.add<popl::Switch>("", "identity", "Use Identity Data", &identityData);
        op.add<popl::Switch>("", "fixed", "Use Fixed Data", &fixedData);
        op.add<popl::Switch>("", "emulate", "Unconditionally Emulate dpas", &emulate);
        op.add<popl::Switch>("", "wallclock", "Measure Wallclock Time", &wallclock);
        op.add<popl::Switch>("", "skipinit", "Do Not Initialize Buffers", &skipinit);
        op.add<popl::Switch>("", "roundrobin", "Use Round Robin Scheduling", &roundRobin);
        op.add<popl::Value<float>>("", "threshold", "Local Error Threshold", threshold, &threshold);
        op.add<popl::Value<size_t>, popl::Attribute::advanced>("", "mask", "Test Mask", mask, &mask);
        bool printUsage = false;
        try {
            op.parse(argc, argv);
        } catch (std::exception& e) {
            fprintf(stderr, "Error: %s\n\n", e.what());
            printUsage = true;
        }

        if (printUsage || !op.unknown_options().empty() || !op.non_option_args().empty()) {
            fprintf(stderr,
                "Usage: flashattention [options]\n"
                "%s", op.help().c_str());
            return -1;
        }
    }

    // Derived values:
    C = NH * D;

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platformIndex >= platforms.size()) {
        printf("Requested platform index is %d, but only %zu platforms were found.\n",
            platformIndex, platforms.size());
        return -1;
    }

    printf("Running on platform: %s\n",
        platforms[platformIndex].getInfo<CL_PLATFORM_NAME>().c_str() );

    std::vector<cl::Device> devices;
    platforms[platformIndex].getDevices(CL_DEVICE_TYPE_ALL, &devices);
    if (deviceIndex >= devices.size()) {
        printf("Requested device index is %d, but only %zu devices were found.\n",
            deviceIndex, devices.size());
    }

    cl::Device& device = devices[deviceIndex];
    printf("Running on device: %s (%uCUs, %uMHz)\n",
        device.getInfo<CL_DEVICE_NAME>().c_str(),
        device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>(),
        device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>());
    printf("Running on drivers: %s\n",
        device.getInfo<CL_DRIVER_VERSION>().c_str());

    auto minSubGroupSize = findMinSubGroupSize(device);

    bool has_simd8 = minSubGroupSize == 8;
    bool emulate_tN8 = true;
    bool emulate_tN16 = true;
    if (!emulate && checkDeviceForExtension(device, "cl_intel_subgroup_matrix_multiply_accumulate")) {
        printf("Found support for cl_intel_subgroup_matrix_multiply_accumulate, min sub-group size is: %zu\n", minSubGroupSize);
        switch(minSubGroupSize) {
            case 8: emulate_tN8 = false; break;
            case 16: emulate_tN16 = false; break;
            default: break;
        }
    }

    buildOptions += " -DHAS_SIMD8=" + std::to_string(has_simd8);
    buildOptions += " -DEMULATE_tN8=" + std::to_string(emulate_tN8);
    buildOptions += " -DEMULATE_tN16=" + std::to_string(emulate_tN16);
    buildOptions += " -DEMULATE_tN16=" + std::to_string(emulate_tN16);
    buildOptions += " -DB=" + std::to_string(B);
    buildOptions += " -DT=" + std::to_string(T);
    buildOptions += " -DNH=" + std::to_string(NH);
    buildOptions += " -DC=" + std::to_string(C);
    buildOptions += " -DCAUSAL=" + std::to_string(causal ? 1 : 0);

    printf("Config:\n");
    printf("\tBatch Size B: %d\n", B);
    printf("\tSequence Length T: %d\n", T);
    printf("\tNumber of Heads NH: %d\n", NH);
    printf("\tHead Dimension D: %d\n", D);
    printf("\tChannels C: %d * %d = %d\n", NH, D, C);
    printf("\tTest Iterations: %d\n", testIterations);
    printf("\tWork-group Size: %zu\n", wgSize);
    printf("\tValidating data?: %s\n", validate ? "true" : "false");
    printf("\tFixed data?: %s\n", fixedData ? "true" : "false");
    printf("\tWallclock time?: %s\n", wallclock ? "true" : "false");
    printf("\tEmulate dpas for tN=8?: %s\n", emulate_tN8 ? "true" : "false");
    printf("\tEmulate dpas for tN=16?: %s\n", emulate_tN16 ? "true" : "false");

    cl::Context context{device};
    cl::CommandQueue queue{context, device, CL_QUEUE_PROFILING_ENABLE};

    printf("Reading program source from file: %s\n", fileName.c_str() );
    std::string kernelString = readStringFromFile(fileName.c_str());

    printf("Building program with build options: %s\n",
        buildOptions.empty() ? "(none)" : buildOptions.c_str() );
    cl::Program program{ context, kernelString };
    program.build(buildOptions.c_str());
    for( auto& device : program.getInfo<CL_PROGRAM_DEVICES>() )
    {
        printf("Program build log for device %s:\n",
            device.getInfo<CL_DEVICE_NAME>().c_str() );
        printf("%s\n",
            program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device).c_str() );
    }

    std::vector<float> out_vec   (B * T * C);
    std::vector<float> preatt_vec(B * NH * T * T);
    std::vector<float> att_vec   (B * NH * T * T);

    std::vector<float>  q_vec(B * T * C);
    std::vector<float>  k_vec(B * T * C);
    std::vector<float>  v_vec(B * T * C);
    fill_input(q_vec, q_vec.size());
    fill_input(k_vec, k_vec.size());
    fill_input(v_vec, v_vec.size());

    if (validate) {
        {
            printf("Computing 3p reference... "); fflush(stdout);
            auto start = test_clock::now();
            attention_forward_3p_cpu(out_vec, preatt_vec, att_vec, q_vec, k_vec, v_vec);
            std::chrono::duration<float> sw_time = test_clock::now() - start;
            auto elapsed = sw_time.count();
            printf("done in %f seconds.\n", elapsed);
        }

#if 0
        {
            std::vector<float> tout_vec   (B * T * C);
            std::vector<float> tpreatt_vec(B * NH * T * T);
            std::vector<float> tatt_vec   (B * NH * T * T);

            printf("Computing 2p reference... "); fflush(stdout);
            auto start = test_clock::now();
            attention_forward_2p_cpu(tout_vec, tpreatt_vec, tatt_vec, q_vec, k_vec, v_vec);
            std::chrono::duration<float> sw_time = test_clock::now() - start;
            auto elapsed = sw_time.count();
            printf("done in %f seconds.\n", elapsed);

            printf("Checking results: preatt... "); fflush(stdout);
            check_results(tpreatt_vec, preatt_vec);
            printf("Checking results: att... "); fflush(stdout);
            check_results(tatt_vec, att_vec);
            printf("Checking results: out... "); fflush(stdout);
            check_results(tout_vec, out_vec);
            printf(" done!\n");
        }
#endif

#if 0
        {
            std::vector<float> tout_vec   (B * T * C);

            printf("Computing 1p reference... "); fflush(stdout);
            auto start = test_clock::now();
            attention_forward_1p_cpu(tout_vec, q_vec, k_vec, v_vec);
            std::chrono::duration<float> sw_time = test_clock::now() - start;
            auto elapsed = sw_time.count();
            printf("done in %f seconds.\n", elapsed);

            printf("Checking results: out... "); fflush(stdout);
            check_results(tout_vec, out_vec);
            printf(" done!\n");
        }
#endif

#if 0
        {
            std::vector<float> tout_vec   (B * T * C);

            printf("Computing 1p column blocked reference... "); fflush(stdout);
            auto start = test_clock::now();
            attention_forward_1p_colblock_cpu(tout_vec, q_vec, k_vec, v_vec);
            std::chrono::duration<float> sw_time = test_clock::now() - start;
            auto elapsed = sw_time.count();
            printf("done in %f seconds.\n", elapsed);

            printf("Checking results: out... "); fflush(stdout);
            check_results(tout_vec, out_vec);
            printf(" done!\n");
        }
#endif

#if 0
        {
            std::vector<float> tout_vec   (B * T * C);

            printf("Computing 1p row and column blocked reference... "); fflush(stdout);
            auto start = test_clock::now();
            attention_forward_1p_rcblock_cpu(tout_vec, q_vec, k_vec, v_vec);
            std::chrono::duration<float> sw_time = test_clock::now() - start;
            auto elapsed = sw_time.count();
            printf("done in %f seconds.\n", elapsed);

            printf("Checking results: out... "); fflush(stdout);
            check_results(tout_vec, out_vec);
            printf(" done!\n");
        }
#endif
    }

    printf("Creating source buffers...\n");
    cl::Buffer q{context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, B * T * C * sizeof(float), q_vec.data()};
    cl::Buffer k{context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, B * T * C * sizeof(float), k_vec.data()};
    cl::Buffer v{context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, B * T * C * sizeof(float), v_vec.data()};

    cl::Buffer out      {context, CL_MEM_READ_WRITE, B * T * C      * sizeof(float)};
    cl::Buffer preatt   {context, CL_MEM_READ_WRITE, B * NH * T * T * sizeof(float)};
    cl::Buffer att      {context, CL_MEM_READ_WRITE, B * NH * T * T * sizeof(float)};

    printf("Running tests...\n");

    if (mask & 0x1) {
        naive_3p_attention_forward(context, program, queue, out, preatt, att, q, k, v, wgSize, out_vec, preatt_vec, att_vec);
    }

    if (mask & 0x10) {
        flash_attention_minimal_forward(context, program, queue, out, q, k, v, wgSize, out_vec);
    }

    if (mask & 0x20) {
        flash_attention_forward("flash_attention", context, program, queue, out, q, k, v, wgSize, out_vec);
        flash_attention_forward("flash_attention_blocked", context, program, queue, out, q, k, v, wgSize, out_vec);
    }

    if (mask & 0x40) {
        flash_attention_forward_wg("flash_attention_wg", context, program, queue, out, q, k, v, wgSize, out_vec);
    }

    printf("Done.\n");

    return 0;
}
