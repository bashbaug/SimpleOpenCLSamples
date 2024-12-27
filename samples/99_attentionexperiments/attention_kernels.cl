// These should be defined by build options, but these are reasonable defaults,
// and allowing them to be defined here helps intellisense.
#if !defined(B)
#define B 8
#endif
#if !defined(T)
#define T 1024
#endif
#if !defined(C)
#define C 768
#endif
#if !defined(NH)
#define NH 12
#endif

#define D (C / NH)  // head size

kernel void naive_3p_query_key(global float* preatt, global const float* q_base, global const float* k_base)
{
    // Note: Global work size is B * NH * T * T
    int idx = get_global_id(0);

    int b = idx / (NH * T * T);
    int nh = (idx / (T * T)) % NH;
    int tk = idx % T;
    int tq = (idx / T) % T;

    // Note: q, k, v are (B, NH, T, D)
    global const float* q = q_base + ((b * NH + nh) * T + tq) * D;
    global const float* k = k_base + ((b * NH + nh) * T + tk) * D;

    if (CAUSAL && tk > tq) {
        preatt[idx] = -INFINITY;    // causal mask
    }
    else {
        float val = 0.0f;
        for (int d = 0; d < D; d++) {
            val += q[d] * k[d];
        }
        val *= 1.0f / sqrt((float)D);
        preatt[idx] = val;
    }
}

kernel void naive_3p_softmax(global float* att, global const float* preatt)
{
    // Note: Global work size is B * T * NH
    int idx = get_global_id(0);

    int h = idx % NH;
    int t = (idx / NH) % T;
    int b = idx / (NH * T);

    const int tCheck = CAUSAL ? t + 1 : T;

    global const float* preatt_bth = preatt + b*NH*T*T + h*T*T + t*T;
    global float* att_bth = att + b*NH*T*T + h*T*T + t*T;

    // find maxval
    float maxval = -10000.0f; // TODO something better
    for (int t2 = 0; t2 < tCheck; t2++) {
        float val = preatt_bth[t2];
        maxval = fmax(val, maxval);
    }

    // calculate the exp and keep track of sum
    float expsum = 0.0f;
    for (int t2 = 0; t2 < tCheck; t2++) {
        float expv = native_exp(preatt_bth[t2] - maxval);
        expsum += expv;
        att_bth[t2] = expv;
    }
    float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

    // normalize to get the softmax
    for (int t2 = 0; t2 < T; t2++) {
        if (t2 < tCheck) {
            att_bth[t2] *= expsum_inv;
        } else {
            // causal attention mask. not strictly necessary to set to zero here
            // only doing this explicitly for debugging and checking to PyTorch
            att_bth[t2] = 0.0f;
        }
    }
}

kernel void naive_3p_value(global float* out, global const float* att, global const float* v_base)
{
    // Note: Global work size is B * T * NH
    int idx = get_global_id(0);
    int nh = idx % NH;
    int t = (idx / NH) % T;
    int b = idx / (NH * T);

    const int tCheck = CAUSAL ? t + 1 : T;

    // Note: out is (B, NH, T, D)
    global float* out_bth = out + b * NH * T * D + nh * T * D + t * D;
    global const float* att_bth = att + b*NH*T*T + nh*T*T + t*T;

    for (int i = 0; i < D; i++) {
        out_bth[i] = 0.0f;
    }
    for (int t2 = 0; t2 < tCheck; t2++) {
        // Note: q, k, v are (B, NH, T, D)
        global const float* v = v_base + + b * NH * T * D + nh * T * D + t2 * D;
        float att_btht2 = att_bth[t2];
        for (int i = 0; i < D; i++) {
            out_bth[i] += att_btht2 * v[i];
        }
    }
}

// This is a port of the minimal flash attention, see:
// https://github.com/tspeterkim/flash-attention-minimal
kernel void flash_attention_minimal(
    global const float* Q, global const float* K, global const float* V,
    const int Bc, const int Br, // Note: this implementation requires Bc == Br!!
    const float softmax_scale,
    global float* l, global float* m, global float* O,
    local float* Qc,
    local float* Kc,
    local float* Vc,
    local float* SP)    // Used for both S and P
{
    // scale the scale, so we can use exp2 instead of exp
    const float adjusted_scale = softmax_scale * M_LOG2E_F;

    // Note: Global Work Size is (B * Bc, NH)
    // Note: Local Work Size is (Bc, 1)
    //  --> Group ID is (batch index, head index)
    const int b = get_group_id(0);
    const int nh = get_group_id(1);

    //  --> Local ID is row or column index
    const int rc = get_local_id(0);

    // Note: q, k, v are (B, NH, T, D)
    const int qkv_offset = (b * NH * T * D) + (nh * T * D);

    // Note: l, m are (B, NH, T)
    const int lm_offset = (b * NH * T) + (nh * T);  // offset for l and m

    for (int to = 0; to < T; to += Bc) {
        // Load K, V to SLM
        // Each work-item loads one row of Kc and Vc.
        for (int d = 0; d < D; d++) {
            Kc[(rc * D) + d] = K[qkv_offset + ((to + rc) * D) + d];
            Vc[(rc * D) + d] = V[qkv_offset + ((to + rc) * D) + d];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int ti = 0; ti < T; ti += Br)  {
            // Load Q to SLM
            // Each work-item loads one row of Qc
            for (int d = 0; d < D; d++) {
                Qc[(rc * D) + d] = Q[qkv_offset + ((ti + rc) * D) + d];
            }

            // Compute SP = QK^T, mi_local = rowmax(SP)
            // Each work-item computes one row of S,
            // from one row of Qc and Kc.
            float mi_local = -INFINITY;
            for (int y = 0; y < Bc; y++) {
                float qk = 0;
                if (CAUSAL && ti + rc < to + y) {
                    qk = -INFINITY;
                }
                else {
                    for (int d = 0; d < D; d++) {
                        qk += Qc[(rc * D) + d] * Kc[(y * D) + d];
                    }
                    qk *= adjusted_scale;
                }
                SP[(Bc * rc) + y] = qk;
                mi_local = fmax(qk, mi_local);
            }

            // SP = exp(SP - mi_local), vm = rowsum(SP)
            // implement softmax with causal masking
            float vm = 0;
            for (int y = 0; y < Bc; y++) {
                SP[(Bc * rc) + y] = native_exp2(SP[(Bc * rc) + y] - mi_local);
                vm += SP[(Bc * rc) + y];
            }

            // Compute new m and l
            float mim1 = m[lm_offset + ti + rc];
            float dim1 = l[lm_offset + ti + rc];

            float mi = fmax(mim1, mi_local);
            float di = dim1 * native_exp2(mim1 - mi) + vm * native_exp2(mi_local - mi);

            float om = dim1 * native_exp2(mim1 - mi) / di;
            vm = native_exp2(mi_local - mi) / di;

            // Write O, l, m to HBM
            for (int d = 0; d < D; d++) {
                float pv = 0;  // Pij * Vc
                for (int y = 0; y < Bc; y++) {
                    //if (to * Bc + y >= T) {
                    //    break;
                    //}
                    pv += SP[(Bc * rc) + y] * Vc[(y * D) + d];
                }
                // O is (B, NH, T, D)
                O[qkv_offset + ((ti + rc) * D) + d] =
                    om * O[qkv_offset + ((ti + rc) * D) + d] +
                    vm * pv;
            }

            m[lm_offset + ti + rc] = mi;
            l[lm_offset + ti + rc] = di;
        }
        barrier(CLK_LOCAL_MEM_FENCE); // otherwise, thread can use the wrong Kc, Vc in inner loop
    }
}

// This is a very basic flash attention kernel.
// For this kernel, each work-item computes one row of D elements of the output.
// There is no caching of the Q or O data.
// There is also no sharing of the K or V data.
kernel void flash_attention(
    global const float* restrict Q, global const float* restrict K, global const float* restrict V,
    global float* restrict O,
    const float scale)
{
    // Note: all data is arranged: B x NH x T x D

    // scale the scale, so we can use exp2 instead of exp
    const float adjusted_scale = scale * M_LOG2E_F;

    int to = get_global_id(0);
    int nh = get_global_id(1);
    int b = get_global_id(2);

    global float* o = O + b * NH * T * D + nh * T * D + to * D;

    global const float* q = Q + b * NH * T * D + nh * T * D + to * D;
    float mi = -INFINITY;
    float di = 0.0f;

    for (int ti = 0; ti < T; ti++) {
        global const float* k = K + b * NH * T * D + nh * T * D + ti * D;
        global const float* v = V + b * NH * T * D + nh * T * D + ti * D;

        // Compute qk = QK^T
        float qk = 0.0f;
        if (CAUSAL && to < ti) {
            qk = -INFINITY;
        }
        else {
            for (int d = 0; d < D; d++) {
                qk += q[d] * k[d];
            }
            qk *= adjusted_scale;
        }

        // Update the running maximum
        float mim1 = mi;
        mi = fmax(mim1, qk);

        // softmax(qk)
        float smxi = native_exp2(qk - mi);

        // Update di
        float alpha = native_exp2(mim1 - mi);
        di = di * alpha + smxi;

        // Update the un-scaled output from softmax(qk) and V
        for (int d = 0; d < D; d++) {
            o[d] = o[d] * alpha + smxi * v[d];
        }
    }

    // Epilog scaling (flash attention 2)
    for (int d = 0; d < D; d++) {
        o[d] = o[d] * native_recip(di);
    }
}

// This is a slightly more complicated flash attention kernel.
// For this kernel, each work-item still computes one row of D elements of the output.
#define BLOCK_N D
kernel void flash_attention_blocked(
    global const float* restrict Q, global const float* restrict K, global const float* restrict V,
    global float* restrict O,
    const float scale)
{
    // scale the scale, so we can use exp2 instead of exp
    const float adjusted_scale = scale * M_LOG2E_F;

    int to = get_global_id(0);
    int nh = get_global_id(1);
    int b = get_global_id(2);

    global float* o = O + b * NH * T * D + nh * T * D + to * D;

    global const float* q = Q + b * NH * T * D + nh * T * D + to * D;
    float mi = -INFINITY;
    float di = 0.0f;

    for (int ti = 0; ti < T; ti+=BLOCK_N) {
        global const float* k = K + b * NH * T * D + nh * T * D + ti * D;
        global const float* v = V + b * NH * T * D + nh * T * D + ti * D;

        // Compute qk = QK^T
        float qk[BLOCK_N];
        for (int tt = 0; tt < BLOCK_N; tt++) {
            qk[tt] = 0.0f;
            for (int d = 0; d < D; d++) {
                qk[tt] += q[d] * k[(tt * D) + d];
            }
            qk[tt] *= adjusted_scale;
            if (CAUSAL) {
                qk[tt] = (to < ti + tt) ? -INFINITY : qk[tt];
            }
        }

        // Update the running maqkmum
        float mim1 = mi;
        for (int tt = 0; tt < BLOCK_N; tt++) {
            mi = fmax(mi, qk[tt]);  // max reduction
        }

        // softmax(qk)
        float l = 0.0f;
        float smqk[BLOCK_N];
        for (int tt = 0; tt < BLOCK_N; tt++) {
            smqk[tt] = native_exp2(qk[tt] - mi);
            l = l + smqk[tt];   // add reduction
        }

        // Update di
        float alpha = native_exp2(mim1 - mi);
        di = di * alpha + l;

#if 1
        // Update the un-scaled output from softmax(qk) and V
        for (int d = 0; d < D; d++) {
            o[d] = o[d] * alpha;
        }
        for (int d = 0; d < D; d++) {
            float update = 0.0f;
            for (int tt = 0; tt < BLOCK_N; tt++) {
                update += smqk[tt] * v[(tt * D) + d];
            }
            o[d] += update;
        }
#else
        // Update the un-scaled output from softmax(qk) and V
        for (int d = 0; d < D; d++) {
            float update = o[d] * alpha;
            for (int tt = 0; tt < BLOCK_N; tt++) {
                update += smqk[tt] * v[(tt * D) + d];
            }
            o[d] = update;
        }
#endif
    }

    // Epilog scaling (flash attention 2)
    for (int d = 0; d < D; d++) {
        o[d] = o[d] * native_recip(di);
    }
}
#undef BLOCK_N


// This is a slightly more complicated flash attention kernel.
// For this kernel, each work-group computes one row of D elements of the output.
__attribute__((reqd_work_group_size(D, 1, 1)))
kernel void flash_attention_wg(
    global const float* restrict Q, global const float* restrict K, global const float* restrict V,
    global float* restrict O,
    const float scale)
{
    // scale the scale, so we can use exp2 instead of exp
    const float adjusted_scale = scale * M_LOG2E_F;

    const int tt = get_local_id(0);

    const int to = get_group_id(0);
    const int nh = get_group_id(1);
    const int b = get_group_id(2);

    global float* o = O + b * NH * T * D + nh * T * D + to * D;
    float acc = o[tt];

    global const float* q = Q + b * NH * T * D + nh * T * D + to * D;
    float mi = -INFINITY;
    float di = 0.0f;

    local float scratch[D];

    for (int ti = 0; ti < T; ti+=D) {
        global const float* k = K + b * NH * T * D + nh * T * D + ti * D;
        global const float* v = V + b * NH * T * D + nh * T * D + ti * D;

        // Compute qk = QK^T
        float qk = 0.0f;
        for (int d = 0; d < D; d++) {
            qk += q[d] * k[(tt * D) + d];
        }
        qk *= adjusted_scale;
        if (CAUSAL) {
            qk = (to < ti + tt) ? -INFINITY : qk;
        }

        // Update the running maximum
        float mim1 = mi;
        barrier(CLK_LOCAL_MEM_FENCE);
        scratch[tt] = qk;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int d = 0; d < D; d++) {
            mi = fmax(mi, scratch[d]);  // max reduction of qk
        }

        // softmax(qk)
        float l = 0.0f;
        float smqk = native_exp2(qk - mi);
        barrier(CLK_LOCAL_MEM_FENCE);
        scratch[tt] = smqk;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int d = 0; d < D; d++) {
            l = l + scratch[d]; // add reduction of smqk
        }

        // Update di
        float alpha = native_exp2(mim1 - mi);
        di = di * alpha + l;

        // Update the un-scaled output from softmax(qk) and V
        acc = acc * alpha;
        for (int d = 0; d < D; d++) {
            acc += scratch[d] * v[(d * D) + tt];
        }
    }

    // Epilog scaling (flash attention 2)
    o[tt] = acc * native_recip(di);
}
