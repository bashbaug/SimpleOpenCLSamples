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

kernel void naive_3p_query_key(global float* preatt, global const float* q_base, global const float* k_base)
{
    const int D = C / NH; // head size

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
    const int D = C / NH; // head size

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
    const int Tc, const int Tr,
    const int Bc, const int Br, // Note: this implementation requires Bc == Br!!
    const float softmax_scale,
    global float* l, global float* m, global float* O,
    local float* Qc,
    local float* Kc,
    local float* Vc,
    local float* SP)    // Used for both S and P
{
    const int D = C / NH; // head size

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

    for (int j = 0; j < Tc; j++) {
        // Load K, V to SLM
        // Each work-item loads one row of Kc and Vc.
        for (int d = 0; d < D; d++) {
            Kc[(rc * D) + d] = K[qkv_offset + ((Bc * j + rc) * D) + d];
            Vc[(rc * D) + d] = V[qkv_offset + ((Bc * j + rc) * D) + d];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int i = 0; i < Tr; i++)  {
            // Load Q to SLM
            // Each work-item loads one row of Qc
            for (int d = 0; d < D; d++) {
                Qc[(rc * D) + d] = Q[qkv_offset + ((Bc * i + rc) * D) + d];
            }

            // Compute SP = QK^T, mi_local = rowmax(SP)
            // Each work-item computes one row of S,
            // from one row of Qc and Kc.
            float mi_local = -INFINITY;
            for (int y = 0; y < Bc; y++) {
                float xi = 0;
                if (CAUSAL && i * Br + rc < j * Bc + y) {
                    xi = -INFINITY;
                }
                else {
                    for (int d = 0; d < D; d++) {
                        xi += Qc[(rc * D) + d] * Kc[(y * D) + d];
                    }
                    xi *= softmax_scale;
                }
                SP[(Bc * rc) + y] = xi;
                mi_local = fmax(xi, mi_local);
            }

            // SP = exp(SP - mi_local), vm = rowsum(SP)
            // implement softmax with causal masking
            float vm = 0;
            for (int y = 0; y < Bc; y++) {
                SP[(Bc * rc) + y] = native_exp(SP[(Bc * rc) + y] - mi_local);
                vm += SP[(Bc * rc) + y];
            }

            // Compute new m and l
            float mim1 = m[lm_offset + (Br * i) + rc];
            float dim1 = l[lm_offset + (Br * i) + rc];

            float mi = fmax(mim1, mi_local);
            float di = dim1 * native_exp(mim1 - mi) + vm * native_exp(mi_local - mi);

            float om = dim1 * native_exp(mim1 - mi) / di;
            vm = native_exp(mi_local - mi) / di;

            // Write O, l, m to HBM
            for (int d = 0; d < D; d++) {
                float pv = 0;  // Pij * Vc
                for (int y = 0; y < Bc; y++) {
                    //if (j * Bc + y >= T) {
                    //    break;
                    //}
                    pv += SP[(Bc * rc) + y] * Vc[(y * D) + d];
                }
                // O is (B, NH, T, D)
                O[qkv_offset + ((Bc * i + rc) * D) + d] =
                    om * O[qkv_offset + ((Bc * i + rc) * D) + d] +
                     vm * pv;
            }

            m[lm_offset + (Br * i) + rc] = mi;
            l[lm_offset + (Br * i) + rc] = di;
        }
        barrier(CLK_LOCAL_MEM_FENCE); // otherwise, thread can use the wrong Kc, Vc in inner loop
    }
}

kernel void flash_attention(
    global const float* Q, global const float* K, global const float* V,
    global float* O,
    const float scale)
{
    const int D = C / NH; // head size

    int b = get_global_id(0);
    int nh = get_global_id(1);
    int to = get_global_id(2);

    float* o = O + b * NH * T * D + nh * T * D + to * D;

    const float* q = Q + b * NH * T * D + nh * T * D + to * D;
    float mi = -INFINITY;
    float di = 0.0f;

    for (int ti = 0; ti < T; ti++) {
        const float* k = K + b * NH * T * D + nh * T * D + ti * D;
        const float* v = V + b * NH * T * D + nh * T * D + ti * D;

        // Compute xi = QK^T
        float xi = 0.0;
        if (CAUSAL && to < ti) {
            xi = -INFINITY;
        }
        else {
            for (int d = 0; d < D; d++) {
                xi += q[d] * k[d];
            }
            xi *= scale;
        }

        // Update the running maximum
        float mim1 = mi;
        mi = fmax(mim1, xi);

        // softmax(xi)
        float smxi = native_exp(xi - mi);

        // Update di
        float dim1 = di;
        float exp_dmim1mi = native_exp(mim1 - mi);
        di = dim1 * exp_dmim1mi + smxi;

        // Compute the output from dim1, di, softmax(xi), and v
        float om = dim1 * exp_dmim1mi / di;
        float vm = smxi / di;
        for (int d = 0; d < D; d++) {
            o[d] = o[d] * om + vm * v[d];
        }
    }
}
