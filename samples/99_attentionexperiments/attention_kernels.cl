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

kernel void naive_query_key(global float* preatt, global const float* q_base, global const float* k_base)
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

    if (tk > tq) {
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

kernel void naive_softmax(global float* att, global const float* preatt)
{
    // Note: Global work size is B * T * NH
    int idx = get_global_id(0);

    int h = idx % NH;
    int t = (idx / NH) % T;
    int b = idx / (NH * T);

    global const float* preatt_bth = preatt + b*NH*T*T + h*T*T + t*T;
    global float* att_bth = att + b*NH*T*T + h*T*T + t*T;

    // find maxval
    float maxval = -10000.0f; // TODO something better
    for (int t2 = 0; t2 <= t; t2++) {
        if (preatt_bth[t2] > maxval) {
            maxval = preatt_bth[t2];
        }
    }

    // calculate the exp and keep track of sum
    float expsum = 0.0f;
    for (int t2 = 0; t2 <= t; t2++) {
        float expv = native_exp(preatt_bth[t2] - maxval);
        expsum += expv;
        att_bth[t2] = expv;
    }
    float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

    // normalize to get the softmax
    for (int t2 = 0; t2 < T; t2++) {
        if (t2 <= t) {
            att_bth[t2] *= expsum_inv;
        } else {
            // causal attention mask. not strictly necessary to set to zero here
            // only doing this explicitly for debugging and checking to PyTorch
            att_bth[t2] = 0.0f;
        }
    }
}

kernel void naive_value(global float* out, global const float* att, global const float* v)
{
    const int D = C / NH; // head size

    // Note: Global work size is B * T * NH
    int idx = get_global_id(0);
    int nh = idx % NH;
    int t = (idx / NH) % T;
    int b = idx / (NH * T);

    // Note: out is (B, NH, T, D)
    global float* out_bth = out + b * NH * T * D + nh * T * D + t * D;
    global const float* att_bth = att + b*NH*T*T + nh*T*T + t*T;

    for (int i = 0; i < D; i++) {
        out_bth[i] = 0.0f;
    }
    for (int t2 = 0; t2 <= t; t2++) {
        // Note: q, k, v are (B, NH, T, D)
        global const float* value_t2 = v + + b * NH * T * D + nh * T * D + t2 * D;
        float att_btht2 = att_bth[t2];
        for (int i = 0; i < D; i++) {
            out_bth[i] += att_btht2 * value_t2[i];
        }
    }
}

kernel void flash_attention(
    global const float* Q, global const float* K, global const float* V,
    const int Tc, const int Tr,
    const int Bc, const int Br,
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
    const int tx = get_local_id(0);
    const int b = get_group_id(0);
    const int nh = get_group_id(1);

    // Note: q, k, v are (B, NH, T, D)
    const int qkv_offset = (b * NH * T * D) + (nh * T * D);

    // Note: l, m are (B, NH, T)
    const int lm_offset = (b * NH * T) + (nh * T);  // offset for l and m

    for (int j = 0; j < Tc; j++) {
        // Load K, V to SLM
        for (int x = 0; x < D; x++) {
            Kc[(tx * D) + x] = K[qkv_offset + ((Bc * j + tx) * D) + x];
            Vc[(tx * D) + x] = V[qkv_offset + ((Bc * j + tx) * D) + x];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int i = 0; i < Tr; i++)  {
            // if past the end of the sequence, break
            if (i * Br + tx >= T) {
                break;
            }

            // Load Q to SRAM
            for (int x = 0; x < D; x++) {
                Qc[(tx * D) + x] = Q[qkv_offset + ((Bc * i + tx) * D) + x];
            }

            // SP = QK^T, row_m = rowmax(SP)
            // with causal masking
            float row_m = -INFINITY;
            for (int y = 0; y < Bc; y++) {
                //if (j * Bc + y >= T) {
                //    break;
                //}
                float sum = 0;
                for (int x = 0; x < D; x++) {
                    sum += Qc[(tx * D) + x] * Kc[(y * D) + x];
                }
                sum *= softmax_scale;
                if (i * Br + tx < j * Bc + y)
                    sum = -INFINITY;
                SP[(Bc * tx) + y] = sum;

                if (sum > row_m)
                    row_m = sum;
            }

            // SP = exp(SP - row_m), row_l = rowsum(SP)
            // implement softmax with causal masking
            float row_l = 0;
            for (int y = 0; y < Bc; y++) {
                //if (j * Bc + y >= T) {
                //    break;
                //}
                if (i * Br + tx < j * Bc + y)
                    SP[(Bc * tx) + y] = 0;
                else
                    SP[(Bc * tx) + y] = native_exp(SP[(Bc * tx) + y] - row_m);
                row_l += SP[(Bc * tx) + y];
            }

            // Compute new m and l
            float row_m_prev = m[lm_offset + (Br * i) + tx];
            float row_l_prev = l[lm_offset + (Br * i) + tx];

            float row_m_new = max(row_m_prev, row_m);
            float row_l_new = row_l_prev * native_exp(row_m_prev - row_m_new) + row_l * native_exp(row_m - row_m_new);

            // Write O, l, m to HBM
            for (int x = 0; x < D; x++) {
                float pv = 0;  // Pij * Vc
                for (int y = 0; y < Bc; y++) {
                    if (j * Bc + y >= T) {
                        break;
                    }
                    pv += SP[(Bc * tx) + y] * Vc[(y * D) + x];
                }
                // O is (B, NH, T, D)
                O[qkv_offset + ((Bc * i + tx) * D) + x] =
                    (1 / row_l_new) *
                    (row_l_prev * native_exp(row_m_prev - row_m_new) * O[qkv_offset + ((Bc * i + tx) * D) + x] + 
                        native_exp(row_m - row_m_new) * pv);
            }

            m[lm_offset + (Br * i) + tx] = row_m_new;
            l[lm_offset + (Br * i) + tx] = row_l_new;
        }
        barrier(CLK_LOCAL_MEM_FENCE); // otherwise, thread can use the wrong Kc, Vc in inner loop
    }
}
