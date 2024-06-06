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

kernel void naive_query_key_old(global float* preatt, global const float* inp)
{
    const int D = C / NH; // head size

    int idx = get_global_id(0);
    int total_threads = B * NH * T * T;

    if (idx < total_threads) {
        int t2 = idx % T;
        int t = (idx / T) % T;
        if (t2 > t) {
            // autoregressive mask
            preatt[idx] = -INFINITY;
            return;
        }
        int h = (idx / (T * T)) % NH;
        int b = idx / (NH * T * T);

        int C3 = C*3;
        global const float* query_t = inp + b * T * C3 + t * C3 + h * D;
        global const float* key_t2 = inp + b * T * C3 + t2 * C3 + h * D + C; // +C because it's key

        // (query_t) dot (key_t2)
        float val = 0.0f;
        for (int i = 0; i < D; i++) {
            val += query_t[i] * key_t2[i];
        }
        val *= 1.0f / sqrt((float)D);
        preatt[idx] = val;
    }
}

kernel void naive_query_key(global float* preatt, global const float* q, global const float* k)
{
    // Note: q, k, v are (B, NH, N, D)
    const int D = C / NH; // head size

    // Note: Global work size is B * NH * T * T
    int idx = get_global_id(0);
    int total_threads = B * NH * T * T;

    int t2 = idx % T;
    int t = (idx / T) % T;
    if (t2 > t) {
        // autoregressive mask
        preatt[idx] = -INFINITY;
        return;
    }
    int nh = (idx / (T * T)) % NH;
    int b = idx / (NH * T * T);

    global const float* query_t = q + b * NH * T * D + nh * T * D + t * D;
    global const float* key_t2 = k + b * NH * T * D + nh * T * D + t2 * D;

    // (query_t) dot (key_t2)
    float val = 0.0f;
    for (int i = 0; i < D; i++) {
        val += query_t[i] * key_t2[i];
    }
    val *= 1.0f / sqrt((float)D);
    preatt[idx] = val;
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
        // could try native_exp...
        float expv = exp(preatt_bth[t2] - maxval);
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

kernel void naive_value_old(global float* out, global const float* att, global const float* inp)
{
    int idx = get_global_id(0);
    int total_threads = B * T * NH;

    if (idx < total_threads) {
        int h = idx % NH;
        int t = (idx / NH) % T;
        int b = idx / (NH * T);

        int C3 = C*3;
        int D = C / NH; // head size

        global float* out_bth = out + b * T * C + t * C + h * D;
        global const float* att_bth = att + b*NH*T*T + h*T*T + t*T;

        for (int i = 0; i < D; i++) { out_bth[i] = 0.0f; }
        for (int t2 = 0; t2 <= t; t2++) {
            global const float* value_t2 = inp + b * T * C3 + t2 * C3 + h * D + C*2; // +C*2 because it's value
            float att_btht2 = att_bth[t2];
            for (int i = 0; i < D; i++) {
                out_bth[i] += att_btht2 * value_t2[i];
            }
        }
    }
}

kernel void naive_value(global float* out, global const float* att, global const float* v)
{
    // Note: q, k, v are (B, NH, N, D)
    const int D = C / NH; // head size

    // Note: Global work size is B * T * NH
    int idx = get_global_id(0);

    int nh = idx % NH;
    int t = (idx / NH) % T;
    int b = idx / (NH * T);

    global float* out_bth = out + b * T * C + t * C + nh * D;
    global const float* att_bth = att + b*NH*T*T + nh*T*T + t*T;

    for (int i = 0; i < D; i++) {
        out_bth[i] = 0.0f;
    }
    for (int t2 = 0; t2 <= t; t2++) {
        global const float* value_t2 = v +  + b * NH * T * D + nh * T * D + t2 * D;
        float att_btht2 = att_bth[t2];
        for (int i = 0; i < D; i++) {
            out_bth[i] += att_btht2 * value_t2[i];
        }
    }
}

kernel void flash_attention(
    global const float* Q, global const float* K, global const float* V,
    const int N, const int d,
    const int Tc, const int Tr,
    const int Bc, const int Br,
    const float softmax_scale,
    global float* l, global float* m, global float* O,
    local float* sram )
{
    int tx = get_local_id(0);
    int bx = get_group_id(0); int by = get_group_id(1);  // batch and head index

    // Offset into Q,K,V,O,l,m - different for each batch and head
    int qkv_offset = (bx * get_num_groups(1) * N * d) + (by * N * d);  // get_num_groups(1) = nh
    int lm_offset = (bx * get_num_groups(1) * N) + (by * N);  // offset for l and m

    // Define SRAM for Q,K,V,S
    int tile_size = Bc * d;  // size of Qi, Kj, Vj
    local float* Qi = sram;
    local float* Kj = &sram[tile_size];
    local float* Vj = &sram[tile_size * 2];
    local float* S = &sram[tile_size * 3];

    for (int j = 0; j < Tc; j++) {

        // Load Kj, Vj to SRAM
        for (int x = 0; x < d; x++) {
            Kj[(tx * d) + x] = K[qkv_offset + (tile_size * j) + (tx * d) + x];
            Vj[(tx * d) + x] = V[qkv_offset + (tile_size * j) + (tx * d) + x];
        }
        barrier(CLK_LOCAL_MEM_FENCE); // such that the inner loop can use the correct Kj, Vj

        for (int i = 0; i < Tr; i++)  {
            // if past the end of the sequence, break
            if (i * Br + tx >= N) {
                break;
            }

            // Load Qi to SRAM, l and m to registers
            for (int x = 0; x < d; x++) {
                Qi[(tx * d) + x] = Q[qkv_offset + (tile_size * i) + (tx * d) + x];
            }
            float row_m_prev = m[lm_offset + (Br * i) + tx];
            float row_l_prev = l[lm_offset + (Br * i) + tx];

            // S = QK^T, row_m = rowmax(S)
            // S[tx][y] = Sum_{x = 0}^{d-1} {Qi[tx][x] * Kj[y][x]}
            // row_m = Max_{y = 0}^{Bc-1} S[tx][y]
            // with causal masking
            float row_m = -INFINITY;
            for (int y = 0; y < Bc; y++) {
                if (j * Bc + y >= N) {
                    break;
                }
                float sum = 0;
                for (int x = 0; x < d; x++) {
                    sum += Qi[(tx * d) + x] * Kj[(y * d) + x];
                }
                sum *= softmax_scale;
                if (i * Br + tx < j * Bc + y)
                    sum = -INFINITY;
                S[(Bc * tx) + y] = sum;

                if (sum > row_m)
                    row_m = sum;
            }

            // implement softmax with causal masking
            // P = exp(S - row_m), row_l = rowsum(P)
            // P[tx][y] = exp(S[tx][y] - row_m)
            float row_l = 0;
            for (int y = 0; y < Bc; y++) {
                if (j * Bc + y >= N) {
                    break;
                }
                if (i * Br + tx < j * Bc + y)
                    S[(Bc * tx) + y] = 0;
                else
                    S[(Bc * tx) + y] = native_exp(S[(Bc * tx) + y] - row_m);
                row_l += S[(Bc * tx) + y];
            }

            // Compute new m and l
            float row_m_new = max(row_m_prev, row_m);
            float row_l_new = (native_exp(row_m_prev - row_m_new) * row_l_prev) + (native_exp(row_m - row_m_new) * row_l);

            // Write O, l, m to HBM
            for (int x = 0; x < d; x++) {
                float pv = 0;  // Pij * Vj
                for (int y = 0; y < Bc; y++) {
                    if (j * Bc + y >= N) {
                        break;
                    }
                    pv += S[(Bc * tx) + y] * Vj[(y * d) + x];
                }
                O[qkv_offset + (tile_size * i) + (tx * d) + x] = (1 / row_l_new) \
                    * ((row_l_prev * native_exp(row_m_prev - row_m_new) * O[qkv_offset + (tile_size * i) + (tx * d) + x]) \
                    + (native_exp(row_m - row_m_new) * pv));
            }
            m[lm_offset + (Br * i) + tx] = row_m_new;
            l[lm_offset + (Br * i) + tx] = row_l_new;
        }
        barrier(CLK_LOCAL_MEM_FENCE); // otherwise, thread can use the wrong Kj, Vj in inner loop
    }
}

kernel void permute(global float* q, global float* k, global float* v, global const float* inp)
{
    const int N = T;
    const int d = C / NH;

    // okay so now, this kernel wants Q,K,V to all be of shape (B, NH, N, d)
    // but instead, we have a single tensor QKV (inp) of shape (B, N, 3, NH, d)
    int idx = get_global_id(0);

    // Q[b][nh_][n][d_] = inp[b][n][0][nh_][d_]

    int b = idx / (NH * N * d);
    int rest = idx % (NH * N * d);
    int nh_ = rest / (N * d);
    rest = rest % (N * d);
    int n = rest / d;
    int d_ = rest % d;

    int inp_idx = \
        (b * N * 3 * NH * d)
        +   (n * 3 * NH * d)
        +       (0 * NH * d)
        +          (nh_ * d)
        +                d_;

    q[idx] = inp[inp_idx];
    k[idx] = inp[inp_idx + NH * d];
    v[idx] = inp[inp_idx + 2 * (NH * d)];
}

kernel void unpermute(global const float* temp, global float *out)
{
    const int N = T;
    const int d = C / NH;

    // out has shape (B, nh, N, d) but we need to unpermute it to (B, N, nh, d)
    int idx = get_global_id(0);

    // out[b][n][nh_][d_] <- temp[b][nh_][n][d_]

    int b = idx / (NH * N * d);
    int rest = idx % (NH * N * d);
    int nh_ = rest / (N * d);
    rest = rest % (N * d);
    int n = rest / d;
    int d_ = rest % d;

    int other_idx = (b * NH * N * d) + (n * NH * d) + (nh_ * d) + d_;
    out[other_idx] = temp[idx];
}
