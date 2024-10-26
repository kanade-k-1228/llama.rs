use super::{layers::*, param::HyperParam};

use super::{context::Context, weight::Weight};

pub fn decode(
    tok: usize, // new token
    pos: usize, // new token position
    hp: &HyperParam,
    ctx: &mut Context,
    w: &Weight,
) -> Vec<f32> {
    let head_dim = hp.dim / hp.head;
    let norm = 1.0 / (head_dim as f32).sqrt(); // sm(QK/√d)V の 1/√d

    // Embedding
    let emb = w.tok_emb_table[tok].clone();

    let layer_out = (0..hp.layer).fold(emb, |attn_input, layer| {
        // --------------------------------------------------------------------------------
        // Attention

        // 1. RMS Normalize
        let attn_norm = rms_norm(&attn_input, &w.attn_norm[layer]);

        // 2. Weight Multiple
        let attn_wqx = matmul(&attn_norm, &w.attn_wq[layer]);
        let attn_wkx = matmul(&attn_norm, &w.attn_wk[layer]);
        let attn_wvx = matmul(&attn_norm, &w.attn_wv[layer]);

        // 3. RoPE for each head
        let mut attn_q_r = vec![0.0; hp.dim];
        let mut attn_k_r = vec![0.0; hp.dim];
        for head in 0..hp.head {
            rope(
                &mut attn_q_r,
                &mut attn_k_r,
                &attn_wqx,
                &attn_wkx,
                &w.rope_cos[pos],
                &w.rope_sin[pos],
                head * head_dim,
                head_dim,
            );
        }

        // 4. Key / Value Cache
        ctx.k_cache[layer][pos] = attn_k_r;
        ctx.v_cache[layer][pos] = attn_wvx;

        // 5. Multi-Head Attention
        // TODO! Grouped-Query Attention
        let attn_val = (0..hp.head)
            .flat_map(|head| {
                let head_range = (head * head_dim)..((head + 1) * head_dim);

                // 5-1. QK
                let attn_qk = qk_mul(
                    &attn_q_r,
                    &ctx.k_cache[layer],
                    head_range.clone(),
                    0..(pos + 1),
                );

                // 5-2. softmax( QK/√d )
                let attn_score = softmax(&mul(&attn_qk, norm));

                // 5-3. softmax(QK/√d) . V
                qkv_mul(
                    &attn_score,
                    &ctx.v_cache[layer],
                    head_range.clone(),
                    0..(pos + 1),
                )
            })
            .collect::<Vec<_>>();

        // 6. Output (Merge Heads)
        let attn_out = matmul(&attn_val, &w.attn_wo[layer]);

        // 7. Res connect
        let attn_res = add_v(&attn_input, &attn_out);

        // --------------------------------------------------------------------------------
        // FFN

        // 1. RMS Normalize
        let ffn_norm = rms_norm(&attn_res, &w.ffn_norm[layer]);

        // 2. w1 . x
        let ffn_w1x = matmul(&ffn_norm, &w.ffn_w1[layer]);

        // 3. w3 . x
        let ffn_w3x = matmul(&ffn_norm, &w.ffn_w3[layer]);

        // 4. silu( w1x )
        let ffn_act = silu_vec(&ffn_w1x);

        // 5. silu(w1x) * w3x
        let ffn_dot = mul_v(&ffn_act, &ffn_w3x);

        // 6. w2 . silu(w1x)*w3x
        let ffn_out = matmul(&ffn_dot, &w.ffn_w2[layer]);

        // 7. Res connect
        let ffn_res = add_v(&attn_res, &ffn_out);

        ffn_res
    });

    // Final RMS Normalize
    let final_norm = rms_norm(&layer_out, &w.final_norm);

    // Classifier into logits
    let logits = matmul(&final_norm, &w.tok_emb_table);

    logits
}
