use anyhow::{bail, Result};
use det_num::ops::{attention_softmax, rope_rotate_pairs_in_place};
use det_num::{Acc, Act};
use raster::List;
use rayon::prelude::*;

use crate::tensor::{
    dot_bits, linear_slab, mac_weighted_value, pack_i32_page, requantize_acc, unpack_page_i32s,
    Matrix, Slab,
};
use ::prefill_range::input::{
    add_row, gelu, rms_norm, scale_row, value_rms_norm, ActivationRow, ActivationSequence, KeyRow,
    LayerParams, PleLayerInputs, PleRow, TransformerLayer,
};

pub struct PrefillRangeDirectInputs<'a> {
    pub activations: &'a ActivationSequence,
    pub layer: &'a TransformerLayer,
    pub donor_kv: &'a ActivationSequence,
    pub ple: &'a PleLayerInputs,
}

pub fn run_prefill_range_direct(
    inputs: PrefillRangeDirectInputs<'_>,
) -> Result<ActivationSequence> {
    let params = validate_layer_params(&inputs.layer.params)?;
    let weights = LayerWeights::from_layer(inputs.layer, &params)?;
    let input_rows = activation_rows(inputs.activations, params.hidden_size as usize)?;

    if input_rows.rows() == 0 {
        return Ok(ActivationSequence {
            rows: List::new(),
            errors: List::new(),
            kv: List::new(),
            start_position: inputs.activations.start_position,
        });
    }

    let (queries, own_keys) = project_tokens(&input_rows, &weights, &params)?;
    let donor_keys = inputs.donor_kv.kv.as_slice();
    let ple_rows = inputs.ple.rows.as_slice();
    let output_rows = attend_tokens(&queries, &own_keys, donor_keys, ple_rows, &weights, &params)?;

    Ok(ActivationSequence {
        rows: List::from(output_rows),
        errors: List::new(),
        kv: List::from(own_keys),
        start_position: inputs.activations.start_position,
    })
}

struct InputRows {
    token_ids: Vec<u32>,
    slab: Slab,
}

impl InputRows {
    fn rows(&self) -> usize {
        self.slab.rows()
    }
}

struct QueryState {
    position: u32,
    token_id: u32,
    q: Vec<i32>,
    residual: Vec<i32>,
}

struct LayerWeights {
    w_q: Matrix,
    w_k: Matrix,
    w_v: Matrix,
    w_o: Matrix,
    w_gate: Matrix,
    w_up: Matrix,
    w_down: Matrix,
    ple_input_gate: Matrix,
    ple_layer_projection: Matrix,
}

impl LayerWeights {
    fn from_layer(layer: &TransformerLayer, params: &LayerParams) -> Result<Self> {
        let hidden = params.hidden_size as usize;
        let ffn = params.ffn_size as usize;
        let q_len = params.num_heads as usize * params.head_dim as usize;
        let kv_len = params.num_kv_heads as usize * params.head_dim as usize;
        let ple_width = params.ple_width as usize;
        Ok(Self {
            w_q: Matrix::from_region("w_q", &layer.w_q, q_len, hidden)?,
            w_k: Matrix::from_region("w_k", &layer.w_k, kv_len, hidden)?,
            w_v: Matrix::from_region("w_v", &layer.w_v, kv_len, hidden)?,
            w_o: Matrix::from_region("w_o", &layer.w_o, hidden, q_len)?,
            w_gate: Matrix::from_region("w_gate", &layer.w_gate, ffn, hidden)?,
            w_up: Matrix::from_region("w_up", &layer.w_up, ffn, hidden)?,
            w_down: Matrix::from_region("w_down", &layer.w_down, hidden, ffn)?,
            ple_input_gate: Matrix::from_region(
                "ple_input_gate",
                &layer.ple_input_gate,
                ple_width,
                hidden,
            )?,
            ple_layer_projection: Matrix::from_region(
                "ple_layer_projection",
                &layer.ple_layer_projection,
                hidden,
                ple_width,
            )?,
        })
    }
}

fn validate_layer_params(params: &LayerParams) -> Result<LayerParams> {
    let hidden = params.hidden_size as usize;
    let ffn = params.ffn_size as usize;
    let heads = params.num_heads as usize;
    let kv_heads = params.num_kv_heads as usize;
    let head_dim = params.head_dim as usize;
    if hidden == 0 || ffn == 0 {
        bail!("transformer layer requires non-zero hidden_size and ffn_size");
    }
    if heads == 0 || kv_heads == 0 || head_dim == 0 {
        bail!("transformer layer requires non-zero num_heads, num_kv_heads and head_dim");
    }
    if heads % kv_heads != 0 {
        bail!(
            "layer {}: {heads} query heads do not group evenly over {kv_heads} kv heads",
            params.layer_idx
        );
    }
    let rotary_dim = params.rotary_dim as usize;
    if rotary_dim % 2 != 0 {
        bail!(
            "layer {}: rotary_dim {rotary_dim} must be even",
            params.layer_idx
        );
    }
    if rotary_dim > head_dim {
        bail!(
            "layer {}: rotary_dim {rotary_dim} exceeds head_dim {head_dim}",
            params.layer_idx
        );
    }
    let freq_base_dim = params.rope_freq_base_dim as usize;
    if rotary_dim > 0 && (freq_base_dim < 2 || freq_base_dim % 2 != 0) {
        bail!(
            "layer {}: rope_freq_base_dim {freq_base_dim} must be even and at least 2",
            params.layer_idx
        );
    }
    if rotary_dim > 0 && params.rope_base <= 0 {
        bail!("layer {}: rope_base must be positive", params.layer_idx);
    }
    if params.norm_eps < 0 {
        bail!("rms-norm epsilon must be non-negative");
    }

    for (name, page, values) in [
        ("norm_input", &params.norm_input, hidden),
        ("norm_post_attn", &params.norm_post_attn, hidden),
        ("norm_pre_ffw", &params.norm_pre_ffw, hidden),
        ("norm_post_ffw", &params.norm_post_ffw, hidden),
        ("q_norm", &params.q_norm, head_dim),
        ("k_norm", &params.k_norm, head_dim),
    ] {
        let expected = values * 4;
        if page.len() != expected {
            bail!(
                "layer {} {name} has {} bytes, expected {expected} ({values} values)",
                params.layer_idx,
                page.len()
            );
        }
    }
    Ok(params.clone())
}

fn activation_rows(activations: &ActivationSequence, hidden: usize) -> Result<InputRows> {
    let mut token_ids = Vec::with_capacity(activations.rows.len());
    let mut rows = Vec::with_capacity(activations.rows.len());
    for (row_idx, row) in activations.rows.iter().enumerate() {
        let values = unpack_page_i32s(&row.values)?;
        if values.len() != hidden {
            bail!(
                "activation row {row_idx} has {} values, expected hidden_size {hidden}",
                values.len()
            );
        }
        token_ids.push(row.token_id);
        rows.push(values);
    }
    Ok(InputRows {
        token_ids,
        slab: Slab::from_rows(rows)?,
    })
}

fn project_tokens(
    rows: &InputRows,
    weights: &LayerWeights,
    params: &LayerParams,
) -> Result<(Vec<QueryState>, Vec<KeyRow>)> {
    let norm_input = unpack_page_i32s(&params.norm_input)?;
    let q_norm = unpack_page_i32s(&params.q_norm)?;
    let k_norm = unpack_page_i32s(&params.k_norm)?;
    let hidden = params.hidden_size as usize;
    let normed = normed_input_slab(rows, &norm_input, params.norm_eps, hidden)?;
    let mut q = linear_slab(&normed, &weights.w_q)?;
    let mut k = linear_slab(&normed, &weights.w_k)?;
    let mut v = linear_slab(&normed, &weights.w_v)?;

    let head_dim = params.head_dim as usize;
    for row_idx in 0..rows.rows() {
        finish_qkv_row(
            row_idx as u32,
            q.row_mut(row_idx),
            k.row_mut(row_idx),
            v.row_mut(row_idx),
            &q_norm,
            &k_norm,
            params,
        )?;
    }

    let mut queries = Vec::with_capacity(rows.rows());
    let mut keys = Vec::with_capacity(rows.rows());
    for row_idx in 0..rows.rows() {
        let position = row_idx as u32;
        let q_row = q.row(row_idx).to_vec();
        let k_row = k.row(row_idx).to_vec();
        let v_row = v.row(row_idx).to_vec();
        let expected_q = params.num_heads as usize * head_dim;
        let expected_kv = params.num_kv_heads as usize * head_dim;
        if q_row.len() != expected_q || k_row.len() != expected_kv || v_row.len() != expected_kv {
            bail!(
                "Q/K/V width mismatch at position {position}: q {} k {} v {}",
                q_row.len(),
                k_row.len(),
                v_row.len()
            );
        }
        queries.push(QueryState {
            position,
            token_id: rows.token_ids[row_idx],
            q: q_row,
            residual: rows.slab.row(row_idx).to_vec(),
        });
        keys.push(KeyRow {
            position,
            k: pack_i32_page(&k_row),
            v: pack_i32_page(&v_row),
        });
    }
    Ok((queries, keys))
}

fn normed_input_slab(
    rows: &InputRows,
    norm_input: &[i32],
    norm_eps: i64,
    hidden: usize,
) -> Result<Slab> {
    let mut normed = Slab::zeroed(rows.rows(), hidden);
    if crate::tensor::use_parallel() {
        normed
            .as_flat_mut()
            .par_chunks_mut(hidden.max(1))
            .enumerate()
            .try_for_each(|(row_idx, out)| {
                out.copy_from_slice(rows.slab.row(row_idx));
                rms_norm(out, norm_input, norm_eps).map_err(anyhow::Error::msg)
            })?;
    } else {
        for row_idx in 0..rows.rows() {
            let out = normed.row_mut(row_idx);
            out.copy_from_slice(rows.slab.row(row_idx));
            rms_norm(out, norm_input, norm_eps).map_err(anyhow::Error::msg)?;
        }
    }
    Ok(normed)
}

fn finish_qkv_row(
    position: u32,
    q: &mut [i32],
    k: &mut [i32],
    v: &mut [i32],
    q_norm: &[i32],
    k_norm: &[i32],
    params: &LayerParams,
) -> Result<()> {
    let head_dim = params.head_dim as usize;
    for head in q.chunks_mut(head_dim) {
        rms_norm(head, q_norm, params.norm_eps).map_err(anyhow::Error::msg)?;
        apply_rope(head, params, position);
    }
    for head in k.chunks_mut(head_dim) {
        rms_norm(head, k_norm, params.norm_eps).map_err(anyhow::Error::msg)?;
        apply_rope(head, params, position);
    }
    for head in v.chunks_mut(head_dim) {
        value_rms_norm(head, params.norm_eps).map_err(anyhow::Error::msg)?;
    }
    Ok(())
}

fn apply_rope(head: &mut [i32], params: &LayerParams, position: u32) {
    let rotary_dim = params.rotary_dim as usize;
    if rotary_dim == 0 || position == 0 {
        return;
    }
    let mut lanes: Vec<Act> = head.iter().map(|bits| Act::from_bits(*bits)).collect();
    rope_rotate_pairs_in_place(
        &mut lanes,
        rotary_dim,
        params.rope_freq_base_dim as usize,
        Acc::from_bits(params.rope_base),
        position as usize,
    );
    for (lane, rotated) in head.iter_mut().zip(lanes.iter()) {
        *lane = rotated.to_bits();
    }
}

fn attend_tokens(
    queries: &[QueryState],
    own_keys: &[KeyRow],
    donor_keys: &[KeyRow],
    ple_rows: &[PleRow],
    weights: &LayerWeights,
    params: &LayerParams,
) -> Result<Vec<ActivationRow>> {
    if crate::tensor::use_parallel() {
        queries
            .par_iter()
            .map(|query| attend_token(query, own_keys, donor_keys, ple_rows, weights, params))
            .collect()
    } else {
        queries
            .iter()
            .map(|query| attend_token(query, own_keys, donor_keys, ple_rows, weights, params))
            .collect()
    }
}

fn attend_token(
    query: &QueryState,
    own_keys: &[KeyRow],
    donor_keys: &[KeyRow],
    ple_rows: &[PleRow],
    weights: &LayerWeights,
    params: &LayerParams,
) -> Result<ActivationRow> {
    let active_keys = if params.kv_donor_layer >= 0 {
        donor_keys
    } else {
        own_keys
    };
    let context = attention_context(query, active_keys, params)?;
    let attn_proj = weights.w_o.matvec(&context)?;
    let residual = attn_residual(query, attn_proj, params)?;
    let ff_in = pre_ff_norm(&residual, params)?;
    let gate = weights.w_gate.matvec(&ff_in)?;
    let up = weights.w_up.matvec(&ff_in)?;
    let gated = gelu_mul(gate, up)?;
    let ff = weights.w_down.matvec(&gated)?;
    let xs = finish_mlp(&residual, &ff_in, ff, params)?;
    let ple_row = ple_rows
        .get(query.position as usize)
        .ok_or_else(|| anyhow::anyhow!("missing PLE row at position {}", query.position))?;
    let ple_gate = weights.ple_input_gate.matvec(&xs)?;
    let ple_gated = ple_gate_mul(ple_gate, ple_row)?;
    let projected = weights.ple_layer_projection.matvec(&ple_gated)?;
    let values = finish_layer(&xs, projected, params)?;
    Ok(ActivationRow {
        token_id: query.token_id,
        values: pack_i32_page(&values),
    })
}

fn attention_context(
    query: &QueryState,
    keys: &[KeyRow],
    params: &LayerParams,
) -> Result<Vec<i32>> {
    let heads = params.num_heads as usize;
    let kv_heads = params.num_kv_heads as usize;
    let head_dim = params.head_dim as usize;
    let group = heads / kv_heads.max(1);
    let mut visible = Vec::new();
    let mut scores = Vec::new();
    for (index, key) in keys.iter().enumerate() {
        if !is_visible(query.position, key.position, params.sliding_window) {
            continue;
        }
        let k = unpack_page_i32s(&key.k)?;
        if k.len() != kv_heads * head_dim {
            bail!(
                "attention key width mismatch at position {}: got {}, expected {}",
                key.position,
                k.len(),
                kv_heads * head_dim
            );
        }
        for head in 0..heads {
            let kv_head = head / group.max(1);
            let q_head = &query.q[head * head_dim..(head + 1) * head_dim];
            let k_head = &k[kv_head * head_dim..(kv_head + 1) * head_dim];
            scores.push(dot_bits(q_head, k_head));
        }
        visible.push((index, key));
    }
    if visible.is_empty() {
        bail!("attention saw no keys at or before this token");
    }

    let count = visible.len();
    let mut weights = scores.clone();
    let mut logits = Vec::with_capacity(count);
    for head in 0..heads {
        logits.clear();
        for key in 0..count {
            logits.push(Act::from_bits(scores[key * heads + head]));
        }
        let head_weights = attention_softmax(&logits);
        for (key, weight) in head_weights.iter().enumerate() {
            weights[key * heads + head] = weight.to_bits();
        }
    }

    let start = window_start(query.position, params.sliding_window) as usize;
    let mut acc = vec![0_i64; heads * head_dim];
    for (list_index, key) in &visible {
        let slot = list_index
            .checked_sub(start)
            .ok_or_else(|| anyhow::anyhow!("visible key precedes the attention window"))?;
        if slot >= count {
            bail!("visible key falls outside the weight list");
        }
        let v = unpack_page_i32s(&key.v)?;
        if v.len() != kv_heads * head_dim {
            bail!(
                "attention value width mismatch at position {}: got {}, expected {}",
                key.position,
                v.len(),
                kv_heads * head_dim
            );
        }
        for head in 0..heads {
            let kv_head = head / group.max(1);
            let weight = weights[slot * heads + head];
            let v_head = &v[kv_head * head_dim..(kv_head + 1) * head_dim];
            for (lane, value) in v_head.iter().enumerate() {
                mac_weighted_value(&mut acc[head * head_dim + lane], weight, *value);
            }
        }
    }
    Ok(acc.into_iter().map(requantize_acc).collect())
}

fn window_start(query_position: u32, sliding_window: u32) -> u32 {
    if sliding_window == 0 {
        0
    } else {
        query_position
            .saturating_add(1)
            .saturating_sub(sliding_window)
    }
}

fn is_visible(query_position: u32, key_position: u32, sliding_window: u32) -> bool {
    key_position <= query_position
        && (sliding_window == 0 || query_position - key_position < sliding_window)
}

fn attn_residual(
    query: &QueryState,
    mut attn_proj: Vec<i32>,
    params: &LayerParams,
) -> Result<Vec<i32>> {
    rms_norm(
        &mut attn_proj,
        &unpack_page_i32s(&params.norm_post_attn)?,
        params.norm_eps,
    )
    .map_err(anyhow::Error::msg)?;
    add_row(&mut attn_proj, &query.residual).map_err(anyhow::Error::msg)?;
    Ok(attn_proj)
}

fn pre_ff_norm(residual: &[i32], params: &LayerParams) -> Result<Vec<i32>> {
    let mut normed = residual.to_vec();
    rms_norm(
        &mut normed,
        &unpack_page_i32s(&params.norm_pre_ffw)?,
        params.norm_eps,
    )
    .map_err(anyhow::Error::msg)?;
    Ok(normed)
}

fn gelu_mul(mut gate: Vec<i32>, up: Vec<i32>) -> Result<Vec<i32>> {
    if gate.len() != up.len() {
        bail!("gate/up width mismatch: {} vs {}", gate.len(), up.len());
    }
    for (value, up_value) in gate.iter_mut().zip(&up) {
        *value = ::prefill_range::input::mul(gelu(*value), *up_value);
    }
    Ok(gate)
}

fn finish_mlp(
    residual: &[i32],
    _ff_in: &[i32],
    mut ff: Vec<i32>,
    params: &LayerParams,
) -> Result<Vec<i32>> {
    rms_norm(
        &mut ff,
        &unpack_page_i32s(&params.norm_post_ffw)?,
        params.norm_eps,
    )
    .map_err(anyhow::Error::msg)?;
    add_row(&mut ff, residual).map_err(anyhow::Error::msg)?;
    Ok(ff)
}

fn ple_gate_mul(mut gate: Vec<i32>, per_layer_input: &PleRow) -> Result<Vec<i32>> {
    let ple = unpack_page_i32s(&per_layer_input.values)?;
    if gate.len() != ple.len() {
        bail!(
            "per-layer input has {} values, gate has {}",
            ple.len(),
            gate.len()
        );
    }
    for (slot, embedded) in gate.iter_mut().zip(ple.iter()) {
        *slot = ::prefill_range::input::mul(gelu(*slot), *embedded);
    }
    Ok(gate)
}

fn finish_layer(xs: &[i32], mut projected: Vec<i32>, params: &LayerParams) -> Result<Vec<i32>> {
    rms_norm(
        &mut projected,
        &unpack_page_i32s(&params.ple_post_norm)?,
        params.norm_eps,
    )
    .map_err(anyhow::Error::msg)?;
    add_row(&mut projected, xs).map_err(anyhow::Error::msg)?;
    if params.layer_scalar != 0 {
        scale_row(&mut projected, params.layer_scalar);
    }
    Ok(projected)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ::prefill_range::input::{pack_i32s, LayerParams};

    fn tiny_params() -> LayerParams {
        LayerParams {
            layer_idx: 0,
            hidden_size: 2,
            ffn_size: 2,
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: 2,
            sliding_window: 0,
            attn_scale: 0,
            layer_scalar: 0,
            norm_eps: 0,
            rope_base: 1_i64 << 32,
            rotary_dim: 0,
            rope_freq_base_dim: 2,
            kv_donor_layer: -1,
            norm_input: pack_i32s(&[1 << 16, 1 << 16]),
            norm_post_attn: pack_i32s(&[1 << 16, 1 << 16]),
            norm_pre_ffw: pack_i32s(&[1 << 16, 1 << 16]),
            norm_post_ffw: pack_i32s(&[1 << 16, 1 << 16]),
            q_norm: pack_i32s(&[1 << 16, 1 << 16]),
            k_norm: pack_i32s(&[1 << 16, 1 << 16]),
            ple_width: 2,
            ple_post_norm: pack_i32s(&[1 << 16, 1 << 16]),
        }
    }

    #[test]
    fn rejects_invalid_head_grouping() {
        let mut params = tiny_params();
        params.num_heads = 3;
        params.num_kv_heads = 2;
        assert!(validate_layer_params(&params)
            .unwrap_err()
            .to_string()
            .contains("do not group evenly"));
    }

    #[test]
    fn sliding_visibility_matches_wip_predicate() {
        assert!(is_visible(3, 2, 2));
        assert!(!is_visible(3, 1, 2));
        assert!(!is_visible(3, 4, 0));
        assert!(is_visible(3, 0, 0));
    }
}
