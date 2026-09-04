use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use inference_artifacts::{
    read_json, DirectInferManifest, DirectInferShape, DIRECT_INFER_ARTIFACTS_DIR,
    DIRECT_INFER_MANIFEST_JSON,
};
use prompt_prepare::input::{merge_bucket_of, vocab_bucket_of, BpeMerge, BpePieces, MergeBucket};
use prompt_prepare::input::{PromptTokenizer, TokenEntry, VocabBucket};
use raster::{Bytes, List};

use crate::detwgt::{DetwgtMatrixView, MmapDetwgt};

const ONE: i32 = 1 << 16;
const END_OF_WORD: &str = "</w>";
const BOS_TOKEN: &str = "<bos>";
const TURN_OPEN: &str = "<|turn>";
const TURN_CLOSE: &str = "<turn|>";
const NEWLINE_TOKEN: &str = "\n";

pub const MISSING_DIRECT_MANIFEST: &str =
    "run raster-inference model import ... to generate direct-infer artifacts";

pub struct DirectInferenceModel {
    pub manifest: DirectInferManifest,
    pub manifest_path: PathBuf,
    tokenizer: serde_json::Value,
    config_text: serde_json::Value,
    detwgt: MmapDetwgt,
    ple_params: Vec<prefill_prepare_aux::input::PleLayerParams>,
    layer_params: Vec<prefill_range::input::LayerParams>,
    final_head_params: prefill_finalize::input::FinalHeadParams,
}

pub struct DirectPleLayer<'a> {
    pub params: prefill_prepare_aux::input::PleLayerParams,
    pub embeddings: DetwgtMatrixView<'a>,
    pub embedding_start: usize,
    pub projection: DetwgtMatrixView<'a>,
}

pub struct DirectTransformerLayer<'a> {
    pub params: prefill_range::input::LayerParams,
    pub w_q: DetwgtMatrixView<'a>,
    pub w_k: DetwgtMatrixView<'a>,
    pub w_v: DetwgtMatrixView<'a>,
    pub w_o: DetwgtMatrixView<'a>,
    pub w_gate: DetwgtMatrixView<'a>,
    pub w_up: DetwgtMatrixView<'a>,
    pub w_down: DetwgtMatrixView<'a>,
    pub ple_input_gate: DetwgtMatrixView<'a>,
    pub ple_layer_projection: DetwgtMatrixView<'a>,
}

pub struct DirectFinalHead<'a> {
    pub params: prefill_finalize::input::FinalHeadParams,
    pub projection: DetwgtMatrixView<'a>,
}

impl DirectInferenceModel {
    pub fn load(manifest_path: &Path) -> Result<Self> {
        if !manifest_path.is_file() {
            bail!(
                "missing direct-infer manifest {}; {MISSING_DIRECT_MANIFEST}",
                manifest_path.display()
            );
        }
        let manifest: DirectInferManifest = read_json(manifest_path)?;
        if manifest.version != 1 {
            bail!("unsupported direct-infer manifest v{}", manifest.version);
        }

        let manifest_dir = manifest_path
            .parent()
            .ok_or_else(|| anyhow::anyhow!("direct manifest has no parent directory"))?;
        let config_path = resolve_manifest_path(manifest_dir, &manifest.bundle.config_path);
        let tokenizer_path = resolve_manifest_path(manifest_dir, &manifest.bundle.tokenizer_path);
        let detwgt_path = resolve_manifest_path(manifest_dir, &manifest.bundle.model_detwgt_path);

        let config: serde_json::Value = read_json_file(&config_path)?;
        let tokenizer: serde_json::Value = read_json_file(&tokenizer_path)?;
        let config_text = config
            .get("text_config")
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("config.json has no text_config"))?;
        let detwgt = MmapDetwgt::open(&detwgt_path)?;
        let ple_params = build_ple_params(&manifest.shape, &detwgt)?;
        let layer_params = build_layer_params(&manifest.shape, &detwgt)?;
        let final_head_params = build_final_head_params(&manifest.shape, &detwgt)?;

        Ok(Self {
            manifest,
            manifest_path: manifest_path.to_path_buf(),
            tokenizer,
            config_text,
            detwgt,
            ple_params,
            layer_params,
            final_head_params,
        })
    }

    pub fn default_manifest_path(base_dir: &Path) -> PathBuf {
        base_dir
            .join(DIRECT_INFER_ARTIFACTS_DIR)
            .join(DIRECT_INFER_MANIFEST_JSON)
    }

    pub fn prompt_inputs(&self) -> Result<prompt_prepare::input::PromptTokenization> {
        let tokenizer = self.prompt_tokenizer()?;
        let pieces = BpePieces {
            pieces: List::from(self.manifest.prompt.initial_pieces.clone()),
        };
        staged_infer::kernels::prompt_prepare::run_prompt_prepare_direct(
            staged_infer::kernels::prompt_prepare::PromptPrepareDirectInputs {
                tokenizer: &tokenizer,
                initial_pieces: &pieces,
            },
        )
    }

    pub fn embedding_view(&self) -> Result<DetwgtMatrixView<'_>> {
        self.detwgt.matrix(
            "model.language_model.embed_tokens.weight",
            self.shape().vocab_size as usize,
            self.shape().hidden_size as usize,
        )
    }

    pub fn scaled_embedding_row(&self, token_id: u32) -> Result<Vec<i32>> {
        let row = self.embedding_view()?.row_values(token_id as usize)?;
        let scale = det_num::Act::from_bits(self.shape().embedding_scale);
        Ok(row
            .into_iter()
            .map(|bits| det_num::ops::mul_sat(det_num::Act::from_bits(bits), scale).to_bits())
            .collect())
    }

    pub fn embedding_table(&self) -> Result<input_embedding::input::EmbeddingTable> {
        let mut values = Vec::with_capacity(
            self.shape().vocab_size as usize * self.shape().hidden_size as usize,
        );
        for token_id in 0..self.shape().vocab_size as usize {
            values.extend(
                self.detwgt
                    .row("model.language_model.embed_tokens.weight", token_id)?,
            );
        }
        Ok(input_embedding::input::EmbeddingTable {
            hidden_size: self.shape().hidden_size,
            embedding_scale: self.shape().embedding_scale,
            values: paged_i32s(&values)?,
        })
    }

    pub fn ple_layer(&self, layer_idx: usize) -> Result<prefill_prepare_aux::input::PleLayer> {
        let shape = self.shape();
        let start = layer_idx * shape.hidden_size_per_layer_input as usize;
        let end = start + shape.hidden_size_per_layer_input as usize;
        let embeddings = self.detwgt.column_slice(
            "model.language_model.embed_tokens_per_layer.weight",
            start,
            end,
        )?;
        let projection = self.detwgt.rows(
            "model.language_model.per_layer_model_projection.weight",
            start,
            end,
        )?;
        Ok(prefill_prepare_aux::input::PleLayer {
            params: prefill_prepare_aux::input::PleLayerParams {
                layer_idx: layer_idx as u32,
                hidden_size: shape.hidden_size,
                ple_width: shape.hidden_size_per_layer_input,
                embedding_scale: shape.ple_embedding_scale,
                projection_scalar: shape.ple_projection_scalar,
                input_scale: shape.ple_input_scale,
                norm_eps: shape.norm_eps,
                norm_weights: pack_i32_page(
                    &self
                        .detwgt
                        .values("model.language_model.per_layer_projection_norm.weight")?,
                ),
            },
            embeddings: paged_i32s(&embeddings)?,
            projection: paged_i32s(&projection)?,
        })
    }

    pub fn direct_ple_layer(&self, layer_idx: usize) -> Result<DirectPleLayer<'_>> {
        let shape = self.shape();
        let embedding_start = layer_idx * shape.hidden_size_per_layer_input as usize;
        let embedding_width = shape.hidden_size_per_layer_input as usize;
        let embedding_cols = shape.num_hidden_layers as usize * embedding_width;
        let projection = self.detwgt.matrix_rows(
            "model.language_model.per_layer_model_projection.weight",
            embedding_start,
            embedding_start + embedding_width,
            shape.hidden_size as usize,
        )?;
        Ok(DirectPleLayer {
            params: self.ple_params[layer_idx].clone(),
            embeddings: self.detwgt.matrix(
                "model.language_model.embed_tokens_per_layer.weight",
                shape.vocab_size as usize,
                embedding_cols,
            )?,
            embedding_start,
            projection,
        })
    }

    pub fn transformer_layer(
        &self,
        layer_idx: usize,
    ) -> Result<prefill_range::input::TransformerLayer> {
        let shape = self.shape();
        let at = |suffix: &str| format!("model.language_model.layers.{layer_idx}.{suffix}");
        let gate_dims = self
            .detwgt
            .tensor(&at("mlp.gate_proj.weight"))?
            .dims
            .clone();
        let ffn = *gate_dims
            .first()
            .ok_or_else(|| anyhow::anyhow!("gate projection has no row dimension"))?
            as u32;
        let head_dim = self.head_dim_at(layer_idx);
        let (rope_base, rotary_dim, rope_freq_base_dim) = self.rope_at(layer_idx);
        let kv_donor_layer = self.kv_donor_layer(layer_idx);
        let (donor_a_layer, donor_b_layer) = self.donor_candidates();
        let w_v = self
            .detwgt
            .values(&at("self_attn.v_proj.weight"))
            .or_else(|_| self.detwgt.values(&at("self_attn.k_proj.weight")))?;

        Ok(prefill_range::input::TransformerLayer {
            params: prefill_range::input::LayerParams {
                layer_idx: layer_idx as u32,
                hidden_size: shape.hidden_size,
                ffn_size: ffn,
                num_heads: shape.num_attention_heads,
                num_kv_heads: shape.num_key_value_heads,
                head_dim,
                sliding_window: if self.is_sliding(layer_idx) {
                    shape.sliding_window
                } else {
                    0
                },
                attn_scale: inv_sqrt_q16(head_dim as usize),
                layer_scalar: self
                    .detwgt
                    .values(&at("layer_scalar"))
                    .ok()
                    .and_then(|values| values.first().copied())
                    .unwrap_or(0),
                norm_eps: shape.norm_eps,
                rope_base,
                rotary_dim,
                rope_freq_base_dim,
                kv_donor_layer,
                donor_a_layer,
                donor_b_layer,
                norm_input: pack_i32_page(&self.detwgt.values(&at("input_layernorm.weight"))?),
                norm_post_attn: pack_i32_page(
                    &self.detwgt.values(&at("post_attention_layernorm.weight"))?,
                ),
                norm_pre_ffw: pack_i32_page(
                    &self
                        .detwgt
                        .values(&at("pre_feedforward_layernorm.weight"))?,
                ),
                norm_post_ffw: pack_i32_page(
                    &self
                        .detwgt
                        .values(&at("post_feedforward_layernorm.weight"))?,
                ),
                q_norm: pack_i32_page(&self.detwgt.values(&at("self_attn.q_norm.weight"))?),
                k_norm: pack_i32_page(&self.detwgt.values(&at("self_attn.k_norm.weight"))?),
                ple_width: shape.hidden_size_per_layer_input,
                ple_post_norm: pack_i32_page(
                    &self
                        .detwgt
                        .values(&at("post_per_layer_input_norm.weight"))?,
                ),
            },
            w_q: paged_i32s(&self.detwgt.values(&at("self_attn.q_proj.weight"))?)?,
            w_k: paged_i32s(&self.detwgt.values(&at("self_attn.k_proj.weight"))?)?,
            w_v: paged_i32s(&w_v)?,
            w_o: paged_i32s(&self.detwgt.values(&at("self_attn.o_proj.weight"))?)?,
            w_gate: paged_i32s(&self.detwgt.values(&at("mlp.gate_proj.weight"))?)?,
            w_up: paged_i32s(&self.detwgt.values(&at("mlp.up_proj.weight"))?)?,
            w_down: paged_i32s(&self.detwgt.values(&at("mlp.down_proj.weight"))?)?,
            ple_input_gate: paged_i32s(&self.detwgt.values(&at("per_layer_input_gate.weight"))?)?,
            ple_layer_projection: paged_i32s(
                &self.detwgt.values(&at("per_layer_projection.weight"))?,
            )?,
        })
    }

    pub fn direct_transformer_layer(&self, layer_idx: usize) -> Result<DirectTransformerLayer<'_>> {
        let shape = self.shape();
        let at = |suffix: &str| format!("model.language_model.layers.{layer_idx}.{suffix}");
        let params = self.layer_params[layer_idx].clone();
        let ffn = params.ffn_size;
        let head_dim = params.head_dim;
        let q_len = shape.num_attention_heads as usize * head_dim as usize;
        let kv_len = shape.num_key_value_heads as usize * head_dim as usize;
        let hidden = shape.hidden_size as usize;
        let ple_width = shape.hidden_size_per_layer_input as usize;
        let v_name = at("self_attn.v_proj.weight");
        let k_name = at("self_attn.k_proj.weight");
        let v_tensor = if self.detwgt.tensor(&v_name).is_ok() {
            v_name
        } else {
            k_name.clone()
        };

        Ok(DirectTransformerLayer {
            params,
            w_q: self
                .detwgt
                .matrix(&at("self_attn.q_proj.weight"), q_len, hidden)?,
            w_k: self.detwgt.matrix(&k_name, kv_len, hidden)?,
            w_v: self.detwgt.matrix(&v_tensor, kv_len, hidden)?,
            w_o: self
                .detwgt
                .matrix(&at("self_attn.o_proj.weight"), hidden, q_len)?,
            w_gate: self
                .detwgt
                .matrix(&at("mlp.gate_proj.weight"), ffn as usize, hidden)?,
            w_up: self
                .detwgt
                .matrix(&at("mlp.up_proj.weight"), ffn as usize, hidden)?,
            w_down: self
                .detwgt
                .matrix(&at("mlp.down_proj.weight"), hidden, ffn as usize)?,
            ple_input_gate: self.detwgt.matrix(
                &at("per_layer_input_gate.weight"),
                ple_width,
                hidden,
            )?,
            ple_layer_projection: self.detwgt.matrix(
                &at("per_layer_projection.weight"),
                hidden,
                ple_width,
            )?,
        })
    }

    pub fn final_head(&self) -> Result<prefill_finalize::input::FinalHead> {
        let shape = self.shape();
        let tied = self
            .config_text
            .get("tie_word_embeddings")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        let tensor = if tied {
            "model.language_model.embed_tokens.weight"
        } else {
            "model.language_model.lm_head.weight"
        };
        let mut rows = Vec::with_capacity(shape.vocab_size as usize * shape.hidden_size as usize);
        for token_id in 0..shape.vocab_size as usize {
            rows.extend(self.detwgt.row(tensor, token_id)?);
        }
        Ok(prefill_finalize::input::FinalHead {
            params: prefill_finalize::input::FinalHeadParams {
                hidden_size: shape.hidden_size,
                norm_eps: shape.norm_eps,
                softcap: shape.final_logit_softcap,
                norm_weights: pack_i32_page(
                    &self.detwgt.values("model.language_model.norm.weight")?,
                ),
            },
            projection: paged_i32s(&rows)?,
        })
    }

    pub fn direct_final_head(&self) -> Result<DirectFinalHead<'_>> {
        let shape = self.shape();
        let tied = self
            .config_text
            .get("tie_word_embeddings")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        let tensor = if tied {
            "model.language_model.embed_tokens.weight"
        } else {
            "model.language_model.lm_head.weight"
        };
        Ok(DirectFinalHead {
            params: self.final_head_params.clone(),
            projection: self.detwgt.matrix(
                tensor,
                shape.vocab_size as usize,
                shape.hidden_size as usize,
            )?,
        })
    }

    pub fn decoder_table(&self) -> Result<output_finalize::input::DecoderTable> {
        let model = self
            .tokenizer
            .get("model")
            .ok_or_else(|| anyhow::anyhow!("tokenizer.json has no model"))?;
        let vocab_map = vocab_map(model)?;
        let special_ids: BTreeSet<u32> = self
            .tokenizer
            .get("added_tokens")
            .and_then(serde_json::Value::as_array)
            .into_iter()
            .flatten()
            .filter(|entry| {
                entry
                    .get("special")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false)
            })
            .filter_map(|entry| entry.get("id").and_then(serde_json::Value::as_u64))
            .map(|id| id as u32)
            .collect();
        let eos_ids: BTreeSet<u32> = self.manifest.prompt.eos_token_ids.iter().copied().collect();
        let max_token_id = vocab_map.values().copied().max().unwrap_or(0);
        let mut tokens =
            vec![output_finalize::input::DecoderToken::default(); max_token_id as usize + 1];
        for (token, id) in vocab_map {
            tokens[id as usize] = output_finalize::input::DecoderToken {
                token,
                special: special_ids.contains(&id),
                terminal: eos_ids.contains(&id),
            };
        }
        Ok(output_finalize::input::DecoderTable {
            tokens: List::from(tokens),
        })
    }

    pub fn shape(&self) -> &DirectInferShape {
        &self.manifest.shape
    }

    fn prompt_tokenizer(&self) -> Result<PromptTokenizer> {
        let model = self
            .tokenizer
            .get("model")
            .ok_or_else(|| anyhow::anyhow!("tokenizer.json has no model"))?;
        let vocab_map = vocab_map(model)?;
        let mut vocab: Vec<TokenEntry> = vocab_map
            .into_iter()
            .map(|(token, id)| TokenEntry { token, id })
            .collect();
        vocab.sort_by_key(|entry| entry.id);
        let merges = model
            .get("merges")
            .and_then(serde_json::Value::as_array)
            .map(|merges| {
                merges
                    .iter()
                    .enumerate()
                    .filter_map(|(rank, entry)| parse_merge(rank as u32, entry))
                    .collect()
            })
            .unwrap_or_default();
        let (vocab_bucket_count, vocab_buckets) = bucket_vocab(vocab);
        let (merge_bucket_count, merge_buckets) = bucket_merges(merges);
        Ok(PromptTokenizer {
            vocab_bucket_count,
            merge_bucket_count,
            vocab_buckets: List::from(vocab_buckets),
            merge_buckets: List::from(merge_buckets),
        })
    }

    fn is_sliding(&self, idx: usize) -> bool {
        matches!(
            self.shape().layer_types.get(idx).map(String::as_str),
            Some("sliding_attention")
        )
    }

    fn head_dim_at(&self, idx: usize) -> u32 {
        if self.is_sliding(idx) {
            self.shape().head_dim
        } else {
            self.shape().global_head_dim
        }
    }

    fn rope_at(&self, idx: usize) -> (i64, u32, u32) {
        let head_dim = self.head_dim_at(idx);
        if self.is_sliding(idx) {
            (self.shape().rope_base_sliding, head_dim, head_dim)
        } else {
            let rotary = ((head_dim as i64 * self.shape().full_partial_rotary_factor_q16 as i64)
                / ONE as i64) as u32;
            (self.shape().rope_base_full, rotary, head_dim)
        }
    }

    fn kv_donor_layer(&self, idx: usize) -> i32 {
        let first_shared = (self.shape().num_hidden_layers as usize)
            .saturating_sub(self.shape().num_kv_shared_layers as usize);
        if self.shape().num_kv_shared_layers == 0 || idx < first_shared {
            return -1;
        }
        let Some(attention_type) = self.shape().layer_types.get(idx) else {
            return -1;
        };
        self.shape().layer_types[..first_shared]
            .iter()
            .rposition(|candidate| candidate == attention_type)
            .map(|donor| donor as i32)
            .unwrap_or(-1)
    }

    fn donor_candidates(&self) -> (i32, i32) {
        let mut donors = (0..self.shape().num_hidden_layers as usize)
            .map(|layer| self.kv_donor_layer(layer))
            .filter(|donor| *donor >= 0)
            .collect::<Vec<_>>();
        donors.sort_unstable();
        donors.dedup();
        (
            donors.first().copied().unwrap_or(-2),
            donors.get(1).copied().unwrap_or(-3),
        )
    }
}

fn build_ple_params(
    shape: &DirectInferShape,
    detwgt: &MmapDetwgt,
) -> Result<Vec<prefill_prepare_aux::input::PleLayerParams>> {
    if shape.num_hidden_layers == 0 {
        return Ok(Vec::new());
    }
    let norm_weights =
        pack_i32_page(&detwgt.values("model.language_model.per_layer_projection_norm.weight")?);
    Ok((0..shape.num_hidden_layers)
        .map(|layer_idx| prefill_prepare_aux::input::PleLayerParams {
            layer_idx,
            hidden_size: shape.hidden_size,
            ple_width: shape.hidden_size_per_layer_input,
            embedding_scale: shape.ple_embedding_scale,
            projection_scalar: shape.ple_projection_scalar,
            input_scale: shape.ple_input_scale,
            norm_eps: shape.norm_eps,
            norm_weights: norm_weights.clone(),
        })
        .collect())
}

fn build_layer_params(
    shape: &DirectInferShape,
    detwgt: &MmapDetwgt,
) -> Result<Vec<prefill_range::input::LayerParams>> {
    let (donor_a_layer, donor_b_layer) = donor_candidates(shape);
    (0..shape.num_hidden_layers as usize)
        .map(|layer_idx| {
            let at = |suffix: &str| format!("model.language_model.layers.{layer_idx}.{suffix}");
            let gate_dims = detwgt.tensor(&at("mlp.gate_proj.weight"))?.dims.clone();
            let ffn = *gate_dims
                .first()
                .ok_or_else(|| anyhow::anyhow!("gate projection has no row dimension"))?
                as u32;
            let head_dim = head_dim_at(shape, layer_idx);
            let (rope_base, rotary_dim, rope_freq_base_dim) = rope_at(shape, layer_idx);
            Ok(prefill_range::input::LayerParams {
                layer_idx: layer_idx as u32,
                hidden_size: shape.hidden_size,
                ffn_size: ffn,
                num_heads: shape.num_attention_heads,
                num_kv_heads: shape.num_key_value_heads,
                head_dim,
                sliding_window: if is_sliding(shape, layer_idx) {
                    shape.sliding_window
                } else {
                    0
                },
                attn_scale: inv_sqrt_q16(head_dim as usize),
                layer_scalar: detwgt
                    .values(&at("layer_scalar"))
                    .ok()
                    .and_then(|values| values.first().copied())
                    .unwrap_or(0),
                norm_eps: shape.norm_eps,
                rope_base,
                rotary_dim,
                rope_freq_base_dim,
                kv_donor_layer: kv_donor_layer(shape, layer_idx),
                donor_a_layer,
                donor_b_layer,
                norm_input: pack_i32_page(&detwgt.values(&at("input_layernorm.weight"))?),
                norm_post_attn: pack_i32_page(
                    &detwgt.values(&at("post_attention_layernorm.weight"))?,
                ),
                norm_pre_ffw: pack_i32_page(
                    &detwgt.values(&at("pre_feedforward_layernorm.weight"))?,
                ),
                norm_post_ffw: pack_i32_page(
                    &detwgt.values(&at("post_feedforward_layernorm.weight"))?,
                ),
                q_norm: pack_i32_page(&detwgt.values(&at("self_attn.q_norm.weight"))?),
                k_norm: pack_i32_page(&detwgt.values(&at("self_attn.k_norm.weight"))?),
                ple_width: shape.hidden_size_per_layer_input,
                ple_post_norm: pack_i32_page(
                    &detwgt.values(&at("post_per_layer_input_norm.weight"))?,
                ),
            })
        })
        .collect()
}

fn build_final_head_params(
    shape: &DirectInferShape,
    detwgt: &MmapDetwgt,
) -> Result<prefill_finalize::input::FinalHeadParams> {
    Ok(prefill_finalize::input::FinalHeadParams {
        hidden_size: shape.hidden_size,
        norm_eps: shape.norm_eps,
        softcap: shape.final_logit_softcap,
        norm_weights: pack_i32_page(&detwgt.values("model.language_model.norm.weight")?),
    })
}

fn is_sliding(shape: &DirectInferShape, idx: usize) -> bool {
    matches!(
        shape.layer_types.get(idx).map(String::as_str),
        Some("sliding_attention")
    )
}

fn head_dim_at(shape: &DirectInferShape, idx: usize) -> u32 {
    if is_sliding(shape, idx) {
        shape.head_dim
    } else {
        shape.global_head_dim
    }
}

fn rope_at(shape: &DirectInferShape, idx: usize) -> (i64, u32, u32) {
    let head_dim = head_dim_at(shape, idx);
    if is_sliding(shape, idx) {
        (shape.rope_base_sliding, head_dim, head_dim)
    } else {
        let rotary =
            ((head_dim as i64 * shape.full_partial_rotary_factor_q16 as i64) / ONE as i64) as u32;
        (shape.rope_base_full, rotary, head_dim)
    }
}

fn kv_donor_layer(shape: &DirectInferShape, idx: usize) -> i32 {
    let first_shared =
        (shape.num_hidden_layers as usize).saturating_sub(shape.num_kv_shared_layers as usize);
    if shape.num_kv_shared_layers == 0 || idx < first_shared {
        return -1;
    }
    let Some(attention_type) = shape.layer_types.get(idx) else {
        return -1;
    };
    shape.layer_types[..first_shared]
        .iter()
        .rposition(|candidate| candidate == attention_type)
        .map(|donor| donor as i32)
        .unwrap_or(-1)
}

fn donor_candidates(shape: &DirectInferShape) -> (i32, i32) {
    let mut donors = (0..shape.num_hidden_layers as usize)
        .map(|layer| kv_donor_layer(shape, layer))
        .filter(|donor| *donor >= 0)
        .collect::<Vec<_>>();
    donors.sort_unstable();
    donors.dedup();
    (
        donors.first().copied().unwrap_or(-2),
        donors.get(1).copied().unwrap_or(-3),
    )
}

fn read_json_file(path: &Path) -> Result<serde_json::Value> {
    let bytes =
        std::fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("failed to parse {}", path.display()))
}

fn resolve_manifest_path(manifest_dir: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        manifest_dir.join(path)
    }
}

fn vocab_map(model: &serde_json::Value) -> Result<BTreeMap<String, u32>> {
    model
        .get("vocab")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| anyhow::anyhow!("tokenizer.json has no vocab"))?
        .iter()
        .map(|(token, id)| {
            Ok((
                token.clone(),
                id.as_u64()
                    .ok_or_else(|| anyhow::anyhow!("token '{token}' has non-integer id"))?
                    as u32,
            ))
        })
        .collect()
}

fn bucket_vocab(vocab: Vec<TokenEntry>) -> (u32, Vec<VocabBucket>) {
    let count = bucket_count_for(vocab.len());
    let mut buckets: Vec<Vec<TokenEntry>> = vec![Vec::new(); count as usize];
    for entry in vocab {
        buckets[vocab_bucket_of(&entry.token, count) as usize].push(entry);
    }
    (
        count,
        buckets
            .into_iter()
            .map(|entries| VocabBucket {
                entries: List::from(entries),
            })
            .collect(),
    )
}

fn bucket_merges(merges: Vec<BpeMerge>) -> (u32, Vec<MergeBucket>) {
    let count = bucket_count_for(merges.len());
    let mut buckets: Vec<Vec<BpeMerge>> = vec![Vec::new(); count as usize];
    for rule in merges {
        buckets[merge_bucket_of(&rule.left, &rule.right, count) as usize].push(rule);
    }
    for bucket in &mut buckets {
        bucket.sort_by_key(|rule| rule.rank);
    }
    (
        count,
        buckets
            .into_iter()
            .map(|rules| MergeBucket {
                rules: List::from(rules),
            })
            .collect(),
    )
}

fn bucket_count_for(len: usize) -> u32 {
    (len / 4).max(1) as u32
}

fn parse_merge(rank: u32, entry: &serde_json::Value) -> Option<BpeMerge> {
    let (left, right) = match entry {
        serde_json::Value::String(text) => {
            let mut parts = text.splitn(2, ' ');
            (parts.next()?.to_string(), parts.next()?.to_string())
        }
        serde_json::Value::Array(pair) if pair.len() == 2 => {
            (pair[0].as_str()?.to_string(), pair[1].as_str()?.to_string())
        }
        _ => return None,
    };
    let merged = format!("{left}{right}");
    Some(BpeMerge {
        rank,
        left,
        right,
        merged,
    })
}

fn inv_sqrt_q16(n: usize) -> i32 {
    if n == 0 {
        return 0;
    }
    let scaled = (1.0 / (n as f64).sqrt()) * ONE as f64;
    scaled.round().clamp(i32::MIN as f64, i32::MAX as f64) as i32
}

fn pack_i32_bytes(values: &[i32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn pack_i32_page(values: &[i32]) -> raster::BytesPage {
    raster::BytesPage::__from_parts(0, 0, pack_i32_bytes(values))
}

fn paged_i32s(values: &[i32]) -> Result<Bytes<196_608>> {
    Bytes::<196_608>::paged(pack_i32_bytes(values)).map_err(|error| anyhow::anyhow!("{error}"))
}

#[allow(dead_code)]
fn render_gemma_turns(prompt: &str) -> String {
    format!(
        "{BOS_TOKEN}{TURN_OPEN}user\n{}{TURN_CLOSE}\n{TURN_OPEN}model\n",
        prompt.trim()
    )
}

#[allow(dead_code)]
fn supports_gemma_turns(vocab: &BTreeMap<String, u32>) -> bool {
    [BOS_TOKEN, TURN_OPEN, TURN_CLOSE, NEWLINE_TOKEN]
        .iter()
        .all(|token| vocab.contains_key(*token))
}

#[allow(dead_code)]
fn split_prompt(prompt: &str, vocab: &BTreeMap<String, u32>, specials: &[String]) -> Vec<String> {
    let mut pieces = Vec::new();
    let mut rest = prompt;
    while !rest.is_empty() {
        if let Some(special) = specials
            .iter()
            .find(|token| rest.starts_with(token.as_str()))
        {
            pieces.push(special.clone());
            rest = &rest[special.len()..];
            continue;
        }
        let ch = rest.chars().next().expect("rest is non-empty");
        rest = &rest[ch.len_utf8()..];
        let piece = if ch == ' ' {
            String::from('\u{2581}')
        } else {
            ch.to_string()
        };
        if vocab.contains_key(&piece) {
            pieces.push(piece);
        } else {
            for byte in piece.as_bytes() {
                pieces.push(format!("<0x{byte:02X}>"));
            }
        }
    }
    pieces.push(END_OF_WORD.to_string());
    pieces
}
