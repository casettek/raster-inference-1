use std::path::PathBuf;

use serde::{Deserialize, Serialize};

pub const MODEL_ARTIFACTS_DIR: &str = "model-artifacts";
pub const MODEL_MANIFEST_JSON: &str = "manifest.json";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelManifest {
    pub version: u32,
    pub bundle: DirectInferBundle,
    pub shape: DirectInferShape,
    pub eos_token_ids: Vec<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provenance: Option<DirectInferProvenance>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DirectInferBundle {
    pub model_detwgt_path: PathBuf,
    pub model_detwgt_sha256: String,
    pub config_path: PathBuf,
    pub config_sha256: String,
    pub tokenizer_path: PathBuf,
    pub tokenizer_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DirectInferShape {
    pub hidden_size: u32,
    pub num_hidden_layers: u32,
    pub num_attention_heads: u32,
    pub num_key_value_heads: u32,
    pub head_dim: u32,
    pub global_head_dim: u32,
    pub vocab_size: u32,
    pub hidden_size_per_layer_input: u32,
    pub sliding_window: u32,
    pub layer_types: Vec<String>,
    pub num_kv_shared_layers: u32,
    pub norm_eps: i64,
    pub rope_base_sliding: i64,
    pub rope_base_full: i64,
    pub full_partial_rotary_factor_q16: i32,
    pub embedding_scale: i32,
    pub ple_embedding_scale: i32,
    pub ple_projection_scalar: i32,
    pub ple_input_scale: i32,
    pub final_logit_softcap: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DirectInferProvenance {
    pub raster_manifest_path: PathBuf,
    pub raster_manifest_sha256: String,
}
