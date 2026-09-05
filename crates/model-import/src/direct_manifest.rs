use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use inference_artifacts::{
    write_json, DirectInferBundle, DirectInferImportSettings, DirectInferManifest,
    DirectInferPrompt, DirectInferProvenance, DirectInferShape, DIRECT_INFER_ARTIFACTS_DIR,
    DIRECT_INFER_MANIFEST_JSON,
};
use sha2::{Digest, Sha256};

use super::{
    det_act_bits, f32_to_q16, render_gemma_turns, split_prompt, supports_gemma_turns, ImportConfig,
    Shape,
};

pub fn write_direct_manifest(
    args: &ImportConfig,
    tokenizer: &serde_json::Value,
    eos_ids: &BTreeSet<u32>,
    shape: &Shape,
    text_config: &serde_json::Value,
    raster_manifest: Option<(&Path, &str)>,
) -> Result<PathBuf, Box<dyn Error>> {
    let artifact_dir = PathBuf::from(DIRECT_INFER_ARTIFACTS_DIR);
    let manifest_path = artifact_dir.join(DIRECT_INFER_MANIFEST_JSON);
    let model_detwgt = args.model_dir.join("model.detwgt");
    let config = args.model_dir.join("config.json");
    let tokenizer_path = args.model_dir.join("tokenizer.json");

    let direct = DirectInferManifest {
        version: 1,
        bundle: DirectInferBundle {
            model_detwgt_path: manifest_relative_path(&artifact_dir, &model_detwgt),
            model_detwgt_sha256: sha256_file(&model_detwgt)?,
            config_path: manifest_relative_path(&artifact_dir, &config),
            config_sha256: sha256_file(&config)?,
            tokenizer_path: manifest_relative_path(&artifact_dir, &tokenizer_path),
            tokenizer_sha256: sha256_file(&tokenizer_path)?,
        },
        import: DirectInferImportSettings {
            prompt: args.prompt.clone(),
            raw_prompt: args.raw_prompt,
            tokens: args.tokens,
        },
        prompt: direct_prompt(tokenizer, &args.prompt, args.raw_prompt, eos_ids)?,
        shape: DirectInferShape {
            hidden_size: shape.hidden as u32,
            num_hidden_layers: shape.layers as u32,
            num_attention_heads: shape.heads as u32,
            num_key_value_heads: shape.kv_heads as u32,
            head_dim: shape.head_dim as u32,
            global_head_dim: shape.global_head_dim as u32,
            vocab_size: shape.vocab as u32,
            hidden_size_per_layer_input: shape.ple_width as u32,
            sliding_window: shape.sliding_window,
            layer_types: shape.layer_types.clone(),
            num_kv_shared_layers: shape.num_kv_shared_layers as u32,
            norm_eps: shape.norm_eps,
            rope_base_sliding: (shape.rope_base_sliding as i64) << 32,
            rope_base_full: (shape.rope_base_full as i64) << 32,
            full_partial_rotary_factor_q16: f32_to_q16(shape.full_partial_rotary_factor),
            embedding_scale: det_act_bits((shape.hidden as f32).sqrt()),
            ple_embedding_scale: det_act_bits((shape.ple_width as f32).sqrt()),
            ple_projection_scalar: det_act_bits((shape.hidden as f32).powf(-0.5)),
            ple_input_scale: det_act_bits(2f32.powf(-0.5)),
            final_logit_softcap: text_config
                .get("final_logit_softcapping")
                .and_then(serde_json::Value::as_f64)
                .map(f32_to_q16)
                .unwrap_or(0),
        },
        provenance: raster_manifest.map(|(path, text)| DirectInferProvenance {
            raster_manifest_path: manifest_relative_path(&artifact_dir, path),
            raster_manifest_sha256: format!("{:x}", Sha256::digest(text.as_bytes())),
        }),
    };

    write_json(&manifest_path, &direct)
        .map_err(|error| -> Box<dyn Error> { error.to_string().into() })?;
    println!("wrote {}", manifest_path.display());
    Ok(manifest_path)
}

pub fn warn_partial_direct_manifest_not_refreshed() {
    eprintln!(
        "direct-infer artifacts not refreshed; run a full `raster-inference model import ...` \
         to regenerate {DIRECT_INFER_ARTIFACTS_DIR}/{DIRECT_INFER_MANIFEST_JSON}"
    );
}

fn direct_prompt(
    tokenizer: &serde_json::Value,
    prompt: &str,
    raw_prompt: bool,
    eos_ids: &BTreeSet<u32>,
) -> Result<DirectInferPrompt, Box<dyn Error>> {
    let model = tokenizer
        .get("model")
        .ok_or("tokenizer.json has no model")?;
    let vocab: BTreeMap<String, u32> = model
        .get("vocab")
        .and_then(serde_json::Value::as_object)
        .ok_or("tokenizer.json has no vocab")?
        .iter()
        .map(|(token, id)| {
            let id = id.as_u64().unwrap_or_default() as u32;
            (token.clone(), id)
        })
        .collect();
    let mut special_tokens: Vec<String> = tokenizer
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
        .filter_map(|entry| entry.get("content").and_then(serde_json::Value::as_str))
        .map(str::to_string)
        .collect();
    special_tokens.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));

    let templated = !raw_prompt && supports_gemma_turns(&vocab);
    let rendered_prompt = if templated {
        render_gemma_turns(prompt)
    } else {
        prompt.to_string()
    };
    let initial_pieces = if templated {
        split_prompt(&rendered_prompt, &vocab, &special_tokens)
    } else {
        split_prompt(&rendered_prompt, &vocab, &[])
    };
    Ok(DirectInferPrompt {
        rendered_prompt,
        initial_pieces,
        eos_token_ids: eos_ids.iter().copied().collect(),
    })
}

fn manifest_relative_path(artifact_dir: &Path, source: &Path) -> PathBuf {
    if source.is_absolute() {
        return source.to_path_buf();
    }
    let mut path = PathBuf::new();
    for _ in artifact_dir.components() {
        path.push("..");
    }
    path.join(source)
}

fn sha256_file(path: &Path) -> Result<String, Box<dyn Error>> {
    Ok(format!("{:x}", Sha256::digest(fs::read(path)?)))
}
