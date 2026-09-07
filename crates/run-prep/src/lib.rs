use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use inference_artifacts::{InferenceRunSpec, ModelManifest, PreparedPrompt};
use output_finalize::input::{DecoderTable, DecoderToken};
use prompt_prepare::input::{merge_bucket_of, vocab_bucket_of, BpeMerge, MergeBucket};
use prompt_prepare::input::{BpePieces, PromptTokenizer, TokenEntry, VocabBucket};
use raster::List;

const END_OF_WORD: &str = "</w>";
const BOS_TOKEN: &str = "<bos>";
const TURN_OPEN: &str = "<|turn>";
const TURN_CLOSE: &str = "<turn|>";
const NEWLINE_TOKEN: &str = "\n";

pub fn prepare_prompt(
    tokenizer: &serde_json::Value,
    spec_dir: &Path,
    spec: &InferenceRunSpec,
    model: &ModelManifest,
) -> Result<PreparedPrompt> {
    let model_json = tokenizer_model(tokenizer)?;
    let vocab = vocab_map(model_json)?;
    let prompt = resolve_prompt(spec_dir, spec)?;
    let templated = !spec.raw_prompt && supports_gemma_turns(&vocab);
    let rendered_prompt = if templated {
        render_gemma_turns(&prompt)
    } else {
        prompt.clone()
    };
    let initial_pieces = if templated {
        let special_tokens = special_tokens(tokenizer);
        split_prompt(&rendered_prompt, &vocab, &special_tokens)
    } else {
        split_prompt(&rendered_prompt, &vocab, &[])
    };

    Ok(PreparedPrompt {
        resolved_prompt: prompt,
        rendered_prompt,
        initial_pieces,
        eos_token_ids: model.eos_token_ids.clone(),
    })
}

pub fn prompt_tokenizer(tokenizer: &serde_json::Value) -> Result<PromptTokenizer> {
    let model = tokenizer_model(tokenizer)?;
    let mut vocab: Vec<TokenEntry> = vocab_map(model)?
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

pub fn decoder_table(tokenizer: &serde_json::Value, eos_ids: &[u32]) -> Result<DecoderTable> {
    let model = tokenizer_model(tokenizer)?;
    let vocab_map = vocab_map(model)?;
    let special_ids: BTreeSet<u32> = tokenizer
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
    let max_token_id = vocab_map.values().copied().max().unwrap_or(0);
    let mut tokens = vec![DecoderToken::default(); max_token_id as usize + 1];
    for (token, id) in vocab_map {
        tokens[id as usize] = DecoderToken {
            token,
            special: special_ids.contains(&id),
            terminal: eos_ids.contains(&id),
        };
    }
    Ok(DecoderTable {
        tokens: List::from(tokens),
    })
}

pub fn write_initial_pieces(
    prompt: &PreparedPrompt,
    dir: &Path,
) -> Result<(std::path::PathBuf, std::path::PathBuf, String)> {
    fs::create_dir_all(dir).with_context(|| format!("failed to create {}", dir.display()))?;
    let pieces = BpePieces {
        pieces: List::from(prompt.initial_pieces.clone()),
    };
    let data_path = dir.join("initial_pieces.rastered");
    let index_path = dir.join("initial_pieces.rindex");
    let commitment = raster::write_raster_files(&pieces, &data_path, &index_path)
        .map_err(|error| anyhow::anyhow!("{error}"))?;
    Ok((data_path, index_path, commitment))
}

pub fn render_gemma_turns(prompt: &str) -> String {
    format!(
        "{BOS_TOKEN}{TURN_OPEN}user\n{}{TURN_CLOSE}\n{TURN_OPEN}model\n",
        prompt.trim()
    )
}

pub fn supports_gemma_turns(vocab: &BTreeMap<String, u32>) -> bool {
    [BOS_TOKEN, TURN_OPEN, TURN_CLOSE, NEWLINE_TOKEN]
        .iter()
        .all(|token| vocab.contains_key(*token))
}

pub fn split_prompt(
    prompt: &str,
    vocab: &BTreeMap<String, u32>,
    specials: &[String],
) -> Vec<String> {
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

pub fn vocab_map(model: &serde_json::Value) -> Result<BTreeMap<String, u32>> {
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

pub fn special_tokens(tokenizer: &serde_json::Value) -> Vec<String> {
    let mut tokens: Vec<String> = tokenizer
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
    tokens.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
    tokens
}

fn resolve_prompt(spec_dir: &Path, spec: &InferenceRunSpec) -> Result<String> {
    if let Some(prompt) = spec.prompt.as_ref() {
        return Ok(prompt.clone());
    }
    let prompt_file = spec
        .prompt_file
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("run spec must set `prompt` or `prompt_file`"))?;
    let path = if prompt_file.is_absolute() {
        prompt_file.clone()
    } else {
        spec_dir.join(prompt_file)
    };
    fs::read_to_string(&path)
        .with_context(|| format!("failed to read prompt file {}", path.display()))
}

fn tokenizer_model(tokenizer: &serde_json::Value) -> Result<&serde_json::Value> {
    tokenizer
        .get("model")
        .ok_or_else(|| anyhow::anyhow!("tokenizer.json has no model"))
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

#[cfg(test)]
mod tests {
    use super::*;
    use inference_artifacts::{DirectInferBundle, DirectInferShape};
    use std::path::PathBuf;

    const ONE: i32 = 1 << 16;

    #[test]
    fn gemma_turn_rendering_matches_current_shape() {
        assert_eq!(
            render_gemma_turns(" hello raster "),
            "<bos><|turn>user\nhello raster<turn|>\n<|turn>model\n"
        );
    }

    #[test]
    fn raw_prompt_bypasses_turn_rendering() {
        let tokenizer = gemma_tokenizer();
        let spec = run_spec("h", true);
        let prepared =
            prepare_prompt(&tokenizer, Path::new("."), &spec, &model_manifest()).unwrap();

        assert_eq!(prepared.rendered_prompt, "h");
        assert_eq!(prepared.initial_pieces, vec!["h", "</w>"]);
    }

    #[test]
    fn special_tokens_stay_whole_in_initial_pieces() {
        let tokenizer = gemma_tokenizer();
        let spec = run_spec("hello", false);
        let prepared =
            prepare_prompt(&tokenizer, Path::new("."), &spec, &model_manifest()).unwrap();

        assert!(prepared.initial_pieces.contains(&String::from("<bos>")));
        assert!(prepared.initial_pieces.contains(&String::from("<|turn>")));
    }

    #[test]
    fn byte_fallback_emits_hex_pieces_for_unknown_utf8_bytes() {
        let tokenizer = serde_json::json!({
            "model": { "vocab": {}, "merges": [] },
            "added_tokens": []
        });
        let model = tokenizer.get("model").unwrap();

        assert_eq!(
            split_prompt("é", &vocab_map(model).unwrap(), &[]),
            vec!["<0xC3>", "<0xA9>", "</w>"]
        );
    }

    fn run_spec(prompt: &str, raw_prompt: bool) -> InferenceRunSpec {
        InferenceRunSpec {
            model_manifest: PathBuf::from("model-artifacts/manifest.json"),
            prompt: Some(prompt.to_string()),
            prompt_file: None,
            raw_prompt,
            tokens: 1,
        }
    }

    fn model_manifest() -> ModelManifest {
        ModelManifest {
            version: 2,
            bundle: DirectInferBundle {
                model_detwgt_path: PathBuf::from("model.detwgt"),
                model_detwgt_sha256: String::new(),
                config_path: PathBuf::from("config.json"),
                config_sha256: String::new(),
                tokenizer_path: PathBuf::from("tokenizer.json"),
                tokenizer_sha256: String::new(),
            },
            shape: DirectInferShape {
                hidden_size: 1,
                num_hidden_layers: 0,
                num_attention_heads: 1,
                num_key_value_heads: 1,
                head_dim: 1,
                global_head_dim: 1,
                vocab_size: 1,
                hidden_size_per_layer_input: 1,
                sliding_window: 0,
                layer_types: Vec::new(),
                num_kv_shared_layers: 0,
                norm_eps: 0,
                rope_base_sliding: 10_000_i64 << 32,
                rope_base_full: 1_000_000_i64 << 32,
                full_partial_rotary_factor_q16: ONE,
                embedding_scale: ONE,
                ple_embedding_scale: ONE,
                ple_projection_scalar: ONE,
                ple_input_scale: ONE,
                final_logit_softcap: 0,
            },
            eos_token_ids: vec![1, 2],
            provenance: None,
        }
    }

    fn gemma_tokenizer() -> serde_json::Value {
        serde_json::json!({
            "model": {
                "vocab": {
                    "<bos>": 0,
                    "<|turn>": 1,
                    "<turn|>": 2,
                    "\n": 3,
                    "hello": 4,
                    "user": 5,
                    "model": 6,
                    "h": 7
                },
                "merges": []
            },
            "added_tokens": [
                { "id": 0, "content": "<bos>", "special": true },
                { "id": 1, "content": "<|turn>", "special": true },
                { "id": 2, "content": "<turn|>", "special": true }
            ]
        })
    }
}
