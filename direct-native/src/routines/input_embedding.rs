use std::path::Path;

use anyhow::{bail, Result};
use det_num::ops::mul_sat;
use det_num::Act;
use input_embedding::input::{
    pack_i32s, unpack_i32s_at, ActivationRow, ActivationSequence, EmbeddingTable,
    PromptTokenization,
};
use raster::List;

use crate::artifact_io::{with_main_sequence_scope, with_stage_sequence_scope};
use crate::cache::CachedInputs;

pub struct Inputs {
    pub prompt: PromptTokenization,
    pub embedding: EmbeddingTable,
}

pub fn load_inputs_from_args() -> Result<Inputs> {
    with_main_sequence_scope(|| load_inputs_from_initialized_runtime(&CachedInputs::new()))
}

pub fn load_inputs_from_paths(
    input: &Path,
    input_manifest: &Path,
    cached_inputs: &CachedInputs,
) -> Result<Inputs> {
    with_stage_sequence_scope(input, input_manifest, || {
        load_inputs_from_initialized_runtime(cached_inputs)
    })
}

pub fn run_direct(inputs: &Inputs) -> Result<ActivationSequence> {
    let hidden_size = inputs.embedding.hidden_size;
    if hidden_size == 0 {
        bail!("embedding table requires non-zero hidden_size");
    }

    let page_size = inputs.embedding.values.page_size();
    let pages = inputs.embedding.values.pages();
    let scale = Act::from_bits(inputs.embedding.embedding_scale);
    let mut rows = Vec::with_capacity(inputs.prompt.token_ids.len());

    for token_id in inputs.prompt.token_ids.iter().copied() {
        let byte_off = token_id as u64 * hidden_size as u64 * 4;
        let page_idx = if page_size == 0 {
            0
        } else {
            byte_off / page_size
        };
        let page = pages.get(page_idx as usize).ok_or_else(|| {
            anyhow::anyhow!("token {token_id} has no embedding row of the declared width")
        })?;
        let values = unpack_i32s_at(page, byte_off, hidden_size)
            .map_err(anyhow::Error::msg)
            .map_err(|_| {
                anyhow::anyhow!("token {token_id} has no embedding row of the declared width")
            })?;
        let scaled = values
            .iter()
            .map(|bits| mul_sat(Act::from_bits(*bits), scale).to_bits())
            .collect::<Vec<_>>();
        rows.push(ActivationRow {
            token_id,
            values: pack_i32s(&scaled),
        });
    }

    Ok(ActivationSequence {
        rows: List::from(rows),
        errors: List::new(),
        kv: List::new(),
        start_position: 0,
    })
}

fn load_inputs_from_initialized_runtime(cached_inputs: &CachedInputs) -> Result<Inputs> {
    let binding = raster::start_program(&[
        raster::entry_argument_spec::<PromptTokenization>("prompt"),
        raster::entry_argument_spec::<EmbeddingTable>("embedding"),
    ])?;
    let prompt = match cached_inputs.get("prompt") {
        Some(value) => value.as_embedding_prompt()?,
        None => raster::materialize_auth_return(raster::entry_argument_auth_ref::<
            PromptTokenization,
        >(binding.reference.clone(), "prompt")),
    };
    let embedding = raster::materialize_auth_return(raster::entry_argument_auth_ref::<
        EmbeddingTable,
    >(binding.reference, "embedding"));
    Ok(Inputs { prompt, embedding })
}
