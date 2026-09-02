use std::path::Path;

use anyhow::{bail, Result};
use output_finalize::input::{DecodeEdge, DecoderTable, GeneratedOutput};
use raster::{BytesPage, List};
use sha2::{Digest, Sha256};

use crate::artifact_io::{with_main_sequence_scope, with_stage_sequence_scope};
use crate::cache::CachedInputs;

pub struct Inputs {
    pub edge: DecodeEdge,
    pub decoder: DecoderTable,
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

pub fn run_direct(inputs: &Inputs) -> Result<GeneratedOutput> {
    let mut state = output_finalize::initial_finalize_state();
    for token_id in inputs.edge.generated_token_ids.iter().copied() {
        let token = inputs
            .decoder
            .tokens
            .get(token_id as usize)
            .ok_or_else(|| anyhow::anyhow!("decoder table has no token id {token_id}"))?;
        state = output_finalize::advance_finalize_state(state, token_id, token.clone());
    }
    if !state.pending_bytes.as_slice().is_empty() {
        state
            .text
            .push_str(&String::from_utf8_lossy(state.pending_bytes.as_slice()));
        state.pending_bytes = BytesPage::__from_parts(0, 0, Vec::new());
    }
    state.json.push(']');

    if inputs.edge.has_selected != (state.count > 0) {
        bail!(
            "decode edge selection flag {} disagrees with generated token count {}",
            inputs.edge.has_selected,
            state.count
        );
    }

    let digest = Sha256::digest(state.json.as_bytes());
    Ok(GeneratedOutput {
        generated_token_count: state.count,
        generated_token_ids: List::from(
            inputs
                .edge
                .generated_token_ids
                .iter()
                .copied()
                .collect::<Vec<_>>(),
        ),
        generated_token_ids_sha256: format!("{digest:x}"),
        generated_text: state.text,
        stop_reason: if state.stopped {
            String::from("eos")
        } else {
            String::from("max_new_tokens")
        },
    })
}

fn load_inputs_from_initialized_runtime(cached_inputs: &CachedInputs) -> Result<Inputs> {
    let binding = raster::start_program(&[
        raster::entry_argument_spec::<DecodeEdge>("edge"),
        raster::entry_argument_spec::<DecoderTable>("decoder"),
    ])?;
    let edge = match cached_inputs.get("edge") {
        Some(value) => value.as_output_edge()?,
        None => raster::materialize_auth_return(raster::entry_argument_auth_ref::<DecodeEdge>(
            binding.reference.clone(),
            "edge",
        )),
    };
    let decoder = raster::materialize_auth_return(raster::entry_argument_auth_ref::<DecoderTable>(
        binding.reference,
        "decoder",
    ));
    Ok(Inputs { edge, decoder })
}
