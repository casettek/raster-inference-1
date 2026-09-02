use std::path::Path;

use anyhow::{bail, Result};
use decode_select_token::input::{DecodeEdge, PrefillLogits};
use raster::List;

use crate::artifact_io::{with_main_sequence_scope, with_stage_sequence_scope};
use crate::cache::CachedInputs;

pub struct Inputs {
    pub logits: PrefillLogits,
    pub prior: DecodeEdge,
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

pub fn run_direct(inputs: &Inputs) -> Result<DecodeEdge> {
    let mut best_token = 0;
    let mut best_value = 0;
    let mut has_value = false;

    for entry in inputs.logits.logits.iter() {
        if !has_value || entry.value > best_value {
            has_value = true;
            best_token = entry.token_id;
            best_value = entry.value;
        }
    }

    if !has_value {
        bail!("output decode requires at least one logit to select the next token");
    }

    let mut generated_token_ids = inputs
        .prior
        .generated_token_ids
        .iter()
        .copied()
        .collect::<Vec<_>>();
    generated_token_ids.push(best_token);

    Ok(DecodeEdge {
        has_selected: true,
        decode_position: inputs.logits.decode_position,
        token_id: best_token,
        value: best_value,
        generated_token_ids: List::from(generated_token_ids),
    })
}

fn load_inputs_from_initialized_runtime(cached_inputs: &CachedInputs) -> Result<Inputs> {
    let binding = raster::start_program(&[
        raster::entry_argument_spec::<PrefillLogits>("logits"),
        raster::entry_argument_spec::<DecodeEdge>("prior"),
    ])?;
    let logits = match cached_inputs.get("logits") {
        Some(value) => value.as_decode_logits()?,
        None => raster::materialize_auth_return(raster::entry_argument_auth_ref::<PrefillLogits>(
            binding.reference.clone(),
            "logits",
        )),
    };
    let prior = match cached_inputs.get("prior") {
        Some(value) => value.as_decode_edge()?,
        None => raster::materialize_auth_return(raster::entry_argument_auth_ref::<DecodeEdge>(
            binding.reference,
            "prior",
        )),
    };
    Ok(Inputs { logits, prior })
}
