use std::path::Path;

use anyhow::Result;
use prompt_prepare::input::{BpePieces, PromptTokenization, PromptTokenizer};

use crate::artifact_io::{with_main_sequence_scope, with_stage_sequence_scope};
use crate::cache::CachedInputs;
use crate::kernels::prompt_prepare::{run_prompt_prepare_direct, PromptPrepareDirectInputs};

pub struct Inputs {
    pub tokenizer: PromptTokenizer,
    pub initial_pieces: BpePieces,
}

pub fn load_inputs_from_args() -> Result<Inputs> {
    with_main_sequence_scope(load_inputs_from_initialized_runtime)
}

pub fn load_inputs_from_paths(
    input: &Path,
    input_manifest: &Path,
    _cached_inputs: &CachedInputs,
) -> Result<Inputs> {
    with_stage_sequence_scope(input, input_manifest, load_inputs_from_initialized_runtime)
}

pub fn run_direct(inputs: &Inputs) -> Result<PromptTokenization> {
    run_prompt_prepare_direct(PromptPrepareDirectInputs {
        tokenizer: &inputs.tokenizer,
        initial_pieces: &inputs.initial_pieces,
    })
}

fn load_inputs_from_initialized_runtime() -> Result<Inputs> {
    let binding = raster::start_program(&[
        raster::entry_argument_spec::<PromptTokenizer>("tokenizer"),
        raster::entry_argument_spec::<BpePieces>("initial_pieces"),
    ])?;
    let tokenizer = raster::materialize_auth_return(raster::entry_argument_auth_ref::<
        PromptTokenizer,
    >(binding.reference.clone(), "tokenizer"));
    let initial_pieces = raster::materialize_auth_return(raster::entry_argument_auth_ref::<
        BpePieces,
    >(binding.reference, "initial_pieces"));
    Ok(Inputs {
        tokenizer,
        initial_pieces,
    })
}
