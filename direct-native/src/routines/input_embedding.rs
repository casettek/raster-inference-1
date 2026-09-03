use std::path::Path;

use anyhow::Result;
use input_embedding::input::{ActivationSequence, EmbeddingTable, PromptTokenization};

use crate::artifact_io::{with_main_sequence_scope, with_stage_sequence_scope};
use crate::cache::CachedInputs;
use crate::kernels::input_embedding::{run_input_embedding_direct, InputEmbeddingDirectInputs};

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
    run_input_embedding_direct(InputEmbeddingDirectInputs {
        prompt: &inputs.prompt,
        embedding: &inputs.embedding,
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
