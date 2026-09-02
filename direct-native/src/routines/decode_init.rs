use std::path::Path;

use anyhow::Result;
use decode_select_token::input::DecodeEdge;
use raster::List;

use crate::artifact_io::{with_main_sequence_scope, with_stage_sequence_scope};
use crate::cache::CachedInputs;

pub struct Inputs;

pub fn load_inputs_from_args() -> Result<Inputs> {
    with_main_sequence_scope(|| Ok(Inputs))
}

pub fn load_inputs_from_paths(
    input: &Path,
    input_manifest: &Path,
    _cached_inputs: &CachedInputs,
) -> Result<Inputs> {
    with_stage_sequence_scope(input, input_manifest, || Ok(Inputs))
}

pub fn run_direct(_inputs: &Inputs) -> Result<DecodeEdge> {
    Ok(DecodeEdge {
        has_selected: false,
        decode_position: 0,
        token_id: 0,
        value: 0,
        generated_token_ids: List::new(),
    })
}
