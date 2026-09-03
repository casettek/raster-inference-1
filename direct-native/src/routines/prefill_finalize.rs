use std::path::Path;

use anyhow::Result;
use prefill_finalize::input::{ActivationSequence, FinalHead, PrefillLogits};

use crate::artifact_io::{with_main_sequence_scope, with_stage_sequence_scope};
use crate::cache::CachedInputs;
use crate::kernels::prefill_finalize::{run_prefill_finalize_direct, PrefillFinalizeDirectInputs};

pub struct Inputs {
    pub activations: ActivationSequence,
    pub head: FinalHead,
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

pub fn run_direct(inputs: &Inputs) -> Result<PrefillLogits> {
    run_prefill_finalize_direct(PrefillFinalizeDirectInputs {
        activations: &inputs.activations,
        head: &inputs.head,
    })
}

fn load_inputs_from_initialized_runtime(cached_inputs: &CachedInputs) -> Result<Inputs> {
    let binding = raster::start_program(&[
        raster::entry_argument_spec::<ActivationSequence>("activations"),
        raster::entry_argument_spec::<FinalHead>("head"),
    ])?;
    let activations = match cached_inputs.get("activations") {
        Some(value) => value.as_finalize_activation_sequence()?,
        None => raster::materialize_auth_return(raster::entry_argument_auth_ref::<
            ActivationSequence,
        >(binding.reference.clone(), "activations")),
    };
    let head = raster::materialize_auth_return(raster::entry_argument_auth_ref::<FinalHead>(
        binding.reference,
        "head",
    ));
    Ok(Inputs { activations, head })
}
