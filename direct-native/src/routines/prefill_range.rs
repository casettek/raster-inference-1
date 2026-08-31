use anyhow::Result;
use prefill_range::input::{ActivationSequence, PleLayerInputs, TransformerLayer};

use crate::artifact_io::with_main_sequence_scope;

pub use crate::prefill_range::{run_prefill_range_direct, PrefillRangeDirectInputs};

pub struct Inputs {
    pub activations: ActivationSequence,
    pub layer: TransformerLayer,
    pub donor_kv: ActivationSequence,
    pub ple: PleLayerInputs,
}

pub fn load_inputs_from_args() -> Result<Inputs> {
    with_main_sequence_scope(load_inputs_from_initialized_runtime)
}

pub fn run_direct(inputs: &Inputs) -> Result<ActivationSequence> {
    run_prefill_range_direct(PrefillRangeDirectInputs {
        activations: &inputs.activations,
        layer: &inputs.layer,
        donor_kv: &inputs.donor_kv,
        ple: &inputs.ple,
    })
}

fn load_inputs_from_initialized_runtime() -> Result<Inputs> {
    let binding = raster::start_program(&[
        raster::entry_argument_spec::<ActivationSequence>("activations"),
        raster::entry_argument_spec::<TransformerLayer>("layer"),
        raster::entry_argument_spec::<ActivationSequence>("donor_kv"),
        raster::entry_argument_spec::<PleLayerInputs>("ple"),
    ])?;

    let activations = raster::materialize_auth_return(raster::entry_argument_auth_ref::<
        ActivationSequence,
    >(
        binding.reference.clone(), "activations"
    ));
    let layer = raster::materialize_auth_return(
        raster::entry_argument_auth_ref::<TransformerLayer>(binding.reference.clone(), "layer"),
    );
    let donor_kv = raster::materialize_auth_return(raster::entry_argument_auth_ref::<
        ActivationSequence,
    >(binding.reference.clone(), "donor_kv"));
    let ple = raster::materialize_auth_return(raster::entry_argument_auth_ref::<PleLayerInputs>(
        binding.reference,
        "ple",
    ));

    Ok(Inputs {
        activations,
        layer,
        donor_kv,
        ple,
    })
}
