use anyhow::Result;
use prefill_range::input::{ActivationSequence, PleLayerInputs, TransformerLayer};
use raster_runtime::OutputArtifact;

pub struct OwnedPrefillRangeInputs {
    pub activations: ActivationSequence,
    pub layer: TransformerLayer,
    pub donor_kv: ActivationSequence,
    pub ple: PleLayerInputs,
}

pub struct EncodedArtifact {
    pub data: Vec<u8>,
    pub index: Vec<u8>,
    pub structural_commitment: String,
}

pub fn load_prefill_range_inputs_from_args() -> Result<OwnedPrefillRangeInputs> {
    raster::init();
    raster_runtime::enter_sequence_scope("main");
    let result = load_prefill_range_inputs_from_initialized_runtime();
    raster_runtime::exit_sequence_scope();
    result
}

fn load_prefill_range_inputs_from_initialized_runtime() -> Result<OwnedPrefillRangeInputs> {
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

    Ok(OwnedPrefillRangeInputs {
        activations,
        layer,
        donor_kv,
        ple,
    })
}

pub fn encode_activation_sequence(value: &ActivationSequence) -> Result<EncodedArtifact> {
    let (data, index, structural_commitment) = raster::encode_raster_value(value)?;
    Ok(EncodedArtifact {
        data,
        index,
        structural_commitment,
    })
}

pub fn write_activation_sequence_output(value: &ActivationSequence) -> Result<OutputArtifact> {
    raster_runtime::write_program_output_artifact(value)?
        .ok_or_else(|| anyhow::anyhow!("{} is not set", raster_runtime::OUTPUT_DIR_ENV))
}
