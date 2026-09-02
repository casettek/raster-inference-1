use std::path::Path;

use anyhow::{bail, Result};
use prefill_finalize::input::{
    rms_norm, softcap, unpack_i32s, ActivationSequence, FinalHead, FinalHeadParams, LogitEntry,
    PrefillLogits,
};
use raster::List;
use rayon::prelude::*;

use crate::artifact_io::{with_main_sequence_scope, with_stage_sequence_scope};
use crate::cache::CachedInputs;
use crate::tensor::{self, Matrix};

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
    let params = validate_head(&inputs.head.params)?;
    let position = inputs.activations.rows.iter().last().ok_or_else(|| {
        anyhow::anyhow!("prefill produced no activation rows; there is no final position to score")
    })?;

    let mut values = unpack_i32s(&position.values).map_err(anyhow::Error::msg)?;
    if values.len() != params.hidden_size as usize {
        bail!(
            "final position has {} values, expected hidden_size {}",
            values.len(),
            params.hidden_size
        );
    }
    rms_norm(
        &mut values,
        &unpack_i32s(&params.norm_weights).map_err(anyhow::Error::msg)?,
        params.norm_eps,
    )
    .map_err(anyhow::Error::msg)?;

    let projection = projection_matrix(&inputs.head, &params)?;
    let scores = projection.matvec(&values)?;
    let logits = if tensor::use_parallel() {
        scores
            .par_iter()
            .enumerate()
            .map(|(token_id, value)| LogitEntry {
                token_id: token_id as u32,
                value: softcap(*value, params.softcap),
            })
            .collect::<Vec<_>>()
    } else {
        scores
            .iter()
            .enumerate()
            .map(|(token_id, value)| LogitEntry {
                token_id: token_id as u32,
                value: softcap(*value, params.softcap),
            })
            .collect::<Vec<_>>()
    };

    Ok(PrefillLogits {
        decode_position: inputs.activations.start_position + inputs.activations.rows.len() as u32,
        logits: List::from(logits),
        errors: List::new(),
    })
}

fn validate_head(params: &FinalHeadParams) -> Result<FinalHeadParams> {
    if params.hidden_size == 0 {
        bail!("output head requires a non-zero hidden_size");
    }
    if params.norm_eps < 0 {
        bail!("rms-norm epsilon must be non-negative");
    }
    if params.softcap < 0 {
        bail!("final logit softcap must be non-negative (0 disables it)");
    }
    let expected = params.hidden_size as usize * 4;
    if params.norm_weights.len() != expected {
        bail!(
            "final norm weights have {} bytes, expected {expected}",
            params.norm_weights.len()
        );
    }
    Ok(params.clone())
}

fn projection_matrix(head: &FinalHead, params: &FinalHeadParams) -> Result<Matrix> {
    let stride = params.hidden_size as u64 * 4;
    let byte_len = head.projection.byte_len();
    if stride == 0 || byte_len % stride != 0 {
        bail!("projection has {byte_len} bytes, expected a whole number of hidden-width rows");
    }
    Matrix::from_region(
        "projection",
        &head.projection,
        (byte_len / stride) as usize,
        params.hidden_size as usize,
    )
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
