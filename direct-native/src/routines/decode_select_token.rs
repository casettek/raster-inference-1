use anyhow::{bail, Result};
use decode_select_token::input::{PrefillLogits, SelectedToken};

use crate::artifact_io::with_main_sequence_scope;

pub struct Inputs {
    pub logits: PrefillLogits,
}

pub fn load_inputs_from_args() -> Result<Inputs> {
    with_main_sequence_scope(load_inputs_from_initialized_runtime)
}

pub fn run_direct(inputs: &Inputs) -> Result<SelectedToken> {
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

    Ok(SelectedToken {
        decode_position: inputs.logits.decode_position,
        token_id: best_token,
        value: best_value,
    })
}

fn load_inputs_from_initialized_runtime() -> Result<Inputs> {
    let binding = raster::start_program(&[raster::entry_argument_spec::<PrefillLogits>("logits")])?;
    let logits = raster::materialize_auth_return(raster::entry_argument_auth_ref::<PrefillLogits>(
        binding.reference,
        "logits",
    ));
    Ok(Inputs { logits })
}
