use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use inference_artifacts::{
    InferStageTiming, InferenceResult, InferenceRunReport, InferenceTimings,
};

use crate::model::DirectInferenceModel;
use crate::view_kernels::{
    advance_decode_edge, run_decode_embed_view, run_input_embedding_view, run_ple_prepare_view,
    run_prefill_range_view, score_next_token_view, NextTokenScore,
};
use crate::DirectInferenceConfig;

#[derive(Debug, Default)]
pub struct DirectInferenceExecutor;

impl DirectInferenceExecutor {
    pub fn run(&self, config: DirectInferenceConfig) -> Result<InferenceResult> {
        Ok(self.run_with_report(config)?.result)
    }

    pub fn run_with_report(&self, config: DirectInferenceConfig) -> Result<InferenceRunReport> {
        let infer_started = Instant::now();
        let mut timings = Vec::new();

        let load_started = Instant::now();
        let model = DirectInferenceModel::load(&config.direct_manifest_path)
            .context("failed to load direct-infer model")?;
        timings.push(phase_timing("load_direct_model", load_started.elapsed()));

        let prompt_started = Instant::now();
        let prompt = model.prompt_inputs()?;
        timings.push(phase_timing("prompt_prepare", prompt_started.elapsed()));

        let embedding_started = Instant::now();
        let empty_or_embedding = run_input_embedding_view(&model, &prompt)?;
        timings.push(phase_timing("input_embedding", embedding_started.elapsed()));

        let prefill_started = Instant::now();
        let (mut prior_layers, mut score) = run_layers_and_score(
            &model,
            empty_or_embedding.clone(),
            empty_or_embedding.clone(),
            None,
            &mut timings,
            "prefill",
        )?;
        timings.push(phase_timing("prefill", prefill_started.elapsed()));

        let decode_started = Instant::now();
        let mut edge = host_kernels::kernels::decode_init::run_decode_init_direct(
            host_kernels::kernels::decode_init::DecodeInitDirectInputs,
        )?;
        for token_idx in 0..model.manifest.import.tokens {
            edge = advance_decode_edge(score, &edge);
            if token_idx + 1 == model.manifest.import.tokens {
                break;
            }
            let decode_seed = run_decode_embed_view(&model, &edge)?;
            let (next_prior_layers, next_score) = run_layers_and_score(
                &model,
                decode_seed,
                empty_or_embedding.clone(),
                Some(&prior_layers),
                &mut timings,
                &format!("decode_t{token_idx}"),
            )?;
            prior_layers = next_prior_layers;
            score = next_score;
        }
        timings.push(phase_timing("decode", decode_started.elapsed()));

        let finalize_started = Instant::now();
        let output_edge = output_finalize::input::DecodeEdge {
            has_selected: edge.has_selected,
            decode_position: edge.decode_position,
            token_id: edge.token_id,
            value: edge.value,
            generated_token_ids: edge.generated_token_ids.clone(),
        };
        let decoder = model.decoder_table()?;
        let output = host_kernels::kernels::output_finalize::run_output_finalize_direct(
            host_kernels::kernels::output_finalize::OutputFinalizeDirectInputs {
                edge: &output_edge,
                decoder: &decoder,
            },
        )?;
        let result = InferenceResult {
            generated_token_count: output.generated_token_count,
            generated_token_ids: output.generated_token_ids.iter().copied().collect(),
            generated_token_ids_sha256: output.generated_token_ids_sha256,
            generated_text: output.generated_text,
            stop_reason: output.stop_reason,
        };
        timings.push(phase_timing("output_finalize", finalize_started.elapsed()));

        Ok(InferenceRunReport {
            result,
            timings: InferenceTimings {
                total_duration: infer_started.elapsed(),
                stages: timings,
                aux_waves: Vec::new(),
            },
        })
    }
}

fn run_layers_and_score(
    model: &DirectInferenceModel,
    seed: prefill_range::input::ActivationSequence,
    empty_or_embedding: prefill_range::input::ActivationSequence,
    prior_layers: Option<&[prefill_range::input::ActivationSequence]>,
    timings: &mut Vec<InferStageTiming>,
    label: &str,
) -> Result<(
    Vec<prefill_range::input::ActivationSequence>,
    NextTokenScore,
)> {
    let mut current = seed;
    let mut layer_outputs = Vec::with_capacity(model.shape().num_hidden_layers as usize);
    for layer_idx in 0..model.shape().num_hidden_layers as usize {
        let aux_started = Instant::now();
        let ple_layer = model.direct_ple_layer(layer_idx)?;
        let ple = run_ple_prepare_view(&current, &ple_layer)?;
        timings.push(phase_timing(
            &format!("{label}_prefill_prepare_aux_l{layer_idx}"),
            aux_started.elapsed(),
        ));

        let range_started = Instant::now();
        let layer = model.direct_transformer_layer(layer_idx)?;
        let prior_kv = prior_layers
            .and_then(|layers| layers.get(layer_idx))
            .cloned()
            .unwrap_or_else(|| empty_or_embedding.clone());
        let donor_a_kv = donor_input(
            &layer_outputs,
            layer.params.donor_a_layer,
            &empty_or_embedding,
        );
        let donor_b_kv = donor_input(
            &layer_outputs,
            layer.params.donor_b_layer,
            &empty_or_embedding,
        );
        current =
            run_prefill_range_view(&current, &layer, &prior_kv, &donor_a_kv, &donor_b_kv, &ple)?;
        layer_outputs.push(current.clone());
        timings.push(phase_timing(
            &format!("{label}_prefill_range_l{layer_idx}"),
            range_started.elapsed(),
        ));
    }

    let finalize_started = Instant::now();
    let head = model.direct_final_head()?;
    let score = score_next_token_view(&current, &head)?;
    timings.push(phase_timing(
        &format!("{label}_prefill_finalize"),
        finalize_started.elapsed(),
    ));
    Ok((layer_outputs, score))
}

fn donor_input(
    layers: &[prefill_range::input::ActivationSequence],
    donor_layer: i32,
    fallback: &prefill_range::input::ActivationSequence,
) -> prefill_range::input::ActivationSequence {
    if donor_layer < 0 {
        return fallback.clone();
    }
    layers
        .get(donor_layer as usize)
        .cloned()
        .unwrap_or_else(|| fallback.clone())
}

fn phase_timing(stage: &str, duration: Duration) -> InferStageTiming {
    InferStageTiming {
        stage: stage.to_string(),
        routine: String::from("direct-infer"),
        input_synthesis_duration: Duration::ZERO,
        input_load_duration: Duration::ZERO,
        kernel_duration: duration,
        encode_duration: Duration::ZERO,
        total_duration: duration,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_infer_uses_direct_manifest_by_default() {
        let config = DirectInferenceConfig::from_current_dir().expect("config should build");
        assert_eq!(
            config.direct_manifest_path,
            DirectInferenceModel::default_manifest_path(&config.base_dir)
        );
    }
}
