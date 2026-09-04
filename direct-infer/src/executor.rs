use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use decode_embed::input as decode_embed_input;
use inference_artifacts::InferenceResult;
use staged_infer::cache::CachedStageValue;
use staged_infer::{InferStageTiming, InferenceRunReport, InferenceTimings};

use crate::model::DirectInferenceModel;
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
        let prompt_value = CachedStageValue::PromptTokenization(prompt);
        timings.push(phase_timing("prompt_prepare", prompt_started.elapsed()));

        let embedding_started = Instant::now();
        let embedding = model.embedding_table()?;
        let embedding_prompt = prompt_value.as_embedding_prompt()?;
        let embedded = staged_infer::kernels::input_embedding::run_input_embedding_direct(
            staged_infer::kernels::input_embedding::InputEmbeddingDirectInputs {
                prompt: &embedding_prompt,
                embedding: &embedding,
            },
        )?;
        let embedded_value = CachedStageValue::EmbeddedActivations(embedded);
        let empty_or_embedding = embedded_value.as_range_activation_sequence()?;
        timings.push(phase_timing("input_embedding", embedding_started.elapsed()));

        let prefill_started = Instant::now();
        let (mut prior_layers, mut logits) = run_layers_and_finalize(
            &model,
            empty_or_embedding.clone(),
            empty_or_embedding.clone(),
            None,
            &mut timings,
            "prefill",
        )?;
        timings.push(phase_timing("prefill", prefill_started.elapsed()));

        let decode_started = Instant::now();
        let mut edge = staged_infer::kernels::decode_init::run_decode_init_direct(
            staged_infer::kernels::decode_init::DecodeInitDirectInputs,
        )?;
        for token_idx in 0..model.manifest.import.tokens {
            let select_logits = CachedStageValue::PrefillLogits(logits).as_decode_logits()?;
            edge = staged_infer::kernels::decode_select_token::run_decode_select_token_direct(
                staged_infer::kernels::decode_select_token::DecodeSelectTokenDirectInputs {
                    logits: &select_logits,
                    prior: &edge,
                },
            )?;
            let selected = CachedStageValue::DecodeEdge(edge.clone()).as_decode_embed_edge()?;
            let decode_embedding = decode_embedding(&embedding);
            let decoded = staged_infer::kernels::decode_embed::run_decode_embed_direct(
                staged_infer::kernels::decode_embed::DecodeEmbedDirectInputs {
                    selected: &selected,
                    embedding: &decode_embedding,
                },
            )?;
            let decode_seed =
                CachedStageValue::DecodeActivations(decoded).as_range_activation_sequence()?;
            let (next_prior_layers, next_logits) = run_layers_and_finalize(
                &model,
                decode_seed,
                empty_or_embedding.clone(),
                Some(&prior_layers),
                &mut timings,
                &format!("decode_t{token_idx}"),
            )?;
            prior_layers = next_prior_layers;
            logits = next_logits;
        }
        timings.push(phase_timing("decode", decode_started.elapsed()));

        let finalize_started = Instant::now();
        let output_edge = CachedStageValue::DecodeEdge(edge).as_output_edge()?;
        let decoder = model.decoder_table()?;
        let output = staged_infer::kernels::output_finalize::run_output_finalize_direct(
            staged_infer::kernels::output_finalize::OutputFinalizeDirectInputs {
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

fn run_layers_and_finalize(
    model: &DirectInferenceModel,
    seed: prefill_range::input::ActivationSequence,
    empty_or_embedding: prefill_range::input::ActivationSequence,
    prior_layers: Option<&[prefill_range::input::ActivationSequence]>,
    timings: &mut Vec<InferStageTiming>,
    label: &str,
) -> Result<(
    Vec<prefill_range::input::ActivationSequence>,
    prefill_finalize::input::PrefillLogits,
)> {
    let mut current = seed;
    let mut layer_outputs = Vec::with_capacity(model.shape().num_hidden_layers as usize);
    for layer_idx in 0..model.shape().num_hidden_layers as usize {
        let aux_started = Instant::now();
        let embedded =
            CachedStageValue::RangeActivations(current.clone()).as_aux_activation_sequence()?;
        let ple_layer = model.ple_layer(layer_idx)?;
        let ple = staged_infer::kernels::prefill_prepare_aux::run_prefill_prepare_aux_direct(
            staged_infer::kernels::prefill_prepare_aux::PrefillPrepareAuxDirectInputs {
                embedded: &embedded,
                layer: &ple_layer,
            },
        )?;
        timings.push(phase_timing(
            &format!("{label}_prefill_prepare_aux_l{layer_idx}"),
            aux_started.elapsed(),
        ));

        let range_started = Instant::now();
        let range_ple = CachedStageValue::PleLayerInputs(ple).as_range_ple_inputs()?;
        let layer = model.transformer_layer(layer_idx)?;
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
        current = staged_infer::kernels::prefill_range::run_prefill_range_direct(
            staged_infer::kernels::prefill_range::PrefillRangeDirectInputs {
                activations: &current,
                layer: &layer,
                layer_cache_key: None,
                prior_kv: &prior_kv,
                donor_a_kv: &donor_a_kv,
                donor_b_kv: &donor_b_kv,
                ple: &range_ple,
            },
        )?;
        layer_outputs.push(current.clone());
        timings.push(phase_timing(
            &format!("{label}_prefill_range_l{layer_idx}"),
            range_started.elapsed(),
        ));
    }

    let finalize_started = Instant::now();
    let final_activations =
        CachedStageValue::RangeActivations(current).as_finalize_activation_sequence()?;
    let head = model.final_head()?;
    let logits = staged_infer::kernels::prefill_finalize::run_prefill_finalize_direct(
        staged_infer::kernels::prefill_finalize::PrefillFinalizeDirectInputs {
            activations: &final_activations,
            head: &head,
        },
    )?;
    timings.push(phase_timing(
        &format!("{label}_prefill_finalize"),
        finalize_started.elapsed(),
    ));
    Ok((layer_outputs, logits))
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

fn decode_embedding(
    embedding: &input_embedding::input::EmbeddingTable,
) -> decode_embed_input::EmbeddingTable {
    decode_embed_input::EmbeddingTable {
        hidden_size: embedding.hidden_size,
        embedding_scale: embedding.embedding_scale,
        values: embedding.values.clone(),
    }
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
