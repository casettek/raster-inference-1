use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use raster_runtime::OutputArtifact;
use serde::Serialize;

use crate::artifact_io::{encode_output, write_output, EncodedArtifact};
use crate::cache::{CachedInputs, CachedStageValue};

pub mod decode_select_token;
pub mod input_embedding;
pub mod prefill_finalize;
pub mod prefill_prepare_aux;
pub mod prefill_range;
pub mod prompt_prepare;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StageKind {
    PromptPrepare,
    InputEmbedding,
    PrefillPrepareAux { layer: usize },
    PrefillRange { layer: usize },
    PrefillFinalize,
    DecodeSelectToken,
}

#[derive(Debug)]
pub struct DirectCompareOutput {
    pub encoded: EncodedArtifact,
    pub timings: DirectTimings,
}

pub struct DirectPublishOutput {
    pub artifact: OutputArtifact,
    pub timings: DirectTimings,
    pub output: CachedStageValue,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirectTimings {
    pub input_load_duration: Duration,
    pub kernel_duration: Duration,
    pub encode_write_duration: Duration,
    pub direct_stage_duration: Duration,
}

impl StageKind {
    pub fn from_stage_spec(project: &str, name: &str) -> Result<Self> {
        let kind = Self::from_stage_name(name)?;
        if kind.project() != project {
            bail!(
                "stage `{name}` uses project `{project}`, expected `{}` for direct-native {}",
                kind.project(),
                kind.routine()
            );
        }
        Ok(kind)
    }

    pub fn from_stage_dir(stage_dir: &Path) -> Result<Self> {
        let name = stage_dir
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| anyhow::anyhow!("selected stage has no valid UTF-8 directory name"))?;
        Self::from_stage_name(name)
    }

    pub fn from_stage_name(name: &str) -> Result<Self> {
        match name {
            "prompt_prepare" => Ok(Self::PromptPrepare),
            "input_embedding" => Ok(Self::InputEmbedding),
            "prefill_finalize" => Ok(Self::PrefillFinalize),
            "decode_select_token" => Ok(Self::DecodeSelectToken),
            _ if name.starts_with("prefill_prepare_aux_l") => Ok(Self::PrefillPrepareAux {
                layer: parse_layer_index(name, "prefill_prepare_aux_l")?,
            }),
            _ if name.starts_with("prefill_range_l") => Ok(Self::PrefillRange {
                layer: parse_layer_index(name, "prefill_range_l")?,
            }),
            _ => bail!("stage `{name}` is not supported by direct-native"),
        }
    }

    pub fn project(&self) -> &'static str {
        match self {
            Self::PromptPrepare => "prompt-prepare",
            Self::InputEmbedding => "input-embedding",
            Self::PrefillPrepareAux { .. } => "prefill-prepare-aux",
            Self::PrefillRange { .. } => "prefill-range",
            Self::PrefillFinalize => "prefill-finalize",
            Self::DecodeSelectToken => "decode-select-token",
        }
    }

    pub fn routine(&self) -> &'static str {
        match self {
            Self::PromptPrepare => "prompt_prepare",
            Self::InputEmbedding => "input_embedding",
            Self::PrefillPrepareAux { .. } => "prefill_prepare_aux",
            Self::PrefillRange { .. } => "prefill_range",
            Self::PrefillFinalize => "prefill_finalize",
            Self::DecodeSelectToken => "decode_select_token",
        }
    }

    pub fn instance(&self) -> Option<usize> {
        match self {
            Self::PrefillPrepareAux { layer } | Self::PrefillRange { layer } => Some(*layer),
            Self::PromptPrepare
            | Self::InputEmbedding
            | Self::PrefillFinalize
            | Self::DecodeSelectToken => None,
        }
    }
}

pub fn run_for_compare(kind: &StageKind) -> Result<DirectCompareOutput> {
    match kind {
        StageKind::PromptPrepare => compare(
            prompt_prepare::load_inputs_from_args,
            prompt_prepare::run_direct,
            "prompt_prepare",
        ),
        StageKind::InputEmbedding => compare(
            input_embedding::load_inputs_from_args,
            input_embedding::run_direct,
            "input_embedding",
        ),
        StageKind::PrefillPrepareAux { .. } => compare(
            prefill_prepare_aux::load_inputs_from_args,
            prefill_prepare_aux::run_direct,
            "prefill_prepare_aux",
        ),
        StageKind::PrefillRange { .. } => compare(
            prefill_range::load_inputs_from_args,
            prefill_range::run_direct,
            "prefill_range",
        ),
        StageKind::PrefillFinalize => compare(
            prefill_finalize::load_inputs_from_args,
            prefill_finalize::run_direct,
            "prefill_finalize",
        ),
        StageKind::DecodeSelectToken => compare(
            decode_select_token::load_inputs_from_args,
            decode_select_token::run_direct,
            "decode_select_token",
        ),
    }
}

pub fn run_and_publish(kind: &StageKind) -> Result<DirectPublishOutput> {
    match kind {
        StageKind::PromptPrepare => publish(
            prompt_prepare::load_inputs_from_args,
            prompt_prepare::run_direct,
            "prompt_prepare",
            CachedStageValue::PromptTokenization,
        ),
        StageKind::InputEmbedding => publish(
            input_embedding::load_inputs_from_args,
            input_embedding::run_direct,
            "input_embedding",
            CachedStageValue::EmbeddedActivations,
        ),
        StageKind::PrefillPrepareAux { .. } => publish(
            prefill_prepare_aux::load_inputs_from_args,
            prefill_prepare_aux::run_direct,
            "prefill_prepare_aux",
            CachedStageValue::PleLayerInputs,
        ),
        StageKind::PrefillRange { .. } => publish(
            prefill_range::load_inputs_from_args,
            prefill_range::run_direct,
            "prefill_range",
            CachedStageValue::RangeActivations,
        ),
        StageKind::PrefillFinalize => publish(
            prefill_finalize::load_inputs_from_args,
            prefill_finalize::run_direct,
            "prefill_finalize",
            CachedStageValue::PrefillLogits,
        ),
        StageKind::DecodeSelectToken => publish(
            decode_select_token::load_inputs_from_args,
            decode_select_token::run_direct,
            "decode_select_token",
            CachedStageValue::SelectedToken,
        ),
    }
}

pub fn run_and_publish_from_paths(
    kind: &StageKind,
    input_path: &Path,
    input_manifest_path: &Path,
    cached_inputs: &CachedInputs,
) -> Result<DirectPublishOutput> {
    match kind {
        StageKind::PromptPrepare => publish(
            || {
                prompt_prepare::load_inputs_from_paths(
                    input_path,
                    input_manifest_path,
                    cached_inputs,
                )
            },
            prompt_prepare::run_direct,
            "prompt_prepare",
            CachedStageValue::PromptTokenization,
        ),
        StageKind::InputEmbedding => publish(
            || {
                input_embedding::load_inputs_from_paths(
                    input_path,
                    input_manifest_path,
                    cached_inputs,
                )
            },
            input_embedding::run_direct,
            "input_embedding",
            CachedStageValue::EmbeddedActivations,
        ),
        StageKind::PrefillPrepareAux { .. } => publish(
            || {
                prefill_prepare_aux::load_inputs_from_paths(
                    input_path,
                    input_manifest_path,
                    cached_inputs,
                )
            },
            prefill_prepare_aux::run_direct,
            "prefill_prepare_aux",
            CachedStageValue::PleLayerInputs,
        ),
        StageKind::PrefillRange { .. } => publish(
            || {
                prefill_range::load_inputs_from_paths(
                    input_path,
                    input_manifest_path,
                    cached_inputs,
                )
            },
            prefill_range::run_direct,
            "prefill_range",
            CachedStageValue::RangeActivations,
        ),
        StageKind::PrefillFinalize => publish(
            || {
                prefill_finalize::load_inputs_from_paths(
                    input_path,
                    input_manifest_path,
                    cached_inputs,
                )
            },
            prefill_finalize::run_direct,
            "prefill_finalize",
            CachedStageValue::PrefillLogits,
        ),
        StageKind::DecodeSelectToken => publish(
            || {
                decode_select_token::load_inputs_from_paths(
                    input_path,
                    input_manifest_path,
                    cached_inputs,
                )
            },
            decode_select_token::run_direct,
            "decode_select_token",
            CachedStageValue::SelectedToken,
        ),
    }
}

fn compare<I, O>(
    load: impl FnOnce() -> Result<I>,
    run: impl FnOnce(&I) -> Result<O>,
    label: &str,
) -> Result<DirectCompareOutput>
where
    O: Serialize,
{
    let direct_stage_started = Instant::now();
    let input_load_started = Instant::now();
    let inputs = load().with_context(|| format!("failed to load {label} stage inputs"))?;
    let input_load_duration = input_load_started.elapsed();

    let kernel_started = Instant::now();
    let output = run(&inputs).with_context(|| format!("direct-native {label} execution failed"))?;
    let kernel_duration = kernel_started.elapsed();

    let encode_write_started = Instant::now();
    let encoded = encode_output(&output)
        .with_context(|| format!("failed to encode direct-native {label} output"))?;
    let encode_write_duration = encode_write_started.elapsed();

    Ok(DirectCompareOutput {
        encoded,
        timings: DirectTimings {
            input_load_duration,
            kernel_duration,
            encode_write_duration,
            direct_stage_duration: direct_stage_started.elapsed(),
        },
    })
}

fn publish<I, O>(
    load: impl FnOnce() -> Result<I>,
    run: impl FnOnce(&I) -> Result<O>,
    label: &str,
    cache_value: impl FnOnce(O) -> CachedStageValue,
) -> Result<DirectPublishOutput>
where
    O: Serialize,
{
    let direct_stage_started = Instant::now();
    let input_load_started = Instant::now();
    let inputs = load().with_context(|| format!("failed to load {label} stage inputs"))?;
    let input_load_duration = input_load_started.elapsed();

    let kernel_started = Instant::now();
    let output = run(&inputs).with_context(|| format!("direct-native {label} execution failed"))?;
    let kernel_duration = kernel_started.elapsed();

    let encode_write_started = Instant::now();
    let artifact = write_output(&output)
        .with_context(|| format!("failed to write direct-native {label} output"))?;
    let encode_write_duration = encode_write_started.elapsed();
    let output = cache_value(output);

    Ok(DirectPublishOutput {
        artifact,
        timings: DirectTimings {
            input_load_duration,
            kernel_duration,
            encode_write_duration,
            direct_stage_duration: direct_stage_started.elapsed(),
        },
        output,
    })
}

fn parse_layer_index(name: &str, prefix: &str) -> Result<usize> {
    let raw = name
        .strip_prefix(prefix)
        .expect("caller checked stage name prefix");
    raw.parse::<usize>()
        .with_context(|| format!("failed to parse layer index from `{name}`"))
}
