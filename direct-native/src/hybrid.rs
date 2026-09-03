use std::collections::BTreeMap;
use std::ffi::OsString;
use std::fs;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use raster_core::input::payload_structural_root;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::cache::{CachedInputs, CachedStageValue, StageOutputCache};
use crate::routines::{self, StageKind};
use crate::shadow::{parity_dir, EXECUTION_TIMES_JSON};

#[derive(Debug)]
pub struct HybridRun {
    pub chain_dir: PathBuf,
    pub selected_stage_dir: Option<PathBuf>,
}

#[derive(Debug)]
struct Manifest {
    chain: ChainSpec,
}

#[derive(Debug, Clone)]
struct ChainSpec {
    stage: Vec<StageSpec>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
struct StageSpec {
    name: String,
    project: String,
    #[serde(default)]
    inputs: BTreeMap<String, InputBinding>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum InputBinding {
    External(ExternalRef),
    From(String),
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
struct ExternalRef {
    path: String,
    #[serde(default)]
    index_path: Option<String>,
    commitment: String,
}

#[derive(Debug, Deserialize)]
struct RasterTomlDoc {
    chain: ChainTable,
}

#[derive(Debug, Deserialize)]
struct ChainTable {
    #[serde(default, rename = "input")]
    inputs: BTreeMap<String, InputDecl>,
    #[serde(default, rename = "stage")]
    stages: Vec<toml::Spanned<StageSpec>>,
    #[serde(default, rename = "repeat")]
    repeats: Vec<toml::Spanned<RepeatSpec>>,
}

#[derive(Debug)]
enum ChainItem {
    Stage(StageSpec),
    Repeat(RepeatSpec),
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum InputDecl {
    Indexed(IndexedInputDecl),
    Single(ExternalRef),
}

#[derive(Debug, Deserialize)]
struct IndexedInputDecl {
    index: String,
    path: String,
    #[serde(default)]
    index_path: Option<String>,
    commitments: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct RepeatSpec {
    name: String,
    index: String,
    #[serde(default)]
    start: u32,
    count: u32,
    #[serde(default, rename = "stage")]
    stages: Vec<RepeatStageSpec>,
    #[serde(default)]
    exports: BTreeMap<String, ExportDecl>,
}

#[derive(Debug, Deserialize)]
struct RepeatStageSpec {
    name: String,
    project: String,
    #[serde(default)]
    index: Option<String>,
    #[serde(default)]
    start: u32,
    #[serde(default)]
    count: Option<u32>,
    #[serde(default)]
    inputs: BTreeMap<String, RepeatBinding>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum RepeatBinding {
    From {
        from: String,
        #[serde(default)]
        first: Option<String>,
    },
    Input {
        input: String,
    },
    External {
        external: ExternalRef,
    },
}

#[derive(Debug, Deserialize)]
struct ExportDecl {
    stage: String,
    entry: String,
}

#[derive(Clone, Copy)]
struct TemplateIndex<'a> {
    name: &'a str,
    value: u32,
    start: u32,
}

enum RenderedTemplate {
    Text(String),
    Underflow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StageDispatch {
    DirectNative,
    RasterReference,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirectStageBackend {
    InProcess,
    Subprocess,
}

#[derive(Debug)]
struct StageOutput {
    payload_commitment: Vec<u8>,
    structural_commitment: Vec<u8>,
}

struct DirectStageRun {
    duration: Duration,
    output: Option<CachedStageValue>,
}

struct ChainRunState {
    output_commitments: Vec<Option<Vec<u8>>>,
    execution_times: Vec<Option<Duration>>,
    aux_waves: Vec<AuxWaveExecutionTime>,
    output_cache: StageOutputCache,
    selected_stage_dir: Option<PathBuf>,
}

struct AuxStageJob {
    idx: usize,
    name: String,
    kind: StageKind,
    input_json_path: PathBuf,
    input_manifest_path: PathBuf,
    stage_dir: PathBuf,
}

struct AuxStageResult {
    idx: usize,
    name: String,
    duration: Duration,
    output: StageOutput,
}

struct AuxStageBatch {
    results: Vec<AuxStageResult>,
    wall_duration: Duration,
    parallelism: usize,
}

#[derive(Debug, Serialize)]
struct ExecutionTimesDocument {
    version: u32,
    stages: Vec<StageExecutionTime>,
    total_exec_duration_ns: u128,
    total_wall_duration_ns: u128,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    aux_waves: Vec<AuxWaveExecutionTime>,
}

#[derive(Debug, Serialize)]
struct StageExecutionTime {
    name: String,
    exec_duration_ns: u128,
}

#[derive(Clone, Debug, Serialize)]
struct AuxWaveExecutionTime {
    name: String,
    first_stage: String,
    last_stage: String,
    stage_count: usize,
    parallelism: usize,
    wall_duration_ns: u128,
    stage_duration_sum_ns: u128,
}

pub fn run(
    raster_stage: Option<&str>,
    current_exe: &Path,
    direct_backend: DirectStageBackend,
) -> Result<HybridRun> {
    let base_dir = std::env::current_dir().context("failed to read current directory")?;
    let manifest = read_manifest(&base_dir.join("Raster.toml"))?;
    validate_supported_stages(&manifest.chain.stage)?;
    if let Some(raster_stage) = raster_stage {
        validate_reference_stage(&manifest.chain.stage, raster_stage)?;
    }

    let chain_dir = create_chain_dir(&base_dir)?;
    println!(
        "direct-native chain run  {}  ({} stages)",
        chain_run_id_label(&chain_dir),
        manifest.chain.stage.len()
    );
    println!("  dir: {}", chain_dir.display());
    match raster_stage {
        Some(stage) => {
            println!("  mode: unauthenticated hybrid (--no-auth; Raster reference: {stage})");
        }
        None => println!("  mode: unauthenticated direct-native (--no-auth)"),
    }
    println!();

    let chain_started = Instant::now();
    let stage_index = build_stage_index(&manifest.chain.stage)?;
    let mut state = ChainRunState {
        output_commitments: vec![None; manifest.chain.stage.len()],
        execution_times: vec![None; manifest.chain.stage.len()],
        aux_waves: Vec::new(),
        output_cache: StageOutputCache::default(),
        selected_stage_dir: None,
    };

    let mut idx = 0;
    while idx < manifest.chain.stage.len() {
        if let Some(aux_range) = aux_wave_range(&manifest.chain.stage, idx) {
            run_aux_wave(
                aux_range.clone(),
                &manifest.chain.stage,
                &base_dir,
                &chain_dir,
                &stage_index,
                current_exe,
                direct_backend,
                raster_stage,
                &mut state,
            )?;
            idx = aux_range.end;
            continue;
        }

        run_one_stage(
            idx,
            &manifest.chain.stage,
            &base_dir,
            &chain_dir,
            &stage_index,
            current_exe,
            direct_backend,
            raster_stage,
            &mut state,
        )?;
        idx += 1;
    }

    let chain_wall_duration = chain_started.elapsed();
    write_execution_times(
        &chain_dir,
        &manifest.chain.stage,
        &state.execution_times,
        chain_wall_duration,
        &state.aux_waves,
    )?;
    print_direct_timing_summary(
        &manifest.chain.stage,
        &state.execution_times,
        chain_wall_duration,
        &state.aux_waves,
    )?;
    if raster_stage.is_some() && state.selected_stage_dir.is_none() {
        bail!("selected Raster reference stage did not run");
    }

    Ok(HybridRun {
        chain_dir,
        selected_stage_dir: state.selected_stage_dir,
    })
}

fn read_manifest(path: &Path) -> Result<Manifest> {
    let text =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    text.parse::<toml::Value>()
        .context("failed to parse Raster.toml as TOML")?;
    let doc: RasterTomlDoc = toml::from_str(&text).context("failed to decode Raster.toml chain")?;
    Ok(Manifest {
        chain: expand_chain_table(doc.chain)?,
    })
}

fn expand_chain_table(table: ChainTable) -> Result<ChainSpec> {
    let inputs = flatten_input_decls(table.inputs)?;
    let mut stages = Vec::new();
    let mut exports = BTreeMap::new();

    for item in merge_chain_items(table.stages, table.repeats) {
        match item {
            ChainItem::Stage(stage) => stages.push(stage),
            ChainItem::Repeat(repeat) => {
                expand_repeat(&repeat, &inputs, &mut stages, &mut exports)?
            }
        }
    }

    resolve_export_sources(&mut stages, &exports);
    Ok(ChainSpec { stage: stages })
}

fn merge_chain_items(
    stages: Vec<toml::Spanned<StageSpec>>,
    repeats: Vec<toml::Spanned<RepeatSpec>>,
) -> Vec<ChainItem> {
    let mut items: Vec<(usize, ChainItem)> = stages
        .into_iter()
        .map(|stage| (stage.span().start, ChainItem::Stage(stage.into_inner())))
        .chain(
            repeats
                .into_iter()
                .map(|repeat| (repeat.span().start, ChainItem::Repeat(repeat.into_inner()))),
        )
        .collect();
    items.sort_by_key(|(offset, _)| *offset);
    items.into_iter().map(|(_, item)| item).collect()
}

fn flatten_input_decls(
    decls: BTreeMap<String, InputDecl>,
) -> Result<BTreeMap<String, ExternalRef>> {
    let mut inputs = BTreeMap::new();
    for (name, decl) in decls {
        match decl {
            InputDecl::Single(external) => {
                if inputs.insert(name.clone(), external).is_some() {
                    bail!("chain input `{name}` is declared more than once");
                }
            }
            InputDecl::Indexed(indexed) => {
                for (member, external) in indexed.flatten(&name)? {
                    if inputs.insert(member.clone(), external).is_some() {
                        bail!("chain input `{member}` is declared more than once");
                    }
                }
            }
        }
    }
    Ok(inputs)
}

impl IndexedInputDecl {
    fn flatten(self, family: &str) -> Result<Vec<(String, ExternalRef)>> {
        let placeholder = format!("{{{}}}", self.index);
        if !self.path.contains(&placeholder) {
            bail!(
                "[chain.input.{family}]: path '{}' does not mention '{placeholder}'",
                self.path
            );
        }

        Ok(self
            .commitments
            .into_iter()
            .enumerate()
            .map(|(idx, commitment)| {
                let index = idx.to_string();
                let render = |value: &str| value.replace(&placeholder, &index);
                (
                    format!("{family}_{idx}"),
                    ExternalRef {
                        path: render(&self.path),
                        index_path: self.index_path.as_deref().map(render),
                        commitment,
                    },
                )
            })
            .collect())
    }
}

fn expand_repeat(
    repeat: &RepeatSpec,
    inputs: &BTreeMap<String, ExternalRef>,
    stages: &mut Vec<StageSpec>,
    exports: &mut BTreeMap<String, String>,
) -> Result<()> {
    for outer in repeat.start..repeat.start + repeat.count {
        let outer_index = TemplateIndex {
            name: &repeat.index,
            value: outer,
            start: repeat.start,
        };
        for stage in &repeat.stages {
            if let (Some(inner_name), Some(inner_count)) = (&stage.index, stage.count) {
                for inner in stage.start..stage.start + inner_count {
                    let inner_index = TemplateIndex {
                        name: inner_name,
                        value: inner,
                        start: stage.start,
                    };
                    push_repeat_stage(stage, inputs, &[outer_index, inner_index], stages)?;
                }
            } else {
                push_repeat_stage(stage, inputs, &[outer_index], stages)?;
            }
        }
    }

    let export_index = TemplateIndex {
        name: &repeat.index,
        value: repeat.start + repeat.count.saturating_sub(1),
        start: repeat.start,
    };
    for (name, export) in &repeat.exports {
        let source = if repeat.count == 0 {
            export.entry.clone()
        } else {
            render_text(&export.stage, &[export_index])?
        };
        exports.insert(format!("{}.{}", repeat.name, name), source);
    }

    Ok(())
}

fn push_repeat_stage(
    template: &RepeatStageSpec,
    named_inputs: &BTreeMap<String, ExternalRef>,
    indexes: &[TemplateIndex<'_>],
    stages: &mut Vec<StageSpec>,
) -> Result<()> {
    let mut inputs = BTreeMap::new();
    for (param, binding) in &template.inputs {
        inputs.insert(
            param.clone(),
            expand_repeat_binding(binding, named_inputs, indexes)?,
        );
    }
    stages.push(StageSpec {
        name: render_text(&template.name, indexes)?,
        project: template.project.clone(),
        inputs,
    });
    Ok(())
}

fn expand_repeat_binding(
    binding: &RepeatBinding,
    named_inputs: &BTreeMap<String, ExternalRef>,
    indexes: &[TemplateIndex<'_>],
) -> Result<InputBinding> {
    match binding {
        RepeatBinding::From { from, first } => match render_template(from, indexes)? {
            RenderedTemplate::Text(source) => Ok(InputBinding::From(source)),
            RenderedTemplate::Underflow => {
                let first = first.as_ref().ok_or_else(|| {
                    anyhow::anyhow!(
                        "repeat binding `{from}` underflows but has no `first` fallback"
                    )
                })?;
                Ok(InputBinding::From(render_text(first, indexes)?))
            }
        },
        RepeatBinding::Input { input } => {
            let name = render_text(input, indexes)?;
            let external = named_inputs.get(&name).ok_or_else(|| {
                anyhow::anyhow!("repeat binding names unknown chain input `{name}`")
            })?;
            Ok(InputBinding::External(external.clone()))
        }
        RepeatBinding::External { external } => Ok(InputBinding::External(external.clone())),
    }
}

fn resolve_export_sources(stages: &mut [StageSpec], exports: &BTreeMap<String, String>) {
    for stage in stages {
        for binding in stage.inputs.values_mut() {
            if let InputBinding::From(source) = binding {
                if let Some(target) = exports.get(source) {
                    *source = target.clone();
                }
            }
        }
    }
}

fn render_text(template: &str, indexes: &[TemplateIndex<'_>]) -> Result<String> {
    match render_template(template, indexes)? {
        RenderedTemplate::Text(value) => Ok(value),
        RenderedTemplate::Underflow => {
            bail!("template `{template}` underflows outside a `first` binding")
        }
    }
}

fn render_template(template: &str, indexes: &[TemplateIndex<'_>]) -> Result<RenderedTemplate> {
    let mut rendered = String::new();
    let mut rest = template;
    while let Some(open) = rest.find('{') {
        rendered.push_str(&rest[..open]);
        let after_open = &rest[open + 1..];
        let close = after_open.find('}').ok_or_else(|| {
            anyhow::anyhow!("template `{template}` has an unterminated placeholder")
        })?;
        let placeholder = &after_open[..close];
        let (name, previous) = placeholder
            .strip_suffix("-1")
            .map(|name| (name, true))
            .unwrap_or((placeholder, false));
        let index = indexes
            .iter()
            .find(|index| index.name == name)
            .ok_or_else(|| {
                anyhow::anyhow!("template `{template}` references unknown index `{name}`")
            })?;
        if previous && index.value == index.start {
            return Ok(RenderedTemplate::Underflow);
        }
        let value = if previous {
            index.value - 1
        } else {
            index.value
        };
        rendered.push_str(&value.to_string());
        rest = &after_open[close + 1..];
    }
    rendered.push_str(rest);
    Ok(RenderedTemplate::Text(rendered))
}

fn validate_supported_stages(stages: &[StageSpec]) -> Result<()> {
    for stage in stages {
        StageKind::from_stage_spec(&stage.project, &stage.name)?;
    }
    Ok(())
}

fn build_stage_index(stages: &[StageSpec]) -> Result<BTreeMap<String, usize>> {
    let mut stage_index = BTreeMap::new();
    for (idx, stage) in stages.iter().enumerate() {
        if stage_index.insert(stage.name.clone(), idx).is_some() {
            bail!("chain contains duplicate stage name `{}`", stage.name);
        }
    }
    Ok(stage_index)
}

fn validate_reference_stage(stages: &[StageSpec], raster_stage: &str) -> Result<()> {
    let count = stages
        .iter()
        .filter(|stage| stage.name == raster_stage)
        .count();
    if count != 1 {
        bail!("expected exactly one `{raster_stage}` stage in Raster.toml, found {count}");
    }
    let stage = stages
        .iter()
        .find(|stage| stage.name == raster_stage)
        .expect("stage existence checked above");
    StageKind::from_stage_spec(&stage.project, &stage.name)?;
    Ok(())
}

fn dispatch_for_stage(stage: &StageSpec, raster_stage: Option<&str>) -> Result<StageDispatch> {
    StageKind::from_stage_spec(&stage.project, &stage.name)?;
    if raster_stage == Some(stage.name.as_str()) {
        Ok(StageDispatch::RasterReference)
    } else {
        Ok(StageDispatch::DirectNative)
    }
}

fn run_one_stage(
    idx: usize,
    stages: &[StageSpec],
    base_dir: &Path,
    chain_dir: &Path,
    stage_index: &BTreeMap<String, usize>,
    current_exe: &Path,
    direct_backend: DirectStageBackend,
    raster_stage: Option<&str>,
    state: &mut ChainRunState,
) -> Result<()> {
    let stage = &stages[idx];
    println!(
        "▸ stage {}/{}  {}   ({})",
        idx + 1,
        stages.len(),
        stage.name,
        stage.project
    );

    let stage_dir = chain_dir.join(&stage.name);
    fs::create_dir_all(&stage_dir)
        .with_context(|| format!("failed to create {}", stage_dir.display()))?;

    let (input_json_path, input_manifest_path) = synthesize_inputs(
        stage,
        &stage_dir,
        base_dir,
        chain_dir,
        &state.output_commitments,
        stage_index,
    )?;

    let stage_run = run_prepared_stage(
        stage,
        base_dir,
        current_exe,
        direct_backend,
        raster_stage,
        &input_json_path,
        &input_manifest_path,
        &stage_dir,
        stage_index,
        state,
    )?;

    finish_stage(idx, stage, stage_run, stage_dir, state)
}

fn run_prepared_stage(
    stage: &StageSpec,
    base_dir: &Path,
    current_exe: &Path,
    direct_backend: DirectStageBackend,
    raster_stage: Option<&str>,
    input_json_path: &Path,
    input_manifest_path: &Path,
    stage_dir: &Path,
    stage_index: &BTreeMap<String, usize>,
    state: &mut ChainRunState,
) -> Result<DirectStageRun> {
    let dispatch = dispatch_for_stage(stage, raster_stage)?;
    match dispatch {
        StageDispatch::DirectNative => {
            let kind = StageKind::from_stage_spec(&stage.project, &stage.name)?;
            println!("    direct-native {} …", kind.routine());
            let cached_inputs = cached_inputs_for_stage(
                stage,
                stage_index,
                &state.output_commitments,
                &state.output_cache,
            )?;
            run_direct_native_stage(
                direct_backend,
                current_exe,
                &kind,
                &stage.name,
                input_json_path,
                input_manifest_path,
                stage_dir,
                &cached_inputs,
            )
        }
        StageDispatch::RasterReference => {
            let kind = StageKind::from_stage_spec(&stage.project, &stage.name)?;
            println!("    raster reference {} …", kind.routine());
            let duration = run_raster_stage(
                stage,
                base_dir,
                input_json_path,
                input_manifest_path,
                stage_dir,
            )?;
            println!("    direct-native parity check …");
            run_compare_stage(current_exe, stage_dir)?;
            state.selected_stage_dir = Some(stage_dir.to_path_buf());
            Ok(DirectStageRun {
                duration,
                output: None,
            })
        }
    }
}

fn finish_stage(
    idx: usize,
    stage: &StageSpec,
    stage_run: DirectStageRun,
    stage_dir: PathBuf,
    state: &mut ChainRunState,
) -> Result<()> {
    state.execution_times[idx] = Some(stage_run.duration);
    let output = collect_output(&stage_dir)?;
    if let Some(value) = stage_run.output {
        output_cache_insert(
            &mut state.output_cache,
            &stage.name,
            &output.structural_commitment,
            value,
        );
    }
    print_stage_output(&output);
    state.output_commitments[idx] = Some(output.structural_commitment);
    Ok(())
}

fn output_cache_insert(
    output_cache: &mut StageOutputCache,
    stage_name: &str,
    structural_commitment: &[u8],
    value: CachedStageValue,
) {
    output_cache.insert(stage_name, structural_commitment.to_vec(), value);
}

fn aux_wave_range(stages: &[StageSpec], start: usize) -> Option<Range<usize>> {
    let stage = stages.get(start)?;
    if !is_prefill_prepare_aux_stage(stage) {
        return None;
    }
    let end = stages[start..]
        .iter()
        .position(|stage| !is_prefill_prepare_aux_stage(stage))
        .map(|offset| start + offset)
        .unwrap_or(stages.len());
    (end - start > 1).then_some(start..end)
}

fn is_prefill_prepare_aux_stage(stage: &StageSpec) -> bool {
    matches!(
        StageKind::from_stage_spec(&stage.project, &stage.name),
        Ok(StageKind::PrefillPrepareAux { .. })
    )
}

fn run_aux_wave(
    range: Range<usize>,
    stages: &[StageSpec],
    base_dir: &Path,
    chain_dir: &Path,
    stage_index: &BTreeMap<String, usize>,
    current_exe: &Path,
    direct_backend: DirectStageBackend,
    raster_stage: Option<&str>,
    state: &mut ChainRunState,
) -> Result<()> {
    println!(
        "▸ stages {}-{}  prefill_prepare_aux   ({} parallel stages)",
        range.start + 1,
        range.end,
        range.end - range.start
    );

    let mut jobs = Vec::new();
    let mut reference_idx = None;
    for idx in range.clone() {
        let stage = &stages[idx];
        ensure_stage_inputs_ready(stage, &state.output_commitments, stage_index)?;
        match dispatch_for_stage(stage, raster_stage)? {
            StageDispatch::RasterReference => {
                if reference_idx.replace(idx).is_some() {
                    bail!("aux wave cannot contain more than one Raster reference stage");
                }
            }
            StageDispatch::DirectNative => jobs.push(prepare_aux_stage_job(
                idx,
                stages,
                base_dir,
                chain_dir,
                stage_index,
                state,
            )?),
        }
    }

    if let Some(idx) = reference_idx {
        run_one_stage(
            idx,
            stages,
            base_dir,
            chain_dir,
            stage_index,
            current_exe,
            direct_backend,
            raster_stage,
            state,
        )?;
    }

    let batch = run_aux_stage_jobs(current_exe, jobs)?;
    let stage_duration_sum = batch
        .results
        .iter()
        .map(|result| result.duration.as_nanos())
        .sum();
    if let (Some(first), Some(last)) = (batch.results.first(), batch.results.last()) {
        state.aux_waves.push(AuxWaveExecutionTime {
            name: String::from("prefill_prepare_aux"),
            first_stage: first.name.clone(),
            last_stage: last.name.clone(),
            stage_count: batch.results.len(),
            parallelism: batch.parallelism,
            wall_duration_ns: batch.wall_duration.as_nanos(),
            stage_duration_sum_ns: stage_duration_sum,
        });
    }
    for result in batch.results {
        state.execution_times[result.idx] = Some(result.duration);
        print_stage_output(&result.output);
        state.output_commitments[result.idx] = Some(result.output.structural_commitment);
    }

    for idx in range {
        if state.output_commitments[idx].is_none() {
            bail!(
                "aux wave did not produce output for stage `{}`",
                stages[idx].name
            );
        }
    }
    Ok(())
}

fn prepare_aux_stage_job(
    idx: usize,
    stages: &[StageSpec],
    base_dir: &Path,
    chain_dir: &Path,
    stage_index: &BTreeMap<String, usize>,
    state: &ChainRunState,
) -> Result<AuxStageJob> {
    let stage = &stages[idx];
    println!(
        "  queued stage {}/{}  {}   ({})",
        idx + 1,
        stages.len(),
        stage.name,
        stage.project
    );
    let stage_dir = chain_dir.join(&stage.name);
    fs::create_dir_all(&stage_dir)
        .with_context(|| format!("failed to create {}", stage_dir.display()))?;
    let (input_json_path, input_manifest_path) = synthesize_inputs(
        stage,
        &stage_dir,
        base_dir,
        chain_dir,
        &state.output_commitments,
        stage_index,
    )?;
    Ok(AuxStageJob {
        idx,
        name: stage.name.clone(),
        kind: StageKind::from_stage_spec(&stage.project, &stage.name)?,
        input_json_path,
        input_manifest_path,
        stage_dir,
    })
}

fn run_aux_stage_jobs(current_exe: &Path, jobs: Vec<AuxStageJob>) -> Result<AuxStageBatch> {
    if jobs.is_empty() {
        return Ok(AuxStageBatch {
            results: Vec::new(),
            wall_duration: Duration::default(),
            parallelism: 0,
        });
    }
    let parallelism = aux_parallelism(jobs.len());
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(parallelism)
        .build()
        .context("failed to build aux stage worker pool")?;
    let started = Instant::now();
    let mut results = pool.install(|| {
        jobs.into_par_iter()
            .map(|job| run_aux_stage_job(current_exe, job))
            .collect::<Result<Vec<_>>>()
    })?;
    let wall_duration = started.elapsed();
    results.sort_by_key(|result| result.idx);
    Ok(AuxStageBatch {
        results,
        wall_duration,
        parallelism,
    })
}

fn run_aux_stage_job(current_exe: &Path, job: AuxStageJob) -> Result<AuxStageResult> {
    let stage_run = run_direct_native_stage_subprocess(
        current_exe,
        &job.kind,
        &job.input_json_path,
        &job.input_manifest_path,
        &job.stage_dir,
    )
    .with_context(|| format!("failed to run parallel aux stage `{}`", job.name))?;
    let output = collect_output(&job.stage_dir)
        .with_context(|| format!("failed to collect parallel aux stage `{}`", job.name))?;
    Ok(AuxStageResult {
        idx: job.idx,
        name: job.name,
        duration: stage_run.duration,
        output,
    })
}

fn ensure_stage_inputs_ready(
    stage: &StageSpec,
    outputs: &[Option<Vec<u8>>],
    stage_index: &BTreeMap<String, usize>,
) -> Result<()> {
    for (param, binding) in &stage.inputs {
        let InputBinding::From(producer) = binding else {
            continue;
        };
        let producer_idx = *stage_index.get(producer).ok_or_else(|| {
            anyhow::anyhow!(
                "stage '{}': parameter '{param}' is fed from unknown stage '{producer}'",
                stage.name
            )
        })?;
        if outputs.get(producer_idx).and_then(Option::as_ref).is_none() {
            bail!(
                "stage '{}': parameter '{param}' is fed from '{producer}', which has not run",
                stage.name
            );
        }
    }
    Ok(())
}

fn aux_parallelism(job_count: usize) -> usize {
    let requested = std::env::var("DIRECT_NATIVE_AUX_PARALLELISM")
        .ok()
        .and_then(|value| value.parse::<usize>().ok());
    let available = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(4);
    aux_parallelism_from(job_count, requested, available)
}

fn aux_parallelism_from(
    job_count: usize,
    requested: Option<usize>,
    available_parallelism: usize,
) -> usize {
    if job_count == 0 {
        return 0;
    }
    requested
        .filter(|value| *value > 0)
        .unwrap_or_else(|| available_parallelism.clamp(1, 4))
        .clamp(1, job_count)
}

fn cached_inputs_for_stage(
    stage: &StageSpec,
    stage_index: &BTreeMap<String, usize>,
    outputs: &[Option<Vec<u8>>],
    output_cache: &StageOutputCache,
) -> Result<CachedInputs> {
    let mut cached_inputs = CachedInputs::new();
    for (param, binding) in &stage.inputs {
        let InputBinding::From(producer) = binding else {
            continue;
        };
        let Some(producer_idx) = stage_index.get(producer).copied() else {
            continue;
        };
        let Some(expected_commitment) = outputs.get(producer_idx).and_then(Option::as_ref) else {
            continue;
        };
        if let Some(value) = output_cache.get(producer, expected_commitment)? {
            cached_inputs.insert(param.clone(), value);
        }
    }
    Ok(cached_inputs)
}

fn synthesize_inputs(
    stage: &StageSpec,
    stage_dir: &Path,
    base_dir: &Path,
    chain_dir: &Path,
    outputs: &[Option<Vec<u8>>],
    stage_index: &BTreeMap<String, usize>,
) -> Result<(PathBuf, PathBuf)> {
    let mut input_entries: Vec<(String, serde_json::Value)> = Vec::new();
    let mut manifest_entries: Vec<(String, serde_json::Value)> = Vec::new();

    for (param, binding) in &stage.inputs {
        let (path, index_path, commitment) = match binding {
            InputBinding::External(ext) => {
                let path = absolute(base_dir, &ext.path);
                let index_path = ext
                    .index_path
                    .as_ref()
                    .map(|path| absolute(base_dir, path))
                    .unwrap_or_else(|| path.with_extension("rindex"));
                (path, index_path, ext.commitment.clone())
            }
            InputBinding::From(producer) => {
                let producer_idx = *stage_index.get(producer).ok_or_else(|| {
                    anyhow::anyhow!(
                        "stage '{}': parameter '{param}' is fed from stage '{producer}', which has not run",
                        stage.name
                    )
                })?;
                let structural = outputs.get(producer_idx).and_then(Option::as_ref).ok_or_else(|| {
                    anyhow::anyhow!(
                        "stage '{}': parameter '{param}' is fed from stage '{producer}', which has not run",
                        stage.name
                    )
                })?;
                if structural.is_empty() {
                    bail!(
                        "stage '{}': parameter '{param}' is fed from '{producer}', which produced no output",
                        stage.name
                    );
                }
                let producer_dir = chain_dir.join(producer);
                (
                    producer_dir.join("output.bin"),
                    producer_dir.join("output.rindex"),
                    hex::encode(structural),
                )
            }
        };

        input_entries.push((
            param.clone(),
            serde_json::json!({
                "path": path.to_string_lossy(),
                "index_path": index_path.to_string_lossy(),
                "load_preference": load_preference_for_binding(binding),
            }),
        ));
        manifest_entries.push((
            param.clone(),
            serde_json::json!({ "type": "sha256", "encoding": "raster", "commitment": commitment }),
        ));
    }

    let input_json_path = stage_dir.join("input.json");
    let input_manifest_path = stage_dir.join("input_manifest.json");
    fs::write(
        &input_json_path,
        serde_json::to_vec_pretty(&serde_json::Value::Object(
            input_entries.into_iter().collect(),
        ))
        .context("failed to serialize input.json")?,
    )
    .with_context(|| format!("failed to write {}", input_json_path.display()))?;
    fs::write(
        &input_manifest_path,
        serde_json::to_vec_pretty(&serde_json::Value::Object(
            manifest_entries.into_iter().collect(),
        ))
        .context("failed to serialize input_manifest.json")?,
    )
    .with_context(|| format!("failed to write {}", input_manifest_path.display()))?;

    Ok((input_json_path, input_manifest_path))
}

fn load_preference_for_binding(binding: &InputBinding) -> &'static str {
    match binding {
        InputBinding::External(_) => "mmap",
        InputBinding::From(_) => "read",
    }
}

fn run_raster_stage(
    stage: &StageSpec,
    base_dir: &Path,
    input_json_path: &Path,
    input_manifest_path: &Path,
    stage_dir: &Path,
) -> Result<Duration> {
    let project_dir = base_dir.join(&stage.project);
    let manifest_path = project_dir.join("Cargo.toml");
    if !manifest_path.is_file() {
        bail!(
            "stage '{}' has no Cargo manifest at {}",
            stage.name,
            manifest_path.display()
        );
    }

    let mut command = Command::new("cargo");
    command
        .current_dir(&project_dir)
        .args(["run", "--release", "--manifest-path", "Cargo.toml", "--"])
        .arg("--input")
        .arg(input_json_path)
        .arg("--input-manifest")
        .arg(input_manifest_path)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());
    apply_stage_env(&mut command, stage_dir);

    run_timed_command(command, &format!("Raster stage '{}'", stage.name))
}

fn run_direct_native_stage(
    backend: DirectStageBackend,
    current_exe: &Path,
    kind: &StageKind,
    stage_name: &str,
    input_json_path: &Path,
    input_manifest_path: &Path,
    stage_dir: &Path,
    cached_inputs: &CachedInputs,
) -> Result<DirectStageRun> {
    match backend {
        DirectStageBackend::InProcess => run_direct_native_stage_in_process(
            kind,
            stage_name,
            input_json_path,
            input_manifest_path,
            stage_dir,
            cached_inputs,
        ),
        DirectStageBackend::Subprocess => run_direct_native_stage_subprocess(
            current_exe,
            kind,
            input_json_path,
            input_manifest_path,
            stage_dir,
        ),
    }
}

fn run_direct_native_stage_subprocess(
    current_exe: &Path,
    kind: &StageKind,
    input_json_path: &Path,
    input_manifest_path: &Path,
    stage_dir: &Path,
) -> Result<DirectStageRun> {
    let mut command = Command::new(current_exe);
    command
        .arg("--run-stage")
        .arg(stage_dir)
        .arg("--input")
        .arg(input_json_path)
        .arg("--input-manifest")
        .arg(input_manifest_path)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());
    apply_stage_env(&mut command, stage_dir);

    Ok(DirectStageRun {
        duration: run_timed_command(command, &format!("direct-native {} stage", kind.routine()))?,
        output: None,
    })
}

fn run_direct_native_stage_in_process(
    kind: &StageKind,
    stage_name: &str,
    input_json_path: &Path,
    input_manifest_path: &Path,
    stage_dir: &Path,
    cached_inputs: &CachedInputs,
) -> Result<DirectStageRun> {
    let _env = StageEnvGuard::apply(stage_dir);
    let started = Instant::now();
    let direct = routines::run_and_publish_from_paths(
        kind,
        input_json_path,
        input_manifest_path,
        cached_inputs,
    )?;
    let duration = started.elapsed();
    println!(
        "direct-native {stage_name}: output {} structural={} stage={}",
        direct.artifact.data_path.display(),
        direct.artifact.commitment,
        format_duration(duration),
    );
    Ok(DirectStageRun {
        duration,
        output: Some(direct.output),
    })
}

fn run_compare_stage(current_exe: &Path, stage_dir: &Path) -> Result<()> {
    let prior_report = parity_dir(stage_dir).join("report.json");
    if let Err(error) = fs::remove_file(&prior_report) {
        if error.kind() != std::io::ErrorKind::NotFound {
            return Err(error)
                .with_context(|| format!("failed to remove {}", prior_report.display()));
        }
    }

    let mut command = Command::new(current_exe);
    command
        .arg("--compare-stage")
        .arg(stage_dir)
        .arg("--input")
        .arg(stage_dir.join("input.json"))
        .arg("--input-manifest")
        .arg(stage_dir.join("input_manifest.json"))
        .env(raster_runtime::auth::AUTH_ENV, "0")
        .env_remove(raster_runtime::TRACE_PATH_ENV)
        .env_remove(raster_runtime::TRACE_FORMAT_ENV)
        .env_remove(raster_runtime::OUTPUT_DIR_ENV)
        .env_remove(raster_runtime::PROFILE_PATH_ENV)
        .env_remove(raster_runtime::PROFILE_STREAM_PATH_ENV)
        .env_remove(raster_runtime::PROFILE_RUN_ID_ENV)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());

    let status = command
        .status()
        .context("failed to start direct-native comparison child")?;
    if !status.success() {
        bail!(
            "direct-native parity check for {} failed ({status})",
            stage_dir.display()
        );
    }
    Ok(())
}

fn run_timed_command(mut command: Command, label: &str) -> Result<Duration> {
    let started = Instant::now();
    let status = command
        .status()
        .with_context(|| format!("failed to start {label}"))?;
    let duration = started.elapsed();
    if !status.success() {
        bail!("{label} exited unsuccessfully ({status})");
    }
    Ok(duration)
}

fn collect_output(stage_dir: &Path) -> Result<StageOutput> {
    let output_bin = stage_dir.join("output.bin");
    let bytes = fs::read(&output_bin)
        .with_context(|| format!("failed to read {}", output_bin.display()))?;
    let payload_commitment = Sha256::digest(&bytes).to_vec();
    let structural_commitment = payload_structural_root(&bytes)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "{} is not a well-formed Raster payload",
                output_bin.display()
            )
        })?
        .to_vec();

    let manifest_commitment = read_output_manifest_commitment(stage_dir)?;
    if manifest_commitment != hex::encode(&structural_commitment) {
        bail!(
            "{}: output_manifest commitment {manifest_commitment} disagrees with the recomputed structural root {}",
            stage_dir.display(),
            hex::encode(&structural_commitment)
        );
    }

    Ok(StageOutput {
        payload_commitment,
        structural_commitment,
    })
}

fn read_output_manifest_commitment(stage_dir: &Path) -> Result<String> {
    let path = stage_dir.join("output_manifest.json");
    let text =
        fs::read_to_string(&path).with_context(|| format!("failed to read {}", path.display()))?;
    let doc: serde_json::Value = serde_json::from_str(&text)
        .with_context(|| format!("failed to parse {}", path.display()))?;
    doc.get("output")
        .and_then(|value| value.get("commitment"))
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| anyhow::anyhow!("{} has no output.commitment", path.display()))
}

fn write_execution_times(
    chain_dir: &Path,
    stages: &[StageSpec],
    execution_times: &[Option<Duration>],
    total_wall_duration: Duration,
    aux_waves: &[AuxWaveExecutionTime],
) -> Result<()> {
    let path = chain_dir.join(EXECUTION_TIMES_JSON);
    if execution_times.len() != stages.len() {
        bail!(
            "execution timing slot count {} does not match stage count {}",
            execution_times.len(),
            stages.len()
        );
    }
    let mut total_exec_duration_ns = 0;
    let mut timing_stages = Vec::with_capacity(stages.len());
    for (idx, (stage, duration)) in stages.iter().zip(execution_times).enumerate() {
        let duration = (*duration).ok_or_else(|| {
            anyhow::anyhow!(
                "missing execution timing for stage {} `{}`",
                idx + 1,
                stage.name
            )
        })?;
        total_exec_duration_ns += duration.as_nanos();
        timing_stages.push(StageExecutionTime {
            name: stage.name.clone(),
            exec_duration_ns: duration.as_nanos(),
        });
    }
    let document = ExecutionTimesDocument {
        version: 2,
        stages: timing_stages,
        total_exec_duration_ns,
        total_wall_duration_ns: total_wall_duration.as_nanos(),
        aux_waves: aux_waves.to_vec(),
    };
    fs::write(
        &path,
        serde_json::to_vec_pretty(&document).context("failed to encode execution-times.json")?,
    )
    .with_context(|| format!("failed to write {}", path.display()))
}

fn print_direct_timing_summary(
    stages: &[StageSpec],
    execution_times: &[Option<Duration>],
    total_wall_duration: Duration,
    _aux_waves: &[AuxWaveExecutionTime],
) -> Result<()> {
    let stage_duration_sum = execution_times
        .iter()
        .enumerate()
        .map(|(idx, duration)| {
            duration.ok_or_else(|| {
                anyhow::anyhow!(
                    "missing execution timing for stage {} `{}`",
                    idx + 1,
                    stages[idx].name
                )
            })
        })
        .try_fold(Duration::default(), |sum, duration| {
            duration.map(|duration| sum + duration)
        })?;

    println!("direct-native timing:");
    println!(
        "  elapsed wall time: {}",
        format_duration(total_wall_duration)
    );
    println!(
        "  stage duration sum: {}",
        format_duration(stage_duration_sum)
    );
    Ok(())
}

fn create_chain_dir(base_dir: &Path) -> Result<PathBuf> {
    let root = base_dir
        .join("target")
        .join("direct-native")
        .join("chains-no-auth");
    fs::create_dir_all(&root).with_context(|| format!("failed to create {}", root.display()))?;
    let chain_dir = root.join(chain_run_id());
    fs::create_dir_all(&chain_dir)
        .with_context(|| format!("failed to create {}", chain_dir.display()))?;
    Ok(chain_dir)
}

fn chain_run_id() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{nanos:020}-pid{}", std::process::id())
}

fn chain_run_id_label(chain_dir: &Path) -> String {
    chain_dir
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("?")
        .to_string()
}

fn absolute(base_dir: &Path, path: &str) -> PathBuf {
    let path = Path::new(path);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base_dir.join(path)
    }
}

fn apply_stage_env(command: &mut Command, stage_dir: &Path) {
    command
        .env(raster_runtime::auth::AUTH_ENV, "0")
        .env(raster_runtime::OUTPUT_DIR_ENV, stage_dir)
        .env_remove(raster_runtime::TRACE_PATH_ENV)
        .env_remove(raster_runtime::TRACE_FORMAT_ENV)
        .env_remove(raster_runtime::PROFILE_PATH_ENV)
        .env_remove(raster_runtime::PROFILE_STREAM_PATH_ENV)
        .env_remove(raster_runtime::PROFILE_RUN_ID_ENV);
}

struct StageEnvGuard {
    saved: Vec<(&'static str, Option<OsString>)>,
}

impl StageEnvGuard {
    fn apply(stage_dir: &Path) -> Self {
        let guard = Self::capture(&[
            raster_runtime::auth::AUTH_ENV,
            raster_runtime::OUTPUT_DIR_ENV,
            raster_runtime::TRACE_PATH_ENV,
            raster_runtime::TRACE_FORMAT_ENV,
            raster_runtime::PROFILE_PATH_ENV,
            raster_runtime::PROFILE_STREAM_PATH_ENV,
            raster_runtime::PROFILE_RUN_ID_ENV,
        ]);
        std::env::set_var(raster_runtime::auth::AUTH_ENV, "0");
        std::env::set_var(raster_runtime::OUTPUT_DIR_ENV, stage_dir);
        for name in [
            raster_runtime::TRACE_PATH_ENV,
            raster_runtime::TRACE_FORMAT_ENV,
            raster_runtime::PROFILE_PATH_ENV,
            raster_runtime::PROFILE_STREAM_PATH_ENV,
            raster_runtime::PROFILE_RUN_ID_ENV,
        ] {
            std::env::remove_var(name);
        }
        guard
    }

    fn capture(names: &[&'static str]) -> Self {
        Self {
            saved: names
                .iter()
                .map(|name| (*name, std::env::var_os(name)))
                .collect(),
        }
    }
}

impl Drop for StageEnvGuard {
    fn drop(&mut self) {
        for (name, value) in self.saved.drain(..).rev() {
            match value {
                Some(value) => std::env::set_var(name, value),
                None => std::env::remove_var(name),
            }
        }
    }
}

fn short_hex(bytes: &[u8]) -> String {
    let full = hex::encode(bytes);
    full.chars().take(12).collect::<String>() + "..."
}

fn print_stage_output(output: &StageOutput) {
    println!(
        "    output.bin  payload={}  structural={}",
        short_hex(&output.payload_commitment),
        short_hex(&output.structural_commitment)
    );
    println!();
}

fn format_duration(duration: Duration) -> String {
    format_duration_ns(duration.as_nanos())
}

fn format_duration_ns(ns: u128) -> String {
    if ns < 1_000 {
        format!("{ns}ns")
    } else if ns < 1_000_000 {
        format!("{:.2}µs", ns as f64 / 1_000.0)
    } else if ns < 1_000_000_000 {
        format!("{:.2}ms", ns as f64 / 1_000_000.0)
    } else {
        format!("{:.2}s", ns as f64 / 1_000_000_000.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT_TEMP_DIR: AtomicUsize = AtomicUsize::new(0);

    fn test_base_dir() -> PathBuf {
        let id = NEXT_TEMP_DIR.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "direct-native-hybrid-test-{}-{id}",
            std::process::id()
        ))
    }

    #[test]
    fn real_manifest_dispatches_one_reference_and_remaining_stages_native() {
        let manifest_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("Raster.toml");
        let manifest = read_manifest(&manifest_path).unwrap();
        let mut direct = 0usize;
        let mut reference = 0usize;

        for stage in &manifest.chain.stage {
            match dispatch_for_stage(stage, Some("prefill_range_l13")).unwrap() {
                StageDispatch::DirectNative => direct += 1,
                StageDispatch::RasterReference => reference += 1,
            }
        }

        assert_eq!(manifest.chain.stage.len(), 221);
        assert_eq!(direct, 220);
        assert_eq!(reference, 1);
    }

    #[test]
    fn real_manifest_dispatches_every_stage_native_without_reference() {
        let manifest_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("Raster.toml");
        let manifest = read_manifest(&manifest_path).unwrap();
        let mut direct = 0usize;
        let mut reference = 0usize;

        for stage in &manifest.chain.stage {
            match dispatch_for_stage(stage, None).unwrap() {
                StageDispatch::DirectNative => direct += 1,
                StageDispatch::RasterReference => reference += 1,
            }
        }

        assert_eq!(manifest.chain.stage.len(), 221);
        assert_eq!(direct, 221);
        assert_eq!(reference, 0);
    }

    #[test]
    fn real_manifest_expands_decode_repeat_and_export() {
        let manifest_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("Raster.toml");
        let manifest = read_manifest(&manifest_path).unwrap();
        let index = build_stage_index(&manifest.chain.stage).unwrap();

        assert!(index.contains_key("decode_select_t0"));
        assert!(index.contains_key("decode_embed_t1"));
        assert!(index.contains_key("decode_range_t1_l34"));

        let output = manifest.chain.stage.last().unwrap();
        assert_eq!(output.name, "output_finalize");
        assert_eq!(
            output.inputs.get("edge"),
            Some(&InputBinding::From(String::from("decode_select_t1")))
        );
    }

    #[test]
    fn prefill_only_manifest_resolves_zero_count_decode_export() {
        let manifest_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("Raster.toml.prefill-only");
        let manifest = read_manifest(&manifest_path).unwrap();
        let index = build_stage_index(&manifest.chain.stage).unwrap();

        assert!(!index.contains_key("decode_select_t0"));
        let output = manifest.chain.stage.last().unwrap();
        assert_eq!(output.name, "output_finalize");
        assert_eq!(
            output.inputs.get("edge"),
            Some(&InputBinding::From(String::from("decode_init")))
        );
    }

    #[test]
    fn synthesized_inputs_cover_external_and_chained_bindings() {
        let base = test_base_dir();
        let chain_dir = base.join("run");
        let producer_dir = chain_dir.join("producer");
        let stage_dir = chain_dir.join("consumer");
        fs::create_dir_all(&producer_dir).unwrap();
        fs::create_dir_all(&stage_dir).unwrap();

        let stage = StageSpec {
            name: "consumer".into(),
            project: "consumer-project".into(),
            inputs: BTreeMap::from([
                (
                    "external_arg".into(),
                    InputBinding::External(ExternalRef {
                        path: "external/value.rastered".into(),
                        index_path: None,
                        commitment: "abc123".into(),
                    }),
                ),
                ("chained_arg".into(), InputBinding::From("producer".into())),
            ]),
        };
        let stage_index = BTreeMap::from([("producer".into(), 0usize)]);
        let outputs = vec![Some(vec![0xde, 0xad, 0xbe, 0xef])];

        let (input_json, input_manifest) = synthesize_inputs(
            &stage,
            &stage_dir,
            &base,
            &chain_dir,
            &outputs,
            &stage_index,
        )
        .unwrap();
        let input: serde_json::Value =
            serde_json::from_slice(&fs::read(input_json).unwrap()).unwrap();
        let manifest: serde_json::Value =
            serde_json::from_slice(&fs::read(input_manifest).unwrap()).unwrap();

        assert_eq!(
            input["external_arg"]["path"].as_str(),
            Some(
                base.join("external/value.rastered")
                    .to_string_lossy()
                    .as_ref()
            )
        );
        assert_eq!(
            input["external_arg"]["index_path"].as_str(),
            Some(
                base.join("external/value.rindex")
                    .to_string_lossy()
                    .as_ref()
            )
        );
        assert_eq!(
            input["chained_arg"]["path"].as_str(),
            Some(producer_dir.join("output.bin").to_string_lossy().as_ref())
        );
        assert_eq!(input["external_arg"]["load_preference"], "mmap");
        assert_eq!(input["chained_arg"]["load_preference"], "read");
        assert_eq!(manifest["external_arg"]["commitment"], "abc123");
        assert_eq!(manifest["chained_arg"]["commitment"], "deadbeef");

        fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn cached_inputs_follow_from_bindings_by_commitment() {
        let stage = StageSpec {
            name: "consumer".into(),
            project: "input-embedding".into(),
            inputs: BTreeMap::from([("prompt".into(), InputBinding::From("producer".into()))]),
        };
        let stage_index = BTreeMap::from([("producer".into(), 0usize)]);
        let outputs = vec![Some(vec![0xaa, 0xbb])];
        let mut output_cache = StageOutputCache::default();
        output_cache.insert(
            "producer",
            vec![0xaa, 0xbb],
            CachedStageValue::PromptTokenization(prompt_prepare::input::PromptTokenization {
                token_ids: raster::List::from(vec![1, 2]),
            }),
        );

        let inputs =
            cached_inputs_for_stage(&stage, &stage_index, &outputs, &output_cache).unwrap();

        assert!(inputs.contains_key("prompt"));
    }

    #[test]
    fn real_manifest_identifies_prefill_prepare_aux_wave() {
        let manifest_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("Raster.toml");
        let manifest = read_manifest(&manifest_path).unwrap();

        let range = aux_wave_range(&manifest.chain.stage, 2).unwrap();

        assert_eq!(range, 2..37);
        assert!(aux_wave_range(&manifest.chain.stage, 0).is_none());
        assert!(aux_wave_range(&manifest.chain.stage, 37).is_none());
        for stage in &manifest.chain.stage[range] {
            assert!(is_prefill_prepare_aux_stage(stage));
            assert_eq!(
                stage.inputs.get("embedded"),
                Some(&InputBinding::From(String::from("input_embedding")))
            );
        }
    }

    #[test]
    fn raster_reference_inside_aux_wave_dispatches_once() {
        let manifest_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("Raster.toml");
        let manifest = read_manifest(&manifest_path).unwrap();
        let range = aux_wave_range(&manifest.chain.stage, 2).unwrap();
        let mut direct = 0usize;
        let mut reference = Vec::new();

        for stage in &manifest.chain.stage[range] {
            match dispatch_for_stage(stage, Some("prefill_prepare_aux_l5")).unwrap() {
                StageDispatch::DirectNative => direct += 1,
                StageDispatch::RasterReference => reference.push(stage.name.as_str()),
            }
        }

        assert_eq!(direct, 34);
        assert_eq!(reference, ["prefill_prepare_aux_l5"]);
    }

    #[test]
    fn synthesized_inputs_require_manifest_indexed_producer_output() {
        let base = test_base_dir();
        let chain_dir = base.join("run");
        let stage_dir = chain_dir.join("consumer");
        fs::create_dir_all(&stage_dir).unwrap();

        let stage = StageSpec {
            name: "consumer".into(),
            project: "input-embedding".into(),
            inputs: BTreeMap::from([("prompt".into(), InputBinding::From("producer".into()))]),
        };
        let stage_index = BTreeMap::from([("producer".into(), 3usize)]);
        let outputs = vec![None, None, None, Some(vec![0xaa, 0xbb])];

        let (_input_json, input_manifest) = synthesize_inputs(
            &stage,
            &stage_dir,
            &base,
            &chain_dir,
            &outputs,
            &stage_index,
        )
        .unwrap();
        let manifest: serde_json::Value =
            serde_json::from_slice(&fs::read(input_manifest).unwrap()).unwrap();

        assert_eq!(manifest["prompt"]["commitment"], "aabb");

        fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn synthesized_inputs_reject_missing_producer_output() {
        let base = test_base_dir();
        let chain_dir = base.join("run");
        let stage_dir = chain_dir.join("consumer");
        fs::create_dir_all(&stage_dir).unwrap();

        let stage = StageSpec {
            name: "consumer".into(),
            project: "input-embedding".into(),
            inputs: BTreeMap::from([("prompt".into(), InputBinding::From("producer".into()))]),
        };
        let stage_index = BTreeMap::from([("producer".into(), 0usize)]);
        let outputs = vec![None];

        let error = synthesize_inputs(
            &stage,
            &stage_dir,
            &base,
            &chain_dir,
            &outputs,
            &stage_index,
        )
        .unwrap_err();

        assert!(error.to_string().contains("has not run"));

        fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn aux_parallelism_is_bounded_and_overridable() {
        assert_eq!(aux_parallelism_from(0, None, 8), 0);
        assert_eq!(aux_parallelism_from(35, None, 16), 4);
        assert_eq!(aux_parallelism_from(2, None, 16), 2);
        assert_eq!(aux_parallelism_from(35, Some(6), 16), 6);
        assert_eq!(aux_parallelism_from(35, Some(0), 16), 4);
    }
}
