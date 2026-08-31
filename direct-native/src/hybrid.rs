use std::collections::BTreeMap;
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use raster_core::input::payload_structural_root;
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

#[derive(Debug, Deserialize)]
struct Manifest {
    chain: ChainSpec,
}

#[derive(Debug, Deserialize)]
struct ChainSpec {
    stage: Vec<StageSpec>,
}

#[derive(Debug, Deserialize)]
struct StageSpec {
    name: String,
    project: String,
    #[serde(default)]
    inputs: BTreeMap<String, InputBinding>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum InputBinding {
    External(ExternalRef),
    From(String),
}

#[derive(Debug, Deserialize)]
struct ExternalRef {
    path: String,
    #[serde(default)]
    index_path: Option<String>,
    commitment: String,
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

#[derive(Debug, Serialize)]
struct ExecutionTimesDocument {
    version: u32,
    stages: Vec<StageExecutionTime>,
    total_exec_duration_ns: u128,
}

#[derive(Debug, Serialize)]
struct StageExecutionTime {
    name: String,
    exec_duration_ns: u128,
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
        None => println!("  mode: unauthenticated direct-native (--no-auth; no Raster reference)"),
    }
    println!();

    let mut output_commitments: Vec<Vec<u8>> = Vec::new();
    let mut stage_index: BTreeMap<String, usize> = BTreeMap::new();
    let mut execution_times: Vec<(String, Duration)> = Vec::new();
    let mut output_cache = StageOutputCache::default();
    let mut selected_stage_dir = None;

    for (idx, stage) in manifest.chain.stage.iter().enumerate() {
        println!(
            "▸ stage {}/{}  {}   ({})",
            idx + 1,
            manifest.chain.stage.len(),
            stage.name,
            stage.project
        );

        let stage_dir = chain_dir.join(&stage.name);
        fs::create_dir_all(&stage_dir)
            .with_context(|| format!("failed to create {}", stage_dir.display()))?;

        let (input_json_path, input_manifest_path) = synthesize_inputs(
            stage,
            &stage_dir,
            &base_dir,
            &chain_dir,
            &output_commitments,
            &stage_index,
        )?;

        let dispatch = dispatch_for_stage(stage, raster_stage)?;
        let stage_run = match dispatch {
            StageDispatch::DirectNative => {
                let kind = StageKind::from_stage_spec(&stage.project, &stage.name)?;
                println!("    direct-native {} …", kind.routine());
                let cached_inputs = cached_inputs_for_stage(
                    stage,
                    &stage_index,
                    &output_commitments,
                    &output_cache,
                )?;
                run_direct_native_stage(
                    direct_backend,
                    current_exe,
                    &kind,
                    &stage.name,
                    &input_json_path,
                    &input_manifest_path,
                    &stage_dir,
                    &cached_inputs,
                )?
            }
            StageDispatch::RasterReference => {
                let kind = StageKind::from_stage_spec(&stage.project, &stage.name)?;
                println!("    raster reference {} …", kind.routine());
                let duration = run_raster_stage(
                    stage,
                    &base_dir,
                    &input_json_path,
                    &input_manifest_path,
                    &stage_dir,
                )?;
                println!("    direct-native parity check …");
                run_compare_stage(current_exe, &stage_dir)?;
                selected_stage_dir = Some(stage_dir.clone());
                DirectStageRun {
                    duration,
                    output: None,
                }
            }
        };

        execution_times.push((stage.name.clone(), stage_run.duration));
        let output = collect_output(&stage_dir)?;
        if let Some(value) = stage_run.output {
            output_cache.insert(&stage.name, output.structural_commitment.clone(), value);
        }
        println!(
            "    output.bin  payload={}  structural={}",
            short_hex(&output.payload_commitment),
            short_hex(&output.structural_commitment)
        );
        println!("    (no trace, no commitment - hybrid --no-auth)");
        println!();

        output_commitments.push(output.structural_commitment);
        stage_index.insert(stage.name.clone(), idx);
    }

    write_execution_times(&chain_dir, &execution_times)?;
    if raster_stage.is_some() && selected_stage_dir.is_none() {
        bail!("selected Raster reference stage did not run");
    }
    println!("no chain-commitment written (hybrid --no-auth)");
    if raster_stage.is_none() {
        println!("no Raster reference selected; parity comparison skipped");
    }

    Ok(HybridRun {
        chain_dir,
        selected_stage_dir,
    })
}

fn read_manifest(path: &Path) -> Result<Manifest> {
    let text =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    text.parse::<toml::Value>()
        .context("failed to parse Raster.toml as TOML")?;
    toml::from_str(&text).context("failed to decode Raster.toml chain")
}

fn validate_supported_stages(stages: &[StageSpec]) -> Result<()> {
    for stage in stages {
        StageKind::from_stage_spec(&stage.project, &stage.name)?;
    }
    Ok(())
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

fn cached_inputs_for_stage(
    stage: &StageSpec,
    stage_index: &BTreeMap<String, usize>,
    outputs: &[Vec<u8>],
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
        let Some(expected_commitment) = outputs.get(producer_idx) else {
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
    outputs: &[Vec<u8>],
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
                let structural = outputs.get(producer_idx).ok_or_else(|| {
                    anyhow::anyhow!(
                        "stage '{}': parameter '{param}' refers to missing output for stage '{producer}'",
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

fn write_execution_times(chain_dir: &Path, execution_times: &[(String, Duration)]) -> Result<()> {
    let path = chain_dir.join(EXECUTION_TIMES_JSON);
    let total_exec_duration_ns = execution_times
        .iter()
        .map(|(_, duration)| duration.as_nanos())
        .sum();
    let document = ExecutionTimesDocument {
        version: 1,
        stages: execution_times
            .iter()
            .map(|(name, duration)| StageExecutionTime {
                name: name.clone(),
                exec_duration_ns: duration.as_nanos(),
            })
            .collect(),
        total_exec_duration_ns,
    };
    fs::write(
        &path,
        serde_json::to_vec_pretty(&document).context("failed to encode execution-times.json")?,
    )
    .with_context(|| format!("failed to write {}", path.display()))
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

fn format_duration(duration: Duration) -> String {
    let ns = duration.as_nanos();
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

        assert_eq!(manifest.chain.stage.len(), 74);
        assert_eq!(direct, 73);
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

        assert_eq!(manifest.chain.stage.len(), 74);
        assert_eq!(direct, 74);
        assert_eq!(reference, 0);
    }

    #[test]
    fn synthesized_inputs_cover_external_and_chained_bindings() {
        let base =
            std::env::temp_dir().join(format!("direct-native-hybrid-test-{}", std::process::id()));
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
        let outputs = vec![vec![0xde, 0xad, 0xbe, 0xef]];

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
        let outputs = vec![vec![0xaa, 0xbb]];
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
}
