use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command as ProcessCommand, ExitCode, ExitStatus};

use anyhow::{bail, Context, Result};
use clap::{Args, Parser, Subcommand};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Debug, Parser)]
#[command(name = "raster-inference")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Run one direct-native stage and publish its canonical output artifact.
    #[arg(long = "run-stage", alias = "run-prefill-range-stage", hide = true)]
    run_stage: Option<PathBuf>,

    /// Compare one already-completed no-auth stage without rerunning the chain.
    #[arg(long = "compare-stage", alias = "compare-prefill-range-stage")]
    compare_stage: Option<PathBuf>,

    #[arg(long, hide = true)]
    input: Option<PathBuf>,

    #[arg(long = "input-manifest", hide = true)]
    input_manifest: Option<PathBuf>,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Model preparation workflows.
    Model {
        #[command(subcommand)]
        command: ModelCommand,
    },
    /// Proposer claim workflows.
    Claim {
        #[command(subcommand)]
        command: ClaimCommand,
    },
    /// Fast unconstrained deterministic inference.
    Infer,
    /// Challenge workflows reserved for the verifier path.
    Challenge {
        #[command(subcommand)]
        command: ChallengeCommand,
    },
    /// Fault proof workflows reserved for the proof path.
    Fault {
        #[command(subcommand)]
        command: FaultCommand,
    },
}

#[derive(Debug, Subcommand)]
enum ModelCommand {
    /// Import a model bundle into committed Raster inference externals.
    Import(ModelImportArgs),
}

#[derive(Debug, Args)]
struct ModelImportArgs {
    #[arg(long = "model")]
    model_dir: PathBuf,
    #[arg(long, default_value = "hello raster")]
    prompt: String,
    #[arg(long = "raw-prompt")]
    raw_prompt: bool,
    #[arg(long, default_value_t = 1)]
    tokens: u32,
    #[arg(long)]
    manifest: Option<PathBuf>,
    #[arg(long = "only-tokenizer")]
    only_tokenizer: bool,
    #[arg(long = "only-layers")]
    only_layers: bool,
    #[arg(long = "only-embedding")]
    only_embedding: bool,
    #[arg(long = "only-ple")]
    only_ple: bool,
}

#[derive(Debug, Subcommand)]
enum ClaimCommand {
    /// Build a direct-native checkpoint claim.
    Build(ClaimBuildArgs),
}

#[derive(Debug, Args)]
struct ClaimBuildArgs {
    /// Use the previous per-stage child-process direct-native runner.
    #[arg(long = "direct-subprocess", hide = true)]
    direct_subprocess: bool,
}

#[derive(Debug, Subcommand)]
enum ChallengeCommand {
    /// Locate the first divergent checkpoint in a claim.
    Locate,
    /// Build a challenge bundle for a divergent claim.
    Build,
}

#[derive(Debug, Subcommand)]
enum FaultCommand {
    /// Prove a bad Raster challenge.
    Prove,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CheckpointTrace {
    pub version: u32,
    pub chain_dir: PathBuf,
    pub manifest_path: PathBuf,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub execution_times_path: Option<PathBuf>,
    pub checkpoints: Vec<Checkpoint>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Checkpoint {
    pub stage: String,
    pub output_commitment: String,
    pub output_path: PathBuf,
    pub output_index_path: PathBuf,
    pub output_manifest_path: PathBuf,
    pub output_sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exec_duration_ns: Option<u128>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ClaimBundle {
    pub version: u32,
    pub executor: String,
    pub raster_stage: Option<String>,
    pub chain_dir: PathBuf,
    pub manifest_path: PathBuf,
    pub checkpoint_trace_path: PathBuf,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub execution_times_path: Option<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub final_output: Option<CheckpointRef>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CheckpointRef {
    pub stage: String,
    pub output_commitment: String,
}

#[derive(Debug, Deserialize)]
struct OutputManifest {
    output: OutputManifestEntry,
}

#[derive(Debug, Deserialize)]
struct OutputManifestEntry {
    commitment: String,
}

#[derive(Debug, Deserialize)]
struct ExecutionTimesDocument {
    stages: Vec<StageExecutionTime>,
}

#[derive(Debug, Deserialize)]
struct StageExecutionTime {
    name: String,
    exec_duration_ns: u128,
}

pub fn run_from_env() -> Result<ExitCode> {
    execute(Cli::parse())
}

pub fn execute_from<I, T>(args: I) -> Result<ExitCode>
where
    I: IntoIterator<Item = T>,
    T: Into<std::ffi::OsString> + Clone,
{
    execute(Cli::try_parse_from(args)?)
}

pub fn parse_for_test<I, T>(args: I) -> Result<()>
where
    I: IntoIterator<Item = T>,
    T: Into<std::ffi::OsString> + Clone,
{
    Cli::try_parse_from(args)?;
    Ok(())
}

fn execute(cli: Cli) -> Result<ExitCode> {
    if let Some(stage_dir) = cli.run_stage.as_ref() {
        require_input_args(&cli)?;
        direct_native::shadow::run_hidden_stage_direct(stage_dir)?;
        return Ok(ExitCode::SUCCESS);
    }

    if let Some(stage_dir) = cli.compare_stage {
        if cli.input.is_none() || cli.input_manifest.is_none() {
            let status = run_hidden_compare(&stage_dir)?;
            return finish_shadow_run(status, &stage_dir);
        }
        let report = direct_native::shadow::run_hidden_stage_compare(&stage_dir)?;
        return Ok(if report.matched {
            ExitCode::SUCCESS
        } else {
            ExitCode::from(1)
        });
    }

    match cli.command {
        Some(Commands::Model {
            command: ModelCommand::Import(args),
        }) => {
            model_import::run_from_args(args.into_model_import_args())
                .map_err(|error| anyhow::anyhow!("{error}"))?;
            Ok(ExitCode::SUCCESS)
        }
        Some(Commands::Claim {
            command: ClaimCommand::Build(args),
        }) => build_claim(args),
        Some(Commands::Infer) => bail!("raster-inference infer is not implemented yet"),
        Some(Commands::Challenge { command }) => match command {
            ChallengeCommand::Locate => {
                bail!("raster-inference challenge locate is not implemented yet")
            }
            ChallengeCommand::Build => {
                bail!("raster-inference challenge build is not implemented yet")
            }
        },
        Some(Commands::Fault { command }) => match command {
            FaultCommand::Prove => bail!("raster-inference fault prove is not implemented yet"),
        },
        None => bail!(
            "raster-inference requires a workflow command; try `model import`, `claim build`, or `infer`"
        ),
    }
}

fn build_claim(args: ClaimBuildArgs) -> Result<ExitCode> {
    let exe = std::env::current_exe().context("failed to locate current executable")?;
    let backend = if args.direct_subprocess {
        direct_native::hybrid::DirectStageBackend::Subprocess
    } else {
        direct_native::hybrid::DirectStageBackend::InProcess
    };
    let run = direct_native::hybrid::run(None, &exe, backend)?;
    let manifest_path = std::env::current_dir()
        .context("failed to read current directory")?
        .join("Raster.toml");
    let (trace_path, bundle_path) = write_claim_artifacts(&run.chain_dir, &manifest_path)?;
    println!("checkpoint trace: {}", trace_path.display());
    println!("claim bundle: {}", bundle_path.display());
    Ok(ExitCode::SUCCESS)
}

pub fn write_claim_artifacts(
    chain_dir: &Path,
    manifest_path: &Path,
) -> Result<(PathBuf, PathBuf)> {
    let trace = build_checkpoint_trace(chain_dir, manifest_path)?;
    let trace_path = chain_dir.join("checkpoint_trace.json");
    write_json(&trace_path, &trace)?;

    let final_output = trace.checkpoints.last().map(|checkpoint| CheckpointRef {
        stage: checkpoint.stage.clone(),
        output_commitment: checkpoint.output_commitment.clone(),
    });
    let bundle = ClaimBundle {
        version: 1,
        executor: String::from("checkpointed-direct-native"),
        raster_stage: None,
        chain_dir: chain_dir.to_path_buf(),
        manifest_path: manifest_path.to_path_buf(),
        checkpoint_trace_path: trace_path.clone(),
        execution_times_path: trace.execution_times_path.clone(),
        final_output,
    };
    let bundle_path = chain_dir.join("claim_bundle.json");
    write_json(&bundle_path, &bundle)?;
    Ok((trace_path, bundle_path))
}

pub fn build_checkpoint_trace(chain_dir: &Path, manifest_path: &Path) -> Result<CheckpointTrace> {
    let execution_times_path = chain_dir.join(direct_native::shadow::EXECUTION_TIMES_JSON);
    let execution_times = if execution_times_path.is_file() {
        Some(read_execution_times(&execution_times_path)?)
    } else {
        None
    };

    let mut checkpoints = Vec::new();
    for entry in fs::read_dir(chain_dir)
        .with_context(|| format!("failed to read chain directory {}", chain_dir.display()))?
    {
        let entry =
            entry.with_context(|| format!("failed to read entry in {}", chain_dir.display()))?;
        let stage_dir = entry.path();
        if !stage_dir.is_dir() {
            continue;
        }
        let Some(stage) = stage_dir.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        let output_manifest_path = stage_dir.join("output_manifest.json");
        if !output_manifest_path.is_file() {
            continue;
        }
        let output_path = stage_dir.join("output.bin");
        let output_index_path = stage_dir.join("output.rindex");
        let manifest: OutputManifest = serde_json::from_slice(
            &fs::read(&output_manifest_path)
                .with_context(|| format!("failed to read {}", output_manifest_path.display()))?,
        )
        .with_context(|| format!("failed to parse {}", output_manifest_path.display()))?;
        let output = fs::read(&output_path)
            .with_context(|| format!("failed to read {}", output_path.display()))?;
        checkpoints.push(Checkpoint {
            stage: stage.to_string(),
            output_commitment: manifest.output.commitment,
            output_path,
            output_index_path,
            output_manifest_path,
            output_sha256: format!("{:x}", Sha256::digest(&output)),
            exec_duration_ns: execution_times
                .as_ref()
                .and_then(|timings| timings.get(stage).copied()),
        });
    }
    checkpoints.sort_by_key(|checkpoint| {
        execution_times
            .as_ref()
            .and_then(|timings| timings.get(&checkpoint.stage).copied())
            .is_none()
    });
    checkpoints.sort_by(|left, right| match execution_times.as_ref() {
        Some(timings) => timings
            .order(&left.stage)
            .cmp(&timings.order(&right.stage))
            .then_with(|| left.stage.cmp(&right.stage)),
        None => left.stage.cmp(&right.stage),
    });

    Ok(CheckpointTrace {
        version: 1,
        chain_dir: chain_dir.to_path_buf(),
        manifest_path: manifest_path.to_path_buf(),
        execution_times_path: execution_times.map(|_| execution_times_path),
        checkpoints,
    })
}

fn read_execution_times(path: &Path) -> Result<ExecutionTimesIndex> {
    let document: ExecutionTimesDocument = serde_json::from_slice(
        &fs::read(path).with_context(|| format!("failed to read {}", path.display()))?,
    )
    .with_context(|| format!("failed to parse {}", path.display()))?;
    Ok(ExecutionTimesIndex::from(document))
}

struct ExecutionTimesIndex {
    order: BTreeMap<String, usize>,
    durations: BTreeMap<String, u128>,
}

impl ExecutionTimesIndex {
    fn get(&self, stage: &str) -> Option<&u128> {
        self.durations.get(stage)
    }

    fn order(&self, stage: &str) -> usize {
        self.order.get(stage).copied().unwrap_or(usize::MAX)
    }
}

impl From<ExecutionTimesDocument> for ExecutionTimesIndex {
    fn from(document: ExecutionTimesDocument) -> Self {
        let mut order = BTreeMap::new();
        let mut durations = BTreeMap::new();
        for (idx, stage) in document.stages.into_iter().enumerate() {
            order.insert(stage.name.clone(), idx);
            durations.insert(stage.name, stage.exec_duration_ns);
        }
        Self { order, durations }
    }
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    fs::write(
        path,
        serde_json::to_vec_pretty(value).context("failed to encode JSON artifact")?,
    )
    .with_context(|| format!("failed to write {}", path.display()))
}

fn require_input_args(cli: &Cli) -> Result<()> {
    if cli.input.is_none() || cli.input_manifest.is_none() {
        bail!("hidden stage mode requires --input and --input-manifest");
    }
    Ok(())
}

fn finish_shadow_run(status: ExitStatus, stage_dir: &Path) -> Result<ExitCode> {
    match direct_native::shadow::read_shadow_report(stage_dir) {
        Ok(report) => {
            let chain_dir = stage_dir
                .parent()
                .ok_or_else(|| anyhow::anyhow!("selected stage has no chain run directory"))?;
            print!("{}", render_timing_summary_or_warning(chain_dir, &report));
        }
        Err(error) if status.success() => return Err(error),
        Err(_) => {}
    }
    Ok(exit_code_from_status(status))
}

fn render_timing_summary_or_warning(
    chain_dir: &Path,
    report: &direct_native::shadow::ShadowReport,
) -> String {
    direct_native::shadow::read_chain_execution_times(chain_dir)
        .and_then(|timings| direct_native::shadow::render_timing_summary(&timings, report))
        .unwrap_or_else(|error| format!("\ntiming summary unavailable: {error:#}\n"))
}

fn run_hidden_compare(stage_dir: &Path) -> Result<ExitStatus> {
    let input = stage_dir.join("input.json");
    let input_manifest = stage_dir.join("input_manifest.json");
    let prior_report = direct_native::shadow::parity_dir(stage_dir).join("report.json");
    if let Err(error) = fs::remove_file(&prior_report) {
        if error.kind() != std::io::ErrorKind::NotFound {
            return Err(error)
                .with_context(|| format!("failed to remove {}", prior_report.display()));
        }
    }
    let exe = std::env::current_exe().context("failed to locate current executable")?;
    let status = comparison_command(&exe, stage_dir, &input, &input_manifest)
        .status()
        .context("failed to start direct-native comparison child")?;
    Ok(status)
}

fn comparison_command(
    exe: &Path,
    stage_dir: &Path,
    input: &Path,
    input_manifest: &Path,
) -> ProcessCommand {
    let mut command = ProcessCommand::new(exe);
    command
        .arg("--compare-stage")
        .arg(stage_dir)
        .arg("--input")
        .arg(input)
        .arg("--input-manifest")
        .arg(input_manifest)
        .env(raster_runtime::auth::AUTH_ENV, "0")
        .env_remove(raster_runtime::TRACE_PATH_ENV)
        .env_remove(raster_runtime::TRACE_FORMAT_ENV)
        .env_remove(raster_runtime::OUTPUT_DIR_ENV)
        .env_remove(raster_runtime::PROFILE_PATH_ENV)
        .env_remove(raster_runtime::PROFILE_STREAM_PATH_ENV)
        .env_remove(raster_runtime::PROFILE_RUN_ID_ENV);
    command
}

fn exit_code_from_status(status: ExitStatus) -> ExitCode {
    match status.code() {
        Some(code) => ExitCode::from(code.clamp(0, u8::MAX as i32) as u8),
        None => ExitCode::from(1),
    }
}

impl ModelImportArgs {
    fn into_model_import_args(self) -> Vec<String> {
        let mut args = vec![
            String::from("--model"),
            self.model_dir.to_string_lossy().into_owned(),
            String::from("--prompt"),
            self.prompt,
            String::from("--tokens"),
            self.tokens.to_string(),
        ];
        if self.raw_prompt {
            args.push(String::from("--raw-prompt"));
        }
        if let Some(manifest) = self.manifest {
            args.push(String::from("--manifest"));
            args.push(manifest.to_string_lossy().into_owned());
        }
        if self.only_tokenizer {
            args.push(String::from("--only-tokenizer"));
        }
        if self.only_layers {
            args.push(String::from("--only-layers"));
        }
        if self.only_embedding {
            args.push(String::from("--only-embedding"));
        }
        if self.only_ple {
            args.push(String::from("--only-ple"));
        }
        args
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn parses_phase_one_workflows() {
        parse_for_test([
            "raster-inference",
            "model",
            "import",
            "--model",
            "fixtures/model",
            "--prompt",
            "hello",
            "--tokens",
            "2",
        ])
        .unwrap();
        parse_for_test(["raster-inference", "claim", "build"]).unwrap();
        parse_for_test(["raster-inference", "infer"]).unwrap();
    }

    #[test]
    fn infer_is_reserved_for_later() {
        let error = execute_from(["raster-inference", "infer"]).unwrap_err();

        assert!(error.to_string().contains("not implemented yet"));
    }

    #[test]
    fn model_import_args_forward_existing_flags() {
        let args = ModelImportArgs {
            model_dir: PathBuf::from("model"),
            prompt: String::from("hi"),
            raw_prompt: true,
            tokens: 3,
            manifest: Some(PathBuf::from("Raster.toml")),
            only_tokenizer: true,
            only_layers: false,
            only_embedding: true,
            only_ple: false,
        }
        .into_model_import_args();

        assert_eq!(
            args,
            [
                "--model",
                "model",
                "--prompt",
                "hi",
                "--tokens",
                "3",
                "--raw-prompt",
                "--manifest",
                "Raster.toml",
                "--only-tokenizer",
                "--only-embedding",
            ]
            .map(String::from)
            .to_vec()
        );
    }

    #[test]
    fn claim_artifacts_summarize_chain_outputs() {
        let base = temp_dir("claim-artifacts");
        fs::create_dir_all(base.join("stage_a")).unwrap();
        fs::create_dir_all(base.join("stage_b")).unwrap();
        fs::write(base.join("stage_a").join("output.bin"), b"stage-a").unwrap();
        fs::write(base.join("stage_a").join("output.rindex"), b"index-a").unwrap();
        fs::write(
            base.join("stage_a").join("output_manifest.json"),
            r#"{"output":{"type":"sha256","encoding":"raster","commitment":"aaa"}}"#,
        )
        .unwrap();
        fs::write(base.join("stage_b").join("output.bin"), b"stage-b").unwrap();
        fs::write(base.join("stage_b").join("output.rindex"), b"index-b").unwrap();
        fs::write(
            base.join("stage_b").join("output_manifest.json"),
            r#"{"output":{"type":"sha256","encoding":"raster","commitment":"bbb"}}"#,
        )
        .unwrap();
        fs::write(
            base.join(direct_native::shadow::EXECUTION_TIMES_JSON),
            r#"{"version":2,"stages":[{"name":"stage_b","exec_duration_ns":20},{"name":"stage_a","exec_duration_ns":10}],"total_exec_duration_ns":30}"#,
        )
        .unwrap();

        let manifest_path = base.join("Raster.toml");
        fs::write(&manifest_path, "[chain]\nname = \"test\"\n").unwrap();
        let (trace_path, bundle_path) = write_claim_artifacts(&base, &manifest_path).unwrap();
        let trace: CheckpointTrace =
            serde_json::from_slice(&fs::read(trace_path).unwrap()).unwrap();
        let bundle: ClaimBundle =
            serde_json::from_slice(&fs::read(bundle_path).unwrap()).unwrap();

        assert_eq!(trace.checkpoints[0].stage, "stage_b");
        assert_eq!(trace.checkpoints[0].exec_duration_ns, Some(20));
        assert_eq!(trace.checkpoints[1].stage, "stage_a");
        assert_eq!(
            bundle.final_output,
            Some(CheckpointRef {
                stage: String::from("stage_a"),
                output_commitment: String::from("aaa")
            })
        );

        fs::remove_dir_all(base).unwrap();
    }

    fn temp_dir(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "raster-inference-cli-{label}-{}-{nanos}",
            std::process::id()
        ))
    }
}
