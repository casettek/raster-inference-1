use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command as ProcessCommand, ExitCode, ExitStatus};

use anyhow::{bail, Context, Result};
use clap::{Args, Parser, Subcommand};

mod artifacts;
mod challenge;
mod claim;

pub use artifacts::{
    build_checkpoint_trace, write_claim_artifacts, ChallengeBundle, ChallengeTrace, Checkpoint,
    CheckpointTrace, ClaimBundle, ClaimEndpoint, Divergence, DivergenceReason, ReplayPackage,
    CHALLENGE_BUNDLE_JSON, CHALLENGE_TRACE_JSON, CHECKPOINT_HASHES_TXT, CHECKPOINT_TRACE_JSON,
    CLAIM_BUNDLE_JSON, DIVERGENCE_JSON, REPLAY_PACKAGE_JSON,
};
pub use challenge::{
    build_challenge, ChallengeBuildOptions, ChallengeBuildOutcome, ChallengeBuildResult,
    ChallengeInput,
};
pub use claim::{build_claim, ClaimBuildOptions, ClaimBuildResult};

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
    Build(ChallengeBuildArgs),
}

#[derive(Debug, Subcommand)]
enum FaultCommand {
    /// Prove a bad Raster challenge.
    Prove,
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
            model_import::import_model(args.into_import_config())
                .map_err(|error| anyhow::anyhow!("{error}"))?;
            Ok(ExitCode::SUCCESS)
        }
        Some(Commands::Claim {
            command: ClaimCommand::Build(args),
        }) => {
            warn_if_debug_build("claim build");
            let result = build_claim(args.into_options()?)?;
            println!("checkpoint trace: {}", result.checkpoint_trace_path.display());
            println!("checkpoint hashes: {}", result.checkpoint_hashes_path.display());
            println!("claim bundle: {}", result.claim_bundle_path.display());
            Ok(ExitCode::SUCCESS)
        }
        Some(Commands::Infer) => bail!("raster-inference infer is not implemented yet"),
        Some(Commands::Challenge { command }) => match command {
            ChallengeCommand::Locate => {
                bail!("raster-inference challenge locate is not implemented yet")
            }
            ChallengeCommand::Build(args) => {
                warn_if_debug_build("challenge build");
                let result = build_challenge(args.into_options()?)?;
                print_challenge_result(&result);
                Ok(ExitCode::SUCCESS)
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

#[cfg(debug_assertions)]
fn warn_if_debug_build(workflow: &str) {
    eprintln!(
        "warning: raster-inference {workflow} is running from a debug build; \
         use `cargo run --release --manifest-path raster-inference-cli/Cargo.toml -- {workflow}` \
         for the optimized direct-native path"
    );
}

#[cfg(not(debug_assertions))]
fn warn_if_debug_build(_workflow: &str) {}

#[derive(Debug, Args)]
struct ChallengeBuildArgs {
    /// Existing checkpoint trace to challenge.
    #[arg(long)]
    trace: PathBuf,

    /// Use the previous per-stage child-process direct-native runner.
    #[arg(long = "direct-subprocess", hide = true)]
    direct_subprocess: bool,
}

fn require_input_args(cli: &Cli) -> Result<()> {
    if cli.input.is_none() || cli.input_manifest.is_none() {
        bail!("hidden stage mode requires --input and --input-manifest");
    }
    Ok(())
}

fn print_challenge_result(result: &ChallengeBuildResult) {
    match &result.outcome {
        ChallengeBuildOutcome::NoDivergence {
            claimed_trace_path,
            recomputed_trace_path,
            recomputed_chain_dir,
        } => {
            println!("no divergence found");
            println!("claimed trace: {}", claimed_trace_path.display());
            println!("recomputed trace: {}", recomputed_trace_path.display());
            println!("recomputed chain: {}", recomputed_chain_dir.display());
        }
        ChallengeBuildOutcome::DivergenceBuilt {
            divergence,
            replay_package,
            challenge_bundle_path,
            ..
        } => {
            println!("divergence found: {}", divergence.stage);
            println!("reason: {:?}", divergence.reason);
            if let Some(claimed) = divergence.claimed.as_ref() {
                println!("claimed commitment: {}", claimed.output_commitment);
                println!("claimed sha256: {}", claimed.output_sha256);
            }
            if let Some(recomputed) = divergence.recomputed.as_ref() {
                println!("recomputed commitment: {}", recomputed.output_commitment);
                println!("recomputed sha256: {}", recomputed.output_sha256);
            }
            println!(
                "raster fingerprint: {}",
                replay_package.commit_path.display()
            );
            println!("challenge bundle: {}", challenge_bundle_path.display());
        }
    }
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
    fn into_import_config(self) -> model_import::ImportConfig {
        model_import::ImportConfig {
            model_dir: self.model_dir,
            prompt: self.prompt,
            raw_prompt: self.raw_prompt,
            tokens: self.tokens,
            manifest: self.manifest,
            only_tokenizer: self.only_tokenizer,
            only_layers: self.only_layers,
            only_embedding: self.only_embedding,
            only_ple: self.only_ple,
        }
    }
}

impl ClaimBuildArgs {
    fn into_options(self) -> Result<ClaimBuildOptions> {
        let current_exe = std::env::current_exe().context("failed to locate current executable")?;
        let direct_backend = if self.direct_subprocess {
            direct_native::hybrid::DirectStageBackend::Subprocess
        } else {
            direct_native::hybrid::DirectStageBackend::InProcess
        };
        ClaimBuildOptions::from_current_dir(current_exe, direct_backend)
    }
}

impl ChallengeBuildArgs {
    fn into_options(self) -> Result<ChallengeBuildOptions> {
        let input = ChallengeInput::Trace(self.trace);
        let current_exe = std::env::current_exe().context("failed to locate current executable")?;
        let direct_backend = if self.direct_subprocess {
            direct_native::hybrid::DirectStageBackend::Subprocess
        } else {
            direct_native::hybrid::DirectStageBackend::InProcess
        };
        ChallengeBuildOptions::from_current_dir(current_exe, direct_backend, input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        parse_for_test([
            "raster-inference",
            "challenge",
            "build",
            "--trace",
            "checkpoint_trace.json",
        ])
        .unwrap();
        parse_for_test(["raster-inference", "infer"]).unwrap();
    }

    #[test]
    fn infer_is_reserved_for_later() {
        let error = execute_from(["raster-inference", "infer"]).unwrap_err();

        assert!(error.to_string().contains("not implemented yet"));
    }

    #[test]
    fn model_import_args_build_typed_config() {
        let config = ModelImportArgs {
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
        .into_import_config();

        assert_eq!(
            config,
            model_import::ImportConfig {
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
        );
    }
}
