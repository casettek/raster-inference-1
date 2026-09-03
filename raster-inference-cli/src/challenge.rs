use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{bail, Context, Result};
use direct_native::hybrid::DirectStageBackend;
use direct_native::{CheckpointedInferenceConfig, CheckpointedInferenceExecutor, ParityPolicy};

use crate::artifacts::{
    build_checkpoint_trace, read_challenge_bundle, read_checkpoint_trace,
    write_challenge_artifacts, write_checkpoint_trace_artifact, ChallengeBundle, Checkpoint,
    CheckpointTrace, Divergence, DivergenceReason, ReplayPackage,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChallengeInput {
    Trace(PathBuf),
}

#[derive(Debug, Clone)]
pub struct ChallengeBuildOptions {
    pub base_dir: PathBuf,
    pub manifest_path: PathBuf,
    pub current_exe: PathBuf,
    pub direct_backend: DirectStageBackend,
    pub input: ChallengeInput,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChallengeBuildResult {
    pub outcome: ChallengeBuildOutcome,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChallengeBuildOutcome {
    NoDivergence {
        claimed_trace_path: PathBuf,
        recomputed_trace_path: PathBuf,
        recomputed_chain_dir: PathBuf,
    },
    DivergenceBuilt {
        divergence: Divergence,
        divergence_path: PathBuf,
        replay_package: ReplayPackage,
        replay_package_path: PathBuf,
        challenge_trace_path: PathBuf,
        challenge_bundle_path: PathBuf,
        challenge_bundle: ChallengeBundle,
    },
}

pub fn build_challenge(options: ChallengeBuildOptions) -> Result<ChallengeBuildResult> {
    let (claimed_trace_path, claimed_trace) = load_claimed_trace(&options.input)?;
    let executor = CheckpointedInferenceExecutor;
    let recomputed = executor.run(CheckpointedInferenceConfig {
        base_dir: options.base_dir.clone(),
        manifest_path: options.manifest_path.clone(),
        current_exe: options.current_exe,
        direct_backend: options.direct_backend,
        parity_policy: ParityPolicy::Skip,
    })?;
    let recomputed_trace = build_checkpoint_trace(&recomputed.chain_dir, &options.manifest_path)?;
    let recomputed_trace_path =
        write_checkpoint_trace_artifact(&recomputed.chain_dir, &recomputed_trace)?;

    let Some(divergence) = locate_divergence(
        &claimed_trace_path,
        &claimed_trace,
        &recomputed_trace_path,
        &recomputed_trace,
    ) else {
        return Ok(ChallengeBuildResult {
            outcome: ChallengeBuildOutcome::NoDivergence {
                claimed_trace_path,
                recomputed_trace_path,
                recomputed_chain_dir: recomputed.chain_dir,
            },
        });
    };

    let recomputed_checkpoint = divergence.recomputed.as_ref().ok_or_else(|| {
        anyhow::anyhow!(
            "cannot build a Raster challenge for `{}` because the recomputed trace has no matching checkpoint",
            divergence.stage
        )
    })?;
    if recomputed_checkpoint.stage != divergence.stage {
        bail!(
            "cannot build a Raster challenge for stage-name divergence: claimed `{}`, recomputed `{}`",
            divergence.stage,
            recomputed_checkpoint.stage
        );
    }

    let challenge_dir = recomputed
        .chain_dir
        .join("challenge")
        .join(sanitize_stage_name(&divergence.stage));
    let replay_run_dir = challenge_dir.join("raster-replay");
    seed_replay_directory(
        &recomputed.chain_dir,
        &recomputed_trace,
        divergence.checkpoint_index,
        &replay_run_dir,
    )?;
    let replay_package = run_raster_replay(&options.base_dir, &replay_run_dir, &divergence.stage)?;
    let (divergence_path, replay_package_path, challenge_trace_path, challenge_bundle_path) =
        write_challenge_artifacts(
            &challenge_dir,
            &claimed_trace_path,
            &recomputed_trace_path,
            &divergence,
            &replay_package,
        )?;
    let challenge_bundle = read_challenge_bundle(&challenge_bundle_path)?;

    Ok(ChallengeBuildResult {
        outcome: ChallengeBuildOutcome::DivergenceBuilt {
            divergence,
            divergence_path,
            replay_package,
            replay_package_path,
            challenge_trace_path,
            challenge_bundle_path,
            challenge_bundle,
        },
    })
}

pub fn locate_divergence(
    claimed_trace_path: &Path,
    claimed: &CheckpointTrace,
    recomputed_trace_path: &Path,
    recomputed: &CheckpointTrace,
) -> Option<Divergence> {
    let checkpoint_count = claimed.checkpoints.len().max(recomputed.checkpoints.len());
    for index in 0..checkpoint_count {
        let claimed_checkpoint = claimed.checkpoints.get(index);
        let recomputed_checkpoint = recomputed.checkpoints.get(index);
        let Some(reason) = divergence_reason(claimed_checkpoint, recomputed_checkpoint) else {
            continue;
        };
        let stage = claimed_checkpoint
            .or(recomputed_checkpoint)
            .map(|checkpoint| checkpoint.stage.clone())
            .unwrap_or_else(|| String::from("<unknown>"));
        return Some(Divergence {
            version: 1,
            checkpoint_index: index,
            stage,
            reason,
            claimed_trace_path: claimed_trace_path.to_path_buf(),
            recomputed_trace_path: recomputed_trace_path.to_path_buf(),
            claimed: claimed_checkpoint.cloned(),
            recomputed: recomputed_checkpoint.cloned(),
        });
    }
    None
}

fn divergence_reason(
    claimed: Option<&Checkpoint>,
    recomputed: Option<&Checkpoint>,
) -> Option<DivergenceReason> {
    match (claimed, recomputed) {
        (None, None) => None,
        (None, Some(_)) => Some(DivergenceReason::MissingClaimedCheckpoint),
        (Some(_), None) => Some(DivergenceReason::MissingRecomputedCheckpoint),
        (Some(claimed), Some(recomputed)) if claimed.stage != recomputed.stage => {
            Some(DivergenceReason::StageName)
        }
        (Some(claimed), Some(recomputed))
            if claimed.input_commitment != recomputed.input_commitment =>
        {
            Some(DivergenceReason::InputCommitment)
        }
        (Some(claimed), Some(recomputed))
            if claimed.output_commitment != recomputed.output_commitment =>
        {
            Some(DivergenceReason::OutputCommitment)
        }
        (Some(claimed), Some(recomputed)) if claimed.output_sha256 != recomputed.output_sha256 => {
            Some(DivergenceReason::OutputSha256)
        }
        _ => None,
    }
}

fn load_claimed_trace(input: &ChallengeInput) -> Result<(PathBuf, CheckpointTrace)> {
    match input {
        ChallengeInput::Trace(path) => {
            let trace = read_checkpoint_trace(path)?;
            Ok((path.clone(), trace))
        }
    }
}

fn run_raster_replay(base_dir: &Path, replay_run_dir: &Path, stage: &str) -> Result<ReplayPackage> {
    fs::create_dir_all(replay_run_dir)
        .with_context(|| format!("failed to create {}", replay_run_dir.display()))?;
    let status = Command::new("cargo")
        .current_dir(base_dir)
        .args(["raster", "chain", "run", "--quiet-timings", "--run"])
        .arg(replay_run_dir)
        .arg("--stage")
        .arg(stage)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .context("failed to start authenticated Raster stage replay")?;
    if !status.success() {
        bail!("authenticated Raster stage replay for `{stage}` failed ({status})");
    }
    replay_package(stage, replay_run_dir)
}

fn replay_package(stage: &str, replay_run_dir: &Path) -> Result<ReplayPackage> {
    let stage_dir = replay_run_dir.join(stage);
    let package = ReplayPackage {
        version: 1,
        stage: stage.to_string(),
        replay_run_dir: replay_run_dir.to_path_buf(),
        stage_dir: stage_dir.clone(),
        input_path: stage_dir.join("input.json"),
        input_manifest_path: stage_dir.join("input_manifest.json"),
        output_path: stage_dir.join("output.bin"),
        output_index_path: stage_dir.join("output.rindex"),
        output_manifest_path: stage_dir.join("output_manifest.json"),
        commit_path: stage_dir.join("commit.bin"),
    };
    for path in [
        &package.input_path,
        &package.input_manifest_path,
        &package.output_path,
        &package.output_index_path,
        &package.output_manifest_path,
        &package.commit_path,
    ] {
        if !path.is_file() {
            bail!(
                "authenticated Raster replay did not write {}",
                path.display()
            );
        }
    }
    Ok(package)
}

fn seed_replay_directory(
    source_chain_dir: &Path,
    trace: &CheckpointTrace,
    checkpoint_index: usize,
    replay_run_dir: &Path,
) -> Result<()> {
    if replay_run_dir.exists() {
        fs::remove_dir_all(replay_run_dir)
            .with_context(|| format!("failed to reset {}", replay_run_dir.display()))?;
    }
    fs::create_dir_all(replay_run_dir)
        .with_context(|| format!("failed to create {}", replay_run_dir.display()))?;
    for checkpoint in trace.checkpoints.iter().take(checkpoint_index) {
        let source_stage_dir = source_chain_dir.join(&checkpoint.stage);
        copy_dir_recursive(&source_stage_dir, &replay_run_dir.join(&checkpoint.stage))
            .with_context(|| format!("failed to seed producer `{}`", checkpoint.stage))?;
    }
    Ok(())
}

fn copy_dir_recursive(source: &Path, destination: &Path) -> Result<()> {
    fs::create_dir_all(destination)
        .with_context(|| format!("failed to create {}", destination.display()))?;
    for entry in
        fs::read_dir(source).with_context(|| format!("failed to read {}", source.display()))?
    {
        let entry =
            entry.with_context(|| format!("failed to read entry in {}", source.display()))?;
        let source_path = entry.path();
        let destination_path = destination.join(entry.file_name());
        let file_type = entry
            .file_type()
            .with_context(|| format!("failed to inspect {}", source_path.display()))?;
        if file_type.is_dir() {
            copy_dir_recursive(&source_path, &destination_path)?;
        } else if file_type.is_file() {
            fs::copy(&source_path, &destination_path).with_context(|| {
                format!(
                    "failed to copy {} to {}",
                    source_path.display(),
                    destination_path.display()
                )
            })?;
        }
    }
    Ok(())
}

fn sanitize_stage_name(stage: &str) -> String {
    stage
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

impl ChallengeBuildOptions {
    pub fn from_current_dir(
        current_exe: PathBuf,
        direct_backend: DirectStageBackend,
        input: ChallengeInput,
    ) -> Result<Self> {
        let base_dir = std::env::current_dir().context("failed to read current directory")?;
        Ok(Self {
            manifest_path: base_dir.join("Raster.toml"),
            base_dir,
            current_exe,
            direct_backend,
            input,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn identical_traces_have_no_divergence() {
        let trace = trace(vec![checkpoint("stage_a", "aaa", "111")]);

        assert_eq!(
            locate_divergence(
                Path::new("claimed.json"),
                &trace,
                Path::new("recomputed.json"),
                &trace,
            ),
            None
        );
    }

    #[test]
    fn commitment_mismatch_is_first_divergence() {
        let claimed = trace(vec![
            checkpoint("stage_a", "aaa", "111"),
            checkpoint("stage_b", "bbb", "222"),
        ]);
        let recomputed = trace(vec![
            checkpoint("stage_a", "aaa", "111"),
            checkpoint("stage_b", "ccc", "222"),
        ]);

        let divergence = locate_divergence(
            Path::new("claimed.json"),
            &claimed,
            Path::new("recomputed.json"),
            &recomputed,
        )
        .unwrap();

        assert_eq!(divergence.checkpoint_index, 1);
        assert_eq!(divergence.stage, "stage_b");
        assert_eq!(divergence.reason, DivergenceReason::OutputCommitment);
    }

    #[test]
    fn missing_checkpoint_is_divergence() {
        let claimed = trace(vec![checkpoint("stage_a", "aaa", "111")]);
        let recomputed = trace(vec![
            checkpoint("stage_a", "aaa", "111"),
            checkpoint("stage_b", "bbb", "222"),
        ]);

        let divergence = locate_divergence(
            Path::new("claimed.json"),
            &claimed,
            Path::new("recomputed.json"),
            &recomputed,
        )
        .unwrap();

        assert_eq!(divergence.checkpoint_index, 1);
        assert_eq!(
            divergence.reason,
            DivergenceReason::MissingClaimedCheckpoint
        );
    }

    #[test]
    fn replay_seed_copies_prior_stages_without_mutating_source() {
        let base = temp_dir("seed");
        let source_run = base.join("source");
        let replay_run = base.join("replay");
        fs::create_dir_all(&source_run).unwrap();
        write_stage_dir(&source_run, "stage_a");
        write_stage_dir(&source_run, "stage_b");
        let trace = CheckpointTrace {
            checkpoints: vec![
                checkpoint("stage_a", "aaa", "111"),
                checkpoint("stage_b", "bbb", "222"),
            ],
        };

        seed_replay_directory(&source_run, &trace, 1, &replay_run).unwrap();

        assert!(replay_run.join("stage_a").join("output.bin").is_file());
        assert!(!replay_run.join("stage_b").exists());
        assert!(source_run.join("stage_b").join("output.bin").is_file());

        fs::remove_dir_all(base).unwrap();
    }

    fn trace(checkpoints: Vec<Checkpoint>) -> CheckpointTrace {
        CheckpointTrace { checkpoints }
    }

    fn checkpoint(stage: &str, commitment: &str, sha: &str) -> Checkpoint {
        Checkpoint {
            stage: stage.to_string(),
            input_commitment: String::from("input"),
            output_commitment: commitment.to_string(),
            output_sha256: sha.to_string(),
        }
    }

    fn write_stage_dir(base: &Path, stage: &str) {
        let stage_dir = base.join(stage);
        fs::create_dir_all(&stage_dir).unwrap();
        fs::write(stage_dir.join("input.json"), b"{}").unwrap();
        fs::write(stage_dir.join("input_manifest.json"), b"{}").unwrap();
        fs::write(stage_dir.join("output.bin"), b"output").unwrap();
        fs::write(stage_dir.join("output.rindex"), b"index").unwrap();
        fs::write(stage_dir.join("output_manifest.json"), b"manifest").unwrap();
    }

    fn temp_dir(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "raster-inference-cli-challenge-{label}-{}-{nanos}",
            std::process::id()
        ))
    }
}
