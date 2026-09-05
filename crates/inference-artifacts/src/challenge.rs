use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::checkpoint::{read_checkpoint_from_stage_dir, Checkpoint};
use crate::io::{read_json, write_json};

pub const DIVERGENCE_JSON: &str = "divergence.json";
pub const CHALLENGE_TRACE_JSON: &str = "challenge_trace.json";
pub const REPLAY_PACKAGE_JSON: &str = "replay_package.json";
pub const CHALLENGE_BUNDLE_JSON: &str = "challenge_bundle.json";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Divergence {
    pub version: u32,
    pub checkpoint_index: usize,
    pub stage: String,
    pub reason: DivergenceReason,
    pub claimed_trace_path: PathBuf,
    pub recomputed_trace_path: PathBuf,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub claimed: Option<Checkpoint>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recomputed: Option<Checkpoint>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DivergenceReason {
    StageName,
    InputCommitment,
    OutputCommitment,
    OutputSha256,
    MissingClaimedCheckpoint,
    MissingRecomputedCheckpoint,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChallengeTrace {
    pub version: u32,
    pub stage: String,
    pub divergence_path: PathBuf,
    pub replay_package_path: PathBuf,
    pub raster_commit_path: PathBuf,
    pub raster_output_commitment: String,
    pub raster_output_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReplayPackage {
    pub version: u32,
    pub stage: String,
    pub replay_run_dir: PathBuf,
    pub stage_dir: PathBuf,
    pub input_path: PathBuf,
    pub input_manifest_path: PathBuf,
    pub output_path: PathBuf,
    pub output_index_path: PathBuf,
    pub output_manifest_path: PathBuf,
    pub commit_path: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChallengeBundle {
    pub version: u32,
    pub stage: String,
    pub source_trace_path: PathBuf,
    pub recomputed_trace_path: PathBuf,
    pub divergence_path: PathBuf,
    pub challenge_trace_path: PathBuf,
    pub replay_package_path: PathBuf,
    pub raster_commit_path: PathBuf,
}

pub fn read_challenge_bundle(path: &Path) -> Result<ChallengeBundle> {
    read_json(path)
}

pub fn write_challenge_artifacts(
    challenge_dir: &Path,
    source_trace_path: &Path,
    recomputed_trace_path: &Path,
    divergence: &Divergence,
    replay_package: &ReplayPackage,
) -> Result<(PathBuf, PathBuf, PathBuf, PathBuf)> {
    std::fs::create_dir_all(challenge_dir)
        .with_context(|| format!("failed to create {}", challenge_dir.display()))?;

    let divergence_path = challenge_dir.join(DIVERGENCE_JSON);
    write_json(&divergence_path, divergence)?;

    let replay_package_path = challenge_dir.join(REPLAY_PACKAGE_JSON);
    write_json(&replay_package_path, replay_package)?;

    let replay_output =
        read_checkpoint_from_stage_dir(&replay_package.stage, &replay_package.stage_dir)?;
    let challenge_trace = ChallengeTrace {
        version: 1,
        stage: replay_package.stage.clone(),
        divergence_path: divergence_path.clone(),
        replay_package_path: replay_package_path.clone(),
        raster_commit_path: replay_package.commit_path.clone(),
        raster_output_commitment: replay_output.output_commitment,
        raster_output_sha256: replay_output.output_sha256,
    };
    let challenge_trace_path = challenge_dir.join(CHALLENGE_TRACE_JSON);
    write_json(&challenge_trace_path, &challenge_trace)?;

    let bundle = ChallengeBundle {
        version: 1,
        stage: replay_package.stage.clone(),
        source_trace_path: source_trace_path.to_path_buf(),
        recomputed_trace_path: recomputed_trace_path.to_path_buf(),
        divergence_path: divergence_path.clone(),
        challenge_trace_path: challenge_trace_path.clone(),
        replay_package_path: replay_package_path.clone(),
        raster_commit_path: replay_package.commit_path.clone(),
    };
    let bundle_path = challenge_dir.join(CHALLENGE_BUNDLE_JSON);
    write_json(&bundle_path, &bundle)?;

    Ok((
        divergence_path,
        replay_package_path,
        challenge_trace_path,
        bundle_path,
    ))
}
