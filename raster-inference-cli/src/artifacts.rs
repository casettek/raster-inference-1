use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const CHECKPOINT_TRACE_JSON: &str = "checkpoint_trace.json";
pub const CHECKPOINT_HASHES_TXT: &str = "checkpoint_hashes.txt";
pub const CLAIM_BUNDLE_JSON: &str = "claim_bundle.json";
pub const DIVERGENCE_JSON: &str = "divergence.json";
pub const CHALLENGE_TRACE_JSON: &str = "challenge_trace.json";
pub const REPLAY_PACKAGE_JSON: &str = "replay_package.json";
pub const CHALLENGE_BUNDLE_JSON: &str = "challenge_bundle.json";

/// Ordered routine-boundary checkpoints from one checkpointed inference run.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(transparent)]
pub struct CheckpointTrace {
    pub checkpoints: Vec<Checkpoint>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Checkpoint {
    pub stage: String,
    pub input_commitment: String,
    pub output_commitment: String,
    pub output_sha256: String,
}

/// Contract-shaped summary of a proposer claim.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ClaimBundle {
    pub version: u32,
    pub input: ClaimEndpoint,
    pub output: ClaimEndpoint,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ClaimEndpoint {
    pub stage: String,
    pub commitment: String,
}

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
}

pub fn write_claim_artifacts(
    chain_dir: &Path,
    manifest_path: &Path,
) -> Result<(PathBuf, PathBuf, PathBuf)> {
    let trace = build_checkpoint_trace(chain_dir, manifest_path)?;
    let trace_path = chain_dir.join(CHECKPOINT_TRACE_JSON);
    write_json(&trace_path, &trace)?;
    let hashes_path = write_checkpoint_hashes_artifact(chain_dir, &trace)?;

    let first = trace
        .checkpoints
        .first()
        .ok_or_else(|| anyhow::anyhow!("cannot build a claim bundle from an empty trace"))?;
    let last = trace
        .checkpoints
        .last()
        .ok_or_else(|| anyhow::anyhow!("cannot build a claim bundle from an empty trace"))?;
    let bundle = ClaimBundle {
        version: 1,
        input: ClaimEndpoint {
            stage: first.stage.clone(),
            commitment: first.input_commitment.clone(),
        },
        output: ClaimEndpoint {
            stage: last.stage.clone(),
            commitment: last.output_commitment.clone(),
        },
    };
    let bundle_path = chain_dir.join(CLAIM_BUNDLE_JSON);
    write_json(&bundle_path, &bundle)?;
    Ok((trace_path, hashes_path, bundle_path))
}

pub fn read_checkpoint_trace(path: &Path) -> Result<CheckpointTrace> {
    read_json(path)
}

pub fn read_challenge_bundle(path: &Path) -> Result<ChallengeBundle> {
    read_json(path)
}

pub fn write_checkpoint_trace_artifact(
    chain_dir: &Path,
    trace: &CheckpointTrace,
) -> Result<PathBuf> {
    let trace_path = chain_dir.join(CHECKPOINT_TRACE_JSON);
    write_json(&trace_path, trace)?;
    Ok(trace_path)
}

pub fn write_checkpoint_hashes_artifact(
    chain_dir: &Path,
    trace: &CheckpointTrace,
) -> Result<PathBuf> {
    let hashes_path = chain_dir.join(CHECKPOINT_HASHES_TXT);
    let mut output = checkpoint_hashes(trace)?.join("\n");
    if !output.is_empty() {
        output.push('\n');
    }
    fs::write(&hashes_path, output)
        .with_context(|| format!("failed to write {}", hashes_path.display()))?;
    Ok(hashes_path)
}

pub fn checkpoint_hashes(trace: &CheckpointTrace) -> Result<Vec<String>> {
    trace
        .checkpoints
        .iter()
        .map(|checkpoint| {
            let encoded = serde_json::to_vec(checkpoint)
                .context("failed to encode checkpoint for hashing")?;
            Ok(format!("{:x}", Sha256::digest(encoded)))
        })
        .collect()
}

pub fn write_challenge_artifacts(
    challenge_dir: &Path,
    source_trace_path: &Path,
    recomputed_trace_path: &Path,
    divergence: &Divergence,
    replay_package: &ReplayPackage,
) -> Result<(PathBuf, PathBuf, PathBuf, PathBuf)> {
    fs::create_dir_all(challenge_dir)
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

pub fn build_checkpoint_trace(chain_dir: &Path, _manifest_path: &Path) -> Result<CheckpointTrace> {
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
        checkpoints.push(read_checkpoint_from_stage_dir(stage, &stage_dir)?);
    }
    checkpoints.sort_by(|left, right| match execution_times.as_ref() {
        Some(timings) => timings
            .order(&left.stage)
            .cmp(&timings.order(&right.stage))
            .then_with(|| left.stage.cmp(&right.stage)),
        None => left.stage.cmp(&right.stage),
    });

    Ok(CheckpointTrace { checkpoints })
}

pub fn read_checkpoint_from_stage_dir(stage: &str, stage_dir: &Path) -> Result<Checkpoint> {
    let input_manifest_path = stage_dir.join("input_manifest.json");
    let output_manifest_path = stage_dir.join("output_manifest.json");
    let output_path = stage_dir.join("output.bin");
    let input_manifest = fs::read(&input_manifest_path)
        .with_context(|| format!("failed to read {}", input_manifest_path.display()))?;
    let manifest: OutputManifest = serde_json::from_slice(
        &fs::read(&output_manifest_path)
            .with_context(|| format!("failed to read {}", output_manifest_path.display()))?,
    )
    .with_context(|| format!("failed to parse {}", output_manifest_path.display()))?;
    let output = fs::read(&output_path)
        .with_context(|| format!("failed to read {}", output_path.display()))?;
    Ok(Checkpoint {
        stage: stage.to_string(),
        input_commitment: format!("{:x}", Sha256::digest(&input_manifest)),
        output_commitment: manifest.output.commitment,
        output_sha256: format!("{:x}", Sha256::digest(&output)),
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
}

impl ExecutionTimesIndex {
    fn order(&self, stage: &str) -> usize {
        self.order.get(stage).copied().unwrap_or(usize::MAX)
    }
}

impl From<ExecutionTimesDocument> for ExecutionTimesIndex {
    fn from(document: ExecutionTimesDocument) -> Self {
        let mut order = BTreeMap::new();
        for (idx, stage) in document.stages.into_iter().enumerate() {
            order.insert(stage.name, idx);
        }
        Self { order }
    }
}

fn read_json<T>(path: &Path) -> Result<T>
where
    T: for<'de> Deserialize<'de>,
{
    serde_json::from_slice(
        &fs::read(path).with_context(|| format!("failed to read {}", path.display()))?,
    )
    .with_context(|| format!("failed to parse {}", path.display()))
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    fs::write(
        path,
        serde_json::to_vec_pretty(value).context("failed to encode JSON artifact")?,
    )
    .with_context(|| format!("failed to write {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn claim_artifacts_summarize_chain_outputs() {
        let base = temp_dir("claim-artifacts");
        fs::create_dir_all(&base).unwrap();
        write_stage(&base, "stage_a", "aaa", b"stage-a");
        write_stage(&base, "stage_b", "bbb", b"stage-b");
        fs::write(
            base.join(direct_native::shadow::EXECUTION_TIMES_JSON),
            r#"{"version":2,"stages":[{"name":"stage_b","exec_duration_ns":20},{"name":"stage_a","exec_duration_ns":10}],"total_exec_duration_ns":30}"#,
        )
        .unwrap();

        let manifest_path = base.join("Raster.toml");
        fs::write(&manifest_path, "[chain]\nname = \"test\"\n").unwrap();
        let (trace_path, hashes_path, bundle_path) =
            write_claim_artifacts(&base, &manifest_path).unwrap();
        assert_eq!(trace_path.file_name().unwrap(), CHECKPOINT_TRACE_JSON);
        assert_eq!(hashes_path.file_name().unwrap(), CHECKPOINT_HASHES_TXT);
        assert_eq!(bundle_path.file_name().unwrap(), CLAIM_BUNDLE_JSON);

        let trace: CheckpointTrace =
            serde_json::from_slice(&fs::read(&trace_path).unwrap()).unwrap();
        let bundle: ClaimBundle = serde_json::from_slice(&fs::read(&bundle_path).unwrap()).unwrap();

        assert_eq!(trace.checkpoints[0].stage, "stage_b");
        assert_eq!(trace.checkpoints[0].output_commitment, "bbb");
        assert_eq!(trace.checkpoints[1].stage, "stage_a");
        assert_eq!(checkpoint_hashes(&trace).unwrap().len(), 2);
        assert_eq!(fs::read_to_string(&hashes_path).unwrap().lines().count(), 2);
        assert!(trace_path.is_file());
        assert!(hashes_path.is_file());
        let input_commitment = format!("{:x}", Sha256::digest(b"input-manifest"));
        assert_eq!(bundle.input.stage, "stage_b");
        assert_eq!(bundle.input.commitment, input_commitment);
        assert_eq!(
            bundle.output,
            ClaimEndpoint {
                stage: String::from("stage_a"),
                commitment: String::from("aaa")
            }
        );

        fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn checkpoint_trace_without_execution_times_uses_stage_name_order() {
        let base = temp_dir("claim-artifacts-fallback-order");
        fs::create_dir_all(&base).unwrap();
        write_stage(&base, "stage_b", "bbb", b"stage-b");
        write_stage(&base, "stage_a", "aaa", b"stage-a");
        fs::write(base.join("not-a-stage.txt"), b"ignored").unwrap();
        fs::create_dir_all(base.join("scratch")).unwrap();

        let manifest_path = base.join("Raster.toml");
        fs::write(&manifest_path, "[chain]\nname = \"test\"\n").unwrap();
        let trace = build_checkpoint_trace(&base, &manifest_path).unwrap();

        assert_eq!(
            trace
                .checkpoints
                .iter()
                .map(|checkpoint| checkpoint.stage.as_str())
                .collect::<Vec<_>>(),
            ["stage_a", "stage_b"]
        );

        fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn claim_artifacts_round_trip_as_json() {
        let trace = CheckpointTrace {
            checkpoints: vec![Checkpoint {
                stage: String::from("output_finalize"),
                input_commitment: String::from("input"),
                output_commitment: String::from("abc"),
                output_sha256: String::from("deadbeef"),
            }],
        };
        let bundle = ClaimBundle {
            version: 1,
            input: ClaimEndpoint {
                stage: String::from("prompt_prepare"),
                commitment: String::from("input"),
            },
            output: ClaimEndpoint {
                stage: String::from("output_finalize"),
                commitment: String::from("abc"),
            },
        };

        assert_eq!(
            serde_json::from_slice::<CheckpointTrace>(&serde_json::to_vec(&trace).unwrap())
                .unwrap(),
            trace
        );
        assert_eq!(
            serde_json::from_slice::<ClaimBundle>(&serde_json::to_vec(&bundle).unwrap()).unwrap(),
            bundle
        );
    }

    #[test]
    fn challenge_artifacts_summarize_raster_replay() {
        let base = temp_dir("challenge-artifacts");
        let replay_run_dir = base.join("replay");
        fs::create_dir_all(&replay_run_dir).unwrap();
        write_stage(
            &replay_run_dir,
            "stage_a",
            "raster-commitment",
            b"raster-output",
        );
        let stage_dir = replay_run_dir.join("stage_a");
        fs::write(stage_dir.join("input.json"), b"{}").unwrap();
        fs::write(stage_dir.join("input_manifest.json"), b"{}").unwrap();
        fs::write(stage_dir.join("commit.bin"), b"commit").unwrap();

        let divergence = Divergence {
            version: 1,
            checkpoint_index: 0,
            stage: String::from("stage_a"),
            reason: DivergenceReason::OutputCommitment,
            claimed_trace_path: PathBuf::from("claimed.json"),
            recomputed_trace_path: PathBuf::from("recomputed.json"),
            claimed: Some(Checkpoint {
                stage: String::from("stage_a"),
                input_commitment: String::from("input"),
                output_commitment: String::from("claimed"),
                output_sha256: String::from("111"),
            }),
            recomputed: Some(Checkpoint {
                stage: String::from("stage_a"),
                input_commitment: String::from("input"),
                output_commitment: String::from("recomputed"),
                output_sha256: String::from("222"),
            }),
        };
        let replay_package = ReplayPackage {
            version: 1,
            stage: String::from("stage_a"),
            replay_run_dir: replay_run_dir.clone(),
            stage_dir: stage_dir.clone(),
            input_path: stage_dir.join("input.json"),
            input_manifest_path: stage_dir.join("input_manifest.json"),
            output_path: stage_dir.join("output.bin"),
            output_index_path: stage_dir.join("output.rindex"),
            output_manifest_path: stage_dir.join("output_manifest.json"),
            commit_path: stage_dir.join("commit.bin"),
        };

        let (divergence_path, replay_package_path, challenge_trace_path, bundle_path) =
            write_challenge_artifacts(
                &base.join("challenge"),
                Path::new("claimed.json"),
                Path::new("recomputed.json"),
                &divergence,
                &replay_package,
            )
            .unwrap();

        assert_eq!(divergence_path.file_name().unwrap(), DIVERGENCE_JSON);
        assert_eq!(
            replay_package_path.file_name().unwrap(),
            REPLAY_PACKAGE_JSON
        );
        assert_eq!(
            challenge_trace_path.file_name().unwrap(),
            CHALLENGE_TRACE_JSON
        );
        assert_eq!(bundle_path.file_name().unwrap(), CHALLENGE_BUNDLE_JSON);

        let bundle: ChallengeBundle =
            serde_json::from_slice(&fs::read(&bundle_path).unwrap()).unwrap();
        assert_eq!(bundle.stage, "stage_a");
        assert_eq!(bundle.raster_commit_path, replay_package.commit_path);

        let challenge_trace: ChallengeTrace =
            serde_json::from_slice(&fs::read(&challenge_trace_path).unwrap()).unwrap();
        assert_eq!(
            challenge_trace.raster_output_commitment,
            "raster-commitment"
        );

        fs::remove_dir_all(base).unwrap();
    }

    fn write_stage(base: &Path, stage: &str, commitment: &str, output: &[u8]) {
        let stage_dir = base.join(stage);
        fs::create_dir_all(&stage_dir).unwrap();
        fs::write(stage_dir.join("input_manifest.json"), b"input-manifest").unwrap();
        fs::write(stage_dir.join("output.bin"), output).unwrap();
        fs::write(stage_dir.join("output.rindex"), b"index").unwrap();
        fs::write(
            stage_dir.join("output_manifest.json"),
            format!(
                r#"{{"output":{{"type":"sha256","encoding":"raster","commitment":"{commitment}"}}}}"#
            ),
        )
        .unwrap();
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
