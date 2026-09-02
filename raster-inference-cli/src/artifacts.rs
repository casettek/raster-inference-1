use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const CHECKPOINT_TRACE_JSON: &str = "checkpoint_trace.json";
pub const CLAIM_BUNDLE_JSON: &str = "claim_bundle.json";
pub const CLAIM_EXECUTOR: &str = "checkpointed-direct-native";

/// Ordered routine-boundary checkpoints from one checkpointed inference run.
///
/// Paths are local filesystem paths as written by the current repo run. A later
/// packaging step can make claims portable without changing this local trace.
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

/// Small entrypoint artifact for a proposer claim.
///
/// The bundle identifies the executor and points at the trace that carries the
/// full checkpoint list. It intentionally stays local and lightweight for now.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ClaimBundle {
    pub version: u32,
    pub executor: String,
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

pub fn write_claim_artifacts(chain_dir: &Path, manifest_path: &Path) -> Result<(PathBuf, PathBuf)> {
    let trace = build_checkpoint_trace(chain_dir, manifest_path)?;
    let trace_path = chain_dir.join(CHECKPOINT_TRACE_JSON);
    write_json(&trace_path, &trace)?;

    let final_output = trace.checkpoints.last().map(|checkpoint| CheckpointRef {
        stage: checkpoint.stage.clone(),
        output_commitment: checkpoint.output_commitment.clone(),
    });
    let bundle = ClaimBundle {
        version: 1,
        executor: String::from(CLAIM_EXECUTOR),
        chain_dir: chain_dir.to_path_buf(),
        manifest_path: manifest_path.to_path_buf(),
        checkpoint_trace_path: trace_path.clone(),
        execution_times_path: trace.execution_times_path.clone(),
        final_output,
    };
    let bundle_path = chain_dir.join(CLAIM_BUNDLE_JSON);
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
        let (trace_path, bundle_path) = write_claim_artifacts(&base, &manifest_path).unwrap();
        assert_eq!(trace_path.file_name().unwrap(), CHECKPOINT_TRACE_JSON);
        assert_eq!(bundle_path.file_name().unwrap(), CLAIM_BUNDLE_JSON);

        let trace: CheckpointTrace =
            serde_json::from_slice(&fs::read(&trace_path).unwrap()).unwrap();
        let bundle: ClaimBundle = serde_json::from_slice(&fs::read(&bundle_path).unwrap()).unwrap();

        assert_eq!(trace.checkpoints[0].stage, "stage_b");
        assert_eq!(trace.checkpoints[0].output_commitment, "bbb");
        assert_eq!(trace.checkpoints[0].exec_duration_ns, Some(20));
        assert_eq!(trace.checkpoints[1].stage, "stage_a");
        assert_eq!(trace.checkpoints[1].exec_duration_ns, Some(10));
        assert_eq!(bundle.executor, CLAIM_EXECUTOR);
        assert_eq!(bundle.checkpoint_trace_path, trace_path);
        assert_eq!(
            bundle.final_output,
            Some(CheckpointRef {
                stage: String::from("stage_a"),
                output_commitment: String::from("aaa")
            })
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

        assert_eq!(trace.execution_times_path, None);
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
            version: 1,
            chain_dir: PathBuf::from("run"),
            manifest_path: PathBuf::from("Raster.toml"),
            execution_times_path: Some(PathBuf::from("execution-times.json")),
            checkpoints: vec![Checkpoint {
                stage: String::from("output_finalize"),
                output_commitment: String::from("abc"),
                output_path: PathBuf::from("output_finalize/output.bin"),
                output_index_path: PathBuf::from("output_finalize/output.rindex"),
                output_manifest_path: PathBuf::from("output_finalize/output_manifest.json"),
                output_sha256: String::from("deadbeef"),
                exec_duration_ns: Some(42),
            }],
        };
        let bundle = ClaimBundle {
            version: 1,
            executor: String::from(CLAIM_EXECUTOR),
            chain_dir: trace.chain_dir.clone(),
            manifest_path: trace.manifest_path.clone(),
            checkpoint_trace_path: PathBuf::from(CHECKPOINT_TRACE_JSON),
            execution_times_path: trace.execution_times_path.clone(),
            final_output: Some(CheckpointRef {
                stage: String::from("output_finalize"),
                output_commitment: String::from("abc"),
            }),
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

    fn write_stage(base: &Path, stage: &str, commitment: &str, output: &[u8]) {
        let stage_dir = base.join(stage);
        fs::create_dir_all(&stage_dir).unwrap();
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
