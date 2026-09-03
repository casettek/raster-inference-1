use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use output_finalize::input::GeneratedOutput;

use crate::cache::{CachedStageValue, StageOutputCache};
use crate::hybrid::{self, StageSpec};
use crate::routines::{self, StageKind};

#[derive(Debug, Clone)]
pub struct UnconstrainedInferenceConfig {
    pub base_dir: PathBuf,
    pub manifest_path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InferenceResult {
    pub generated_token_count: u32,
    pub generated_token_ids: Vec<u32>,
    pub generated_token_ids_sha256: String,
    pub generated_text: String,
    pub stop_reason: String,
}

#[derive(Debug, Default)]
pub struct UnconstrainedInferenceExecutor;

struct InferRunState {
    output_commitments: Vec<Option<Vec<u8>>>,
    output_cache: StageOutputCache,
    last_output: Option<CachedStageValue>,
}

struct InferenceTempDir {
    path: PathBuf,
}

impl UnconstrainedInferenceExecutor {
    pub fn run(&self, config: UnconstrainedInferenceConfig) -> Result<InferenceResult> {
        ensure_manifest_exists(&config.manifest_path)?;
        let manifest = hybrid::read_manifest(&config.manifest_path)?;
        hybrid::validate_supported_stages(&manifest.chain.stage)?;
        let stage_index = hybrid::build_stage_index(&manifest.chain.stage)?;
        let temp_dir = InferenceTempDir::create()?;

        let mut state = InferRunState {
            output_commitments: vec![None; manifest.chain.stage.len()],
            output_cache: StageOutputCache::default(),
            last_output: None,
        };

        for (idx, stage) in manifest.chain.stage.iter().enumerate() {
            run_stage(
                idx,
                stage,
                &config.base_dir,
                temp_dir.path(),
                &stage_index,
                &mut state,
            )?;
        }

        match state.last_output {
            Some(CachedStageValue::GeneratedOutput(output)) => Ok(output.into()),
            Some(other) => bail!(
                "infer expected final stage to produce generated output, got {:?}",
                other
            ),
            None => bail!("infer cannot run an empty chain"),
        }
    }
}

impl UnconstrainedInferenceConfig {
    pub fn from_current_dir() -> Result<Self> {
        let base_dir = std::env::current_dir().context("failed to read current directory")?;
        Ok(Self {
            manifest_path: base_dir.join("Raster.toml"),
            base_dir,
        })
    }
}

impl From<GeneratedOutput> for InferenceResult {
    fn from(output: GeneratedOutput) -> Self {
        Self {
            generated_token_count: output.generated_token_count,
            generated_token_ids: output.generated_token_ids.iter().copied().collect(),
            generated_token_ids_sha256: output.generated_token_ids_sha256,
            generated_text: output.generated_text,
            stop_reason: output.stop_reason,
        }
    }
}

fn run_stage(
    idx: usize,
    stage: &StageSpec,
    base_dir: &Path,
    staging_dir: &Path,
    stage_index: &std::collections::BTreeMap<String, usize>,
    state: &mut InferRunState,
) -> Result<()> {
    let stage_dir = staging_dir.join(&stage.name);
    fs::create_dir_all(&stage_dir)
        .with_context(|| format!("failed to create {}", stage_dir.display()))?;

    let cached_inputs = hybrid::cached_inputs_for_stage(
        stage,
        stage_index,
        &state.output_commitments,
        &state.output_cache,
    )?;
    let (input_json_path, input_manifest_path) = hybrid::synthesize_inputs(
        stage,
        &stage_dir,
        base_dir,
        staging_dir,
        &state.output_commitments,
        stage_index,
    )?;
    let kind = StageKind::from_stage_spec(&stage.project, &stage.name)?;
    let direct = routines::run_cached_from_paths(
        &kind,
        &input_json_path,
        &input_manifest_path,
        &cached_inputs,
    )
    .with_context(|| format!("infer failed at stage `{}`", stage.name))?;
    let structural_commitment = hex::decode(&direct.encoded.structural_commitment)
        .with_context(|| format!("stage `{}` produced a non-hex commitment", stage.name))?;

    state.output_cache.insert(
        stage.name.clone(),
        structural_commitment.clone(),
        direct.output.clone(),
    );
    state.output_commitments[idx] = Some(structural_commitment);
    state.last_output = Some(direct.output);
    Ok(())
}

fn ensure_manifest_exists(path: &Path) -> Result<()> {
    if !path.is_file() {
        bail!(
            "infer requires an imported workspace with {}",
            path.display()
        );
    }
    Ok(())
}

impl InferenceTempDir {
    fn create() -> Result<Self> {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .context("system clock is before UNIX_EPOCH")?
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "raster-inference-infer-{}-{nanos}",
            std::process::id()
        ));
        fs::create_dir_all(&path)
            .with_context(|| format!("failed to create {}", path.display()))?;
        Ok(Self { path })
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for InferenceTempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_from_current_dir_points_at_root_manifest() {
        let cwd = std::env::current_dir().unwrap();
        let config = UnconstrainedInferenceConfig::from_current_dir().unwrap();

        assert_eq!(config.base_dir, cwd);
        assert_eq!(config.manifest_path, cwd.join("Raster.toml"));
    }

    #[test]
    fn missing_manifest_is_rejected_before_execution() {
        let base_dir = std::env::temp_dir().join(format!(
            "direct-native-infer-missing-{}",
            std::process::id()
        ));
        fs::create_dir_all(&base_dir).unwrap();
        let manifest_path = base_dir.join("Raster.toml");

        let error = UnconstrainedInferenceExecutor
            .run(UnconstrainedInferenceConfig {
                base_dir: base_dir.clone(),
                manifest_path,
            })
            .unwrap_err();

        assert!(error.to_string().contains("imported workspace"));
        fs::remove_dir_all(base_dir).unwrap();
    }
}
