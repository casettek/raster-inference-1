use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

use crate::hybrid::{self, DirectStageBackend};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParityPolicy {
    Skip,
    ReferenceStage(String),
}

#[derive(Debug, Clone)]
pub struct CheckpointedInferenceConfig {
    pub base_dir: PathBuf,
    pub manifest_path: PathBuf,
    pub current_exe: PathBuf,
    pub direct_backend: DirectStageBackend,
    pub parity_policy: ParityPolicy,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointedInferenceResult {
    pub chain_dir: PathBuf,
    pub selected_stage_dir: Option<PathBuf>,
}

#[derive(Debug, Default)]
pub struct CheckpointedInferenceExecutor;

impl CheckpointedInferenceExecutor {
    pub fn run(&self, config: CheckpointedInferenceConfig) -> Result<CheckpointedInferenceResult> {
        ensure_supported_manifest_path(&config.base_dir, &config.manifest_path)?;
        let _cwd = CurrentDirGuard::enter(&config.base_dir)?;
        let raster_stage = match &config.parity_policy {
            ParityPolicy::Skip => None,
            ParityPolicy::ReferenceStage(stage) => Some(stage.as_str()),
        };
        let run = hybrid::run(raster_stage, &config.current_exe, config.direct_backend)?;
        Ok(CheckpointedInferenceResult {
            chain_dir: run.chain_dir,
            selected_stage_dir: run.selected_stage_dir,
        })
    }
}

fn ensure_supported_manifest_path(base_dir: &Path, manifest_path: &Path) -> Result<()> {
    let expected = base_dir.join("Raster.toml");
    if manifest_path != expected {
        bail!(
            "checkpointed inference currently requires manifest path {}; got {}",
            expected.display(),
            manifest_path.display()
        );
    }
    Ok(())
}

struct CurrentDirGuard {
    saved: PathBuf,
}

impl CurrentDirGuard {
    fn enter(base_dir: &Path) -> Result<Self> {
        let saved = std::env::current_dir().context("failed to read current directory")?;
        std::env::set_current_dir(base_dir)
            .with_context(|| format!("failed to enter {}", base_dir.display()))?;
        Ok(Self { saved })
    }
}

impl Drop for CurrentDirGuard {
    fn drop(&mut self) {
        let _ = std::env::set_current_dir(&self.saved);
    }
}

impl CheckpointedInferenceConfig {
    pub fn from_current_dir(
        current_exe: PathBuf,
        direct_backend: DirectStageBackend,
    ) -> Result<Self> {
        let base_dir = std::env::current_dir().context("failed to read current directory")?;
        Ok(Self {
            manifest_path: base_dir.join("Raster.toml"),
            base_dir,
            current_exe,
            direct_backend,
            parity_policy: ParityPolicy::Skip,
        })
    }

    pub fn reference_stage(mut self, stage: impl Into<String>) -> Self {
        self.parity_policy = ParityPolicy::ReferenceStage(stage.into());
        self
    }

    pub fn with_current_exe_from_env(direct_backend: DirectStageBackend) -> Result<Self> {
        Self::from_current_dir(
            std::env::current_exe().context("failed to locate current executable")?,
            direct_backend,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_from_current_dir_points_at_root_manifest() {
        let cwd = std::env::current_dir().unwrap();
        let config = CheckpointedInferenceConfig::from_current_dir(
            PathBuf::from("bin"),
            DirectStageBackend::InProcess,
        )
        .unwrap();

        assert_eq!(config.base_dir, cwd);
        assert_eq!(config.manifest_path, cwd.join("Raster.toml"));
        assert_eq!(config.parity_policy, ParityPolicy::Skip);
    }

    #[test]
    fn manifest_path_must_match_current_hybrid_contract() {
        let error =
            ensure_supported_manifest_path(Path::new("/repo"), Path::new("/repo/other.toml"))
                .unwrap_err();

        assert!(error.to_string().contains("requires manifest path"));
    }
}
