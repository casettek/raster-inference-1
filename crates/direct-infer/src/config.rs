use std::path::PathBuf;

use inference_artifacts::{DIRECT_INFER_ARTIFACTS_DIR, DIRECT_INFER_MANIFEST_JSON};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectInferenceConfig {
    pub base_dir: PathBuf,
    pub direct_manifest_path: PathBuf,
}

impl DirectInferenceConfig {
    pub fn from_current_dir() -> anyhow::Result<Self> {
        let base_dir = std::env::current_dir()?;
        Ok(Self {
            direct_manifest_path: base_dir
                .join(DIRECT_INFER_ARTIFACTS_DIR)
                .join(DIRECT_INFER_MANIFEST_JSON),
            base_dir,
        })
    }
}
