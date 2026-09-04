use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectInferenceConfig {
    pub base_dir: PathBuf,
    pub manifest_path: PathBuf,
}

impl DirectInferenceConfig {
    pub fn from_current_dir() -> anyhow::Result<Self> {
        let base_dir = std::env::current_dir()?;
        Ok(Self {
            manifest_path: base_dir
                .join("direct-infer-artifacts")
                .join("manifest.json"),
            base_dir,
        })
    }
}
