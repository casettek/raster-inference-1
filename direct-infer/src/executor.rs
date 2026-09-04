use anyhow::{bail, Result};
use inference_artifacts::InferenceResult;

use crate::DirectInferenceConfig;

#[derive(Debug, Default)]
pub struct DirectInferenceExecutor;

impl DirectInferenceExecutor {
    pub fn run(&self, _config: DirectInferenceConfig) -> Result<InferenceResult> {
        bail!(
            "direct-infer is scaffolded but not implemented yet; \
             use the staged artifactless fallback until the native runtime exists"
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_infer_is_scaffolded_but_not_implemented() {
        let error = DirectInferenceExecutor
            .run(DirectInferenceConfig {
                base_dir: ".".into(),
                manifest_path: "direct-infer-artifacts/manifest.json".into(),
            })
            .unwrap_err();

        assert!(error.to_string().contains("not implemented"));
    }
}
