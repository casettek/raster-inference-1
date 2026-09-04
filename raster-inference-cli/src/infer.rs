use anyhow::{Context, Result};
use staged_infer::{
    ArtifactlessStagedInferenceConfig, ArtifactlessStagedInferenceExecutor, InferenceRunReport,
};

pub fn run_infer() -> Result<InferenceRunReport> {
    ArtifactlessStagedInferenceExecutor.run_with_report(
        ArtifactlessStagedInferenceConfig::from_current_dir()
            .context("failed to build infer configuration")?,
    )
}
