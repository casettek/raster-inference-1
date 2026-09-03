use anyhow::{Context, Result};
use direct_native::{
    InferenceResult, UnconstrainedInferenceConfig, UnconstrainedInferenceExecutor,
};

pub fn run_infer() -> Result<InferenceResult> {
    UnconstrainedInferenceExecutor.run(
        UnconstrainedInferenceConfig::from_current_dir()
            .context("failed to build infer configuration")?,
    )
}
