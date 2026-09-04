use anyhow::{Context, Result};
use direct_infer::{DirectInferenceConfig, DirectInferenceExecutor};
use staged_infer::InferenceRunReport;

pub fn run_infer() -> Result<InferenceRunReport> {
    DirectInferenceExecutor.run_with_report(
        DirectInferenceConfig::from_current_dir().context("failed to build infer configuration")?,
    )
}
