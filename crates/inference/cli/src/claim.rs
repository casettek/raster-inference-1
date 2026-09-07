use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use staged_infer::chain_runner::StagedExecutionBackend;
use staged_infer::{
    CheckpointedInferenceConfig, CheckpointedInferenceExecutor, CheckpointedInferenceResult,
    ParityPolicy,
};

use inference_artifacts::{
    read_json, read_run_spec, write_claim_artifacts_with_prepared_run, write_json, InferenceResult,
    ModelManifest, PreparedRun, PREPARED_RUN_JSON,
};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone)]
pub struct ClaimBuildOptions {
    pub base_dir: PathBuf,
    pub run_spec_path: PathBuf,
    pub current_exe: PathBuf,
    pub staged_backend: StagedExecutionBackend,
}

struct PreparedClaimRun {
    prepared_run: PreparedRun,
    manifest_path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClaimBuildResult {
    pub chain_dir: PathBuf,
    pub checkpoint_trace_path: PathBuf,
    pub checkpoint_hashes_path: PathBuf,
    pub claim_bundle_path: PathBuf,
    pub final_result: Option<InferenceResult>,
}

pub fn build_claim(options: ClaimBuildOptions) -> Result<ClaimBuildResult> {
    let prepared = prepare_claim_run(&options)?;
    let executor = CheckpointedInferenceExecutor;
    let checkpointed = executor.run(CheckpointedInferenceConfig {
        base_dir: options.base_dir,
        manifest_path: prepared.manifest_path.clone(),
        current_exe: options.current_exe,
        staged_backend: options.staged_backend,
        parity_policy: ParityPolicy::Skip,
    })?;
    write_claim_result(
        checkpointed,
        &prepared.manifest_path,
        &prepared.prepared_run,
    )
}

fn write_claim_result(
    checkpointed: CheckpointedInferenceResult,
    manifest_path: &PathBuf,
    prepared_run: &PreparedRun,
) -> Result<ClaimBuildResult> {
    let prepared_run_path = prepared_run
        .run_manifest_path
        .as_ref()
        .and_then(|path| path.parent())
        .map(|dir| dir.join(PREPARED_RUN_JSON));
    let (checkpoint_trace_path, checkpoint_hashes_path, claim_bundle_path) =
        write_claim_artifacts_with_prepared_run(
            &checkpointed.chain_dir,
            manifest_path,
            prepared_run_path.as_deref(),
        )?;
    Ok(ClaimBuildResult {
        chain_dir: checkpointed.chain_dir,
        checkpoint_trace_path,
        checkpoint_hashes_path,
        claim_bundle_path,
        final_result: checkpointed.final_result,
    })
}

impl ClaimBuildOptions {
    pub fn from_current_dir(
        current_exe: PathBuf,
        staged_backend: StagedExecutionBackend,
        run_spec_path: PathBuf,
    ) -> Result<Self> {
        let base_dir = std::env::current_dir().context("failed to read current directory")?;
        Ok(Self {
            run_spec_path,
            base_dir,
            current_exe,
            staged_backend,
        })
    }
}

fn prepare_claim_run(options: &ClaimBuildOptions) -> Result<PreparedClaimRun> {
    let run_spec_path = absolute(&options.base_dir, &options.run_spec_path);
    let run_spec = read_run_spec(&run_spec_path)?;
    let run_spec_dir = run_spec_path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("run spec has no parent directory"))?;
    let model_manifest_path = absolute(run_spec_dir, &run_spec.model_manifest);
    let model_manifest: ModelManifest = read_json(&model_manifest_path)?;
    let model_manifest_dir = model_manifest_path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("model manifest has no parent directory"))?;
    let tokenizer_path = absolute(model_manifest_dir, &model_manifest.bundle.tokenizer_path);
    let tokenizer: serde_json::Value = read_json(&tokenizer_path)?;
    let prompt = run_prep::prepare_prompt(&tokenizer, run_spec_dir, &run_spec, &model_manifest)?;

    let run_dir = options
        .base_dir
        .join("target")
        .join("raster-inference")
        .join("runs")
        .join(run_id(
            &run_spec_path,
            &prompt.rendered_prompt,
            run_spec.tokens,
        ));
    let prompt_dir = run_dir.join("prompt");
    let (pieces_path, pieces_index_path, pieces_commitment) =
        run_prep::write_initial_pieces(&prompt, &prompt_dir)?;

    let template_path = model_manifest
        .provenance
        .as_ref()
        .map(|provenance| absolute(model_manifest_dir, &provenance.raster_manifest_path))
        .unwrap_or_else(|| options.base_dir.join("Raster.toml"));
    let template = fs::read_to_string(&template_path).with_context(|| {
        format!(
            "failed to read manifest template {}",
            template_path.display()
        )
    })?;
    let run_manifest = render_run_manifest(
        &template,
        &pieces_path,
        &pieces_index_path,
        &pieces_commitment,
        run_spec.tokens,
    )?;
    let manifest_path = run_dir.join("Raster.toml");
    fs::create_dir_all(&run_dir)
        .with_context(|| format!("failed to create {}", run_dir.display()))?;
    fs::write(&manifest_path, &run_manifest)
        .with_context(|| format!("failed to write {}", manifest_path.display()))?;
    let run_manifest_sha256 = format!("{:x}", Sha256::digest(run_manifest.as_bytes()));
    let prepared_run = PreparedRun {
        version: 1,
        run_spec_path,
        model_manifest_path: model_manifest_path.clone(),
        model_manifest_sha256: sha256_file(&model_manifest_path)?,
        prompt,
        tokens: run_spec.tokens,
        run_manifest_path: Some(manifest_path.clone()),
        run_manifest_sha256: Some(run_manifest_sha256),
    };
    write_json(&run_dir.join(PREPARED_RUN_JSON), &prepared_run)?;

    Ok(PreparedClaimRun {
        prepared_run,
        manifest_path,
    })
}

fn render_run_manifest(
    template: &str,
    pieces_path: &Path,
    pieces_index_path: &Path,
    pieces_commitment: &str,
    tokens: u32,
) -> Result<String> {
    let pieces_line = format!(
        "inputs.initial_pieces = {{ external = {{ path = {:?}, index_path = {:?}, commitment = {:?} }} }}",
        pieces_path.to_string_lossy(),
        pieces_index_path.to_string_lossy(),
        pieces_commitment
    );
    let mut replaced_pieces = false;
    let mut replaced_decode_count = false;
    let mut in_decode_repeat = false;
    let mut out = Vec::new();
    let has_initial_pieces_placeholder = template
        .lines()
        .any(|line| line.trim_start().starts_with("inputs.initial_pieces ="));

    for line in template.lines() {
        if line.trim() == "[[chain.repeat]]" {
            in_decode_repeat = true;
        }
        if line.trim_start().starts_with("inputs.initial_pieces =") {
            if replaced_pieces {
                anyhow::bail!("manifest template has multiple initial_pieces bindings");
            }
            out.push(pieces_line.clone());
            replaced_pieces = true;
            continue;
        }
        out.push(line.to_string());
        if !has_initial_pieces_placeholder
            && !replaced_pieces
            && line.trim_start().starts_with("inputs.tokenizer =")
        {
            out.push(pieces_line.clone());
            replaced_pieces = true;
        }
        if in_decode_repeat && line.trim_start().starts_with("count =") {
            *out.last_mut().expect("line was just pushed") = format!("count = {tokens}");
            replaced_decode_count = true;
            in_decode_repeat = false;
        }
    }

    if !replaced_pieces {
        anyhow::bail!("manifest template has no prompt_prepare tokenizer binding");
    }
    if !replaced_decode_count {
        anyhow::bail!("manifest template has no decode repeat count to replace");
    }
    out.push(String::new());
    Ok(out.join("\n"))
}

fn absolute(base: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base.join(path)
    }
}

fn sha256_file(path: &Path) -> Result<String> {
    Ok(format!("{:x}", Sha256::digest(fs::read(path)?)))
}

fn run_id(run_spec_path: &Path, rendered_prompt: &str, tokens: u32) -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let digest = Sha256::digest(format!(
        "{}\0{}\0{}\0{}",
        run_spec_path.display(),
        rendered_prompt,
        tokens,
        nanos
    ));
    hex::encode(digest)[..16].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use inference_artifacts::{CHECKPOINT_TRACE_JSON, CLAIM_BUNDLE_JSON};
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn options_from_current_dir_use_root_manifest() {
        let saved = std::env::current_dir().unwrap();
        let base = temp_dir("claim-options");
        fs::create_dir_all(&base).unwrap();
        let expected_base = base.canonicalize().unwrap();
        std::env::set_current_dir(&base).unwrap();

        let options = ClaimBuildOptions::from_current_dir(
            PathBuf::from("raster-inference"),
            StagedExecutionBackend::InProcess,
            PathBuf::from("inference.toml"),
        )
        .unwrap();

        std::env::set_current_dir(saved).unwrap();
        fs::remove_dir_all(&base).unwrap();

        assert_eq!(options.base_dir, expected_base);
        assert_eq!(options.run_spec_path, PathBuf::from("inference.toml"));
        assert_eq!(options.current_exe, PathBuf::from("raster-inference"));
        assert_eq!(options.staged_backend, StagedExecutionBackend::InProcess);
    }

    #[test]
    fn claim_result_writes_existing_artifact_names() {
        let base = temp_dir("claim-result");
        fs::create_dir_all(&base).unwrap();
        write_stage(&base, "output_finalize", "abc", b"output");
        let manifest_path = base.join("Raster.toml");
        fs::write(&manifest_path, "[chain]\nname = \"test\"\n").unwrap();

        let result = write_claim_result(
            CheckpointedInferenceResult {
                chain_dir: base.clone(),
                selected_stage_dir: None,
                final_result: None,
            },
            &manifest_path,
            &PreparedRun {
                version: 1,
                run_spec_path: base.join("inference.toml"),
                model_manifest_path: base.join("model-artifacts/manifest.json"),
                model_manifest_sha256: String::from("model-sha"),
                prompt: inference_artifacts::PreparedPrompt {
                    resolved_prompt: String::from("hello"),
                    rendered_prompt: String::from("hello"),
                    initial_pieces: vec![String::from("hello"), String::from("</w>")],
                    eos_token_ids: Vec::new(),
                },
                tokens: 1,
                run_manifest_path: Some(manifest_path.clone()),
                run_manifest_sha256: Some(String::from("run-sha")),
            },
        )
        .unwrap();

        assert_eq!(result.chain_dir, base);
        assert_eq!(
            result.checkpoint_trace_path.file_name().unwrap(),
            CHECKPOINT_TRACE_JSON
        );
        assert_eq!(
            result.checkpoint_hashes_path.file_name().unwrap(),
            inference_artifacts::CHECKPOINT_HASHES_TXT
        );
        assert_eq!(
            result.claim_bundle_path.file_name().unwrap(),
            CLAIM_BUNDLE_JSON
        );
        assert_eq!(result.final_result, None);

        fs::remove_dir_all(result.chain_dir).unwrap();
    }

    #[test]
    fn run_manifest_replaces_prompt_artifact_and_decode_count() {
        let template = r#"[chain]
name = "test"

[[chain.stage]]
name = "prompt_prepare"
project = "raster-stages/prompt-prepare"
inputs.tokenizer = { external = { path = "tokenizer.rastered", index_path = "tokenizer.rindex", commitment = "tok" } }
inputs.initial_pieces = { external = { path = "old.rastered", index_path = "old.rindex", commitment = "old" } }

[[chain.repeat]]
name = "decode"
index = "t"
count = 1

  [[chain.repeat.stage]]
name = "decode_select_t{t}"
project = "raster-stages/decode-select-token"
"#;

        let manifest = render_run_manifest(
            template,
            Path::new("/tmp/run/prompt/initial_pieces.rastered"),
            Path::new("/tmp/run/prompt/initial_pieces.rindex"),
            "abc",
            7,
        )
        .unwrap();

        assert!(manifest.contains("path = \"/tmp/run/prompt/initial_pieces.rastered\""));
        assert!(manifest.contains("commitment = \"abc\""));
        assert!(manifest.contains("count = 7"));
        assert!(!manifest.contains("old.rastered"));
        assert_eq!(manifest.matches("inputs.initial_pieces =").count(), 1);
    }

    fn write_stage(base: &PathBuf, stage: &str, commitment: &str, output: &[u8]) {
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
