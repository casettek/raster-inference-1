pub mod challenge;
pub mod checkpoint;
pub mod claim;
pub mod direct;
pub mod inference;
pub mod io;

pub use challenge::{
    read_challenge_bundle, write_challenge_artifacts, ChallengeBundle, ChallengeTrace, Divergence,
    DivergenceReason, ReplayPackage, CHALLENGE_BUNDLE_JSON, CHALLENGE_TRACE_JSON, DIVERGENCE_JSON,
    REPLAY_PACKAGE_JSON,
};
pub use checkpoint::{
    build_checkpoint_trace, checkpoint_hashes, read_checkpoint_from_stage_dir,
    read_checkpoint_trace, write_checkpoint_hashes_artifact, write_checkpoint_trace_artifact,
    Checkpoint, CheckpointTrace, CHECKPOINT_HASHES_TXT, CHECKPOINT_TRACE_JSON,
    EXECUTION_TIMES_JSON,
};
pub use claim::{write_claim_artifacts, ClaimBundle, ClaimEndpoint, CLAIM_BUNDLE_JSON};
pub use direct::{
    DirectInferBundle, DirectInferImportSettings, DirectInferManifest, DirectInferPrompt,
    DirectInferProvenance, DirectInferShape, DIRECT_INFER_ARTIFACTS_DIR,
    DIRECT_INFER_MANIFEST_JSON,
};
pub use inference::{
    InferAuxWaveTiming, InferStageTiming, InferenceResult, InferenceRunReport, InferenceTimings,
};
pub use io::{read_json, write_json};

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn claim_artifacts_summarize_chain_outputs() {
        let base = temp_dir("claim-artifacts");
        fs::create_dir_all(&base).unwrap();
        write_stage(&base, "stage_a", "aaa", b"stage-a");
        write_stage(&base, "stage_b", "bbb", b"stage-b");
        fs::write(
            base.join(EXECUTION_TIMES_JSON),
            r#"{"version":2,"stages":[{"name":"stage_b","exec_duration_ns":20},{"name":"stage_a","exec_duration_ns":10}],"total_exec_duration_ns":30}"#,
        )
        .unwrap();

        let manifest_path = base.join("Raster.toml");
        fs::write(&manifest_path, "[chain]\nname = \"test\"\n").unwrap();
        let (trace_path, hashes_path, bundle_path) =
            write_claim_artifacts(&base, &manifest_path).unwrap();
        assert_eq!(trace_path.file_name().unwrap(), CHECKPOINT_TRACE_JSON);
        assert_eq!(hashes_path.file_name().unwrap(), CHECKPOINT_HASHES_TXT);
        assert_eq!(bundle_path.file_name().unwrap(), CLAIM_BUNDLE_JSON);

        let trace: CheckpointTrace =
            serde_json::from_slice(&fs::read(&trace_path).unwrap()).unwrap();
        let bundle: ClaimBundle = serde_json::from_slice(&fs::read(&bundle_path).unwrap()).unwrap();

        assert_eq!(trace.checkpoints[0].stage, "stage_b");
        assert_eq!(trace.checkpoints[0].output_commitment, "bbb");
        assert_eq!(trace.checkpoints[1].stage, "stage_a");
        assert_eq!(checkpoint_hashes(&trace).unwrap().len(), 2);
        assert_eq!(fs::read_to_string(&hashes_path).unwrap().lines().count(), 2);
        assert!(trace_path.is_file());
        assert!(hashes_path.is_file());
        let input_commitment = format!("{:x}", Sha256::digest(b"input-manifest"));
        assert_eq!(bundle.input.stage, "stage_b");
        assert_eq!(bundle.input.commitment, input_commitment);
        assert_eq!(
            bundle.output,
            ClaimEndpoint {
                stage: String::from("stage_a"),
                commitment: String::from("aaa")
            }
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
            checkpoints: vec![Checkpoint {
                stage: String::from("output_finalize"),
                input_commitment: String::from("input"),
                output_commitment: String::from("abc"),
                output_sha256: String::from("deadbeef"),
            }],
        };
        let bundle = ClaimBundle {
            version: 1,
            input: ClaimEndpoint {
                stage: String::from("prompt_prepare"),
                commitment: String::from("input"),
            },
            output: ClaimEndpoint {
                stage: String::from("output_finalize"),
                commitment: String::from("abc"),
            },
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

    #[test]
    fn direct_infer_manifest_round_trips_as_json() {
        let manifest = DirectInferManifest {
            version: 1,
            bundle: DirectInferBundle {
                model_detwgt_path: PathBuf::from("../model.detwgt"),
                model_detwgt_sha256: String::from("model-sha"),
                config_path: PathBuf::from("../config.json"),
                config_sha256: String::from("config-sha"),
                tokenizer_path: PathBuf::from("../tokenizer.json"),
                tokenizer_sha256: String::from("tokenizer-sha"),
            },
            import: DirectInferImportSettings {
                prompt: String::from("hello"),
                raw_prompt: true,
                tokens: 2,
            },
            prompt: DirectInferPrompt {
                rendered_prompt: String::from("hello"),
                initial_pieces: vec![String::from("hello"), String::from("</w>")],
                eos_token_ids: vec![1, 2],
            },
            shape: DirectInferShape {
                hidden_size: 4,
                num_hidden_layers: 1,
                num_attention_heads: 2,
                num_key_value_heads: 1,
                head_dim: 2,
                global_head_dim: 2,
                vocab_size: 8,
                hidden_size_per_layer_input: 2,
                sliding_window: 16,
                layer_types: vec![String::from("sliding_attention")],
                num_kv_shared_layers: 0,
                norm_eps: 0,
                rope_base_sliding: 10_000_i64 << 32,
                rope_base_full: 1_000_000_i64 << 32,
                full_partial_rotary_factor_q16: 1 << 16,
                embedding_scale: 1 << 16,
                ple_embedding_scale: 1 << 16,
                ple_projection_scalar: 1 << 16,
                ple_input_scale: 1 << 16,
                final_logit_softcap: 0,
            },
            provenance: Some(DirectInferProvenance {
                raster_manifest_path: PathBuf::from("../Raster.toml"),
                raster_manifest_sha256: String::from("raster-sha"),
            }),
        };

        assert_eq!(
            serde_json::from_slice::<DirectInferManifest>(&serde_json::to_vec(&manifest).unwrap())
                .unwrap(),
            manifest
        );
    }

    #[test]
    fn challenge_artifacts_summarize_raster_replay() {
        let base = temp_dir("challenge-artifacts");
        let replay_run_dir = base.join("replay");
        fs::create_dir_all(&replay_run_dir).unwrap();
        write_stage(
            &replay_run_dir,
            "stage_a",
            "raster-commitment",
            b"raster-output",
        );
        let stage_dir = replay_run_dir.join("stage_a");
        fs::write(stage_dir.join("input.json"), b"{}").unwrap();
        fs::write(stage_dir.join("input_manifest.json"), b"{}").unwrap();
        fs::write(stage_dir.join("commit.bin"), b"commit").unwrap();

        let divergence = Divergence {
            version: 1,
            checkpoint_index: 0,
            stage: String::from("stage_a"),
            reason: DivergenceReason::OutputCommitment,
            claimed_trace_path: PathBuf::from("claimed.json"),
            recomputed_trace_path: PathBuf::from("recomputed.json"),
            claimed: Some(Checkpoint {
                stage: String::from("stage_a"),
                input_commitment: String::from("input"),
                output_commitment: String::from("claimed"),
                output_sha256: String::from("111"),
            }),
            recomputed: Some(Checkpoint {
                stage: String::from("stage_a"),
                input_commitment: String::from("input"),
                output_commitment: String::from("recomputed"),
                output_sha256: String::from("222"),
            }),
        };
        let replay_package = ReplayPackage {
            version: 1,
            stage: String::from("stage_a"),
            replay_run_dir: replay_run_dir.clone(),
            stage_dir: stage_dir.clone(),
            input_path: stage_dir.join("input.json"),
            input_manifest_path: stage_dir.join("input_manifest.json"),
            output_path: stage_dir.join("output.bin"),
            output_index_path: stage_dir.join("output.rindex"),
            output_manifest_path: stage_dir.join("output_manifest.json"),
            commit_path: stage_dir.join("commit.bin"),
        };

        let (divergence_path, replay_package_path, challenge_trace_path, bundle_path) =
            write_challenge_artifacts(
                &base.join("challenge"),
                Path::new("claimed.json"),
                Path::new("recomputed.json"),
                &divergence,
                &replay_package,
            )
            .unwrap();

        assert_eq!(divergence_path.file_name().unwrap(), DIVERGENCE_JSON);
        assert_eq!(
            replay_package_path.file_name().unwrap(),
            REPLAY_PACKAGE_JSON
        );
        assert_eq!(
            challenge_trace_path.file_name().unwrap(),
            CHALLENGE_TRACE_JSON
        );
        assert_eq!(bundle_path.file_name().unwrap(), CHALLENGE_BUNDLE_JSON);

        let bundle: ChallengeBundle =
            serde_json::from_slice(&fs::read(&bundle_path).unwrap()).unwrap();
        assert_eq!(bundle.stage, "stage_a");
        assert_eq!(bundle.raster_commit_path, replay_package.commit_path);

        let challenge_trace: ChallengeTrace =
            serde_json::from_slice(&fs::read(&challenge_trace_path).unwrap()).unwrap();
        assert_eq!(
            challenge_trace.raster_output_commitment,
            "raster-commitment"
        );

        fs::remove_dir_all(base).unwrap();
    }

    fn write_stage(base: &Path, stage: &str, commitment: &str, output: &[u8]) {
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
