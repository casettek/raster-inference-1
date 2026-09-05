# Claim Build Workflow

`claim build` reruns inference through the checkpointed staged path and writes the phase-two claim artifacts.

```bash
cargo run --release -p raster-inference-cli -- claim build
# or
just claim
```

Flow:

1. The CLI builds a `CheckpointedInferenceConfig` from the repo root.
2. `staged-infer` reads the root `Raster.toml` and expands the chain stages.
3. Each stage synthesizes `input.json` and `input_manifest.json`, runs the host implementation, and writes Raster-compatible output artifacts.
4. `inference-artifacts` scans the completed chain directory and writes the claim files.

Per-stage checkpoint files live under `target/staged-infer/chains-no-auth/<run>/<stage>/`:

- `input.json`
- `input_manifest.json`
- `output.bin`
- `output.rindex`
- `output_manifest.json`

Claim files live beside the generated checkpoint tree:

- `checkpoint_trace.json`: ordered checkpoint records.
- `checkpoint_hashes.txt`: one SHA-256 per checkpoint record.
- `claim_bundle.json`: whole-claim input and output commitments.

The claim path must preserve per-stage artifacts even if the host execution becomes faster internally. Those files are the surface the challenge path verifies.
