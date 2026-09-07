# Claim Build Workflow

`claim build` reruns inference through the checkpointed staged path and writes the phase-two claim artifacts.

```bash
cargo run --release -p raster-inference-cli -- claim build --run inference.toml
# or
just claim inference.toml
```

Flow:

1. The CLI loads the run spec and prompt-free model manifest.
2. Prompt pieces and `prepared_run.json` are written under `target/raster-inference/runs/<run>/`.
3. A run-specific `Raster.toml` is generated in the same run directory with the requested prompt artifact and token count.
4. `staged-infer` reads the run manifest and expands the chain stages.
5. Each stage synthesizes `input.json` and `input_manifest.json`, runs the host implementation, and writes Raster-compatible output artifacts.
6. `inference-artifacts` scans the completed chain directory and writes the claim files.

Per-stage checkpoint files live under `target/staged-infer/chains-no-auth/<run>/<stage>/`:

- `input.json`
- `input_manifest.json`
- `output.bin`
- `output.rindex`
- `output_manifest.json`

Claim files live beside the generated checkpoint tree:

- `checkpoint_trace.json`: ordered checkpoint records.
- `checkpoint_hashes.txt`: one SHA-256 per checkpoint record.
- `claim_bundle.json`: whole-claim input/output commitments plus the checkpoint trace and prepared-run metadata references.

The claim path must preserve per-stage artifacts even if the host execution becomes faster internally. Those files are the surface the challenge path verifies.
