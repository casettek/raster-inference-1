# Challenge Build Workflow

`challenge build` consumes a compact claimed checkpoint hash list plus frozen run context, recomputes the checkpointed run, stops at the first observed divergent hash, and replays that one divergent stage through Raster.

```bash
cargo run --release -p raster-inference-cli -- challenge build \
  --claim-context target/staged-infer/runs/.../claim_bundle.json \
  --checkpoints target/staged-infer/runs/.../checkpoints.txt
# or
just challenge \
  target/staged-infer/runs/.../claim_bundle.json \
  target/staged-infer/runs/.../checkpoints.txt
```

Flow:

1. Load the claimed `checkpoints.txt` and use `claim_bundle.json` only as today's local carrier for `prepared_run.json`.
2. Run the checkpointed staged executor again with the frozen run manifest from `prepared_run.json`.
3. After each recomputed checkpoint, hash the verifier's local checkpoint record and compare it to the claimed hash at the same index.
4. If no divergence exists, report the claimed and recomputed traces.
5. If a divergence exists, seed a replay directory with all prior recomputed stages.
6. Run `cargo raster chain run --stage <stage>` for the divergent stage.
7. Write the challenge artifacts.

Challenge hash verification happens during recomputation. Sequential stages stop immediately at the first mismatch. Parallel aux waves still execute with the existing parallelism; if a mismatch is found in the middle of a completed aux batch, later batch outputs may exist on disk but are ignored for the challenge trace and replay seed.

Divergence reasons are checked in this order:

- Checkpoint hash mismatch.
- Missing claimed or recomputed checkpoint hash.

Challenge outputs live under the recomputed run's `challenge/<stage>/` directory:

- `divergence.json`
- `replay_package.json`
- `challenge_trace.json`
- `challenge_bundle.json`

For manual fraud tests, corrupt a copy of the claimed hash list and challenge that copy:

```bash
just corrupt target/staged-infer/runs/.../checkpoints.txt
just challenge \
  target/staged-infer/runs/.../claim_bundle.json \
  target/staged-infer/runs/.../checkpoints.corrupt.txt
```

`--claim` and `--trace` remain lower-level debug inputs that compare full checkpoint records. `--trace` recomputes with the current root manifest because a trace alone has no frozen run metadata. `challenge locate` and `fault prove` are reserved commands; `challenge build` is the implemented verifier workflow today.
