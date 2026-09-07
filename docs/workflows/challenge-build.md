# Challenge Build Workflow

`challenge build` consumes a claim bundle, recomputes the checkpointed run from the claim's frozen prepared-run metadata, locates the first divergence, and replays that one divergent stage through Raster.

```bash
cargo run --release -p raster-inference-cli -- challenge build \
  --claim target/staged-infer/chains-no-auth/.../claim_bundle.json
# or
just challenge target/staged-infer/chains-no-auth/.../claim_bundle.json
```

Flow:

1. Load the claimed `claim_bundle.json`, its `checkpoint_trace.json`, and its `prepared_run.json`.
2. Run the checkpointed staged executor again with the frozen run manifest from `prepared_run.json`.
3. Compare traces in stage order and stop at the first mismatch.
4. If no divergence exists, report the claimed and recomputed traces.
5. If a divergence exists, seed a replay directory with all prior recomputed stages.
6. Run `cargo raster chain run --stage <stage>` for the divergent stage.
7. Write the challenge artifacts.

Divergence reasons are checked in this order:

- Stage name mismatch.
- Input commitment mismatch.
- Output commitment mismatch.
- Output payload SHA-256 mismatch.
- Missing claimed or recomputed checkpoint.

Challenge outputs live under the recomputed run's `challenge/<stage>/` directory:

- `divergence.json`
- `replay_package.json`
- `challenge_trace.json`
- `challenge_bundle.json`

`--trace` remains a lower-level debug input, but it recomputes with the current root manifest because a trace alone has no frozen run metadata. `challenge locate` and `fault prove` are reserved commands; `challenge build` is the implemented verifier workflow today.
