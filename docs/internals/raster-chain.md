# Raster Chain Internals

The root `Raster.toml` is a chain manifest: it has a `[chain]` table and no `[program]` table. The member directories under `raster-stages/` are ordinary Raster programs with their own `Cargo.toml`, `Raster.lock`, no-std tile library, and sequence entry point.

```text
model import
  -> tokenizer and prompt pieces
  -> embedding table
  -> PLE layer externals
  -> transformer layer externals
  -> output head and decoder
  -> Raster.toml chain manifest
```

Stage outputs are linked by structural commitment, not Rust type name. Boundary structs that cross stage edges therefore need to stay field-for-field compatible between producer and consumer crates.

The central activation boundary is:

```text
ActivationSequence {
  rows,
  errors,
  kv,
  start_position,
}
```

High-level stage order:

1. `prompt-prepare`: tokenize prompt pieces into token ids.
2. `input-embedding`: gather embedding rows for token ids.
3. `prefill-prepare-aux`: compute per-layer PLE inputs.
4. `prefill-range`: run one transformer layer and publish activation/KV state.
5. `prefill-finalize`: score the output head.
6. Decode repeat: select token, embed token, run per-layer aux/range stages, finalize logits.
7. `output-finalize`: decode generated ids into text and publish `InferenceResult` fields.

The layer loop lives in the chain manifest rather than in one Raster program. A generated model expands one program over many stage instances, preserving bounded replay units and per-stage auditability.
