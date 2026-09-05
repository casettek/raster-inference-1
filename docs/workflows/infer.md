# Infer Workflow

`infer` is the fast deterministic path. It uses `direct-infer`, reads `direct-infer-artifacts/manifest.json`, and does not write staged checkpoint artifacts.

```bash
cargo run --release -p raster-inference-cli -- infer
# or
just infer
```

Flow:

1. `raster-inference-cli` dispatches `infer` to `direct_infer::DirectInferenceExecutor`.
2. `direct-infer` loads the direct manifest, tokenizer metadata, and mmap-backed DETWGT weights.
3. The executor runs prompt preparation, embedding, prefill layers, decode, and output finalization in host memory.
4. The command prints an `InferenceResult` and timing summary.

Output:

- Generated token count.
- Generated token ids.
- SHA-256 of the generated token id list.
- Stop reason.
- Generated text.
- Direct-infer timing summary.

This path is deterministic inference only. It is intentionally free of the per-stage checkpoint tree used by claims and challenges.
