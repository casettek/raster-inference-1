# Model Import

`model import` is the shared setup step for every workflow. It reads a Gemma bundle and writes the committed artifacts the repo uses afterward.

```bash
cargo run --release -p raster-inference-cli -- model import \
  --model ../raster-inference/assets/tiny-gemma-dev \
  --prompt "hello raster"
```

Inputs:

- `model.detwgt`: deterministic Q16.16 model weights.
- `config.json`: model shape, attention/window settings, softcap, and normalization parameters.
- `tokenizer.json`: vocabulary, merges, added tokens, and EOS metadata.

Outputs:

- Root `Raster.toml`, the canonical chain manifest for checkpointed and challenge workflows.
- Stage externals under `raster-stages/*`, written as `.rastered` and `.rindex` pairs with manifest commitments.
- `direct-infer-artifacts/manifest.json`, the host-runtime manifest consumed by `infer`.

Useful partial modes:

- `--only-direct`: refresh only `direct-infer-artifacts/manifest.json`.
- `--only-tokenizer`: refresh prompt/decoder externals.
- `--only-layers`, `--only-embedding`, `--only-ple`: refresh targeted weight externals.

`model import` is setup, not a peer to `infer`, `claim build`, or `challenge build`: the workflows all assume imported artifacts already exist.
