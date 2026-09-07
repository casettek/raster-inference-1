# Model Import

`model import` is the shared setup step for every workflow. It reads a Gemma bundle and writes the committed artifacts the repo uses afterward.

```bash
cargo run --release -p raster-inference-cli -- model import \
  --model ../raster-inference/assets/tiny-gemma-dev
```

Inputs:

- `model.detwgt`: deterministic Q16.16 model weights.
- `config.json`: model shape, attention/window settings, softcap, and normalization parameters.
- `tokenizer.json`: vocabulary, merges, added tokens, and EOS metadata.

Outputs:

- Root `Raster.toml`, a prompt-free chain manifest template.
- Stage externals under `raster-stages/*`, written as `.rastered` and `.rindex` pairs with manifest commitments.
- `model-artifacts/manifest.json`, the prompt-free model manifest consumed by run specs.

Useful partial modes:

- `--only-direct`: refresh only `model-artifacts/manifest.json`.
- `--only-tokenizer`: refresh tokenizer/decoder externals.
- `--only-layers`, `--only-embedding`, `--only-ple`: refresh targeted weight externals.

`model import` no longer accepts `--prompt`, `--raw-prompt`, or `--tokens`. Runtime prompt text and token count live in `inference.toml`.
