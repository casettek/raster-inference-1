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
- Reusable Raster externals under `runtime/model-artifacts/<model-id>/raster/*`, written as `.rastered` and `.rindex` pairs with manifest commitments.
- `runtime/model-artifacts/<model-id>/manifest.json`, the prompt-free model manifest consumed by run specs.

By default, `<model-id>` is the sanitized model bundle directory name. Pass `--model-id <id>` to choose it explicitly, or `--artifact-root <dir>` to place model-scoped artifact directories somewhere other than `runtime/model-artifacts`.

`initial_pieces` is deliberately not imported with the model. It is generated
from each prompt during run preparation and written under that run's `prompt/`
directory.

Useful partial modes:

- `--only-direct`: refresh only `runtime/model-artifacts/<model-id>/manifest.json`.
- `--only-tokenizer`: refresh tokenizer/decoder externals.
- `--only-layers`, `--only-embedding`, `--only-ple`: refresh targeted weight externals.

`model import` no longer accepts `--prompt`, `--raw-prompt`, or `--tokens`. Runtime prompt text and token count live in `inference.toml`.
