# raster-chain-inference

`raster-chain-inference` runs Gemma inference in three developer-facing modes over the same imported model inputs:

- `infer`: fast deterministic inference from a run spec and `runtime/model-artifacts/<model-id>/manifest.json`.
- `claim build`: a checkpointed deterministic rerun that emits a compact claim.
- `challenge build`: verifier rerun from a claim trace, divergence detection, and Raster replay of the first divergent stage.

`model import` is the shared setup step. It turns a model bundle into committed Raster externals and a prompt-free model manifest. `infer`, `claim build`, and `challenge build` prepare prompt-specific run artifacts from `inference.toml`.

## Quick Start

```bash
# Import committed externals and the direct-infer manifest.
cargo run --release -p raster-inference-cli -- model import \
  --model ../raster-inference/assets/tiny-gemma-dev

# Fast deterministic inference, no checkpoint tree.
cargo run --release -p raster-inference-cli -- infer --run inference.toml

# Proposer claim with every staged checkpoint.
cargo run --release -p raster-inference-cli -- claim build --run inference.toml

# Verifier challenge from a prior claim bundle.
cargo run --release -p raster-inference-cli -- challenge build \
  --claim-context target/staged-infer/runs/.../claim_bundle.json \
  --checkpoints target/staged-infer/runs/.../checkpoints.txt
```

The same commands are available through `just` recipes:

```bash
just import ../raster-inference/assets/tiny-gemma-dev
just infer inference.toml
just claim inference.toml
just challenge \
  target/staged-infer/runs/.../claim_bundle.json \
  target/staged-infer/runs/.../checkpoints.txt
```

## Workflows

- [Model import](docs/workflows/model-import.md): bundle inputs, generated externals, root manifest, and direct-infer manifest.
- [Infer](docs/workflows/infer.md): fast deterministic inference through `direct-infer`.
- [Claim build](docs/workflows/claim-build.md): checkpointed staged inference and claim artifacts.
- [Challenge build](docs/workflows/challenge-build.md): trace comparison, divergence packaging, and Raster replay.

## Internals

- [Raster chain](docs/internals/raster-chain.md): stage expansion, boundary types, and chain-program layout.
- [Artifacts](docs/internals/artifacts.md): stable JSON contracts and per-stage checkpoint files.
- [Development](docs/internals/development.md): workspace commands, stage checks, and lower-level debug surfaces.
- [Issues](docs/issues/README.md): known program gaps and upstream Raster handoffs.
- [Proposals](docs/proposals/): design notes and historical decisions.

## Layout

```text
Cargo.toml                  # workspace manifest
justfile                    # workflow and test shortcuts
Raster.toml                 # prompt-free generated chain manifest template
inference.toml              # default run spec
crates/
  model/
    import/                 # model bundle -> committed inputs/manifests
    run-prep/               # prompt/run preparation helpers
    detwgt/                 # DETWGT eager and mmap readers
  inference/
    cli/                    # developer-facing workflow CLI
    direct/                 # fast deterministic runtime
    staged/                 # checkpointed chain executor and parity tooling
    artifacts/              # shared JSON contracts and artifact I/O
    kernels/                # shared deterministic inference kernels
  det-num/                  # deterministic numeric primitives
raster-stages/              # verifiable Raster program crates
runtime/                    # imported model artifacts, prompt fixtures, generated manifest examples
manifests/                  # alternate generated manifests and examples
docs/                       # workflow, internals, issues, proposals
```

`raster-stages/*` are intentionally left outside `crates/`: each directory is a Raster program boundary with its own `Cargo.toml`, `Raster.lock`, no-std tile library, and sequence entry point. Reusable imported model externals live under `runtime/model-artifacts/<model-id>/raster/`, prompt fixtures live under `runtime/prompt-fixtures/`, and generated chain manifests point each stage at the files it needs there. The root `Raster.toml` is a chain manifest with a `[chain]` table and no `[program]` table.
