# Artifact Contracts

The stable contract types and filenames live in `crates/inference-artifacts`.

## Direct Infer

`direct-infer-artifacts/manifest.json` records the host-runtime inputs for `infer`:

- Model, config, and tokenizer paths plus SHA-256 hashes.
- Import settings such as prompt, raw-prompt mode, and token count.
- Rendered prompt pieces and EOS token ids.
- Model shape and fixed-point scaling parameters.
- Optional provenance pointing back to the Raster manifest used during import.

## Checkpoints

A checkpoint trace is an ordered list of per-stage records:

- Stage name.
- SHA-256 of `input_manifest.json`.
- Output structural commitment from `output_manifest.json`.
- SHA-256 of `output.bin`.

Every checkpointed stage directory must include:

- `input.json`
- `input_manifest.json`
- `output.bin`
- `output.rindex`
- `output_manifest.json`

## Claims

`claim build` writes:

- `checkpoint_trace.json`
- `checkpoint_hashes.txt`
- `claim_bundle.json`

The claim bundle is intentionally compact: it records the first checkpoint input commitment and the final checkpoint output commitment.

## Challenges

`challenge build` writes these files for a divergent stage:

- `divergence.json`
- `replay_package.json`
- `challenge_trace.json`
- `challenge_bundle.json`

The challenge bundle indexes the source trace, recomputed trace, divergence record, replay package, Raster commit path, and challenge trace.
