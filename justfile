set positional-arguments

default:
    just --list

import model:
    cargo run --release -p raster-inference-cli -- model import --model "{{model}}"

infer run="inference.toml":
    cargo run --release -p raster-inference-cli -- infer --run "{{run}}"

claim run="inference.toml":
    cargo run --release -p raster-inference-cli -- claim build --run "{{run}}"

challenge claim:
    cargo run --release -p raster-inference-cli -- challenge build --claim "{{claim}}"

test-workspace:
    cargo test --workspace

test-direct:
    cargo test -p direct-infer

test-staged:
    cargo test -p staged-infer

test-cli:
    cargo test -p raster-inference-cli

test-artifacts:
    cargo test -p inference-artifacts

stage-check stage:
    cargo check --manifest-path "raster-stages/{{stage}}/Cargo.toml" --no-default-features
