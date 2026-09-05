set positional-arguments

default:
    just --list

import model prompt="hello raster" tokens="1":
    cargo run --release -p raster-inference-cli -- model import --model "{{model}}" --prompt "{{prompt}}" --tokens "{{tokens}}"

infer:
    cargo run --release -p raster-inference-cli -- infer

claim:
    cargo run --release -p raster-inference-cli -- claim build

challenge trace:
    cargo run --release -p raster-inference-cli -- challenge build --trace "{{trace}}"

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
