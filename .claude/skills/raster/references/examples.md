# Worked examples — a single program and a 3-stage chain

Real code from `raster-examples/raster-pipeline`: one computation split into
three chained programs (`normalize → aggregate → report`). Use these as the
template for new programs — every skill rule shows up here in its natural
place.

```text
  measurements + threshold                                    Report
        │                                                       ▲
        ▼            Filtered            Stats                  │
  ┌────────────┐  ───────────▶  ┌────────────┐  ─────────▶ ┌────────────┐
  │  phase 1   │                │  phase 2   │             │  phase 3   │
  │ normalize  │   output.bin   │ aggregate  │  output.bin │  report    │
  └────────────┘                └────────────┘             └────────────┘
```

## 1. A complete single program (phase 1 — `normalize`)

Filters a committed list of readings by a committed threshold. Demonstrates:
entry arguments, `select!`, an output-building recur tile, storage-backed
`main` return.

### `src/lib.rs` — the `no_std` tile library

```rust
//! `no_std` so the same tiles compile into RISC0 replay guests.
#![no_std]

extern crate alloc;

use raster::prelude::*;    // brings List, Block, Materializable, select!, tile, …
use serde::{Deserialize, Serialize};

/// Raw, committed input to the whole pipeline: a labelled column of readings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct Measurements {
    pub label: alloc::string::String,
    pub samples: List<u64>,
}

/// Phase 1 output / phase 2 input: the readings that survived the threshold.
///
/// The field layout here MUST match `Filtered` in `phase2-aggregate`, because
/// the chain links the two stages by the structural commitment of this value.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct Filtered {
    pub label: alloc::string::String,
    pub kept: List<u64>,
}

/// Recur tile: fold each sample into a growing `Filtered`, keeping only the
/// values `>= threshold`. The label is set once, on the first iteration.
#[tile(kind = recur)]
pub fn keep_above(
    input: RecurInput<u64>,
    output: RecurOutput<Filtered>,
    label: alloc::string::String,
    threshold: u64,
) -> RecurOutput<Filtered> {
    let mut output = output;
    if input.is_first() {
        output.label().set(label);
    }
    let value = input.into_value();
    if value >= threshold {
        output.kept().push(value);
    }
    output
}
```

Note what the tile does and doesn't see: plain `u64`s and `String`s in, a
draft handle threaded through — no proofs, no storage, no references. The
`if` on `value` is fine — control flow *inside a tile* is ordinary
computation; it's sequences that must stay linear.

### `src/main.rs` — the `std` binary: one sequence, zero computation

```rust
use raster::prelude::*;

use normalize::*;

/// Phase 1 entrypoint.
///
/// Binds two committed externals declared as `main` parameters. Its return
/// value is the program's authorized output (`output.bin`), which the chain
/// feeds into phase 2 as its `filtered` parameter.
#[sequence]
fn main(readings: Measurements, threshold: u64) -> Filtered {
    let label = select!(String, readings.clone().label);
    let samples = select!(List<u64>, readings.samples);
    let threshold = select!(u64, threshold);

    let filtered = call_recur!(
        tile = keep_above,
        input = samples,
        output = new!(Filtered),
        args = (label, threshold)
    );

    raster::println!("phase1 normalize → {:?}", filtered);
    filtered
}
```

Every skill rule, visible: selections pick exactly the sub-values the loop
needs; the recur input is a storage-backed `select!` result; the return value
is the `call_recur!` binding (storage-backed → valid `ProgramEnd`); the only
"statements" are selects, one call, one println, one return.

### Input fixtures — `bin/gen_input.rs` (behind a `gen-input` feature)

```rust
let measurements_commitment = raster::write_raster_files(
    &measurements,
    &out_dir.join("measurements.rastered"),
    &out_dir.join("measurements.rindex"),
)?;
// ...writes input.json + input_manifest.json with the printed commitments
```

Produces per entry argument: a raster-encoded value file + `.rindex`, an
`input.json` entry `{ "path", "index_path", "load_preference": "read" }`, and
an `input_manifest.json` entry
`{ "type": "sha256", "encoding": "raster", "commitment": "…" }`.

`Cargo.toml` posture that makes the split work:

```toml
[features]
default = ["std"]
std = ["raster/std"]
gen-input = ["dep:serde_json"]

[dependencies]
raster = { path = "…", default-features = false }
serde  = { version = "1.0", default-features = false, features = ["derive", "alloc"] }

[[bin]]
name = "gen_input"
path = "bin/gen_input.rs"
required-features = ["gen-input"]
```

## 2. A consumer stage (phase 2 — `aggregate`)

Shows the other recur mode (state-only fold) and an assembly tile. Its
`filtered` parameter is bound by the chain to phase 1's authorized output —
same authoring surface as an external input.

```rust
// lib.rs (no_std): Filtered duplicated field-for-field from phase 1,
// plus this stage's own types:

/// Loop-carried accumulator for the fold.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct Acc {
    pub count: u64,
    pub sum: u64,
    pub max: u64,
}

/// State-only recur tile: reduce the kept samples to `(count, sum, max)`.
#[tile(kind = recur)]
pub fn fold_stats(input: RecurInput<u64>, state: RecurState<Acc>) -> RecurState<Acc> {
    let mut state = state;
    let value = input.into_value();
    state.count += 1;
    state.sum += value;
    if value > state.max {
        state.max = value;
    }
    state
}

/// Combine the carried label with the folded accumulator into `Stats`.
#[tile]
pub fn assemble_stats(label: String, count: u64, sum: u64, max: u64) -> Stats {
    Stats { label, count, sum, max }
}
```

```rust
// main.rs:
#[sequence]
fn main(filtered: Filtered) -> Stats {
    let label = select!(String, filtered.clone().label);
    let kept = select!(List<u64>, filtered.kept);

    let acc = call_recur!(
        tile = fold_stats,
        input = kept,
        state = Acc { count: 0, sum: 0, max: 0 },
        args = ()
    );

    let count = select!(u64, acc.clone().count);
    let sum = select!(u64, acc.clone().sum);
    let max = select!(u64, acc.max);

    let stats = call!(assemble_stats, label, count, sum, max);
    raster::println!("phase2 aggregate → {:?}", stats);
    stats
}
```

The pattern to copy: fold with `state = ...`, then `select!` the individual
fields out of the final state and hand them to a small assembly tile whose
output becomes the stage result. (Phase 3 repeats the same shape with a pure
formatting tile — integer-scaled mean, `mean_scaled = (sum * 100) / count`,
because floats are banned from deterministic tiles.)

## 3. The chain manifest — root `Raster.toml`

A chain project's root manifest has a `[chain]` table and NO `[program]`
table (the Cargo virtual-workspace analogue). Members are ordinary programs
in subdirectories — own `Cargo.toml` + `Raster.lock`, no per-member
`Raster.toml`.

```toml
[chain]
name = "raster-pipeline"
version = "0.1.0"

# Stages run top-to-bottom. Each `main` parameter is bound either to an
# `external` committed input or `from` an earlier stage's authorized output.
[[chain.stage]]
name = "normalize"
project = "phase1-normalize"
inputs.readings  = { external = { path = "phase1-normalize/measurements.rastered", index_path = "phase1-normalize/measurements.rindex", commitment = "3c15…ab6b" } }
inputs.threshold = { external = { path = "phase1-normalize/threshold.rastered",    index_path = "phase1-normalize/threshold.rindex",    commitment = "3cb3…d952" } }

[[chain.stage]]
name = "aggregate"
project = "phase2-aggregate"
inputs.filtered = { from = "normalize" }

[[chain.stage]]
name = "report"
project = "phase3-report"
inputs.stats = { from = "aggregate" }
```

Chain-specific rules this encodes:

- **Binding is by name**: `inputs.filtered` matches phase 2's
  `fn main(filtered: Filtered)` parameter.
- **Links are by structural commitment**, not Rust type name: `Filtered` is
  defined identically in phases 1 and 2 (`Stats` in 2 and 3). Changing a
  field in one crate but not the other breaks the link at `chain audit` — a
  comment on both definitions ("MUST match …") is the house convention.
- **External commitments** come from the producing `gen_input` — paste the
  printed values into the manifest.
- The chain is linear: one authorized output feeds the next stage's named
  input; a `main() -> ()` stage can't feed anything downstream.

Run/verify from the chain root (manifest is discovered, no path argument):

```bash
cargo raster chain run                  # run stages, thread outputs, commit
cargo raster chain audit                # links + identities, no proving
cargo raster chain audit --execution    # + re-run stages against commits
cargo raster chain fraud-prove          # succinct receipt naming a faulty stage
cargo raster chain fraud-verify
```

## 4. Negative example — the "fake Raster program"

The most common failure mode when converting existing code: keep the native
function's shape and sprinkle `#[tile]` on it. The result compiles, runs
natively, and is not a Raster program — it is one giant unprovable step with
a cosmetic sequence in front of it.

```rust
// ❌ DO NOT write this.
#[derive(Serialize, Deserialize, Selectable)]
pub struct LargeInput {
    pub records: List<Record>,        // collections are List<T>, never Vec<T>
    pub rules: List<Rule>,
    pub lookup_table: List<LookupEntry>,
}

#[sequence]
fn main(input: LargeInput) -> Result<FinalOutput> {
    let records = select!(List<Record>, input.clone().records);
    let rules = select!(List<Rule>, input.clone().rules);
    let lookup_table = select!(List<LookupEntry>, input.lookup_table);

    // Each of these is a whole `List` handed to a tile — a COMPILE ERROR now,
    // because `List<T>` is not `Materializable`. Shown to name the anti-pattern.
    let output = call!(process_everything, records, rules, lookup_table)?;
    Ok(output)
}

#[tile]
pub fn process_everything(
    records: List<Record>,            // ← rejected: not Materializable
    rules: List<Rule>,                // ← rejected
    lookup_table: List<LookupEntry>,  // ← rejected
) -> Result<FinalOutput> {
    // loops over all records
    // scans all rules
    // scans the whole lookup table
    // builds the entire final output
}
```

The selections themselves are legal, but the whole design does not compile any
more — passing a `List` to a tile is rejected at the boundary. Even setting the
type enforcement aside, six distinct problems make the *shape* wrong:

### 4.1 The sequence is only cosmetic

It does not orchestrate verifiable steps — it selects entire collections and
hands them to one tile. One step in the CFS, everything interesting invisible
inside it. Orchestration means the *loop structure* of the computation is
visible in the schema:

```rust
// ❌ cosmetic routing (and now a compile error at the call! boundary)
let all_records = select!(List<Record>, input.records);
let output = call!(process_everything, all_records);

// ✅ real orchestration — the iteration IS the schema
let records = select!(List<Record>, input.records);
let output = call_recur!(
    tile = process_record_chunk,
    input = records,
    chunk = 64,
    output = new!(FinalOutput),
    args = ()
);
```

### 4.2 Whole collections materialized into one tile

Every tile call materializes its inputs (SKILL.md §2). This is why `List<T>` is
not `Materializable`: a whole collection as a parameter would be decoded,
Merkleized, and replayed in ONE zkVM unit — potentially beyond what RISC0 replay
can afford (memory is hard-capped; proving cost scales with every byte touched).
The type system now blocks it outright; the `Block<T>` window is the affordable
substitute.

```rust
// ❌ whole collection as tile input — does not compile (List not Materializable)
#[tile]
pub fn process_rules(rules: List<Rule>) -> Summary

// ✅ one element (or one chunk) per replay unit
#[tile(kind = recur)]
pub fn process_one_rule(
    input: RecurInput<Rule>,
    state: RecurState<RuleState>,
) -> RecurState<RuleState>
```

### 4.3 Wrapper structs do not fix it

A struct containing a `List` is still a large tile input — and the type system
knows it: a struct with a `List<T>` field is `Selectable` but **not**
`Materializable`, so wrapping the collection does not smuggle it across the
boundary. The tile below fails to compile for the same reason `process_rules`
did:

```rust
// ❌ same problem, one indirection deeper — still rejected
#[derive(Serialize, Deserialize, Selectable)]
pub struct RuleSet {
    pub rules: List<Rule>,   // makes RuleSet non-Materializable
}

#[tile]
pub fn process_rules(rule_set: RuleSet) -> Summary   // ← rejected: RuleSet not Materializable
```

This generalizes: no newtype, no field nesting, no "context object" makes a
big collection small — and none of them is `Materializable`. Select the scalar
fields a tile actually needs; iterate the collection with recur. (Same trap in
recur `args` — see `recur.md` §2.)

### 4.4 A giant output rebuilt and committed at once

Outputs cost what inputs cost: the returned value is serialized, Merkleized,
and committed at the tile boundary. The `#[tile]` macro asserts the return type
is `Materializable` too, so a tile that returns a collection (or a struct with a
`List` field) is rejected — the input `List<Record>` here is *also* rejected:

```rust
// ❌ builds and commits the entire output in one step — rejected at BOTH ends
#[tile]
pub fn build_output(records: List<Record>) -> FinalOutput {  // arg not Materializable;
    let mut output = FinalOutput::default();                 // return also checked
    for record in records {
        output.items.push(transform(record));
    }
    output
}

// ✅ append one increment per step through the draft protocol
#[tile(kind = recur)]
pub fn append_output_item(
    input: RecurInput<Record>,
    output: RecurOutput<FinalOutput>,
) -> RecurOutput<FinalOutput> {
    let mut output = output;
    output.items().push(transform(input.into_value()));
    output
}
```

### 4.5 Runtime/internal handles are not an escape hatch

Do not smuggle runtime plumbing into program types or tile signatures to
dodge materialization:

```rust
// ❌ leaks the runtime into the authored program model
pub struct WorkState {
    pub records_ref: StorageRef,
}

#[tile]
pub fn process(state: WorkState) -> WorkState
```

A `StorageRef` (or any internal handle) inside a program type is not
authorized dataflow — the protocol cannot bind what flowed through it, and
the tile body gains an unaccounted door into storage. The ONLY protocol
types allowed in tile signatures are the sanctioned ones: `Draft<T>` and the
`Recur*` family. Everything else a tile receives is a plain materialized
value.

### 4.6 Native success is misleading

The bad program above passes `cargo check` and `cargo raster run`. That
proves nothing — the failure is in the *replay shape*, which native
execution never exercises. A Raster program is not done until:

- large inputs are chunked (recur / slice selections);
- sequences route steps — the computation's structure is visible in the CFS;
- every tile is bounded in input, compute, AND output;
- repeated work uses recur tiles or recur sequences, never in-tile loops
  over collections;
- large outputs grow through drafts, never one rebuilt return value;
- no wrapper struct hides a large collection;
- no runtime/internal reference leaks through program types.

**The short version:** if the Raster rewrite has the same shape as the
original native function, just with `#[tile]` added, it is probably wrong.
The native version is one function; the Raster version is a *schema of small
steps*. More steps, each smaller — that is what being verifiable costs, and
recur/chunk is what keeps that cost linear instead of painful.

## 5. A byte-region program — sweep and random access

Byte data is `Bytes<P>`; the only value that crosses a tile boundary is one
`BytesPage`. This is the same `List`/`Block` split one granularity down, so the
driver is the ordinary list recur.

### `src/input.rs` — declaring the region

```rust
#[derive(Serialize, Deserialize, Selectable)]
pub struct ModelFile {
    pub row_stride: u32,                  // 16 KiB — 4096 i32 per row

    /// Where each layer's weights start, in bytes, within `weights`.
    pub layers: List<LayerEntry>,

    /// 256 KiB pages = 16 rows/page. A multiple of `row_stride`, so no row
    /// straddles a page boundary — the first sizing rule, ahead of cycles.
    #[page_size = 262_144]
    pub weights: Bytes<262_144>,
}

#[derive(Serialize, Deserialize, Selectable)]
pub struct LayerEntry {
    pub layer_id: u32,
    pub byte_offset: u64,                 // global offset into `weights`
    pub byte_len: u64,
}
```

`ModelFile` has `List` fields, so it is `Selectable` but not `Materializable` —
passing the whole model into a tile is the existing compile error. `Bytes<P>` is
never `Materializable` for the same reason.

### `bin/gen_input.rs` — writing the artifact

```rust
let raw: Vec<u8> = load_weights("model.safetensors")?;
let model = ModelFile {
    row_stride: 4096 * 4,
    layers:     List::from(layer_table_for(&raw)),   // offsets page-aligned
    weights:    Bytes::<{ ModelFile::WEIGHTS_PAGE_SIZE }>::paged(raw)?,
};
let commitment = write_raster_files(
    &model, Path::new("model.rastered"), Path::new("model.rindex"))?;
```

`paged` is the only constructor — a region without a page size has no addressing
scheme. Use the generated `WEIGHTS_PAGE_SIZE` const so the fixture cannot drift
from the declaration.

### Sweeping the whole region

```rust
#[sequence]
fn main(model: ModelFile) -> Accum {
    // `.pages` is written explicitly: `call_recur!` iterates a `List<T>`, and
    // the hop must be visible to the flow resolver or a sweep across a
    // `call_seq!` boundary fails the audit.
    let pages = select!(List<BytesPage>, model.weights.pages);
    call_recur!(tile = accumulate_page, input = pages, state = Accum::default())
}

#[tile(kind = recur, description = "Accumulate one page", estimated_cycles = 1_200_000)]
pub fn accumulate_page(input: RecurInput<BytesPage>, state: Accum) -> Accum {
    let page = input.into_value();
    let bytes = page.as_slice();
    assert!(bytes.len() % 4 == 0, "page is not i32-aligned");   // last page is short

    let mut state = state;
    for word in bytes.chunks_exact(4) {
        state.sum += i64::from(i32::from_le_bytes(word.try_into().unwrap()));
    }
    state
}
```

Add `chunk = 4` if per-replay overhead dominates; the step then takes
`RecurInput<Block<BytesPage>>` and nothing else changes.

### Random access at a computed byte offset

A byte offset is not a page index, and converting between them is arithmetic —
so it happens in a tile, not in the selector.

```rust
#[sequence]
fn main(model: ModelFile, request: LayerRequest) -> LayerOut {
    let layer_id = select!(u32, request.layer_id);
    let layers   = select!(List<LayerEntry>, model.clone().layers);
    let entry    = select!(LayerEntry, layers[layer_id]);

    let start = select!(u64, entry.clone().byte_offset);
    let len   = select!(u64, entry.byte_len);

    // The region's own committed page size — a selectable field, not a literal,
    // so the index is bound to the artifact it addresses.
    let page_size = select!(u64, model.clone().weights.page_size);
    let page_idx  = call!(page_of, clone!(start), page_size);   // replayed, proven

    let page = select!(BytesPage, model.weights[page_idx]);     // ordinary BoundIndex

    call!(apply_layer, page, start, len)
}

#[tile(description = "Page containing a byte offset", estimated_cycles = 200)]
pub fn page_of(byte_offset: u64, page_size: u64) -> u64 {
    byte_offset / page_size
}
```

`page_idx` is an ordinary authorized citation — no new proof step, no new audit
arm. Note `start` is passed to `apply_layer` *alongside* the page: a page's
Merkle leaf must be stable regardless of which byte was requested, so the
request cannot live inside the page value. The consuming tile computes
`local = start - page.offset()`.

### The three ways to get this wrong

```rust
// 1. Byte data as a list of scalars. One index node and one Merkle leaf per
//    byte: a 1 GiB region needs ~120 GB of .rindex. Not slow — impossible.
pub weights: List<u8>,

// 2. Hex in a String. 2x on disk, 2x in the replay input, plus a nibble
//    decoder burning 2.6-5.2M cycles in every tile.
pub weights: String,

// 3. A recur driven by byte ranges. The page count wobbles by alignment
//    (⌈L/P⌉ or ⌈L/P⌉+1), so chunking fails on the first short iteration.
let window = select!(Block<BytesPage>, model.weights[0..1_048_576]);
call_recur!(tile = accumulate_page, input = window, /* ... */);
```

Ranges are for random access; recur is for sweeps.
