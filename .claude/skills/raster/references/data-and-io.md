# Data, selection, drafts, and program I/O

How data enters a Raster program, moves between steps in authenticated form,
and leaves as an output another program can consume.

Everything here implements one chain (SKILL.md §2): committed entry arguments
are the only data source; every intermediate value a sequence holds is an
authorized **reference** into committed storage; tiles are the only place a
value is **materialized** (decoded + committed — the expensive operation);
dynamically built data grows through drafts; `main`'s return closes the chain
as the authorized output. Every construct below is a link in that chain — if
a piece of data can't name its link, it doesn't belong in the program.

## 1. `select!` — authenticated sub-value access

`select!(Type, source.path)` decodes a committed value and produces a
`SelectionCommitment` binding the selected bytes to the source's committed
root. It is the ONLY way to reach into a value inside a sequence.

```rust
// field access:
let name = select!(String, personal_data.clone().name);

// chained fields + indexing:
let line = select!(String, personal_data.addresses[0].lines[0]);

// two-step selection through a bound intermediate — a whole collection is List<T>:
let addresses = select!(List<Address>, personal_data.addresses);
let second    = select!(Address, addresses[1]);

// contiguous range — ONE commitment, and it yields a Block<String>: the only
// collection type a tile may take. A tile taking Block<String> records a single
// external binding for the whole window:
let two_lines = select!(Block<String>, personal_data.addresses[0].lines[0..2]);

// whole value re-selection: legal for a struct with NO List field. A struct that
// contains a List<T> is Selectable but not Materializable, so it cannot be passed
// whole into a tile — select the scalar fields instead.
let whole = select!(SmallConfig, config_bin.clone());

// selecting from a tile / recur result:
let stats = call_recur!(tile = track_max_len, input = lines, state = MaxLenState { max_len: 0 }, args = ());
let max   = select!(u64, stats.max_len);
```

Rules:

- Every struct type traversed by a selector path MUST derive `Selectable`
  (alongside `Serialize`/`Deserialize`); keep those types in the `no_std`
  library so guests can decode them. Collection fields are `List<T>`, never
  `Vec<T>` (the derive rejects `Vec`).
- Paths = field access, `[index]`, `[a..b]` ranges. No method calls except
  `.clone()` on the source binding.
- **Target-type vocabulary:** a whole-collection select names `List<T>`; a
  range `[a..b]` select names `Block<T>`. The macro rejects a `Block` target
  without a range and a range target that isn't `Block`. Byte regions are
  `Bytes<P>`; a page select names `BytesPage` (with an index) and a covering
  byte range names `Block<BytesPage>`. Literal indexes on `Bytes` are in bytes
  and must be page-aligned; binding indexes are already page indexes.
- **Paged fixtures:** write with `Bytes::<N>::paged(bytes)` (or
  `Bytes::<{ Type::FIELD_PAGE_SIZE }>::paged(bytes)`). Prefer `mmap` for large
  regions. Changing `N` / `#[page_size]` is both an artifact regeneration and
  an identity change (`InterfaceDecl.schema_hash`).
- **Derive `Selectable`/`Serialize`/`Deserialize`; never hand-write them.** The
  derive is what makes the host's selector path and the guest's decode agree on
  the same bytes; a manual impl (or a manual `Default`/`Ord`/`PartialEq` a tile
  relies on) can silently diverge between native execution and replay, which
  surfaces as an audit divergence with a clean native run. Any other computation
  in an `impl` block on a Rastered type is governed by SKILL.md §3, "Where code
  may live".
- `.clone()` the source when it is used again later in the sequence.
- Select the smallest thing the consuming tile needs. Feeding blocks: one
  range selection → one `Block<T>` tile input beats N single-element selections.

## 2. Drafts — multi-tile object construction

`Draft<T>` lets several tiles build one object with authenticated lineage.

```rust
let draft = new!(CollectiveGreeting);                       // empty draft
let draft = call!(set_draft_greeting_title, "Title".to_string(), draft);
let draft = call!(push_draft_greeting_line, "Line 1".to_string(), draft);
let draft = call!(push_draft_greeting_line, "Line 2".to_string(), draft);
let greeting = finalize(draft);                             // materialized T
let title = select!(String, greeting.clone().title);        // select! from it
```

Tile side: take `Draft<T>` (position doesn't matter), mutate through set-once
accessors, return it:

```rust
#[tile(kind = iter)]
pub fn set_draft_greeting_title(
    title: String,
    draft: Draft<CollectiveGreeting>,
) -> Draft<CollectiveGreeting> {
    let mut draft = draft;
    draft.title().set(title);      // scalar field: set-once
    draft
}
// list fields: draft.lines().push(value) — append-only
```

Hard rules (compile-time UI-tested):

- Draft handles are **linear**: cloning one is a compile error; using a
  binding again after passing it to a `call!` is a compile error. Rebind at
  every step: `let draft = call!(step, ..., draft);`
- Scalar fields are set-once — a second `.set()` fails at runtime.
- `finalize(draft)` fails if required fields were never set (this is also why
  empty recur inputs can fail output finalization).

## 3. Entry arguments — `input.json` + `input_manifest.json`

`#[sequence] fn main(personal_data: PersonalData, seed: u64)` declares two
entry arguments. Each is resolved lazily, by parameter name, from two files:

**`input.json` (private — paths only, never inline values):**

```json
{
  "personal_data": { "path": "personal_data.bin", "load_preference": "read" },
  "seed":          { "path": "seed.bin",          "load_preference": "mmap" }
}
```

- Only the `{ "path", "load_preference": "read" | "mmap" }` object form is
  valid. Inline strings/numbers/objects are rejected.
- Referenced files contain **postcard-serialized** values of the parameter's
  declared Rust type (or raster-encoded data with an `.rindex`, which is
  self-describing and also supports cross-process selection in the
  commit/audit pipeline; postcard entries are limited to in-process use).
- **Any input a `call_recur!` / `call_recur_seq!` sweeps must be raster-encoded**
  — `index_path` + `encoding = "raster"` in the manifest. This is a hard
  requirement, not a performance preference: postcard is sequential and carries
  no index, so `rows[i]` cannot be located without decoding everything before
  it, and opening a recur over one fails at the call site. `--commit`/`--audit`
  already imposes the same constraint on *every* argument it must build a
  selection witness for, so raster encoding is the safe default for anything
  larger than a scalar.

**`input_manifest.json` (public — commitments):** one entry per argument name.
For postcard-encoded arguments used with `select!`, the commitment is the
selection-tree structural root (`raster::postcard_structural_commitment`); for
raw files it is the SHA-256 of the file bytes.

**Regenerating fixtures** — never hand-edit the `.bin`/manifest files. Follow
the hello-tiles pattern: a `bin/gen_input.rs` behind a `gen-input` feature
that writes the value files AND the manifest with matching commitments:

```bash
cargo run --features gen-input --bin gen_input -- .
cargo raster run --input input.json --input-manifest input_manifest.json
```

Adding/renaming/retyping a `main` parameter ⇒ regenerate fixtures, or the run
fails to resolve/authorize the argument.

**An entry argument must carry information.** A commitment attests *which
bytes*, never *that the bytes mean anything* — so a field the generator
computes from another field of the same input (`rounds: (0..chars.len())`,
`count: items.len()`, a duplicated key) is committed to nothing and
constrains nothing. Two failure modes follow, and both survive the whole
check ladder:

- **Derived scalars that the program then trusts.** Only pre-derive a value
  when it is a static property of an already-committed table and the
  derivation is part of the fixture's meaning (e.g. `unk_id` /
  `max_special_len` in `raster-tokenizer`'s `GemmaTokenizer` — covered by the
  same commitment as the vocabulary they come from). If the program would
  behave differently for a *wrong* derived value, it must compute it in a
  tile.
- **Fabricated collections used to drive control flow.** A counter list
  committed so a `call_recur!` has something of the right length to iterate
  hands the loop's trip count to whoever writes the fixture. This is the
  worst version of the fake recur precisely because it is committed; see
  `references/recur.md` §2, "the committed counter list".

The test: could a verifier disagree with this field? If its value is forced
by the rest of the input, it belongs in a tile or nowhere.

## 4. Program output — the `ProgramEnd` contract

`main`'s return value is the program's **authorized output**:

- `fn main(...)` returning `()` → empty output, always fine.
- A value-returning `main` MUST return a **storage-backed** value: the binding
  of a `call!`, `call_seq!`, `call_recur!`, or `select!`. The value must
  provably live in committed storage (produced by a verified tile).
- Returning an inline literal or locally computed value is a runtime error at
  the ProgramEnd boundary — there is no storage lineage to commit to.
- The protocol attests **success only**: a failed program has no authorized
  output.

The runtime exports the output as a raster-encoded artifact
(`output.bin` + index + `output_manifest.json`) — the SAME format as external
inputs, which is what makes chaining possible.

## 5. Chains — output N = input N+1

A chain is declared in a `Raster.toml` `[chain]` table (or `chain.json`); see
`docs/proposals/program-chain.md` for the manifest format. The contract that
matters when authoring member programs:

- Stage N's authorized output artifact becomes stage N+1's committed external
  input. So stage N+1's `main` entry-argument **type must match** stage N's
  return type — same Rust type, same (shared) crate definition ideally.
- A stage with `main() -> ()` cannot feed a following stage.

Commands:

```bash
cargo raster chain run                      # run all stages, thread outputs, commit
cargo raster chain audit                    # verify links + identities (public)
cargo raster chain audit --execution        # + re-run stages against commits
cargo raster chain fraud-prove              # produce a chain fraud receipt
cargo raster chain fraud-verify             # verify a receipt
```

## 6. Program identity — what it is, why it exists, how it's used

### What

The **program identity** (`program_commitment`) is one hash that names a
Raster program as a stable, verifiable object. It commits to the program's
**static** definition — everything that is true of the program on *every*
run:

- **the interface** — declared input names/types/encodings and the output
  type/encoding (`Raster.toml`);
- **the control flow** — the CFS: topology, tile/sequence names, arities,
  dataflow bindings, entry-argument names;
- **the code** — the tile image-id registry (`TileId → image id`). This is
  the part that actually pins each tile's semantics: the CFS alone is
  code-blind (it carries names and arities, not code), so two programs with
  identical shape but different tile bodies would otherwise hash the same.

It deliberately excludes the **dynamic** side of a run — input values, the
trace fingerprint, output commitments — and the protocol's own guest image
ids (so a protocol upgrade doesn't change *what program you have*). One
attested execution is named by the tuple:

```text
( program_commitment,          ← WHICH program        (static)
  input_manifest_commitment,   ← which authorized inputs   (per-run)
  fingerprint,                 ← which actual data path    (per-run)
  output_manifest_commitment ) ← which authorized output   (per-run)
```

Boundary note: a literal at a call site (`call!(f, 42)`) is per-run data,
bound by the fingerprint; a constant inside a tile *body* is program
identity, baked into that tile's image id. The dividing line is the tile
boundary — exactly where code identity is measured.

### Why

Without it, no claim can name its subject. "This program mapped input I to
output O" is meaningless if "this program" is unspecified — a prover could
substitute **any** tile binary whose output happens to match the recorded
witness, or present a different program with the same shape. The identity
closes that hole: the transition guest receives the canonical definition
bytes, hashes them itself to derive `program_commitment`, and drives its
checks from those same bytes — so "the program this receipt is about" and
"the program this hash names" cannot diverge. Chains need it for the same
reason: every stage checkpoint carries the stage's `program_commitment`, and
`chain audit` recomputes each stage's identity and flags a mismatch as
**identity fraud**.

### The three artifacts

| File | Written by | In VCS | Role |
| --- | --- | --- | --- |
| `Raster.toml` | you (optional) | yes | declared interface: `[program]` name/version, `[inputs.<name>]` type+encoding, `[output]` type+encoding |
| `Raster.lock` | `cargo raster build` | **yes — commit it** | the recorded claim: `program_commitment`, per-tile `{image_id, source_hash}`, toolchain |
| `target/raster/program.bin` | `cargo raster build` | no (gitignored) | canonical definition bytes — the hash preimage AND the guest's verification input; a pure regenerable cache |

- `Raster.lock` is not the identity itself — it is the reproducible *claim*
  of it (Cargo.lock semantics), tying the commitment to a source revision.
  Never hand-edit it.
- `program.bin` is safe to delete; every consumer regenerates it — the
  prover by recompiling tiles and drift-checking image ids against
  `Raster.lock`, the light audit path by reassembling the identical bytes
  from checked-in files only (source CFS + `Raster.lock` image ids, **no
  toolchain needed**).
- `Raster.toml` is optional (absent, the manifest is synthesized from the
  crate name + `main`'s signature) but when present it is **enforced**: at
  build, declared inputs must match `main`'s entry arguments and `[output]`
  must match whether `main` returns a value — drift is a build error. Chain
  wiring reads each stage's `Raster.toml` to validate output→input
  compatibility before anything runs.

### Authoring rules

- Any change to a tile body, a sequence, or `main`'s signature changes the
  identity. That is the point — re-locking is a deliberate act:

```bash
cargo raster program            # show identity (commitment, interface, tiles)
cargo raster program --verify   # recompute from source, check against Raster.lock
```

- Run `--verify` after every change set; an *unexpected* mismatch means
  program behavior drifted when you thought it hadn't (or someone else's
  change rode along). An expected mismatch → rebuild, review, commit the new
  `Raster.lock` together with the source change.
- In chain projects, member stages carry their own `Raster.lock` (no
  per-member `Raster.toml` needed); `chain audit`'s identity ✓ per stage
  depends on those locks being committed and current.

## 7. Byte regions — `Bytes<P>` and `BytesPage`

### The tile-side contract

A tile never receives a `Bytes<P>` region; it receives one `BytesPage`, or a
`Block<BytesPage>` under `chunk = N`. What a page gives you:

| on a `BytesPage` | |
| --- | --- |
| `page.offset()` | global byte offset of the page start — committed, audited `== index × page_size` |
| `page.index()` | page number — committed |
| `page.as_slice()` | the bytes; `page_size` long **except on the last page** |
| `page.len()` | actual length — assert your element alignment, never assume it |
| the offset you *asked for* | **not on the page** — it arrives as its own authorized tile argument |
| mutation | none: no `DerefMut`, no public constructor |
| `Debug` | prints `{ index, offset, len }`, never the payload |

Two rules follow, and both are load-bearing:

```rust
#[tile(kind = recur, description = "Accumulate one page", estimated_cycles = 1_200_000)]
pub fn accumulate_page(input: RecurInput<BytesPage>, state: Accum) -> Accum {
    let page = input.into_value();
    let bytes = page.as_slice();
    // 1. `len` is input, not a constant. The final page is short.
    assert!(bytes.len() % 4 == 0, "page is not i32-aligned");

    let mut state = state;
    for word in bytes.chunks_exact(4) {
        state.sum += i64::from(i32::from_le_bytes(word.try_into().unwrap()));
    }
    state
}

#[tile(description = "Apply one layer", estimated_cycles = 900_000)]
pub fn apply_layer(page: BytesPage, start: u64, len: u64) -> LayerOut {
    // 2. Position inside the page is `start - page.offset()`. `start` is passed
    //    alongside because a page's Merkle leaf must be stable regardless of
    //    which byte was asked for — putting the request in the page value would
    //    change its root.
    let local = (start - page.offset()) as usize;
    assert!(local + len as usize <= page.len(), "layer spans more than one page");
    /* ... */
    LayerOut::default()
}
```

**Pagination is visible in the tile no matter how the sequence is written.**
Hiding it in the selector changes the sequence's spelling, not the tile's
arithmetic.

### Sizing `page_size`

Stride first, then budget — see SKILL.md §2 for the formula and the comparison
against `List<u8>` / hex. Two more rules:

- **Keep record offsets page-aligned at import.** It costs nothing in the
  fixture generator and turns a runtime abort ("spans more than one page") into
  an import-time invariant.
- **Compute addresses from committed headers, never on the host.** A layer table
  *inside* the artifact, cited via `BoundIndex`, keeps the address in the
  authorization chain. A host-computed offset surfaces as `external`/`Inline` in
  `cfs.json` — it is not authorized, and the audit will say so.

### Measuring, so sizing is not guesswork

Build with `--features profiling` and read `TileProfileRecord`:

| field | what it tells you |
| --- | --- |
| `input_bytes` | **the replay unit size** — the number `page_size` controls, and what the input-commitment hash is charged on |
| `output_bytes` | the other half of the budget; a large per-iteration output is the usual cause of a slow sweep |
| `user_duration_ns` vs `raster_overhead_ns` | whether you are paying for work or for framing — small pages mean more replays, so fixed overhead dominates |

Sweep a page-size candidate, read `input_bytes`, confirm it equals your declared
`page_size` on every iteration but the last, and check `user_duration_ns` is
still the dominant term. If overhead dominates, the page is too small.

### Host memory

- Give a large region its own external input with `load_preference: "mmap"`.
  Mapping does not allocate; a page selection is one slice plus one copy.
- Selecting **`.pages` then one page at a time** keeps host memory at
  `O(page + witness)`. Selecting the region whole (`select!(Bytes<P>, …)`) walks
  every page node into RAM — that spelling is for holding a reference, not for
  reading.
