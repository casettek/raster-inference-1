---
name: raster
description: >
  Write or modify verifiable Raster programs: tiles, sequences, recur-tiles and
  recur-sequences over blocks of data, program inputs/outputs, chains. MUST be
  used for any change to a Raster project's src/, input fixtures, or Raster.toml.
  Enforces the rules that keep every computation re-executable as a RISC0 guest.
---

# Writing verifiable Raster programs

**Tiles compute on materialized authorized data; sequences route authorized
references.** Every step of a Raster program must be re-executable inside the
RISC0 zkVM and bound to committed data. Almost every violation of the rules
below **still compiles and runs natively** — it fails later, at
commit/audit/fraud-proof time, or worse, silently produces a program that
cannot be verified at all. Never treat `cargo build` as "done"; done is
defined by the check ladder in §9.

If a line in a sequence does anything other than **name a step, select data,
or bind a result**, it belongs in a tile. If a feature seems to require plain
Rust orchestration (an `if`, a loop, a computed argument), do NOT emulate it —
restructure it into tiles, or tell the user the model doesn't support it yet.

**The shape test:** if a Raster rewrite has the same shape as the original
native function — one entry, whole collections in, whole result out — just
with `#[tile]` added, it is probably wrong. A real Raster program has MORE
steps than the native version, each bounded and small. The full anatomy of
this failure ("the fake Raster program") is dissected in
`references/examples.md` §4.

## 1. Program layout

```text
my-program/
  Cargo.toml
  Raster.toml            # program identity (+ optional [chain] table)
  src/
    lib.rs               # #![no_std] — ALL tiles live here (or its modules)
    input.rs             # shared data types: Serialize/Deserialize + Selectable
    main.rs              # std binary — ALL sequences + #[sequence] fn main
  input.json             # private: entry-arg name -> {path, load_preference}
  input_manifest.json    # public: entry-arg name -> commitment
```

- `src/lib.rs` MUST start with `#![no_std]` and `extern crate alloc;`. Tiles
  are compiled into RISC0 guests with `default-features = false` — any `std`
  leakage in the tile library breaks guest builds.
- Tile and sequence IDs are the bare function names. They MUST be unique
  across the whole program (the registry is keyed by name only).
- Types that cross tile boundaries or get `select!`ed MUST derive
  `Serialize, Deserialize` and — for `select!` paths into them — `Selectable`.
  Keep them in the `no_std` library so both host and guest see them.

**Chain-project layout** (multi-program pipeline) differs: the root `Raster.toml`
holds a `[chain]` table and NO `[program]` table — the analogue of a Cargo
virtual-workspace manifest. Member stages are ordinary programs in
subdirectories with their own `Cargo.toml` + `Raster.lock` but **no
per-member `Raster.toml`** — each stage's identity comes from its
`Raster.lock` / cached `target/raster/program.bin`:

```text
my-pipeline/
  Raster.toml            # [chain] table only — stages, input bindings
  stage-a/               # a normal single program (Cargo.toml, Raster.lock, src/)
  stage-b/
```

## 2. The data model — one authorization chain, references by default

Internalize this before writing any code; every other rule derives from it.

There is exactly ONE way data becomes trustworthy in a Raster program, and it
forms an unbroken chain:

```text
committed entry arguments (input_manifest.json)
      │  select! / call! — every hop recorded and witnessed
      ▼
storage-backed intermediate values (every tile output, every selection)
      │  drafts / recur outputs for anything built dynamically
      ▼
authorized program output (ProgramEnd → output.bin + manifest)
```

- **The only source of data is `main`'s entry arguments.** They are committed
  in `input_manifest.json` and resolved lazily on first use. Nothing
  verifiable can enter the program through a side door — a literal or a
  computed value in a sequence has no commitment and no lineage.
- **Sequences hold authorized references, never data.** Every binding in a
  sequence body — the result of `call!`, `call_seq!`, `call_recur!`,
  `select!`, and `main`'s parameters themselves — is an authorized reference
  (`AuthRef`) to a value living in committed storage, NOT the value itself.
  Routing a reference is cheap; that is why sequences route. A sequence never
  opens a reference: field access, comparison, printing a field — all
  forbidden (§4).
- **Tiles receive materialized values.** At the `call!` boundary Raster
  materializes each argument's reference into the plain typed value and
  records the binding witnesses; the tile body computes on plain `T` and
  never sees proofs, references, or storage. Its return value is stored and
  committed on exit, and the call site binds a fresh authorized reference to
  it. So: authorization lives at the boundaries; the tile stays plain Rust.
- **Dynamically built data goes through drafts.** Any object or collection
  that doesn't come from input must grow inside the authorized draft protocol
  — `new!`/`Draft<T>` with set-once writes (§6), or `RecurOutput` in loops
  (§7). Never assemble a collection ad hoc in sequence code (unverifiable)
  and never return a giant rebuilt collection from a tile (unaffordable, see
  below).
- **Materialization is the cost center — avoid every unnecessary one.**
  Materializing means decoding AND committing: Raster-Merkleizing a large
  value dominates a tile call's cost. Concretely:
  - select the smallest sub-value a tile actually needs — the field, not the
    struct; never pass a whole object where one field suffices;
  - feed a block of elements as ONE contiguous slice selection
    (`lines[0..2]`), not the whole collection and not N single-element
    selections;
  - keep tile OUTPUTS as small as inputs — a tile that returns a big object
    pays Merkleization for all of it on every call; thread a `Draft` and
    append instead of returning rebuilt collections;
  - don't `select!` a whole value just to "have it" — select only at the
    point where a tile consumes it;
  - `clone!(binding)` clones a reference (cheap); materializing that binding
    into a tile is what costs.

### Collections are `List<T>`; the only tile-visible window is `Block<T>`

`Vec` is **not** a Rastered type. The bounded-materialization rule is enforced by
the type system, not just prose (`crates/raster-core/src/collections.rs`):

- **`List<T>`** — the unbounded collection. Use it for every collection field in
  a Rastered data type and for whole-collection references. `List` is
  `Selectable` (you can `select!` an element, a range, or the whole reference)
  and it is the source a `call_recur!` iterates. It is **not** `Materializable`:
  it can never be passed whole into a tile.
- **`Block<T>`** — a bounded window, and the ONLY collection type that may cross
  a tile boundary. It is produced solely by the framework from operations whose
  size bound is pinned in the CFS: a literal-range `select!` (`xs[a..b]`) or a
  `chunk = N` recur step. A tile that needs several elements at once takes a
  `Block<T>`.
- **`Materializable`** — the marker trait a value needs to be materialized into a
  tile (the dual of `Selectable`: `Selectable` reaches *into* stored data,
  `Materializable` brings it out *whole*). Scalars, `String`, `Block<T>`, and
  derived structs whose fields are all `Materializable` qualify. A struct with a
  `List<T>` field is `Selectable` but **not** `Materializable` — select its
  scalar fields, never pass the whole struct into a tile.

Consequences, all now **compile errors** (not runtime/audit failures):

- a tile argument or return of type `Vec<T>` or `List<T>` → rejected
  (`"cannot be materialized into a tile"`);
- a `Vec<T>` field in a `#[derive(Selectable)]` struct → rejected (`"not a
  Rastered type … must be List<T>"`);
- `select!(Block<T>, …)` without a range, or a range not named `Block` → rejected;
- an inline `call!(f, vec![…])` → rejected (no lineage, and not `Materializable`).

Write `List<T>` in data types and whole-collection selects; write `Block<T>` in
tile signatures fed by a range select or a `chunk = N` step. Full contract:
`references/data-and-io.md` §1 and `references/recur.md`.

### Raw bytes are `Bytes<P>`; the only tile-visible window is `BytesPage`

Byte data gets the same split, with the granularity on the Rust type (`select!`
is a proc macro and cannot see a field attribute):

```rust
#[page_size = 262_144]
pub weights: Bytes<262_144>,
```

`#[page_size = n]` must equal `Bytes<N>` — a cross-check, not a second source of
truth. The derive emits `WEIGHTS_PAGE_SIZE`.

- **`Bytes<P>`** — a paged byte region. `Selectable`, never `Materializable`.
- **`BytesPage`** — one page, the only byte value that may cross a tile boundary.
  It carries committed `index()` and `offset()`.
- Sweep with `select!(List<BytesPage>, region.pages)` then `call_recur!`. Never
  pass `Bytes` as a recur input (compile error naming `.pages`).
- To reach a byte offset, convert it to a page index **in a tile**
  (`call!(page_of, offset, page_size)`), then index the region with the result.
  A binding index is already a page index. Literal indexes/ranges on `Bytes` are
  in **bytes** and converted to page units at expansion; unaligned literals are
  a compile error. Pass the offset to the consuming tile —
  `local = offset - page.offset()`.
- Never model byte data as `List<u8>` or hex in a `String`.
- Changing `#[page_size]` / `Bytes<N>` changes `program_commitment` via
  `InterfaceDecl.schema_hash`. Re-import the artifact.

**A page is the replay unit, so `page_size` is the single knob that sets tile
cost.** Pick it in this order — the first constraint is not negotiable, the
second is a budget:

1. **A multiple of your record stride**, so no record straddles a page. A
   straddling record forces loop-carried stitching state in every tile and makes
   the last page a special case. Get this wrong and no amount of tuning helps.
2. **Then the largest page that stays under your per-replay cycle budget.**
   Roughly `cycles ≈ page_size × (per-byte work + ~1.1 for the input-commitment
   hash)` — SHA-256 runs ~1.06 cycles/byte with the risc0 accelerator, and it is
   charged on *every* replay. Bigger pages amortize the fixed per-replay
   overhead, so go as large as the budget allows.

Why the vocabulary is worth obeying, for a 1 GiB region at 256 KiB pages:

| | `List<u8>` | hex in a `String` | **`Bytes<P>`** |
| --- | --- | --- | --- |
| `.rindex` | ~120 GB — **infeasible** | ~460 KB | ~460 KB |
| tile input per page | — | 512 KiB | **256 KiB** |
| decode cycles per page | — | ~2.6–5.2 M | **0** |

Hex decode is the term that carries the difference; `List<u8>` does not merely
cost more, it cannot be built (one index node and one Merkle leaf per byte).
Cycle figures are order-of-magnitude — **measure before quoting them**.

Measure with `--features profiling`: `TileProfileRecord.input_bytes` is the
replay unit size, and `output_bytes` catches the other half of the budget (see
`references/data-and-io.md` §7).

## 3. Tiles — all computation, always written for the zkVM

A tile's unit of existence is a **RISC0 replay**: postcard input bytes in,
committed output bytes out, executed inside the guest with bounded cycles.
When creating ANY tile, ask first: *can this function be replayed in the
zkVM guest, bit-identically to the native run, at a cycle cost someone can
afford to prove?* If the answer needs a caveat, redesign before writing.

A tile is a non-generic **free function** annotated with `#[tile(...)]`:

```rust
use raster::prelude::*;
use raster::println;

#[tile(kind = iter, description = "Greets a user", estimated_cycles = 1000)]
pub fn greet(name: String) -> String {
    let greeting = format!("Hello, {}!", name);
    println!("greet: {}", greeting);   // raster::println! — CLI captures it
    greeting
}
```

Hard rules:

- **Attribute syntax is key/value only.** `#[tile(kind = iter)]` (default) or
  `#[tile(kind = recur)]`. NEVER write `#[tile(recur)]` — it is silently
  ignored and the tile stays `iter`. Optional keys: `description`,
  `estimated_cycles`, `max_memory`. Unknown keys are silently ignored, so
  typos vanish — double-check spelling.
- **Signature:** no `self`, no generics, no `where` clauses, parameters are
  simple identifiers (`x: T` — never destructuring patterns; those can panic
  the macro). Inputs and the return type must be postcard-serializable serde
  types. Multiple logical outputs = return ONE struct/tuple; the caller
  `select!`s into it (destructuring a call result is forbidden, §4).
- **Plain bodies:** the tile receives already-materialized, already-authorized
  values (§2). Never try to accept references, proofs, or storage handles in
  a tile signature (`Draft<T>`/`Recur*` protocol types are the sanctioned
  exceptions). Never reference macro-generated `__raster_*` symbols.
- **Fallible tiles** return the prelude `Result<T>` (Raster's terminal
  execution result; errors are strings): `Err(String::from("MissingName"))`.
  Callers propagate with `?` on the `call!` result.
- **Determinism (unenforced — you are the only guard).** Native execution and
  guest replay must agree **bit-for-bit**. Inside tiles:
  - no I/O, filesystem, network, clock, randomness, threads, env vars
    (`no_std` blocks most, but don't smuggle them via dependencies);
  - no `HashMap`/`HashSet` (iteration order) — use `BTreeMap`/`BTreeSet`/`Vec`;
  - no floating point unless bit-identical cross-target behavior is proven —
    prefer integers/fixed-point;
  - no pointer/address-derived values, no `#[cfg(target_...)]` logic in tile
    bodies;
  - logging ONLY via `raster::println!`.
- **Tiles must stay small — in cycles AND in I/O.** Each tile call is one
  zkVM replay unit, and both its input and output are materialized and
  committed (§2). You *cannot* take a whole collection as one argument or
  return one: `Vec<T>`/`List<T>` are not `Materializable`, so both positions
  are compile errors (§2). Walk data in `Block<T>` windows (`select!` ranges)
  and recur calls (§7), build results through drafts. Use `estimated_cycles`
  to document known-heavy tiles.

### Where code may live — the tile boundary is what pins it

Program identity commits to the **tile image-id registry** (§8): code reachable
from a tile body is compiled into a guest and pinned by that tile's image id.
Code that only the host ever calls is in `program_commitment` nowhere. So the
rule is not "no methods on data types" — a tile body is plain Rust and may call
helpers freely — it is that **anything the program's result depends on must be
reachable from a tile body**. An `impl` block is the easiest place to break
that without noticing.

| Code | Verdict |
| --- | --- |
| helper fn / method called from a **tile body** | fine — inside the image id, pinned, replayed in the guest |
| helper called from **both** a tile and the fixture generator | fine, and the sanctioned way to keep host and guest byte-identical (`vocab_bucket_of` in `raster-tokenizer`) — it must obey the tile determinism rules |
| method used **only** by host fixture code to *encode* a value | fine — representation, not logic; the committed bytes are the program's input |
| method used by host fixture code to **derive** a value the program then trusts | ❌ computation outside every image id (§2, and `references/data-and-io.md` §3) |
| method called from a **sequence** body | ❌ computation hidden from the CFS (§4) — the grammar forbids bare calls; a method call on a binding or inside a `select!` path is the sneak |
| hand-written `Serialize`/`Deserialize`/`Selectable`/`Default`/`Ord` on a Rastered type | ❌ derive only — these decide what gets committed and how selector paths resolve; a hand-written impl can make host and guest disagree about the same value |
| a native oracle (`encode_prompt_native`) used by tests | fine, and useful — but it is a *reference*, never a path the program takes |

The test for any `impl` block on a Rastered type: **would the program's output
change if this method were wrong?** If yes, it must be called from a tile. If
it only shapes bytes on their way into `input.json`, it is encoding and may
stay on the host.

## 4. Sequences — orchestration only

A sequence is a free function annotated with `#[sequence]` in the `std`
binary. It moves **authorized references** between steps (§2) — it never
opens them. Its body is written in a **restricted grammar** — treat
everything outside this list as forbidden:

| Allowed in a sequence body | Form |
| --- | --- |
| tile call | `call!(tile_name, args...)` — optionally with `?` / `.expect(...)` |
| sub-sequence call | `call_seq!(seq_name, args...)` |
| recur over a list | `call_recur!( ... )` / `call_recur_seq!( ... )` (§7) |
| data selection | `select!(Type, binding.path)` (§5) |
| draft creation / finish | `new!(Type)` / `finalize(draft)` (§6) |
| explicit storage ref | `storage!(Type, reference)` |
| binding a result | `let x = <one of the above>;` — simple identifier only |
| cloning a binding | `clone!(binding)` — clones the reference, cheap |
| output | `raster::println!(...)` / `println!(...)` for debug |
| return | last expression = a binding or call result |

**Forbidden in sequence bodies** (compiles, breaks verifiability):

- **Bare calls**: `greet(name)` — invisible to the CFS. Always `call!` /
  `call_seq!`.
- **Nested calls**: `call!(exclaim, call!(greet, name))` — decompose:

  ```rust
  let greeting = call!(greet, name);
  let result = call!(exclaim, greeting);
  ```

- **Destructuring bindings**: `let (a, b) = call!(split, x);` — bind one name,
  then `select!` the parts.
- **Computed arguments**: `call!(f, x + 1)`, `call!(f, format!("{x}"))`,
  `call!(f, vec![a, b])` — these become unauthenticated inline inputs with no
  lineage (§2). Move the computation into a tile. Plain literals (`42`,
  `"Chunked greeting".to_string()`) are acceptable as configuration-style
  arguments.
- **Opening a reference**: field access, indexing, comparison, or arithmetic
  on a binding outside `select!` — a sequence binding is a reference, not a
  value; there is nothing legitimate to read on it.
- **Control flow**: `if` / `match` / `for` / `while` / early `return` on
  runtime values. The CFS is linear. The only sanctioned iteration is
  `call_recur!`/`call_recur_seq!`; the only early exits are
  `RecurControl::Break` inside a recur step (§7) and `?` on fallible calls.
- **Any arithmetic / string building / collection manipulation.** That is
  computation. Tile.

Sequences may call sequences (`call_seq!`); the callee's own `#[sequence]`
wrapper emits its trace boundary events. Fallible sequences mirror fallible
tiles: return `Result<T>`, propagate with `?`.

## 5. Selecting data — `select!`

`select!(Type, source.path)` is the only sanctioned way to reach INTO a
referenced value (entry argument, tile output, finalized draft). It produces
an authenticated selection commitment; plain field access in a sequence is
forbidden (§4).

```rust
let name         = select!(String, personal_data.clone().name);
let addresses    = select!(List<Address>, personal_data.addresses);  // whole list = List<T>
let second       = select!(Address, addresses[1]);
let first_line   = select!(String, second.lines[0]);
// Contiguous range = ONE commitment, ONE tile input, and yields a Block<T>:
let two_lines    = select!(Block<String>, personal_data.addresses[0].lines[0..2]);
```

- Paths support field access, indexing, and contiguous ranges `[a..b]`.
- A **whole-collection** select names `List<T>`; a **range** `[a..b]` select
  names `Block<T>` (the macro rejects either target used the other way, §2).
- Every struct traversed by a path must derive `Selectable`.
- `clone!` the source binding when it is used again later. Inside a `select!`
  path a bare `.clone()` is still the spelling (`personal_data.clone().name`) —
  it is part of the selector expression, not a sequence step.
- **Select the smallest value a tile actually needs, exactly where it is
  consumed** (§2 cost rules): a field beats the struct, one slice selection
  beats N element selections, and an unused selection is pure waste.

### Indexing by an authorized value

An index may also be a **binding in scope**, which is how you look an element up
by a value the program computed instead of scanning for it:

```rust
let token_id = select!(u32, prompt.token_ids[0]);
let row      = select!(EmbeddingRow, table.rows[token_id]);   // O(log n) proof

// Inside a recur sequence, the item itself is the index:
let wanted = into_ref!(input);                 // AuthRef, nothing materialized
let row    = select!(Row, rows[wanted]);
```

- **`into_ref!(handle)`** unwraps a recur-sequence item to its `AuthRef`. A
  `RecurSequenceInput<T>` is a handle, not a reference: it does not implement
  `SelectSource`, so `select!` cannot reach into it, and passing it to a tile
  materializes the whole item. `into_ref!` is what lets you index by an item, or
  select one field out of a wide one, without materializing anything. It is a
  **macro, not a method** — the CFS attributes provenance by recognizing the
  grammar's macros, so a bare method call would leave the local unattributed
  (`InputSource::Inline`): an argument the schema pins to nothing.

- The index must be an `AuthRef<uN>` (`u8`/`u16`/`u32`/`u64`) — an authorized
  value. A literal, a computed expression (`i + 1`), a `.clone()`, or a signed
  integer is rejected: an index with no lineage is one a prover could choose.
  Pass the binding **bare**; it is borrowed, so the same index can locate
  several values.
- Ranges keep literal bounds. There is no dynamic `[a..b]` yet.
- **An out-of-range index aborts the run with no output** — it cannot be
  handled, because a list has no non-membership proof. Do not use a dynamic
  index where "not found" is an expected outcome; that still needs a scan.
- This replaces the scan-and-match idiom for *positional* lookup. Key→value
  lookup (a string to an id) is still a scan — there is no map type.

See `docs/proposals/dynamic-index-selection.md`.

Details and the input-fixture format: `references/data-and-io.md`.

## 6. Building outputs across tiles — drafts

Dynamically built data MUST grow through the authorized draft protocol (§2).
To build one object with several tiles, thread a `Draft<T>` through them:

```rust
// in the sequence:
let draft = new!(CollectiveGreeting);
let draft = call!(set_title, "Greeting".to_string(), draft);
let draft = call!(push_line, "Hello".to_string(), draft);
let greeting = finalize(draft);                    // materialized CollectiveGreeting
let title = select!(String, greeting.clone().title);

// tiles take and return the draft:
#[tile(kind = iter)]
pub fn push_line(line: String, draft: Draft<CollectiveGreeting>) -> Draft<CollectiveGreeting> {
    let mut draft = draft;
    draft.lines().push(line);       // set-once accessors: .field().set(v), .list().push(v)
    draft
}
```

Draft handles are **linear**: never clone one, never reuse one after passing
it to a call — rebind (`let draft = call!(...)`) every step. Fields are
set-once: a second `.set()` on the same field fails at runtime. This is also
the cheap path: each step appends its increment instead of re-materializing
and re-committing the whole object (§2).

## 7. Blocks of data — recur tiles and recur sequences

Never loop in a sequence. To process a list, pick from this decision tree:

| Need | Use |
| --- | --- |
| one value / sub-value | `select!` |
| a bounded `Block<T>` window as one tile input | `select!` with `[a..b]` |
| fold list → single summary value | `call_recur!` + `state = ...` |
| map list → one built object | `call_recur!` + `output = new!(T)` |
| fold AND build together | `call_recur!` + `state` + `output`, step returns the `(state, output)` tuple |
| early stop | state+output step returning `RecurControl` (`Continue`/`Break`) |
| step should see N elements at a time | add `chunk = N` (step takes `RecurInput<Block<T>>`) |
| several tiles per element | `#[sequence(kind = recur)]` + `call_recur_seq!` |
| sweep a byte region | `call_recur!` with `input = select!(List<BytesPage>, region.pages)` |
| several pages per replay unit | `chunk = N` (step takes `RecurInput<Block<BytesPage>>`) |
| the page at a computed byte offset | `call!(page_of, offset, page_size)` then `select!(BytesPage, region[page_idx])` |
| whole `Bytes` into one tile | **NEVER** (compile error — not `Materializable`) |
| recur driven by byte ranges | **NEVER** — page count wobbles and fails the chunk rules |
| whole `List<T>` into one tile | **NEVER** (a compile error — `List` is not `Materializable`) |

**Placement is not negotiable** — each recur slot has a fixed role:

- `input` — THE collection being iterated, and it must be the *real* work
  items (chunks, rows, records). This is the **only** place a collection
  ever goes. Driving a recur with a synthetic counter list (`List<u32>` of
  "rounds") while the real data hides in `state`/`args` is the same
  violation wearing a loop costume — an unused `input` parameter in the
  step is the tell (`references/recur.md` §2, "the fake recur").
  **Committing that counter list as part of an entry argument does not
  launder it.** A `rounds: List<u32>` field sitting beside the real data,
  filled with `0..data.len()`, passes every rung of §9 — it is genuinely
  storage-backed and genuinely committed — and it is still the same
  violation, now with the loop's trip count chosen by whoever writes the
  fixture: a short list yields a truncated result with a valid proof over
  it. A field derivable from another field of the same input is not data.
  Full dissection and the sanctioned alternatives: `references/recur.md`
  §2, "the committed counter list".
- `input` **must come from a raster-encoded source.** This is an error, not a
  tuning knob: a recur reaches its items one at a time through the source's
  raster index, and a postcard external has no index, so `rows[i]` cannot be
  located without decoding everything before it. Opening a recur over one fails
  with

  ```text
  call_recur! requires a raster-indexed List source;
  re-encode this input with encoding = "raster"
  ```

  Declare iterable inputs with `index_path` + `encoding = "raster"`;
  `examples/hello-tiles/bin/gen_input.rs` shows the pattern. Internally stored
  values (tile outputs, finalized drafts, `store_value` results) always carry a
  raster index, so only **external** inputs need the declaration. The source is
  never materialized: the loop bound comes from an authenticated 41-byte
  metadata selection and each item from its own indexed read.
- `state` — a tiny loop-carried value (counters, running max, a small
  accumulator struct). It is re-committed on **every** iteration, so it must
  stay scalar-small; anything that *grows* belongs in `output` (append-only
  draft — pays only the increment).
- `args` — small per-call constants: a label, a threshold, a limit. NEVER a
  collection. `args` are materialized tile arguments, so a `List<T>` (or `Vec`)
  there is now a **compile error** (`args` must be `Materializable`); a `Block<T>`
  would type-check but still smuggles the whole window into every iteration —
  the model violation the `args`-collection ban has always been about. Reduce
  the other collection to a scalar first (see `references/recur.md`).
- **One recur = one collection.** A step body must never loop over a second
  collection smuggled in via `args` or `state` (item-vs-other-collection
  scans, joins). Computation happens around a small amount of data, period.
  If per-item work feels too granular, coarsen with `chunk = N` — never with
  fatter params. Restructuring patterns for cross-collection problems:
  `references/recur.md`.

Minimal example (output-building recur):

```rust
// sequence side — input must be a storage-backed List:
let address_lines = select!(List<String>, personal_data.addresses[0].lines);
let greeting = call_recur!(
    tile = build_greeting_line,
    input = address_lines,
    output = new!(CollectiveGreeting),
    args = ("Recur-built".to_string(),)
);

// tile side — fixed shape: input first, then state and/or output, extras last:
#[tile(kind = recur)]
pub fn build_greeting_line(
    input: RecurInput<String>,
    output: RecurOutput<CollectiveGreeting>,
    title: String,
) -> RecurOutput<CollectiveGreeting> {
    let mut output = output;
    if input.is_first() {
        output.title().set(title);
    }
    output.lines().push(input.into_value());
    output
}
```

Each iteration is its own replay unit: the source list stays ONE authenticated
binding while every step materializes only its element (or chunk) — that is
the affordable way to touch large data (§2). The macro rigidly validates step
shapes; get them from `references/recur.md` (all three modes, chunking, early
termination, recur sequences, empty-input semantics) rather than improvising.

## 8. Program boundary — `main`, inputs, output, chains

- The program entry point is `#[sequence] fn main(...)`. It cannot be
  `kind = recur`.
- **Entry arguments**: each `main` parameter is bound by name to a committed
  external input — `input.json` (private paths) + `input_manifest.json`
  (public commitments). This is the ONLY door data enters through (§2).
  Adding/renaming a parameter means regenerating the fixtures (see
  `references/data-and-io.md`).
- **Authorized output**: `main`'s return value is the program output
  (`ProgramEnd`). It MUST be `()` or a **storage-backed** value — the result
  of a `call!`/`call_seq!`/`call_recur!`/`select!`. Returning a literal or
  locally-built value is an error: the output must provably sit at the end of
  the authorization chain. The runtime exports it as `output.bin` + manifest,
  format-compatible with external inputs.
- **Chains**: in a chain project (root `[chain]` Raster.toml, §1), stage N's
  output artifact is stage N+1's committed input. Each stage's `main`
  parameter is bound in the manifest either to a committed external
  (`inputs.x = { external = { path, index_path, commitment } }`) or to an
  earlier stage's output (`inputs.x = { from = "stage-name" }`). The link is
  by **structural commitment**, not Rust type name — so the boundary type is
  defined (field-for-field identically) in BOTH stages' crates. Full worked
  chain: `references/examples.md`.
- **Program identity**: every verifiable claim ("this program mapped input I
  to output O") must be able to *name the program* — that name is the
  `program_commitment`: a hash over the program's static definition
  (declared interface + CFS control flow + the tile image-id registry that
  pins each tile's actual code). Without it a prover could substitute
  different tile binaries behind the same program shape. It is recorded in
  `Raster.lock` (checked in — commit it, never hand-edit it) and cached as
  `target/raster/program.bin` (regenerable, gitignored). Any change to
  tiles, sequences, or `main`'s signature changes the identity: run
  `cargo raster program --verify` and re-lock deliberately; an unexpected
  mismatch means program behavior drifted. Full explanation:
  `references/data-and-io.md` §6.

## 9. Check ladder — the definition of done

Run in order; do not skip rungs. `cargo build` passing means nothing yet.

**Rung 0 is not authoritative.** A plain `cargo run` executes the program with
authenticated storage off: tile outputs are passed as ordinary Rust values,
nothing is hashed or stored between tiles, and no trace is written — so nothing
from it can be committed or audited. It is the fastest way to iterate on
*logic*. A result that matters must be reproduced at rung 3 or above.

The two modes agree on values for any type obeying RAS-203a (postcard
round-trip is the identity); a disagreement is a violation of that rule, not a
quirk of the mode — see the failure table below.

```bash
# 0. Fastest iteration — plain Rust, no storage, no trace, not authoritative.
#    Whole surface including drafts and recur; values match rung 3, but storage
#    bindings do not exist and nothing here can be committed. ~6x on the example.
cargo run -- --input input.json --input-manifest input_manifest.json

# 1. Both compilation postures (tiles must stay no_std-clean):
cargo check
cargo check -p <lib-crate> --no-default-features

# 2. CFS sanity — verify every intended step appears, with real bindings:
cargo raster cfs && cat target/raster/cfs.json
#    Red flag: an argument you meant as dataflow showing up as
#    {"type": "external"} instead of seq_input/prior-output binding.

# 3. Native run with committed inputs — the first authoritative rung:
cargo raster run --input input.json --input-manifest input_manifest.json

# 4. Commit/audit round-trip (the real verifiability test):
cargo raster run --input input.json --input-manifest input_manifest.json \
  --commit commit.bin --fraud-proof-window-size 32
cargo raster run --input input.json --input-manifest input_manifest.json \
  --audit commit.bin

# 5. Program identity:
cargo raster program --verify

# 6. Chains only:
cargo raster chain run && cargo raster chain audit --execution
```

If any rung fails, map the failure back to a rule before touching code:

| Symptom | Likely violated rule |
| --- | --- |
| step missing from cfs.json | bare call instead of `call!`/`call_seq!` (§4) |
| binding shows as `external` in CFS | computed argument / destructured `let` (§4) |
| guest build failure | `std` leakage into the tile library (§1) |
| "requires a selectable storage list source" | `call_recur!` input not storage-backed (§7) |
| set-once / finalize failure | draft reuse, double-set, or empty recur input (§6, §7) |
| audit divergence with clean native run | nondeterminism in a tile (§3) |
| rung 0 and rung 3 disagree on a value | a tile type whose postcard round-trip is not the identity — usually a `#[serde(skip)]` field, which authenticated mode clears between tiles and rung 0 carries through (RAS-203a) |
| ProgramEnd error on return | `main` returning a non-storage-backed value (§8) |
| run is unexpectedly slow / heavy | unnecessary materialization: whole-object selections, oversized tile outputs (§2) |
| `call_recur! requires a raster-indexed List source` | recur source is a postcard external — re-declare it with `index_path` + `encoding = "raster"` (§7) |
| `.rindex` far larger than the data file | byte data modelled as `List<u8>` instead of `Bytes<P>` (§2) |
| "artifact page size does not match declared `Bytes<N>`" at load | artifact written with a different `#[page_size]` — regenerate the fixture (§2) |
| recur over pages fails `check_previous_chunk_was_full` | recur driven by byte ranges; page count wobbles by alignment — sweep `.pages` instead (§2, `references/recur.md` §1) |
| tile aborts on `page is not i32-aligned` / spans two pages | `page_size` is not a multiple of the record stride (§2) |

A green ladder is necessary, not sufficient: model violations that keep the
mechanics intact — a fake recur, a committed counter list, computation hidden
behind one CFS step — pass all six rungs. Re-read §2 and §7 against the code
before calling it done.

## References

- `references/recur.md` — complete recur-tile/recur-sequence contract: the
  three modes, parameter ordering, `RecurControl`, `chunk`, recur sequences,
  empty inputs.
- `references/data-and-io.md` — `select!` paths and `Selectable`, drafts in
  depth, `input.json`/`input_manifest.json` fixtures and regeneration, output
  artifacts, chain wiring.
- `references/examples.md` — complete worked code (from
  `raster-examples/raster-pipeline`): a full single program (tiles + `main` +
  input fixtures + Cargo.toml posture), a consumer stage, the chain manifest
  that links three programs, and the dissected negative example ("the fake
  Raster program" — §4): cosmetic sequences, whole-collection tiles, wrapper
  structs, giant outputs, smuggled runtime handles.
