# Recur tiles and recur sequences — the complete contract

Recur calls are the ONLY sanctioned way to iterate data in a Raster program.
One `call_recur!`/`call_recur_seq!` site becomes a schema-described loop: each
iteration is its own trace step (its own zkVM replay unit), and the site itself
is a step committing the loop's result. The shapes below are rigidly validated
by the macros — do not improvise.

## 1. The input source

The `input = ...` of a recur call MUST be a **storage-backed `List<T>`**
(for bytes: `select!(List<BytesPage>, region.pages)` — never `Bytes` itself,
and never a byte range; page count wobbles and fails the chunk rules):

- a `select!(List<T>, ...)` result (the normal case),
- a prior `call!`/`call_recur!` binding whose value is a `List<T>`,
- an explicit `storage!(List<T>, reference)` (advanced/test scaffolding),
- a `List<T>` reference passed in as a recur-*sequence* arg (it travels as an
  `AuthRef`, never materialized — this is how the two-collection pattern is
  expressed, §"Restructuring" below).

A plain Rust `Vec` literal or computed vector does not type-check as a source
(`List` is the storage-backed list type); an inline value is rejected at runtime
with `"call_recur! requires a selectable storage list source"`.

## 2. What goes where — input vs state vs params (do not hack the model)

A recur site has four slots, and each has a fixed, non-negotiable role. The
model's whole guarantee — every replay unit touches a bounded amount of data —
lives in this placement:

| Slot | Role | Size discipline |
| --- | --- | --- |
| `input` | THE collection being iterated | the only slot where a collection ever goes; each iteration materializes one element (or one `chunk`) |
| `state` | loop-carried intermediate | tiny — counters, running max, a small struct; re-committed EVERY iteration, so its size is a per-iteration cost |
| `output` | the growing result | append-only draft — the sanctioned place for anything that accumulates; pays only the increment per iteration |
| `args` | per-call constants | small config values: a label, threshold, limit; identical in every iteration |

Consequences, spelled out:

- **Growing data goes in `output`, never in `state`.** `state` is
  re-serialized and re-committed wholesale each iteration; a `Vec` that grows
  inside `state` turns N iterations into O(N²) committed bytes. `output`'s
  draft protocol appends, so it pays only for what each iteration adds.
- **`args` is not a data channel.** `args` are materialized tile arguments, so
  a `List<T>` (or `Vec<T>`) through `args` is now a **compile error** (`args`
  must be `Materializable`). The subtler version still to avoid: passing a
  `Block<T>` window through `args` *does* type-check, but it re-materializes
  that whole window into every single iteration's replay unit — N ×
  window-Merkleization, each replay as big as the thing recur exists to avoid.
  Reduce the other collection to a scalar first (see "Restructuring" below).
- **One recur = one collection.** The step body must never iterate a second
  unbounded thing, wherever it came from.

### The forbidden pattern — item × other-collection scans

```rust
// FORBIDDEN — second collection smuggled through args, inner loop in the body.
// This no longer even compiles: `Vec<u64>`/`List<u64>` as a tile arg is not
// `Materializable`. (Shown to name the anti-pattern; the type system now blocks
// the direct form, and the reasoning applies to any windowed workaround.)
#[tile(kind = recur)]
pub fn match_reading(
    input: RecurInput<u64>,
    state: RecurState<MatchCount>,
    reference_table: Vec<u64>,          // ← a whole collection as a "param" — REJECTED
) -> RecurState<MatchCount> {
    let mut state = state;
    let value = input.into_value();
    for r in &reference_table {          // ← unbounded loop inside one replay unit
        if *r == value {
            state.hits += 1;
        }
    }
    state
}
```

This is not a style issue — it interrupts the Raster model. The point of a
program being "built around Raster" instead of "Raster bolted onto a program"
is that **computation only ever happens around a small amount of data**, and
the protocol can therefore replay, commit, and fraud-prove every step at
bounded cost. Nested loops over a second collection make the per-step cost
proportional to that collection's size — unprovable in practice, and
unaccounted for in the schema (the CFS sees one step; the inner loop is
invisible, exactly like computation hidden in a sequence).

### The fake recur — a synthetic input driving a smuggled state machine

The subtler cousin of args-smuggling: the recur *runs*, but its input is not
the data — it's a dummy counter list whose only job is to force N
repetitions, while the real data and a mutable native-style state machine
hide in `state`/`args`:

```rust
// ❌ DO NOT write this — every marked line is a violation:
#[derive(Serialize, Deserialize)]
pub struct WorkState {
    pub storage_ref: StorageRef,     // ← internal runtime handle in user state
    pub done: bool,
}

#[sequence(kind = recur)]
pub fn process_round(
    _round: RecurSequenceInput<u32>, // ← unused fake counter as the "input"
    state: RecurSequenceState<WorkState>,
    chunks: List<Chunk>,             // ← whole collection through an arg
) -> RecurSequenceState<WorkState> {
    let opened = call!(open_state, state);
    let scan = call_recur!(
        tile = scan_chunk,
        input = chunks,              // ← inner recur fed from an arg
        state = ScanState::initial(),
        args = (opened.clone(),)     // ← wrapper hiding the StorageRef
    );
    call!(close_state, opened, scan)
}
```

Each red flag, and what it tells you:

- **An unused (or `_`-prefixed) recur input is a confession** — if the step
  doesn't consume `input`, the iteration isn't over data; something else is
  the real driver, and it's hiding in the wrong slot.
- **A materialized collection as an arg** — passing the data as a *tile* arg is
  now a compile error (`List`/`Vec` aren't `Materializable`). Note the one
  legitimate exception this example gets *wrong* for other reasons: a `List<T>`
  passed as a recur-**sequence** arg travels as an `AuthRef` (never
  materialized) and MAY feed an inner `call_recur!` — that is the sanctioned
  two-collection pattern (see "Restructuring" below). What damns *this* snippet
  is the fake counter input and the `StorageRef` in state, not the List arg.
- **Internal handles (`StorageRef` etc.) inside state structs** — user
  program state must be plain data. A runtime reference in `state` (or
  wrapped in an "opened" helper object passed via `args`) gives the steps an
  unaccounted door into storage; the protocol can't bind what flowed through
  it. Only the sanctioned protocol types (`Draft<T>`, `Recur*`) may appear
  in signatures — never raw runtime plumbing.
- **The overall shape**: this is a native `while !done` state machine wearing
  a recur costume. The CFS shows "iterate some u32s"; the actual computation
  — which chunks, what order, what state transitions — is invisible. Same
  disease as computation hidden in a sequence.

**Rule: the recur input IS the work.** Each iteration must be driven by a
real, meaningful item — a chunk, a row, a record, a block. If you find
yourself building a `List<u32>` of round numbers to iterate over, stop: either
the real collection should be the input (`input = data_blocks`, not
`input = fake_round_numbers`), or what you want is a growing output — which
is a draft (`output = new!(Output)` + append per item), not rounds.

### The committed counter list — a fake recur laundered through an entry argument

The most dangerous form of the fake recur does not appear in a sequence at
all. It appears in the **input type**, where a counter list is added beside
the real data so that a `call_recur!` has something of the right length to
iterate:

```rust
// ❌ DO NOT write this — `rounds` is not data.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct PromptInput {
    pub chars: List<PromptChar>,
    /// One element per possible BPE merge round. Length is `chars.len()`: each
    /// merge removes one piece, and there are at most `chars.len()` pieces.
    pub rounds: List<u32>,
}

impl PromptInput {
    pub fn new(text: &str) -> Self {
        let chars: Vec<PromptChar> = text.char_indices().map(/* ... */).collect();
        let rounds: Vec<u32> = (0..chars.len() as u32).collect();   // ← fabricated
        Self { chars: chars.into(), rounds: rounds.into() }
    }
}
```

**Why this one is worse than the in-program version: it passes every check.**
`rounds` is a genuine storage-backed `List<u32>`, committed in
`input_manifest.json`, reachable by `select!`, and legal as a `call_recur!`
source. `cargo raster cfs` shows a proper `seq_input` binding, not an
`external`; the run, the commit/audit round-trip and `program --verify` all
succeed. There is no rung of the check ladder that catches it. The commitment
launders the counter into looking like data — but a commitment only attests
*which bytes*, never *that the bytes mean anything*.

The tells, in the order you will notice them:

- **The field is derivable from another field of the same input.**
  `rounds.len() == chars.len()` and `rounds[i] == i`. It carries zero
  information about the prompt, so its commitment constrains nothing. A
  committed argument must be something a verifier could disagree about.
- **The step never consumes its item** — it reads `input.index()` /
  `is_first()` at most, and the real work reaches it through `state`/`args`.
- **The loop count is a bound someone chose**, not the collection being
  processed. The comment justifying the length ("at most `chars.len()`
  pieces") is the giveaway: it is reasoning about a *budget*, and budgets do
  not belong in the data.

**The security consequence, which is the decisive argument.** Once the trip
count lives in the input, it is chosen by whoever writes the fixture. Commit a
`rounds` list that is too short and the loop stops early: the program produces
a *wrong* result — a partially-merged tokenization — with a completely valid
proof over it. The commitment binds the rounds list; nothing binds it to being
long enough, because "long enough" is a property of the computation, not of
the bytes. A program must never let its own termination be an input.

**What is actually wanted here is a data-dependent trip count** (loop until
BPE reaches its fixpoint), and Raster cannot express one today: `state` cannot
carry a collection — a `List<T>` is not `Materializable` and a `Block<T>`
cannot drive a recur — so no loop can transform a collection. See
`docs/proposals/loop-carried-state.md`. Until that lands, the options are:

1. **Recur over the real collection** if the algorithm admits it (one pass per
   element, accumulating into `output`).
2. **Unroll a fixed number of rounds** as explicit `call_seq!` sites and end
   with a tile that *asserts convergence* (`call!(assert_merges_complete, …)?`).
   This is verbose and pins an arbitrary constant into `program_commitment` —
   but it is honest, and honest in the exact place the counter list is not: the
   budget is visible in the program identity where a reader can audit it, and
   an over-long input **fails the run** instead of returning a truncated answer.
   This is what `raster-inference/raster-tokenizer` does (`src/main.rs`,
   `merge_all_rounds`).
3. **Say the model doesn't support it yet.** Preferable to shipping a program
   whose loop bound is attacker-chosen.

The contrast worth keeping: `input` as a **budget** is fine when the budget
*is* the data — recurring over the piece list because L pieces admit at most
L−1 merges bounds the loop by construction. Fabricating a parallel list of
integers to stand in for that same bound is not the same thing, even though
the two have equal length.

### Restructuring cross-collection problems (the legitimate ways)

- **Reduce the other collection first.** If the per-item check only needs a
  summary of collection B (a max, a sum, a count, a small set of thresholds),
  fold B with its own recur into a small `state` result, `select!` the scalar
  fields out, and pass those as `args` to the recur over A. Two linear
  passes, every step bounded.
- **Zip at the boundary.** If the computation is index-aligned (`a[i]` vs
  `b[i]`), build ONE collection of small pair items first — in the
  `gen_input` fixture, in a previous chain stage, or with a prior recur that
  drafts a combined list — then recur over the pairs. One collection, one
  pass.
- **Nest via a recur sequence (the two-collection pattern).** Iterate A with a
  `#[sequence(kind = recur)]`, and pass `List<B>` as a recur-sequence arg — it
  travels as an `AuthRef`, so passing it costs nothing and materializes nothing.
  Inside, `call_recur!(tile = scan_b, input = b_list, …, args = (item,))` walks
  B with A's current element as a scalar arg. Each replay unit touches one
  element of A × one element/`Block` of B.
- **Coarsen with `chunk`, not with params.** If per-element steps are too
  granular (too many tiny replay units), `chunk = N` hands each iteration a
  `Block<T>` window of the SAME collection — bounded by a literal pinned in the
  CFS. That is the sanctioned knob for granularity; widening `args` is not.
- **True joins over two huge collections** (every-item × every-item) are not
  expressible as one recur — and that is by design, because their cost is
  inherently quadratic. Restructure the data (pre-index, pre-bucket, reduce)
  in earlier stages until every step is small; if the problem genuinely
  cannot be decomposed, say so to the user instead of smuggling.

## 3. Recur tile shapes — three modes

Parameter order is fixed and macro-enforced:

```text
fn step(
    input:  RecurInput<T>,          // ALWAYS first
    state:  RecurState<S>,          // optional — BEFORE output if present
    output: RecurOutput<O>,         // optional — needs at least one of state/output
    ...extra_args                   // plain per-call constants, always last
) -> <return type must match the mode>
```

Macro errors you will hit if you deviate (verbatim):

- fewer than 2 params → ``must accept at least `(input, state)` or `(input, output)` ``
- first param not `RecurInput<T>` → ``must start with `input: RecurInput<T>` ``
- output before state → ``must place `state: RecurState<S>` before `output: RecurOutput<T>` ``
- wrong return type → rejected (see `crates/raster/tests/ui/recur_tile_invalid_return.rs`)

### Mode A — state-only (fold / reduce)

Return the state. Use for list → single summary value.

```rust
#[derive(Clone, Debug, Serialize, Deserialize, Selectable)]
pub struct LineLengthStats { pub max_len: u64 }

#[tile(kind = recur)]
pub fn compute_recur_max_line_len(
    input: RecurInput<String>,
    state: RecurState<LineLengthStats>,
) -> RecurState<LineLengthStats> {
    let mut state = state;
    let len = input.value().len() as u64;
    if len > state.max_len {
        state.max_len = len;
    }
    state
}

// call site:
let stats = call_recur!(
    tile = compute_recur_max_line_len,
    input = address_lines,
    state = LineLengthStats { max_len: 0 },   // initial state expression
    args = ()
);
let max = select!(u64, stats.max_len);        // select into the final state
```

`RecurState<S>` derefs to `S` — read/mutate fields directly. The `state = ...`
initializer at the call site is an ordinary expression of type `S`.

### Mode B — output-only (map / build)

Return the output. Use for list → one built object (draft semantics).

```rust
#[tile(kind = recur)]
pub fn build_recur_draft_greeting(
    input: RecurInput<String>,
    output: RecurOutput<CollectiveGreeting>,
    title: String,                             // extra arg
) -> RecurOutput<CollectiveGreeting> {
    let mut output = output;
    if input.is_first() {
        output.title().set(title);             // set-once: guard with is_first()
    }
    output.lines().push(input.into_value());
    output
}

// call site:
let greeting = call_recur!(
    tile = build_recur_draft_greeting,
    input = address_lines.clone(),
    output = new!(CollectiveGreeting),
    args = ("Recur-built greeting".to_string(),)
);
```

`RecurOutput<O>` is a draft handle: set-once accessors
(`.field().set(v)`, `.list().push(v)`), linear (rebind every iteration —
the macro-generated driver threads it for you). The site finalizes the draft
and binds the materialized `O`.

### Mode C — state + output

Two valid return forms:

- `(RecurState<S>, RecurOutput<O>)` — plain tuple; runs over the whole list:

  ```rust
  #[tile(kind = recur)]
  pub fn process_chunk(
      input: RecurInput<Chunk>,
      state: RecurState<SmallState>,
      output: RecurOutput<Output>,
      config: Config,
  ) -> (RecurState<SmallState>, RecurOutput<Output>) {
      let chunk = input.into_value();
      let mut state = state;
      let mut output = output;
      let result = process_bounded_chunk(chunk, &config);
      state.count += 1;
      state.checksum = update_checksum(state.checksum, &result);
      output.items().push(result);
      (state, output)
  }
  ```

- `RecurControl<(RecurState<S>, RecurOutput<O>)>` — when the loop must be able
  to stop early. This is the ONLY early-exit mechanism in the model:

```rust
#[tile(kind = recur)]
pub fn build_limited_recur_greeting(
    input: RecurInput<String>,
    state: RecurState<GreetingLimitState>,
    output: RecurOutput<CollectiveGreeting>,
    title: String,
    limit: u64,
) -> RecurControl<(RecurState<GreetingLimitState>, RecurOutput<CollectiveGreeting>)> {
    let mut state = state;
    let mut output = output;
    if input.is_first() {
        output.title().set(title);
    }
    state.seen += 1;
    output.lines().push(input.into_value());

    if state.seen >= limit {
        RecurControl::Break((state, output))     // stop after this element
    } else {
        RecurControl::Continue((state, output))
    }
}

// call site (state AND output):
let limited = call_recur!(
    tile = build_limited_recur_greeting,
    input = address_lines,
    state = GreetingLimitState { seen: 0 },
    output = new!(CollectiveGreeting),
    args = ("State+output recur greeting".to_string(), 2)
);
```

## 4. `call_recur!` named-argument form

```text
call_recur!(
    tile   = <recur tile name>,          // required, first
    input  = <storage-backed list>,      // required
    chunk  = <integer literal>,          // optional — must be a literal (pinned in CFS)
    state  = <initial S expression>,     // required iff tile has RecurState
    output = <new!(O) or draft handle>,  // required iff tile has RecurOutput
    args   = (<extras>,)                 // required, LAST — () if none
)
```

- `state`/`output` presence must exactly match the tile's mode; the macro
  rejects mismatches: ``call_recur! requires `state = ...` and/or `output = ...` ``.
- `args = (...)` is always last; error otherwise:
  ``requires `state = ...` and/or `output = ...` before `args = (...)` ``.
- The binding of the call is the finalized result: `S` (state-only), `O`
  (output-only), or the pair's materialized parts for state+output — select
  into it like any other value.

## 5. Chunked iteration — `chunk = N`

`chunk = N` makes each iteration receive up to `N` elements as a bounded window:
the step's element type changes from `T` to `Block<T>`, and the loop runs
`ceil(len / N)` times while the source stays ONE authenticated binding. `Block<T>`
is the sanctioned way a step sees several elements at once — the framework builds
it with the bound `N` pinned in the CFS.

**How a chunk is proved.** The source type stays `AuthRef<List<T>>` — there is no
`List<Block<T>>` anywhere in the storage tree, so there would be nothing to prove
membership in. Each iteration is a **range selection** over the real list,
`[i·N, min((i+1)·N, len))`, carrying a `ListRange` proof that folds to the same
committed list root a single-element selection folds to. Only the final chunk is
ever short. This is why `N` must be a literal: the driver, not an adapter,
computes the ranges, and the audit checks each iteration's span against the
program's declared shape.

```rust
#[tile(kind = recur)]
pub fn collect_line_chunk(
    input: RecurInput<Block<String>>,    // note: Block<String>, not String
    output: RecurOutput<CollectiveGreeting>,
    title: String,
) -> RecurOutput<CollectiveGreeting> {
    let mut output = output;
    if input.is_first() {
        output.title().set(title);
    }
    let chunk_index = input.index();     // chunk index, not element index
    let chunk = input.into_value();
    output.lines().push(format!("chunk {}: {}", chunk_index, chunk.join(", ")));
    output
}

let chunked = call_recur!(
    tile = collect_line_chunk,
    input = address_lines,
    chunk = 2,                            // MUST be an integer literal
    output = new!(CollectiveGreeting),
    args = ("Chunked greeting".to_string(),)
);
```

`chunk` must be a literal — a named constant fails:
``call_recur! `chunk = ...` must be an integer literal so it can be pinned in the CFS``.

### Sweeping a byte region

A `Bytes<P>` sweep is an ordinary list recur over `select!(List<BytesPage>,
region.pages)`; `chunk = N` gives the step a `RecurInput<Block<BytesPage>>` and
works unchanged. Two constraints are specific to bytes:

**Never drive a recur with byte ranges.** A byte range returns the whole pages
*covering* it, so a request of length `L` yields `⌈L/P⌉` or `⌈L/P⌉ + 1` pages
depending on where it starts:

| request (`P` = 256 KiB) | pages |
| --- | --- |
| `[0 .. 262_144)` | 1 |
| `[1 .. 262_145)` | 2 |
| `[0 .. 1_048_576)` | 4 |
| `[1 .. 1_048_577)` | 5 |

Chunking requires every chunk to be exactly the declared size except the last,
so an alternating 4/5 count fails `check_previous_chunk_was_full` on the first
short iteration. **Ranges are for random access; recur is for sweeps.** A sweep
over `.pages` is aligned by construction and never wobbles.

**Straddling records are a `page_size` bug, not a tile problem.** If a record
crosses a page boundary the step needs loop-carried stitching state — the
partial record from the previous page — which is exactly the "state as a
smuggled accumulator" pattern §2 forbids, and it makes the final page a special
case on top. Choose `page_size` as a multiple of the record stride and the
problem does not exist. This constraint outranks every performance
consideration; tune the page size for cycles only *within* the set of stride
multiples.

## 6. `RecurInput` API inside the step

- `input.value()` — borrow the element (or chunk)
- `input.into_value()` — take it
- `input.index()` — 0-based iteration index (chunk index under `chunk = N`)
- `input.is_first()` — `index() == 0`; use to guard set-once initialization

## 7. Empty inputs

An empty source list **skips the step function entirely** and goes straight to
finalization. Finalization succeeds only if the untouched output schema
materializes without set-once writes:

- state-only: fine — the initial state is the result;
- output-bearing modes: fails if `O` has required set-once fields that were
  never written. Design outputs for this, or guarantee non-empty input.

## 8. Recur sequences — several tiles per element

When one element needs multiple tile steps, write a recur *sequence* and drive
it with `call_recur_seq!`. Same three modes, with `RecurSequence*` wrapper
types:

```rust
#[sequence(kind = recur)]
fn collect_prefixed_lines(
    input: RecurSequenceInput<String>,        // first
    output: RecurSequenceOutput<LineBundle>,  // state and/or output, same ordering
    prefix: String,                           // extras last
) -> RecurSequenceOutput<LineBundle> {
    // Body = the restricted sequence grammar. The handles are OPAQUE here:
    // pass them to tiles; the tiles receive plain values / Draft<T>.
    let line = call!(prefix_line, input, prefix);
    call!(append_prefixed_line, output, line)
}

// The tiles are ORDINARY tiles — no Recur* types in their signatures:
#[tile]
fn prefix_line(line: String, prefix: String) -> String { /* ... */ }

#[tile]
fn append_prefixed_line(output: Draft<LineBundle>, line: String) -> Draft<LineBundle> { /* ... */ }

// call site:
let bundle = call_recur_seq!(
    sequence = collect_prefixed_lines,
    input = storage!(List<String>, source),
    output = output,                          // new!(T) or an already-threaded draft
    args = (prefix_arg,)
);
```

Recur-sequence-specific rules (macro/UI-test enforced):

- The body MUST NOT read the input handle itself — no `input.value()` /
  `.into_value()` (that would be sequence-level computation). Pass `input` to
  a tile; materialization happens at the call boundary.
- MUST NOT return `RecurControl` — early termination belongs to recur *tiles*.
  Return the handle(s): `RecurSequenceState<S>`, `RecurSequenceOutput<O>`, or
  the tuple `(RecurSequenceState<S>, RecurSequenceOutput<O>)`.
- Tile handles (`RecurInput`/`RecurState`/`RecurOutput`) are rejected in
  recur-sequence signatures — use the `RecurSequence*` forms.
- `main` cannot be `#[sequence(kind = recur)]`.

## 9. Choosing tile-recur vs sequence-recur

- One computation per element → recur tile (`call_recur!`).
- A pipeline per element (transform, then accumulate; validate, then build) →
  recur sequence (`call_recur_seq!`) so each stage stays its own small,
  replayable tile.
- Heavy per-element work in one tile body is a red flag either way — split it.
