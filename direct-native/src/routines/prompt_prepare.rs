use std::path::Path;

use anyhow::{bail, Result};
use prompt_prepare::input::{
    merge_bucket_of, vocab_bucket_of, BpePieces, MergeMatch, MergeStep, PromptTokenization,
    PromptTokenizer, UNK_TOKEN_ID,
};
use raster::List;

use crate::artifact_io::{with_main_sequence_scope, with_stage_sequence_scope};
use crate::cache::CachedInputs;

const MERGE_ROUNDS: usize = 8;
const TERMINATOR: &str = "</w>";

pub struct Inputs {
    pub tokenizer: PromptTokenizer,
    pub initial_pieces: BpePieces,
}

pub fn load_inputs_from_args() -> Result<Inputs> {
    with_main_sequence_scope(load_inputs_from_initialized_runtime)
}

pub fn load_inputs_from_paths(
    input: &Path,
    input_manifest: &Path,
    _cached_inputs: &CachedInputs,
) -> Result<Inputs> {
    with_stage_sequence_scope(input, input_manifest, load_inputs_from_initialized_runtime)
}

pub fn run_direct(inputs: &Inputs) -> Result<PromptTokenization> {
    let mut pieces = inputs
        .initial_pieces
        .pieces
        .iter()
        .cloned()
        .collect::<Vec<_>>();

    for _ in 0..MERGE_ROUNDS {
        pieces = merge_round(&pieces, &inputs.tokenizer)?;
    }

    let remaining = count_remaining_merges(&pieces, &inputs.tokenizer)?;
    if remaining != 0 {
        bail!(
            "{remaining} adjacent pair(s) would still merge after the last round; \
             the merge unroll in `main` is too short for this prompt"
        );
    }

    let token_ids = pieces
        .iter()
        .filter(|piece| piece.as_str() != TERMINATOR)
        .map(|piece| resolve_piece_token(piece, &inputs.tokenizer))
        .collect::<Result<Vec<_>>>()?;

    Ok(PromptTokenization {
        token_ids: List::from(token_ids),
    })
}

fn merge_round(pieces: &[String], tokenizer: &PromptTokenizer) -> Result<Vec<String>> {
    let mut output = Vec::new();
    let mut pending = String::new();
    let mut has_pending = false;

    for piece in pieces {
        let step = MergeStep {
            pending: pending.clone(),
            has_pending,
            piece: piece.clone(),
        };
        let hit = find_merge(&step, tokenizer)?;

        if step.has_pending && !hit.matched {
            output.push(step.pending.clone());
        }
        if step.piece == TERMINATOR {
            output.push(step.piece.clone());
        }

        if hit.matched {
            pending = hit.merged;
        } else {
            pending = step.piece;
        }
        has_pending = true;
    }

    Ok(output)
}

fn count_remaining_merges(pieces: &[String], tokenizer: &PromptTokenizer) -> Result<u32> {
    let mut previous = String::new();
    let mut has_previous = false;
    let mut remaining = 0_u32;

    for piece in pieces {
        let step = MergeStep {
            pending: previous,
            has_pending: has_previous,
            piece: piece.clone(),
        };
        let hit = find_merge(&step, tokenizer)?;
        remaining = remaining.saturating_add(if hit.matched { 1 } else { 0 });
        previous = step.piece;
        has_previous = true;
    }

    Ok(remaining)
}

fn find_merge(step: &MergeStep, tokenizer: &PromptTokenizer) -> Result<MergeMatch> {
    let mut hit = MergeMatch {
        matched: false,
        rank: 0,
        merged: String::new(),
    };
    if !step.has_pending {
        return Ok(hit);
    }

    let bucket_idx =
        merge_bucket_of(&step.pending, &step.piece, tokenizer.merge_bucket_count) as usize;
    let bucket = tokenizer.merge_buckets.get(bucket_idx).ok_or_else(|| {
        anyhow::anyhow!(
            "merge bucket index {bucket_idx} is out of range for {} buckets",
            tokenizer.merge_buckets.len()
        )
    })?;
    for rule in bucket.rules.iter() {
        let applies = rule.left == step.pending && rule.right == step.piece;
        if applies && (!hit.matched || rule.rank < hit.rank) {
            hit.matched = true;
            hit.rank = rule.rank;
            hit.merged = rule.merged.clone();
        }
    }
    Ok(hit)
}

fn resolve_piece_token(piece: &str, tokenizer: &PromptTokenizer) -> Result<u32> {
    let bucket_idx = vocab_bucket_of(piece, tokenizer.vocab_bucket_count) as usize;
    let bucket = tokenizer.vocab_buckets.get(bucket_idx).ok_or_else(|| {
        anyhow::anyhow!(
            "vocab bucket index {bucket_idx} is out of range for {} buckets",
            tokenizer.vocab_buckets.len()
        )
    })?;
    Ok(bucket
        .entries
        .iter()
        .find(|entry| entry.token == piece)
        .map(|entry| entry.id)
        .unwrap_or(UNK_TOKEN_ID))
}

fn load_inputs_from_initialized_runtime() -> Result<Inputs> {
    let binding = raster::start_program(&[
        raster::entry_argument_spec::<PromptTokenizer>("tokenizer"),
        raster::entry_argument_spec::<BpePieces>("initial_pieces"),
    ])?;
    let tokenizer = raster::materialize_auth_return(raster::entry_argument_auth_ref::<
        PromptTokenizer,
    >(binding.reference.clone(), "tokenizer"));
    let initial_pieces = raster::materialize_auth_return(raster::entry_argument_auth_ref::<
        BpePieces,
    >(binding.reference, "initial_pieces"));
    Ok(Inputs {
        tokenizer,
        initial_pieces,
    })
}
