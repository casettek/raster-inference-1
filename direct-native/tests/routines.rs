use decode_select_token::input::{DecodeEdge, LogitEntry, PrefillLogits};
use direct_native::routines;
use input_embedding::input::{EmbeddingTable, PromptTokenization};
use output_finalize::input::{DecoderTable, DecoderToken};
use prefill_finalize::input::{FinalHead, FinalHeadParams};
use prefill_prepare_aux::input::{PleLayer, PleLayerParams};
use prompt_prepare::input::{BpePieces, MergeBucket, PromptTokenizer, TokenEntry, VocabBucket};
use raster::{Bytes, List};

const ONE: i32 = 1 << 16;

fn bytes_of_i32s(values: &[i32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn paged(values: &[i32]) -> Bytes<196_608> {
    Bytes::<196_608>::paged(bytes_of_i32s(values)).unwrap()
}

#[test]
fn decode_select_token_keeps_first_token_on_tie() {
    let output =
        routines::decode_select_token::run_direct(&routines::decode_select_token::Inputs {
            logits: PrefillLogits {
                decode_position: 9,
                logits: List::from(vec![
                    LogitEntry {
                        token_id: 5,
                        value: 10,
                    },
                    LogitEntry {
                        token_id: 3,
                        value: 10,
                    },
                ]),
                errors: List::new(),
            },
            prior: DecodeEdge {
                has_selected: true,
                decode_position: 8,
                token_id: 11,
                value: 1,
                generated_token_ids: List::from(vec![11, 12]),
            },
        })
        .unwrap();

    assert!(output.has_selected);
    assert_eq!(output.decode_position, 9);
    assert_eq!(output.token_id, 5);
    assert_eq!(output.value, 10);
    assert_eq!(output.generated_token_ids.as_slice(), &[11, 12, 5]);
}

#[test]
fn decode_init_outputs_empty_edge() {
    let output = routines::decode_init::run_direct(&routines::decode_init::Inputs).unwrap();

    assert!(!output.has_selected);
    assert_eq!(output.decode_position, 0);
    assert_eq!(output.token_id, 0);
    assert_eq!(output.value, 0);
    assert!(output.generated_token_ids.is_empty());
}

#[test]
fn decode_embed_uses_selected_decode_position() {
    let output = routines::decode_embed::run_direct(&routines::decode_embed::Inputs {
        selected: decode_embed::input::DecodeEdge {
            has_selected: true,
            decode_position: 9,
            token_id: 1,
            value: 10,
            generated_token_ids: List::from(vec![1]),
        },
        embedding: decode_embed::input::EmbeddingTable {
            hidden_size: 2,
            embedding_scale: ONE,
            values: decode_embed::input::EmbeddingTable {
                hidden_size: 2,
                embedding_scale: ONE,
                values: paged(&[0, 0, ONE, 2 * ONE]),
            }
            .values,
        },
    })
    .unwrap();

    assert_eq!(output.rows.len(), 1);
    assert_eq!(output.rows[0].token_id, 1);
    assert_eq!(output.start_position, 9);
    assert!(output.errors.is_empty());
}

#[test]
fn output_finalize_decodes_generated_ids() {
    let output = routines::output_finalize::run_direct(&routines::output_finalize::Inputs {
        edge: output_finalize::input::DecodeEdge {
            has_selected: true,
            decode_position: 2,
            token_id: 1,
            value: 10,
            generated_token_ids: List::from(vec![0, 1]),
        },
        decoder: DecoderTable {
            tokens: List::from(vec![
                DecoderToken {
                    token: "▁Hi".to_string(),
                    special: false,
                    terminal: false,
                },
                DecoderToken {
                    token: "!".to_string(),
                    special: false,
                    terminal: false,
                },
            ]),
        },
    })
    .unwrap();

    assert_eq!(output.generated_token_count, 2);
    assert_eq!(output.generated_token_ids.as_slice(), &[0, 1]);
    assert_eq!(output.generated_text, " Hi!");
    assert_eq!(output.stop_reason, "max_new_tokens");
}

#[test]
fn prompt_prepare_resolves_vocab_bucket() {
    let output = routines::prompt_prepare::run_direct(&routines::prompt_prepare::Inputs {
        tokenizer: PromptTokenizer {
            vocab_bucket_count: 1,
            merge_bucket_count: 1,
            vocab_buckets: List::from(vec![VocabBucket {
                entries: List::from(vec![TokenEntry {
                    token: "hello".to_string(),
                    id: 42,
                }]),
            }]),
            merge_buckets: List::from(vec![MergeBucket { rules: List::new() }]),
        },
        initial_pieces: BpePieces {
            pieces: List::from(vec!["hello".to_string(), "</w>".to_string()]),
        },
    })
    .unwrap();

    assert_eq!(output.token_ids.as_slice(), &[42]);
}

#[test]
fn input_embedding_gathers_and_scales_rows() {
    let output = routines::input_embedding::run_direct(&routines::input_embedding::Inputs {
        prompt: PromptTokenization {
            token_ids: List::from(vec![1]),
        },
        embedding: EmbeddingTable {
            hidden_size: 2,
            embedding_scale: ONE,
            values: paged(&[0, 0, ONE, 2 * ONE]),
        },
    })
    .unwrap();

    assert_eq!(output.rows.len(), 1);
    assert_eq!(output.rows[0].token_id, 1);
    assert!(output.errors.is_empty());
    assert!(output.kv.is_empty());
}

#[test]
fn prefill_prepare_aux_publishes_one_row() {
    let norm = prefill_prepare_aux::input::pack_i32s(&[ONE, ONE]);
    let activation = prefill_prepare_aux::input::ActivationSequence {
        rows: List::from(vec![prefill_prepare_aux::input::ActivationRow {
            token_id: 0,
            values: prefill_prepare_aux::input::pack_i32s(&[ONE, 0]),
        }]),
        errors: List::new(),
        kv: List::new(),
        start_position: 0,
    };

    let output =
        routines::prefill_prepare_aux::run_direct(&routines::prefill_prepare_aux::Inputs {
            embedded: activation,
            layer: PleLayer {
                params: PleLayerParams {
                    layer_idx: 7,
                    hidden_size: 2,
                    ple_width: 2,
                    embedding_scale: ONE,
                    projection_scalar: ONE,
                    input_scale: ONE,
                    norm_eps: 0,
                    norm_weights: norm.clone(),
                },
                embeddings: paged(&[ONE, 0]),
                projection: paged(&[ONE, 0, 0, ONE]),
            },
        })
        .unwrap();

    assert_eq!(output.layer_idx, 7);
    assert_eq!(output.rows.len(), 1);
    assert!(output.errors.is_empty());
}

#[test]
fn prefill_prepare_aux_preserves_prompt_row_order() {
    let norm = prefill_prepare_aux::input::pack_i32s(&[ONE, ONE]);
    let row = |token_id, values: &[i32]| prefill_prepare_aux::input::ActivationRow {
        token_id,
        values: prefill_prepare_aux::input::pack_i32s(values),
    };
    let layer = || PleLayer {
        params: PleLayerParams {
            layer_idx: 7,
            hidden_size: 2,
            ple_width: 2,
            embedding_scale: ONE,
            projection_scalar: ONE,
            input_scale: ONE,
            norm_eps: 0,
            norm_weights: norm.clone(),
        },
        embeddings: paged(&[0, 0, ONE, 0, 0, ONE]),
        projection: paged(&[ONE, 0, 0, ONE]),
    };
    let run = |rows| {
        routines::prefill_prepare_aux::run_direct(&routines::prefill_prepare_aux::Inputs {
            embedded: prefill_prepare_aux::input::ActivationSequence {
                rows: List::from(rows),
                errors: List::new(),
                kv: List::new(),
                start_position: 0,
            },
            layer: layer(),
        })
        .unwrap()
    };

    let output = run(vec![
        row(2, &[ONE, 0]),
        row(0, &[0, ONE]),
        row(1, &[ONE, ONE]),
    ]);
    let expected = [
        run(vec![row(2, &[ONE, 0])]).rows[0].values.clone(),
        run(vec![row(0, &[0, ONE])]).rows[0].values.clone(),
        run(vec![row(1, &[ONE, ONE])]).rows[0].values.clone(),
    ];

    assert_eq!(output.rows.len(), 3);
    assert!(output.errors.is_empty());
    assert_eq!(output.rows[0].values, expected[0]);
    assert_eq!(output.rows[1].values, expected[1]);
    assert_eq!(output.rows[2].values, expected[2]);
}

#[test]
fn prefill_finalize_scores_projection_rows() {
    let norm = prefill_finalize::input::pack_i32s(&[ONE, ONE]);
    let activations = prefill_finalize::input::ActivationSequence {
        rows: List::from(vec![prefill_finalize::input::ActivationRow {
            token_id: 7,
            values: prefill_finalize::input::pack_i32s(&[ONE, 0]),
        }]),
        errors: List::new(),
        kv: List::new(),
        start_position: 4,
    };

    let output = routines::prefill_finalize::run_direct(&routines::prefill_finalize::Inputs {
        activations,
        head: FinalHead {
            params: FinalHeadParams {
                hidden_size: 2,
                norm_eps: 0,
                softcap: 0,
                norm_weights: norm,
            },
            projection: paged(&[ONE, 0, 0, ONE]),
        },
    })
    .unwrap();

    assert_eq!(output.decode_position, 5);
    assert_eq!(output.logits.len(), 2);
    assert!(output.errors.is_empty());
}
