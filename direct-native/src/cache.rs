use std::collections::BTreeMap;

use anyhow::{bail, Result};
use raster::List;

use decode_select_token::input as decode_input;
use input_embedding::input as embedding_input;
use prefill_finalize::input as finalize_input;
use prefill_prepare_aux::input as aux_input;
use prefill_range::input as range_input;
use prompt_prepare::input as prompt_input;

#[derive(Clone, Debug)]
pub enum CachedStageValue {
    PromptTokenization(prompt_input::PromptTokenization),
    EmbeddedActivations(embedding_input::ActivationSequence),
    PleLayerInputs(aux_input::PleLayerInputs),
    RangeActivations(range_input::ActivationSequence),
    PrefillLogits(finalize_input::PrefillLogits),
    SelectedToken(decode_input::SelectedToken),
}

#[derive(Clone, Debug)]
pub struct CachedStageOutput {
    pub structural_commitment: Vec<u8>,
    pub value: CachedStageValue,
}

#[derive(Default)]
pub struct StageOutputCache {
    outputs: BTreeMap<String, CachedStageOutput>,
}

pub type CachedInputs = BTreeMap<String, CachedStageValue>;

impl StageOutputCache {
    pub fn insert(
        &mut self,
        stage: impl Into<String>,
        structural_commitment: Vec<u8>,
        value: CachedStageValue,
    ) {
        self.outputs.insert(
            stage.into(),
            CachedStageOutput {
                structural_commitment,
                value,
            },
        );
    }

    pub fn get(
        &self,
        stage: &str,
        expected_structural_commitment: &[u8],
    ) -> Result<Option<CachedStageValue>> {
        let Some(output) = self.outputs.get(stage) else {
            return Ok(None);
        };
        if output.structural_commitment != expected_structural_commitment {
            bail!("cached direct-native output for stage `{stage}` has stale commitment");
        }
        Ok(Some(output.value.clone()))
    }
}

impl CachedStageValue {
    pub fn as_embedding_prompt(&self) -> Result<embedding_input::PromptTokenization> {
        match self {
            Self::PromptTokenization(value) => Ok(embedding_input::PromptTokenization {
                token_ids: value.token_ids.clone(),
            }),
            _ => bail!("cached value is not a prompt tokenization"),
        }
    }

    pub fn as_aux_activation_sequence(&self) -> Result<aux_input::ActivationSequence> {
        match self {
            Self::EmbeddedActivations(value) => Ok(aux_activation_from_embedding(value)),
            Self::RangeActivations(value) => Ok(aux_activation_from_range(value)),
            _ => bail!("cached value is not an activation sequence"),
        }
    }

    pub fn as_range_activation_sequence(&self) -> Result<range_input::ActivationSequence> {
        match self {
            Self::EmbeddedActivations(value) => Ok(range_activation_from_embedding(value)),
            Self::RangeActivations(value) => Ok(value.clone()),
            _ => bail!("cached value is not an activation sequence"),
        }
    }

    pub fn as_range_ple_inputs(&self) -> Result<range_input::PleLayerInputs> {
        match self {
            Self::PleLayerInputs(value) => Ok(range_ple_from_aux(value)),
            _ => bail!("cached value is not PLE layer inputs"),
        }
    }

    pub fn as_finalize_activation_sequence(&self) -> Result<finalize_input::ActivationSequence> {
        match self {
            Self::EmbeddedActivations(value) => Ok(finalize_activation_from_embedding(value)),
            Self::RangeActivations(value) => Ok(finalize_activation_from_range(value)),
            _ => bail!("cached value is not an activation sequence"),
        }
    }

    pub fn as_decode_logits(&self) -> Result<decode_input::PrefillLogits> {
        match self {
            Self::PrefillLogits(value) => Ok(decode_logits_from_finalize(value)),
            _ => bail!("cached value is not prefill logits"),
        }
    }
}

fn map_list<T, U>(source: &List<T>, mut convert: impl FnMut(&T) -> U) -> List<U> {
    List::from(
        source
            .iter()
            .map(|value| convert(value))
            .collect::<Vec<_>>(),
    )
}

fn aux_activation_from_embedding(
    source: &embedding_input::ActivationSequence,
) -> aux_input::ActivationSequence {
    aux_input::ActivationSequence {
        rows: map_list(&source.rows, |row| aux_input::ActivationRow {
            token_id: row.token_id,
            values: row.values.clone(),
        }),
        errors: source.errors.clone(),
        kv: map_list(&source.kv, |row| aux_input::KeyRow {
            position: row.position,
            k: row.k.clone(),
            v: row.v.clone(),
        }),
    }
}

fn aux_activation_from_range(
    source: &range_input::ActivationSequence,
) -> aux_input::ActivationSequence {
    aux_input::ActivationSequence {
        rows: map_list(&source.rows, |row| aux_input::ActivationRow {
            token_id: row.token_id,
            values: row.values.clone(),
        }),
        errors: source.errors.clone(),
        kv: map_list(&source.kv, |row| aux_input::KeyRow {
            position: row.position,
            k: row.k.clone(),
            v: row.v.clone(),
        }),
    }
}

fn range_activation_from_embedding(
    source: &embedding_input::ActivationSequence,
) -> range_input::ActivationSequence {
    range_input::ActivationSequence {
        rows: map_list(&source.rows, |row| range_input::ActivationRow {
            token_id: row.token_id,
            values: row.values.clone(),
        }),
        errors: source.errors.clone(),
        kv: map_list(&source.kv, |row| range_input::KeyRow {
            position: row.position,
            k: row.k.clone(),
            v: row.v.clone(),
        }),
    }
}

fn finalize_activation_from_embedding(
    source: &embedding_input::ActivationSequence,
) -> finalize_input::ActivationSequence {
    finalize_input::ActivationSequence {
        rows: map_list(&source.rows, |row| finalize_input::ActivationRow {
            token_id: row.token_id,
            values: row.values.clone(),
        }),
        errors: source.errors.clone(),
        kv: map_list(&source.kv, |row| finalize_input::KeyRow {
            position: row.position,
            k: row.k.clone(),
            v: row.v.clone(),
        }),
    }
}

fn finalize_activation_from_range(
    source: &range_input::ActivationSequence,
) -> finalize_input::ActivationSequence {
    finalize_input::ActivationSequence {
        rows: map_list(&source.rows, |row| finalize_input::ActivationRow {
            token_id: row.token_id,
            values: row.values.clone(),
        }),
        errors: source.errors.clone(),
        kv: map_list(&source.kv, |row| finalize_input::KeyRow {
            position: row.position,
            k: row.k.clone(),
            v: row.v.clone(),
        }),
    }
}

fn range_ple_from_aux(source: &aux_input::PleLayerInputs) -> range_input::PleLayerInputs {
    range_input::PleLayerInputs {
        layer_idx: source.layer_idx,
        rows: map_list(&source.rows, |row| range_input::PleRow {
            values: row.values.clone(),
        }),
        errors: source.errors.clone(),
    }
}

fn decode_logits_from_finalize(
    source: &finalize_input::PrefillLogits,
) -> decode_input::PrefillLogits {
    decode_input::PrefillLogits {
        decode_position: source.decode_position,
        logits: map_list(&source.logits, |entry| decode_input::LogitEntry {
            token_id: entry.token_id,
            value: entry.value,
        }),
        errors: source.errors.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_rejects_stale_commitment() {
        let mut cache = StageOutputCache::default();
        cache.insert(
            "producer",
            vec![0xde, 0xad],
            CachedStageValue::PromptTokenization(prompt_input::PromptTokenization {
                token_ids: List::from(vec![1, 2, 3]),
            }),
        );

        let error = cache.get("producer", &[0xbe, 0xef]).unwrap_err();

        assert!(error.to_string().contains("stale commitment"));
    }

    #[test]
    fn cached_prompt_converts_to_embedding_input() {
        let cached = CachedStageValue::PromptTokenization(prompt_input::PromptTokenization {
            token_ids: List::from(vec![7, 8]),
        });

        let prompt = cached.as_embedding_prompt().unwrap();

        assert_eq!(prompt.token_ids.as_slice(), &[7, 8]);
    }
}
