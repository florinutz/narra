//! Named Entity Recognition service using BERT-based token classification.
//!
//! Extracts person, location, organization, and miscellaneous entities from text.
//! Uses dslim/bert-base-NER (BIO tagging) with lazy annotation caching.

use std::sync::Arc;

use async_trait::async_trait;
use tracing::{info, warn};

use crate::db::connection::NarraDb;
use crate::embedding::candle_backend::{download_model, select_device, TokenClassifier};
use crate::models::annotation::{
    get_annotation, upsert_annotation, AnnotationCreate, NerEntity, NerOutput,
};
use crate::NarraError;

const NER_MODEL_REPO: &str = "dslim/bert-base-NER";
const NER_MODEL_VERSION: &str = "bert-base-ner-v1";
const NER_MODEL_TYPE: &str = "ner";

/// Service trait for named entity recognition.
#[async_trait]
pub trait NerService: Send + Sync {
    /// Get entities for an entity, using cached annotations when fresh.
    async fn get_entities(&self, entity_id: &str, text: &str) -> Result<NerOutput, NarraError>;

    /// Extract entities from text without caching (one-off extraction).
    async fn extract_entities(&self, text: &str) -> Result<NerOutput, NarraError>;

    /// Whether the NER model is loaded and available.
    fn is_available(&self) -> bool;
}

/// Local NER service using candle BERT token classifier.
pub struct LocalNerService {
    classifier: Option<Arc<TokenClassifier>>,
    db: Arc<NarraDb>,
    available: bool,
}

impl LocalNerService {
    /// Create a new local NER service.
    ///
    /// Downloads and loads the BERT NER model eagerly. If model loading fails,
    /// the service will be unavailable but won't error (graceful degradation).
    pub fn new(db: Arc<NarraDb>) -> Self {
        let files = match download_model(NER_MODEL_REPO, None) {
            Ok(files) => files,
            Err(e) => {
                warn!(
                    "Failed to download NER model: {}. Entity extraction will be unavailable.",
                    e
                );
                return Self {
                    classifier: None,
                    db,
                    available: false,
                };
            }
        };

        let device = select_device();

        match TokenClassifier::new(&files, device) {
            Ok(classifier) => {
                info!(
                    "NER classifier loaded ({}, {} labels via candle)",
                    NER_MODEL_REPO,
                    classifier.num_labels()
                );
                Self {
                    classifier: Some(Arc::new(classifier)),
                    db,
                    available: true,
                }
            }
            Err(e) => {
                warn!(
                    "Failed to load NER model: {}. Entity extraction will be unavailable.",
                    e
                );
                Self {
                    classifier: None,
                    db,
                    available: false,
                }
            }
        }
    }
}

#[async_trait]
impl NerService for LocalNerService {
    async fn get_entities(&self, entity_id: &str, text: &str) -> Result<NerOutput, NarraError> {
        // Check for fresh cached annotation
        if let Ok(Some(annotation)) = get_annotation(&self.db, entity_id, NER_MODEL_TYPE).await {
            if !annotation.stale {
                if let Ok(output) = serde_json::from_value::<NerOutput>(annotation.output) {
                    return Ok(output);
                }
                warn!("Corrupt NER annotation for {}, recomputing", entity_id);
            }
        }

        // Extract entities
        let output = self.extract_entities(text).await?;

        // Cache as annotation
        let output_json = serde_json::to_value(&output)
            .map_err(|e| NarraError::Database(format!("Failed to serialize NER output: {}", e)))?;

        if let Err(e) = upsert_annotation(
            &self.db,
            AnnotationCreate {
                entity_id: entity_id.to_string(),
                model_type: NER_MODEL_TYPE.to_string(),
                model_version: NER_MODEL_VERSION.to_string(),
                output: output_json,
            },
        )
        .await
        {
            warn!("Failed to cache NER annotation for {}: {}", entity_id, e);
        }

        Ok(output)
    }

    async fn extract_entities(&self, text: &str) -> Result<NerOutput, NarraError> {
        let classifier = self
            .classifier
            .as_ref()
            .ok_or_else(|| NarraError::Database("NER model not loaded".to_string()))?
            .clone();

        let text_owned = text.to_string();

        let result = tokio::task::spawn_blocking(move || {
            let texts = vec![text_owned];
            classifier.extract_entities(&texts)
        })
        .await
        .map_err(|e| NarraError::Database(format!("Task join error: {}", e)))?
        .map_err(|e| NarraError::Database(format!("NER extraction error: {}", e)))?;

        let entities: Vec<NerEntity> = result
            .into_iter()
            .next()
            .unwrap_or_default()
            .into_iter()
            .map(|e| NerEntity {
                text: e.text,
                label: e.label,
                score: e.score,
                start: e.start,
                end: e.end,
            })
            .collect();

        let entity_count = entities.len();

        Ok(NerOutput {
            entities,
            entity_count,
        })
    }

    fn is_available(&self) -> bool {
        self.available
    }
}

// ============================================================================
// No-op service (for tests / graceful degradation)
// ============================================================================

/// No-op NER service for testing.
pub struct NoopNerService;

impl Default for NoopNerService {
    fn default() -> Self {
        Self::new()
    }
}

impl NoopNerService {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl NerService for NoopNerService {
    async fn get_entities(&self, _entity_id: &str, _text: &str) -> Result<NerOutput, NarraError> {
        Err(NarraError::Database(
            "NER service is not available (noop)".to_string(),
        ))
    }

    async fn extract_entities(&self, _text: &str) -> Result<NerOutput, NarraError> {
        Err(NarraError::Database(
            "NER service is not available (noop)".to_string(),
        ))
    }

    fn is_available(&self) -> bool {
        false
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noop_ner_service_is_not_available() {
        let service = NoopNerService::new();
        assert!(!service.is_available());
    }

    #[tokio::test]
    async fn test_noop_ner_service_get_entities_returns_error() {
        let service = NoopNerService::new();
        let result = service.get_entities("character:alice", "some text").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_noop_ner_service_extract_entities_returns_error() {
        let service = NoopNerService::new();
        let result = service.extract_entities("some text").await;
        assert!(result.is_err());
    }

    #[test]
    fn test_noop_ner_service_default() {
        let service = NoopNerService;
        assert!(!service.is_available());
    }

    #[test]
    fn test_ner_output_serialization_roundtrip() {
        let output = NerOutput {
            entities: vec![
                NerEntity {
                    text: "Alice".to_string(),
                    label: "PER".to_string(),
                    score: 0.98,
                    start: 0,
                    end: 5,
                },
                NerEntity {
                    text: "Manor".to_string(),
                    label: "LOC".to_string(),
                    score: 0.85,
                    start: 20,
                    end: 25,
                },
            ],
            entity_count: 2,
        };
        let json = serde_json::to_value(&output).expect("serialize");
        let roundtrip: NerOutput = serde_json::from_value(json).expect("deserialize");
        assert_eq!(roundtrip.entity_count, 2);
        assert_eq!(roundtrip.entities[0].text, "Alice");
        assert_eq!(roundtrip.entities[0].label, "PER");
        assert_eq!(roundtrip.entities[1].text, "Manor");
        assert_eq!(roundtrip.entities[1].label, "LOC");
    }

    // -- Property-based tests --

    mod prop_tests {
        use super::*;
        use proptest::prelude::*;

        fn arb_ner_output() -> impl Strategy<Value = NerOutput> {
            proptest::collection::vec(
                (
                    "[A-Z][a-z]{2,10}", // text
                    prop::sample::select(vec!["PER", "LOC", "ORG", "MISC"]),
                    0.0f32..=1.0f32, // score
                    0usize..1000,    // start
                    0usize..1000,    // end
                ),
                0..20,
            )
            .prop_map(|entities| {
                let ner_entities: Vec<NerEntity> = entities
                    .into_iter()
                    .map(|(text, label, score, start, end)| NerEntity {
                        text,
                        label: label.to_string(),
                        score,
                        start,
                        end: end.max(start),
                    })
                    .collect();
                let count = ner_entities.len();
                NerOutput {
                    entities: ner_entities,
                    entity_count: count,
                }
            })
        }

        proptest! {
            #[test]
            fn prop_entity_count_matches_len(output in arb_ner_output()) {
                prop_assert_eq!(output.entity_count, output.entities.len());
            }

            #[test]
            fn prop_serialization_roundtrip(output in arb_ner_output()) {
                let json = serde_json::to_value(&output).expect("serialize");
                let roundtrip: NerOutput = serde_json::from_value(json).expect("deserialize");
                prop_assert_eq!(roundtrip.entity_count, output.entity_count);
                prop_assert_eq!(roundtrip.entities.len(), output.entities.len());
                for (orig, rt) in output.entities.iter().zip(roundtrip.entities.iter()) {
                    prop_assert_eq!(&orig.text, &rt.text);
                    prop_assert_eq!(&orig.label, &rt.label);
                }
            }

            #[test]
            fn prop_end_gte_start(output in arb_ner_output()) {
                for e in &output.entities {
                    prop_assert!(e.end >= e.start, "end ({}) must be >= start ({})", e.end, e.start);
                }
            }

            #[test]
            fn prop_labels_are_known(output in arb_ner_output()) {
                let known = ["PER", "LOC", "ORG", "MISC"];
                for e in &output.entities {
                    prop_assert!(
                        known.contains(&e.label.as_str()),
                        "Unknown NER label: {}",
                        e.label
                    );
                }
            }
        }
    }

    #[test]
    fn test_ner_entity_fields() {
        let entity = NerEntity {
            text: "New York".to_string(),
            label: "LOC".to_string(),
            score: 0.92,
            start: 10,
            end: 18,
        };
        assert_eq!(entity.text, "New York");
        assert_eq!(entity.label, "LOC");
        assert!((entity.score - 0.92).abs() < 0.001);
        assert_eq!(entity.start, 10);
        assert_eq!(entity.end, 18);
    }
}
