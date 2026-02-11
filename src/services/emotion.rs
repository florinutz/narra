//! Emotion classification service using GoEmotions (RoBERTa-base, 28 labels).
//!
//! Classifies entity text into emotion labels with lazy annotation caching:
//! fresh annotations are returned from the database, stale/missing ones
//! trigger recomputation via the [`SequenceClassifier`].

use std::sync::Arc;

use async_trait::async_trait;
use tracing::{info, warn};

use crate::db::connection::NarraDb;
use crate::embedding::candle_backend::{download_model, select_device, SequenceClassifier};
use crate::models::annotation::{
    get_annotation, upsert_annotation, AnnotationCreate, EmotionOutput, EmotionScore,
};
use crate::NarraError;

/// Default sigmoid activation threshold for GoEmotions multi-label output.
const EMOTION_THRESHOLD: f32 = 0.3;

const EMOTION_MODEL_REPO: &str = "SamLowe/roberta-base-go_emotions";
const EMOTION_MODEL_VERSION: &str = "roberta-base-go_emotions-v1";
const EMOTION_MODEL_TYPE: &str = "emotion";

/// Service trait for emotion classification.
#[async_trait]
pub trait EmotionService: Send + Sync {
    /// Get emotions for an entity, using cached annotations when fresh.
    ///
    /// 1. Checks for a fresh annotation in the database
    /// 2. If stale or missing, classifies the provided text
    /// 3. Caches the result as an annotation
    async fn get_emotions(&self, entity_id: &str, text: &str) -> Result<EmotionOutput, NarraError>;

    /// Classify text without caching (one-off classification).
    async fn classify_text(&self, text: &str) -> Result<EmotionOutput, NarraError>;

    /// Whether the emotion model is loaded and available.
    fn is_available(&self) -> bool;
}

/// Local emotion service using candle GoEmotions model.
pub struct LocalEmotionService {
    classifier: Option<Arc<SequenceClassifier>>,
    db: Arc<NarraDb>,
    available: bool,
}

impl LocalEmotionService {
    /// Create a new local emotion service.
    ///
    /// Downloads and loads the GoEmotions model eagerly. If model loading fails,
    /// the service will be unavailable but won't error (graceful degradation).
    pub fn new(db: Arc<NarraDb>) -> Self {
        let files = match download_model(EMOTION_MODEL_REPO, None) {
            Ok(files) => files,
            Err(e) => {
                warn!(
                    "Failed to download emotion model: {}. Emotion classification will be unavailable.",
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

        match SequenceClassifier::new(&files, device) {
            Ok(classifier) => {
                info!(
                    "Emotion classifier loaded ({}, {} labels via candle)",
                    EMOTION_MODEL_REPO,
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
                    "Failed to load emotion model: {}. Emotion classification will be unavailable.",
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

/// Convert raw classifier output to typed EmotionOutput.
fn to_emotion_output(scores: Vec<(String, f32)>) -> EmotionOutput {
    let mut sorted: Vec<EmotionScore> = scores
        .into_iter()
        .map(|(label, score)| EmotionScore { label, score })
        .collect();
    sorted.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let dominant = sorted
        .first()
        .map(|s| s.label.clone())
        .unwrap_or_else(|| "neutral".to_string());
    let active_count = sorted
        .iter()
        .filter(|s| s.score >= EMOTION_THRESHOLD)
        .count();

    EmotionOutput {
        scores: sorted,
        dominant,
        active_count,
    }
}

#[async_trait]
impl EmotionService for LocalEmotionService {
    async fn get_emotions(&self, entity_id: &str, text: &str) -> Result<EmotionOutput, NarraError> {
        // 1. Check for fresh cached annotation
        if let Ok(Some(annotation)) = get_annotation(&self.db, entity_id, EMOTION_MODEL_TYPE).await
        {
            if !annotation.stale {
                // Parse cached output
                if let Ok(output) = serde_json::from_value::<EmotionOutput>(annotation.output) {
                    return Ok(output);
                }
                // Cached output is corrupt — fall through to recompute
                warn!("Corrupt emotion annotation for {}, recomputing", entity_id);
            }
        }

        // 2. Classify text
        let output = self.classify_text(text).await?;

        // 3. Cache as annotation
        let output_json = serde_json::to_value(&output).map_err(|e| {
            NarraError::Database(format!("Failed to serialize emotion output: {}", e))
        })?;

        if let Err(e) = upsert_annotation(
            &self.db,
            AnnotationCreate {
                entity_id: entity_id.to_string(),
                model_type: EMOTION_MODEL_TYPE.to_string(),
                model_version: EMOTION_MODEL_VERSION.to_string(),
                output: output_json,
            },
        )
        .await
        {
            warn!(
                "Failed to cache emotion annotation for {}: {}",
                entity_id, e
            );
        }

        Ok(output)
    }

    async fn classify_text(&self, text: &str) -> Result<EmotionOutput, NarraError> {
        let classifier = self
            .classifier
            .as_ref()
            .ok_or_else(|| NarraError::Database("Emotion model not loaded".to_string()))?
            .clone();

        let text_owned = text.to_string();

        let result = tokio::task::spawn_blocking(move || {
            let texts = vec![text_owned];
            let results = classifier.classify(&texts)?;
            Ok::<Vec<Vec<(String, f32)>>, anyhow::Error>(results)
        })
        .await
        .map_err(|e| NarraError::Database(format!("Task join error: {}", e)))?
        .map_err(|e| NarraError::Database(format!("Emotion classification error: {}", e)))?;

        let scores = result
            .into_iter()
            .next()
            .ok_or_else(|| NarraError::Database("Empty classification result".to_string()))?;

        Ok(to_emotion_output(scores))
    }

    fn is_available(&self) -> bool {
        self.available
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_emotion_output_sorts_descending() {
        let scores = vec![
            ("sadness".to_string(), 0.1),
            ("joy".to_string(), 0.9),
            ("anger".to_string(), 0.5),
        ];
        let output = to_emotion_output(scores);
        assert_eq!(output.scores[0].label, "joy");
        assert_eq!(output.scores[1].label, "anger");
        assert_eq!(output.scores[2].label, "sadness");
    }

    #[test]
    fn test_to_emotion_output_dominant_is_highest() {
        let scores = vec![
            ("fear".to_string(), 0.8),
            ("joy".to_string(), 0.2),
        ];
        let output = to_emotion_output(scores);
        assert_eq!(output.dominant, "fear");
    }

    #[test]
    fn test_to_emotion_output_active_count_threshold() {
        let scores = vec![
            ("joy".to_string(), 0.9),
            ("anger".to_string(), 0.3),    // exactly at threshold — should be active
            ("sadness".to_string(), 0.29),  // below threshold — not active
            ("fear".to_string(), 0.01),
        ];
        let output = to_emotion_output(scores);
        assert_eq!(output.active_count, 2);
    }

    #[test]
    fn test_to_emotion_output_empty_input() {
        let output = to_emotion_output(vec![]);
        assert!(output.scores.is_empty());
        assert_eq!(output.dominant, "neutral");
        assert_eq!(output.active_count, 0);
    }

    #[test]
    fn test_to_emotion_output_all_below_threshold() {
        let scores = vec![
            ("joy".to_string(), 0.1),
            ("sadness".to_string(), 0.05),
        ];
        let output = to_emotion_output(scores);
        assert_eq!(output.active_count, 0);
        assert_eq!(output.dominant, "joy"); // still the highest, even if below threshold
    }

    #[test]
    fn test_to_emotion_output_all_above_threshold() {
        let scores = vec![
            ("joy".to_string(), 0.9),
            ("love".to_string(), 0.7),
            ("optimism".to_string(), 0.4),
        ];
        let output = to_emotion_output(scores);
        assert_eq!(output.active_count, 3);
    }

    #[test]
    fn test_to_emotion_output_nan_handling() {
        let scores = vec![
            ("joy".to_string(), f32::NAN),
            ("sadness".to_string(), 0.5),
        ];
        // Should not panic — NaN treated as Equal in sort
        let output = to_emotion_output(scores);
        assert_eq!(output.scores.len(), 2);
    }

    #[test]
    fn test_noop_emotion_service_is_not_available() {
        let service = NoopEmotionService::new();
        assert!(!service.is_available());
    }

    #[tokio::test]
    async fn test_noop_emotion_service_get_emotions_returns_error() {
        let service = NoopEmotionService::new();
        let result = service.get_emotions("character:alice", "some text").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_noop_emotion_service_classify_text_returns_error() {
        let service = NoopEmotionService::new();
        let result = service.classify_text("some text").await;
        assert!(result.is_err());
    }

    #[test]
    fn test_noop_emotion_service_default() {
        let service = NoopEmotionService::default();
        assert!(!service.is_available());
    }

    #[test]
    fn test_emotion_output_serialization_roundtrip() {
        let output = EmotionOutput {
            scores: vec![
                EmotionScore { label: "joy".to_string(), score: 0.9 },
                EmotionScore { label: "anger".to_string(), score: 0.3 },
            ],
            dominant: "joy".to_string(),
            active_count: 2,
        };
        let json = serde_json::to_value(&output).expect("serialize");
        let roundtrip: EmotionOutput = serde_json::from_value(json).expect("deserialize");
        assert_eq!(roundtrip.dominant, "joy");
        assert_eq!(roundtrip.scores.len(), 2);
        assert_eq!(roundtrip.active_count, 2);
    }
}

/// No-op emotion service for testing.
pub struct NoopEmotionService;

impl Default for NoopEmotionService {
    fn default() -> Self {
        Self::new()
    }
}

impl NoopEmotionService {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl EmotionService for NoopEmotionService {
    async fn get_emotions(
        &self,
        _entity_id: &str,
        _text: &str,
    ) -> Result<EmotionOutput, NarraError> {
        Err(NarraError::Database(
            "Emotion service is not available (noop)".to_string(),
        ))
    }

    async fn classify_text(&self, _text: &str) -> Result<EmotionOutput, NarraError> {
        Err(NarraError::Database(
            "Emotion service is not available (noop)".to_string(),
        ))
    }

    fn is_available(&self) -> bool {
        false
    }
}
