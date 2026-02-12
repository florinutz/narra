//! Theme classification service using zero-shot NLI (cross-encoder/nli-roberta-base).
//!
//! Classifies entity text into narrative themes by constructing NLI hypotheses
//! ("This text is about {theme}") and using the entailment probability as the
//! relevance score. Uses lazy annotation caching: fresh annotations are returned
//! from the database, stale/missing ones trigger recomputation.

use std::sync::Arc;

use async_trait::async_trait;
use tracing::{info, warn};

use crate::db::connection::NarraDb;
use crate::embedding::candle_backend::{download_model, select_device, NliClassifier};
use crate::models::annotation::{
    get_annotation, upsert_annotation, AnnotationCreate, ThemeOutput, ThemeScore,
};
use crate::NarraError;

/// Default entailment threshold for considering a theme active.
const THEME_THRESHOLD: f32 = 0.5;

const THEME_MODEL_REPO: &str = "cross-encoder/nli-roberta-base";
const THEME_MODEL_VERSION: &str = "nli-roberta-base-v1";
const THEME_MODEL_TYPE: &str = "theme";

/// Default narrative themes for fiction analysis.
pub const DEFAULT_NARRATIVE_THEMES: &[&str] = &[
    "love",
    "betrayal",
    "revenge",
    "identity",
    "power",
    "loss",
    "redemption",
    "sacrifice",
    "justice",
    "freedom",
    "deception",
    "loyalty",
    "corruption",
    "growth",
    "family",
    "war",
    "survival",
    "ambition",
    "fate",
    "morality",
];

/// Service trait for theme classification.
#[async_trait]
pub trait ThemeService: Send + Sync {
    /// Get themes for an entity, using cached annotations when fresh.
    ///
    /// 1. Checks for a fresh annotation in the database
    /// 2. If stale or missing, classifies the provided text against candidate themes
    /// 3. Caches the result as an annotation
    async fn get_themes(
        &self,
        entity_id: &str,
        text: &str,
        themes: Option<&[String]>,
    ) -> Result<ThemeOutput, NarraError>;

    /// Classify text against themes without caching (one-off classification).
    async fn classify_themes(
        &self,
        text: &str,
        themes: Option<&[String]>,
    ) -> Result<ThemeOutput, NarraError>;

    /// Whether the NLI model is loaded and available.
    fn is_available(&self) -> bool;
}

/// Local theme service using candle NLI model.
pub struct LocalThemeService {
    classifier: Option<Arc<NliClassifier>>,
    db: Arc<NarraDb>,
    available: bool,
}

impl LocalThemeService {
    /// Create a new local theme service.
    ///
    /// Downloads and loads the NLI model eagerly. If model loading fails,
    /// the service will be unavailable but won't error (graceful degradation).
    pub fn new(db: Arc<NarraDb>) -> Self {
        let files = match download_model(THEME_MODEL_REPO, None) {
            Ok(files) => files,
            Err(e) => {
                warn!(
                    "Failed to download theme model: {}. Theme classification will be unavailable.",
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

        match NliClassifier::new(&files, device) {
            Ok(classifier) => {
                info!(
                    "Theme classifier loaded ({} via candle NLI)",
                    THEME_MODEL_REPO,
                );
                Self {
                    classifier: Some(Arc::new(classifier)),
                    db,
                    available: true,
                }
            }
            Err(e) => {
                warn!(
                    "Failed to load theme model: {}. Theme classification will be unavailable.",
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

/// Convert raw NLI entailment scores to typed ThemeOutput.
fn to_theme_output(scores: Vec<(String, f32)>) -> ThemeOutput {
    let mut sorted: Vec<ThemeScore> = scores
        .into_iter()
        .map(|(label, score)| ThemeScore { label, score })
        .collect();
    sorted.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let dominant = sorted
        .first()
        .map(|s| s.label.clone())
        .unwrap_or_else(|| "none".to_string());
    let active_count = sorted.iter().filter(|s| s.score >= THEME_THRESHOLD).count();

    ThemeOutput {
        themes: sorted,
        dominant,
        active_count,
    }
}

#[async_trait]
impl ThemeService for LocalThemeService {
    async fn get_themes(
        &self,
        entity_id: &str,
        text: &str,
        themes: Option<&[String]>,
    ) -> Result<ThemeOutput, NarraError> {
        // Only use cache if default themes (custom themes bypass cache)
        if themes.is_none() {
            if let Ok(Some(annotation)) =
                get_annotation(&self.db, entity_id, THEME_MODEL_TYPE).await
            {
                if !annotation.stale {
                    if let Ok(output) = serde_json::from_value::<ThemeOutput>(annotation.output) {
                        return Ok(output);
                    }
                    warn!("Corrupt theme annotation for {}, recomputing", entity_id);
                }
            }
        }

        // Classify text
        let output = self.classify_themes(text, themes).await?;

        // Cache as annotation (only for default themes)
        if themes.is_none() {
            let output_json = serde_json::to_value(&output).map_err(|e| {
                NarraError::Database(format!("Failed to serialize theme output: {}", e))
            })?;

            if let Err(e) = upsert_annotation(
                &self.db,
                AnnotationCreate {
                    entity_id: entity_id.to_string(),
                    model_type: THEME_MODEL_TYPE.to_string(),
                    model_version: THEME_MODEL_VERSION.to_string(),
                    output: output_json,
                },
            )
            .await
            {
                warn!("Failed to cache theme annotation for {}: {}", entity_id, e);
            }
        }

        Ok(output)
    }

    async fn classify_themes(
        &self,
        text: &str,
        themes: Option<&[String]>,
    ) -> Result<ThemeOutput, NarraError> {
        let classifier = self
            .classifier
            .as_ref()
            .ok_or_else(|| NarraError::Database("Theme model not loaded".to_string()))?
            .clone();

        // Build hypothesis pairs: (text, "This text is about {theme}")
        let candidate_labels: Vec<String> = match themes {
            Some(custom) => custom.to_vec(),
            None => DEFAULT_NARRATIVE_THEMES
                .iter()
                .map(|s| s.to_string())
                .collect(),
        };

        let text_owned = text.to_string();
        let labels = candidate_labels.clone();

        let result = tokio::task::spawn_blocking(move || {
            let pairs: Vec<(String, String)> = labels
                .iter()
                .map(|theme| (text_owned.clone(), format!("This text is about {}.", theme)))
                .collect();
            classifier.classify_pairs(&pairs)
        })
        .await
        .map_err(|e| NarraError::Database(format!("Task join error: {}", e)))?
        .map_err(|e| NarraError::Database(format!("Theme classification error: {}", e)))?;

        let scores: Vec<(String, f32)> = candidate_labels.into_iter().zip(result).collect();

        Ok(to_theme_output(scores))
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
    fn test_to_theme_output_sorts_descending() {
        let scores = vec![
            ("loyalty".to_string(), 0.2),
            ("betrayal".to_string(), 0.9),
            ("love".to_string(), 0.5),
        ];
        let output = to_theme_output(scores);
        assert_eq!(output.themes[0].label, "betrayal");
        assert_eq!(output.themes[1].label, "love");
        assert_eq!(output.themes[2].label, "loyalty");
    }

    #[test]
    fn test_to_theme_output_dominant_is_highest() {
        let scores = vec![("power".to_string(), 0.8), ("love".to_string(), 0.2)];
        let output = to_theme_output(scores);
        assert_eq!(output.dominant, "power");
    }

    #[test]
    fn test_to_theme_output_active_count_threshold() {
        let scores = vec![
            ("betrayal".to_string(), 0.9),
            ("love".to_string(), 0.5),   // exactly at threshold — active
            ("power".to_string(), 0.49), // below threshold — not active
            ("fate".to_string(), 0.01),
        ];
        let output = to_theme_output(scores);
        assert_eq!(output.active_count, 2);
    }

    #[test]
    fn test_to_theme_output_empty_input() {
        let output = to_theme_output(vec![]);
        assert!(output.themes.is_empty());
        assert_eq!(output.dominant, "none");
        assert_eq!(output.active_count, 0);
    }

    #[test]
    fn test_to_theme_output_all_below_threshold() {
        let scores = vec![("love".to_string(), 0.3), ("fate".to_string(), 0.1)];
        let output = to_theme_output(scores);
        assert_eq!(output.active_count, 0);
        assert_eq!(output.dominant, "love"); // still highest, even if below threshold
    }

    #[test]
    fn test_to_theme_output_all_above_threshold() {
        let scores = vec![
            ("betrayal".to_string(), 0.9),
            ("love".to_string(), 0.7),
            ("power".to_string(), 0.6),
        ];
        let output = to_theme_output(scores);
        assert_eq!(output.active_count, 3);
    }

    #[test]
    fn test_to_theme_output_nan_handling() {
        let scores = vec![("love".to_string(), f32::NAN), ("fate".to_string(), 0.5)];
        // Should not panic — NaN treated as Equal in sort
        let output = to_theme_output(scores);
        assert_eq!(output.themes.len(), 2);
    }

    #[test]
    fn test_noop_theme_service_is_not_available() {
        let service = NoopThemeService::new();
        assert!(!service.is_available());
    }

    #[tokio::test]
    async fn test_noop_theme_service_get_themes_returns_error() {
        let service = NoopThemeService::new();
        let result = service
            .get_themes("character:alice", "some text", None)
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_noop_theme_service_classify_themes_returns_error() {
        let service = NoopThemeService::new();
        let result = service.classify_themes("some text", None).await;
        assert!(result.is_err());
    }

    #[test]
    fn test_noop_theme_service_default() {
        let service = NoopThemeService::default();
        assert!(!service.is_available());
    }

    // -- Property-based tests --

    mod prop_tests {
        use super::*;
        use proptest::prelude::*;

        fn arb_scores() -> impl Strategy<Value = Vec<(String, f32)>> {
            proptest::collection::vec(("[a-z]{3,10}", 0.0f32..=1.0f32), 0..20)
        }

        proptest! {
            #[test]
            fn prop_output_sorted_descending(scores in arb_scores()) {
                let output = to_theme_output(scores);
                for window in output.themes.windows(2) {
                    prop_assert!(
                        window[0].score >= window[1].score
                            || window[0].score.is_nan()
                            || window[1].score.is_nan(),
                        "Scores must be sorted descending"
                    );
                }
            }

            #[test]
            fn prop_output_length_preserved(scores in arb_scores()) {
                let n = scores.len();
                let output = to_theme_output(scores);
                prop_assert_eq!(output.themes.len(), n);
            }

            #[test]
            fn prop_active_count_correct(scores in arb_scores()) {
                let output = to_theme_output(scores);
                let expected = output.themes.iter().filter(|s| s.score >= THEME_THRESHOLD).count();
                prop_assert_eq!(output.active_count, expected);
            }

            #[test]
            fn prop_dominant_is_first_or_none(scores in arb_scores()) {
                let output = to_theme_output(scores);
                if output.themes.is_empty() {
                    prop_assert_eq!(output.dominant, "none");
                } else {
                    prop_assert_eq!(output.dominant, output.themes[0].label.clone());
                }
            }

            #[test]
            fn prop_idempotent(scores in arb_scores()) {
                let output1 = to_theme_output(scores.clone());
                let re_input: Vec<(String, f32)> = output1.themes.iter()
                    .map(|s| (s.label.clone(), s.score))
                    .collect();
                let output2 = to_theme_output(re_input);
                prop_assert_eq!(output1.dominant, output2.dominant);
                prop_assert_eq!(output1.active_count, output2.active_count);
                prop_assert_eq!(output1.themes.len(), output2.themes.len());
            }
        }
    }

    #[test]
    fn test_theme_output_serialization_roundtrip() {
        let output = ThemeOutput {
            themes: vec![
                ThemeScore {
                    label: "betrayal".to_string(),
                    score: 0.9,
                },
                ThemeScore {
                    label: "love".to_string(),
                    score: 0.3,
                },
            ],
            dominant: "betrayal".to_string(),
            active_count: 1,
        };
        let json = serde_json::to_value(&output).expect("serialize");
        let roundtrip: ThemeOutput = serde_json::from_value(json).expect("deserialize");
        assert_eq!(roundtrip.dominant, "betrayal");
        assert_eq!(roundtrip.themes.len(), 2);
        assert_eq!(roundtrip.active_count, 1);
    }

    #[test]
    fn test_default_narrative_themes_not_empty() {
        assert!(!DEFAULT_NARRATIVE_THEMES.is_empty());
        assert!(DEFAULT_NARRATIVE_THEMES.len() >= 15);
    }
}

/// No-op theme service for testing.
pub struct NoopThemeService;

impl Default for NoopThemeService {
    fn default() -> Self {
        Self::new()
    }
}

impl NoopThemeService {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl ThemeService for NoopThemeService {
    async fn get_themes(
        &self,
        _entity_id: &str,
        _text: &str,
        _themes: Option<&[String]>,
    ) -> Result<ThemeOutput, NarraError> {
        Err(NarraError::Database(
            "Theme service is not available (noop)".to_string(),
        ))
    }

    async fn classify_themes(
        &self,
        _text: &str,
        _themes: Option<&[String]>,
    ) -> Result<ThemeOutput, NarraError> {
        Err(NarraError::Database(
            "Theme service is not available (noop)".to_string(),
        ))
    }

    fn is_available(&self) -> bool {
        false
    }
}
