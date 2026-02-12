//! Streaming annotation pipeline for batch entity processing.
//!
//! Streams entities through emotion → theme → NER classifiers in parallel,
//! using `tokio-stream` for backpressure and `async-stream` for ergonomic
//! stream construction. Designed for throughput when annotating many entities.

use std::sync::Arc;

use futures::StreamExt;
use serde::Serialize;

use crate::db::connection::NarraDb;
use crate::models::annotation::{EmotionOutput, NerOutput, ThemeOutput};
use crate::services::progress::ProgressReporter;
use crate::services::{EmotionService, NerService, ThemeService};
use crate::utils::sanitize::validate_table;
use crate::NarraError;

/// Result of annotating a single entity.
#[derive(Debug, Clone, Serialize)]
pub struct AnnotationResult {
    pub entity_id: String,
    pub entity_name: String,
    pub emotions: Option<EmotionOutput>,
    pub themes: Option<ThemeOutput>,
    pub entities: Option<NerOutput>,
    pub errors: Vec<String>,
}

/// Summary of a batch annotation run.
#[derive(Debug, Clone, Serialize)]
pub struct BatchAnnotationReport {
    pub total_processed: usize,
    pub emotion_successes: usize,
    pub theme_successes: usize,
    pub ner_successes: usize,
    pub errors: usize,
    pub results: Vec<AnnotationResult>,
}

/// Configuration for which classifiers to run.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub run_emotions: bool,
    pub run_themes: bool,
    pub run_ner: bool,
    /// Max concurrency for parallel entity processing.
    pub concurrency: usize,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            run_emotions: true,
            run_themes: true,
            run_ner: true,
            concurrency: 4,
        }
    }
}

/// Entity with its text for annotation.
#[derive(Debug, Clone)]
struct EntityText {
    entity_id: String,
    entity_name: String,
    text: String,
}

/// Streaming annotation pipeline.
pub struct AnnotationPipeline {
    db: Arc<NarraDb>,
    emotion_service: Arc<dyn EmotionService + Send + Sync>,
    theme_service: Arc<dyn ThemeService + Send + Sync>,
    ner_service: Arc<dyn NerService + Send + Sync>,
}

impl AnnotationPipeline {
    pub fn new(
        db: Arc<NarraDb>,
        emotion_service: Arc<dyn EmotionService + Send + Sync>,
        theme_service: Arc<dyn ThemeService + Send + Sync>,
        ner_service: Arc<dyn NerService + Send + Sync>,
    ) -> Self {
        Self {
            db,
            emotion_service,
            theme_service,
            ner_service,
        }
    }

    /// Run the annotation pipeline on all entities of the given types.
    ///
    /// Returns a stream of `AnnotationResult` items as they complete,
    /// and a final `BatchAnnotationReport` summary.
    pub async fn annotate_all(
        &self,
        entity_types: &[&str],
        config: PipelineConfig,
        progress: Arc<dyn ProgressReporter>,
    ) -> Result<BatchAnnotationReport, NarraError> {
        // Fetch all entities with their composite text
        let entities = self.fetch_entity_texts(entity_types).await?;
        let total = entities.len();

        if total == 0 {
            return Ok(BatchAnnotationReport {
                total_processed: 0,
                emotion_successes: 0,
                theme_successes: 0,
                ner_successes: 0,
                errors: 0,
                results: vec![],
            });
        }

        // Stream entities through the pipeline with bounded concurrency
        let emotion_svc = self.emotion_service.clone();
        let theme_svc = self.theme_service.clone();
        let ner_svc = self.ner_service.clone();

        let run_emotions = config.run_emotions && emotion_svc.is_available();
        let run_themes = config.run_themes && theme_svc.is_available();
        let run_ner = config.run_ner && ner_svc.is_available();

        let stream = tokio_stream::iter(entities).map(move |entity| {
            let emotion_svc = emotion_svc.clone();
            let theme_svc = theme_svc.clone();
            let ner_svc = ner_svc.clone();

            async move {
                let mut result = AnnotationResult {
                    entity_id: entity.entity_id.clone(),
                    entity_name: entity.entity_name.clone(),
                    emotions: None,
                    themes: None,
                    entities: None,
                    errors: vec![],
                };

                // Run classifiers in parallel for each entity
                let (emotion_result, theme_result, ner_result) = tokio::join!(
                    async {
                        if run_emotions {
                            Some(
                                emotion_svc
                                    .get_emotions(&entity.entity_id, &entity.text)
                                    .await,
                            )
                        } else {
                            None
                        }
                    },
                    async {
                        if run_themes {
                            Some(
                                theme_svc
                                    .get_themes(&entity.entity_id, &entity.text, None)
                                    .await,
                            )
                        } else {
                            None
                        }
                    },
                    async {
                        if run_ner {
                            Some(ner_svc.get_entities(&entity.entity_id, &entity.text).await)
                        } else {
                            None
                        }
                    },
                );

                if let Some(Ok(emotions)) = emotion_result {
                    result.emotions = Some(emotions);
                } else if let Some(Err(e)) = emotion_result {
                    result.errors.push(format!("emotion: {}", e));
                }

                if let Some(Ok(themes)) = theme_result {
                    result.themes = Some(themes);
                } else if let Some(Err(e)) = theme_result {
                    result.errors.push(format!("theme: {}", e));
                }

                if let Some(Ok(entities)) = ner_result {
                    result.entities = Some(entities);
                } else if let Some(Err(e)) = ner_result {
                    result.errors.push(format!("ner: {}", e));
                }

                result
            }
        });

        // Buffer unordered for concurrency
        let mut buffered = stream.buffer_unordered(config.concurrency);

        let mut results = Vec::with_capacity(total);
        let mut emotion_ok = 0usize;
        let mut theme_ok = 0usize;
        let mut ner_ok = 0usize;
        let mut error_count = 0usize;
        let mut processed = 0usize;

        progress
            .report(0.0, 1.0, Some(format!("Annotating {} entities", total)))
            .await;

        while let Some(result) = buffered.next().await {
            if result.emotions.is_some() {
                emotion_ok += 1;
            }
            if result.themes.is_some() {
                theme_ok += 1;
            }
            if result.entities.is_some() {
                ner_ok += 1;
            }
            if !result.errors.is_empty() {
                error_count += 1;
            }
            processed += 1;
            progress
                .report(
                    processed as f64 / total as f64,
                    1.0,
                    Some(format!(
                        "Annotated {}/{}: {}",
                        processed, total, result.entity_name
                    )),
                )
                .await;
            results.push(result);
        }

        Ok(BatchAnnotationReport {
            total_processed: total,
            emotion_successes: emotion_ok,
            theme_successes: theme_ok,
            ner_successes: ner_ok,
            errors: error_count,
            results,
        })
    }

    /// Annotate a single entity — used for targeted re-annotation.
    pub async fn annotate_one(
        &self,
        entity_id: &str,
        entity_name: &str,
        text: &str,
        config: &PipelineConfig,
    ) -> AnnotationResult {
        let mut result = AnnotationResult {
            entity_id: entity_id.to_string(),
            entity_name: entity_name.to_string(),
            emotions: None,
            themes: None,
            entities: None,
            errors: vec![],
        };

        let run_emotions = config.run_emotions && self.emotion_service.is_available();
        let run_themes = config.run_themes && self.theme_service.is_available();
        let run_ner = config.run_ner && self.ner_service.is_available();

        let (emotion_result, theme_result, ner_result) = tokio::join!(
            async {
                if run_emotions {
                    Some(self.emotion_service.get_emotions(entity_id, text).await)
                } else {
                    None
                }
            },
            async {
                if run_themes {
                    Some(self.theme_service.get_themes(entity_id, text, None).await)
                } else {
                    None
                }
            },
            async {
                if run_ner {
                    Some(self.ner_service.get_entities(entity_id, text).await)
                } else {
                    None
                }
            },
        );

        if let Some(Ok(emotions)) = emotion_result {
            result.emotions = Some(emotions);
        } else if let Some(Err(e)) = emotion_result {
            result.errors.push(format!("emotion: {}", e));
        }

        if let Some(Ok(themes)) = theme_result {
            result.themes = Some(themes);
        } else if let Some(Err(e)) = theme_result {
            result.errors.push(format!("theme: {}", e));
        }

        if let Some(Ok(entities)) = ner_result {
            result.entities = Some(entities);
        } else if let Some(Err(e)) = ner_result {
            result.errors.push(format!("ner: {}", e));
        }

        result
    }

    /// Fetch composite text for entities of the given types.
    async fn fetch_entity_texts(
        &self,
        entity_types: &[&str],
    ) -> Result<Vec<EntityText>, NarraError> {
        let mut all_entities = Vec::new();

        for entity_type in entity_types {
            #[derive(serde::Deserialize)]
            struct Row {
                id: surrealdb::RecordId,
                composite_text: Option<String>,
                name: Option<String>,
                title: Option<String>,
                description: Option<String>,
            }

            let safe_table = validate_table(entity_type)?;
            let query = format!(
                "SELECT id, composite_text, name, title, description FROM {}",
                safe_table
            );

            let mut resp = self.db.query(&query).await?;
            let rows: Vec<Row> = resp.take(0)?;

            for row in rows {
                let text = row
                    .composite_text
                    .or(row.description.clone())
                    .or(row.name.clone())
                    .or(row.title.clone());

                if let Some(text) = text {
                    let display = row
                        .name
                        .or(row.title)
                        .or(row.description)
                        .unwrap_or_else(|| row.id.to_string());

                    all_entities.push(EntityText {
                        entity_id: row.id.to_string(),
                        entity_name: display,
                        text,
                    });
                }
            }
        }

        Ok(all_entities)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_config_default() {
        let config = PipelineConfig::default();
        assert!(config.run_emotions);
        assert!(config.run_themes);
        assert!(config.run_ner);
        assert_eq!(config.concurrency, 4);
    }

    #[test]
    fn test_batch_report_empty() {
        let report = BatchAnnotationReport {
            total_processed: 0,
            emotion_successes: 0,
            theme_successes: 0,
            ner_successes: 0,
            errors: 0,
            results: vec![],
        };
        assert_eq!(report.total_processed, 0);
        assert!(report.results.is_empty());
    }

    #[test]
    fn test_annotation_result_no_errors() {
        let result = AnnotationResult {
            entity_id: "character:alice".to_string(),
            entity_name: "Alice".to_string(),
            emotions: None,
            themes: None,
            entities: None,
            errors: vec![],
        };
        assert!(result.errors.is_empty());
    }

    // -- Property-based tests --

    mod prop_tests {
        use super::*;
        use proptest::prelude::*;

        fn arb_annotation_result() -> impl Strategy<Value = AnnotationResult> {
            (
                "[a-z]+:[a-z]+",
                "[A-Z][a-z]{2,10}",
                proptest::option::of(Just(EmotionOutput {
                    scores: vec![],
                    dominant: "neutral".to_string(),
                    active_count: 0,
                })),
                proptest::option::of(Just(ThemeOutput {
                    themes: vec![],
                    dominant: "none".to_string(),
                    active_count: 0,
                })),
                proptest::option::of(Just(NerOutput {
                    entities: vec![],
                    entity_count: 0,
                })),
                proptest::collection::vec("error: [a-z ]{5,20}", 0..3),
            )
                .prop_map(|(id, name, emotions, themes, entities, errors)| {
                    AnnotationResult {
                        entity_id: id,
                        entity_name: name,
                        emotions,
                        themes,
                        entities,
                        errors,
                    }
                })
        }

        fn arb_batch_report() -> impl Strategy<Value = BatchAnnotationReport> {
            proptest::collection::vec(arb_annotation_result(), 0..20).prop_map(|results| {
                let total = results.len();
                let emotion_ok = results.iter().filter(|r| r.emotions.is_some()).count();
                let theme_ok = results.iter().filter(|r| r.themes.is_some()).count();
                let ner_ok = results.iter().filter(|r| r.entities.is_some()).count();
                let errors = results.iter().filter(|r| !r.errors.is_empty()).count();
                BatchAnnotationReport {
                    total_processed: total,
                    emotion_successes: emotion_ok,
                    theme_successes: theme_ok,
                    ner_successes: ner_ok,
                    errors,
                    results,
                }
            })
        }

        proptest! {
            #[test]
            fn prop_report_counts_consistent(report in arb_batch_report()) {
                prop_assert!(
                    report.emotion_successes <= report.total_processed,
                    "Emotion successes ({}) can't exceed total ({})",
                    report.emotion_successes, report.total_processed
                );
                prop_assert!(
                    report.theme_successes <= report.total_processed,
                    "Theme successes ({}) can't exceed total ({})",
                    report.theme_successes, report.total_processed
                );
                prop_assert!(
                    report.ner_successes <= report.total_processed,
                    "NER successes ({}) can't exceed total ({})",
                    report.ner_successes, report.total_processed
                );
                prop_assert!(
                    report.errors <= report.total_processed,
                    "Errors ({}) can't exceed total ({})",
                    report.errors, report.total_processed
                );
            }

            #[test]
            fn prop_report_results_len_matches_total(report in arb_batch_report()) {
                prop_assert_eq!(
                    report.results.len(),
                    report.total_processed,
                    "Results length should match total_processed"
                );
            }

            #[test]
            fn prop_report_serializes(report in arb_batch_report()) {
                let json = serde_json::to_value(&report);
                prop_assert!(json.is_ok(), "Report should serialize to JSON");
            }
        }
    }
}
