//! Annotation model: generic ML model outputs cached per entity.
//!
//! One annotation per (entity_id, model_type) pair. The `output` field is
//! model-specific JSON — typed wrappers like [`EmotionOutput`] provide
//! convenience deserialization over the raw JSON.

use crate::db::connection::NarraDb;
use crate::NarraError;
use serde::{Deserialize, Serialize};
use surrealdb::{Datetime, RecordId};

/// A cached ML model annotation for an entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Annotation {
    pub id: RecordId,
    /// Entity ID in "table:key" format (e.g., "character:alice")
    pub entity_id: String,
    /// Model type identifier (e.g., "emotion")
    pub model_type: String,
    /// Model version string (e.g., "roberta-base-go_emotions-v1")
    pub model_version: String,
    /// Model-specific JSON output
    pub output: serde_json::Value,
    /// When this annotation was computed
    pub computed_at: Datetime,
    /// Whether the annotation needs recomputation
    pub stale: bool,
}

/// Data for creating or upserting an annotation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnotationCreate {
    pub entity_id: String,
    pub model_type: String,
    pub model_version: String,
    pub output: serde_json::Value,
}

/// Typed emotion classification output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionOutput {
    /// Emotion scores sorted descending by score
    pub scores: Vec<EmotionScore>,
    /// Label of the highest-scoring emotion
    pub dominant: String,
    /// Number of labels above the activation threshold
    pub active_count: usize,
}

/// A single emotion label and its score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionScore {
    pub label: String,
    pub score: f32,
}

/// Typed theme classification output (zero-shot NLI).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThemeOutput {
    /// Theme scores sorted descending by score
    pub themes: Vec<ThemeScore>,
    /// Label of the highest-scoring theme
    pub dominant: String,
    /// Number of themes above the activation threshold
    pub active_count: usize,
}

/// A single theme label and its entailment score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThemeScore {
    pub label: String,
    pub score: f32,
}

/// Typed NER extraction output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NerOutput {
    /// Extracted entities sorted by position
    pub entities: Vec<NerEntity>,
    /// Total number of entities found
    pub entity_count: usize,
}

/// A single named entity extracted from text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NerEntity {
    /// Entity text as it appears in the input
    pub text: String,
    /// Entity type: PER, LOC, ORG, MISC
    pub label: String,
    /// Confidence score (0..1, averaged across subword tokens)
    pub score: f32,
    /// Character start offset in input text
    pub start: usize,
    /// Character end offset in input text
    pub end: usize,
}

// ============================================================================
// Annotation CRUD Operations
// ============================================================================

/// Upsert an annotation by (entity_id, model_type).
///
/// Uses SurrealDB's compound unique index — if an annotation already exists
/// for this entity+model combination, it is replaced.
pub async fn upsert_annotation(
    db: &NarraDb,
    data: AnnotationCreate,
) -> Result<Annotation, NarraError> {
    let mut result = db
        .query(
            "DELETE FROM annotation WHERE entity_id = $entity_id AND model_type = $model_type; \
             CREATE annotation SET \
             entity_id = $entity_id, \
             model_type = $model_type, \
             model_version = $model_version, \
             output = $output, \
             computed_at = time::now(), \
             stale = false",
        )
        .bind(("entity_id", data.entity_id))
        .bind(("model_type", data.model_type))
        .bind(("model_version", data.model_version))
        .bind(("output", data.output))
        .await?;

    // The CREATE is statement index 1 (after the DELETE at index 0)
    let annotation: Option<Annotation> = result.take(1)?;
    annotation.ok_or_else(|| NarraError::Database("Failed to upsert annotation".into()))
}

/// Get a single annotation for an entity and model type.
pub async fn get_annotation(
    db: &NarraDb,
    entity_id: &str,
    model_type: &str,
) -> Result<Option<Annotation>, NarraError> {
    let mut result = db
        .query(
            "SELECT * FROM annotation \
             WHERE entity_id = $entity_id AND model_type = $model_type \
             LIMIT 1",
        )
        .bind(("entity_id", entity_id.to_string()))
        .bind(("model_type", model_type.to_string()))
        .await?;

    let annotations: Vec<Annotation> = result.take(0)?;
    Ok(annotations.into_iter().next())
}

/// Get all annotations for an entity.
pub async fn get_entity_annotations(
    db: &NarraDb,
    entity_id: &str,
) -> Result<Vec<Annotation>, NarraError> {
    let mut result = db
        .query("SELECT * FROM annotation WHERE entity_id = $entity_id")
        .bind(("entity_id", entity_id.to_string()))
        .await?;

    let annotations: Vec<Annotation> = result.take(0)?;
    Ok(annotations)
}

/// Mark all annotations for an entity as stale.
///
/// Returns the number of annotations marked stale.
pub async fn mark_annotations_stale(db: &NarraDb, entity_id: &str) -> Result<usize, NarraError> {
    let mut result = db
        .query(
            "UPDATE annotation SET stale = true \
             WHERE entity_id = $entity_id AND stale = false",
        )
        .bind(("entity_id", entity_id.to_string()))
        .await?;

    let updated: Vec<Annotation> = result.take(0)?;
    Ok(updated.len())
}

/// Get stale annotations, optionally filtered by model type.
pub async fn get_stale_annotations(
    db: &NarraDb,
    model_type: Option<&str>,
    limit: usize,
) -> Result<Vec<Annotation>, NarraError> {
    let mut result = if let Some(mt) = model_type {
        db.query(
            "SELECT * FROM annotation \
             WHERE stale = true AND model_type = $model_type \
             ORDER BY computed_at ASC \
             LIMIT $limit",
        )
        .bind(("model_type", mt.to_string()))
        .bind(("limit", limit as i64))
        .await?
    } else {
        db.query(
            "SELECT * FROM annotation \
             WHERE stale = true \
             ORDER BY computed_at ASC \
             LIMIT $limit",
        )
        .bind(("limit", limit as i64))
        .await?
    };

    let annotations: Vec<Annotation> = result.take(0)?;
    Ok(annotations)
}

/// Delete all annotations for an entity.
pub async fn delete_entity_annotations(db: &NarraDb, entity_id: &str) -> Result<usize, NarraError> {
    let mut result = db
        .query("DELETE FROM annotation WHERE entity_id = $entity_id RETURN BEFORE")
        .bind(("entity_id", entity_id.to_string()))
        .await?;

    let deleted: Vec<Annotation> = result.take(0).unwrap_or_default();
    Ok(deleted.len())
}
