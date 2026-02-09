//! Arc tracking service for character evolution analysis.
//!
//! Measures how entities evolve over time through embedding snapshots,
//! compares trajectories between entities, and captures point-in-time moments.

use async_trait::async_trait;
use serde::Serialize;
use std::sync::Arc;
use surrealdb::{engine::local::Db, Surreal};

use crate::utils::math::{cosine_similarity, vector_subtract};
use crate::NarraError;

// ---------------------------------------------------------------------------
// Result types (presentation-agnostic)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct ArcSnapshotEntry {
    pub delta: Option<f32>,
    pub cumulative: f32,
    pub event: Option<String>,
    pub timestamp: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ArcHistoryResult {
    pub entity_id: String,
    pub total_snapshots: usize,
    pub net_displacement: f32,
    pub cumulative_drift: f32,
    pub assessment: String,
    pub snapshots: Vec<ArcSnapshotEntry>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ArcComparisonResult {
    pub entity_a: String,
    pub entity_b: String,
    pub initial_similarity: f32,
    pub current_similarity: f32,
    pub convergence_delta: f32,
    pub convergence: String,
    pub trajectory_similarity: f32,
    pub trajectory: String,
    pub snapshots_a: usize,
    pub snapshots_b: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct ArcMomentResult {
    pub entity_type: String,
    pub delta_magnitude: Option<f32>,
    pub event_title: Option<String>,
    pub created_at: String,
}

// ---------------------------------------------------------------------------
// Intermediate data types (from DB)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ArcSnapshotData {
    pub embedding: Vec<f32>,
    pub delta_magnitude: Option<f32>,
    pub event_title: Option<String>,
    pub created_at: String,
    pub entity_type: Option<String>,
}

#[derive(Debug, Clone)]
pub struct SnapshotEmbeddingData {
    pub embedding: Vec<f32>,
}

// ---------------------------------------------------------------------------
// Pure functions
// ---------------------------------------------------------------------------

/// Qualitative arc displacement assessment.
pub fn arc_assessment(displacement: f32) -> &'static str {
    if displacement < 0.02 {
        "essentially unchanged"
    } else if displacement < 0.1 {
        "minor evolution"
    } else if displacement < 0.3 {
        "significant evolution"
    } else {
        "dramatic transformation"
    }
}

/// Convergence assessment from similarity delta.
pub fn convergence_assessment(delta: f32) -> &'static str {
    if delta.abs() < 0.02 {
        "stable (no significant convergence or divergence)"
    } else if delta > 0.0 {
        "converging (becoming more similar)"
    } else {
        "diverging (becoming less similar)"
    }
}

/// Trajectory assessment from trajectory cosine similarity.
pub fn trajectory_assessment(similarity: f32) -> &'static str {
    if similarity > 0.5 {
        "similar trajectories (evolving in the same direction)"
    } else if similarity < -0.3 {
        "opposite trajectories (evolving in opposing directions)"
    } else {
        "independent trajectories (evolving in unrelated directions)"
    }
}

// ---------------------------------------------------------------------------
// Data provider trait
// ---------------------------------------------------------------------------

#[async_trait]
pub trait ArcDataProvider: Send + Sync {
    /// Fetch arc snapshots for an entity, ordered by created_at ASC.
    async fn fetch_arc_snapshots(
        &self,
        entity_id: &str,
        limit: usize,
    ) -> Result<Vec<ArcSnapshotData>, NarraError>;

    /// Fetch snapshot embeddings for an entity in given order, with optional limit.
    async fn fetch_snapshot_embeddings(
        &self,
        entity_id: &str,
        order: &str,
        limit: Option<usize>,
    ) -> Result<Vec<SnapshotEmbeddingData>, NarraError>;

    /// Fetch the timestamp of an event.
    async fn fetch_event_timestamp(
        &self,
        event_id: &str,
    ) -> Result<Option<surrealdb::Datetime>, NarraError>;

    /// Fetch the nearest arc snapshot for an entity before a given time.
    async fn fetch_moment_snapshot(
        &self,
        entity_id: &str,
        before: Option<&surrealdb::Datetime>,
    ) -> Result<Option<ArcSnapshotData>, NarraError>;
}

// ---------------------------------------------------------------------------
// SurrealDB implementation
// ---------------------------------------------------------------------------

pub struct SurrealArcDataProvider {
    db: Arc<Surreal<Db>>,
}

impl SurrealArcDataProvider {
    pub fn new(db: Arc<Surreal<Db>>) -> Self {
        Self { db }
    }
}

#[async_trait]
impl ArcDataProvider for SurrealArcDataProvider {
    async fn fetch_arc_snapshots(
        &self,
        entity_id: &str,
        limit: usize,
    ) -> Result<Vec<ArcSnapshotData>, NarraError> {
        let query = format!(
            "SELECT *, event_id.title AS event_title FROM arc_snapshot \
             WHERE entity_id = {} ORDER BY created_at ASC LIMIT {}",
            entity_id, limit
        );
        let mut resp = self.db.query(&query).await?;

        #[derive(serde::Deserialize)]
        struct Row {
            embedding: Vec<f32>,
            delta_magnitude: Option<f32>,
            event_title: Option<String>,
            created_at: String,
            entity_type: Option<String>,
        }

        let rows: Vec<Row> = resp.take(0)?;
        Ok(rows
            .into_iter()
            .map(|r| ArcSnapshotData {
                embedding: r.embedding,
                delta_magnitude: r.delta_magnitude,
                event_title: r.event_title,
                created_at: r.created_at,
                entity_type: r.entity_type,
            })
            .collect())
    }

    async fn fetch_snapshot_embeddings(
        &self,
        entity_id: &str,
        order: &str,
        limit: Option<usize>,
    ) -> Result<Vec<SnapshotEmbeddingData>, NarraError> {
        let limit_clause = limit.map(|n| format!(" LIMIT {}", n)).unwrap_or_default();
        let query = format!(
            "SELECT embedding, created_at FROM arc_snapshot WHERE entity_id = {} ORDER BY created_at {}{}",
            entity_id, order, limit_clause
        );
        let mut resp = self.db.query(&query).await?;

        #[derive(serde::Deserialize)]
        struct Row {
            embedding: Vec<f32>,
        }

        let rows: Vec<Row> = resp.take(0)?;
        Ok(rows
            .into_iter()
            .map(|r| SnapshotEmbeddingData {
                embedding: r.embedding,
            })
            .collect())
    }

    async fn fetch_event_timestamp(
        &self,
        event_id: &str,
    ) -> Result<Option<surrealdb::Datetime>, NarraError> {
        let event_ref = if event_id.starts_with("event:") {
            event_id.to_string()
        } else {
            format!("event:{}", event_id)
        };
        let query = format!("SELECT VALUE created_at FROM {}", event_ref);
        let mut resp = self.db.query(&query).await?;
        let timestamps: Vec<surrealdb::Datetime> = resp.take(0).unwrap_or_default();
        Ok(timestamps.into_iter().next())
    }

    async fn fetch_moment_snapshot(
        &self,
        entity_id: &str,
        before: Option<&surrealdb::Datetime>,
    ) -> Result<Option<ArcSnapshotData>, NarraError> {
        let mut resp = if let Some(event_time) = before {
            let query = format!(
                "SELECT *, event_id.title AS event_title FROM arc_snapshot \
                 WHERE entity_id = {} AND created_at <= $event_time \
                 ORDER BY created_at DESC LIMIT 1",
                entity_id
            );
            self.db
                .query(&query)
                .bind(("event_time", event_time.clone()))
                .await?
        } else {
            let query = format!(
                "SELECT *, event_id.title AS event_title FROM arc_snapshot \
                 WHERE entity_id = {} ORDER BY created_at DESC LIMIT 1",
                entity_id
            );
            self.db.query(&query).await?
        };

        #[derive(serde::Deserialize)]
        struct Row {
            entity_type: String,
            delta_magnitude: Option<f32>,
            event_title: Option<String>,
            created_at: String,
        }

        let rows: Vec<Row> = resp.take(0)?;
        Ok(rows.into_iter().next().map(|r| ArcSnapshotData {
            embedding: vec![], // Not needed for moment result
            delta_magnitude: r.delta_magnitude,
            event_title: r.event_title,
            created_at: r.created_at,
            entity_type: Some(r.entity_type),
        }))
    }
}

// ---------------------------------------------------------------------------
// ArcService
// ---------------------------------------------------------------------------

pub struct ArcService {
    data: Arc<dyn ArcDataProvider>,
}

impl ArcService {
    pub fn new(db: Arc<Surreal<Db>>) -> Self {
        Self {
            data: Arc::new(SurrealArcDataProvider::new(db)),
        }
    }

    pub fn with_provider(data: Arc<dyn ArcDataProvider>) -> Self {
        Self { data }
    }

    /// Analyze arc history for an entity.
    pub async fn analyze_history(
        &self,
        entity_id: &str,
        limit: usize,
    ) -> Result<ArcHistoryResult, NarraError> {
        let snapshots = self.data.fetch_arc_snapshots(entity_id, limit).await?;

        if snapshots.is_empty() {
            return Ok(ArcHistoryResult {
                entity_id: entity_id.to_string(),
                total_snapshots: 0,
                net_displacement: 0.0,
                cumulative_drift: 0.0,
                assessment: "no data".to_string(),
                snapshots: vec![],
            });
        }

        let first_embedding = &snapshots[0].embedding;
        let last_embedding = &snapshots[snapshots.len() - 1].embedding;
        let net_displacement = 1.0 - cosine_similarity(first_embedding, last_embedding);

        let mut cumulative_drift = 0.0f32;
        let entries: Vec<ArcSnapshotEntry> = snapshots
            .iter()
            .map(|s| {
                if let Some(d) = s.delta_magnitude {
                    cumulative_drift += d;
                }
                ArcSnapshotEntry {
                    delta: s.delta_magnitude,
                    cumulative: cumulative_drift,
                    event: s.event_title.clone(),
                    timestamp: s.created_at.clone(),
                }
            })
            .collect();

        Ok(ArcHistoryResult {
            entity_id: entity_id.to_string(),
            total_snapshots: entries.len(),
            net_displacement,
            cumulative_drift,
            assessment: arc_assessment(net_displacement).to_string(),
            snapshots: entries,
        })
    }

    /// Compare arc trajectories between two entities.
    pub async fn analyze_comparison(
        &self,
        entity_a: &str,
        entity_b: &str,
        window: Option<&str>,
    ) -> Result<ArcComparisonResult, NarraError> {
        // Parse window parameter
        let limit = if let Some(w) = window {
            if let Some(n_str) = w.strip_prefix("recent:") {
                let n: usize = n_str.parse().map_err(|_| {
                    NarraError::Validation(format!(
                        "Invalid window format: '{}'. Use 'recent:N'",
                        w
                    ))
                })?;
                Some(n)
            } else {
                return Err(NarraError::Validation(format!(
                    "Invalid window format: '{}'. Use 'recent:N'",
                    w
                )));
            }
        } else {
            None
        };

        let order = if window.is_some() { "DESC" } else { "ASC" };

        let mut snaps_a = self
            .data
            .fetch_snapshot_embeddings(entity_a, order, limit)
            .await?;
        let mut snaps_b = self
            .data
            .fetch_snapshot_embeddings(entity_b, order, limit)
            .await?;

        if snaps_a.is_empty() || snaps_b.is_empty() {
            let missing = if snaps_a.is_empty() {
                entity_a
            } else {
                entity_b
            };
            return Err(NarraError::NotFound {
                entity_type: "arc_snapshots".to_string(),
                id: missing.to_string(),
            });
        }

        // Reverse to chronological if fetched DESC
        if window.is_some() {
            snaps_a.reverse();
            snaps_b.reverse();
        }

        let first_sim = cosine_similarity(&snaps_a[0].embedding, &snaps_b[0].embedding);
        let latest_sim = cosine_similarity(
            &snaps_a[snaps_a.len() - 1].embedding,
            &snaps_b[snaps_b.len() - 1].embedding,
        );
        let convergence_delta = latest_sim - first_sim;

        let delta_a = vector_subtract(&snaps_a[snaps_a.len() - 1].embedding, &snaps_a[0].embedding);
        let delta_b = vector_subtract(&snaps_b[snaps_b.len() - 1].embedding, &snaps_b[0].embedding);
        let trajectory_sim = cosine_similarity(&delta_a, &delta_b);

        Ok(ArcComparisonResult {
            entity_a: entity_a.to_string(),
            entity_b: entity_b.to_string(),
            initial_similarity: first_sim,
            current_similarity: latest_sim,
            convergence_delta,
            convergence: convergence_assessment(convergence_delta).to_string(),
            trajectory_similarity: trajectory_sim,
            trajectory: trajectory_assessment(trajectory_sim).to_string(),
            snapshots_a: snaps_a.len(),
            snapshots_b: snaps_b.len(),
        })
    }

    /// Get a point-in-time arc snapshot.
    pub async fn analyze_moment(
        &self,
        entity_id: &str,
        event_id: Option<&str>,
    ) -> Result<ArcMomentResult, NarraError> {
        let event_time = if let Some(eid) = event_id {
            let ts = self.data.fetch_event_timestamp(eid).await?;
            if ts.is_none() {
                return Err(NarraError::NotFound {
                    entity_type: "event".to_string(),
                    id: eid.to_string(),
                });
            }
            ts
        } else {
            None
        };

        let snap = self
            .data
            .fetch_moment_snapshot(entity_id, event_time.as_ref())
            .await?
            .ok_or_else(|| {
                let context = if event_id.is_some() {
                    "at that event"
                } else {
                    "at all"
                };
                NarraError::NotFound {
                    entity_type: "arc_snapshot".to_string(),
                    id: format!("{} {}", entity_id, context),
                }
            })?;

        Ok(ArcMomentResult {
            entity_type: snap.entity_type.unwrap_or_else(|| "unknown".to_string()),
            delta_magnitude: snap.delta_magnitude,
            event_title: snap.event_title,
            created_at: snap.created_at,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arc_assessment_thresholds() {
        assert_eq!(arc_assessment(0.01), "essentially unchanged");
        assert_eq!(arc_assessment(0.02), "minor evolution");
        assert_eq!(arc_assessment(0.09), "minor evolution");
        assert_eq!(arc_assessment(0.1), "significant evolution");
        assert_eq!(arc_assessment(0.29), "significant evolution");
        assert_eq!(arc_assessment(0.3), "dramatic transformation");
        assert_eq!(arc_assessment(0.9), "dramatic transformation");
    }

    #[test]
    fn test_convergence_assessment() {
        assert_eq!(
            convergence_assessment(0.01),
            "stable (no significant convergence or divergence)"
        );
        assert_eq!(
            convergence_assessment(-0.01),
            "stable (no significant convergence or divergence)"
        );
        assert_eq!(
            convergence_assessment(0.1),
            "converging (becoming more similar)"
        );
        assert_eq!(
            convergence_assessment(-0.1),
            "diverging (becoming less similar)"
        );
    }

    #[test]
    fn test_trajectory_assessment() {
        assert_eq!(
            trajectory_assessment(0.7),
            "similar trajectories (evolving in the same direction)"
        );
        assert_eq!(
            trajectory_assessment(-0.5),
            "opposite trajectories (evolving in opposing directions)"
        );
        assert_eq!(
            trajectory_assessment(0.0),
            "independent trajectories (evolving in unrelated directions)"
        );
        assert_eq!(
            trajectory_assessment(0.3),
            "independent trajectories (evolving in unrelated directions)"
        );
    }
}
