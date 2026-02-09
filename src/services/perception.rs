//! Perception analysis service for measuring perspective gaps and shifts.
//!
//! Computes how accurately characters perceive each other by comparing
//! perspective embeddings against reality, tracking perceptual convergence
//! and divergence over time.

use async_trait::async_trait;
use serde::Serialize;
use std::sync::Arc;
use surrealdb::{engine::local::Db, Surreal};

use crate::utils::math::cosine_similarity;
use crate::NarraError;

// ---------------------------------------------------------------------------
// Result types (presentation-agnostic)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct PerceptionGapResult {
    pub observer_name: String,
    pub target_name: String,
    pub gap: f32,
    pub similarity: f32,
    pub assessment: String,
    pub perception: Option<String>,
    pub feelings: Option<String>,
    pub tension_level: Option<i32>,
    pub history_notes: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ObserverGapEntry {
    pub observer_name: String,
    pub gap: f32,
    pub assessment: String,
    pub embedding: Vec<f32>,
    pub agrees_with: Option<String>,
    pub disagrees_with: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PerceptionMatrixResult {
    pub target_name: String,
    pub has_real_embedding: bool,
    pub observers: Vec<ObserverGapEntry>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ShiftSnapshotEntry {
    pub delta: Option<f32>,
    pub gap: Option<f32>,
    pub event: Option<String>,
    pub timestamp: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct PerceptionShiftResult {
    pub observer_name: String,
    pub target_name: String,
    pub snapshots: Vec<ShiftSnapshotEntry>,
    pub trajectory: String,
}

// ---------------------------------------------------------------------------
// Intermediate data types (from DB)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct PerceivesEdgeData {
    pub embedding: Vec<f32>,
    pub observer_name: Option<String>,
    pub target_name: Option<String>,
    pub perception: Option<String>,
    pub feelings: Option<String>,
    pub tension_level: Option<i32>,
    pub history_notes: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ObserverPerspectiveData {
    pub embedding: Vec<f32>,
    pub observer_name: Option<String>,
}

#[derive(Debug, Clone)]
pub struct PerspectiveSnapshotData {
    pub embedding: Vec<f32>,
    pub delta_magnitude: Option<f32>,
    pub created_at: String,
    pub event_title: Option<String>,
}

#[derive(Debug, Clone)]
pub struct CharacterSnapshotData {
    pub embedding: Vec<f32>,
    pub created_at: String,
}

// ---------------------------------------------------------------------------
// Pure functions
// ---------------------------------------------------------------------------

/// Qualitative gap assessment.
pub fn gap_assessment(gap: f32) -> &'static str {
    if gap < 0.05 {
        "remarkably accurate"
    } else if gap < 0.15 {
        "fairly accurate"
    } else if gap < 0.30 {
        "notable blind spots"
    } else if gap < 0.50 {
        "significantly distorted"
    } else {
        "dramatically wrong"
    }
}

/// Compute gap trajectory label from first/last gaps.
pub fn trajectory_label(first_gap: f32, last_gap: f32) -> &'static str {
    let delta = last_gap - first_gap;
    if delta < -0.02 {
        "converging"
    } else if delta > 0.02 {
        "diverging"
    } else {
        "stable"
    }
}

// ---------------------------------------------------------------------------
// Data provider trait
// ---------------------------------------------------------------------------

#[async_trait]
pub trait PerceptionDataProvider: Send + Sync {
    async fn fetch_perceives_edge(
        &self,
        observer_id: &str,
        target_id: &str,
    ) -> Result<Option<PerceivesEdgeData>, NarraError>;

    async fn fetch_real_embedding(&self, entity_id: &str) -> Result<Option<Vec<f32>>, NarraError>;

    async fn fetch_observer_perspectives(
        &self,
        target_id: &str,
        limit: usize,
    ) -> Result<Vec<ObserverPerspectiveData>, NarraError>;

    async fn fetch_perceives_edge_id(
        &self,
        observer_id: &str,
        target_id: &str,
    ) -> Result<Option<String>, NarraError>;

    async fn fetch_perspective_snapshots(
        &self,
        perceives_id: &str,
    ) -> Result<Vec<PerspectiveSnapshotData>, NarraError>;

    async fn fetch_character_snapshots(
        &self,
        target_id: &str,
    ) -> Result<Vec<CharacterSnapshotData>, NarraError>;
}

// ---------------------------------------------------------------------------
// SurrealDB implementation
// ---------------------------------------------------------------------------

pub struct SurrealPerceptionDataProvider {
    db: Arc<Surreal<Db>>,
}

impl SurrealPerceptionDataProvider {
    pub fn new(db: Arc<Surreal<Db>>) -> Self {
        Self { db }
    }
}

#[async_trait]
impl PerceptionDataProvider for SurrealPerceptionDataProvider {
    async fn fetch_perceives_edge(
        &self,
        observer_id: &str,
        target_id: &str,
    ) -> Result<Option<PerceivesEdgeData>, NarraError> {
        let query = format!(
            "SELECT embedding, in.name AS observer_name, out.name AS target_name, \
             perception, feelings, tension_level, history_notes \
             FROM perceives \
             WHERE in = {} AND out = {} AND embedding IS NOT NONE",
            observer_id, target_id
        );

        let mut resp = self.db.query(&query).await?;

        #[derive(serde::Deserialize)]
        struct Row {
            embedding: Vec<f32>,
            observer_name: Option<String>,
            target_name: Option<String>,
            perception: Option<String>,
            feelings: Option<String>,
            tension_level: Option<i32>,
            history_notes: Option<String>,
        }

        let rows: Vec<Row> = resp.take(0)?;
        Ok(rows.into_iter().next().map(|r| PerceivesEdgeData {
            embedding: r.embedding,
            observer_name: r.observer_name,
            target_name: r.target_name,
            perception: r.perception,
            feelings: r.feelings,
            tension_level: r.tension_level,
            history_notes: r.history_notes,
        }))
    }

    async fn fetch_real_embedding(&self, entity_id: &str) -> Result<Option<Vec<f32>>, NarraError> {
        let query = format!("SELECT VALUE embedding FROM {}", entity_id);
        let mut resp = self.db.query(&query).await?;
        let embeddings: Vec<Option<Vec<f32>>> = resp.take(0).unwrap_or_default();
        Ok(embeddings.into_iter().next().flatten())
    }

    async fn fetch_observer_perspectives(
        &self,
        target_id: &str,
        limit: usize,
    ) -> Result<Vec<ObserverPerspectiveData>, NarraError> {
        let query = format!(
            "SELECT embedding, in.name AS observer_name \
             FROM perceives WHERE out = {} AND embedding IS NOT NONE LIMIT {}",
            target_id, limit
        );

        let mut resp = self.db.query(&query).await?;

        #[derive(serde::Deserialize)]
        struct Row {
            embedding: Vec<f32>,
            observer_name: Option<String>,
        }

        let rows: Vec<Row> = resp.take(0)?;
        Ok(rows
            .into_iter()
            .map(|r| ObserverPerspectiveData {
                embedding: r.embedding,
                observer_name: r.observer_name,
            })
            .collect())
    }

    async fn fetch_perceives_edge_id(
        &self,
        observer_id: &str,
        target_id: &str,
    ) -> Result<Option<String>, NarraError> {
        let query = format!(
            "SELECT id FROM perceives WHERE in = {} AND out = {}",
            observer_id, target_id
        );
        let mut resp = self.db.query(&query).await?;

        #[derive(serde::Deserialize)]
        struct Row {
            id: surrealdb::sql::Thing,
        }

        let rows: Vec<Row> = resp.take(0)?;
        Ok(rows.into_iter().next().map(|r| r.id.to_string()))
    }

    async fn fetch_perspective_snapshots(
        &self,
        perceives_id: &str,
    ) -> Result<Vec<PerspectiveSnapshotData>, NarraError> {
        let query = format!(
            "SELECT embedding, delta_magnitude, created_at, event_id.title AS event_title \
             FROM arc_snapshot \
             WHERE entity_id = {} AND entity_type = 'perspective' \
             ORDER BY created_at ASC",
            perceives_id
        );
        let mut resp = self.db.query(&query).await?;

        #[derive(serde::Deserialize)]
        struct Row {
            embedding: Vec<f32>,
            delta_magnitude: Option<f32>,
            created_at: String,
            event_title: Option<String>,
        }

        let rows: Vec<Row> = resp.take(0)?;
        Ok(rows
            .into_iter()
            .map(|r| PerspectiveSnapshotData {
                embedding: r.embedding,
                delta_magnitude: r.delta_magnitude,
                created_at: r.created_at,
                event_title: r.event_title,
            })
            .collect())
    }

    async fn fetch_character_snapshots(
        &self,
        target_id: &str,
    ) -> Result<Vec<CharacterSnapshotData>, NarraError> {
        let query = format!(
            "SELECT embedding, created_at FROM arc_snapshot \
             WHERE entity_id = {} AND entity_type = 'character' \
             ORDER BY created_at ASC",
            target_id
        );
        let mut resp = self.db.query(&query).await?;

        #[derive(serde::Deserialize)]
        struct Row {
            embedding: Vec<f32>,
            created_at: String,
        }

        let rows: Vec<Row> = resp.take(0).unwrap_or_default();
        Ok(rows
            .into_iter()
            .map(|r| CharacterSnapshotData {
                embedding: r.embedding,
                created_at: r.created_at,
            })
            .collect())
    }
}

// ---------------------------------------------------------------------------
// PerceptionService
// ---------------------------------------------------------------------------

pub struct PerceptionService {
    data: Arc<dyn PerceptionDataProvider>,
}

impl PerceptionService {
    pub fn new(db: Arc<Surreal<Db>>) -> Self {
        Self {
            data: Arc::new(SurrealPerceptionDataProvider::new(db)),
        }
    }

    pub fn with_provider(data: Arc<dyn PerceptionDataProvider>) -> Self {
        Self { data }
    }

    /// Measure the perception gap between observer and target.
    pub async fn analyze_gap(
        &self,
        observer_id: &str,
        target_id: &str,
    ) -> Result<PerceptionGapResult, NarraError> {
        let edge = self
            .data
            .fetch_perceives_edge(observer_id, target_id)
            .await?
            .ok_or_else(|| NarraError::NotFound {
                entity_type: "perceives_edge".to_string(),
                id: format!("{} -> {}", observer_id, target_id),
            })?;

        let real_embedding = self
            .data
            .fetch_real_embedding(target_id)
            .await?
            .ok_or_else(|| NarraError::NotFound {
                entity_type: "embedding".to_string(),
                id: target_id.to_string(),
            })?;

        let similarity = cosine_similarity(&edge.embedding, &real_embedding);
        let gap = 1.0 - similarity;
        let assessment = gap_assessment(gap).to_string();

        let obs_fallback = name_from_id(observer_id);
        let tgt_fallback = name_from_id(target_id);

        Ok(PerceptionGapResult {
            observer_name: edge.observer_name.unwrap_or(obs_fallback),
            target_name: edge.target_name.unwrap_or(tgt_fallback),
            gap,
            similarity,
            assessment,
            perception: edge.perception,
            feelings: edge.feelings,
            tension_level: edge.tension_level,
            history_notes: edge.history_notes,
        })
    }

    /// Build a perception matrix for all observers of a target.
    pub async fn analyze_matrix(
        &self,
        target_id: &str,
        limit: usize,
    ) -> Result<PerceptionMatrixResult, NarraError> {
        let perspectives = self
            .data
            .fetch_observer_perspectives(target_id, limit)
            .await?;

        if perspectives.is_empty() {
            return Err(NarraError::NotFound {
                entity_type: "perspectives".to_string(),
                id: target_id.to_string(),
            });
        }

        let real_embedding = self.data.fetch_real_embedding(target_id).await?;
        let has_real_embedding = real_embedding.is_some();
        let target_name = name_from_id(target_id);

        // Compute per-observer gap
        let mut observer_data: Vec<ObserverGapEntry> = perspectives
            .iter()
            .map(|p| {
                let name = p.observer_name.as_deref().unwrap_or("?").to_string();
                let gap = real_embedding
                    .as_ref()
                    .map(|real| 1.0 - cosine_similarity(&p.embedding, real))
                    .unwrap_or(0.0);
                ObserverGapEntry {
                    observer_name: name,
                    gap,
                    assessment: gap_assessment(gap).to_string(),
                    embedding: p.embedding.clone(),
                    agrees_with: None,
                    disagrees_with: None,
                }
            })
            .collect();

        // Sort by accuracy (lowest gap first)
        observer_data.sort_by(|a, b| {
            a.gap
                .partial_cmp(&b.gap)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Compute pairwise agreement
        let embeddings: Vec<Vec<f32>> = observer_data.iter().map(|o| o.embedding.clone()).collect();
        let names: Vec<String> = observer_data
            .iter()
            .map(|o| o.observer_name.clone())
            .collect();

        for i in 0..observer_data.len() {
            let mut agreements: Vec<(&str, f32)> = embeddings
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(j, emb)| (names[j].as_str(), cosine_similarity(&embeddings[i], emb)))
                .collect();
            agreements.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            observer_data[i].agrees_with = agreements.first().map(|(n, _)| n.to_string());
            observer_data[i].disagrees_with = if agreements.len() > 1 {
                agreements.last().map(|(n, _)| n.to_string())
            } else {
                None
            };
        }

        Ok(PerceptionMatrixResult {
            target_name,
            has_real_embedding,
            observers: observer_data,
        })
    }

    /// Analyze how a perception has shifted over time.
    pub async fn analyze_shift(
        &self,
        observer_id: &str,
        target_id: &str,
    ) -> Result<PerceptionShiftResult, NarraError> {
        let perceives_id = self
            .data
            .fetch_perceives_edge_id(observer_id, target_id)
            .await?
            .ok_or_else(|| NarraError::NotFound {
                entity_type: "perceives_edge".to_string(),
                id: format!("{} -> {}", observer_id, target_id),
            })?;

        let persp_snaps = self.data.fetch_perspective_snapshots(&perceives_id).await?;

        if persp_snaps.is_empty() {
            return Err(NarraError::NotFound {
                entity_type: "perspective_snapshots".to_string(),
                id: format!("{} -> {}", observer_id, target_id),
            });
        }

        let char_snaps = self.data.fetch_character_snapshots(target_id).await?;

        let observer_name = name_from_id(observer_id);
        let target_name = name_from_id(target_id);

        // Build snapshot entries with gap computation
        let snapshots: Vec<ShiftSnapshotEntry> = persp_snaps
            .iter()
            .map(|psnap| {
                let gap = if !char_snaps.is_empty() {
                    let nearest = char_snaps.iter().min_by_key(|cs| {
                        if cs.created_at <= psnap.created_at {
                            0usize
                        } else {
                            1usize
                        }
                    });
                    nearest.map(|n| 1.0 - cosine_similarity(&psnap.embedding, &n.embedding))
                } else {
                    None
                };
                ShiftSnapshotEntry {
                    delta: psnap.delta_magnitude,
                    gap,
                    event: psnap.event_title.clone(),
                    timestamp: psnap.created_at.clone(),
                }
            })
            .collect();

        // Compute trajectory
        let trajectory = if char_snaps.len() >= 2 && persp_snaps.len() >= 2 {
            let first_gap = 1.0
                - cosine_similarity(
                    &persp_snaps.first().unwrap().embedding,
                    &char_snaps.first().unwrap().embedding,
                );
            let last_gap = 1.0
                - cosine_similarity(
                    &persp_snaps.last().unwrap().embedding,
                    &char_snaps.last().unwrap().embedding,
                );
            let label = trajectory_label(first_gap, last_gap);
            match label {
                "converging" => format!(
                    "Converging ({:.4} -> {:.4}): {} is getting more accurate about {}",
                    first_gap, last_gap, observer_name, target_name
                ),
                "diverging" => format!(
                    "Diverging ({:.4} -> {:.4}): {} is drifting further from reality about {}",
                    first_gap, last_gap, observer_name, target_name
                ),
                _ => format!(
                    "Stable ({:.4} -> {:.4}): {}'s accuracy about {} is unchanged",
                    first_gap, last_gap, observer_name, target_name
                ),
            }
        } else {
            "Need more snapshots for gap trajectory analysis".to_string()
        };

        Ok(PerceptionShiftResult {
            observer_name,
            target_name,
            snapshots,
            trajectory,
        })
    }
}

/// Extract the bare name from a table:key ID.
fn name_from_id(entity_id: &str) -> String {
    entity_id
        .split(':')
        .next_back()
        .unwrap_or(entity_id)
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gap_assessment_thresholds() {
        assert_eq!(gap_assessment(0.01), "remarkably accurate");
        assert_eq!(gap_assessment(0.04), "remarkably accurate");
        assert_eq!(gap_assessment(0.05), "fairly accurate");
        assert_eq!(gap_assessment(0.14), "fairly accurate");
        assert_eq!(gap_assessment(0.15), "notable blind spots");
        assert_eq!(gap_assessment(0.29), "notable blind spots");
        assert_eq!(gap_assessment(0.30), "significantly distorted");
        assert_eq!(gap_assessment(0.49), "significantly distorted");
        assert_eq!(gap_assessment(0.50), "dramatically wrong");
        assert_eq!(gap_assessment(0.90), "dramatically wrong");
    }

    #[test]
    fn test_trajectory_label() {
        assert_eq!(trajectory_label(0.5, 0.3), "converging");
        assert_eq!(trajectory_label(0.3, 0.5), "diverging");
        assert_eq!(trajectory_label(0.3, 0.31), "stable");
        assert_eq!(trajectory_label(0.3, 0.3), "stable");
    }

    #[test]
    fn test_name_from_id() {
        assert_eq!(name_from_id("character:alice"), "alice");
        assert_eq!(name_from_id("alice"), "alice");
        assert_eq!(name_from_id("perceives:abc123"), "abc123");
    }
}
