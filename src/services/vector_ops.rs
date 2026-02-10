//! Embedding arithmetic for narrative analysis.
//!
//! Operates on entity embeddings to derive narrative insights:
//! - Growth vectors: where is a character heading?
//! - Misperception vectors: what does an observer get wrong?
//! - Convergence analysis: are two entities becoming more alike?
//! - Semantic midpoints: what bridges two concepts?

use std::sync::Arc;

use crate::db::connection::NarraDb;
use serde::{Deserialize, Serialize};

use crate::utils::math::{cosine_similarity, vector_midpoint, vector_normalize, vector_subtract};
use crate::NarraError;

/// Result of computing a growth vector for an entity.
#[derive(Debug, Clone, Serialize)]
pub struct GrowthVectorResult {
    pub entity_id: String,
    pub entity_name: String,
    /// Number of arc snapshots used
    pub snapshot_count: usize,
    /// Cosine similarity between first and last snapshot (1.0 = no change)
    pub total_drift: f32,
    /// Entities nearest to the extrapolated trajectory
    pub trajectory_neighbors: Vec<TrajectoryNeighbor>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TrajectoryNeighbor {
    pub entity_id: String,
    pub entity_name: String,
    pub entity_type: String,
    /// How aligned this entity is with the growth direction
    pub alignment_score: f32,
}

/// Result of computing a misperception vector.
#[derive(Debug, Clone, Serialize)]
pub struct MisperceptionResult {
    pub observer_id: String,
    pub observer_name: String,
    pub target_id: String,
    pub target_name: String,
    /// Cosine distance between perception and reality (0 = accurate, 1 = maximally wrong)
    pub perception_gap: f32,
    /// Entities most similar to the misperception direction
    pub misperception_neighbors: Vec<TrajectoryNeighbor>,
}

/// A single data point in convergence analysis.
#[derive(Debug, Clone, Serialize)]
pub struct ConvergencePoint {
    pub snapshot_index: usize,
    pub similarity: f32,
    pub event_label: Option<String>,
}

/// Result of convergence analysis between two entities.
#[derive(Debug, Clone, Serialize)]
pub struct ConvergenceResult {
    pub entity_a_id: String,
    pub entity_a_name: String,
    pub entity_b_id: String,
    pub entity_b_name: String,
    /// Per-snapshot similarity trend
    pub trend: Vec<ConvergencePoint>,
    /// Positive = converging, negative = diverging
    pub convergence_rate: f32,
    /// Current similarity
    pub current_similarity: f32,
}

/// Result of semantic midpoint search.
#[derive(Debug, Clone, Serialize)]
pub struct MidpointResult {
    pub entity_a_id: String,
    pub entity_b_id: String,
    /// Entities nearest to the midpoint embedding
    pub neighbors: Vec<TrajectoryNeighbor>,
}

/// Service for embedding arithmetic operations.
pub struct VectorOpsService {
    db: Arc<NarraDb>,
}

impl VectorOpsService {
    pub fn new(db: Arc<NarraDb>) -> Self {
        Self { db }
    }

    /// Compute growth vector: latest_embedding - first_embedding.
    /// Finds entities nearest to the trajectory extrapolation.
    pub async fn growth_vector(
        &self,
        entity_id: &str,
        limit: usize,
    ) -> Result<GrowthVectorResult, NarraError> {
        #[derive(Deserialize)]
        struct Snapshot {
            embedding: Vec<f32>,
        }

        // Fetch arc snapshots ordered by time
        let query = format!(
            "SELECT embedding FROM arc_snapshot \
             WHERE entity_id = {} ORDER BY created_at ASC",
            entity_id
        );
        let mut resp = self.db.query(&query).await?;
        let snapshots: Vec<Snapshot> = resp.take(0)?;

        if snapshots.len() < 2 {
            return Err(NarraError::Database(format!(
                "Need at least 2 arc snapshots for growth vector (found {}). \
                 Run 'narra world baseline-arcs' first.",
                snapshots.len()
            )));
        }

        let first = &snapshots[0].embedding;
        let last = &snapshots[snapshots.len() - 1].embedding;
        let growth = vector_subtract(last, first);
        let total_drift = 1.0 - cosine_similarity(first, last);

        // Get entity name
        let entity_name = self.get_entity_name(entity_id).await?;

        // Find entities whose embeddings align with the growth direction
        let trajectory_neighbors = self
            .find_aligned_entities(&growth, entity_id, limit)
            .await?;

        Ok(GrowthVectorResult {
            entity_id: entity_id.to_string(),
            entity_name,
            snapshot_count: snapshots.len(),
            total_drift,
            trajectory_neighbors,
        })
    }

    /// Compute misperception vector: observer's perception embedding minus target's actual embedding.
    pub async fn misperception_vector(
        &self,
        observer_id: &str,
        target_id: &str,
        limit: usize,
    ) -> Result<MisperceptionResult, NarraError> {
        // Get observer's perception of target
        #[derive(Deserialize)]
        struct PerceptionRow {
            embedding: Option<Vec<f32>>,
        }

        let query = format!(
            "SELECT embedding FROM perceives \
             WHERE in = {} AND out = {} LIMIT 1",
            observer_id, target_id
        );
        let mut resp = self.db.query(&query).await?;
        let perception: Option<PerceptionRow> = resp.take(0)?;

        let perception_embedding = perception.and_then(|p| p.embedding).ok_or_else(|| {
            NarraError::Database(format!(
                "No perception embedding found for {} -> {}. Run backfill first.",
                observer_id, target_id
            ))
        })?;

        // Get target's actual embedding
        #[derive(Deserialize)]
        struct EmbeddingRow {
            embedding: Option<Vec<f32>>,
        }

        let query = format!("SELECT embedding FROM {} LIMIT 1", target_id);
        let mut resp = self.db.query(&query).await?;
        let target_row: Option<EmbeddingRow> = resp.take(0)?;

        let target_embedding = target_row.and_then(|r| r.embedding).ok_or_else(|| {
            NarraError::Database(format!(
                "No embedding found for {}. Run backfill first.",
                target_id
            ))
        })?;

        let misperception = vector_subtract(&perception_embedding, &target_embedding);
        let perception_gap = 1.0 - cosine_similarity(&perception_embedding, &target_embedding);

        let observer_name = self.get_entity_name(observer_id).await?;
        let target_name = self.get_entity_name(target_id).await?;

        let misperception_neighbors = self
            .find_aligned_entities(&misperception, observer_id, limit)
            .await?;

        Ok(MisperceptionResult {
            observer_id: observer_id.to_string(),
            observer_name,
            target_id: target_id.to_string(),
            target_name,
            perception_gap,
            misperception_neighbors,
        })
    }

    /// Analyze convergence/divergence between two entities over time.
    pub async fn convergence_analysis(
        &self,
        entity_a: &str,
        entity_b: &str,
        window: Option<usize>,
    ) -> Result<ConvergenceResult, NarraError> {
        #[derive(Deserialize)]
        struct Snapshot {
            embedding: Vec<f32>,
            #[serde(default)]
            trigger_event: Option<String>,
        }

        let limit = window.unwrap_or(50);

        let query_a = format!(
            "SELECT embedding, trigger_event FROM arc_snapshot \
             WHERE entity_id = {} ORDER BY created_at DESC LIMIT {}",
            entity_a, limit
        );
        let mut resp_a = self.db.query(&query_a).await?;
        let mut snaps_a: Vec<Snapshot> = resp_a.take(0)?;
        snaps_a.reverse(); // oldest first

        let query_b = format!(
            "SELECT embedding, trigger_event FROM arc_snapshot \
             WHERE entity_id = {} ORDER BY created_at DESC LIMIT {}",
            entity_b, limit
        );
        let mut resp_b = self.db.query(&query_b).await?;
        let mut snaps_b: Vec<Snapshot> = resp_b.take(0)?;
        snaps_b.reverse();

        // Align by index (zip shortest)
        let pair_count = snaps_a.len().min(snaps_b.len());
        let mut trend = Vec::with_capacity(pair_count);

        for i in 0..pair_count {
            let sim = cosine_similarity(&snaps_a[i].embedding, &snaps_b[i].embedding);
            trend.push(ConvergencePoint {
                snapshot_index: i,
                similarity: sim,
                event_label: snaps_a[i].trigger_event.clone(),
            });
        }

        // Compute convergence rate as slope of similarity trend
        let convergence_rate = if trend.len() >= 2 {
            let first_sim = trend[0].similarity;
            let last_sim = trend[trend.len() - 1].similarity;
            (last_sim - first_sim) / (trend.len() - 1) as f32
        } else {
            0.0
        };

        let current_similarity = trend.last().map(|p| p.similarity).unwrap_or(0.0);

        let entity_a_name = self.get_entity_name(entity_a).await?;
        let entity_b_name = self.get_entity_name(entity_b).await?;

        Ok(ConvergenceResult {
            entity_a_id: entity_a.to_string(),
            entity_a_name,
            entity_b_id: entity_b.to_string(),
            entity_b_name,
            trend,
            convergence_rate,
            current_similarity,
        })
    }

    /// Find entities nearest to the midpoint of two entity embeddings.
    pub async fn semantic_midpoint(
        &self,
        entity_a: &str,
        entity_b: &str,
        limit: usize,
    ) -> Result<MidpointResult, NarraError> {
        let emb_a = self.get_entity_embedding(entity_a).await?;
        let emb_b = self.get_entity_embedding(entity_b).await?;

        let midpoint = vector_midpoint(&emb_a, &emb_b);

        let neighbors = self
            .find_nearest_entities(&midpoint, &[entity_a, entity_b], limit)
            .await?;

        Ok(MidpointResult {
            entity_a_id: entity_a.to_string(),
            entity_b_id: entity_b.to_string(),
            neighbors,
        })
    }

    // =========================================================================
    // Internal helpers
    // =========================================================================

    async fn get_entity_name(&self, entity_id: &str) -> Result<String, NarraError> {
        #[derive(Deserialize)]
        struct NameRow {
            #[serde(default)]
            name: Option<String>,
            #[serde(default)]
            title: Option<String>,
        }

        let query = format!("SELECT name, title FROM {} LIMIT 1", entity_id);
        let mut resp = self.db.query(&query).await?;
        let row: Option<NameRow> = resp.take(0)?;

        Ok(row
            .and_then(|r| r.name.or(r.title))
            .unwrap_or_else(|| entity_id.to_string()))
    }

    async fn get_entity_embedding(&self, entity_id: &str) -> Result<Vec<f32>, NarraError> {
        #[derive(Deserialize)]
        struct EmbRow {
            embedding: Option<Vec<f32>>,
        }

        let query = format!("SELECT embedding FROM {} LIMIT 1", entity_id);
        let mut resp = self.db.query(&query).await?;
        let row: Option<EmbRow> = resp.take(0)?;

        row.and_then(|r| r.embedding).ok_or_else(|| {
            NarraError::Database(format!(
                "No embedding found for {}. Run backfill first.",
                entity_id
            ))
        })
    }

    /// Find entities whose embeddings align with a direction vector.
    async fn find_aligned_entities(
        &self,
        direction: &[f32],
        exclude_id: &str,
        limit: usize,
    ) -> Result<Vec<TrajectoryNeighbor>, NarraError> {
        let normalized = vector_normalize(direction);
        self.find_nearest_entities(&normalized, &[exclude_id], limit)
            .await
    }

    /// Find entities nearest to a target vector (by cosine similarity).
    async fn find_nearest_entities(
        &self,
        target: &[f32],
        exclude_ids: &[&str],
        limit: usize,
    ) -> Result<Vec<TrajectoryNeighbor>, NarraError> {
        let tables = ["character", "location", "event", "scene"];
        let mut all_neighbors = Vec::new();

        for table in &tables {
            let name_field = match *table {
                "event" | "scene" => "title",
                _ => "name",
            };

            #[derive(Deserialize)]
            struct Row {
                id: surrealdb::RecordId,
                name: String,
                score: f32,
            }

            let query = format!(
                "SELECT id, {name_field} AS name, \
                 vector::similarity::cosine(embedding, $vec) AS score \
                 FROM {table} WHERE embedding IS NOT NONE \
                 ORDER BY score DESC LIMIT {k}",
                name_field = name_field,
                table = table,
                k = limit * 2,
            );

            let mut resp = self.db.query(&query).bind(("vec", target.to_vec())).await?;
            let rows: Vec<Row> = resp.take(0).unwrap_or_default();

            for row in rows {
                let id_str = row.id.to_string();
                if exclude_ids.contains(&id_str.as_str()) {
                    continue;
                }
                all_neighbors.push(TrajectoryNeighbor {
                    entity_id: id_str,
                    entity_name: row.name,
                    entity_type: table.to_string(),
                    alignment_score: row.score,
                });
            }
        }

        all_neighbors.sort_by(|a, b| {
            b.alignment_score
                .partial_cmp(&a.alignment_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        all_neighbors.truncate(limit);

        Ok(all_neighbors)
    }
}
