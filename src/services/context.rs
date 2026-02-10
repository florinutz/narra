use crate::db::connection::NarraDb;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

use crate::repository::{RelationshipRepository, SurrealRelationshipRepository};
use crate::services::summary::{CachedSummaryService, DetailLevel, SummaryService};
use crate::session::SessionStateManager;
use crate::NarraError;

pub use crate::services::summary::EntityFullContent;

/// An entity with its context relevance score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredEntity {
    /// Entity ID (e.g., "character:alice")
    pub id: String,
    /// Entity type (character, location, event, scene)
    pub entity_type: String,
    /// Display name
    pub name: String,
    /// Entity content (summarized or full based on detail level)
    pub content: Option<String>,
    /// Whether content is summarized
    pub is_summarized: bool,
    /// Relevance score (higher = more relevant)
    pub score: f32,
    /// Breakdown of score components
    pub score_breakdown: ScoreBreakdown,
}

/// Breakdown of how relevance score was calculated.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ScoreBreakdown {
    /// Score from direct mention in query
    pub mention_score: f32,
    /// Score from recent access
    pub recency_score: f32,
    /// Score from graph proximity to mentioned entities
    pub proximity_score: f32,
    /// Score from explicit pin
    pub pin_score: f32,
}

/// Context retrieval response with token estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextResponse {
    /// Scored and sorted entities
    pub entities: Vec<ScoredEntity>,
    /// Estimated token count for this context
    pub estimated_tokens: usize,
    /// Whether context was truncated due to budget
    pub truncated: bool,
}

/// Configuration for context retrieval.
#[derive(Debug, Clone)]
pub struct ContextConfig {
    /// Maximum entities to return
    pub max_entities: usize,
    /// Token budget (approximate)
    pub token_budget: usize,
    /// Graph traversal depth for proximity scoring
    pub graph_depth: usize,
    /// Explicit entity pins (always include these)
    pub pinned_entities: Vec<String>,
    /// Detail level for entity content (default: Summary)
    pub detail_level: DetailLevel,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            max_entities: 20,
            token_budget: 4000,
            graph_depth: 3,
            pinned_entities: Vec::new(),
            detail_level: DetailLevel::default(), // Summary
        }
    }
}

/// Context service trait for relevance-based entity retrieval.
#[async_trait]
pub trait ContextService: Send + Sync {
    /// Get relevant context for a query/topic.
    ///
    /// Scores entities by:
    /// - Direct mentions in the query (10.0 points)
    /// - Recency of access (5.0 / position in recent list)
    /// - Graph proximity to mentioned entities (3.0 / distance)
    /// - Explicit pins (2.0 points)
    async fn get_context(
        &self,
        mentioned_entities: &[String],
        config: ContextConfig,
    ) -> Result<ContextResponse, NarraError>;

    /// Record an entity access (updates recency tracking).
    async fn record_access(&self, entity_id: &str);

    /// Pin an entity (always include in context).
    async fn pin_entity(&self, entity_id: &str);

    /// Unpin an entity.
    async fn unpin_entity(&self, entity_id: &str);

    /// Get the current hot entities (most accessed).
    async fn get_hot_entities(&self, limit: usize) -> Vec<String>;

    /// Get full content for a specific entity (SUMM-03).
    /// Use when user requests full detail after seeing summary.
    async fn get_entity_full_detail(
        &self,
        entity_id: &str,
    ) -> Result<Option<EntityFullContent>, NarraError>;
}

/// Cached implementation of ContextService with hot entity tracking.
pub struct CachedContextService {
    db: Arc<NarraDb>,
    relationship_repo: Arc<SurrealRelationshipRepository>,
    summary_service: Arc<CachedSummaryService>,
    /// Session state manager (single source of truth for pinned/recent)
    session_manager: Arc<SessionStateManager>,
}

impl CachedContextService {
    /// Create a new cached context service.
    pub fn new(db: Arc<NarraDb>, session_manager: Arc<SessionStateManager>) -> Self {
        Self {
            db: db.clone(),
            relationship_repo: Arc::new(SurrealRelationshipRepository::new(db.clone())),
            summary_service: Arc::new(CachedSummaryService::with_defaults(db)),
            session_manager,
        }
    }

    /// Calculate relevance score for an entity.
    fn calculate_score(
        &self,
        entity_id: &str,
        mentioned: &[String],
        recent: &[String],
        graph_distances: &HashMap<String, usize>,
        pinned: &std::collections::HashSet<String>,
    ) -> (f32, ScoreBreakdown) {
        calculate_score(entity_id, mentioned, recent, graph_distances, pinned)
    }
}

#[async_trait]
impl ContextService for CachedContextService {
    async fn get_context(
        &self,
        mentioned_entities: &[String],
        config: ContextConfig,
    ) -> Result<ContextResponse, NarraError> {
        use crate::models::note::get_entity_notes;

        let recent = self.session_manager.get_recent(100).await;
        let pinned: std::collections::HashSet<String> = self
            .session_manager
            .get_pinned()
            .await
            .into_iter()
            .collect();

        // Build graph distances for mentioned entities
        let mut graph_distances: HashMap<String, usize> = HashMap::new();
        for mentioned in mentioned_entities {
            graph_distances.insert(mentioned.clone(), 0);

            // Get connected entities up to configured depth
            let connected = self
                .relationship_repo
                .get_connected_entities(mentioned, config.graph_depth)
                .await?;

            for (idx, connected_id) in connected.iter().enumerate() {
                let distance = idx + 1; // Approximate distance by BFS order
                graph_distances
                    .entry(connected_id.clone())
                    .or_insert(distance);
            }
        }

        // Collect candidate entities: mentioned + connected + recent + pinned
        let mut candidates: std::collections::HashSet<String> = std::collections::HashSet::new();
        candidates.extend(mentioned_entities.iter().cloned());
        candidates.extend(graph_distances.keys().cloned());
        candidates.extend(recent.iter().cloned());
        candidates.extend(pinned.iter().cloned());
        candidates.extend(config.pinned_entities.iter().cloned());

        // Score and sort candidates
        let mut scored: Vec<ScoredEntity> = Vec::new();

        // Track parent entity scores for note scoring
        let mut parent_scores: HashMap<String, f32> = HashMap::new();

        for candidate in &candidates {
            // Get entity summary based on detail level
            let summary = self
                .summary_service
                .get_entity_content(candidate, config.detail_level)
                .await?;

            if let Some(entity_summary) = summary {
                let (score, breakdown) = self.calculate_score(
                    candidate,
                    mentioned_entities,
                    &recent,
                    &graph_distances,
                    &pinned,
                );

                parent_scores.insert(candidate.clone(), score);

                scored.push(ScoredEntity {
                    id: entity_summary.id,
                    entity_type: entity_summary.entity_type,
                    name: entity_summary.name,
                    content: Some(entity_summary.content),
                    is_summarized: entity_summary.is_summarized,
                    score,
                    score_breakdown: breakdown,
                });
            }
        }

        // Fetch and include notes attached to mentioned entities
        // Notes are scored based on parent entity relevance (slightly lower priority)
        let mut seen_note_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
        for entity_id in mentioned_entities {
            if let Ok(notes) = get_entity_notes(&self.db, entity_id).await {
                let parent_score = parent_scores.get(entity_id).copied().unwrap_or(0.0);
                // Notes get 80% of parent's score (lower priority than primary entities)
                let note_score_multiplier = 0.8;

                for note in notes {
                    let note_id = note.id.to_string();
                    if seen_note_ids.contains(&note_id) {
                        continue;
                    }
                    seen_note_ids.insert(note_id.clone());

                    let note_score = parent_score * note_score_multiplier;
                    let content = format!("{}\n\n(Attached to {})", note.body, entity_id);

                    // Token estimation: ~4 chars per token, notes typically moderate size
                    let estimated_tokens = content.len().div_ceil(4);

                    scored.push(ScoredEntity {
                        id: note_id,
                        entity_type: "note".to_string(),
                        name: note.title,
                        content: Some(content),
                        is_summarized: estimated_tokens > 80, // Summarized if over ~80 tokens
                        score: note_score,
                        score_breakdown: ScoreBreakdown {
                            mention_score: 0.0,
                            recency_score: 0.0,
                            proximity_score: note_score, // Use proximity since attached to mentioned entity
                            pin_score: 0.0,
                        },
                    });
                }
            }
        }

        // Sort by score descending
        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Apply token budget
        let mut total_tokens = 0;
        let mut truncated = false;
        let mut final_entities: Vec<ScoredEntity> = Vec::new();

        for entity in scored {
            // Use actual token estimate from summary
            let entity_tokens = entity
                .content
                .as_ref()
                .map(|c| c.len().div_ceil(4))
                .unwrap_or(100);

            if total_tokens + entity_tokens > config.token_budget && !final_entities.is_empty() {
                truncated = true;
                break;
            }
            total_tokens += entity_tokens;
            final_entities.push(entity);

            if final_entities.len() >= config.max_entities {
                break;
            }
        }

        Ok(ContextResponse {
            entities: final_entities,
            estimated_tokens: total_tokens,
            truncated,
        })
    }

    async fn record_access(&self, entity_id: &str) {
        self.session_manager.record_access(entity_id).await;
    }

    async fn pin_entity(&self, entity_id: &str) {
        self.session_manager.pin_entity(entity_id).await;
    }

    async fn unpin_entity(&self, entity_id: &str) {
        self.session_manager.unpin_entity(entity_id).await;
    }

    async fn get_hot_entities(&self, limit: usize) -> Vec<String> {
        self.session_manager.get_recent(limit).await
    }

    async fn get_entity_full_detail(
        &self,
        entity_id: &str,
    ) -> Result<Option<EntityFullContent>, NarraError> {
        self.summary_service.get_full_content(entity_id).await
    }
}

/// Standalone score calculation (extracted for testability).
fn calculate_score(
    entity_id: &str,
    mentioned: &[String],
    recent: &[String],
    graph_distances: &HashMap<String, usize>,
    pinned: &std::collections::HashSet<String>,
) -> (f32, ScoreBreakdown) {
    let mut breakdown = ScoreBreakdown::default();

    // Mention score (10.0 if directly mentioned)
    if mentioned.contains(&entity_id.to_string()) {
        breakdown.mention_score = 10.0;
    }

    // Recency score (5.0 / position, capped at 5.0)
    if let Some(pos) = recent.iter().position(|e| e == entity_id) {
        breakdown.recency_score = 5.0 / (pos as f32 + 1.0);
    }

    // Proximity score (3.0 / distance for connected entities)
    if let Some(&distance) = graph_distances.get(entity_id) {
        if distance > 0 {
            breakdown.proximity_score = 3.0 / distance as f32;
        }
    }

    // Pin score (2.0 if pinned)
    if pinned.contains(entity_id) {
        breakdown.pin_score = 2.0;
    }

    let total = breakdown.mention_score
        + breakdown.recency_score
        + breakdown.proximity_score
        + breakdown.pin_score;

    (total, breakdown)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn empty_distances() -> HashMap<String, usize> {
        HashMap::new()
    }

    fn empty_pinned() -> HashSet<String> {
        HashSet::new()
    }

    #[test]
    fn test_mentioned_entity_gets_10_points() {
        let mentioned = vec!["character:alice".to_string()];
        let (score, breakdown) = calculate_score(
            "character:alice",
            &mentioned,
            &[],
            &empty_distances(),
            &empty_pinned(),
        );
        assert!((breakdown.mention_score - 10.0).abs() < 0.01);
        assert!((score - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_first_recent_gets_5_points() {
        let recent = vec!["character:alice".to_string(), "character:bob".to_string()];
        let (_, breakdown) = calculate_score(
            "character:alice",
            &[],
            &recent,
            &empty_distances(),
            &empty_pinned(),
        );
        assert!((breakdown.recency_score - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_second_recent_gets_2_5_points() {
        let recent = vec!["character:alice".to_string(), "character:bob".to_string()];
        let (_, breakdown) = calculate_score(
            "character:bob",
            &[],
            &recent,
            &empty_distances(),
            &empty_pinned(),
        );
        assert!((breakdown.recency_score - 2.5).abs() < 0.01);
    }

    #[test]
    fn test_proximity_distance_1_gives_3_points() {
        let mut distances = HashMap::new();
        distances.insert("character:bob".to_string(), 1);
        let (_, breakdown) =
            calculate_score("character:bob", &[], &[], &distances, &empty_pinned());
        assert!((breakdown.proximity_score - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_proximity_distance_2_gives_1_5_points() {
        let mut distances = HashMap::new();
        distances.insert("character:bob".to_string(), 2);
        let (_, breakdown) =
            calculate_score("character:bob", &[], &[], &distances, &empty_pinned());
        assert!((breakdown.proximity_score - 1.5).abs() < 0.01);
    }

    #[test]
    fn test_pinned_entity_gets_2_points() {
        let mut pinned = HashSet::new();
        pinned.insert("character:alice".to_string());
        let (_, breakdown) =
            calculate_score("character:alice", &[], &[], &empty_distances(), &pinned);
        assert!((breakdown.pin_score - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_combined_scores_add_correctly() {
        let mentioned = vec!["character:alice".to_string()];
        let recent = vec!["character:alice".to_string()];
        let mut distances = HashMap::new();
        distances.insert("character:alice".to_string(), 1);
        let mut pinned = HashSet::new();
        pinned.insert("character:alice".to_string());

        let (total, breakdown) =
            calculate_score("character:alice", &mentioned, &recent, &distances, &pinned);

        // mention=10 + recency=5 + proximity=3 + pin=2 = 20
        assert!((total - 20.0).abs() < 0.01);
        assert!((breakdown.mention_score - 10.0).abs() < 0.01);
        assert!((breakdown.recency_score - 5.0).abs() < 0.01);
        assert!((breakdown.proximity_score - 3.0).abs() < 0.01);
        assert!((breakdown.pin_score - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_no_signals_scores_zero() {
        let (total, breakdown) = calculate_score(
            "character:nobody",
            &[],
            &[],
            &empty_distances(),
            &empty_pinned(),
        );
        assert!((total).abs() < 0.01);
        assert!((breakdown.mention_score).abs() < 0.01);
        assert!((breakdown.recency_score).abs() < 0.01);
        assert!((breakdown.proximity_score).abs() < 0.01);
        assert!((breakdown.pin_score).abs() < 0.01);
    }
}
