mod query_arc;
mod query_basic;
mod query_composite;
mod query_graph;
mod query_narrative;
mod query_perception;
mod query_search;
mod query_validation;

use crate::mcp::NarraServer;
use crate::mcp::{EntityResult, QueryRequest, QueryResponse, MAX_DEPTH, MAX_LIMIT};
use crate::services::{FilterOp, MetadataFilter};
use base64::{engine::general_purpose, Engine as _};
use rmcp::handler::server::wrapper::Parameters;
use serde::{Deserialize, Serialize};

// Cursor data structure (internal, opaque to clients)
#[derive(Serialize, Deserialize)]
pub(crate) struct CursorData {
    pub(crate) offset: usize,
}

pub(crate) fn create_cursor(offset: usize) -> String {
    let json = serde_json::to_string(&CursorData { offset }).unwrap();
    general_purpose::STANDARD.encode(json)
}

pub(crate) fn parse_cursor(cursor: &str) -> Result<CursorData, String> {
    let decoded = general_purpose::STANDARD
        .decode(cursor)
        .map_err(|_| "Invalid cursor format".to_string())?;
    serde_json::from_slice(&decoded).map_err(|_| "Invalid cursor data".to_string())
}

/// Parse a JSON object into metadata filters.
///
/// Known fields are mapped to appropriate filter operations:
/// - "roles" -> Contains (for array membership)
/// - "name" -> Eq (case-insensitive substring)
/// - "sequence_min" -> Gte
/// - "sequence_max" -> Lte
/// - "loc_type" -> Eq
///
///   Unknown fields are silently ignored.
pub(crate) fn parse_metadata_filter(filter: &serde_json::Value) -> Vec<MetadataFilter> {
    let obj = match filter.as_object() {
        Some(o) => o,
        None => return vec![],
    };

    let mut filters = Vec::new();

    for (key, value) in obj {
        let (op, bind_key) = match key.as_str() {
            "roles" => (FilterOp::Contains, "filter_roles".to_string()),
            "name" => (FilterOp::Eq, "filter_name".to_string()),
            "sequence_min" => (FilterOp::Gte, "filter_sequence_min".to_string()),
            "sequence_max" => (FilterOp::Lte, "filter_sequence_max".to_string()),
            "loc_type" => (FilterOp::Eq, "filter_loc_type".to_string()),
            _ => continue, // Silently ignore unknown fields
        };

        filters.push(MetadataFilter {
            field: key.clone(),
            op,
            bind_key,
            value: value.clone(),
        });
    }

    filters
}

impl NarraServer {
    /// Handler for query tool - implementation called from server.rs
    pub async fn handle_query(
        &self,
        Parameters(request): Parameters<QueryRequest>,
    ) -> Result<QueryResponse, String> {
        match request {
            QueryRequest::Lookup {
                entity_id,
                detail_level,
            } => {
                self.handle_lookup(&entity_id, detail_level.unwrap_or_default())
                    .await
            }
            QueryRequest::Search {
                query,
                entity_types,
                limit,
                cursor,
            } => {
                self.handle_search(
                    &query,
                    entity_types,
                    limit.unwrap_or(20).min(MAX_LIMIT),
                    cursor,
                )
                .await
            }
            QueryRequest::GraphTraversal {
                entity_id,
                depth,
                format,
            } => {
                self.handle_graph_traversal(
                    &entity_id,
                    depth.min(MAX_DEPTH),
                    format.unwrap_or_default(),
                )
                .await
            }
            QueryRequest::Temporal {
                character_id,
                event_id,
                event_name,
            } => {
                self.handle_temporal(&character_id, event_id, event_name)
                    .await
            }
            QueryRequest::Overview { entity_type, limit } => {
                self.handle_overview(&entity_type, limit.unwrap_or(20).min(MAX_LIMIT))
                    .await
            }
            QueryRequest::ListNotes { entity_id, limit } => {
                self.handle_list_notes(entity_id, limit.unwrap_or(50).min(MAX_LIMIT))
                    .await
            }
            QueryRequest::GetFact { fact_id } => self.handle_get_fact(&fact_id).await,
            QueryRequest::ListFacts {
                category,
                enforcement_level,
                search,
                entity_id,
                limit,
                cursor,
            } => {
                self.handle_list_facts(
                    category,
                    enforcement_level,
                    search,
                    entity_id,
                    limit.unwrap_or(20).min(MAX_LIMIT),
                    cursor,
                )
                .await
            }
            QueryRequest::SemanticSearch {
                query,
                entity_types,
                limit,
                filter,
            } => {
                self.handle_semantic_search(
                    &query,
                    entity_types,
                    limit.unwrap_or(10).min(MAX_LIMIT),
                    filter,
                )
                .await
            }
            QueryRequest::HybridSearch {
                query,
                entity_types,
                limit,
                filter,
            } => {
                self.handle_hybrid_search(
                    &query,
                    entity_types,
                    limit.unwrap_or(10).min(MAX_LIMIT),
                    filter,
                )
                .await
            }
            QueryRequest::ReverseQuery {
                entity_id,
                referencing_types,
                limit,
            } => {
                self.handle_reverse_query(
                    &entity_id,
                    referencing_types,
                    limit.unwrap_or(20).min(MAX_LIMIT),
                )
                .await
            }
            QueryRequest::ConnectionPath {
                from_id,
                to_id,
                max_hops,
                include_events,
            } => {
                self.handle_connection_path(
                    &from_id,
                    &to_id,
                    max_hops.unwrap_or(3).min(MAX_DEPTH),
                    include_events.unwrap_or(true),
                )
                .await
            }
            QueryRequest::CentralityMetrics {
                scope,
                metrics,
                limit,
            } => {
                self.handle_centrality_metrics(scope, metrics, limit.unwrap_or(20).min(MAX_LIMIT))
                    .await
            }
            QueryRequest::InfluencePropagation {
                from_character_id,
                knowledge_fact,
                max_depth,
            } => {
                self.handle_influence_propagation(
                    &from_character_id,
                    knowledge_fact,
                    max_depth.unwrap_or(3).min(MAX_DEPTH),
                )
                .await
            }
            QueryRequest::DramaticIronyReport {
                character_id,
                min_scene_threshold,
            } => {
                self.handle_dramatic_irony_report(character_id, min_scene_threshold.unwrap_or(3))
                    .await
            }
            QueryRequest::SemanticJoin {
                query,
                entity_types,
                limit,
            } => {
                self.handle_semantic_join(&query, entity_types, limit.unwrap_or(10).min(MAX_LIMIT))
                    .await
            }
            QueryRequest::ThematicClustering {
                entity_types,
                num_themes,
            } => {
                self.handle_thematic_clustering(entity_types, num_themes)
                    .await
            }
            QueryRequest::SemanticKnowledge {
                query,
                character_id,
                limit,
            } => {
                self.handle_semantic_knowledge(
                    &query,
                    character_id,
                    limit.unwrap_or(10).min(MAX_LIMIT),
                )
                .await
            }
            QueryRequest::SemanticGraphSearch {
                entity_id,
                max_hops,
                query,
                entity_types,
                limit,
            } => {
                self.handle_semantic_graph_search(
                    &entity_id,
                    max_hops.unwrap_or(2).min(MAX_DEPTH),
                    &query,
                    entity_types,
                    limit.unwrap_or(10).min(MAX_LIMIT),
                )
                .await
            }
            QueryRequest::ArcHistory { entity_id, limit } => {
                self.handle_arc_history(&entity_id, limit.unwrap_or(50).min(MAX_LIMIT))
                    .await
            }
            QueryRequest::ArcComparison {
                entity_id_a,
                entity_id_b,
                window,
            } => {
                self.handle_arc_comparison(&entity_id_a, &entity_id_b, window)
                    .await
            }
            QueryRequest::ArcDrift { entity_type, limit } => {
                self.handle_arc_drift(entity_type, limit.unwrap_or(20).min(MAX_LIMIT))
                    .await
            }
            QueryRequest::ArcMoment {
                entity_id,
                event_id,
            } => self.handle_arc_moment(&entity_id, event_id).await,
            QueryRequest::PerspectiveSearch {
                query,
                observer_id,
                target_id,
                limit,
            } => {
                self.handle_perspective_search(
                    &query,
                    observer_id,
                    target_id,
                    limit.unwrap_or(10).min(MAX_LIMIT),
                )
                .await
            }
            QueryRequest::PerceptionGap {
                observer_id,
                target_id,
            } => self.handle_perception_gap(&observer_id, &target_id).await,
            QueryRequest::PerceptionMatrix { target_id, limit } => {
                self.handle_perception_matrix(&target_id, limit.unwrap_or(20).min(MAX_LIMIT))
                    .await
            }
            QueryRequest::PerceptionShift {
                observer_id,
                target_id,
            } => self.handle_perception_shift(&observer_id, &target_id).await,
            QueryRequest::UnresolvedTensions {
                limit,
                min_asymmetry,
                max_shared_scenes,
            } => {
                self.handle_unresolved_tensions(
                    limit.unwrap_or(10).min(MAX_LIMIT),
                    min_asymmetry.unwrap_or(0.1),
                    max_shared_scenes,
                )
                .await
            }
            QueryRequest::ThematicGaps {
                min_cluster_size,
                expected_types,
            } => {
                self.handle_thematic_gaps(min_cluster_size.unwrap_or(3), expected_types)
                    .await
            }
            QueryRequest::SimilarRelationships {
                observer_id,
                target_id,
                edge_type,
                bias,
                limit,
            } => {
                self.handle_similar_relationships(
                    &observer_id,
                    &target_id,
                    edge_type,
                    bias,
                    limit.unwrap_or(10).min(MAX_LIMIT),
                )
                .await
            }
            QueryRequest::KnowledgeConflicts {
                character_id,
                limit,
            } => {
                self.handle_knowledge_conflicts(character_id, limit.unwrap_or(50).min(MAX_LIMIT))
                    .await
            }
            QueryRequest::EmbeddingHealth => self.handle_embedding_health().await,
            QueryRequest::WhatIf {
                character_id,
                fact_id,
                certainty,
                source_character,
            } => {
                self.handle_what_if(
                    &character_id,
                    &fact_id,
                    certainty.as_deref(),
                    source_character.as_deref(),
                )
                .await
            }
            QueryRequest::ValidateEntity { entity_id } => {
                self.handle_validate_entity_query(&entity_id).await
            }
            QueryRequest::InvestigateContradictions {
                entity_id,
                max_depth,
            } => {
                self.handle_investigate_contradictions_query(&entity_id, max_depth.min(MAX_DEPTH))
                    .await
            }
            QueryRequest::KnowledgeAsymmetries {
                character_a,
                character_b,
            } => {
                self.handle_knowledge_asymmetries(&character_a, &character_b)
                    .await
            }
            QueryRequest::SituationReport => self.handle_situation_report().await,
            QueryRequest::CharacterDossier { character_id } => {
                self.handle_character_dossier(&character_id).await
            }
            QueryRequest::ScenePlanning { character_ids } => {
                self.handle_scene_planning(&character_ids).await
            }
            QueryRequest::AnalyzeImpact {
                entity_id,
                proposed_change,
                include_details,
            } => {
                self.handle_analyze_impact_query(&entity_id, proposed_change, include_details)
                    .await
            }
        }
    }

    // Helper methods
    fn estimate_tokens_from_results(&self, results: &[EntityResult]) -> usize {
        results.iter().map(|r| r.content.len() / 4 + 20).sum()
    }

    async fn generate_lookup_hints(&self, entity_id: &str, result: &EntityResult) -> Vec<String> {
        let mut hints = vec![];

        // Suggest graph traversal for connected entities
        hints.push(format!(
            "Use graph_traversal on '{}' to see related entities",
            entity_id
        ));

        // Type-specific hints
        match result.entity_type.as_str() {
            "character" => {
                hints.push("Query temporal knowledge to see what this character knows".to_string());
            }
            "event" => {
                hints.push("Check which characters participated in this event".to_string());
            }
            _ => {}
        }

        hints
    }

    fn generate_search_hints(&self, _query: &str, results: &[EntityResult]) -> Vec<String> {
        let mut hints = vec![];

        if results.is_empty() {
            hints.push("No results found. Try a broader search term.".to_string());
        } else if results.len() == 1 {
            hints.push(format!(
                "Use lookup on '{}' for full details",
                results[0].id
            ));
        } else {
            hints.push(format!(
                "Found {} results. Narrow with entity_types filter if needed.",
                results.len()
            ));
        }

        hints.push(
            "Try hybrid_search for results combining keyword matches with semantic similarity"
                .to_string(),
        );

        hints
    }
}
