mod query_arc;
mod query_basic;
mod query_composite;
mod query_graph;
mod query_intelligence;
mod query_narrative;
mod query_perception;
mod query_search;
mod query_validation;
mod query_vector_ops;

use crate::mcp::NarraServer;
use crate::mcp::{
    EntityResult, QueryInput, QueryRequest, QueryResponse, SearchMetadataFilter, TruncationInfo,
    DEFAULT_TOKEN_BUDGET, MAX_DEPTH, MAX_LIMIT, MAX_TOKEN_BUDGET,
};
use crate::services::{EntityType, FilterOp, MetadataFilter};
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

/// Convert a typed `SearchMetadataFilter` into internal `MetadataFilter` predicates.
pub(crate) fn parse_metadata_filter(filter: &SearchMetadataFilter) -> Vec<MetadataFilter> {
    let mut filters = Vec::new();

    if let Some(ref roles) = filter.roles {
        filters.push(MetadataFilter {
            field: "roles".into(),
            op: FilterOp::Contains,
            bind_key: "filter_roles".into(),
            value: serde_json::json!(roles),
        });
    }
    if let Some(ref name) = filter.name {
        filters.push(MetadataFilter {
            field: "name".into(),
            op: FilterOp::Eq,
            bind_key: "filter_name".into(),
            value: serde_json::json!(name),
        });
    }
    if let Some(seq_min) = filter.sequence_min {
        filters.push(MetadataFilter {
            field: "sequence_min".into(),
            op: FilterOp::Gte,
            bind_key: "filter_sequence_min".into(),
            value: serde_json::json!(seq_min),
        });
    }
    if let Some(seq_max) = filter.sequence_max {
        filters.push(MetadataFilter {
            field: "sequence_max".into(),
            op: FilterOp::Lte,
            bind_key: "filter_sequence_max".into(),
            value: serde_json::json!(seq_max),
        });
    }
    if let Some(ref loc_type) = filter.loc_type {
        filters.push(MetadataFilter {
            field: "loc_type".into(),
            op: FilterOp::Eq,
            bind_key: "filter_loc_type".into(),
            value: serde_json::json!(loc_type),
        });
    }

    filters
}

/// Parse entity type strings into EntityType enums, filtering to embeddable types.
/// Returns empty vec if input is None (callers decide default behavior).
/// Per-tool-type default token budget. Composite reports get more room,
/// simple lookups get less. Falls back to DEFAULT_TOKEN_BUDGET for uncategorized ops.
fn tool_type_budget(request: &QueryRequest) -> usize {
    match request {
        // Composite reports — naturally verbose, single result with rich content
        QueryRequest::SituationReport { .. }
        | QueryRequest::CharacterDossier { .. }
        | QueryRequest::ScenePlanning { .. } => 4000,

        // Single-entity lookups — concise
        QueryRequest::Lookup { .. }
        | QueryRequest::GetFact { .. }
        | QueryRequest::ArcMoment { .. }
        | QueryRequest::CharacterVoice { .. }
        | QueryRequest::Emotions { .. }
        | QueryRequest::Themes { .. }
        | QueryRequest::ExtractEntities { .. } => 1000,

        // Analysis/intelligence tools — medium-high
        QueryRequest::KnowledgeAsymmetries { .. }
        | QueryRequest::PerceptionGap { .. }
        | QueryRequest::WhatIf { .. }
        | QueryRequest::AnalyzeImpact { .. }
        | QueryRequest::KnowledgeGapAnalysis { .. }
        | QueryRequest::InvestigateContradictions { .. }
        | QueryRequest::ConvergenceAnalysis { .. }
        | QueryRequest::DetectPhases { .. }
        | QueryRequest::DetectTransitions { .. }
        | QueryRequest::NarrativeTensions { .. }
        | QueryRequest::InferRoles { .. }
        | QueryRequest::LoadPhases => 3000,

        // Everything else (searches, lists, graphs) — standard
        _ => DEFAULT_TOKEN_BUDGET,
    }
}

pub(crate) fn parse_entity_types(types: Option<Vec<String>>) -> Vec<EntityType> {
    types
        .map(|ts| {
            ts.iter()
                .filter_map(|t| match t.to_lowercase().as_str() {
                    "character" => Some(EntityType::Character),
                    "location" => Some(EntityType::Location),
                    "event" => Some(EntityType::Event),
                    "scene" => Some(EntityType::Scene),
                    "knowledge" => Some(EntityType::Knowledge),
                    "note" => Some(EntityType::Note),
                    "fact" => Some(EntityType::Fact),
                    _ => None,
                })
                .collect()
        })
        .unwrap_or_default()
}

impl NarraServer {
    /// Handler for query tool - implementation called from server.rs
    pub async fn handle_query(
        &self,
        Parameters(input): Parameters<QueryInput>,
    ) -> Result<QueryResponse, String> {
        // Extract per-request budget before consuming input for deserialization
        let request_budget = input.token_budget;

        // Reconstruct the full request object for deserialization
        let mut full_request = serde_json::Map::new();
        full_request.insert("operation".to_string(), serde_json::json!(input.operation));
        full_request.extend(input.params);

        // Deserialize to QueryRequest
        let request: QueryRequest = serde_json::from_value(serde_json::Value::Object(full_request))
            .map_err(|e| format!("Invalid query parameters: {}", e))?;

        // Resolve effective token budget:
        // 1. Per-request token_budget (if caller specified)
        // 2. NARRA_TOKEN_BUDGET env var (if set — global override)
        // 3. Per-tool-type default
        let env_budget: Option<usize> = std::env::var("NARRA_TOKEN_BUDGET")
            .ok()
            .and_then(|s| s.parse().ok());

        let token_budget = request_budget
            .or(env_budget)
            .unwrap_or_else(|| tool_type_budget(&request))
            .min(MAX_TOKEN_BUDGET);

        // Execute query handler
        let mut response = match request {
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
                self.handle_overview(
                    &entity_type,
                    limit.unwrap_or(20).min(MAX_LIMIT),
                    crate::services::noop_progress(),
                )
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
            QueryRequest::UnifiedSearch {
                query,
                mode,
                entity_types,
                limit,
                filter,
                phase,
            } => {
                self.handle_unified_search(
                    &query,
                    &mode,
                    entity_types,
                    limit.unwrap_or(10).min(MAX_LIMIT),
                    filter,
                    phase,
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
            QueryRequest::ArcHistory {
                entity_id,
                facet,
                limit,
            } => {
                self.handle_arc_history(
                    &entity_id,
                    facet.as_deref(),
                    limit.unwrap_or(50).min(MAX_LIMIT),
                )
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
            QueryRequest::SituationReport { detail_level } => {
                self.handle_situation_report(
                    detail_level,
                    token_budget,
                    crate::services::noop_progress(),
                )
                .await
            }
            QueryRequest::CharacterDossier {
                character_id,
                detail_level,
            } => {
                self.handle_character_dossier(
                    &character_id,
                    detail_level,
                    token_budget,
                    crate::services::noop_progress(),
                )
                .await
            }
            QueryRequest::ScenePlanning {
                character_ids,
                detail_level,
            } => {
                self.handle_scene_planning(
                    &character_ids,
                    detail_level,
                    token_budget,
                    crate::services::noop_progress(),
                )
                .await
            }
            QueryRequest::GrowthVector { entity_id, limit } => {
                self.handle_growth_vector(&entity_id, limit).await
            }
            QueryRequest::MisperceptionVector {
                observer_id,
                target_id,
                limit,
            } => {
                self.handle_misperception_vector(&observer_id, &target_id, limit)
                    .await
            }
            QueryRequest::ConvergenceAnalysis {
                entity_a,
                entity_b,
                window,
            } => {
                self.handle_convergence_analysis(&entity_a, &entity_b, window)
                    .await
            }
            QueryRequest::SemanticMidpoint {
                entity_a,
                entity_b,
                limit,
            } => {
                self.handle_semantic_midpoint(&entity_a, &entity_b, limit)
                    .await
            }
            QueryRequest::AnalyzeImpact {
                entity_id,
                proposed_change,
                include_details,
            } => {
                self.handle_analyze_impact_query(&entity_id, proposed_change, include_details)
                    .await
            }
            QueryRequest::TensionMatrix { min_tension, limit } => {
                self.handle_tension_matrix(min_tension, limit).await
            }
            QueryRequest::KnowledgeGapAnalysis { character_id } => {
                self.handle_knowledge_gap_analysis(&character_id).await
            }
            QueryRequest::RelationshipStrengthMap {
                character_id,
                limit,
            } => {
                self.handle_relationship_strength_map(&character_id, limit)
                    .await
            }
            QueryRequest::NarrativeThreads { status, limit } => {
                self.handle_narrative_threads(status.as_deref(), limit)
                    .await
            }
            QueryRequest::CharacterVoice { character_id } => {
                self.handle_character_voice(&character_id).await
            }
            QueryRequest::FacetedSearch {
                query,
                facet,
                limit,
            } => self.handle_faceted_search(&query, &facet, limit).await,
            QueryRequest::MultiFacetSearch {
                query,
                weights,
                limit,
            } => {
                self.handle_multi_facet_search(&query, &weights, limit)
                    .await
            }
            QueryRequest::CharacterFacets { character_id } => {
                self.handle_character_facets(&character_id).await
            }
            QueryRequest::DetectPhases {
                entity_types,
                num_phases,
                content_weight,
                neighborhood_weight,
                temporal_weight,
                save,
            } => {
                self.handle_detect_phases(
                    entity_types,
                    num_phases,
                    content_weight,
                    neighborhood_weight,
                    temporal_weight,
                    save.unwrap_or(false),
                )
                .await
            }
            QueryRequest::LoadPhases => self.handle_load_phases().await,
            QueryRequest::QueryAround {
                anchor_id,
                entity_types,
                limit,
            } => {
                self.handle_query_around(
                    &anchor_id,
                    entity_types,
                    limit.unwrap_or(20).min(MAX_LIMIT),
                )
                .await
            }
            QueryRequest::Emotions { entity_id } => self.handle_emotions(&entity_id).await,
            QueryRequest::Themes { entity_id, themes } => {
                self.handle_themes(&entity_id, themes).await
            }
            QueryRequest::ExtractEntities { entity_id } => {
                self.handle_extract_entities(&entity_id).await
            }
            QueryRequest::NarrativeTensions {
                limit,
                min_severity,
            } => {
                self.handle_narrative_tensions(
                    limit.unwrap_or(20).min(MAX_LIMIT),
                    min_severity.unwrap_or(0.0),
                )
                .await
            }
            QueryRequest::InferRoles { limit } => {
                self.handle_infer_roles(limit.unwrap_or(20).min(MAX_LIMIT))
                    .await
            }
            QueryRequest::DetectTransitions {
                entity_types,
                num_phases,
                content_weight,
                neighborhood_weight,
                temporal_weight,
            } => {
                self.handle_detect_transitions(
                    entity_types,
                    num_phases,
                    content_weight,
                    neighborhood_weight,
                    temporal_weight,
                )
                .await
            }
        }?;

        // Apply token budget enforcement if response exceeds limit
        if response.token_estimate > token_budget && !response.results.is_empty() {
            let (truncated_results, truncation_info) =
                self.apply_token_budget(response.results, token_budget, "query");

            response.results = truncated_results;
            response.truncated = truncation_info;
            response.token_estimate = self.estimate_tokens_from_results(&response.results);
        }

        Ok(response)
    }

    // Helper methods
    fn estimate_tokens_from_results(&self, results: &[EntityResult]) -> usize {
        results.iter().map(|r| r.content.len() / 4 + 20).sum()
    }

    /// Truncate results to fit within token budget while preserving utility.
    /// Returns (truncated_results, truncation_info_opt)
    fn apply_token_budget(
        &self,
        results: Vec<EntityResult>,
        budget: usize,
        _query_name: &str,
    ) -> (Vec<EntityResult>, Option<TruncationInfo>) {
        let full_tokens = self.estimate_tokens_from_results(&results);

        if full_tokens <= budget {
            return (results, None);
        }

        // Binary search for max results that fit budget
        let mut kept_results = Vec::new();
        let mut running_tokens = 50; // Response envelope overhead

        for result in results.iter() {
            let result_tokens = result.content.len() / 4 + 20;
            if running_tokens + result_tokens > budget && !kept_results.is_empty() {
                break;
            }
            running_tokens += result_tokens;
            kept_results.push(result.clone());
        }

        let truncation = TruncationInfo {
            reason: "token_budget".to_string(),
            original_count: results.len(),
            returned_count: kept_results.len(),
            suggestion: format!(
                "Response truncated from {} to {} results to fit token budget. \
                 Use pagination or narrow your query.",
                results.len(),
                kept_results.len()
            ),
        };

        (kept_results, Some(truncation))
    }

    /// Prioritize and truncate hints to max 3 items.
    fn truncate_hints(&self, hints: Vec<String>) -> Vec<String> {
        hints.into_iter().take(3).collect()
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

        self.truncate_hints(hints)
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
            "Try unified_search (mode: hybrid) for results combining keyword matches with semantic similarity"
                .to_string(),
        );

        self.truncate_hints(hints)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp::SearchMetadataFilter;

    #[test]
    fn test_parse_empty_filter() {
        let filter = SearchMetadataFilter::default();
        let result = parse_metadata_filter(&filter);
        assert!(result.is_empty());
    }

    #[test]
    fn test_parse_roles_filter() {
        let filter = SearchMetadataFilter {
            roles: Some("warrior".to_string()),
            ..Default::default()
        };
        let result = parse_metadata_filter(&filter);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].field, "roles");
        assert!(matches!(result[0].op, FilterOp::Contains));
    }

    #[test]
    fn test_parse_sequence_range_filter() {
        let filter = SearchMetadataFilter {
            sequence_min: Some(10),
            sequence_max: Some(50),
            ..Default::default()
        };
        let result = parse_metadata_filter(&filter);
        assert_eq!(result.len(), 2);
        assert!(matches!(result[0].op, FilterOp::Gte));
        assert!(matches!(result[1].op, FilterOp::Lte));
    }

    #[test]
    fn test_parse_all_filters() {
        let filter = SearchMetadataFilter {
            roles: Some("warrior".to_string()),
            name: Some("Alice".to_string()),
            sequence_min: Some(1),
            sequence_max: Some(100),
            loc_type: Some("castle".to_string()),
        };
        let result = parse_metadata_filter(&filter);
        assert_eq!(result.len(), 5);
    }

    #[test]
    fn test_cursor_roundtrip() {
        let cursor = create_cursor(42);
        let parsed = parse_cursor(&cursor).expect("Should parse");
        assert_eq!(parsed.offset, 42);
    }

    #[test]
    fn test_invalid_cursor() {
        let result = parse_cursor("not-valid-base64!!!");
        assert!(result.is_err());
    }
}
