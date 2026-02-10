//! MCP query handlers for vector arithmetic operations.

use crate::mcp::types::{EntityResult, QueryResponse};
use crate::services::VectorOpsService;

use super::NarraServer;

/// Normalize an entity ID: if it doesn't contain ':', try to guess table from context.
fn normalize_id(id: &str, default_table: Option<&str>) -> String {
    if id.contains(':') {
        id.to_string()
    } else if let Some(table) = default_table {
        format!("{}:{}", table, id)
    } else {
        // Best-effort: assume character for bare IDs
        format!("character:{}", id)
    }
}

impl NarraServer {
    pub(crate) async fn handle_reranked_search(
        &self,
        query: &str,
        entity_types: Option<Vec<String>>,
        limit: Option<usize>,
    ) -> Result<QueryResponse, String> {
        use crate::services::{EntityType, SearchFilter};

        let entity_type_list: Vec<EntityType> = entity_types
            .unwrap_or_default()
            .iter()
            .filter_map(|t| match t.as_str() {
                "character" => Some(EntityType::Character),
                "location" => Some(EntityType::Location),
                "event" => Some(EntityType::Event),
                "scene" => Some(EntityType::Scene),
                "knowledge" => Some(EntityType::Knowledge),
                _ => None,
            })
            .collect();

        let limit = limit.unwrap_or(10).min(crate::mcp::types::MAX_LIMIT);
        let filter = SearchFilter {
            entity_types: entity_type_list,
            limit: Some(limit),
            ..Default::default()
        };

        let results = self
            .search_service
            .reranked_search(query, filter)
            .await
            .map_err(|e| format!("Re-ranked search failed: {}", e))?;

        let entity_results: Vec<EntityResult> = results
            .iter()
            .map(|r| EntityResult {
                id: r.id.clone(),
                entity_type: r.entity_type.clone(),
                name: r.name.clone(),
                content: format!("[{}] {} (score: {:.4})", r.entity_type, r.name, r.score),
                confidence: Some(r.score),
                last_modified: None,
            })
            .collect();

        let total = entity_results.len();
        let token_estimate = self.estimate_tokens_from_results(&entity_results);

        Ok(QueryResponse {
            results: entity_results,
            total,
            next_cursor: None,
            hints: vec![
                "Re-ranked results use cross-encoder scoring for better relevance".to_string(),
            ],
            token_estimate,
        })
    }

    pub(crate) async fn handle_growth_vector(
        &self,
        entity_id: &str,
        limit: Option<usize>,
    ) -> Result<QueryResponse, String> {
        let entity_id = normalize_id(entity_id, None);
        let limit = limit.unwrap_or(5).min(20);
        let service = VectorOpsService::new(self.db.clone());
        let result = service
            .growth_vector(&entity_id, limit)
            .await
            .map_err(|e| format!("Growth vector failed: {}", e))?;

        let json = serde_json::to_string_pretty(&result)
            .unwrap_or_else(|_| "Failed to serialize".to_string());

        let entity_result = EntityResult {
            id: result.entity_id.clone(),
            entity_type: "analysis".to_string(),
            name: format!("Growth Vector: {}", result.entity_name),
            content: json,
            confidence: None,
            last_modified: None,
        };

        Ok(QueryResponse {
            results: vec![entity_result],
            total: 1,
            next_cursor: None,
            hints: vec![format!(
                "Based on {} arc snapshots. Total drift: {:.4}",
                result.snapshot_count, result.total_drift
            )],
            token_estimate: 200,
        })
    }

    pub(crate) async fn handle_misperception_vector(
        &self,
        observer_id: &str,
        target_id: &str,
        limit: Option<usize>,
    ) -> Result<QueryResponse, String> {
        let observer_id = normalize_id(observer_id, Some("character"));
        let target_id = normalize_id(target_id, Some("character"));
        let limit = limit.unwrap_or(5).min(20);
        let service = VectorOpsService::new(self.db.clone());
        let result = service
            .misperception_vector(&observer_id, &target_id, limit)
            .await
            .map_err(|e| format!("Misperception vector failed: {}", e))?;

        let json = serde_json::to_string_pretty(&result)
            .unwrap_or_else(|_| "Failed to serialize".to_string());

        let entity_result = EntityResult {
            id: format!("{}_{}", observer_id, target_id),
            entity_type: "analysis".to_string(),
            name: format!(
                "Misperception: {} -> {}",
                result.observer_name, result.target_name
            ),
            content: json,
            confidence: Some(result.perception_gap),
            last_modified: None,
        };

        Ok(QueryResponse {
            results: vec![entity_result],
            total: 1,
            next_cursor: None,
            hints: vec![format!(
                "Perception gap: {:.4} (0=accurate, 1=maximally wrong)",
                result.perception_gap
            )],
            token_estimate: 200,
        })
    }

    pub(crate) async fn handle_convergence_analysis(
        &self,
        entity_a: &str,
        entity_b: &str,
        window: Option<usize>,
    ) -> Result<QueryResponse, String> {
        let entity_a = normalize_id(entity_a, None);
        let entity_b = normalize_id(entity_b, None);
        let service = VectorOpsService::new(self.db.clone());
        let result = service
            .convergence_analysis(&entity_a, &entity_b, window)
            .await
            .map_err(|e| format!("Convergence analysis failed: {}", e))?;

        let json = serde_json::to_string_pretty(&result)
            .unwrap_or_else(|_| "Failed to serialize".to_string());

        let trend_label = if result.convergence_rate > 0.001 {
            "converging"
        } else if result.convergence_rate < -0.001 {
            "diverging"
        } else {
            "stable"
        };

        let entity_result = EntityResult {
            id: format!("{}_{}", entity_a, entity_b),
            entity_type: "analysis".to_string(),
            name: format!(
                "Convergence: {} <-> {}",
                result.entity_a_name, result.entity_b_name
            ),
            content: json,
            confidence: Some(result.current_similarity),
            last_modified: None,
        };

        Ok(QueryResponse {
            results: vec![entity_result],
            total: 1,
            next_cursor: None,
            hints: vec![format!(
                "{} (rate: {:.6}, current similarity: {:.4})",
                trend_label, result.convergence_rate, result.current_similarity
            )],
            token_estimate: 200 + result.trend.len() * 20,
        })
    }

    pub(crate) async fn handle_semantic_midpoint(
        &self,
        entity_a: &str,
        entity_b: &str,
        limit: Option<usize>,
    ) -> Result<QueryResponse, String> {
        let entity_a = normalize_id(entity_a, None);
        let entity_b = normalize_id(entity_b, None);
        let limit = limit.unwrap_or(5).min(20);
        let service = VectorOpsService::new(self.db.clone());
        let result = service
            .semantic_midpoint(&entity_a, &entity_b, limit)
            .await
            .map_err(|e| format!("Semantic midpoint failed: {}", e))?;

        let json = serde_json::to_string_pretty(&result)
            .unwrap_or_else(|_| "Failed to serialize".to_string());

        let entity_result = EntityResult {
            id: format!("{}_{}", entity_a, entity_b),
            entity_type: "analysis".to_string(),
            name: format!("Semantic Midpoint: {} <-> {}", entity_a, entity_b),
            content: json,
            confidence: None,
            last_modified: None,
        };

        Ok(QueryResponse {
            results: vec![entity_result],
            total: 1,
            next_cursor: None,
            hints: vec![format!(
                "Found {} entities near the midpoint",
                result.neighbors.len()
            )],
            token_estimate: 100 + result.neighbors.len() * 30,
        })
    }

    pub(crate) async fn handle_faceted_search(
        &self,
        query: &str,
        facet: &str,
        limit: Option<usize>,
    ) -> Result<QueryResponse, String> {
        use crate::services::SearchFilter;

        let limit = limit.unwrap_or(10).min(crate::mcp::types::MAX_LIMIT);
        let filter = SearchFilter {
            limit: Some(limit),
            ..Default::default()
        };

        let results = self
            .search_service
            .faceted_search(query, facet, filter)
            .await
            .map_err(|e| format!("Faceted search failed: {}", e))?;

        let entity_results: Vec<EntityResult> = results
            .iter()
            .map(|r| EntityResult {
                id: r.id.clone(),
                entity_type: r.entity_type.clone(),
                name: r.name.clone(),
                content: format!(
                    "[{}] {} (facet: {}, score: {:.4})",
                    r.entity_type, r.name, facet, r.score
                ),
                confidence: Some(r.score),
                last_modified: None,
            })
            .collect();

        let total = entity_results.len();
        let token_estimate = self.estimate_tokens_from_results(&entity_results);

        Ok(QueryResponse {
            results: entity_results,
            total,
            next_cursor: None,
            hints: vec![format!(
                "Searched {} facet embeddings for: \"{}\"",
                facet, query
            )],
            token_estimate,
        })
    }

    pub(crate) async fn handle_multi_facet_search(
        &self,
        query: &str,
        weights: &std::collections::HashMap<String, f32>,
        limit: Option<usize>,
    ) -> Result<QueryResponse, String> {
        use crate::services::SearchFilter;

        let limit = limit.unwrap_or(10).min(crate::mcp::types::MAX_LIMIT);
        let filter = SearchFilter {
            limit: Some(limit),
            ..Default::default()
        };

        let results = self
            .search_service
            .multi_facet_search(query, weights, filter)
            .await
            .map_err(|e| format!("Multi-facet search failed: {}", e))?;

        let entity_results: Vec<EntityResult> = results
            .iter()
            .map(|r| EntityResult {
                id: r.id.clone(),
                entity_type: r.entity_type.clone(),
                name: r.name.clone(),
                content: format!(
                    "[{}] {} (weighted score: {:.4})",
                    r.entity_type, r.name, r.score
                ),
                confidence: Some(r.score),
                last_modified: None,
            })
            .collect();

        let total = entity_results.len();
        let token_estimate = self.estimate_tokens_from_results(&entity_results);

        let weights_str = weights
            .iter()
            .map(|(k, v)| format!("{}: {:.2}", k, v))
            .collect::<Vec<_>>()
            .join(", ");

        Ok(QueryResponse {
            results: entity_results,
            total,
            next_cursor: None,
            hints: vec![format!(
                "Multi-facet search with weights: {{ {} }} for: \"{}\"",
                weights_str, query
            )],
            token_estimate,
        })
    }

    pub(crate) async fn handle_character_facets(
        &self,
        character_id: &str,
    ) -> Result<QueryResponse, String> {
        let character_id = normalize_id(character_id, Some("character"));

        // Fetch character with all facet embeddings and composites
        #[derive(serde::Deserialize)]
        struct FacetData {
            identity_embedding: Option<Vec<f32>>,
            identity_composite: Option<String>,
            identity_stale: Option<bool>,
            psychology_embedding: Option<Vec<f32>>,
            psychology_composite: Option<String>,
            psychology_stale: Option<bool>,
            social_embedding: Option<Vec<f32>>,
            social_composite: Option<String>,
            social_stale: Option<bool>,
            narrative_embedding: Option<Vec<f32>>,
            narrative_composite: Option<String>,
            narrative_stale: Option<bool>,
        }

        let query = format!(
            "SELECT identity_embedding, identity_composite, identity_stale, \
             psychology_embedding, psychology_composite, psychology_stale, \
             social_embedding, social_composite, social_stale, \
             narrative_embedding, narrative_composite, narrative_stale FROM {}",
            character_id
        );

        let mut resp = self
            .db
            .query(&query)
            .await
            .map_err(|e| format!("Failed to fetch facets: {}", e))?;

        let facet_data: Option<FacetData> = resp
            .take(0)
            .map_err(|e| format!("Failed to parse facets: {}", e))?;

        let facet_data =
            facet_data.ok_or_else(|| format!("Character not found: {}", character_id))?;

        // Compute inter-facet similarities
        let facets = [
            ("identity", &facet_data.identity_embedding),
            ("psychology", &facet_data.psychology_embedding),
            ("social", &facet_data.social_embedding),
            ("narrative", &facet_data.narrative_embedding),
        ];

        let mut similarities = Vec::new();
        for i in 0..facets.len() {
            for j in (i + 1)..facets.len() {
                if let (Some(emb_a), Some(emb_b)) = (facets[i].1, facets[j].1) {
                    let sim = crate::utils::math::cosine_similarity(emb_a, emb_b);
                    similarities.push(format!("{} <-> {}: {:.4}", facets[i].0, facets[j].0, sim));
                }
            }
        }

        // Build status report
        let mut status_lines = Vec::new();
        status_lines.push(format!("Character Facets: {}\n", character_id));

        status_lines.push("IDENTITY:".to_string());
        status_lines.push(format!(
            "  Status: {}",
            if facet_data.identity_stale.unwrap_or(false) {
                "STALE"
            } else if facet_data.identity_embedding.is_some() {
                "OK"
            } else {
                "MISSING"
            }
        ));
        if let Some(composite) = &facet_data.identity_composite {
            status_lines.push(format!("  Text: {}", composite));
        }

        status_lines.push("\nPSYCHOLOGY:".to_string());
        status_lines.push(format!(
            "  Status: {}",
            if facet_data.psychology_stale.unwrap_or(false) {
                "STALE"
            } else if facet_data.psychology_embedding.is_some() {
                "OK"
            } else {
                "MISSING"
            }
        ));
        if let Some(composite) = &facet_data.psychology_composite {
            status_lines.push(format!("  Text: {}", composite));
        }

        status_lines.push("\nSOCIAL:".to_string());
        status_lines.push(format!(
            "  Status: {}",
            if facet_data.social_stale.unwrap_or(false) {
                "STALE"
            } else if facet_data.social_embedding.is_some() {
                "OK"
            } else {
                "MISSING"
            }
        ));
        if let Some(composite) = &facet_data.social_composite {
            status_lines.push(format!("  Text: {}", composite));
        }

        status_lines.push("\nNARRATIVE:".to_string());
        status_lines.push(format!(
            "  Status: {}",
            if facet_data.narrative_stale.unwrap_or(false) {
                "STALE"
            } else if facet_data.narrative_embedding.is_some() {
                "OK"
            } else {
                "MISSING"
            }
        ));
        if let Some(composite) = &facet_data.narrative_composite {
            status_lines.push(format!("  Text: {}", composite));
        }

        if !similarities.is_empty() {
            status_lines.push("\nINTER-FACET SIMILARITIES:".to_string());
            for sim in similarities {
                status_lines.push(format!("  {}", sim));
            }
        }

        let content = status_lines.join("\n");

        let entity_result = EntityResult {
            id: character_id.clone(),
            entity_type: "analysis".to_string(),
            name: format!("Facet Analysis: {}", character_id),
            content,
            confidence: None,
            last_modified: None,
        };

        Ok(QueryResponse {
            results: vec![entity_result],
            total: 1,
            next_cursor: None,
            hints: vec![
                "Shows status, composites, and inter-facet similarities for all character facets"
                    .to_string(),
            ],
            token_estimate: 300,
        })
    }
}
