use crate::mcp::NarraServer;
use crate::mcp::{EntityResult, QueryResponse, SearchMetadataFilter};
use crate::repository::RelationshipRepository;
use crate::services::{EntityType, SearchFilter};
use crate::utils::math::cosine_similarity;

use super::{parse_entity_types, parse_metadata_filter};

impl NarraServer {
    /// Unified search handler supporting semantic, hybrid, and reranked modes.
    pub(crate) async fn handle_unified_search(
        &self,
        query: &str,
        mode: &str,
        entity_types: Option<Vec<String>>,
        limit: usize,
        metadata_filter: Option<SearchMetadataFilter>,
        phase: Option<usize>,
    ) -> Result<QueryResponse, String> {
        // 1. Parse entity types (shared)
        let type_filter = parse_entity_types(entity_types);

        // 2. Parse metadata filters (shared)
        let metadata = metadata_filter
            .as_ref()
            .map(parse_metadata_filter)
            .unwrap_or_default();

        // 3. Build SearchFilter (shared)
        let filter = SearchFilter {
            entity_types: type_filter,
            limit: Some(limit),
            min_score: None,
            metadata,
        };

        // 4. Dispatch by mode
        let mut response: QueryResponse = (match mode {
            "semantic" => {
                if !self.embedding_service.is_available() {
                    return Err("Semantic search unavailable - embedding model not loaded. Use mode 'hybrid' instead.".to_string());
                }

                let results = self
                    .search_service
                    .semantic_search(query, filter)
                    .await
                    .map_err(|e| format!("Semantic search failed: {}", e))?;

                // Auto-rerank results using cross-encoder for better relevance
                let results = self
                    .search_service
                    .rerank_results(query, results)
                    .await
                    .map_err(|e| format!("Re-ranking failed: {}", e))?;

                let entity_results: Vec<EntityResult> = results
                    .iter()
                    .map(|r| EntityResult {
                        id: r.id.clone(),
                        entity_type: r.entity_type.clone(),
                        name: r.name.clone(),
                        content: r.name.clone(),
                        confidence: Some(r.score),
                        last_modified: None,
                    })
                    .collect();

                let mut hints = Vec::new();
                if entity_results.is_empty() {
                    hints.push("No semantic matches found. Try broader terms or run backfill_embeddings if entities lack embeddings.".to_string());
                } else {
                    hints.push(format!(
                        "Found {} semantically similar entities",
                        entity_results.len()
                    ));
                    hints.push("Use lookup for full details on any entity".to_string());
                }

                let token_estimate = self.estimate_tokens_from_results(&entity_results);

                Ok(QueryResponse {
                    results: entity_results,
                    total: results.len(),
                    next_cursor: None,
                    hints,
                    token_estimate,
                    truncated: None,
                })
            }
            "reranked" => {
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
                        "Re-ranked results use cross-encoder scoring for better relevance"
                            .to_string(),
                    ],
                    token_estimate,
                    truncated: None,
                })
            }
            _ => {
                // "hybrid" (default)
                let results = self
                    .search_service
                    .hybrid_search(query, filter)
                    .await
                    .map_err(|e| format!("Hybrid search failed: {}", e))?;

                let entity_results: Vec<EntityResult> = results
                    .iter()
                    .map(|r| EntityResult {
                        id: r.id.clone(),
                        entity_type: r.entity_type.clone(),
                        name: r.name.clone(),
                        content: r.name.clone(),
                        confidence: Some(r.score),
                        last_modified: None,
                    })
                    .collect();

                let mut hints = Vec::new();

                if !self.embedding_service.is_available() {
                    hints.push(
                        "Semantic component unavailable - showing keyword results only".to_string(),
                    );
                } else {
                    hints.push(
                        "Results combine keyword matching with semantic similarity".to_string(),
                    );
                }

                if entity_results.is_empty() {
                    hints.push(
                        "No matches found. Try a different query or broader terms.".to_string(),
                    );
                } else {
                    hints.push(format!(
                        "Found {} results from hybrid search",
                        entity_results.len()
                    ));
                    hints.push("Use lookup for full details on any entity".to_string());
                }

                let token_estimate = self.estimate_tokens_from_results(&entity_results);

                Ok(QueryResponse {
                    results: entity_results,
                    total: results.len(),
                    next_cursor: None,
                    hints,
                    token_estimate,
                    truncated: None,
                })
            }
        } as Result<QueryResponse, String>)?;

        // Apply phase filtering if requested
        if let Some(phase_id) = phase {
            use crate::services::TemporalService;

            let temporal_service = TemporalService::new(self.db.clone());
            match temporal_service
                .load_or_detect_phases(EntityType::embeddable(), None, None)
                .await
            {
                Ok(phase_result) => {
                    if let Some(target_phase) =
                        phase_result.phases.iter().find(|p| p.phase_id == phase_id)
                    {
                        let phase_entity_ids: std::collections::HashSet<String> = target_phase
                            .members
                            .iter()
                            .map(|m| m.entity_id.clone())
                            .collect();

                        response
                            .results
                            .retain(|r| phase_entity_ids.contains(&r.id));
                        response.total = response.results.len();
                        response.hints.push(format!(
                            "Filtered to phase {}: {}",
                            phase_id, target_phase.label
                        ));
                    } else {
                        let available: Vec<String> = phase_result
                            .phases
                            .iter()
                            .map(|p| format!("{}: {}", p.phase_id, p.label))
                            .collect();
                        response.hints.push(format!(
                            "Phase {} not found. Available: {}",
                            phase_id,
                            available.join(", ")
                        ));
                    }
                }
                Err(e) => {
                    response.hints.push(format!(
                        "Phase filtering skipped (phase detection failed: {})",
                        e
                    ));
                }
            }
        }

        Ok(response)
    }

    pub(crate) async fn handle_reverse_query(
        &self,
        entity_id: &str,
        referencing_types: Option<Vec<String>>,
        limit: usize,
    ) -> Result<QueryResponse, String> {
        use crate::services::graph::MermaidGraphService;

        // Create graph service instance
        let graph_service = MermaidGraphService::new(self.db.clone());

        // Call get_referencing_entities
        let results = graph_service
            .get_referencing_entities(entity_id, referencing_types, limit)
            .await
            .map_err(|e| format!("Reverse query failed: {}", e))?;

        // Convert to EntityResults, fetch details in parallel
        let futures: Vec<_> = results
            .iter()
            .map(|result| {
                let summary_service = self.summary_service.clone();
                let entity_id_str = result.entity_id.clone();
                let entity_type = result.entity_type.clone();
                let reference_field = result.reference_field.clone();
                let target_entity_id = entity_id.to_string();
                async move {
                    let summary = summary_service
                        .get_entity_content(
                            &entity_id_str,
                            crate::services::summary::DetailLevel::Minimal,
                        )
                        .await
                        .map_err(|e| format!("Failed to fetch entity: {}", e))?;

                    Ok::<_, String>(summary.map(|s| EntityResult {
                        id: entity_id_str,
                        entity_type,
                        name: s.name.clone(),
                        content: format!("References {} via {}", target_entity_id, reference_field),
                        confidence: Some(1.0),
                        last_modified: None,
                    }))
                }
            })
            .collect();

        let summaries = futures::future::join_all(futures).await;
        let mut entity_results = Vec::new();
        for result in summaries {
            if let Ok(Some(entity_result)) = result {
                entity_results.push(entity_result);
            }
        }

        let hints = vec![
            format!(
                "Found {} entities referencing {}",
                entity_results.len(),
                entity_id
            ),
            "Use lookup on specific entities for full details".to_string(),
        ];

        let token_estimate = self.estimate_tokens_from_results(&entity_results);

        Ok(QueryResponse {
            results: entity_results,
            total: results.len(),
            next_cursor: None,
            hints,
            token_estimate,
            truncated: None,
        })
    }

    pub(crate) async fn handle_semantic_join(
        &self,
        query: &str,
        entity_types: Option<Vec<String>>,
        limit: usize,
    ) -> Result<QueryResponse, String> {
        // Check if embedding service is available
        if !self.embedding_service.is_available() {
            return Err("Semantic join unavailable - embedding model not loaded. Run backfill_embeddings first.".to_string());
        }

        // Convert entity_types using shared helper, default to all embeddable
        let type_filter = {
            let parsed = parse_entity_types(entity_types);
            if parsed.is_empty() {
                EntityType::embeddable()
            } else {
                parsed
            }
        };

        // Embed the query
        let query_vector = self
            .embedding_service
            .embed_text(query)
            .await
            .map_err(|e| format!("Failed to embed query: {}", e))?;

        // Perform cross-table vector search in parallel via try_join_all
        #[derive(serde::Deserialize)]
        struct SearchResultInternal {
            id: surrealdb::sql::Thing,
            entity_type: String,
            name: String,
            score: f32,
        }

        let k = limit * 2;
        let futures: Vec<_> = type_filter
            .iter()
            .map(|entity_type| {
                let db = self.db.clone();
                let qv = query_vector.clone();
                let (table, name_field) = match entity_type {
                    EntityType::Character => ("character", "name"),
                    EntityType::Location => ("location", "name"),
                    EntityType::Event => ("event", "title"),
                    EntityType::Scene => ("scene", "title"),
                    EntityType::Knowledge => ("knowledge", "fact"),
                    EntityType::Note => ("note", "title"),
                    EntityType::Fact => ("fact", "title"),
                };
                let table = table.to_string();
                let name_field = name_field.to_string();
                async move {
                    let query_str = format!(
                        r#"SELECT id, '{table}' AS entity_type, {name_field} AS name,
                                  vector::similarity::cosine(embedding, $query_vector) AS score
                           FROM {table}
                           WHERE embedding IS NOT NONE
                           ORDER BY score DESC
                           LIMIT {k}"#,
                        table = table,
                        name_field = name_field,
                        k = k,
                    );

                    let mut response = db
                        .query(&query_str)
                        .bind(("query_vector", qv))
                        .await
                        .map_err(|e| format!("Semantic join query failed: {}", e))?;

                    let table_results: Vec<SearchResultInternal> = response
                        .take(0)
                        .map_err(|e| format!("Failed to parse results: {}", e))?;

                    Ok::<_, String>(table_results)
                }
            })
            .collect();

        let table_results = futures::future::try_join_all(futures).await?;
        let mut all_results: Vec<SearchResultInternal> =
            table_results.into_iter().flatten().collect();

        // Sort all results by similarity score descending
        all_results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Truncate to limit
        all_results.truncate(limit);

        // Convert to EntityResults
        let entity_results: Vec<EntityResult> = all_results
            .iter()
            .map(|r| EntityResult {
                id: r.id.to_string(),
                entity_type: r.entity_type.clone(),
                name: r.name.clone(),
                content: r.name.clone(),
                confidence: Some(r.score),
                last_modified: None,
            })
            .collect();

        let hints = vec![
            "Cross-field semantic join: character embeddings encode profile sections, roles, relationships".to_string(),
            format!("Searched across {} entity types simultaneously", type_filter.len()),
            "Results ranked by semantic similarity across all types".to_string(),
        ];

        let token_estimate = self.estimate_tokens_from_results(&entity_results);

        Ok(QueryResponse {
            results: entity_results,
            total: all_results.len(),
            next_cursor: None,
            hints,
            token_estimate,
            truncated: None,
        })
    }

    pub(crate) async fn handle_semantic_knowledge(
        &self,
        query: &str,
        character_id: Option<String>,
        limit: usize,
    ) -> Result<QueryResponse, String> {
        // Check if embedding service is available
        if !self.embedding_service.is_available() {
            return Err("Semantic knowledge search unavailable - embedding model not loaded. Run backfill_embeddings first.".to_string());
        }

        // Generate query embedding
        let query_vector = self
            .embedding_service
            .embed_text(query)
            .await
            .map_err(|e| format!("Failed to embed query: {}", e))?;

        // Build SurrealQL with optional character filter.
        let query_str = if let Some(ref char_id) = character_id {
            let char_key = char_id.split(':').nth(1).unwrap_or(char_id);
            format!(
                r#"SELECT id, 'knowledge' AS entity_type, fact AS name,
                          vector::similarity::cosine(embedding, $query_vector) AS score,
                          character.name AS character_name
                   FROM knowledge
                   WHERE character = character:{char_key}
                     AND embedding IS NOT NONE
                   ORDER BY score DESC
                   LIMIT {limit}"#,
                char_key = char_key,
                limit = limit * 2,
            )
        } else {
            format!(
                r#"SELECT id, 'knowledge' AS entity_type, fact AS name,
                          vector::similarity::cosine(embedding, $query_vector) AS score,
                          character.name AS character_name
                   FROM knowledge
                   WHERE embedding IS NOT NONE
                   ORDER BY score DESC
                   LIMIT {limit}"#,
                limit = limit * 2,
            )
        };

        let mut response = self
            .db
            .query(&query_str)
            .bind(("query_vector", query_vector))
            .await
            .map_err(|e| format!("Semantic knowledge query failed: {}", e))?;

        #[derive(serde::Deserialize)]
        struct KnowledgeSearchResult {
            id: surrealdb::sql::Thing,
            entity_type: String,
            name: String,
            score: f32,
            character_name: Option<String>,
        }

        let results: Vec<KnowledgeSearchResult> = response
            .take(0)
            .map_err(|e| format!("Failed to parse results: {}", e))?;

        // Convert to EntityResults, truncate to limit
        let entity_results: Vec<EntityResult> = results
            .into_iter()
            .take(limit)
            .map(|r| {
                let known_by = r
                    .character_name
                    .map(|n| format!(" (known by {})", n))
                    .unwrap_or_default();

                EntityResult {
                    id: r.id.to_string(),
                    entity_type: r.entity_type,
                    name: r.name.clone(),
                    content: format!("{}{}", r.name, known_by),
                    confidence: Some(r.score),
                    last_modified: None,
                }
            })
            .collect();

        let mut hints = Vec::new();
        if entity_results.is_empty() {
            hints.push("No knowledge matches found. Try broader terms or run backfill_embeddings for knowledge entities.".to_string());
        } else {
            hints.push(format!(
                "Found {} semantically similar knowledge entries",
                entity_results.len()
            ));
            if character_id.is_some() {
                hints.push("Filtered to specific character's knowledge".to_string());
            }
            hints.push("Use lookup for full details on any entity".to_string());
        }

        let total = entity_results.len();
        let token_estimate = self.estimate_tokens_from_results(&entity_results);

        Ok(QueryResponse {
            results: entity_results,
            total,
            next_cursor: None,
            hints,
            token_estimate,
            truncated: None,
        })
    }

    pub(crate) async fn handle_semantic_graph_search(
        &self,
        entity_id: &str,
        max_hops: usize,
        query: &str,
        entity_types: Option<Vec<String>>,
        limit: usize,
    ) -> Result<QueryResponse, String> {
        // Check if embedding service is available
        if !self.embedding_service.is_available() {
            return Err(
                "Semantic graph search unavailable - embedding model not loaded.".to_string(),
            );
        }

        // Phase 1: Graph traversal — get candidate IDs within max_hops
        let connected = self
            .relationship_repo
            .get_connected_entities(entity_id, max_hops)
            .await
            .map_err(|e| format!("Graph traversal failed: {}", e))?;

        if connected.is_empty() {
            return Ok(QueryResponse {
                results: vec![],
                total: 0,
                next_cursor: None,
                hints: vec![
                    format!(
                        "No connected entities found within {} hops of {}",
                        max_hops, entity_id
                    ),
                    "Try increasing max_hops or check entity relationships".to_string(),
                ],
                token_estimate: 0,
                truncated: None,
            });
        }

        // Optional type filter
        let type_filter: Option<Vec<String>> =
            entity_types.map(|types| types.into_iter().map(|t| t.to_lowercase()).collect());

        // Phase 2: Embed query
        let query_vector = self
            .embedding_service
            .embed_text(query)
            .await
            .map_err(|e| format!("Failed to embed query: {}", e))?;

        // Phase 3: Batch fetch embeddings per table, then rank by cosine similarity
        // Group candidates by table
        let mut by_table: std::collections::HashMap<String, Vec<String>> =
            std::collections::HashMap::new();
        for id in &connected {
            let table = id.split(':').next().unwrap_or("unknown").to_string();

            // Apply type filter if specified
            if let Some(ref filter) = type_filter {
                if !filter.contains(&table) {
                    continue;
                }
            }

            by_table.entry(table).or_default().push(id.clone());
        }

        // Fetch embeddings per table in parallel
        let futures: Vec<_> = by_table
            .into_iter()
            .map(|(table, ids)| {
                let db = self.db.clone();
                let qv = query_vector.clone();
                async move {
                    let name_field = match table.as_str() {
                        "event" | "scene" | "note" | "fact" => "title",
                        "knowledge" => "fact",
                        _ => "name",
                    };

                    let id_csv: String =
                        ids.iter().map(|id| id.to_string()).collect::<Vec<_>>().join(", ");

                    let query_str = format!(
                        "SELECT id, '{table}' AS entity_type, {name_field} AS name, embedding FROM {table} WHERE id IN [{ids}] AND embedding IS NOT NONE",
                        table = table,
                        name_field = name_field,
                        ids = id_csv,
                    );

                    let mut response = db
                        .query(&query_str)
                        .await
                        .map_err(|e| format!("Failed to fetch embeddings: {}", e))?;

                    #[derive(serde::Deserialize)]
                    struct EntityWithEmbedding {
                        id: surrealdb::sql::Thing,
                        entity_type: String,
                        name: String,
                        embedding: Vec<f32>,
                    }

                    let entities: Vec<EntityWithEmbedding> =
                        response.take(0).unwrap_or_default();

                    let scored: Vec<(String, String, String, f32)> = entities
                        .into_iter()
                        .map(|entity| {
                            let similarity = cosine_similarity(&entity.embedding, &qv);
                            (
                                entity.id.to_string(),
                                entity.entity_type,
                                entity.name,
                                similarity,
                            )
                        })
                        .collect();

                    Ok::<_, String>(scored)
                }
            })
            .collect();

        let table_results = futures::future::try_join_all(futures).await?;
        let mut scored_results: Vec<(String, String, String, f32)> =
            table_results.into_iter().flatten().collect();

        // Sort by similarity descending
        scored_results.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));
        scored_results.truncate(limit);

        // Convert to EntityResults
        let entity_results: Vec<EntityResult> = scored_results
            .iter()
            .map(|(id, entity_type, name, score)| EntityResult {
                id: id.clone(),
                entity_type: entity_type.clone(),
                name: name.clone(),
                content: name.clone(),
                confidence: Some(*score),
                last_modified: None,
            })
            .collect();

        let hints = vec![
            format!(
                "Found {} entities connected to {} (within {} hops) matching \"{}\"",
                entity_results.len(),
                entity_id,
                max_hops,
                query
            ),
            format!("{} total candidates from graph traversal", connected.len()),
            "Results ranked by semantic similarity to query".to_string(),
        ];

        let total = entity_results.len();
        let token_estimate = self.estimate_tokens_from_results(&entity_results);

        Ok(QueryResponse {
            results: entity_results,
            total,
            next_cursor: None,
            hints,
            token_estimate,
            truncated: None,
        })
    }
}
