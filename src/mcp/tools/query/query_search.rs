use crate::mcp::NarraServer;
use crate::mcp::{EntityResult, QueryResponse};
use crate::repository::RelationshipRepository;
use crate::services::{EntityType, SearchFilter};
use crate::utils::math::cosine_similarity;

use super::parse_metadata_filter;

impl NarraServer {
    pub(crate) async fn handle_semantic_search(
        &self,
        query: &str,
        entity_types: Option<Vec<String>>,
        limit: usize,
        metadata_filter: Option<serde_json::Value>,
    ) -> Result<QueryResponse, String> {
        // Check if embedding service is available
        if !self.embedding_service.is_available() {
            return Err("Semantic search unavailable - embedding model not loaded. Use regular search instead.".to_string());
        }

        // Convert entity_types strings to EntityType enum, filtering to only embeddable types
        let type_filter = if let Some(types) = entity_types {
            let mut embeddable_types = Vec::new();
            let mut non_embeddable = Vec::new();

            for t in types {
                match t.to_lowercase().as_str() {
                    "character" => embeddable_types.push(EntityType::Character),
                    "location" => embeddable_types.push(EntityType::Location),
                    "event" => embeddable_types.push(EntityType::Event),
                    "scene" => embeddable_types.push(EntityType::Scene),
                    "knowledge" => embeddable_types.push(EntityType::Knowledge),
                    "note" => non_embeddable.push("note"),
                    _ => {}
                }
            }

            // If user requested non-embeddable types, we'll add a hint later
            if !non_embeddable.is_empty() {
                tracing::debug!("Filtered out non-embeddable types: {:?}", non_embeddable);
            }

            embeddable_types
        } else {
            vec![]
        };

        // Parse metadata filters
        let metadata = metadata_filter
            .as_ref()
            .map(parse_metadata_filter)
            .unwrap_or_default();

        // Build SearchFilter with entity_types, limit, and metadata
        let filter = SearchFilter {
            entity_types: type_filter,
            limit: Some(limit),
            min_score: None,
            metadata,
        };

        // Call semantic_search
        let results = self
            .search_service
            .semantic_search(query, filter)
            .await
            .map_err(|e| format!("Semantic search failed: {}", e))?;

        // Convert SearchResults to EntityResults
        let entity_results: Vec<EntityResult> = results
            .iter()
            .map(|r| EntityResult {
                id: r.id.clone(),
                entity_type: r.entity_type.clone(),
                name: r.name.clone(),
                content: r.name.clone(), // SearchResult doesn't have snippet, use name
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
        })
    }

    pub(crate) async fn handle_hybrid_search(
        &self,
        query: &str,
        entity_types: Option<Vec<String>>,
        limit: usize,
        metadata_filter: Option<serde_json::Value>,
    ) -> Result<QueryResponse, String> {
        // Convert entity_types strings to EntityType enum, filtering to only embeddable types
        let type_filter = if let Some(types) = entity_types {
            let mut embeddable_types = Vec::new();

            for t in types {
                match t.to_lowercase().as_str() {
                    "character" => embeddable_types.push(EntityType::Character),
                    "location" => embeddable_types.push(EntityType::Location),
                    "event" => embeddable_types.push(EntityType::Event),
                    "scene" => embeddable_types.push(EntityType::Scene),
                    "knowledge" => embeddable_types.push(EntityType::Knowledge),
                    _ => {}
                }
            }

            embeddable_types
        } else {
            vec![]
        };

        // Parse metadata filters
        let metadata = metadata_filter
            .as_ref()
            .map(parse_metadata_filter)
            .unwrap_or_default();

        // Build SearchFilter with entity_types, limit, and metadata
        let filter = SearchFilter {
            entity_types: type_filter,
            limit: Some(limit),
            min_score: None,
            metadata,
        };

        // Call hybrid_search (already handles graceful degradation)
        let results = self
            .search_service
            .hybrid_search(query, filter)
            .await
            .map_err(|e| format!("Hybrid search failed: {}", e))?;

        // Convert SearchResults to EntityResults
        let entity_results: Vec<EntityResult> = results
            .iter()
            .map(|r| EntityResult {
                id: r.id.clone(),
                entity_type: r.entity_type.clone(),
                name: r.name.clone(),
                content: r.name.clone(), // SearchResult doesn't have snippet, use name
                confidence: Some(r.score),
                last_modified: None,
            })
            .collect();

        let mut hints = Vec::new();

        // Check if embedding service is available for better hints
        if !self.embedding_service.is_available() {
            hints.push("Semantic component unavailable - showing keyword results only".to_string());
        } else {
            hints.push("Results combine keyword matching with semantic similarity".to_string());
        }

        if entity_results.is_empty() {
            hints.push("No matches found. Try a different query or broader terms.".to_string());
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
        })
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

        // Convert to EntityResults, fetch details for each
        let mut entity_results = Vec::new();
        for result in &results {
            // Use existing summary service to get entity details
            let summary = self
                .summary_service
                .get_entity_content(
                    &result.entity_id,
                    crate::services::summary::DetailLevel::Minimal,
                )
                .await
                .map_err(|e| format!("Failed to fetch entity: {}", e))?;

            if let Some(summary) = summary {
                entity_results.push(EntityResult {
                    id: result.entity_id.clone(),
                    entity_type: result.entity_type.clone(),
                    name: summary.name.clone(),
                    content: format!("References {} via {}", entity_id, result.reference_field),
                    confidence: Some(1.0),
                    last_modified: None,
                });
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

        // Convert entity_types strings to EntityType enum, filtering to only embeddable types
        let type_filter = if let Some(types) = entity_types {
            types
                .iter()
                .filter_map(|t| match t.to_lowercase().as_str() {
                    "character" => Some(EntityType::Character),
                    "location" => Some(EntityType::Location),
                    "event" => Some(EntityType::Event),
                    "scene" => Some(EntityType::Scene),
                    "knowledge" => Some(EntityType::Knowledge),
                    _ => None,
                })
                .collect()
        } else {
            // Default to all embeddable types
            EntityType::embeddable()
        };

        // Embed the query
        let query_vector = self
            .embedding_service
            .embed_text(query)
            .await
            .map_err(|e| format!("Failed to embed query: {}", e))?;

        // Perform cross-table vector search via HNSW index
        let mut all_results = Vec::new();

        for entity_type in &type_filter {
            let (table, name_field) = match entity_type {
                EntityType::Character => ("character", "name"),
                EntityType::Location => ("location", "name"),
                EntityType::Event => ("event", "title"),
                EntityType::Scene => ("scene", "title"),
                EntityType::Knowledge => ("knowledge", "fact"),
                _ => continue, // Skip non-embeddable types
            };

            // Query this table for vector matches using brute-force cosine similarity.
            // HNSW KNN (<|K|>) doesn't work reliably in SurrealDB 2.6.0 embedded RocksDB mode.
            let k = limit * 2; // Fetch more to allow combining across types
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

            let mut response = self
                .db
                .query(&query_str)
                .bind(("query_vector", query_vector.clone()))
                .await
                .map_err(|e| format!("Semantic join query failed: {}", e))?;

            #[derive(serde::Deserialize)]
            struct SearchResultInternal {
                id: surrealdb::sql::Thing,
                entity_type: String,
                name: String,
                score: f32,
            }

            let table_results: Vec<SearchResultInternal> = response
                .take(0)
                .map_err(|e| format!("Failed to parse results: {}", e))?;

            all_results.extend(table_results);
        }

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
                content: r.name.clone(), // Minimal content for cross-type search
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
        // Uses brute-force cosine similarity — HNSW KNN (<|K|>) doesn't work
        // reliably in SurrealDB 2.6.0 embedded RocksDB mode.
        let query_str = if let Some(ref char_id) = character_id {
            // Extract character key
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

        // Fetch embeddings and compute similarities
        let mut scored_results: Vec<(String, String, String, f32)> = Vec::new(); // (id, entity_type, name, score)

        for (table, ids) in &by_table {
            let name_field = match table.as_str() {
                "event" | "scene" => "title",
                "knowledge" => "fact",
                _ => "name",
            };

            // Build IN list for batch query
            let id_list: Vec<String> = ids.iter().map(|id| id.to_string()).collect();
            let id_csv = id_list.join(", ");

            let query_str = format!(
                "SELECT id, '{table}' AS entity_type, {name_field} AS name, embedding FROM {table} WHERE id IN [{ids}] AND embedding IS NOT NONE",
                table = table,
                name_field = name_field,
                ids = id_csv,
            );

            let mut response = self
                .db
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

            let entities: Vec<EntityWithEmbedding> = response.take(0).unwrap_or_default();

            for entity in entities {
                let similarity = cosine_similarity(&entity.embedding, &query_vector);
                scored_results.push((
                    entity.id.to_string(),
                    entity.entity_type,
                    entity.name,
                    similarity,
                ));
            }
        }

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
        })
    }
}
