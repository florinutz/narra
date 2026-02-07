use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use surrealdb::engine::local::Db;
use surrealdb::Surreal;

use crate::embedding::EmbeddingService;
use crate::NarraError;

/// Entity types for search filtering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EntityType {
    Character,
    Location,
    Event,
    Scene,
    Knowledge,
    Note,
}

impl EntityType {
    /// Get the database table name for this entity type.
    pub fn table_name(&self) -> &'static str {
        match self {
            EntityType::Character => "character",
            EntityType::Location => "location",
            EntityType::Event => "event",
            EntityType::Scene => "scene",
            EntityType::Knowledge => "knowledge",
            EntityType::Note => "note",
        }
    }

    /// Get all entity types.
    pub fn all() -> Vec<EntityType> {
        vec![
            EntityType::Character,
            EntityType::Location,
            EntityType::Event,
            EntityType::Scene,
            EntityType::Knowledge,
            EntityType::Note,
        ]
    }

    /// Get entity types that support embeddings.
    pub fn embeddable() -> Vec<EntityType> {
        vec![
            EntityType::Character,
            EntityType::Location,
            EntityType::Event,
            EntityType::Scene,
            EntityType::Knowledge,
        ]
    }

    /// Check if this entity type supports embeddings.
    pub fn has_embeddings(&self) -> bool {
        matches!(
            self,
            EntityType::Character
                | EntityType::Location
                | EntityType::Event
                | EntityType::Scene
                | EntityType::Knowledge
        )
    }
}

/// Metadata filter operation.
#[derive(Debug, Clone)]
pub enum FilterOp {
    /// field = $param
    Eq,
    /// $param IN field (for arrays like roles)
    Contains,
    /// field >= $param
    Gte,
    /// field <= $param
    Lte,
}

/// A single metadata filter condition.
#[derive(Debug, Clone)]
pub struct MetadataFilter {
    pub field: String,
    pub op: FilterOp,
    pub bind_key: String,
    pub value: serde_json::Value,
}

/// Search filter options.
#[derive(Debug, Clone, Default)]
pub struct SearchFilter {
    /// Filter by entity types (empty = all types)
    pub entity_types: Vec<EntityType>,
    /// Maximum results per entity type
    pub limit: Option<usize>,
    /// Minimum relevance score (0.0 to 1.0)
    pub min_score: Option<f32>,
    /// Metadata filters (applied as WHERE clauses)
    pub metadata: Vec<MetadataFilter>,
}

/// Internal search result from database (with RecordId).
#[derive(Debug, Clone, Deserialize)]
struct SearchResultInternal {
    /// Entity ID as RecordId from SurrealDB
    id: surrealdb::RecordId,
    /// Entity type
    entity_type: String,
    /// Display name or title
    name: String,
    /// Relevance score from BM25
    score: f32,
}

/// A search result with relevance score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Entity ID (e.g., "character:alice")
    pub id: String,
    /// Entity type
    pub entity_type: String,
    /// Display name or title
    pub name: String,
    /// Relevance score from BM25
    pub score: f32,
}

impl From<SearchResultInternal> for SearchResult {
    fn from(internal: SearchResultInternal) -> Self {
        Self {
            id: internal.id.to_string(),
            entity_type: internal.entity_type,
            name: internal.name,
            // BM25 scores can be negative; normalize to positive for API consumers
            score: internal.score.abs(),
        }
    }
}

/// Search service trait for finding entities by text.
#[async_trait]
pub trait SearchService: Send + Sync {
    /// Search entities by text query.
    ///
    /// Uses full-text search with BM25 ranking. Searches name/title fields
    /// with optional filtering by entity type.
    async fn search(
        &self,
        query: &str,
        filter: SearchFilter,
    ) -> Result<Vec<SearchResult>, NarraError>;

    /// Fuzzy search with typo tolerance.
    ///
    /// Uses rapidfuzz for similarity matching. Returns entities where
    /// name similarity exceeds threshold (0.0 to 1.0, default 0.7).
    async fn fuzzy_search(
        &self,
        query: &str,
        threshold: f64,
        filter: SearchFilter,
    ) -> Result<Vec<SearchResult>, NarraError>;

    /// Semantic vector search - find entities by meaning.
    ///
    /// Returns entities ranked by semantic similarity to query.
    /// Falls back gracefully if embeddings unavailable.
    async fn semantic_search(
        &self,
        query: &str,
        filter: SearchFilter,
    ) -> Result<Vec<SearchResult>, NarraError>;

    /// Hybrid search combining BM25 keyword + semantic vector results.
    ///
    /// Uses reciprocal rank fusion (RRF) for result merging.
    /// Falls back to keyword-only if embeddings unavailable.
    async fn hybrid_search(
        &self,
        query: &str,
        filter: SearchFilter,
    ) -> Result<Vec<SearchResult>, NarraError>;
}

/// SurrealDB implementation of SearchService.
pub struct SurrealSearchService {
    db: Arc<Surreal<Db>>,
    embedding_service: Arc<dyn EmbeddingService + Send + Sync>,
}

/// Build SQL WHERE clause fragment and bindings from metadata filters for a given entity type.
///
/// Uses a hardcoded whitelist of allowed fields per entity type to prevent injection.
/// Unknown fields or fields that don't apply to the current entity type are silently ignored.
fn build_filter_clause(
    entity_type: EntityType,
    metadata: &[MetadataFilter],
) -> (String, Vec<(String, serde_json::Value)>) {
    let mut clauses = Vec::new();
    let mut bindings = Vec::new();

    for filter in metadata {
        let allowed = matches!(
            (entity_type, filter.field.as_str()),
            (EntityType::Character, "roles")
                | (EntityType::Character, "name")
                | (EntityType::Event, "sequence_min")
                | (EntityType::Event, "sequence_max")
                | (EntityType::Location, "loc_type")
        );

        if !allowed {
            continue;
        }

        let clause = match (filter.field.as_str(), &filter.op) {
            ("roles", FilterOp::Contains) => {
                format!(" AND ${} IN roles", filter.bind_key)
            }
            ("name", FilterOp::Eq) => {
                format!(
                    " AND string::lowercase(name) CONTAINS string::lowercase(${})",
                    filter.bind_key
                )
            }
            ("sequence_min", FilterOp::Gte) => {
                format!(" AND sequence >= ${}", filter.bind_key)
            }
            ("sequence_max", FilterOp::Lte) => {
                format!(" AND sequence <= ${}", filter.bind_key)
            }
            ("loc_type", FilterOp::Eq) => {
                format!(" AND loc_type = ${}", filter.bind_key)
            }
            _ => continue,
        };

        clauses.push(clause);
        bindings.push((filter.bind_key.clone(), filter.value.clone()));
    }

    (clauses.join(""), bindings)
}

/// Standard RRF constant (k=60 is conventional).
const RRF_K: f32 = 60.0;

/// Apply Reciprocal Rank Fusion to merge two ranked result lists.
///
/// Each result's score becomes `sum(1 / (k + rank))` across lists where it appears.
/// Results are returned sorted by RRF score descending, then by id ascending for stability.
pub fn apply_rrf(
    keyword_results: &[SearchResult],
    semantic_results: &[SearchResult],
) -> Vec<SearchResult> {
    let mut rrf_scores: std::collections::HashMap<String, (SearchResult, f32)> =
        std::collections::HashMap::new();

    for (rank, result) in keyword_results.iter().enumerate() {
        let rrf_contribution = 1.0 / (RRF_K + (rank + 1) as f32);
        rrf_scores
            .entry(result.id.clone())
            .or_insert_with(|| (result.clone(), 0.0))
            .1 += rrf_contribution;
    }

    for (rank, result) in semantic_results.iter().enumerate() {
        let rrf_contribution = 1.0 / (RRF_K + (rank + 1) as f32);
        rrf_scores
            .entry(result.id.clone())
            .or_insert_with(|| (result.clone(), 0.0))
            .1 += rrf_contribution;
    }

    let mut merged: Vec<SearchResult> = rrf_scores
        .into_values()
        .map(|(mut result, rrf_score)| {
            result.score = rrf_score;
            result
        })
        .collect();

    merged.sort_by(|a, b| match b.score.partial_cmp(&a.score) {
        Some(std::cmp::Ordering::Equal) | None => a.id.cmp(&b.id),
        Some(ordering) => ordering,
    });

    merged
}

impl SurrealSearchService {
    pub fn new(
        db: Arc<Surreal<Db>>,
        embedding_service: Arc<dyn EmbeddingService + Send + Sync>,
    ) -> Self {
        Self {
            db,
            embedding_service,
        }
    }

    /// Build a full-text search query for a specific entity type.
    fn build_search_query(entity_type: EntityType, limit: usize) -> String {
        let table = entity_type.table_name();
        let name_field = match entity_type {
            EntityType::Event | EntityType::Scene => "title",
            EntityType::Knowledge => "fact",
            EntityType::Note => "title", // Notes use title as display name
            _ => "name",
        };

        // Return id as RecordId (will be converted to String in From impl)
        // ORDER BY score DESC, id ASC ensures stable pagination order
        format!(
            r#"SELECT id, '{table}' AS entity_type, {name_field} AS name, search::score(1) AS score
               FROM {table}
               WHERE {name_field} @1@ $query
               ORDER BY score DESC, id ASC
               LIMIT {limit}"#,
            table = table,
            name_field = name_field,
            limit = limit
        )
    }

    /// Build full-text search queries for notes (searches both title and body).
    /// Returns two queries: one for title, one for body.
    fn build_note_search_queries(limit: usize) -> (String, String) {
        let title_query = format!(
            r#"SELECT id, 'note' AS entity_type, title AS name, search::score(1) AS score
               FROM note
               WHERE title @1@ $query
               ORDER BY score DESC, id ASC
               LIMIT {limit}"#,
            limit = limit
        );

        let body_query = format!(
            r#"SELECT id, 'note' AS entity_type, title AS name, search::score(1) AS score
               FROM note
               WHERE body @1@ $query
               ORDER BY score DESC, id ASC
               LIMIT {limit}"#,
            limit = limit
        );

        (title_query, body_query)
    }
}

#[async_trait]
impl SearchService for SurrealSearchService {
    async fn search(
        &self,
        query: &str,
        filter: SearchFilter,
    ) -> Result<Vec<SearchResult>, NarraError> {
        let entity_types = if filter.entity_types.is_empty() {
            EntityType::all()
        } else {
            filter.entity_types
        };

        let limit = filter.limit.unwrap_or(20);
        let min_score = filter.min_score.unwrap_or(0.0);

        // Launch all entity type queries in parallel
        let futures: Vec<_> = entity_types
            .iter()
            .map(|&entity_type| {
                let db = self.db.clone();
                let query_text = query.to_string();
                async move {
                    if entity_type == EntityType::Note {
                        let (title_query, body_query) = Self::build_note_search_queries(limit);

                        let mut title_response = db
                            .query(&title_query)
                            .bind(("query", query_text.clone()))
                            .await?;
                        let title_results: Vec<SearchResultInternal> = title_response.take(0)?;

                        let mut body_response =
                            db.query(&body_query).bind(("query", query_text)).await?;
                        let body_results: Vec<SearchResultInternal> = body_response.take(0)?;

                        // Deduplicate notes by ID, title results take priority
                        let mut seen = std::collections::HashSet::new();
                        let mut results = Vec::new();
                        for r in title_results.into_iter().chain(body_results) {
                            let id_str = r.id.to_string();
                            if seen.insert(id_str) {
                                results.push(SearchResult::from(r));
                            }
                        }
                        Ok::<_, NarraError>(results)
                    } else {
                        let query_str = Self::build_search_query(entity_type, limit);
                        let mut response = db.query(&query_str).bind(("query", query_text)).await?;
                        let internal: Vec<SearchResultInternal> = response.take(0)?;
                        Ok(internal.into_iter().map(SearchResult::from).collect())
                    }
                }
            })
            .collect();

        let results_per_type = futures::future::join_all(futures).await;

        let mut all_results = Vec::new();
        for type_results in results_per_type {
            for result in type_results? {
                if result.score >= min_score {
                    all_results.push(result);
                }
            }
        }

        // Sort by score descending, then by id ascending for stable pagination
        all_results.sort_by(|a, b| match b.score.partial_cmp(&a.score) {
            Some(std::cmp::Ordering::Equal) | None => a.id.cmp(&b.id),
            Some(ordering) => ordering,
        });

        // Apply overall limit
        if all_results.len() > limit {
            all_results.truncate(limit);
        }

        Ok(all_results)
    }

    async fn fuzzy_search(
        &self,
        query: &str,
        threshold: f64,
        filter: SearchFilter,
    ) -> Result<Vec<SearchResult>, NarraError> {
        use rapidfuzz::distance::levenshtein;

        let entity_types = if filter.entity_types.is_empty() {
            EntityType::all()
        } else {
            filter.entity_types
        };

        let limit = filter.limit.unwrap_or(20);

        // Safety ceiling: fetch at most 500 entities per table to prevent
        // unbounded scans on large datasets. Sufficient for fiction-scale worlds.
        const FETCH_LIMIT: usize = 500;

        // Launch all entity type fetches in parallel
        let futures: Vec<_> = entity_types
            .iter()
            .map(|&entity_type| {
                let db = self.db.clone();
                async move {
                    let table = entity_type.table_name();
                    let name_field = match entity_type {
                        EntityType::Event | EntityType::Scene => "title",
                        EntityType::Knowledge => "fact",
                        EntityType::Note => "title",
                        _ => "name",
                    };

                    let query_str = format!(
                        "SELECT id, '{table}' AS entity_type, {name_field} AS name FROM {table} LIMIT {limit}",
                        table = table,
                        name_field = name_field,
                        limit = FETCH_LIMIT,
                    );

                    let mut response = db.query(&query_str).await?;

                    #[derive(Deserialize)]
                    struct NameOnly {
                        id: surrealdb::RecordId,
                        entity_type: String,
                        name: String,
                    }

                    let entities: Vec<NameOnly> = response.take(0).unwrap_or_default();

                    // For notes, also fetch body data
                    let note_bodies = if entity_type == EntityType::Note {
                        let body_query = format!(
                            "SELECT id, 'note' AS entity_type, title AS name, body FROM note LIMIT {}",
                            FETCH_LIMIT
                        );
                        let mut body_response = db.query(&body_query).await?;

                        #[derive(Deserialize)]
                        struct NoteWithBody {
                            id: surrealdb::RecordId,
                            entity_type: String,
                            name: String,
                            body: String,
                        }

                        let notes: Vec<NoteWithBody> = body_response.take(0).unwrap_or_default();
                        Some(notes.into_iter().map(|n| (n.id.to_string(), n.entity_type, n.name, n.body)).collect::<Vec<_>>())
                    } else {
                        None
                    };

                    Ok::<_, NarraError>((entity_type, entities.into_iter().map(|e| (e.id.to_string(), e.entity_type, e.name)).collect::<Vec<_>>(), note_bodies))
                }
            })
            .collect();

        let fetched = futures::future::join_all(futures).await;

        let mut all_results = Vec::new();
        let mut seen_note_ids = std::collections::HashSet::new();
        let query_lower = query.to_lowercase();

        for fetch_result in fetched {
            let (entity_type, entities, note_bodies) = fetch_result?;

            for (id_str, et, name) in entities {
                let name_lower = name.to_lowercase();
                let similarity =
                    levenshtein::normalized_similarity(query_lower.chars(), name_lower.chars());

                if similarity >= threshold {
                    if entity_type == EntityType::Note && !seen_note_ids.insert(id_str.clone()) {
                        continue;
                    }

                    all_results.push(SearchResult {
                        id: id_str,
                        entity_type: et,
                        name,
                        score: similarity as f32,
                    });
                }
            }

            // Process note bodies
            if let Some(bodies) = note_bodies {
                for (id_str, et, name, body) in bodies {
                    if seen_note_ids.contains(&id_str) {
                        continue;
                    }

                    let body_lower = body.to_lowercase();
                    let similarity =
                        levenshtein::normalized_similarity(query_lower.chars(), body_lower.chars());

                    if similarity >= threshold {
                        seen_note_ids.insert(id_str.clone());
                        all_results.push(SearchResult {
                            id: id_str,
                            entity_type: et,
                            name,
                            score: similarity as f32,
                        });
                    }
                }
            }
        }

        // Sort by similarity descending, then by id ascending for stable pagination
        all_results.sort_by(|a, b| match b.score.partial_cmp(&a.score) {
            Some(std::cmp::Ordering::Equal) | None => a.id.cmp(&b.id),
            Some(ordering) => ordering,
        });

        // Apply limit
        if all_results.len() > limit {
            all_results.truncate(limit);
        }

        Ok(all_results)
    }

    async fn semantic_search(
        &self,
        query: &str,
        filter: SearchFilter,
    ) -> Result<Vec<SearchResult>, NarraError> {
        // Graceful degradation if embeddings unavailable
        if !self.embedding_service.is_available() {
            return Ok(vec![]);
        }

        // Generate query embedding once, share via Arc to avoid cloning per type
        let query_vector = Arc::new(self.embedding_service.embed_text(query).await?);

        // Determine entity types to search (only embeddable types)
        let entity_types = if filter.entity_types.is_empty() {
            EntityType::embeddable()
        } else {
            filter
                .entity_types
                .into_iter()
                .filter(|t| t.has_embeddings())
                .collect()
        };

        let limit = filter.limit.unwrap_or(20);
        let min_score = filter.min_score.unwrap_or(0.0);

        // Launch all entity type queries in parallel
        let futures: Vec<_> = entity_types
            .iter()
            .map(|&entity_type| {
                let db = self.db.clone();
                let qv = Arc::clone(&query_vector);
                let metadata = filter.metadata.clone();
                async move {
                    let table = entity_type.table_name();
                    let name_field = match entity_type {
                        EntityType::Event | EntityType::Scene => "title",
                        EntityType::Knowledge => "fact",
                        _ => "name",
                    };

                    let (filter_clause, filter_bindings) =
                        build_filter_clause(entity_type, &metadata);

                    // Brute-force cosine similarity (HNSW unreliable in embedded RocksDB mode)
                    let k = limit * 2;
                    let query_str = format!(
                        r#"SELECT id, '{table}' AS entity_type, {name_field} AS name,
                                  vector::similarity::cosine(embedding, $query_vector) AS score
                           FROM {table}
                           WHERE embedding IS NOT NONE{filter_clause}
                           ORDER BY score DESC
                           LIMIT {k}"#,
                        table = table,
                        name_field = name_field,
                        filter_clause = filter_clause,
                        k = k,
                    );

                    let mut query_builder =
                        db.query(&query_str).bind(("query_vector", (*qv).clone()));

                    for (key, value) in &filter_bindings {
                        query_builder = query_builder.bind((key.clone(), value.clone()));
                    }

                    let mut response = query_builder.await?;
                    let internal: Vec<SearchResultInternal> = response.take(0)?;
                    Ok::<_, NarraError>(
                        internal
                            .into_iter()
                            .map(SearchResult::from)
                            .collect::<Vec<_>>(),
                    )
                }
            })
            .collect();

        let results_per_type = futures::future::join_all(futures).await;

        let mut all_results = Vec::new();
        for type_results in results_per_type {
            for result in type_results? {
                if result.score >= min_score {
                    all_results.push(result);
                }
            }
        }

        // Sort by score descending, then by id ascending for stable pagination
        all_results.sort_by(|a, b| match b.score.partial_cmp(&a.score) {
            Some(std::cmp::Ordering::Equal) | None => a.id.cmp(&b.id),
            Some(ordering) => ordering,
        });

        // Apply overall limit
        if all_results.len() > limit {
            all_results.truncate(limit);
        }

        Ok(all_results)
    }

    async fn hybrid_search(
        &self,
        query: &str,
        filter: SearchFilter,
    ) -> Result<Vec<SearchResult>, NarraError> {
        // Fallback to keyword search if embeddings unavailable
        if !self.embedding_service.is_available() {
            return self.search(query, filter).await;
        }

        // Generate query embedding once, share via Arc
        let query_vector = Arc::new(self.embedding_service.embed_text(query).await?);

        // Determine entity types to search (only embeddable types for hybrid)
        let entity_types = if filter.entity_types.is_empty() {
            EntityType::embeddable()
        } else {
            filter
                .entity_types
                .into_iter()
                .filter(|t| t.has_embeddings())
                .collect()
        };

        let limit = filter.limit.unwrap_or(20);
        let min_score = filter.min_score;

        // Launch keyword + semantic queries for all types in parallel
        let futures: Vec<_> = entity_types
            .iter()
            .map(|&entity_type| {
                let db = self.db.clone();
                let query_text = query.to_string();
                let qv = Arc::clone(&query_vector);
                let metadata = filter.metadata.clone();
                async move {
                    let table = entity_type.table_name();
                    let name_field = match entity_type {
                        EntityType::Event | EntityType::Scene => "title",
                        EntityType::Knowledge => "fact",
                        _ => "name",
                    };

                    let (filter_clause, filter_bindings) =
                        build_filter_clause(entity_type, &metadata);

                    // Build both queries
                    let keyword_query = format!(
                        r#"SELECT id, '{table}' AS entity_type, {name_field} AS name, search::score(1) AS score
                           FROM {table}
                           WHERE {name_field} @1@ $query{filter_clause}
                           ORDER BY score DESC
                           LIMIT {limit}"#,
                        table = table,
                        name_field = name_field,
                        filter_clause = filter_clause,
                        limit = limit * 2
                    );

                    let k = limit * 2;
                    let semantic_query = format!(
                        r#"SELECT id, '{table}' AS entity_type, {name_field} AS name,
                                  vector::similarity::cosine(embedding, $query_vector) AS score
                           FROM {table}
                           WHERE embedding IS NOT NONE{filter_clause}
                           ORDER BY score DESC
                           LIMIT {k}"#,
                        table = table,
                        name_field = name_field,
                        filter_clause = filter_clause,
                        k = k,
                    );

                    // Execute keyword query
                    let mut kw_builder =
                        db.query(&keyword_query).bind(("query", query_text));
                    for (key, value) in &filter_bindings {
                        kw_builder = kw_builder.bind((key.clone(), value.clone()));
                    }
                    let mut kw_response = kw_builder.await?;
                    let kw_internal: Vec<SearchResultInternal> = kw_response.take(0)?;

                    // Execute semantic query
                    let mut sem_builder = db
                        .query(&semantic_query)
                        .bind(("query_vector", (*qv).clone()));
                    for (key, value) in &filter_bindings {
                        sem_builder = sem_builder.bind((key.clone(), value.clone()));
                    }
                    let mut sem_response = sem_builder.await?;
                    let sem_internal: Vec<SearchResultInternal> = sem_response.take(0)?;

                    let kw_results: Vec<SearchResult> =
                        kw_internal.into_iter().map(SearchResult::from).collect();
                    let sem_results: Vec<SearchResult> =
                        sem_internal.into_iter().map(SearchResult::from).collect();

                    Ok::<_, NarraError>(apply_rrf(&kw_results, &sem_results))
                }
            })
            .collect();

        let results_per_type = futures::future::join_all(futures).await;

        let mut all_results = Vec::new();
        for type_results in results_per_type {
            for result in type_results? {
                if let Some(min) = min_score {
                    if result.score < min {
                        continue;
                    }
                }
                all_results.push(result);
            }
        }

        // Sort by RRF score descending, then by id ascending for stable pagination
        all_results.sort_by(|a, b| match b.score.partial_cmp(&a.score) {
            Some(std::cmp::Ordering::Equal) | None => a.id.cmp(&b.id),
            Some(ordering) => ordering,
        });

        // Apply overall limit
        if all_results.len() > limit {
            all_results.truncate(limit);
        }

        Ok(all_results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_result(id: &str, entity_type: &str, name: &str, score: f32) -> SearchResult {
        SearchResult {
            id: id.to_string(),
            entity_type: entity_type.to_string(),
            name: name.to_string(),
            score,
        }
    }

    #[test]
    fn test_rrf_overlapping_results() {
        let keyword = vec![
            make_result("character:alice", "character", "Alice", 5.0),
            make_result("character:bob", "character", "Bob", 3.0),
        ];
        let semantic = vec![
            make_result("character:bob", "character", "Bob", 0.9),
            make_result("character:alice", "character", "Alice", 0.8),
        ];

        let merged = apply_rrf(&keyword, &semantic);
        assert_eq!(merged.len(), 2, "Should have 2 unique results");

        // Both appear in both lists, so both get 2 contributions.
        // Alice: 1/(60+1) + 1/(60+2) ≈ 0.01639 + 0.01613 = 0.03252
        // Bob:   1/(60+2) + 1/(60+1) ≈ 0.01613 + 0.01639 = 0.03252
        // Scores should be equal; order breaks by id (alice < bob)
        assert_eq!(merged[0].id, "character:alice");
        assert_eq!(merged[1].id, "character:bob");
        assert!((merged[0].score - merged[1].score).abs() < 1e-6);
    }

    #[test]
    fn test_rrf_disjoint_results() {
        let keyword = vec![make_result("character:alice", "character", "Alice", 5.0)];
        let semantic = vec![make_result("character:bob", "character", "Bob", 0.9)];

        let merged = apply_rrf(&keyword, &semantic);
        assert_eq!(merged.len(), 2);
        // Both get 1/(60+1) ≈ 0.01639; tie broken by id
        assert_eq!(merged[0].id, "character:alice");
        assert_eq!(merged[1].id, "character:bob");
    }

    #[test]
    fn test_rrf_empty_lists() {
        let merged = apply_rrf(&[], &[]);
        assert!(merged.is_empty());
    }

    #[test]
    fn test_rrf_single_source_only() {
        let keyword = vec![
            make_result("event:battle", "event", "Battle", 10.0),
            make_result("event:feast", "event", "Feast", 5.0),
        ];

        let merged = apply_rrf(&keyword, &[]);
        assert_eq!(merged.len(), 2);
        // Rank 1 gets higher RRF than rank 2
        assert_eq!(merged[0].id, "event:battle");
        assert_eq!(merged[1].id, "event:feast");
        assert!(merged[0].score > merged[1].score);
    }

    #[test]
    fn test_rrf_semantic_boost() {
        // Item in both lists should rank higher than item in one list only
        let keyword = vec![
            make_result("character:alice", "character", "Alice", 5.0),
            make_result("character:bob", "character", "Bob", 3.0),
        ];
        let semantic = vec![make_result("character:bob", "character", "Bob", 0.9)];

        let merged = apply_rrf(&keyword, &semantic);
        // Bob: keyword rank 2 + semantic rank 1 = 1/(60+2) + 1/(60+1)
        // Alice: keyword rank 1 only = 1/(60+1)
        // Bob should rank higher than Alice
        assert_eq!(merged[0].id, "character:bob");
        assert_eq!(merged[1].id, "character:alice");
        assert!(merged[0].score > merged[1].score);
    }
}
