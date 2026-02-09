use crate::mcp::NarraServer;
use crate::mcp::{EntityResult, QueryResponse};
use crate::services::perception::PerceptionService;

impl NarraServer {
    pub(crate) async fn handle_perspective_search(
        &self,
        query: &str,
        observer_id: Option<String>,
        target_id: Option<String>,
        limit: usize,
    ) -> Result<QueryResponse, String> {
        // Embed query text
        let query_embedding = self
            .embedding_service
            .embed_text(query)
            .await
            .map_err(|e| format!("Failed to embed query: {}", e))?;

        // Build filter clause with parameterized values
        let mut filters = String::new();
        if observer_id.is_some() {
            filters.push_str(" AND in = $obs_ref");
        }
        if target_id.is_some() {
            filters.push_str(" AND out = $tgt_ref");
        }

        let query_str = format!(
            "SELECT id, in.name AS observer_name, out.name AS target_name, \
             perception, feelings, tension_level, history_notes, embedding, \
             vector::similarity::cosine(embedding, $query_vec) AS score \
             FROM perceives \
             WHERE embedding IS NOT NONE{} \
             ORDER BY score DESC LIMIT $lim",
            filters
        );

        let mut q = self
            .db
            .query(&query_str)
            .bind(("query_vec", query_embedding))
            .bind(("lim", limit));
        if let Some(ref obs_id) = observer_id {
            let (table, key) = obs_id.split_once(':').unwrap_or(("character", obs_id));
            q = q.bind(("obs_ref", surrealdb::RecordId::from((table, key))));
        }
        if let Some(ref tgt_id) = target_id {
            let (table, key) = tgt_id.split_once(':').unwrap_or(("character", tgt_id));
            q = q.bind(("tgt_ref", surrealdb::RecordId::from((table, key))));
        }
        let mut response = q
            .await
            .map_err(|e| format!("Perspective search failed: {}", e))?;

        #[derive(serde::Deserialize)]
        struct PerspectiveResult {
            id: surrealdb::sql::Thing,
            observer_name: Option<String>,
            target_name: Option<String>,
            perception: Option<String>,
            feelings: Option<String>,
            tension_level: Option<i32>,
            history_notes: Option<String>,
            score: f32,
        }

        let results: Vec<PerspectiveResult> = response
            .take(0)
            .map_err(|e| format!("Failed to parse perspective results: {}", e))?;

        let entity_results: Vec<EntityResult> = results
            .iter()
            .map(|r| {
                let obs = r.observer_name.as_deref().unwrap_or("?");
                let tgt = r.target_name.as_deref().unwrap_or("?");

                let mut content_parts = Vec::new();
                if let Some(ref p) = r.perception {
                    content_parts.push(format!("Perception: {}", p));
                }
                if let Some(ref f) = r.feelings {
                    content_parts.push(format!("Feelings: {}", f));
                }
                if let Some(t) = r.tension_level {
                    content_parts.push(format!("Tension: {}/10", t));
                }
                if let Some(ref h) = r.history_notes {
                    content_parts.push(format!("History: {}", h));
                }
                let content = if content_parts.is_empty() {
                    format!("{}'s view of {}", obs, tgt)
                } else {
                    content_parts.join(". ")
                };

                EntityResult {
                    id: r.id.to_string(),
                    entity_type: "perspective".to_string(),
                    name: format!("{} → {}", obs, tgt),
                    content,
                    confidence: Some(r.score),
                    last_modified: None,
                }
            })
            .collect();

        let total = entity_results.len();
        let token_estimate = entity_results
            .iter()
            .map(|r| r.content.len() / 4 + 30)
            .sum();

        let hints = vec![
            format!("Found {} perspectives matching \"{}\"", total, query),
            "Use PerceptionGap to measure accuracy of a specific perspective".to_string(),
        ];

        Ok(QueryResponse {
            results: entity_results,
            total,
            next_cursor: None,
            hints,
            token_estimate,
        })
    }

    pub(crate) async fn handle_perception_gap(
        &self,
        observer_id: &str,
        target_id: &str,
    ) -> Result<QueryResponse, String> {
        let service = PerceptionService::new(self.db.clone());
        let result = service
            .analyze_gap(observer_id, target_id)
            .await
            .map_err(|e| format!("{}", e))?;

        let mut content_parts = vec![
            format!("Perception gap: {:.4} ({})", result.gap, result.assessment),
            format!(
                "Cosine similarity to real {}: {:.4}",
                result.target_name, result.similarity
            ),
        ];

        if let Some(ref p) = result.perception {
            content_parts.push(format!(
                "{} perceives {} as: {}",
                result.observer_name, result.target_name, p
            ));
        }
        if let Some(ref f) = result.feelings {
            content_parts.push(format!("{} feels: {}", result.observer_name, f));
        }
        if let Some(t) = result.tension_level {
            content_parts.push(format!("Tension: {}/10", t));
        }
        if let Some(ref h) = result.history_notes {
            content_parts.push(format!("History: {}", h));
        }

        let er = EntityResult {
            id: format!("perceives:{}->{}", observer_id, target_id),
            entity_type: "perception_gap".to_string(),
            name: format!("{} → {} gap", result.observer_name, result.target_name),
            content: content_parts.join("\n"),
            confidence: Some(1.0 - result.gap),
            last_modified: None,
        };

        // Capitalize assessment for hints
        let cap_assessment = {
            let a = &result.assessment;
            let mut chars = a.chars();
            match chars.next() {
                None => String::new(),
                Some(c) => c.to_uppercase().to_string() + chars.as_str(),
            }
        };

        let mut hints = vec![format!(
            "{}: {} (gap={:.4})",
            cap_assessment, result.observer_name, result.gap
        )];
        if result.gap >= 0.30 {
            hints.push("High gap = dramatic irony potential. This character's view is significantly distorted.".to_string());
        }
        hints.push("Use PerceptionMatrix to compare how all observers see this target".to_string());

        let token_estimate = er.content.len() / 4 + 30;

        Ok(QueryResponse {
            results: vec![er],
            total: 1,
            next_cursor: None,
            hints,
            token_estimate,
        })
    }

    pub(crate) async fn handle_perception_matrix(
        &self,
        target_id: &str,
        limit: usize,
    ) -> Result<QueryResponse, String> {
        let service = PerceptionService::new(self.db.clone());
        let result = service
            .analyze_matrix(target_id, limit)
            .await
            .map_err(|e| format!("{}", e))?;

        let entity_results: Vec<EntityResult> = result
            .observers
            .iter()
            .map(|obs| {
                let mut content = format!("Gap: {:.4} ({})", obs.gap, obs.assessment);
                if let Some(ref closest) = obs.agrees_with {
                    // Recompute similarity for display (observer embeddings are available)
                    content.push_str(&format!(". Agrees most with {}", closest));
                }
                if let Some(ref furthest) = obs.disagrees_with {
                    content.push_str(&format!(". Disagrees most with {}", furthest));
                }

                EntityResult {
                    id: format!("perceives:{}->{}", obs.observer_name, result.target_name),
                    entity_type: "perception_matrix".to_string(),
                    name: format!("{} → {}", obs.observer_name, result.target_name),
                    content,
                    confidence: Some(1.0 - obs.gap),
                    last_modified: None,
                }
            })
            .collect();

        let most_accurate = result
            .observers
            .first()
            .map(|o| o.observer_name.as_str())
            .unwrap_or("?");
        let least_accurate = result
            .observers
            .last()
            .map(|o| o.observer_name.as_str())
            .unwrap_or("?");

        let mut hints = vec![
            format!(
                "{} observers of {}",
                entity_results.len(),
                result.target_name
            ),
            format!(
                "Most accurate: {}. Least accurate: {}.",
                most_accurate, least_accurate
            ),
        ];
        if !result.has_real_embedding {
            hints.push(
                "Warning: Target has no embedding — gap values are all 0. Run BackfillEmbeddings."
                    .to_string(),
            );
        }
        hints.push("Use PerceptionGap for detailed analysis of a specific observer".to_string());

        let total = entity_results.len();
        let token_estimate = entity_results
            .iter()
            .map(|r| r.content.len() / 4 + 30)
            .sum();

        Ok(QueryResponse {
            results: entity_results,
            total,
            next_cursor: None,
            hints,
            token_estimate,
        })
    }

    pub(crate) async fn handle_perception_shift(
        &self,
        observer_id: &str,
        target_id: &str,
    ) -> Result<QueryResponse, String> {
        let service = PerceptionService::new(self.db.clone());
        let result = service
            .analyze_shift(observer_id, target_id)
            .await
            .map_err(|e| format!("{}", e))?;

        let perceives_id_str = format!("perceives:{}->{}", observer_id, target_id);

        let entity_results: Vec<EntityResult> = result
            .snapshots
            .iter()
            .enumerate()
            .map(|(i, snap)| {
                let delta_str = snap
                    .delta
                    .map(|d| format!("{:.4}", d))
                    .unwrap_or_else(|| "baseline".to_string());

                let event_info = snap
                    .event
                    .as_ref()
                    .map(|t| format!(" ({})", t))
                    .unwrap_or_default();

                let gap_info = snap
                    .gap
                    .map(|g| format!(", gap={:.4}", g))
                    .unwrap_or_default();

                let content = format!(
                    "Snapshot {}: delta={}{}{}",
                    i + 1,
                    delta_str,
                    gap_info,
                    event_info
                );

                EntityResult {
                    id: format!("{}-snap-{}", perceives_id_str, i),
                    entity_type: "perception_shift".to_string(),
                    name: format!(
                        "{} → {} snapshot {}",
                        result.observer_name,
                        result.target_name,
                        i + 1
                    ),
                    content,
                    confidence: snap.delta,
                    last_modified: Some(snap.timestamp.clone()),
                }
            })
            .collect();

        let total = entity_results.len();
        let token_estimate = entity_results
            .iter()
            .map(|r| r.content.len() / 4 + 30)
            .sum();

        let hints = vec![
            format!(
                "{} perspective snapshots for {} → {}",
                total, result.observer_name, result.target_name
            ),
            result.trajectory.clone(),
            "Use PerceptionGap for current gap analysis".to_string(),
        ];

        Ok(QueryResponse {
            results: entity_results,
            total,
            next_cursor: None,
            hints,
            token_estimate,
        })
    }

    pub(crate) async fn handle_similar_relationships(
        &self,
        observer_id: &str,
        target_id: &str,
        edge_type: Option<String>,
        bias: Option<String>,
        limit: usize,
    ) -> Result<QueryResponse, String> {
        if !self.embedding_service.is_available() {
            return Err(
                "SimilarRelationships requires embedding service. Run BackfillEmbeddings first."
                    .to_string(),
            );
        }

        // Normalize IDs
        let from_ref = if observer_id.contains(':') {
            observer_id.to_string()
        } else {
            format!("character:{}", observer_id)
        };
        let to_ref = if target_id.contains(':') {
            target_id.to_string()
        } else {
            format!("character:{}", target_id)
        };

        // Find the source edge embedding
        let tables_to_try: Vec<&str> = match edge_type.as_deref() {
            Some("perceives") => vec!["perceives"],
            Some("relates_to") => vec!["relates_to"],
            _ => vec!["perceives", "relates_to"],
        };

        #[derive(serde::Deserialize)]
        struct EdgeEmbedding {
            embedding: Vec<f32>,
        }

        let mut source_embedding: Option<Vec<f32>> = None;
        let mut source_edge_type = "";

        for table in &tables_to_try {
            let query = format!(
                "SELECT embedding FROM {} WHERE in = {} AND out = {} AND embedding IS NOT NONE LIMIT 1",
                table, from_ref, to_ref
            );

            let mut resp = self
                .db
                .query(&query)
                .await
                .map_err(|e| format!("Failed to query {} edges: {}", table, e))?;

            let results: Vec<EdgeEmbedding> = resp.take(0).unwrap_or_default();
            if let Some(edge) = results.into_iter().next() {
                source_embedding = Some(edge.embedding);
                source_edge_type = table;
                break;
            }
        }

        let source_embedding = source_embedding.ok_or_else(|| {
            format!(
                "No edge with embedding found between {} and {}. Run BackfillEmbeddings to generate missing embeddings.",
                from_ref, to_ref
            )
        })?;

        // Build search vector (with optional bias)
        let search_vec = if let Some(ref bias_text) = bias {
            let bias_vec = self
                .embedding_service
                .embed_text(bias_text)
                .await
                .map_err(|e| format!("Failed to embed bias text: {}", e))?;
            // Interpolate: 0.7 * source + 0.3 * bias, then normalize
            let combined: Vec<f32> = source_embedding
                .iter()
                .zip(bias_vec.iter())
                .map(|(s, b)| 0.7 * s + 0.3 * b)
                .collect();
            let norm: f32 = combined.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                combined.iter().map(|x| x / norm).collect()
            } else {
                combined
            }
        } else {
            source_embedding.clone()
        };

        // Search across both edge tables
        let mut all_results: Vec<EntityResult> = Vec::new();

        for table in &["perceives", "relates_to"] {
            let (extra_fields, edge_kind_str) = if *table == "perceives" {
                (
                    "perception, feelings, tension_level, rel_types,",
                    "'perceives'",
                )
            } else {
                ("rel_type, subtype, label,", "'relates_to'")
            };

            let query = format!(
                "SELECT type::string(id) AS edge_id, type::string(in) AS from_id, type::string(out) AS to_id, \
                 in.name AS from_name, out.name AS to_name, \
                 {extra_fields} {edge_kind_str} AS edge_kind, \
                 vector::similarity::cosine(embedding, $search_vec) AS score \
                 FROM {table} \
                 WHERE embedding IS NOT NONE AND !(in = {from_ref} AND out = {to_ref}) \
                 ORDER BY score DESC LIMIT {limit}",
                extra_fields = extra_fields,
                edge_kind_str = edge_kind_str,
                table = table,
                from_ref = from_ref,
                to_ref = to_ref,
                limit = limit,
            );

            let mut resp = self
                .db
                .query(&query)
                .bind(("search_vec", search_vec.clone()))
                .await
                .map_err(|e| format!("Failed to search {} edges: {}", table, e))?;

            #[derive(serde::Deserialize)]
            struct SimilarEdge {
                edge_id: Option<String>,
                from_name: Option<String>,
                to_name: Option<String>,
                edge_kind: Option<String>,
                score: f32,
                // perceives fields
                perception: Option<String>,
                feelings: Option<String>,
                tension_level: Option<i32>,
                rel_types: Option<Vec<String>>,
                // relates_to fields
                rel_type: Option<String>,
                subtype: Option<String>,
                label: Option<String>,
            }

            let edges: Vec<SimilarEdge> = resp.take(0).unwrap_or_default();

            for edge in edges {
                let from_name = edge.from_name.as_deref().unwrap_or("?");
                let to_name = edge.to_name.as_deref().unwrap_or("?");
                let kind = edge.edge_kind.as_deref().unwrap_or(*table);

                let content = if *table == "perceives" {
                    let perception = edge.perception.as_deref().unwrap_or("(none)");
                    let feelings = edge.feelings.as_deref().unwrap_or("(none)");
                    let tension_str = edge
                        .tension_level
                        .map(|t| format!(". Tension: {}/10", t))
                        .unwrap_or_default();
                    let rel_types = edge
                        .rel_types
                        .as_deref()
                        .map(|r| r.join(", "))
                        .unwrap_or_default();
                    format!(
                        "Perception: {}. Feelings: {}{}. Rel types: {}. Similarity: {:.3}",
                        perception, feelings, tension_str, rel_types, edge.score
                    )
                } else {
                    let rel_type = edge.rel_type.as_deref().unwrap_or("(none)");
                    let subtype = edge
                        .subtype
                        .as_deref()
                        .map(|s| format!(", subtype: {}", s))
                        .unwrap_or_default();
                    let label = edge
                        .label
                        .as_deref()
                        .map(|l| format!(". {}", l))
                        .unwrap_or_default();
                    format!(
                        "Type: {}{}{} Similarity: {:.3}",
                        rel_type, subtype, label, edge.score
                    )
                };

                all_results.push(EntityResult {
                    id: edge.edge_id.unwrap_or_else(|| format!("{}:unknown", table)),
                    entity_type: "similar_relationship".to_string(),
                    name: format!("{} → {} ({})", from_name, to_name, kind),
                    content,
                    confidence: Some(edge.score),
                    last_modified: None,
                });
            }
        }

        // Sort by score descending
        all_results.sort_by(|a, b| {
            b.confidence
                .unwrap_or(0.0)
                .partial_cmp(&a.confidence.unwrap_or(0.0))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        all_results.truncate(limit);

        let total = all_results.len();

        let mut hints = vec![format!(
            "Source: {} → {} ({})",
            from_ref, to_ref, source_edge_type
        )];
        if let Some(ref b) = bias {
            hints.push(format!("Biased toward: \"{}\"", b));
        }
        let perceives_count = all_results
            .iter()
            .filter(|r| r.name.contains("perceives"))
            .count();
        let relates_count = all_results
            .iter()
            .filter(|r| r.name.contains("relates_to"))
            .count();
        hints.push(format!(
            "{} perceives, {} relates_to edges in results",
            perceives_count, relates_count
        ));

        let token_estimate = self.estimate_tokens_from_results(&all_results);

        Ok(QueryResponse {
            results: all_results,
            total,
            next_cursor: None,
            hints,
            token_estimate,
        })
    }
}
