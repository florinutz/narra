use crate::mcp::NarraServer;
use crate::mcp::{EntityResult, QueryResponse};
use crate::utils::math::cosine_similarity;

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
        // Find perceives edge with embedding
        let edge_query = format!(
            "SELECT id, embedding, in.name AS observer_name, out.name AS target_name, \
             perception, feelings, tension_level, history_notes \
             FROM perceives \
             WHERE in = {} AND out = {} AND embedding IS NOT NONE",
            observer_id, target_id
        );

        let mut edge_resp = self
            .db
            .query(&edge_query)
            .await
            .map_err(|e| format!("Failed to fetch perceives edge: {}", e))?;

        #[derive(serde::Deserialize)]
        struct PerceivesEdge {
            id: surrealdb::sql::Thing,
            embedding: Vec<f32>,
            observer_name: Option<String>,
            target_name: Option<String>,
            perception: Option<String>,
            feelings: Option<String>,
            tension_level: Option<i32>,
            history_notes: Option<String>,
        }

        let edges: Vec<PerceivesEdge> = edge_resp
            .take(0)
            .map_err(|e| format!("Failed to parse perceives edge: {}", e))?;

        let edge = edges.into_iter().next().ok_or_else(|| {
            format!(
                "No perspective embedding found for {} → {}. Run BackfillEmbeddings first.",
                observer_id, target_id
            )
        })?;

        // Fetch target's real embedding
        let real_query = format!("SELECT VALUE embedding FROM {}", target_id);
        let mut real_resp = self
            .db
            .query(&real_query)
            .await
            .map_err(|e| format!("Failed to fetch target embedding: {}", e))?;

        let real_embeddings: Vec<Option<Vec<f32>>> = real_resp.take(0).unwrap_or_default();
        let real_embedding = real_embeddings
            .into_iter()
            .next()
            .flatten()
            .ok_or_else(|| {
                format!(
                    "Target {} has no embedding. Run BackfillEmbeddings first.",
                    target_id
                )
            })?;

        // Compute gap
        let similarity = cosine_similarity(&edge.embedding, &real_embedding);
        let gap = 1.0 - similarity;

        let obs_name = edge.observer_name.as_deref().unwrap_or("?");
        let tgt_name = edge.target_name.as_deref().unwrap_or("?");

        // Qualitative assessment
        let assessment = if gap < 0.05 {
            "Remarkably accurate"
        } else if gap < 0.15 {
            "Fairly accurate"
        } else if gap < 0.30 {
            "Notable blind spots"
        } else if gap < 0.50 {
            "Significantly distorted"
        } else {
            "Dramatically wrong"
        };

        let mut content_parts = vec![
            format!("Perception gap: {:.4} ({})", gap, assessment),
            format!("Cosine similarity to real {}: {:.4}", tgt_name, similarity),
        ];

        if let Some(ref p) = edge.perception {
            content_parts.push(format!("{} perceives {} as: {}", obs_name, tgt_name, p));
        }
        if let Some(ref f) = edge.feelings {
            content_parts.push(format!("{} feels: {}", obs_name, f));
        }
        if let Some(t) = edge.tension_level {
            content_parts.push(format!("Tension: {}/10", t));
        }
        if let Some(ref h) = edge.history_notes {
            content_parts.push(format!("History: {}", h));
        }

        let result = EntityResult {
            id: edge.id.to_string(),
            entity_type: "perception_gap".to_string(),
            name: format!("{} → {} gap", obs_name, tgt_name),
            content: content_parts.join("\n"),
            confidence: Some(1.0 - gap), // Higher confidence = more accurate perception
            last_modified: None,
        };

        let mut hints = vec![format!("{}: {} (gap={:.4})", assessment, obs_name, gap)];
        if gap >= 0.30 {
            hints.push("High gap = dramatic irony potential. This character's view is significantly distorted.".to_string());
        }
        hints.push("Use PerceptionMatrix to compare how all observers see this target".to_string());

        let token_estimate = result.content.len() / 4 + 30;

        Ok(QueryResponse {
            results: vec![result],
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
        // Fetch all perspective embeddings for target
        let persp_query = format!(
            "SELECT id, embedding, in.name AS observer_name \
             FROM perceives WHERE out = {} AND embedding IS NOT NONE LIMIT {}",
            target_id, limit
        );

        let mut persp_resp = self
            .db
            .query(&persp_query)
            .await
            .map_err(|e| format!("Failed to fetch perspectives: {}", e))?;

        #[derive(serde::Deserialize)]
        struct ObserverPersp {
            id: surrealdb::sql::Thing,
            embedding: Vec<f32>,
            observer_name: Option<String>,
        }

        let perspectives: Vec<ObserverPersp> = persp_resp
            .take(0)
            .map_err(|e| format!("Failed to parse perspectives: {}", e))?;

        if perspectives.is_empty() {
            return Err(format!(
                "No perspective embeddings found for {}. Run BackfillEmbeddings first.",
                target_id
            ));
        }

        // Fetch target's real embedding
        let real_query = format!("SELECT VALUE embedding FROM {}", target_id);
        let mut real_resp = self
            .db
            .query(&real_query)
            .await
            .map_err(|e| format!("Failed to fetch target embedding: {}", e))?;

        let real_embeddings: Vec<Option<Vec<f32>>> = real_resp.take(0).unwrap_or_default();
        let real_embedding = real_embeddings.into_iter().next().flatten();

        let target_name = self.extract_name_from_id(target_id);

        // Compute per-observer gap and pairwise agreement
        struct ObserverData {
            id: String,
            name: String,
            gap: f32,
            embedding: Vec<f32>,
        }

        let mut observer_data: Vec<ObserverData> = perspectives
            .iter()
            .map(|p| {
                let name = p.observer_name.as_deref().unwrap_or("?").to_string();
                let gap = real_embedding
                    .as_ref()
                    .map(|real| 1.0 - cosine_similarity(&p.embedding, real))
                    .unwrap_or(0.0);
                ObserverData {
                    id: p.id.to_string(),
                    name,
                    gap,
                    embedding: p.embedding.clone(),
                }
            })
            .collect();

        // Sort by accuracy (lowest gap first)
        observer_data.sort_by(|a, b| {
            a.gap
                .partial_cmp(&b.gap)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Build results with pairwise agreement info
        let entity_results: Vec<EntityResult> = observer_data
            .iter()
            .enumerate()
            .map(|(i, obs)| {
                let gap_assessment = if obs.gap < 0.05 {
                    "remarkably accurate"
                } else if obs.gap < 0.15 {
                    "fairly accurate"
                } else if obs.gap < 0.30 {
                    "notable blind spots"
                } else if obs.gap < 0.50 {
                    "significantly distorted"
                } else {
                    "dramatically wrong"
                };

                // Find most/least agreeing observers
                let mut agreements: Vec<(&str, f32)> = observer_data
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i)
                    .map(|(_, other)| {
                        let sim = cosine_similarity(&obs.embedding, &other.embedding);
                        (other.name.as_str(), sim)
                    })
                    .collect();
                agreements
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                let mut content = format!("Gap: {:.4} ({})", obs.gap, gap_assessment);
                if let Some((closest_name, closest_sim)) = agreements.first() {
                    content.push_str(&format!(
                        ". Agrees most with {} ({:.3})",
                        closest_name, closest_sim
                    ));
                }
                if let Some((furthest_name, furthest_sim)) = agreements.last() {
                    if agreements.len() > 1 {
                        content.push_str(&format!(
                            ". Disagrees most with {} ({:.3})",
                            furthest_name, furthest_sim
                        ));
                    }
                }

                EntityResult {
                    id: obs.id.clone(),
                    entity_type: "perception_matrix".to_string(),
                    name: format!("{} → {}", obs.name, target_name),
                    content,
                    confidence: Some(1.0 - obs.gap),
                    last_modified: None,
                }
            })
            .collect();

        let most_accurate = observer_data
            .first()
            .map(|o| o.name.as_str())
            .unwrap_or("?");
        let least_accurate = observer_data.last().map(|o| o.name.as_str()).unwrap_or("?");

        let mut hints = vec![
            format!("{} observers of {}", entity_results.len(), target_name),
            format!(
                "Most accurate: {}. Least accurate: {}.",
                most_accurate, least_accurate
            ),
        ];
        if real_embedding.is_none() {
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
        // Find the perceives edge ID
        let edge_query = format!(
            "SELECT id FROM perceives WHERE in = {} AND out = {}",
            observer_id, target_id
        );

        let mut edge_resp = self
            .db
            .query(&edge_query)
            .await
            .map_err(|e| format!("Failed to find perceives edge: {}", e))?;

        #[derive(serde::Deserialize)]
        struct EdgeIdResult {
            id: surrealdb::sql::Thing,
        }

        let edge_ids: Vec<EdgeIdResult> = edge_resp
            .take(0)
            .map_err(|e| format!("Failed to parse perceives edge: {}", e))?;

        let perceives_id = edge_ids.into_iter().next().ok_or_else(|| {
            format!(
                "No perceives edge found from {} to {}",
                observer_id, target_id
            )
        })?;

        let perceives_id_str = perceives_id.id.to_string();

        // Fetch perspective arc snapshots
        let persp_snap_query = format!(
            "SELECT embedding, delta_magnitude, created_at, event_id.title AS event_title \
             FROM arc_snapshot \
             WHERE entity_id = {} AND entity_type = 'perspective' \
             ORDER BY created_at ASC",
            perceives_id_str
        );

        let mut persp_resp = self
            .db
            .query(&persp_snap_query)
            .await
            .map_err(|e| format!("Perception shift query failed: {}", e))?;

        #[derive(serde::Deserialize)]
        struct PerspSnapshot {
            embedding: Vec<f32>,
            delta_magnitude: Option<f32>,
            created_at: String,
            event_title: Option<String>,
        }

        let persp_snaps: Vec<PerspSnapshot> = persp_resp
            .take(0)
            .map_err(|e| format!("Failed to parse perspective snapshots: {}", e))?;

        if persp_snaps.is_empty() {
            return Err(format!(
                "No perspective arc snapshots found for {} → {}. \
                 Run BackfillEmbeddings first, then trigger updates to build perspective history.",
                observer_id, target_id
            ));
        }

        // Fetch target character arc snapshots
        let char_snap_query = format!(
            "SELECT embedding, created_at FROM arc_snapshot \
             WHERE entity_id = {} AND entity_type = 'character' \
             ORDER BY created_at ASC",
            target_id
        );

        let mut char_resp = self
            .db
            .query(&char_snap_query)
            .await
            .map_err(|e| format!("Failed to fetch target snapshots: {}", e))?;

        #[derive(serde::Deserialize)]
        struct CharSnapshot {
            embedding: Vec<f32>,
            created_at: String,
        }

        let char_snaps: Vec<CharSnapshot> = char_resp.take(0).unwrap_or_default();

        let obs_name = self.extract_name_from_id(observer_id);
        let tgt_name = self.extract_name_from_id(target_id);

        // For each perspective snapshot, find nearest-time character snapshot and compute gap
        let entity_results: Vec<EntityResult> = persp_snaps
            .iter()
            .enumerate()
            .map(|(i, psnap)| {
                let delta_str = psnap
                    .delta_magnitude
                    .map(|d| format!("{:.4}", d))
                    .unwrap_or_else(|| "baseline".to_string());

                let event_info = psnap
                    .event_title
                    .as_ref()
                    .map(|t| format!(" ({})", t))
                    .unwrap_or_default();

                // Find nearest character snapshot
                let gap_info = if !char_snaps.is_empty() {
                    // Find snapshot with closest timestamp (simple string comparison works for ISO dates)
                    let nearest = char_snaps.iter().min_by_key(|cs| {
                        // Absolute difference by string comparison (approximate but sufficient)
                        if cs.created_at <= psnap.created_at {
                            psnap.created_at.len() // favor earlier snapshots
                        } else {
                            cs.created_at.len() + 1
                        }
                    });

                    if let Some(nearest_char) = nearest {
                        let gap =
                            1.0 - cosine_similarity(&psnap.embedding, &nearest_char.embedding);
                        format!(", gap={:.4}", gap)
                    } else {
                        String::new()
                    }
                } else {
                    String::new()
                };

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
                    name: format!("{} → {} snapshot {}", obs_name, tgt_name, i + 1),
                    content,
                    confidence: psnap.delta_magnitude,
                    last_modified: Some(psnap.created_at.clone()),
                }
            })
            .collect();

        // Compute gap trajectory if we have character snapshots
        let trajectory_hint = if char_snaps.len() >= 2 && persp_snaps.len() >= 2 {
            // Compare first and last gap
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
            let delta = last_gap - first_gap;

            if delta < -0.02 {
                format!(
                    "Gap converging ({:.4} → {:.4}): {} is getting more accurate about {}",
                    first_gap, last_gap, obs_name, tgt_name
                )
            } else if delta > 0.02 {
                format!(
                    "Gap diverging ({:.4} → {:.4}): {} is drifting further from reality about {}",
                    first_gap, last_gap, obs_name, tgt_name
                )
            } else {
                format!(
                    "Gap stable ({:.4} → {:.4}): {}'s accuracy about {} is unchanged",
                    first_gap, last_gap, obs_name, tgt_name
                )
            }
        } else {
            "Need more snapshots for gap trajectory analysis".to_string()
        };

        let total = entity_results.len();
        let token_estimate = entity_results
            .iter()
            .map(|r| r.content.len() / 4 + 30)
            .sum();

        let hints = vec![
            format!(
                "{} perspective snapshots for {} → {}",
                total, obs_name, tgt_name
            ),
            trajectory_hint,
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
