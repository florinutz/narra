use crate::mcp::NarraServer;
use crate::mcp::{EntityResult, QueryResponse};
use crate::utils::math::{cosine_similarity, vector_subtract};

impl NarraServer {
    pub(crate) async fn handle_arc_history(
        &self,
        entity_id: &str,
        limit: usize,
    ) -> Result<QueryResponse, String> {
        // Fetch snapshots ordered by time ascending
        let query_str = format!(
            "SELECT *, event_id.title AS event_title FROM arc_snapshot \
             WHERE entity_id = {} ORDER BY created_at ASC LIMIT {}",
            entity_id, limit
        );

        let mut response = self
            .db
            .query(&query_str)
            .await
            .map_err(|e| format!("Arc history query failed: {}", e))?;

        #[derive(serde::Deserialize)]
        struct ArcSnapshot {
            id: surrealdb::sql::Thing,
            entity_type: String,
            delta_magnitude: Option<f32>,
            event_title: Option<String>,
            created_at: String,
            embedding: Vec<f32>,
        }

        let snapshots: Vec<ArcSnapshot> = response
            .take(0)
            .map_err(|e| format!("Failed to parse arc snapshots: {}", e))?;

        if snapshots.is_empty() {
            return Ok(QueryResponse {
                results: vec![],
                total: 0,
                next_cursor: None,
                hints: vec![
                    format!("No arc snapshots found for {}", entity_id),
                    "Run BaselineArcSnapshots to capture initial state, then update the entity to generate snapshots".to_string(),
                ],
                token_estimate: 0,
            });
        }

        // Compute cumulative drift and net displacement
        let mut cumulative_drift = 0.0f32;
        let first_embedding = &snapshots[0].embedding;
        let last_embedding = &snapshots[snapshots.len() - 1].embedding;
        let net_displacement = 1.0 - cosine_similarity(first_embedding, last_embedding);

        let entity_name = self.extract_name_from_id(entity_id);
        let total_snapshots = snapshots.len();

        let entity_results: Vec<EntityResult> = snapshots
            .iter()
            .enumerate()
            .map(|(i, snap)| {
                if let Some(delta) = snap.delta_magnitude {
                    cumulative_drift += delta;
                }

                let event_info = snap
                    .event_title
                    .as_ref()
                    .map(|t| format!(" (during: {})", t))
                    .unwrap_or_default();

                let delta_str = snap
                    .delta_magnitude
                    .map(|d| format!("{:.4}", d))
                    .unwrap_or_else(|| "baseline".to_string());

                let content = format!(
                    "Snapshot {}: delta={}, cumulative={:.4}{}",
                    i + 1,
                    delta_str,
                    cumulative_drift,
                    event_info
                );

                EntityResult {
                    id: snap.id.to_string(),
                    entity_type: snap.entity_type.clone(),
                    name: format!("{} snapshot {}", entity_name, i + 1),
                    content,
                    confidence: snap.delta_magnitude,
                    last_modified: Some(snap.created_at.clone()),
                }
            })
            .collect();

        // Qualitative assessment
        let assessment = if net_displacement < 0.02 {
            "essentially unchanged"
        } else if net_displacement < 0.1 {
            "minor evolution"
        } else if net_displacement < 0.3 {
            "significant evolution"
        } else {
            "dramatic transformation"
        };

        let hints = vec![
            format!("{} snapshots for {}", total_snapshots, entity_id),
            format!("Net displacement: {:.4} ({})", net_displacement, assessment),
            format!("Cumulative drift: {:.4}", cumulative_drift),
            "Use ArcComparison to compare trajectories with another entity".to_string(),
        ];

        let token_estimate = entity_results
            .iter()
            .map(|r| r.content.len() / 4 + 30)
            .sum();

        Ok(QueryResponse {
            results: entity_results,
            total: total_snapshots,
            next_cursor: None,
            hints,
            token_estimate,
        })
    }

    pub(crate) async fn handle_arc_comparison(
        &self,
        entity_id_a: &str,
        entity_id_b: &str,
        window: Option<String>,
    ) -> Result<QueryResponse, String> {
        // Parse window parameter
        let limit_clause = if let Some(ref w) = window {
            if let Some(n_str) = w.strip_prefix("recent:") {
                let n: usize = n_str
                    .parse()
                    .map_err(|_| format!("Invalid window format: '{}'. Use 'recent:N'", w))?;
                format!(" LIMIT {}", n)
            } else {
                return Err(format!("Invalid window format: '{}'. Use 'recent:N'", w));
            }
        } else {
            String::new()
        };

        // Fetch snapshots for both entities
        let order = if window.is_some() { "DESC" } else { "ASC" };
        let query_a = format!(
            "SELECT embedding FROM arc_snapshot WHERE entity_id = {} ORDER BY created_at {}{}",
            entity_id_a, order, limit_clause
        );
        let query_b = format!(
            "SELECT embedding FROM arc_snapshot WHERE entity_id = {} ORDER BY created_at {}{}",
            entity_id_b, order, limit_clause
        );

        #[derive(serde::Deserialize)]
        struct SnapshotEmbedding {
            embedding: Vec<f32>,
        }

        let mut resp_a = self
            .db
            .query(&query_a)
            .await
            .map_err(|e| format!("Failed to fetch snapshots for {}: {}", entity_id_a, e))?;
        let mut resp_b = self
            .db
            .query(&query_b)
            .await
            .map_err(|e| format!("Failed to fetch snapshots for {}: {}", entity_id_b, e))?;

        let mut snaps_a: Vec<SnapshotEmbedding> = resp_a
            .take(0)
            .map_err(|e| format!("Failed to parse snapshots: {}", e))?;
        let mut snaps_b: Vec<SnapshotEmbedding> = resp_b
            .take(0)
            .map_err(|e| format!("Failed to parse snapshots: {}", e))?;

        if snaps_a.is_empty() || snaps_b.is_empty() {
            let missing = if snaps_a.is_empty() {
                entity_id_a
            } else {
                entity_id_b
            };
            return Err(format!(
                "No arc snapshots found for {}. Run BaselineArcSnapshots first.",
                missing
            ));
        }

        // If we fetched DESC for window, reverse to get chronological order
        if window.is_some() {
            snaps_a.reverse();
            snaps_b.reverse();
        }

        // Convergence metric
        let a_first = &snaps_a[0];
        let a_last = &snaps_a[snaps_a.len() - 1];
        let b_first = &snaps_b[0];
        let b_last = &snaps_b[snaps_b.len() - 1];

        let first_sim = cosine_similarity(&a_first.embedding, &b_first.embedding);
        let latest_sim = cosine_similarity(&a_last.embedding, &b_last.embedding);
        let convergence_delta = latest_sim - first_sim;

        // Trajectory similarity
        let delta_a = vector_subtract(&a_last.embedding, &a_first.embedding);
        let delta_b = vector_subtract(&b_last.embedding, &b_first.embedding);
        let trajectory_sim = cosine_similarity(&delta_a, &delta_b);

        let name_a = self.extract_name_from_id(entity_id_a);
        let name_b = self.extract_name_from_id(entity_id_b);

        // Interpret convergence
        let convergence_desc = if convergence_delta.abs() < 0.02 {
            "stable relationship (no significant convergence or divergence)"
        } else if convergence_delta > 0.0 {
            "converging (becoming more similar)"
        } else {
            "diverging (becoming less similar)"
        };

        // Interpret trajectory
        let trajectory_desc = if trajectory_sim > 0.5 {
            "similar trajectories (evolving in the same direction)"
        } else if trajectory_sim < -0.3 {
            "opposite trajectories (evolving in opposing directions)"
        } else {
            "independent trajectories (evolving in unrelated directions)"
        };

        let content = format!(
            "Arc Comparison: {} vs {}\n\n\
             Initial similarity: {:.4}\n\
             Current similarity: {:.4}\n\
             Convergence delta: {:+.4} ({})\n\n\
             Trajectory similarity: {:.4} ({})\n\n\
             Snapshots analyzed: {} vs {}",
            name_a,
            name_b,
            first_sim,
            latest_sim,
            convergence_delta,
            convergence_desc,
            trajectory_sim,
            trajectory_desc,
            snaps_a.len(),
            snaps_b.len()
        );

        let result = EntityResult {
            id: format!("arc-comparison-{}-{}", name_a, name_b),
            entity_type: "arc_comparison".to_string(),
            name: format!("{} vs {}", name_a, name_b),
            content,
            confidence: Some(convergence_delta.abs()),
            last_modified: None,
        };

        let hints = vec![
            format!(
                "Convergence: {:+.4} ({})",
                convergence_delta, convergence_desc
            ),
            format!("Trajectory: {:.4} ({})", trajectory_sim, trajectory_desc),
            "Use ArcHistory on each entity for detailed snapshot-by-snapshot view".to_string(),
        ];

        let token_estimate = result.content.len() / 4 + 50;

        Ok(QueryResponse {
            results: vec![result],
            total: 1,
            next_cursor: None,
            hints,
            token_estimate,
        })
    }

    pub(crate) async fn handle_arc_drift(
        &self,
        entity_type: Option<String>,
        limit: usize,
    ) -> Result<QueryResponse, String> {
        // Build type filter
        let type_filter = match &entity_type {
            Some(et) => {
                if !matches!(et.as_str(), "character" | "knowledge") {
                    return Err(format!(
                        "Invalid entity_type: '{}'. Must be 'character' or 'knowledge'.",
                        et
                    ));
                }
                format!(" AND entity_type = '{}'", et)
            }
            None => String::new(),
        };

        // Aggregate drift by entity
        let agg_query = format!(
            "SELECT entity_id, entity_type, \
                    math::sum(delta_magnitude) AS total_drift, \
                    count() AS snapshot_count \
             FROM arc_snapshot \
             WHERE delta_magnitude IS NOT NONE{} \
             GROUP BY entity_id, entity_type",
            type_filter
        );

        let mut response = self
            .db
            .query(&agg_query)
            .await
            .map_err(|e| format!("Arc drift query failed: {}", e))?;

        #[derive(serde::Deserialize)]
        struct DriftResult {
            entity_id: surrealdb::sql::Thing,
            entity_type: String,
            total_drift: f32,
            snapshot_count: i64,
        }

        let mut drift_results: Vec<DriftResult> = response
            .take(0)
            .map_err(|e| format!("Failed to parse drift results: {}", e))?;

        // Sort by total_drift descending in Rust (SurrealDB ORDER BY with GROUP BY may not apply correctly)
        drift_results.sort_by(|a, b| {
            b.total_drift
                .partial_cmp(&a.total_drift)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        drift_results.truncate(limit);

        if drift_results.is_empty() {
            return Ok(QueryResponse {
                results: vec![],
                total: 0,
                next_cursor: None,
                hints: vec![
                    "No arc drift data found. Entities need at least 2 snapshots to show drift.".to_string(),
                    "Run BaselineArcSnapshots, then update entities to generate snapshots with deltas.".to_string(),
                ],
                token_estimate: 0,
            });
        }

        // For each top result, compute net displacement via first + last snapshot
        let mut entity_results = Vec::new();

        for dr in &drift_results {
            let eid = dr.entity_id.to_string();

            // Fetch first and last embeddings
            // Note: SurrealDB requires ORDER BY fields in the SELECT projection,
            // so we select both embedding and created_at.
            let first_query = format!(
                "SELECT embedding, created_at FROM arc_snapshot WHERE entity_id = {} ORDER BY created_at ASC LIMIT 1",
                eid
            );
            let last_query = format!(
                "SELECT embedding, created_at FROM arc_snapshot WHERE entity_id = {} ORDER BY created_at DESC LIMIT 1",
                eid
            );

            #[derive(serde::Deserialize)]
            struct EmbeddingRow {
                embedding: Vec<f32>,
            }

            let mut first_resp = self
                .db
                .query(&first_query)
                .await
                .map_err(|e| format!("Failed to fetch first embedding for {}: {}", eid, e))?;
            let mut last_resp = self
                .db
                .query(&last_query)
                .await
                .map_err(|e| format!("Failed to fetch last embedding for {}: {}", eid, e))?;

            let first_emb: Vec<EmbeddingRow> = first_resp.take(0).unwrap_or_default();
            let last_emb: Vec<EmbeddingRow> = last_resp.take(0).unwrap_or_default();

            let (displacement, efficiency) = match (first_emb.first(), last_emb.first()) {
                (Some(first), Some(last)) => {
                    let disp = 1.0 - cosine_similarity(&first.embedding, &last.embedding);
                    let eff = if dr.total_drift > 0.0 {
                        disp / dr.total_drift
                    } else {
                        0.0
                    };
                    (disp, eff)
                }
                _ => (0.0, 0.0),
            };

            let entity_name = self.extract_name_from_id(&eid);

            let content =
                format!(
                "Total drift: {:.4} | Net displacement: {:.4} | Efficiency: {:.0}% | Snapshots: {}",
                dr.total_drift, displacement, efficiency * 100.0, dr.snapshot_count
            );

            entity_results.push(EntityResult {
                id: eid,
                entity_type: dr.entity_type.clone(),
                name: entity_name,
                content,
                confidence: Some(dr.total_drift),
                last_modified: None,
            });
        }

        let hints = vec![
            format!("Top {} entities by embedding drift", entity_results.len()),
            "High drift = lots of change. Low efficiency = oscillation/regression.".to_string(),
            "Use ArcHistory on specific entities for detailed evolution view".to_string(),
        ];

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

    pub(crate) async fn handle_arc_moment(
        &self,
        entity_id: &str,
        event_id: Option<String>,
    ) -> Result<QueryResponse, String> {
        let mut response = if let Some(ref eid) = event_id {
            // Resolve event timestamp
            let event_ref = if eid.starts_with("event:") {
                eid.clone()
            } else {
                format!("event:{}", eid)
            };
            let time_query = format!("SELECT VALUE created_at FROM {}", event_ref);

            let mut time_resp = self
                .db
                .query(&time_query)
                .await
                .map_err(|e| format!("Failed to resolve event timestamp: {}", e))?;

            let timestamps: Vec<surrealdb::Datetime> = time_resp.take(0).unwrap_or_default();

            if let Some(event_time) = timestamps.first() {
                // Find nearest-before snapshot using parameter binding for datetime
                let snapshot_query = format!(
                    "SELECT *, event_id.title AS event_title FROM arc_snapshot \
                     WHERE entity_id = {} AND created_at <= $event_time \
                     ORDER BY created_at DESC LIMIT 1",
                    entity_id
                );
                self.db
                    .query(&snapshot_query)
                    .bind(("event_time", event_time.clone()))
                    .await
                    .map_err(|e| format!("Arc moment query failed: {}", e))?
            } else {
                return Err(format!("Event not found or has no timestamp: {}", eid));
            }
        } else {
            // Latest snapshot
            let snapshot_query = format!(
                "SELECT *, event_id.title AS event_title FROM arc_snapshot \
                 WHERE entity_id = {} ORDER BY created_at DESC LIMIT 1",
                entity_id
            );
            self.db
                .query(&snapshot_query)
                .await
                .map_err(|e| format!("Arc moment query failed: {}", e))?
        };

        #[derive(serde::Deserialize)]
        struct MomentSnapshot {
            id: surrealdb::sql::Thing,
            entity_type: String,
            delta_magnitude: Option<f32>,
            event_title: Option<String>,
            created_at: String,
        }

        let snapshots: Vec<MomentSnapshot> = response
            .take(0)
            .map_err(|e| format!("Failed to parse moment snapshot: {}", e))?;

        if snapshots.is_empty() {
            let context = if event_id.is_some() {
                "at that event"
            } else {
                "at all"
            };
            return Err(format!(
                "No arc snapshot found for {} {}. Run BaselineArcSnapshots first.",
                entity_id, context
            ));
        }

        let snap = &snapshots[0];
        let entity_name = self.extract_name_from_id(entity_id);

        let event_info = snap
            .event_title
            .as_ref()
            .map(|t| format!(" (event: {})", t))
            .unwrap_or_default();

        let delta_str = snap
            .delta_magnitude
            .map(|d| format!("{:.4}", d))
            .unwrap_or_else(|| "baseline".to_string());

        let moment_desc = if event_id.is_some() {
            "at event"
        } else {
            "latest"
        };

        let content = format!(
            "{} {} snapshot: delta={}, captured at {}{}",
            entity_name, moment_desc, delta_str, snap.created_at, event_info
        );

        let result = EntityResult {
            id: snap.id.to_string(),
            entity_type: snap.entity_type.clone(),
            name: format!("{} ({})", entity_name, moment_desc),
            content,
            confidence: snap.delta_magnitude,
            last_modified: Some(snap.created_at.clone()),
        };

        let hints = vec![
            format!("Snapshot for {} at {}", entity_id, snap.created_at),
            "Use ArcHistory for full timeline view".to_string(),
            "Use ArcComparison to compare with another entity at the same point".to_string(),
        ];

        let token_estimate = result.content.len() / 4 + 30;

        Ok(QueryResponse {
            results: vec![result],
            total: 1,
            next_cursor: None,
            hints,
            token_estimate,
        })
    }

    pub(crate) async fn handle_embedding_health(&self) -> Result<QueryResponse, String> {
        let embeddable_tables = [
            ("character", "name"),
            ("location", "name"),
            ("event", "title"),
            ("scene", "title"),
            ("knowledge", "fact"),
            ("perceives", "perception"),
            ("relates_to", "rel_type"),
        ];

        let mut results = Vec::new();
        let mut overall_total = 0usize;
        let mut overall_embedded = 0usize;
        let mut overall_stale = 0usize;

        for (table, _) in &embeddable_tables {
            // SurrealDB GROUP ALL with conditional count can be unreliable,
            // so use separate queries for reliability
            let total_query = format!("SELECT count() AS cnt FROM {} GROUP ALL", table);
            let embedded_query = format!(
                "SELECT count() AS cnt FROM {} WHERE embedding IS NOT NONE GROUP ALL",
                table
            );
            let stale_query = format!("SELECT count() AS cnt FROM {} WHERE embedding IS NOT NONE AND embedding_stale = true GROUP ALL", table);

            #[derive(serde::Deserialize)]
            struct CountResult {
                cnt: i64,
            }

            let total: i64 = {
                let mut r = self
                    .db
                    .query(&total_query)
                    .await
                    .map_err(|e| format!("Failed to count {}: {}", table, e))?;
                let res: Option<CountResult> = r.take(0).unwrap_or(None);
                res.map(|c| c.cnt).unwrap_or(0)
            };

            let embedded: i64 = {
                let mut r = self
                    .db
                    .query(&embedded_query)
                    .await
                    .map_err(|e| format!("Failed to count embedded {}: {}", table, e))?;
                let res: Option<CountResult> = r.take(0).unwrap_or(None);
                res.map(|c| c.cnt).unwrap_or(0)
            };

            let stale: i64 = {
                let mut r = self
                    .db
                    .query(&stale_query)
                    .await
                    .map_err(|e| format!("Failed to count stale {}: {}", table, e))?;
                let res: Option<CountResult> = r.take(0).unwrap_or(None);
                res.map(|c| c.cnt).unwrap_or(0)
            };

            let coverage = if total > 0 {
                (embedded as f32 / total as f32) * 100.0
            } else {
                0.0
            };

            overall_total += total as usize;
            overall_embedded += embedded as usize;
            overall_stale += stale as usize;

            results.push(EntityResult {
                id: format!("health:{}", table),
                entity_type: "embedding_health".to_string(),
                name: table.to_string(),
                content: format!(
                    "{}/{} embedded ({:.0}%), {} stale",
                    embedded, total, coverage, stale
                ),
                confidence: Some(coverage / 100.0),
                last_modified: None,
            });
        }

        // Summary result
        let overall_coverage = if overall_total > 0 {
            (overall_embedded as f32 / overall_total as f32) * 100.0
        } else {
            0.0
        };

        results.push(EntityResult {
            id: "health:overall".to_string(),
            entity_type: "embedding_health".to_string(),
            name: "OVERALL".to_string(),
            content: format!(
                "{}/{} embedded ({:.0}%), {} stale across all types",
                overall_embedded, overall_total, overall_coverage, overall_stale
            ),
            confidence: Some(overall_coverage / 100.0),
            last_modified: None,
        });

        let total = results.len();

        let mut hints = vec![];
        if overall_embedded < overall_total {
            hints.push("Run BackfillEmbeddings to generate missing embeddings".to_string());
        }
        if overall_stale > 0 {
            hints.push(
                "Stale embeddings will regenerate automatically on next relevant mutation"
                    .to_string(),
            );
        }
        if overall_embedded == overall_total && overall_stale == 0 {
            hints.push("All embeddings are up to date".to_string());
        }

        let token_estimate = self.estimate_tokens_from_results(&results);

        Ok(QueryResponse {
            results,
            total,
            next_cursor: None,
            hints,
            token_estimate,
        })
    }

    pub(crate) async fn handle_what_if(
        &self,
        character_id: &str,
        fact_id: &str,
        certainty: Option<&str>,
        _source_character: Option<&str>,
    ) -> Result<QueryResponse, String> {
        use crate::embedding::composite::{character_composite, knowledge_composite};
        use crate::models::knowledge::get_current_knowledge;

        // Step 1: Validate embedding service
        if !self.embedding_service.is_available() {
            return Err(
                "WhatIf unavailable — embedding model not loaded. Run BackfillEmbeddings first."
                    .to_string(),
            );
        }

        // Normalize IDs
        let char_ref = if character_id.starts_with("character:") {
            character_id.to_string()
        } else {
            format!("character:{}", character_id)
        };
        let fact_ref = if fact_id.starts_with("knowledge:") {
            fact_id.to_string()
        } else {
            format!("knowledge:{}", fact_id)
        };
        let char_key = char_ref.strip_prefix("character:").unwrap_or(character_id);
        let fact_key = fact_ref.strip_prefix("knowledge:").unwrap_or(fact_id);
        let certainty = certainty.unwrap_or("knows");

        // Step 2: Fetch character
        let mut char_resp = self
            .db
            .query(format!("SELECT * FROM {}", char_ref))
            .await
            .map_err(|e| format!("Failed to fetch character: {}", e))?;

        let character: Option<crate::models::Character> = char_resp
            .take(0)
            .map_err(|e| format!("Failed to parse character: {}", e))?;

        let character = character.ok_or_else(|| format!("Character not found: {}", char_ref))?;

        // Fetch fact
        #[derive(serde::Deserialize)]
        struct FactRecord {
            fact: String,
            character_name: Option<String>,
        }

        let fact_query = format!(
            "SELECT fact, character.name AS character_name FROM {}",
            fact_ref
        );

        let mut fact_resp = self
            .db
            .query(&fact_query)
            .await
            .map_err(|e| format!("Failed to fetch fact: {}", e))?;

        let facts: Vec<FactRecord> = fact_resp
            .take(0)
            .map_err(|e| format!("Failed to parse fact: {}", e))?;

        let fact_record = facts
            .into_iter()
            .next()
            .ok_or_else(|| format!("Knowledge fact not found: {}", fact_ref))?;

        // Step 3: Check current knowledge state
        let current_knowledge = get_current_knowledge(&self.db, char_key, &fact_ref)
            .await
            .map_err(|e| format!("Failed to check current knowledge: {}", e))?;

        let already_known_note = if let Some(ref state) = current_knowledge {
            format!("Already known with certainty: {:?}", state.certainty)
        } else {
            "Not currently known by this character".to_string()
        };

        // Step 4: Fetch current embedding
        let emb_query = format!("SELECT VALUE embedding FROM {}", char_ref);
        let mut emb_resp = self
            .db
            .query(&emb_query)
            .await
            .map_err(|e| format!("Failed to fetch character embedding: {}", e))?;

        let embeddings: Vec<Option<Vec<f32>>> = emb_resp.take(0).unwrap_or_default();
        let current_embedding = embeddings.into_iter().next().flatten().ok_or_else(|| {
            format!(
                "Character {} has no embedding. Run BackfillEmbeddings first.",
                char_ref
            )
        })?;

        // Step 5: Build hypothetical composite
        // Fetch relationships
        let mut rel_result = self.db
            .query(format!(
                "SELECT ->relates_to->character.{{name, relationship_type}} AS relationships FROM {}",
                char_ref
            ))
            .await
            .map_err(|e| format!("Failed to fetch relationships: {}", e))?;

        #[derive(serde::Deserialize)]
        struct RelResult {
            relationships: Option<Vec<serde_json::Value>>,
        }

        let rel_data: Option<RelResult> = rel_result.take(0).unwrap_or(None);
        let relationships: Vec<(String, String)> = rel_data
            .and_then(|r| r.relationships)
            .unwrap_or_default()
            .iter()
            .filter_map(|r| {
                let rel_type = r.get("relationship_type")?.as_str()?;
                let name = r.get("name")?.as_str()?;
                Some((rel_type.to_string(), name.to_string()))
            })
            .collect();

        // Fetch perceptions
        #[derive(serde::Deserialize)]
        struct PerceptionRecord {
            target_name: Option<String>,
            perception: Option<String>,
        }

        let mut perc_result = self
            .db
            .query(format!(
                "SELECT out.name AS target_name, perception FROM perceives WHERE in = {}",
                char_ref
            ))
            .await
            .map_err(|e| format!("Failed to fetch perceptions: {}", e))?;

        let perc_records: Vec<PerceptionRecord> = perc_result.take(0).unwrap_or_default();
        let perceptions: Vec<(String, String)> = perc_records
            .into_iter()
            .filter_map(|p| Some((p.target_name?, p.perception?)))
            .collect();

        // Build current composite + append knowledge
        let current_composite = character_composite(&character, &relationships, &perceptions);
        let knowledge_text =
            knowledge_composite(&fact_record.fact, &character.name, certainty, None);
        let hypothetical_composite = format!(
            "{}. Additionally, {}",
            current_composite.trim_end_matches('.'),
            knowledge_text
        );

        // Step 6: Embed hypothetical composite
        let hypothetical_embedding = self
            .embedding_service
            .embed_text(&hypothetical_composite)
            .await
            .map_err(|e| format!("Failed to embed hypothetical composite: {}", e))?;

        // Step 7: Compute delta
        let delta = 1.0 - cosine_similarity(&current_embedding, &hypothetical_embedding);

        let impact_label = if delta < 0.02 {
            "negligible"
        } else if delta < 0.05 {
            "minor"
        } else if delta < 0.10 {
            "moderate"
        } else if delta < 0.20 {
            "major"
        } else {
            "transformative"
        };

        // Step 8: Conflict detection
        let fact_embedding = self
            .embedding_service
            .embed_text(&fact_record.fact)
            .await
            .map_err(|e| format!("Failed to embed fact text: {}", e))?;

        let knowledge_query = format!(
            "SELECT id, fact, embedding FROM knowledge WHERE character = {} AND embedding IS NOT NONE",
            char_ref
        );

        let mut knowledge_resp = self
            .db
            .query(&knowledge_query)
            .await
            .map_err(|e| format!("Failed to fetch character knowledge: {}", e))?;

        #[derive(serde::Deserialize)]
        struct KnowledgeWithEmbedding {
            id: surrealdb::sql::Thing,
            fact: String,
            embedding: Vec<f32>,
        }

        let existing_knowledge: Vec<KnowledgeWithEmbedding> =
            knowledge_resp.take(0).unwrap_or_default();

        let mut conflicts: Vec<EntityResult> = Vec::new();
        for k in &existing_knowledge {
            let sim = cosine_similarity(&fact_embedding, &k.embedding);
            if sim > 0.7 {
                conflicts.push(EntityResult {
                    id: k.id.to_string(),
                    entity_type: "what_if_conflict".to_string(),
                    name: format!("Potential conflict (similarity: {:.3})", sim),
                    content: format!(
                        "Existing knowledge: \"{}\" | Similarity to new fact: {:.4}",
                        k.fact, sim
                    ),
                    confidence: Some(sim),
                    last_modified: None,
                });
            }
        }

        // Step 9: Cascade preview
        let cascade_query = format!(
            "SELECT type::string(in) AS observer_id, in.name AS observer_name \
             FROM perceives WHERE out = {} AND embedding IS NOT NONE",
            char_ref
        );

        let mut cascade_resp = self
            .db
            .query(&cascade_query)
            .await
            .map_err(|e| format!("Failed to fetch cascade targets: {}", e))?;

        #[derive(serde::Deserialize)]
        struct CascadeRecord {
            observer_id: String,
            observer_name: Option<String>,
        }

        let cascade_records: Vec<CascadeRecord> = cascade_resp.take(0).unwrap_or_default();

        // Step 10: Build results
        let mut results: Vec<EntityResult> = Vec::new();

        // Primary result
        let fact_owner = fact_record.character_name.as_deref().unwrap_or("someone");
        let primary_content = format!(
            "Embedding delta: {:.4} ({} impact)\n\
             Character: {} would learn: \"{}\"\n\
             Certainty: {} | Fact originally from: {}\n\
             Current status: {}\n\
             Conflicts detected: {} | Perspectives that would go stale: {}",
            delta,
            impact_label,
            character.name,
            fact_record.fact,
            certainty,
            fact_owner,
            already_known_note,
            conflicts.len(),
            cascade_records.len()
        );

        results.push(EntityResult {
            id: format!("what-if-{}-{}", char_key, fact_key),
            entity_type: "what_if_analysis".to_string(),
            name: format!(
                "What if {} learned \"{}\"?",
                character.name, fact_record.fact
            ),
            content: primary_content,
            confidence: Some(delta),
            last_modified: None,
        });

        // Add conflicts
        results.extend(conflicts.iter().cloned());

        // Add cascade result if any
        if !cascade_records.is_empty() {
            let observer_names: Vec<String> = cascade_records
                .iter()
                .map(|c| {
                    c.observer_name
                        .clone()
                        .unwrap_or_else(|| c.observer_id.clone())
                })
                .collect();

            results.push(EntityResult {
                id: format!("what-if-cascade-{}", char_key),
                entity_type: "what_if_cascade".to_string(),
                name: format!("{} perspectives would go stale", cascade_records.len()),
                content: format!(
                    "These characters' perspectives on {} would need re-embedding: {}",
                    character.name,
                    observer_names.join(", ")
                ),
                confidence: Some(cascade_records.len() as f32 / 10.0),
                last_modified: None,
            });
        }

        let total = results.len();
        let token_estimate = results.iter().map(|r| r.content.len() / 4 + 30).sum();

        let mut hints = vec![format!("Impact: {} (delta={:.4})", impact_label, delta)];
        if current_knowledge.is_some() {
            hints.push(format!(
                "Note: {} already knows this fact — learning would update certainty",
                character.name
            ));
        }
        if !conflicts.is_empty() {
            hints.push(format!(
                "{} semantically similar existing knowledge entries — potential contradictions",
                conflicts.len()
            ));
        }
        if !cascade_records.is_empty() {
            hints.push(format!(
                "{} observer perspectives would go stale",
                cascade_records.len()
            ));
        }
        hints.push("This is a preview — use RecordKnowledge to commit.".to_string());

        Ok(QueryResponse {
            results,
            total,
            next_cursor: None,
            hints,
            token_estimate,
        })
    }
}
