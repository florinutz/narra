use crate::mcp::NarraServer;
use crate::mcp::{EntityResult, QueryResponse};
use crate::models::knowledge::find_knowledge_conflicts;
use crate::services::EntityType;
use crate::utils::math::cosine_similarity;

use super::parse_entity_types;

impl NarraServer {
    pub(crate) async fn handle_dramatic_irony_report(
        &self,
        character_id: Option<String>,
        min_scene_threshold: usize,
    ) -> Result<QueryResponse, String> {
        use crate::services::IronyService;

        // Create IronyService
        let irony_service = IronyService::new(self.db.clone());

        // Call generate_report
        let report = irony_service
            .generate_report(character_id.as_deref(), min_scene_threshold)
            .await
            .map_err(|e| format!("Dramatic irony report failed: {}", e))?;

        // Convert IronyReport to QueryResponse
        // Each KnowledgeAsymmetry becomes an EntityResult
        let entity_results: Vec<EntityResult> = report
            .asymmetries
            .iter()
            .map(|asym| {
                let content = format!(
                    "{} doesn't know: \"{}\" ({} scenes since {} learned it via {})",
                    asym.unknowing_character_name,
                    asym.fact,
                    asym.scenes_since,
                    asym.knowing_character_name,
                    asym.learning_method
                );

                // Map signal_strength to confidence score
                let confidence = match asym.signal_strength.as_str() {
                    "high" => 0.9,
                    "medium" => 0.6,
                    _ => 0.3,
                };

                EntityResult {
                    id: format!(
                        "irony-{}-{}",
                        asym.knowing_character_id, asym.unknowing_character_id
                    ),
                    entity_type: "dramatic_irony".to_string(),
                    name: format!(
                        "{} knows, {} doesn't",
                        asym.knowing_character_name, asym.unknowing_character_name
                    ),
                    content,
                    confidence: Some(confidence),
                    last_modified: None,
                }
            })
            .collect();

        let mut hints = vec![
            format!("Found {} knowledge asymmetries", report.total_asymmetries),
            format!("High-signal opportunities: {}", report.high_signal_count),
        ];

        // Add narrative_opportunities from report
        for opportunity in &report.narrative_opportunities {
            hints.push(opportunity.clone());
        }

        let token_estimate = entity_results
            .iter()
            .map(|r| r.content.len() / 4 + 30)
            .sum();

        Ok(QueryResponse {
            results: entity_results,
            total: report.total_asymmetries,
            next_cursor: None,
            hints,
            token_estimate,
            truncated: None,
        })
    }

    pub(crate) async fn handle_knowledge_asymmetries(
        &self,
        character_a: &str,
        character_b: &str,
    ) -> Result<QueryResponse, String> {
        use crate::services::IronyService;

        let irony_service = IronyService::new(self.db.clone());

        let asymmetries = irony_service
            .detect_asymmetries(character_a, character_b)
            .await
            .map_err(|e| format!("Knowledge asymmetries failed: {}", e))?;

        let total = asymmetries.len();

        let entity_results: Vec<EntityResult> = asymmetries
            .iter()
            .map(|asym| {
                let mut content = format!(
                    "{} doesn't know: \"{}\" ({} scenes since {} learned it via {})",
                    asym.unknowing_character_name,
                    asym.fact,
                    asym.scenes_since,
                    asym.knowing_character_name,
                    asym.learning_method
                );

                if let Some(tension) = asym.tension_level {
                    content.push_str(&format!(" | tension: {}", tension));
                }
                if let Some(ref feelings) = asym.feelings {
                    content.push_str(&format!(" | feelings: {}", feelings));
                }

                let confidence = match asym.signal_strength.as_str() {
                    "high" => 0.9,
                    "medium" => 0.6,
                    _ => 0.3,
                };

                EntityResult {
                    id: format!(
                        "asymmetry-{}-{}-{}",
                        asym.knowing_character_id,
                        asym.unknowing_character_id,
                        asym.about.replace(':', "_")
                    ),
                    entity_type: "knowledge_asymmetry".to_string(),
                    name: format!(
                        "{} knows, {} doesn't: {}",
                        asym.knowing_character_name, asym.unknowing_character_name, asym.about
                    ),
                    content,
                    confidence: Some(confidence),
                    last_modified: None,
                }
            })
            .collect();

        let hints = vec![
            format!(
                "Found {} asymmetries between {} and {}",
                total, character_a, character_b
            ),
            "Sorted by dramatic_weight (scenes_since + tension + enforcement)".to_string(),
        ];

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
            truncated: None,
        })
    }

    pub(crate) async fn handle_thematic_clustering(
        &self,
        entity_types: Option<Vec<String>>,
        num_themes: Option<usize>,
    ) -> Result<QueryResponse, String> {
        use crate::services::ClusteringService;

        // Create ClusteringService
        let clustering_service = ClusteringService::new(self.db.clone());

        // Convert entity_types strings to EntityType enum (default to embeddable)
        let type_filter = {
            let parsed = parse_entity_types(entity_types);
            if parsed.is_empty() {
                EntityType::embeddable()
            } else {
                parsed
            }
        };

        // Call discover_themes
        let clustering_result = clustering_service
            .discover_themes(type_filter, num_themes)
            .await
            .map_err(|e| format!("Thematic clustering failed: {}", e))?;

        // Convert ClusteringResult to QueryResponse
        // Each ThemeCluster becomes an EntityResult
        let total_entities = clustering_result.total_entities;
        let entity_results: Vec<EntityResult> = clustering_result
            .clusters
            .iter()
            .map(|cluster| {
                // Format member list with entity types
                let member_list: String = cluster
                    .members
                    .iter()
                    .take(5) // Show first 5 members
                    .map(|m| format!("{} ({})", m.name, m.entity_type))
                    .collect::<Vec<_>>()
                    .join(", ");

                let content = if cluster.members.len() > 5 {
                    format!("{} ... and {} more", member_list, cluster.members.len() - 5)
                } else {
                    member_list
                };

                let confidence = if total_entities > 0 {
                    cluster.member_count as f32 / total_entities as f32
                } else {
                    0.0
                };

                EntityResult {
                    id: format!("theme-{}", cluster.cluster_id),
                    entity_type: "theme_cluster".to_string(),
                    name: cluster.label.clone(),
                    content,
                    confidence: Some(confidence),
                    last_modified: None,
                }
            })
            .collect();

        let mut hints = vec![
            "Clusters reveal emergent thematic groupings from embeddings".to_string(),
            format!(
                "Discovered {} themes from {} entities",
                entity_results.len(),
                total_entities
            ),
        ];

        if clustering_result.entities_without_embeddings > 0 {
            hints.push(format!(
                "Warning: {} entities lack embeddings - run backfill_embeddings",
                clustering_result.entities_without_embeddings
            ));
        }

        let token_estimate = entity_results
            .iter()
            .map(|r| r.content.len() / 4 + 30)
            .sum();

        Ok(QueryResponse {
            results: entity_results,
            total: clustering_result.clusters.len(),
            next_cursor: None,
            hints,
            token_estimate,
            truncated: None,
        })
    }

    // ========================================================================
    // Tier 4: Narrative Intelligence Tools
    // ========================================================================

    pub(crate) async fn handle_thematic_gaps(
        &self,
        min_cluster_size: usize,
        expected_types: Option<Vec<String>>,
    ) -> Result<QueryResponse, String> {
        use crate::services::ClusteringService;

        let clustering_service = ClusteringService::new(self.db.clone());

        // Run clustering across all embeddable types
        let clustering_result = clustering_service
            .discover_themes(EntityType::embeddable(), None)
            .await
            .map_err(|e| format!("Thematic gap analysis failed: {}", e))?;

        let expected =
            expected_types.unwrap_or_else(|| vec!["character".to_string(), "event".to_string()]);

        let total_clusters = clustering_result.clusters.len();
        let mut gap_results: Vec<(f32, EntityResult)> = Vec::new();

        for cluster in &clustering_result.clusters {
            if cluster.member_count < min_cluster_size {
                continue;
            }

            // Count members by entity_type
            let mut type_counts: std::collections::HashMap<String, usize> =
                std::collections::HashMap::new();
            for member in &cluster.members {
                *type_counts.entry(member.entity_type.clone()).or_insert(0) += 1;
            }

            // Find missing expected types
            let missing: Vec<&String> = expected
                .iter()
                .filter(|t| !type_counts.contains_key(t.as_str()))
                .collect();

            if missing.is_empty() {
                continue;
            }

            let gap_severity = missing.len() as f32 / expected.len() as f32;

            // Generate narrative interpretation per gap pattern
            let present_types: Vec<String> = type_counts.keys().cloned().collect();
            let has_type = |t: &str| type_counts.contains_key(t);

            let interpretation = if has_type("event") && !has_type("character") {
                "Events without protagonists — who drives this theme?"
            } else if has_type("character") && !has_type("event") {
                "Characters without plot — what happens to embody this?"
            } else if !has_type("scene") && missing.contains(&&"scene".to_string()) {
                "No scenes ground this theme — where does it come to life?"
            } else if has_type("character")
                && !has_type("knowledge")
                && missing.contains(&&"knowledge".to_string())
            {
                "Characters lack associated knowledge — what do they know?"
            } else {
                "Theme has structural gaps — consider adding missing entity types."
            };

            // Top member names for content
            let top_names: Vec<String> = cluster
                .members
                .iter()
                .take(5)
                .map(|m| format!("{} ({})", m.name, m.entity_type))
                .collect();

            let missing_str: Vec<&str> = missing.iter().map(|s| s.as_str()).collect();

            let content = format!(
                "Members: {} | Present types: {} | Missing types: {} | {}\nTop members: {}",
                cluster.member_count,
                present_types.join(", "),
                missing_str.join(", "),
                interpretation,
                top_names.join(", ")
            );

            gap_results.push((
                gap_severity,
                EntityResult {
                    id: format!("gap-{}", cluster.cluster_id),
                    entity_type: "thematic_gap".to_string(),
                    name: cluster.label.clone(),
                    content,
                    confidence: Some(gap_severity),
                    last_modified: None,
                },
            ));
        }

        // Sort by gap severity descending
        gap_results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let entity_results: Vec<EntityResult> = gap_results.into_iter().map(|(_, r)| r).collect();
        let gaps_found = entity_results.len();

        let hints = vec![
            format!(
                "Analyzed {} clusters, {} with gaps",
                total_clusters, gaps_found
            ),
            format!("Expected types per cluster: {}", expected.join(", ")),
            "Use ThematicClustering for raw cluster data".to_string(),
        ];

        let token_estimate = entity_results
            .iter()
            .map(|r| r.content.len() / 4 + 30)
            .sum();

        Ok(QueryResponse {
            results: entity_results,
            total: gaps_found,
            next_cursor: None,
            hints,
            token_estimate,
            truncated: None,
        })
    }

    pub(crate) async fn handle_unresolved_tensions(
        &self,
        limit: usize,
        min_asymmetry: f32,
        max_shared_scenes: Option<usize>,
    ) -> Result<QueryResponse, String> {
        // Step 1: Fetch all perceives edges with embeddings
        let edge_query = "SELECT type::string(in) AS observer_id, type::string(out) AS target_id, \
                          in.name AS observer_name, out.name AS target_name, embedding \
                          FROM perceives WHERE embedding IS NOT NONE";

        let mut edge_resp = self
            .db
            .query(edge_query)
            .await
            .map_err(|e| format!("Failed to fetch perceives edges: {}", e))?;

        #[derive(serde::Deserialize)]
        struct PerceivesEdgeData {
            observer_id: String,
            target_id: String,
            observer_name: Option<String>,
            target_name: Option<String>,
            embedding: Vec<f32>,
        }

        let edges: Vec<PerceivesEdgeData> = edge_resp
            .take(0)
            .map_err(|e| format!("Failed to parse perceives edges: {}", e))?;

        if edges.is_empty() {
            return Ok(QueryResponse {
                results: vec![],
                total: 0,
                next_cursor: None,
                hints: vec![
                    "No perceives edges with embeddings found.".to_string(),
                    "Create relationships and run BackfillEmbeddings to enable tension analysis."
                        .to_string(),
                ],
                token_estimate: 0,
                truncated: None,
            });
        }

        // Step 2: Build map (observer, target) -> (names, embedding)
        // Find bidirectional pairs
        let mut edge_map: std::collections::HashMap<(String, String), (String, String, Vec<f32>)> =
            std::collections::HashMap::new();

        for edge in &edges {
            edge_map.insert(
                (edge.observer_id.clone(), edge.target_id.clone()),
                (
                    edge.observer_name
                        .clone()
                        .unwrap_or_else(|| edge.observer_id.clone()),
                    edge.target_name
                        .clone()
                        .unwrap_or_else(|| edge.target_id.clone()),
                    edge.embedding.clone(),
                ),
            );
        }

        // Step 3: Find bidirectional pairs and compute asymmetry
        struct TensionPair {
            char_a_id: String,
            char_b_id: String,
            char_a_name: String,
            char_b_name: String,
            asymmetry: f32,
            shared_scene_count: usize,
            tension_score: f32,
        }

        let mut seen_pairs: std::collections::HashSet<(String, String)> =
            std::collections::HashSet::new();
        let mut tension_pairs: Vec<TensionPair> = Vec::new();

        // Collect all character IDs involved in bidirectional pairs for scene query
        let mut all_char_ids: std::collections::HashSet<String> = std::collections::HashSet::new();

        for ((obs, tgt), (obs_name, _tgt_name, a_emb)) in &edge_map {
            // Canonical key to avoid processing same pair twice
            let canonical = if obs < tgt {
                (obs.clone(), tgt.clone())
            } else {
                (tgt.clone(), obs.clone())
            };

            if seen_pairs.contains(&canonical) {
                continue;
            }

            // Check if reverse edge exists
            if let Some((_b_obs_name, b_tgt_name, b_emb)) =
                edge_map.get(&(tgt.clone(), obs.clone()))
            {
                let asymmetry = 1.0 - cosine_similarity(a_emb, b_emb);

                if asymmetry >= min_asymmetry {
                    all_char_ids.insert(obs.clone());
                    all_char_ids.insert(tgt.clone());

                    tension_pairs.push(TensionPair {
                        char_a_id: obs.clone(),
                        char_b_id: tgt.clone(),
                        char_a_name: obs_name.clone(),
                        char_b_name: b_tgt_name.clone(),
                        asymmetry,
                        shared_scene_count: 0, // filled below
                        tension_score: 0.0,    // filled below
                    });
                }

                seen_pairs.insert(canonical);
            }
        }

        if tension_pairs.is_empty() {
            return Ok(QueryResponse {
                results: vec![],
                total: 0,
                next_cursor: None,
                hints: vec![
                    "No bidirectional perception pairs found above the asymmetry threshold."
                        .to_string(),
                    format!("Try lowering min_asymmetry (current: {:.2})", min_asymmetry),
                ],
                token_estimate: 0,
                truncated: None,
            });
        }

        // Step 4: Batch-fetch shared scenes
        if !all_char_ids.is_empty() {
            let char_id_list: Vec<String> = all_char_ids.iter().cloned().collect();
            let char_csv = char_id_list.join(", ");

            let scene_query = format!(
                "SELECT type::string(in) AS char_id, type::string(out) AS scene_id \
                 FROM participates_in WHERE in IN [{}]",
                char_csv
            );

            let mut scene_resp = self
                .db
                .query(&scene_query)
                .await
                .map_err(|e| format!("Failed to fetch scene participation: {}", e))?;

            #[derive(serde::Deserialize)]
            struct ParticipationRecord {
                char_id: String,
                scene_id: String,
            }

            let participations: Vec<ParticipationRecord> = scene_resp.take(0).unwrap_or_default();

            // Build char_id -> set of scene_ids
            let mut char_scenes: std::collections::HashMap<
                String,
                std::collections::HashSet<String>,
            > = std::collections::HashMap::new();

            for p in &participations {
                char_scenes
                    .entry(p.char_id.clone())
                    .or_default()
                    .insert(p.scene_id.clone());
            }

            // Compute shared scene counts per pair
            for pair in &mut tension_pairs {
                let a_scenes = char_scenes.get(&pair.char_a_id);
                let b_scenes = char_scenes.get(&pair.char_b_id);

                pair.shared_scene_count = match (a_scenes, b_scenes) {
                    (Some(a), Some(b)) => a.intersection(b).count(),
                    _ => 0,
                };
            }
        }

        // Step 5: Apply max_shared_scenes filter and compute tension score
        if let Some(max_scenes) = max_shared_scenes {
            tension_pairs.retain(|p| p.shared_scene_count <= max_scenes);
        }

        for pair in &mut tension_pairs {
            pair.tension_score = pair.asymmetry * (1.0 / (pair.shared_scene_count as f32 + 1.0));
        }

        // Step 6: Sort by tension_score desc, truncate
        tension_pairs.sort_by(|a, b| {
            b.tension_score
                .partial_cmp(&a.tension_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        tension_pairs.truncate(limit);

        let entity_results: Vec<EntityResult> = tension_pairs
            .iter()
            .map(|pair| {
                let asymmetry_label = if pair.asymmetry < 0.05 {
                    "negligible"
                } else if pair.asymmetry < 0.15 {
                    "mild"
                } else if pair.asymmetry < 0.30 {
                    "significant"
                } else if pair.asymmetry < 0.50 {
                    "major"
                } else {
                    "extreme"
                };

                let content = format!(
                    "Asymmetry: {:.4} ({}) | Shared scenes: {} | Tension score: {:.4}\n\
                 {} and {} see each other very differently but have {} shared scenes. \
                 This is a scene waiting to be written.",
                    pair.asymmetry,
                    asymmetry_label,
                    pair.shared_scene_count,
                    pair.tension_score,
                    pair.char_a_name,
                    pair.char_b_name,
                    if pair.shared_scene_count == 0 {
                        "no"
                    } else {
                        "few"
                    }
                );

                EntityResult {
                    id: format!(
                        "tension-{}-{}",
                        pair.char_a_id.replace(':', "_"),
                        pair.char_b_id.replace(':', "_")
                    ),
                    entity_type: "unresolved_tension".to_string(),
                    name: format!("{} ↔ {}", pair.char_a_name, pair.char_b_name),
                    content,
                    confidence: Some(pair.tension_score),
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
            format!("{} unresolved tensions found", total),
            "Scoring: asymmetry × (1 / (shared_scenes + 1)) — high asymmetry + few scenes = high tension".to_string(),
            "Use PerceptionGap on specific pairs for detailed analysis".to_string(),
        ];

        Ok(QueryResponse {
            results: entity_results,
            total,
            next_cursor: None,
            hints,
            token_estimate,
            truncated: None,
        })
    }

    pub(crate) async fn handle_knowledge_conflicts(
        &self,
        character_id: Option<String>,
        limit: usize,
    ) -> Result<QueryResponse, String> {
        let conflicts = find_knowledge_conflicts(&self.db)
            .await
            .map_err(|e| format!("Knowledge conflicts query failed: {}", e))?;

        let mut entity_results: Vec<EntityResult> = Vec::new();

        for conflict in &conflicts {
            for state in &conflict.conflicting_states {
                // Filter by character_id if specified
                if let Some(ref filter_id) = character_id {
                    let filter_key = filter_id.split(':').nth(1).unwrap_or(filter_id);
                    let state_key = state
                        .character_id
                        .split(':')
                        .nth(1)
                        .unwrap_or(&state.character_id);
                    if filter_key != state_key {
                        continue;
                    }
                }

                let truth_info = state
                    .truth_value
                    .as_deref()
                    .map(|tv| format!(" (actual truth: {})", tv))
                    .unwrap_or_default();

                let content = format!(
                    "Character {} believes wrongly about {} with certainty {:?}{}",
                    state.character_id, conflict.target, state.certainty, truth_info
                );

                entity_results.push(EntityResult {
                    id: format!(
                        "conflict-{}-{}",
                        state.character_id.replace(':', "_"),
                        conflict.target.replace(':', "_")
                    ),
                    entity_type: "knowledge_conflict".to_string(),
                    name: format!("{} wrong about {}", state.character_id, conflict.target),
                    content,
                    confidence: Some(1.0),
                    last_modified: None,
                });
            }
        }

        entity_results.truncate(limit);

        let total = entity_results.len();
        let token_estimate = entity_results
            .iter()
            .map(|r| r.content.len() / 4 + 30)
            .sum();

        let hints = vec![
            format!(
                "Found {} knowledge conflicts (BelievesWrongly states)",
                total
            ),
            "These represent characters who are certain about something that is factually wrong"
                .to_string(),
            "Use RecordKnowledge with certainty='believes_wrongly' and truth_value to create more"
                .to_string(),
        ];

        Ok(QueryResponse {
            results: entity_results,
            total,
            next_cursor: None,
            hints,
            token_estimate,
            truncated: None,
        })
    }

    // ========================================================================
    // Narrative Phase Detection
    // ========================================================================

    pub(crate) async fn handle_detect_phases(
        &self,
        entity_types: Option<Vec<String>>,
        num_phases: Option<usize>,
        content_weight: Option<f32>,
        neighborhood_weight: Option<f32>,
        temporal_weight: Option<f32>,
        save: bool,
    ) -> Result<QueryResponse, String> {
        use crate::services::{PhaseWeights, TemporalService};

        let type_filter = {
            let parsed = parse_entity_types(entity_types);
            if parsed.is_empty() {
                EntityType::embeddable()
            } else {
                parsed
            }
        };

        let weights = if content_weight.is_some()
            || neighborhood_weight.is_some()
            || temporal_weight.is_some()
        {
            Some(PhaseWeights {
                content: content_weight.unwrap_or(0.6),
                neighborhood: neighborhood_weight.unwrap_or(0.25),
                temporal: temporal_weight.unwrap_or(0.15),
            })
        } else {
            None
        };

        let service = TemporalService::new(self.db.clone());
        let result = service
            .detect_phases(type_filter, num_phases, weights)
            .await
            .map_err(|e| format!("Phase detection failed: {}", e))?;

        if save {
            service
                .save_phases(&result)
                .await
                .map_err(|e| format!("Failed to save phases: {}", e))?;
        }

        let entity_results: Vec<EntityResult> = result
            .phases
            .iter()
            .map(|phase| {
                let type_breakdown: String = phase
                    .entity_type_counts
                    .iter()
                    .map(|(t, c)| format!("{} {}", c, t))
                    .collect::<Vec<_>>()
                    .join(", ");

                let seq_info = phase
                    .sequence_range
                    .map(|(min, max)| format!(" | timeline: {}-{}", min, max))
                    .unwrap_or_default();

                let member_list: String = phase
                    .members
                    .iter()
                    .take(5)
                    .map(|m| {
                        let pos = m
                            .sequence_position
                            .map(|p| format!(" @{:.2}", p))
                            .unwrap_or_default();
                        format!("{} ({}{})", m.name, m.entity_type, pos)
                    })
                    .collect::<Vec<_>>()
                    .join(", ");

                let content = format!(
                    "{} members ({}){}{}",
                    phase.member_count,
                    type_breakdown,
                    seq_info,
                    if phase.members.len() > 5 {
                        format!(
                            "\nTop members: {} ... and {} more",
                            member_list,
                            phase.members.len() - 5
                        )
                    } else {
                        format!("\nMembers: {}", member_list)
                    }
                );

                EntityResult {
                    id: format!("phase-{}", phase.phase_id),
                    entity_type: "narrative_phase".to_string(),
                    name: format!("Phase {}: {}", phase.phase_id, phase.label),
                    content,
                    confidence: None,
                    last_modified: None,
                }
            })
            .collect();

        let total = entity_results.len();
        let token_estimate = entity_results
            .iter()
            .map(|r| r.content.len() / 4 + 30)
            .sum();

        let mut hints = vec![
            format!(
                "Detected {} phases from {} entities",
                total, result.total_entities
            ),
            format!(
                "Weights: content={:.2}, neighborhood={:.2}, temporal={:.2}",
                result.weights_used.content,
                result.weights_used.neighborhood,
                result.weights_used.temporal
            ),
        ];

        if result.entities_without_embeddings > 0 {
            hints.push(format!(
                "{} entities lack embeddings - run backfill_embeddings",
                result.entities_without_embeddings
            ));
        }
        if result.entities_without_temporal_anchor > 0 {
            hints.push(format!(
                "{} entities have no event anchor (clustered by content only)",
                result.entities_without_temporal_anchor
            ));
        }
        if save {
            hints.push(
                "Phases saved to database — use load_phases to retrieve instantly".to_string(),
            );
        }

        Ok(QueryResponse {
            results: entity_results,
            total,
            next_cursor: None,
            hints,
            token_estimate,
            truncated: None,
        })
    }

    pub(crate) async fn handle_load_phases(&self) -> Result<QueryResponse, String> {
        use crate::services::TemporalService;

        let service = TemporalService::new(self.db.clone());
        let saved = service
            .load_phases()
            .await
            .map_err(|e| format!("Failed to load phases: {}", e))?;

        match saved {
            None => Ok(QueryResponse {
                results: vec![],
                total: 0,
                next_cursor: None,
                hints: vec![
                    "No saved phases found. Use detect_phases with save=true to persist phases."
                        .to_string(),
                ],
                token_estimate: 0,
                truncated: None,
            }),
            Some(result) => {
                let entity_results: Vec<EntityResult> = result
                    .phases
                    .iter()
                    .map(|phase| {
                        let type_breakdown: String = phase
                            .entity_type_counts
                            .iter()
                            .map(|(t, c)| format!("{} {}", c, t))
                            .collect::<Vec<_>>()
                            .join(", ");

                        let seq_info = phase
                            .sequence_range
                            .map(|(min, max)| format!(" | timeline: {}-{}", min, max))
                            .unwrap_or_default();

                        let member_list: String = phase
                            .members
                            .iter()
                            .take(5)
                            .map(|m| format!("{} ({})", m.name, m.entity_type))
                            .collect::<Vec<_>>()
                            .join(", ");

                        let content = format!(
                            "{} members ({}){}{}",
                            phase.member_count,
                            type_breakdown,
                            seq_info,
                            if phase.members.len() > 5 {
                                format!(
                                    "\nTop members: {} ... and {} more",
                                    member_list,
                                    phase.members.len() - 5
                                )
                            } else {
                                format!("\nMembers: {}", member_list)
                            }
                        );

                        EntityResult {
                            id: format!("phase:phase_{}", phase.phase_id),
                            entity_type: "narrative_phase".to_string(),
                            name: format!("Phase {}: {}", phase.phase_id, phase.label),
                            content,
                            confidence: None,
                            last_modified: None,
                        }
                    })
                    .collect();

                let total = entity_results.len();
                let token_estimate = entity_results
                    .iter()
                    .map(|r| r.content.len() / 4 + 30)
                    .sum();

                Ok(QueryResponse {
                    results: entity_results,
                    total,
                    next_cursor: None,
                    hints: vec![format!(
                        "Loaded {} saved phases (use clear_phases to remove)",
                        total
                    )],
                    token_estimate,
                    truncated: None,
                })
            }
        }
    }

    pub(crate) async fn handle_detect_transitions(
        &self,
        entity_types: Option<Vec<String>>,
        num_phases: Option<usize>,
        content_weight: Option<f32>,
        neighborhood_weight: Option<f32>,
        temporal_weight: Option<f32>,
    ) -> Result<QueryResponse, String> {
        use crate::services::{PhaseWeights, TemporalService};

        let type_filter = {
            let parsed = parse_entity_types(entity_types);
            if parsed.is_empty() {
                EntityType::embeddable()
            } else {
                parsed
            }
        };

        let weights = if content_weight.is_some()
            || neighborhood_weight.is_some()
            || temporal_weight.is_some()
        {
            Some(PhaseWeights {
                content: content_weight.unwrap_or(0.6),
                neighborhood: neighborhood_weight.unwrap_or(0.25),
                temporal: temporal_weight.unwrap_or(0.15),
            })
        } else {
            None
        };

        let service = TemporalService::new(self.db.clone());
        let result = service
            .detect_transitions(type_filter, num_phases, weights)
            .await
            .map_err(|e| format!("Transition analysis failed: {}", e))?;

        let entity_results: Vec<EntityResult> = result
            .transitions
            .iter()
            .map(|t| {
                let phase_list = t
                    .phase_labels
                    .iter()
                    .enumerate()
                    .map(|(i, label)| format!("Phase {}: {}", t.phase_ids[i], label))
                    .collect::<Vec<_>>()
                    .join(", ");

                let span_info = t
                    .sequence_span
                    .map(|s| format!(" | sequence span: {:.0}", s))
                    .unwrap_or_default();

                let content = format!(
                    "Bridges {} phases: [{}]{} | strength: {:.3}",
                    t.phase_ids.len(),
                    phase_list,
                    span_info,
                    t.bridge_strength
                );

                EntityResult {
                    id: t.entity_id.clone(),
                    entity_type: t.entity_type.clone(),
                    name: t.name.clone(),
                    content,
                    confidence: Some(t.bridge_strength),
                    last_modified: None,
                }
            })
            .collect();

        let total = entity_results.len();
        let token_estimate = entity_results
            .iter()
            .map(|r| r.content.len() / 4 + 30)
            .sum();

        let mut hints = vec![
            format!(
                "{} bridge entities across {} phases",
                result.total_bridge_entities, result.phases_analyzed
            ),
            "Bridge entities connect narrative arcs — they appear in 2+ phases".to_string(),
        ];

        if !result.phase_connections.is_empty() {
            let connections: String = result
                .phase_connections
                .iter()
                .take(5)
                .map(|(a, b, count)| format!("Phase {}↔{}: {} bridges", a, b, count))
                .collect::<Vec<_>>()
                .join(", ");
            hints.push(format!("Phase connections: {}", connections));
        }

        Ok(QueryResponse {
            results: entity_results,
            total,
            next_cursor: None,
            hints,
            token_estimate,
            truncated: None,
        })
    }

    pub(crate) async fn handle_query_around(
        &self,
        anchor_id: &str,
        entity_types: Option<Vec<String>>,
        limit: usize,
    ) -> Result<QueryResponse, String> {
        use crate::services::TemporalService;

        let type_filter = {
            let parsed = parse_entity_types(entity_types);
            if parsed.is_empty() {
                EntityType::embeddable()
            } else {
                parsed
            }
        };

        let service = TemporalService::new(self.db.clone());
        let result = service
            .query_around(anchor_id, type_filter, limit)
            .await
            .map_err(|e| format!("Query around failed: {}", e))?;

        let entity_results: Vec<EntityResult> = result
            .neighbors
            .iter()
            .map(|n| {
                let mut details = vec![format!("similarity: {:.4}", n.similarity)];
                if n.shared_scenes > 0 {
                    details.push(format!("{} shared scenes", n.shared_scenes));
                }
                if let Some(dist) = n.sequence_distance {
                    details.push(format!("timeline distance: {:.2}", dist));
                }

                EntityResult {
                    id: n.entity_id.clone(),
                    entity_type: n.entity_type.clone(),
                    name: n.name.clone(),
                    content: details.join(" | "),
                    confidence: Some(n.similarity),
                    last_modified: None,
                }
            })
            .collect();

        let total = entity_results.len();
        let token_estimate = entity_results
            .iter()
            .map(|r| r.content.len() / 4 + 30)
            .sum();

        let phase_info = if result.anchor_phases.is_empty() {
            "no phase assignment".to_string()
        } else {
            format!(
                "phase(s): {}",
                result
                    .anchor_phases
                    .iter()
                    .map(|p| p.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        };

        let hints = vec![
            format!(
                "Anchor: {} ({}) - {}",
                result.anchor.name, result.anchor.entity_id, phase_info
            ),
            format!("Found {} narrative neighbors", total),
            "Similarity combines embedding closeness, scene co-occurrence, and timeline proximity"
                .to_string(),
        ];

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
