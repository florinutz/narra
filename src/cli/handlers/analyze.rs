//! CLI handlers for narrative analytics commands.

use anyhow::Result;
use serde::Serialize;

use crate::cli::output::{
    output_json, output_json_list, print_header, print_hint, print_kv, print_table, OutputMode,
};
use crate::cli::resolve::resolve_single;
use crate::init::AppContext;
use crate::repository::KnowledgeRepository;
use crate::services::{
    generate_suggested_fix, CentralityMetric, ClusteringService, CompositeIntelligenceService,
    EntityType, GraphAnalyticsService, InfluenceService, IronyService, VectorOpsService,
};

pub async fn handle_centrality(
    ctx: &AppContext,
    scope: Option<&str>,
    limit: usize,
    mode: OutputMode,
) -> Result<()> {
    let service = GraphAnalyticsService::new(ctx.db.clone());
    let metrics = vec![
        CentralityMetric::Degree,
        CentralityMetric::Betweenness,
        CentralityMetric::Closeness,
    ];
    let results = service
        .compute_centrality(scope.map(|s| s.to_string()), metrics, limit)
        .await
        .map_err(|e| anyhow::anyhow!("Centrality computation failed: {}", e))?;

    if mode == OutputMode::Json {
        output_json_list(&results);
    } else {
        let rows: Vec<Vec<String>> = results
            .iter()
            .enumerate()
            .map(|(i, r)| {
                vec![
                    format!("{}", i + 1),
                    r.character_name.clone(),
                    format!("{:.4}", r.degree),
                    format!("{:.4}", r.betweenness),
                    format!("{:.4}", r.closeness),
                    r.narrative_role.clone(),
                ]
            })
            .collect();
        print_table(
            &[
                "#",
                "Character",
                "Degree",
                "Betweenness",
                "Closeness",
                "Role",
            ],
            rows,
        );
    }

    Ok(())
}

pub async fn handle_influence(
    ctx: &AppContext,
    character: &str,
    depth: usize,
    mode: OutputMode,
) -> Result<()> {
    let service = InfluenceService::new(ctx.db.clone());
    let result = service
        .trace_propagation(character, depth)
        .await
        .map_err(|e| anyhow::anyhow!("Influence propagation failed: {}", e))?;

    if mode == OutputMode::Json {
        output_json(&result);
    } else {
        println!(
            "Influence from '{}': {} reachable, {} unreachable",
            result.source_character,
            result.reachable_characters.len(),
            result.unreachable_characters.len()
        );
        let rows: Vec<Vec<String>> = result
            .reachable_characters
            .iter()
            .map(|p| {
                let chain: Vec<String> = p.steps.iter().map(|s| s.character_name.clone()).collect();
                vec![
                    chain.join(" -> "),
                    format!("{}", p.total_hops),
                    p.path_strength.clone(),
                ]
            })
            .collect();
        print_table(&["Path", "Hops", "Strength"], rows);
    }

    Ok(())
}

pub async fn handle_irony(
    ctx: &AppContext,
    character: Option<&str>,
    threshold: usize,
    mode: OutputMode,
) -> Result<()> {
    let service = IronyService::new(ctx.db.clone());
    let report = service
        .generate_report(character, threshold)
        .await
        .map_err(|e| anyhow::anyhow!("Irony report failed: {}", e))?;

    if mode == OutputMode::Json {
        output_json(&report);
    } else {
        println!(
            "Dramatic Irony Report — {} asymmetries ({} high-signal)",
            report.total_asymmetries, report.high_signal_count
        );
        let rows: Vec<Vec<String>> = report
            .asymmetries
            .iter()
            .map(|a| {
                vec![
                    a.knowing_character_name.clone(),
                    a.unknowing_character_name.clone(),
                    a.fact.clone(),
                    a.signal_strength.clone(),
                    format!("{:.1}", a.dramatic_weight),
                ]
            })
            .collect();
        print_table(&["Knows", "Doesn't know", "Fact", "Signal", "Weight"], rows);

        if !report.narrative_opportunities.is_empty() {
            println!("\nNarrative Opportunities:");
            for opp in &report.narrative_opportunities {
                println!("  - {}", opp);
            }
        }
    }

    Ok(())
}

pub async fn handle_asymmetries(
    ctx: &AppContext,
    char_a: &str,
    char_b: &str,
    mode: OutputMode,
) -> Result<()> {
    let service = IronyService::new(ctx.db.clone());
    let asymmetries = service
        .detect_asymmetries(char_a, char_b)
        .await
        .map_err(|e| anyhow::anyhow!("Asymmetry detection failed: {}", e))?;

    if mode == OutputMode::Json {
        output_json_list(&asymmetries);
    } else {
        println!(
            "Knowledge asymmetries between {} and {}: {}",
            char_a,
            char_b,
            asymmetries.len()
        );
        let rows: Vec<Vec<String>> = asymmetries
            .iter()
            .map(|a| {
                let tension = a
                    .tension_level
                    .map(|t| format!("{}", t))
                    .unwrap_or_else(|| "-".to_string());
                vec![
                    a.knowing_character_name.clone(),
                    a.unknowing_character_name.clone(),
                    a.fact.clone(),
                    a.signal_strength.clone(),
                    tension,
                    format!("{:.1}", a.dramatic_weight),
                ]
            })
            .collect();
        print_table(
            &[
                "Knows",
                "Doesn't know",
                "Fact",
                "Signal",
                "Tension",
                "Weight",
            ],
            rows,
        );
    }

    Ok(())
}

pub async fn handle_conflicts(
    ctx: &AppContext,
    character: Option<&str>,
    limit: usize,
    mode: OutputMode,
) -> Result<()> {
    use crate::models::knowledge::find_knowledge_conflicts;

    let conflicts = find_knowledge_conflicts(&ctx.db)
        .await
        .map_err(|e| anyhow::anyhow!("Knowledge conflicts query failed: {}", e))?;

    // Flatten and filter
    let mut rows_data: Vec<(String, String, String, String)> = Vec::new();
    for conflict in &conflicts {
        for state in &conflict.conflicting_states {
            if let Some(filter) = character {
                let key = state
                    .character_id
                    .split(':')
                    .nth(1)
                    .unwrap_or(&state.character_id);
                let filter_key = filter.split(':').nth(1).unwrap_or(filter);
                if key != filter_key {
                    continue;
                }
            }
            let truth = state
                .truth_value
                .as_deref()
                .unwrap_or("unknown")
                .to_string();
            rows_data.push((
                state.character_id.clone(),
                conflict.target.clone(),
                format!("{:?}", state.certainty),
                truth,
            ));
        }
    }
    rows_data.truncate(limit);

    if mode == OutputMode::Json {
        #[derive(Serialize)]
        struct ConflictRow {
            character: String,
            target: String,
            certainty: String,
            truth_value: String,
        }
        let json_rows: Vec<_> = rows_data
            .iter()
            .map(|(c, t, cert, tv)| ConflictRow {
                character: c.clone(),
                target: t.clone(),
                certainty: cert.clone(),
                truth_value: tv.clone(),
            })
            .collect();
        output_json_list(&json_rows);
    } else {
        println!("Knowledge conflicts (BelievesWrongly): {}", rows_data.len());
        let table_rows: Vec<Vec<String>> = rows_data
            .iter()
            .map(|(c, t, cert, tv)| vec![c.clone(), t.clone(), cert.clone(), tv.clone()])
            .collect();
        print_table(&["Character", "Target", "Certainty", "Truth"], table_rows);
    }

    Ok(())
}

pub async fn handle_tensions(ctx: &AppContext, limit: usize, mode: OutputMode) -> Result<()> {
    // Query perceives edges with tension data
    let mut result = ctx
        .db
        .query(
            "SELECT type::string(in) AS observer, type::string(out) AS target, \
             in.name AS observer_name, out.name AS target_name, \
             tension_level, feelings \
             FROM perceives \
             WHERE tension_level IS NOT NONE AND tension_level >= 5 \
             ORDER BY tension_level DESC \
             LIMIT $limit",
        )
        .bind(("limit", limit))
        .await
        .map_err(|e| anyhow::anyhow!("Tension query failed: {}", e))?;

    #[derive(serde::Deserialize, Serialize)]
    struct TensionRow {
        observer: String,
        target: String,
        observer_name: Option<String>,
        target_name: Option<String>,
        tension_level: i32,
        feelings: Option<String>,
    }

    let tensions: Vec<TensionRow> = result.take(0)?;

    if mode == OutputMode::Json {
        output_json_list(&tensions);
    } else {
        println!("High-tension perception pairs: {}", tensions.len());
        let rows: Vec<Vec<String>> = tensions
            .iter()
            .map(|t| {
                vec![
                    t.observer_name
                        .as_deref()
                        .unwrap_or(&t.observer)
                        .to_string(),
                    t.target_name.as_deref().unwrap_or(&t.target).to_string(),
                    format!("{}", t.tension_level),
                    t.feelings.as_deref().unwrap_or("-").to_string(),
                ]
            })
            .collect();
        print_table(&["Observer", "Target", "Tension", "Feelings"], rows);
    }

    Ok(())
}

pub async fn handle_arc_drift(
    ctx: &AppContext,
    entity_type: Option<&str>,
    limit: usize,
    mode: OutputMode,
) -> Result<()> {
    use crate::utils::math::cosine_similarity;

    // Fetch all arc snapshots grouped by entity
    let mut result = ctx
        .db
        .query(
            "SELECT entity_id, count() AS snapshot_count \
             FROM arc_snapshot \
             GROUP BY entity_id \
             ORDER BY snapshot_count DESC",
        )
        .await
        .map_err(|e| anyhow::anyhow!("Arc drift query failed: {}", e))?;

    #[derive(serde::Deserialize)]
    struct ArcEntity {
        entity_id: String,
        snapshot_count: usize,
    }

    let entities: Vec<ArcEntity> = result.take(0).unwrap_or_default();

    // Filter by entity type if specified, and only entities with >1 snapshot
    let filtered: Vec<&ArcEntity> = entities
        .iter()
        .filter(|e| e.snapshot_count > 1)
        .filter(|e| entity_type.map(|t| e.entity_id.contains(t)).unwrap_or(true))
        .collect();

    #[derive(Serialize)]
    struct DriftRow {
        entity_id: String,
        snapshots: usize,
        drift: f64,
    }

    let mut drift_rows: Vec<DriftRow> = Vec::new();

    for entity in &filtered {
        // Fetch first and last embeddings
        let q_first = format!(
            "SELECT embedding, created_at FROM arc_snapshot WHERE entity_id = {} ORDER BY created_at ASC LIMIT 1",
            entity.entity_id
        );
        let q_last = format!(
            "SELECT embedding, created_at FROM arc_snapshot WHERE entity_id = {} ORDER BY created_at DESC LIMIT 1",
            entity.entity_id
        );

        #[derive(serde::Deserialize)]
        struct EmbOnly {
            embedding: Vec<f32>,
        }

        let first: Option<EmbOnly> = ctx
            .db
            .query(&q_first)
            .await
            .ok()
            .and_then(|mut r| r.take::<Option<EmbOnly>>(0).ok().flatten());
        let last: Option<EmbOnly> = ctx
            .db
            .query(&q_last)
            .await
            .ok()
            .and_then(|mut r| r.take::<Option<EmbOnly>>(0).ok().flatten());

        let drift = match (first, last) {
            (Some(f), Some(l)) => 1.0 - cosine_similarity(&f.embedding, &l.embedding) as f64,
            _ => 0.0,
        };

        drift_rows.push(DriftRow {
            entity_id: entity.entity_id.clone(),
            snapshots: entity.snapshot_count,
            drift,
        });
    }

    // Sort by drift descending, take limit
    drift_rows.sort_by(|a, b| {
        b.drift
            .partial_cmp(&a.drift)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    drift_rows.truncate(limit);

    if mode == OutputMode::Json {
        output_json_list(&drift_rows);
    } else {
        println!("Arc drift ranking:");
        let rows: Vec<Vec<String>> = drift_rows
            .iter()
            .enumerate()
            .map(|(i, r)| {
                vec![
                    format!("{}", i + 1),
                    r.entity_id.clone(),
                    format!("{}", r.snapshots),
                    format!("{:.4}", r.drift),
                ]
            })
            .collect();
        print_table(&["#", "Entity", "Snapshots", "Drift"], rows);
    }

    Ok(())
}

pub async fn handle_situation_report(ctx: &AppContext, mode: OutputMode) -> Result<()> {
    let service = CompositeIntelligenceService::new(ctx.db.clone());
    let report = service
        .situation_report()
        .await
        .map_err(|e| anyhow::anyhow!("Situation report failed: {}", e))?;

    if mode == OutputMode::Json {
        output_json(&report);
    } else {
        println!(
            "Narrative Situation Report — {} irony highlights, {} conflicts, {} high-tension pairs, {} themes",
            report.irony_highlights.len(),
            report.knowledge_conflicts.len(),
            report.high_tension_pairs.len(),
            report.theme_count
        );

        if !report.irony_highlights.is_empty() {
            println!("\nDramatic Irony Highlights:");
            let rows: Vec<Vec<String>> = report
                .irony_highlights
                .iter()
                .map(|a| {
                    vec![
                        a.knowing_character_name.clone(),
                        a.unknowing_character_name.clone(),
                        a.fact.clone(),
                        format!("{:.1}", a.dramatic_weight),
                    ]
                })
                .collect();
            print_table(&["Knows", "Doesn't know", "Fact", "Weight"], rows);
        }

        if !report.knowledge_conflicts.is_empty() {
            println!("\nKnowledge Conflicts:");
            let rows: Vec<Vec<String>> = report
                .knowledge_conflicts
                .iter()
                .map(|c| {
                    vec![
                        c.character_id.clone(),
                        c.target.clone(),
                        c.certainty.clone(),
                        c.truth_value.clone(),
                    ]
                })
                .collect();
            print_table(&["Character", "Target", "Certainty", "Truth"], rows);
        }

        if !report.high_tension_pairs.is_empty() {
            println!("\nHigh-Tension Pairs:");
            let rows: Vec<Vec<String>> = report
                .high_tension_pairs
                .iter()
                .map(|t| {
                    vec![
                        t.observer.clone(),
                        t.target.clone(),
                        format!("{}", t.tension_level),
                        t.feelings.as_deref().unwrap_or("-").to_string(),
                    ]
                })
                .collect();
            print_table(&["Observer", "Target", "Tension", "Feelings"], rows);
        }

        if !report.suggestions.is_empty() {
            println!("\nSuggestions:");
            for s in &report.suggestions {
                println!("  - {}", s);
            }
        }
    }

    Ok(())
}

pub async fn handle_dossier(ctx: &AppContext, character: &str, mode: OutputMode) -> Result<()> {
    let service = CompositeIntelligenceService::new(ctx.db.clone());
    let dossier = service
        .character_dossier(character)
        .await
        .map_err(|e| anyhow::anyhow!("Character dossier failed: {}", e))?;

    if mode == OutputMode::Json {
        output_json(&dossier);
    } else {
        println!("Character Dossier: {}", dossier.name);
        println!(
            "Roles: {}",
            if dossier.roles.is_empty() {
                "none".to_string()
            } else {
                dossier.roles.join(", ")
            }
        );
        println!(
            "Centrality rank: {}",
            dossier
                .centrality_rank
                .map(|r| format!("#{}", r))
                .unwrap_or_else(|| "unranked".to_string())
        );
        println!("Influence reach: {} characters", dossier.influence_reach);
        println!(
            "Knowledge: {} advantages, {} blind spots, {} false beliefs",
            dossier.knowledge_advantages, dossier.knowledge_blind_spots, dossier.false_beliefs
        );
        if let Some(avg) = dossier.avg_tension_toward_them {
            println!("Average tension toward them: {:.1}", avg);
        }

        if !dossier.key_perceptions.is_empty() {
            println!("\nKey Perceptions:");
            let rows: Vec<Vec<String>> = dossier
                .key_perceptions
                .iter()
                .map(|p| {
                    vec![
                        p.observer.clone(),
                        p.tension_level
                            .map(|t| format!("{}", t))
                            .unwrap_or_else(|| "-".to_string()),
                        p.feelings.as_deref().unwrap_or("-").to_string(),
                    ]
                })
                .collect();
            print_table(&["Observer", "Tension", "Feelings"], rows);
        }

        if !dossier.suggestions.is_empty() {
            println!("\nSuggestions:");
            for s in &dossier.suggestions {
                println!("  - {}", s);
            }
        }
    }

    Ok(())
}

pub async fn handle_scene_prep(
    ctx: &AppContext,
    characters: &[String],
    mode: OutputMode,
) -> Result<()> {
    let service = CompositeIntelligenceService::new(ctx.db.clone());
    let plan = service
        .scene_prep(characters)
        .await
        .map_err(|e| anyhow::anyhow!("Scene planning failed: {}", e))?;

    if mode == OutputMode::Json {
        output_json(&plan);
    } else {
        println!(
            "Scene Plan — {} characters, {} irony opportunities",
            plan.characters.len(),
            plan.total_irony_opportunities
        );
        if let Some((a, b, t)) = &plan.highest_tension_pair {
            println!("Highest tension: {} <-> {} (level {})", a, b, t);
        }
        println!("Shared history scenes: {}", plan.shared_history_scenes);

        if !plan.pair_dynamics.is_empty() {
            println!("\nPair Dynamics:");
            let rows: Vec<Vec<String>> = plan
                .pair_dynamics
                .iter()
                .map(|d| {
                    vec![
                        format!("{} <-> {}", d.character_a, d.character_b),
                        format!("{}", d.asymmetries),
                        d.tension_level
                            .map(|t| format!("{}", t))
                            .unwrap_or_else(|| "-".to_string()),
                        format!("{}", d.shared_scene_count),
                        d.feelings.as_deref().unwrap_or("-").to_string(),
                    ]
                })
                .collect();
            print_table(
                &[
                    "Pair",
                    "Asymmetries",
                    "Tension",
                    "Shared Scenes",
                    "Feelings",
                ],
                rows,
            );
        }

        if !plan.applicable_facts.is_empty() {
            println!("\nApplicable Facts:");
            for f in &plan.applicable_facts {
                println!("  - {}", f);
            }
        }

        if !plan.opportunities.is_empty() {
            println!("\nOpportunities:");
            for o in &plan.opportunities {
                println!("  - {}", o);
            }
        }
    }

    Ok(())
}

pub async fn handle_themes(
    ctx: &AppContext,
    types: Option<Vec<String>>,
    clusters: Option<usize>,
    mode: OutputMode,
) -> Result<()> {
    let entity_types: Vec<EntityType> = match types {
        Some(ref type_strs) => type_strs
            .iter()
            .filter_map(|t| match t.as_str() {
                "character" | "char" => Some(EntityType::Character),
                "location" => Some(EntityType::Location),
                "event" => Some(EntityType::Event),
                "scene" => Some(EntityType::Scene),
                "knowledge" => Some(EntityType::Knowledge),
                _ => {
                    eprintln!("Unknown type '{}', skipping", t);
                    None
                }
            })
            .collect(),
        None => EntityType::embeddable(),
    };

    let service = ClusteringService::new(ctx.db.clone());
    let result = service
        .discover_themes(entity_types, clusters)
        .await
        .map_err(|e| anyhow::anyhow!("Thematic clustering failed: {}", e))?;

    if mode == OutputMode::Json {
        output_json(&result);
    } else {
        println!(
            "Thematic Clustering: {} clusters from {} entities ({} without embeddings)",
            result.clusters.len(),
            result.total_entities,
            result.entities_without_embeddings
        );

        for cluster in &result.clusters {
            print_header(&format!(
                "Cluster {}: {} ({} members)",
                cluster.cluster_id, cluster.label, cluster.member_count
            ));
            let rows: Vec<Vec<String>> = cluster
                .members
                .iter()
                .map(|m| {
                    vec![
                        m.entity_type.clone(),
                        m.entity_id.clone(),
                        m.name.clone(),
                        format!("{:.4}", m.centrality),
                    ]
                })
                .collect();
            print_table(&["Type", "ID", "Name", "Centrality"], rows);
        }
    }

    Ok(())
}

pub async fn handle_thematic_gaps(
    ctx: &AppContext,
    min_size: usize,
    expected_types: Option<Vec<String>>,
    mode: OutputMode,
) -> Result<()> {
    let service = ClusteringService::new(ctx.db.clone());
    let clustering_result = service
        .discover_themes(EntityType::embeddable(), None)
        .await
        .map_err(|e| anyhow::anyhow!("Thematic gap analysis failed: {}", e))?;

    let expected =
        expected_types.unwrap_or_else(|| vec!["character".to_string(), "event".to_string()]);

    #[derive(Serialize)]
    struct GapResult {
        cluster_label: String,
        member_count: usize,
        present_types: Vec<String>,
        missing_types: Vec<String>,
        severity: f32,
        interpretation: String,
    }

    let mut gaps: Vec<GapResult> = Vec::new();

    for cluster in &clustering_result.clusters {
        if cluster.member_count < min_size {
            continue;
        }

        let mut type_counts: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        for member in &cluster.members {
            *type_counts.entry(member.entity_type.clone()).or_insert(0) += 1;
        }

        let missing: Vec<String> = expected
            .iter()
            .filter(|t| !type_counts.contains_key(t.as_str()))
            .cloned()
            .collect();

        if missing.is_empty() {
            continue;
        }

        let has_type = |t: &str| type_counts.contains_key(t);
        let interpretation = if has_type("event") && !has_type("character") {
            "Events without protagonists — who drives this theme?"
        } else if has_type("character") && !has_type("event") {
            "Characters without plot — what happens to embody this?"
        } else if !has_type("scene") && missing.contains(&"scene".to_string()) {
            "No scenes ground this theme — where does it come to life?"
        } else {
            "Theme has structural gaps — consider adding missing entity types."
        };

        let present_types: Vec<String> = type_counts.keys().cloned().collect();
        let severity = missing.len() as f32 / expected.len() as f32;

        gaps.push(GapResult {
            cluster_label: cluster.label.clone(),
            member_count: cluster.member_count,
            present_types,
            missing_types: missing,
            severity,
            interpretation: interpretation.to_string(),
        });
    }

    gaps.sort_by(|a, b| {
        b.severity
            .partial_cmp(&a.severity)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    if mode == OutputMode::Json {
        output_json_list(&gaps);
    } else {
        println!(
            "Thematic gaps: {} clusters analyzed, {} with gaps",
            clustering_result.clusters.len(),
            gaps.len()
        );

        let rows: Vec<Vec<String>> = gaps
            .iter()
            .map(|g| {
                vec![
                    g.cluster_label.clone(),
                    format!("{}", g.member_count),
                    g.present_types.join(", "),
                    g.missing_types.join(", "),
                    g.interpretation.clone(),
                ]
            })
            .collect();
        print_table(
            &["Theme", "Members", "Has", "Missing", "Interpretation"],
            rows,
        );
    }

    Ok(())
}

pub async fn handle_temporal(
    ctx: &AppContext,
    character: &str,
    event: Option<String>,
    mode: OutputMode,
    no_semantic: bool,
) -> Result<()> {
    let char_id = resolve_single(ctx, character, no_semantic).await?;
    let char_key = char_id.split(':').nth(1).unwrap_or(&char_id).to_string();

    let states = if let Some(event_ref) = event {
        // Resolve event
        let event_key = if event_ref.contains(':') {
            event_ref
                .split(':')
                .nth(1)
                .unwrap_or(&event_ref)
                .to_string()
        } else {
            // Fuzzy search for event
            let search_svc = if no_semantic {
                None
            } else {
                Some(ctx.search_service.as_ref())
            };
            let filter = crate::services::SearchFilter {
                entity_types: vec![EntityType::Event],
                limit: Some(1),
                ..Default::default()
            };
            let results = if let Some(svc) = search_svc {
                svc.fuzzy_search(&event_ref, 0.8, filter).await?
            } else {
                vec![]
            };
            match results.first() {
                Some(r) => r.id.split(':').nth(1).unwrap_or(&r.id).to_string(),
                None => anyhow::bail!("No event found for '{}'", event_ref),
            }
        };
        ctx.knowledge_repo
            .get_knowledge_at_event(&char_key, &event_key)
            .await
            .map_err(|e| anyhow::anyhow!("Temporal knowledge query failed: {}", e))?
    } else {
        ctx.knowledge_repo
            .get_character_knowledge_states(&char_key)
            .await
            .map_err(|e| anyhow::anyhow!("Knowledge states query failed: {}", e))?
    };

    if mode == OutputMode::Json {
        output_json_list(&states);
    } else {
        println!(
            "Knowledge states for {} ({} entries):",
            char_id,
            states.len()
        );
        let rows: Vec<Vec<String>> = states
            .iter()
            .map(|s| {
                vec![
                    s.target.to_string(),
                    format!("{:?}", s.certainty),
                    format!("{:?}", s.learning_method),
                    s.event
                        .as_ref()
                        .map(|e: &surrealdb::RecordId| e.to_string())
                        .unwrap_or_else(|| "-".to_string()),
                ]
            })
            .collect();
        print_table(&["Target", "Certainty", "Method", "Learned At"], rows);
    }

    Ok(())
}

pub async fn handle_contradictions(
    ctx: &AppContext,
    entity: &str,
    depth: usize,
    mode: OutputMode,
    no_semantic: bool,
) -> Result<()> {
    let entity_id = resolve_single(ctx, entity, no_semantic).await?;

    let (violations, entities_checked) = ctx
        .consistency_service
        .investigate_contradictions(&entity_id, depth)
        .await
        .map_err(|e| anyhow::anyhow!("Contradiction investigation failed: {}", e))?;

    if mode == OutputMode::Json {
        #[derive(Serialize)]
        struct ContradictionReport {
            entity_id: String,
            depth: usize,
            entities_checked: usize,
            violations: Vec<ContradictionRow>,
        }
        #[derive(Serialize)]
        struct ContradictionRow {
            severity: String,
            violation_type: String,
            message: String,
            suggested_fix: Option<String>,
        }
        let rows: Vec<ContradictionRow> = violations
            .iter()
            .map(|v| ContradictionRow {
                severity: format!("{:?}", v.severity).to_uppercase(),
                violation_type: classify_violation(v),
                message: v.message.clone(),
                suggested_fix: generate_suggested_fix(v),
            })
            .collect();
        output_json(&ContradictionReport {
            entity_id,
            depth,
            entities_checked,
            violations: rows,
        });
    } else {
        print_header(&format!(
            "Contradiction Investigation: {} (depth {}, {} entities checked)",
            entity_id, depth, entities_checked
        ));

        if violations.is_empty() {
            println!("  No contradictions found.");
        } else {
            println!("  {} issue(s) found:\n", violations.len());
            let rows: Vec<Vec<String>> = violations
                .iter()
                .map(|v| {
                    vec![
                        format!("{:?}", v.severity).to_uppercase(),
                        classify_violation(v),
                        v.message.clone(),
                        generate_suggested_fix(v).unwrap_or_else(|| "-".to_string()),
                    ]
                })
                .collect();
            print_table(&["Severity", "Type", "Message", "Fix"], rows);
        }
    }

    Ok(())
}

/// Classify a violation into a type category based on message content.
fn classify_violation(v: &crate::services::Violation) -> String {
    if v.message.contains("timeline") || v.message.contains("before learning") {
        "timeline".into()
    } else if v.message.contains("relationship")
        || v.message.contains("Circular")
        || v.message.contains("Asymmetric")
    {
        "relationship".into()
    } else {
        "fact".into()
    }
}

pub async fn handle_impact(
    ctx: &AppContext,
    entity: &str,
    description: Option<String>,
    mode: OutputMode,
    no_semantic: bool,
) -> Result<()> {
    let entity_id = resolve_single(ctx, entity, no_semantic).await?;

    let desc = description.as_deref().unwrap_or("general change");
    let analysis = ctx
        .impact_service
        .analyze_impact(&entity_id, desc, 3)
        .await
        .map_err(|e| anyhow::anyhow!("Impact analysis failed: {}", e))?;

    if mode == OutputMode::Json {
        output_json(&analysis);
    } else {
        print_header(&format!(
            "Impact Analysis: {} (\"{}\")",
            analysis.changed_entity, analysis.change_description
        ));
        print_kv("Total affected", &format!("{}", analysis.total_affected));
        print_kv(
            "Protected entities impacted",
            &format!("{}", analysis.has_protected_impact),
        );

        for severity in &["critical", "high", "medium", "low"] {
            if let Some(entities) = analysis.affected_by_severity.get(*severity) {
                if !entities.is_empty() {
                    println!("\n{} ({}):", severity.to_uppercase(), entities.len());
                    let rows: Vec<Vec<String>> = entities
                        .iter()
                        .map(|e| {
                            vec![
                                e.entity_type.clone(),
                                e.name.clone(),
                                e.reason.clone(),
                                format!("{}", e.distance),
                            ]
                        })
                        .collect();
                    print_table(&["Type", "Name", "Reason", "Distance"], rows);
                }
            }
        }

        if !analysis.warnings.is_empty() {
            println!("\nWarnings:");
            for w in &analysis.warnings {
                println!("  - {}", w);
            }
        }
    }

    Ok(())
}

// =============================================================================
// Phase 2: Vector arithmetic analysis commands
// =============================================================================

pub async fn handle_growth_vector(
    ctx: &AppContext,
    entity: &str,
    limit: usize,
    mode: OutputMode,
    no_semantic: bool,
) -> Result<()> {
    let entity_id = resolve_single(ctx, entity, no_semantic).await?;
    let service = VectorOpsService::new(ctx.db.clone());
    let result = service
        .growth_vector(&entity_id, limit)
        .await
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    if mode == OutputMode::Json {
        output_json(&result);
    } else {
        print_header(&format!("Growth Vector: {}", result.entity_name));
        print_kv("Snapshots", &result.snapshot_count.to_string());
        print_kv("Total drift", &format!("{:.4}", result.total_drift));
        if !result.trajectory_neighbors.is_empty() {
            println!(
                "\n  Trajectory neighbors (where {} is heading):",
                result.entity_name
            );
            let rows: Vec<Vec<String>> = result
                .trajectory_neighbors
                .iter()
                .map(|n| {
                    vec![
                        n.entity_id.clone(),
                        n.entity_name.clone(),
                        n.entity_type.clone(),
                        format!("{:.4}", n.alignment_score),
                    ]
                })
                .collect();
            print_table(&["ID", "Name", "Type", "Alignment"], rows);
        }
    }

    Ok(())
}

pub async fn handle_misperception(
    ctx: &AppContext,
    observer: &str,
    target: &str,
    limit: usize,
    mode: OutputMode,
    no_semantic: bool,
) -> Result<()> {
    let observer_id = resolve_single(ctx, observer, no_semantic).await?;
    let target_id = resolve_single(ctx, target, no_semantic).await?;
    let service = VectorOpsService::new(ctx.db.clone());
    let result = service
        .misperception_vector(&observer_id, &target_id, limit)
        .await
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    if mode == OutputMode::Json {
        output_json(&result);
    } else {
        print_header(&format!(
            "Misperception: {} -> {}",
            result.observer_name, result.target_name
        ));
        print_kv("Perception gap", &format!("{:.4}", result.perception_gap));
        if !result.misperception_neighbors.is_empty() {
            println!("\n  Misperception direction neighbors:");
            let rows: Vec<Vec<String>> = result
                .misperception_neighbors
                .iter()
                .map(|n| {
                    vec![
                        n.entity_id.clone(),
                        n.entity_name.clone(),
                        n.entity_type.clone(),
                        format!("{:.4}", n.alignment_score),
                    ]
                })
                .collect();
            print_table(&["ID", "Name", "Type", "Alignment"], rows);
        }
    }

    Ok(())
}

pub async fn handle_convergence(
    ctx: &AppContext,
    entity_a: &str,
    entity_b: &str,
    window: Option<usize>,
    mode: OutputMode,
    no_semantic: bool,
) -> Result<()> {
    let id_a = resolve_single(ctx, entity_a, no_semantic).await?;
    let id_b = resolve_single(ctx, entity_b, no_semantic).await?;
    let service = VectorOpsService::new(ctx.db.clone());
    let result = service
        .convergence_analysis(&id_a, &id_b, window)
        .await
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    if mode == OutputMode::Json {
        output_json(&result);
    } else {
        print_header(&format!(
            "Convergence: {} <-> {}",
            result.entity_a_name, result.entity_b_name
        ));
        print_kv(
            "Current similarity",
            &format!("{:.4}", result.current_similarity),
        );
        print_kv(
            "Convergence rate",
            &format!(
                "{:.6} ({})",
                result.convergence_rate,
                if result.convergence_rate > 0.001 {
                    "converging"
                } else if result.convergence_rate < -0.001 {
                    "diverging"
                } else {
                    "stable"
                }
            ),
        );
        if !result.trend.is_empty() {
            println!("\n  Similarity trend:");
            let rows: Vec<Vec<String>> = result
                .trend
                .iter()
                .map(|p| {
                    vec![
                        p.snapshot_index.to_string(),
                        format!("{:.4}", p.similarity),
                        p.event_label.clone().unwrap_or_default(),
                    ]
                })
                .collect();
            print_table(&["#", "Similarity", "Event"], rows);
        }
    }

    Ok(())
}

pub async fn handle_midpoint(
    ctx: &AppContext,
    entity_a: &str,
    entity_b: &str,
    limit: usize,
    mode: OutputMode,
    no_semantic: bool,
) -> Result<()> {
    let id_a = resolve_single(ctx, entity_a, no_semantic).await?;
    let id_b = resolve_single(ctx, entity_b, no_semantic).await?;
    let service = VectorOpsService::new(ctx.db.clone());
    let result = service
        .semantic_midpoint(&id_a, &id_b, limit)
        .await
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    if mode == OutputMode::Json {
        output_json(&result);
    } else {
        print_header(&format!(
            "Semantic Midpoint: {} <-> {}",
            result.entity_a_id, result.entity_b_id
        ));
        if result.neighbors.is_empty() {
            println!("  No neighbors found (entities may lack embeddings).");
        } else {
            let rows: Vec<Vec<String>> = result
                .neighbors
                .iter()
                .map(|n| {
                    vec![
                        n.entity_id.clone(),
                        n.entity_name.clone(),
                        n.entity_type.clone(),
                        format!("{:.4}", n.alignment_score),
                    ]
                })
                .collect();
            print_table(&["ID", "Name", "Type", "Similarity"], rows);
        }
    }

    Ok(())
}

pub async fn handle_facets(
    ctx: &AppContext,
    character: &str,
    mode: OutputMode,
    no_semantic: bool,
) -> Result<()> {
    let character_id = resolve_single(ctx, character, no_semantic).await?;

    // Fetch character with all facet embeddings and composites
    #[derive(serde::Deserialize)]
    struct FacetData {
        name: String,
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
        "SELECT name, identity_embedding, identity_composite, identity_stale, \
         psychology_embedding, psychology_composite, psychology_stale, \
         social_embedding, social_composite, social_stale, \
         narrative_embedding, narrative_composite, narrative_stale FROM {}",
        character_id
    );

    let mut resp = ctx.db.query(&query).await?;
    let facet_data: Option<FacetData> = resp.take(0)?;

    let facet_data =
        facet_data.ok_or_else(|| anyhow::anyhow!("Character not found: {}", character_id))?;

    if mode == OutputMode::Json {
        // Build JSON structure
        use serde_json::json;
        let output = json!({
            "character_id": character_id,
            "character_name": facet_data.name,
            "facets": {
                "identity": {
                    "status": if facet_data.identity_stale.unwrap_or(false) { "stale" } else if facet_data.identity_embedding.is_some() { "ok" } else { "missing" },
                    "composite": facet_data.identity_composite,
                },
                "psychology": {
                    "status": if facet_data.psychology_stale.unwrap_or(false) { "stale" } else if facet_data.psychology_embedding.is_some() { "ok" } else { "missing" },
                    "composite": facet_data.psychology_composite,
                },
                "social": {
                    "status": if facet_data.social_stale.unwrap_or(false) { "stale" } else if facet_data.social_embedding.is_some() { "ok" } else { "missing" },
                    "composite": facet_data.social_composite,
                },
                "narrative": {
                    "status": if facet_data.narrative_stale.unwrap_or(false) { "stale" } else if facet_data.narrative_embedding.is_some() { "ok" } else { "missing" },
                    "composite": facet_data.narrative_composite,
                }
            }
        });
        output_json(&output);
        return Ok(());
    }

    print_header(&format!("Character Facets: {}", facet_data.name));

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
                similarities.push((facets[i].0, facets[j].0, sim));
            }
        }
    }

    // Print facet status table
    let mut rows = Vec::new();
    for (facet_name, embedding_opt) in &facets {
        let (status, composite) = match *facet_name {
            "identity" => (
                if facet_data.identity_stale.unwrap_or(false) {
                    "STALE"
                } else if embedding_opt.is_some() {
                    "OK"
                } else {
                    "MISSING"
                },
                facet_data.identity_composite.as_deref(),
            ),
            "psychology" => (
                if facet_data.psychology_stale.unwrap_or(false) {
                    "STALE"
                } else if embedding_opt.is_some() {
                    "OK"
                } else {
                    "MISSING"
                },
                facet_data.psychology_composite.as_deref(),
            ),
            "social" => (
                if facet_data.social_stale.unwrap_or(false) {
                    "STALE"
                } else if embedding_opt.is_some() {
                    "OK"
                } else {
                    "MISSING"
                },
                facet_data.social_composite.as_deref(),
            ),
            "narrative" => (
                if facet_data.narrative_stale.unwrap_or(false) {
                    "STALE"
                } else if embedding_opt.is_some() {
                    "OK"
                } else {
                    "MISSING"
                },
                facet_data.narrative_composite.as_deref(),
            ),
            _ => ("UNKNOWN", None),
        };

        let composite_preview = composite
            .map(|c| {
                if c.len() > 60 {
                    format!("{}...", &c[..60])
                } else {
                    c.to_string()
                }
            })
            .unwrap_or_else(|| "-".to_string());

        rows.push(vec![
            facet_name.to_string(),
            status.to_string(),
            composite_preview,
        ]);
    }

    print_table(&["Facet", "Status", "Composite (preview)"], rows);

    // Print inter-facet similarities
    if !similarities.is_empty() {
        println!("\nInter-facet Similarities:");
        for (facet_a, facet_b, sim) in similarities {
            println!("  {} <-> {}: {:.4}", facet_a, facet_b, sim);
        }
    }

    print_hint("Use 'narra find --facet <facet_name> <query>' to search by specific facet");

    Ok(())
}
