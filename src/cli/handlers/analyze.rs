//! CLI handlers for narrative analytics commands.

use anyhow::Result;
use serde::Serialize;

use crate::cli::output::{output_json, output_json_list, print_table, OutputMode};
use crate::init::AppContext;
use crate::services::{
    CentralityMetric, CompositeIntelligenceService, GraphAnalyticsService, InfluenceService,
    IronyService,
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
