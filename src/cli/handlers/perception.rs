//! CLI handlers for perception analysis commands.

use anyhow::Result;
use serde::Serialize;

use crate::cli::output::{
    create_spinner, output_json, print_header, print_hint, print_kv, print_success, print_table,
    OutputMode,
};
use crate::cli::resolve::resolve_by_name;
use crate::init::AppContext;
use crate::models::{Perception, PerceptionCreate};
use crate::repository::relationship::RelationshipRepository;
use crate::services::perception::PerceptionService;

/// Resolve a single entity to a character ID.
async fn resolve_character(ctx: &AppContext, input: &str, no_semantic: bool) -> Result<String> {
    if input.contains(':') {
        return Ok(input.to_string());
    }
    let search_svc = if no_semantic {
        None
    } else {
        Some(ctx.search_service.as_ref())
    };
    let matches = resolve_by_name(&ctx.db, input, search_svc).await?;
    match matches.len() {
        0 => anyhow::bail!("No entity found for '{}'", input),
        1 => Ok(matches[0].id.clone()),
        _ => {
            eprintln!("Ambiguous name '{}'. Matches:", input);
            for m in &matches {
                eprintln!("  {} ({})", m.id, m.name);
            }
            anyhow::bail!("Use a full ID (type:key) to disambiguate");
        }
    }
}

pub async fn handle_perception_gap(
    ctx: &AppContext,
    observer: &str,
    target: &str,
    mode: OutputMode,
    no_semantic: bool,
) -> Result<()> {
    let observer_id = resolve_character(ctx, observer, no_semantic).await?;
    let target_id = resolve_character(ctx, target, no_semantic).await?;

    let service = PerceptionService::new(ctx.db.clone());
    let result = service.analyze_gap(&observer_id, &target_id).await?;

    if mode == OutputMode::Json {
        #[derive(Serialize)]
        struct PerceptionGapOutput {
            observer: String,
            target: String,
            gap: f32,
            similarity: f32,
            assessment: String,
            perception: Option<String>,
            feelings: Option<String>,
            tension_level: Option<i32>,
        }
        output_json(&PerceptionGapOutput {
            observer: result.observer_name.clone(),
            target: result.target_name.clone(),
            gap: result.gap,
            similarity: result.similarity,
            assessment: result.assessment.clone(),
            perception: result.perception.clone(),
            feelings: result.feelings.clone(),
            tension_level: result.tension_level,
        });
    } else {
        print_header(&format!(
            "Perception Gap: {} -> {}",
            result.observer_name, result.target_name
        ));
        print_kv("Gap", &format!("{:.4} ({})", result.gap, result.assessment));
        print_kv("Cosine similarity", &format!("{:.4}", result.similarity));
        if let Some(ref p) = result.perception {
            print_kv("Perception", p);
        }
        if let Some(ref f) = result.feelings {
            print_kv("Feelings", f);
        }
        if let Some(t) = result.tension_level {
            print_kv("Tension", &format!("{}/10", t));
        }
        if let Some(ref h) = result.history_notes {
            print_kv("History", h);
        }
    }

    Ok(())
}

pub async fn handle_perception_matrix(
    ctx: &AppContext,
    target: &str,
    limit: usize,
    mode: OutputMode,
    no_semantic: bool,
) -> Result<()> {
    let target_id = resolve_character(ctx, target, no_semantic).await?;

    let service = PerceptionService::new(ctx.db.clone());
    let result = service.analyze_matrix(&target_id, limit).await?;

    if mode == OutputMode::Json {
        #[derive(Serialize)]
        struct MatrixRow {
            observer: String,
            gap: f32,
            assessment: String,
            agrees_with: Option<String>,
            disagrees_with: Option<String>,
        }
        let rows: Vec<MatrixRow> = result
            .observers
            .iter()
            .map(|obs| MatrixRow {
                observer: obs.observer_name.clone(),
                gap: obs.gap,
                assessment: obs.assessment.clone(),
                agrees_with: obs.agrees_with.clone(),
                disagrees_with: obs.disagrees_with.clone(),
            })
            .collect();
        output_json(&rows);
    } else {
        println!(
            "Perception Matrix for {} ({} observers):",
            result.target_name,
            result.observers.len()
        );

        if !result.has_real_embedding {
            println!("Warning: Target has no embedding — gap values are all 0.");
        }

        let rows: Vec<Vec<String>> = result
            .observers
            .iter()
            .enumerate()
            .map(|(i, obs)| {
                vec![
                    format!("{}", i + 1),
                    obs.observer_name.clone(),
                    format!("{:.4}", obs.gap),
                    obs.assessment.clone(),
                    obs.agrees_with.as_deref().unwrap_or("-").to_string(),
                    obs.disagrees_with.as_deref().unwrap_or("-").to_string(),
                ]
            })
            .collect();
        print_table(
            &[
                "#",
                "Observer",
                "Gap",
                "Assessment",
                "Agrees With",
                "Disagrees With",
            ],
            rows,
        );
    }

    Ok(())
}

pub async fn handle_perception_shift(
    ctx: &AppContext,
    observer: &str,
    target: &str,
    mode: OutputMode,
    no_semantic: bool,
) -> Result<()> {
    let observer_id = resolve_character(ctx, observer, no_semantic).await?;
    let target_id = resolve_character(ctx, target, no_semantic).await?;

    let service = PerceptionService::new(ctx.db.clone());
    let result = service.analyze_shift(&observer_id, &target_id).await?;

    if mode == OutputMode::Json {
        #[derive(Serialize)]
        struct ShiftRow {
            index: usize,
            delta: Option<f32>,
            gap: Option<f32>,
            event: Option<String>,
            timestamp: String,
        }
        #[derive(Serialize)]
        struct ShiftOutput {
            observer: String,
            target: String,
            snapshots: Vec<ShiftRow>,
            trajectory: String,
        }

        let rows: Vec<ShiftRow> = result
            .snapshots
            .iter()
            .enumerate()
            .map(|(i, s)| ShiftRow {
                index: i + 1,
                delta: s.delta,
                gap: s.gap,
                event: s.event.clone(),
                timestamp: s.timestamp.clone(),
            })
            .collect();

        output_json(&ShiftOutput {
            observer: result.observer_name.clone(),
            target: result.target_name.clone(),
            snapshots: rows,
            trajectory: result.trajectory.clone(),
        });
    } else {
        println!(
            "Perception Shift: {} -> {} ({} snapshots)",
            result.observer_name,
            result.target_name,
            result.snapshots.len()
        );

        let rows: Vec<Vec<String>> = result
            .snapshots
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let delta_str = s
                    .delta
                    .map(|d| format!("{:.4}", d))
                    .unwrap_or_else(|| "baseline".to_string());
                let gap_str = s
                    .gap
                    .map(|g| format!("{:.4}", g))
                    .unwrap_or_else(|| "-".to_string());
                vec![
                    format!("{}", i + 1),
                    delta_str,
                    gap_str,
                    s.event.as_deref().unwrap_or("-").to_string(),
                    s.timestamp.clone(),
                ]
            })
            .collect();
        print_table(&["#", "Delta", "Gap", "Event", "Timestamp"], rows);

        println!("\nTrajectory: {}", result.trajectory);
    }

    Ok(())
}

// =============================================================================
// Creation — record perceptions
// =============================================================================

#[allow(clippy::too_many_arguments)]
pub async fn create_perception(
    ctx: &AppContext,
    observer: &str,
    target: &str,
    perception: &str,
    feelings: Option<&str>,
    tension: Option<i32>,
    rel_types: &[String],
    subtype: Option<&str>,
    history: Option<&str>,
    mode: OutputMode,
) -> Result<()> {
    let observer_id = resolve_character(ctx, observer, false).await?;
    let target_id = resolve_character(ctx, target, false).await?;

    // Extract key parts (strip "character:" prefix if present)
    let observer_key = observer_id.split(':').nth(1).unwrap_or(&observer_id);
    let target_key = target_id.split(':').nth(1).unwrap_or(&target_id);

    let data = PerceptionCreate {
        rel_types: if rel_types.is_empty() {
            vec!["general".to_string()]
        } else {
            rel_types.to_vec()
        },
        subtype: subtype.map(String::from),
        feelings: feelings.map(String::from),
        perception: Some(perception.to_string()),
        tension_level: tension,
        history_notes: history.map(String::from),
    };

    let result: Perception = ctx
        .relationship_repo
        .create_perception(observer_key, target_key, data)
        .await?;

    if mode == OutputMode::Json {
        output_json(&result);
    } else {
        print_success(&format!(
            "Recorded perception: {} -> {} ({})",
            observer, target, result.id
        ));
    }

    Ok(())
}

// =============================================================================
// Perspective Search — semantic search across perception edges
// =============================================================================

#[allow(clippy::too_many_arguments)]
pub async fn handle_perspective_search(
    ctx: &AppContext,
    query: &str,
    observer: Option<&str>,
    target: Option<&str>,
    limit: usize,
    mode: OutputMode,
    no_semantic: bool,
) -> Result<()> {
    if !ctx.embedding_service.is_available() {
        anyhow::bail!("Semantic search unavailable — embedding model not loaded.");
    }

    let observer_id = if let Some(o) = observer {
        Some(resolve_character(ctx, o, no_semantic).await?)
    } else {
        None
    };
    let target_id = if let Some(t) = target {
        Some(resolve_character(ctx, t, no_semantic).await?)
    } else {
        None
    };

    let spinner = create_spinner("Searching perspectives...");

    let query_embedding = ctx.embedding_service.embed_text(query).await?;

    // Build filter clause
    let mut filters = String::new();
    if observer_id.is_some() {
        filters.push_str(" AND in = $obs_ref");
    }
    if target_id.is_some() {
        filters.push_str(" AND out = $tgt_ref");
    }

    let query_str = format!(
        "SELECT id, in.name AS observer_name, out.name AS target_name, \
         perception, feelings, tension_level, \
         vector::similarity::cosine(embedding, $query_vec) AS score \
         FROM perceives \
         WHERE embedding IS NOT NONE{} \
         ORDER BY score DESC LIMIT $lim",
        filters
    );

    let mut q = ctx
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
    let mut response = q.await?;

    #[derive(serde::Deserialize, Serialize)]
    struct PerspectiveResult {
        id: surrealdb::sql::Thing,
        observer_name: Option<String>,
        target_name: Option<String>,
        perception: Option<String>,
        feelings: Option<String>,
        tension_level: Option<i32>,
        score: f32,
    }

    let results: Vec<PerspectiveResult> = response.take(0).unwrap_or_default();

    spinner.finish_and_clear();

    if mode == OutputMode::Json {
        output_json(&results);
        return Ok(());
    }

    println!(
        "Perspective search for '{}': {} results\n",
        query,
        results.len()
    );

    let rows: Vec<Vec<String>> = results
        .iter()
        .map(|r| {
            let obs = r.observer_name.as_deref().unwrap_or("?");
            let tgt = r.target_name.as_deref().unwrap_or("?");
            vec![
                format!("{} -> {}", obs, tgt),
                r.perception.as_deref().unwrap_or("-").to_string(),
                r.feelings.as_deref().unwrap_or("-").to_string(),
                r.tension_level
                    .map(|t| format!("{}/10", t))
                    .unwrap_or_else(|| "-".to_string()),
                format!("{:.4}", r.score),
            ]
        })
        .collect();

    print_table(
        &[
            "Observer -> Target",
            "Perception",
            "Feelings",
            "Tension",
            "Score",
        ],
        rows,
    );

    if results.is_empty() {
        print_hint("No perspective matches. Try broader terms or run 'narra world backfill'.");
    }

    Ok(())
}
