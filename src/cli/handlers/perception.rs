//! CLI handlers for perception analysis commands.

use anyhow::Result;
use serde::Serialize;

use crate::cli::output::{output_json, print_header, print_kv, print_table, OutputMode};
use crate::cli::resolve::resolve_by_name;
use crate::init::AppContext;
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
