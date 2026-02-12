//! World management command handlers: status, health, backfill, export, import, validate, graph.

use std::path::Path;

use anyhow::Result;
use colored::Colorize;
use serde::{Deserialize, Serialize};

use crate::cli::output::{
    create_spinner, output_json, print_error, print_header, print_kv, print_success, print_table,
    OutputMode,
};
use crate::cli::resolve::bare_key;
use crate::init::AppContext;

// =============================================================================
// Status — world overview dashboard
// =============================================================================

#[derive(Deserialize)]
struct CountResult {
    count: usize,
}

pub async fn handle_status(ctx: &AppContext, mode: OutputMode) -> Result<()> {
    let tables = [
        ("character", "Characters"),
        ("location", "Locations"),
        ("event", "Events"),
        ("scene", "Scenes"),
        ("knowledge", "Knowledge"),
        ("relates_to", "Relationships"),
        ("note", "Notes"),
    ];

    #[derive(Serialize)]
    struct EntityStatus {
        entity_type: String,
        total: usize,
        embedded: usize,
        coverage: String,
    }

    let mut statuses = Vec::new();
    let mut total_stale = 0usize;

    for (table, label) in &tables {
        // Total count
        let total_q = format!("SELECT count() AS count FROM {} GROUP ALL", table);
        let mut total_resp = ctx.db.query(&total_q).await?;
        let total_rows: Vec<CountResult> = total_resp.take(0).unwrap_or_default();
        let total = total_rows.first().map(|r| r.count).unwrap_or(0);

        // Embedded count
        let embedded_q = format!(
            "SELECT count() AS count FROM {} WHERE embedding IS NOT NONE GROUP ALL",
            table
        );
        let mut embedded_resp = ctx.db.query(&embedded_q).await?;
        let embedded_rows: Vec<CountResult> = embedded_resp.take(0).unwrap_or_default();
        let embedded = embedded_rows.first().map(|r| r.count).unwrap_or(0);

        // Stale count
        let stale_q = format!(
            "SELECT count() AS count FROM {} WHERE embedding_stale = true GROUP ALL",
            table
        );
        let mut stale_resp = ctx.db.query(&stale_q).await?;
        let stale_rows: Vec<CountResult> = stale_resp.take(0).unwrap_or_default();
        let stale = stale_rows.first().map(|r| r.count).unwrap_or(0);
        total_stale += stale;

        let coverage = if total > 0 {
            format!("{:.0}%", (embedded as f64 / total as f64) * 100.0)
        } else {
            "-".to_string()
        };

        statuses.push(EntityStatus {
            entity_type: label.to_string(),
            total,
            embedded,
            coverage,
        });
    }

    let embedding_available = ctx.embedding_service.is_available();
    let dimensions = ctx.embedding_service.dimensions();
    let model_id = ctx.embedding_service.model_id();
    let provider = ctx.embedding_service.provider_name();

    let model_mismatch = matches!(
        ctx.embedding_model_mismatch,
        crate::embedding::provider::ModelMatch::Mismatch { .. }
    );

    if mode == OutputMode::Json {
        let json = serde_json::json!({
            "data_path": ctx.data_path.display().to_string(),
            "entities": statuses,
            "embedding_available": embedding_available,
            "embedding_model": model_id,
            "embedding_provider": provider,
            "embedding_dimensions": dimensions,
            "embedding_model_mismatch": model_mismatch,
            "stale_count": total_stale,
        });
        output_json(&json);
        return Ok(());
    }

    println!("{}", format!("World: {}", ctx.data_path.display()).bold());
    println!();

    let rows: Vec<Vec<String>> = statuses
        .iter()
        .map(|s| {
            vec![
                s.entity_type.clone(),
                s.total.to_string(),
                s.embedded.to_string(),
                s.coverage.clone(),
            ]
        })
        .collect();

    print_table(&["Entities", "Count", "Embedded", "Coverage"], rows);

    println!();
    if embedding_available {
        println!(
            "  Embedding Model: {} via {} ({} dims) {}",
            model_id,
            provider,
            dimensions,
            "OK".green()
        );
    } else {
        println!(
            "  Embedding Model: {} {}",
            "unavailable".yellow(),
            "(semantic search disabled)".dimmed()
        );
    }
    if model_mismatch {
        if let crate::embedding::provider::ModelMatch::Mismatch {
            stored_model,
            current_model,
            ..
        } = &ctx.embedding_model_mismatch
        {
            println!(
                "  {} Model mismatch: world embedded with '{}', current is '{}'",
                "WARNING:".yellow().bold(),
                stored_model,
                current_model
            );
            println!(
                "  {}",
                "Run 'narra world backfill --force' to re-embed.".dimmed()
            );
        }
    }
    if total_stale > 0 {
        println!(
            "  Stale: {} entities need re-embedding",
            total_stale.to_string().yellow()
        );
    }

    Ok(())
}

// =============================================================================
// Health
// =============================================================================

#[derive(Serialize)]
struct HealthTable {
    table: String,
    total: usize,
    embedded: usize,
    stale: usize,
}

pub async fn handle_health(ctx: &AppContext, mode: OutputMode) -> Result<()> {
    let tables = vec![
        ("character", true),
        ("location", true),
        ("event", true),
        ("scene", true),
        ("knowledge", true),
        ("relates_to", true),
        ("perceives", true),
    ];

    let mut health_rows: Vec<HealthTable> = Vec::new();

    for (table, _) in &tables {
        let total_q = format!("SELECT count() AS count FROM {} GROUP ALL", table);
        let mut total_resp = ctx.db.query(&total_q).await?;
        let total_rows: Vec<CountResult> = total_resp.take(0).unwrap_or_default();
        let total = total_rows.first().map(|r| r.count).unwrap_or(0);

        let embedded_q = format!(
            "SELECT count() AS count FROM {} WHERE embedding IS NOT NONE GROUP ALL",
            table
        );
        let mut embedded_resp = ctx.db.query(&embedded_q).await?;
        let embedded_rows: Vec<CountResult> = embedded_resp.take(0).unwrap_or_default();
        let embedded = embedded_rows.first().map(|r| r.count).unwrap_or(0);

        let stale_q = format!(
            "SELECT count() AS count FROM {} WHERE embedding_stale = true GROUP ALL",
            table
        );
        let mut stale_resp = ctx.db.query(&stale_q).await?;
        let stale_rows: Vec<CountResult> = stale_resp.take(0).unwrap_or_default();
        let stale = stale_rows.first().map(|r| r.count).unwrap_or(0);

        health_rows.push(HealthTable {
            table: table.to_string(),
            total,
            embedded,
            stale,
        });
    }

    if mode == OutputMode::Json {
        output_json(&health_rows);
        return Ok(());
    }

    let rows: Vec<Vec<String>> = health_rows
        .iter()
        .map(|h| {
            let coverage = if h.total > 0 {
                format!("{:.0}%", (h.embedded as f64 / h.total as f64) * 100.0)
            } else {
                "-".to_string()
            };
            vec![
                h.table.clone(),
                h.total.to_string(),
                h.embedded.to_string(),
                h.stale.to_string(),
                coverage,
            ]
        })
        .collect();

    println!("Embedding Health Report\n");
    print_table(&["Table", "Total", "Embedded", "Stale", "Coverage"], rows);
    Ok(())
}

// =============================================================================
// Backfill
// =============================================================================

pub async fn handle_backfill(
    ctx: &AppContext,
    _entity_type: Option<&str>,
    force: bool,
    mode: OutputMode,
) -> Result<()> {
    use crate::embedding::BackfillService;

    // If --force, mark all entities as needing re-embedding
    if force {
        let tables = [
            "character",
            "location",
            "event",
            "scene",
            "knowledge",
            "perceives",
            "relates_to",
        ];
        for table in &tables {
            let query = format!(
                "UPDATE {} SET embedding_stale = true WHERE embedding IS NOT NONE",
                table
            );
            ctx.db.query(&query).await?;
        }
        if mode != OutputMode::Json {
            println!("Marked all entities as stale for re-embedding.");
        }
    }

    let spinner = create_spinner("Generating embeddings...");

    let backfill = BackfillService::new(ctx.db.clone(), ctx.embedding_service.clone());
    let stats = backfill.backfill_all().await?;

    spinner.finish_and_clear();

    if mode == OutputMode::Json {
        output_json(&stats);
    } else {
        println!("\nBackfill complete:");
        println!("  Total entities: {}", stats.total_entities);
        println!("  Embedded:       {}", stats.embedded);
        println!("  Skipped:        {}", stats.skipped);
        println!("  Failed:         {}", stats.failed);
        if !stats.entity_type_stats.is_empty() {
            println!("  By type:");
            for (t, count) in &stats.entity_type_stats {
                println!("    {}: {}", t, count);
            }
        }
    }

    Ok(())
}

// =============================================================================
// Export
// =============================================================================

pub async fn handle_export(
    ctx: &AppContext,
    output: Option<&Path>,
    mode: OutputMode,
) -> Result<()> {
    use crate::services::export::ExportService;

    let spinner = create_spinner("Exporting world data...");

    let export_service = ExportService::new(ctx.db.clone());
    let import = export_service.export_world().await?;

    spinner.finish_and_clear();

    let yaml = serde_yaml_ng::to_string(&import)?;

    let header = format!(
        "# Narra world export\n# Version: {}\n# Exported: {}\n",
        env!("CARGO_PKG_VERSION"),
        chrono::Utc::now().to_rfc3339()
    );
    let content = format!("{}{}", header, yaml);

    let default_path = format!(
        "./narra-export-{}.yaml",
        chrono::Utc::now().format("%Y-%m-%d")
    );
    let output_path = output
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| std::path::PathBuf::from(&default_path));

    std::fs::write(&output_path, &content)?;

    let summary = serde_json::json!({
        "output_path": output_path.display().to_string(),
        "characters": import.characters.len(),
        "locations": import.locations.len(),
        "events": import.events.len(),
        "scenes": import.scenes.len(),
        "relationships": import.relationships.len(),
        "knowledge": import.knowledge.len(),
        "notes": import.notes.len(),
        "facts": import.facts.len(),
    });

    if mode == OutputMode::Json {
        output_json(&summary);
    } else {
        print_success(&format!("Exported world data to {}", output_path.display()));
        println!("  Characters:    {}", import.characters.len());
        println!("  Locations:     {}", import.locations.len());
        println!("  Events:        {}", import.events.len());
        println!("  Scenes:        {}", import.scenes.len());
        println!("  Relationships: {}", import.relationships.len());
        println!("  Knowledge:     {}", import.knowledge.len());
        println!("  Notes:         {}", import.notes.len());
        println!("  Facts:         {}", import.facts.len());
    }

    Ok(())
}

// =============================================================================
// Import
// =============================================================================

pub async fn handle_import(
    ctx: &AppContext,
    file: &Path,
    on_conflict: &str,
    dry_run: bool,
    mode: OutputMode,
) -> Result<()> {
    use crate::mcp::types::{ConflictMode, NarraImport};
    use crate::services::import::ImportService;

    let content = std::fs::read_to_string(file)
        .map_err(|e| anyhow::anyhow!("Failed to read file '{}': {}", file.display(), e))?;

    let import: NarraImport = serde_yaml_ng::from_str(&content)
        .map_err(|e| anyhow::anyhow!("Failed to parse YAML: {}", e))?;

    let conflict_mode = match on_conflict.to_lowercase().as_str() {
        "skip" => ConflictMode::Skip,
        "update" => ConflictMode::Update,
        _ => ConflictMode::Error,
    };

    if dry_run {
        let counts = [
            ("characters", import.characters.len()),
            ("locations", import.locations.len()),
            ("events", import.events.len()),
            ("scenes", import.scenes.len()),
            ("relationships", import.relationships.len()),
            ("knowledge", import.knowledge.len()),
            ("notes", import.notes.len()),
            ("facts", import.facts.len()),
        ];

        let total: usize = counts.iter().map(|(_, c)| c).sum();

        match mode {
            OutputMode::Json => {
                let json_counts: std::collections::HashMap<&str, usize> =
                    counts.iter().copied().collect();
                output_json(&serde_json::json!({
                    "dry_run": true,
                    "total": total,
                    "by_type": json_counts,
                    "on_conflict": on_conflict,
                }));
            }
            OutputMode::Human | OutputMode::Markdown => {
                println!("Dry run — no changes will be made\n");
                let rows: Vec<Vec<String>> = counts
                    .iter()
                    .filter(|(_, c)| *c > 0)
                    .map(|(t, c)| vec![t.to_string(), c.to_string()])
                    .collect();
                print_table(&["Entity Type", "Count"], rows);
                println!("\nTotal: {} entities", total);
                println!("Conflict mode: {}", on_conflict);
            }
        }
        return Ok(());
    }

    let spinner = create_spinner("Importing world data...");

    let import_service = ImportService::new(ctx.db.clone(), ctx.staleness_manager.clone());
    let result = import_service
        .execute_import(import, conflict_mode)
        .await
        .map_err(|e| anyhow::anyhow!("Import failed: {}", e))?;

    spinner.finish_and_clear();

    match mode {
        OutputMode::Json => {
            output_json(&result);
        }
        OutputMode::Human | OutputMode::Markdown => {
            let rows: Vec<Vec<String>> = result
                .by_type
                .iter()
                .filter(|t| t.created > 0 || t.skipped > 0 || t.updated > 0 || !t.errors.is_empty())
                .map(|t| {
                    vec![
                        t.entity_type.clone(),
                        t.created.to_string(),
                        t.skipped.to_string(),
                        t.updated.to_string(),
                        t.errors.len().to_string(),
                    ]
                })
                .collect();

            print_table(
                &["Entity Type", "Created", "Skipped", "Updated", "Errors"],
                rows,
            );

            print_success(&format!(
                "Import complete: {} created, {} skipped, {} updated, {} errors",
                result.total_created,
                result.total_skipped,
                result.total_updated,
                result.total_errors,
            ));

            for type_result in &result.by_type {
                for err in &type_result.errors {
                    print_error(&format!("[{}] {}", type_result.entity_type, err));
                }
            }

            if result.total_created > 0 {
                println!(
                    "\nRun 'narra world backfill' to generate embeddings for imported entities."
                );
            }
        }
    }

    Ok(())
}

// =============================================================================
// Validate
// =============================================================================

pub async fn handle_validate(
    ctx: &AppContext,
    entity_id: Option<&str>,
    mode: OutputMode,
) -> Result<()> {
    match entity_id {
        Some(eid) => {
            let result = ctx
                .consistency_service
                .check_entity_mutation(eid, &serde_json::json!({}))
                .await?;

            if mode == OutputMode::Json {
                output_json(&result);
            } else if result.is_valid {
                print_success(&format!("Entity '{}' is valid (no violations)", eid));
            } else {
                println!("Validation result for '{}':", eid);
                println!("  Valid: {}", result.is_valid);
                println!("  Total violations: {}", result.total_violations);
                for (severity, violations) in &result.violations_by_severity {
                    println!("  {:?}:", severity);
                    for v in violations {
                        println!("    - [{}] {}", v.fact_title, v.message);
                    }
                }
            }
        }
        None => {
            #[derive(Deserialize)]
            struct IdOnly {
                id: surrealdb::RecordId,
            }

            let mut resp = ctx.db.query("SELECT id FROM character").await?;
            let characters: Vec<IdOnly> = resp.take(0).unwrap_or_default();

            let mut total_violations = 0;
            let mut checked = 0;

            for c in &characters {
                let char_id = c.id.to_string();
                let result = ctx
                    .consistency_service
                    .check_entity_mutation(&char_id, &serde_json::json!({}))
                    .await?;
                total_violations += result.total_violations;
                checked += 1;

                if !result.is_valid && mode != OutputMode::Json {
                    println!("  {} has {} violations", char_id, result.total_violations);
                }
            }

            if mode == OutputMode::Json {
                println!(
                    r#"{{"checked": {}, "total_violations": {}}}"#,
                    checked, total_violations
                );
            } else {
                println!(
                    "\nValidation complete: checked {} entities, {} total violations",
                    checked, total_violations
                );
            }
        }
    }

    Ok(())
}

// =============================================================================
// Graph
// =============================================================================

pub async fn handle_graph(
    ctx: &AppContext,
    scope: &str,
    depth: usize,
    output: Option<&Path>,
    phases: bool,
    mode: OutputMode,
) -> Result<()> {
    use crate::services::{GraphOptions, GraphScope, GraphService, MermaidGraphService};

    let graph_service = MermaidGraphService::new(ctx.db.clone());

    let graph_scope = if scope == "full" {
        GraphScope::FullNetwork
    } else {
        let char_id = bare_key(scope, "character");
        GraphScope::CharacterCentered {
            character_id: char_id,
            depth,
        }
    };

    let options = GraphOptions::default();
    let mut mermaid = graph_service.generate_mermaid(graph_scope, options).await?;

    // Append phase coloring if requested
    if phases {
        use crate::services::{EntityType, TemporalService};

        let temporal_service = TemporalService::new(ctx.db.clone());
        match temporal_service
            .load_or_detect_phases(EntityType::embeddable(), None, None)
            .await
        {
            Ok(result) => {
                let phase_section = generate_phase_styles(&result);
                // Insert phase styles before the closing ``` fence
                if let Some(fence_pos) = mermaid.rfind("\n```\n") {
                    mermaid.insert_str(fence_pos, &phase_section);
                } else {
                    mermaid.push_str(&phase_section);
                }
            }
            Err(e) => {
                if mode != OutputMode::Json {
                    eprintln!(
                        "Warning: phase detection failed ({}), generating graph without colors",
                        e
                    );
                }
            }
        }
    }

    if let Some(path) = output {
        std::fs::write(path, &mermaid)?;
        if mode != OutputMode::Json {
            print_success(&format!("Graph written to {}", path.display()));
        }
    }

    if mode == OutputMode::Json {
        let result = serde_json::json!({
            "format": "mermaid",
            "content": mermaid,
            "output_path": output.map(|p| p.display().to_string()),
        });
        output_json(&result);
    } else if output.is_none() {
        println!("{}", mermaid);
    }

    Ok(())
}

/// Generate Mermaid style definitions for phase coloring.
pub fn generate_phase_styles(result: &crate::services::PhaseDetectionResult) -> String {
    // Phase color palette (up to 8 phases, then cycles)
    const PHASE_COLORS: &[(&str, &str)] = &[
        ("#3b82f6", "#dbeafe"), // blue
        ("#ef4444", "#fee2e2"), // red
        ("#22c55e", "#dcfce7"), // green
        ("#f59e0b", "#fef3c7"), // amber
        ("#8b5cf6", "#ede9fe"), // purple
        ("#14b8a6", "#ccfbf1"), // teal
        ("#f97316", "#ffedd5"), // orange
        ("#ec4899", "#fce7f3"), // pink
    ];

    let mut lines = Vec::new();
    lines.push(String::new());
    lines.push("    %% Phase coloring".to_string());

    for phase in &result.phases {
        let (stroke, fill) = PHASE_COLORS[phase.phase_id % PHASE_COLORS.len()];
        let class_name = format!("phase{}", phase.phase_id);
        lines.push(format!(
            "    classDef {} fill:{},stroke:{},stroke-width:2px",
            class_name, fill, stroke
        ));
    }

    // Assign entities to phase classes (only character nodes in the graph)
    for phase in &result.phases {
        let char_ids: Vec<String> = phase
            .members
            .iter()
            .filter(|m| m.entity_type == "character")
            .filter_map(|m| m.entity_id.split(':').nth(1).map(|s| s.to_string()))
            .collect();

        if !char_ids.is_empty() {
            let class_name = format!("phase{}", phase.phase_id);
            lines.push(format!("    class {} {}", char_ids.join(","), class_name));
        }
    }

    // Add phase legend as comment
    lines.push(String::new());
    lines.push("    %% Phase legend:".to_string());
    for phase in &result.phases {
        let (stroke, _) = PHASE_COLORS[phase.phase_id % PHASE_COLORS.len()];
        lines.push(format!(
            "    %% Phase {}: {} ({})",
            phase.phase_id, phase.label, stroke
        ));
    }

    lines.join("\n")
}

// =============================================================================
// Baseline Arc Snapshots
// =============================================================================

pub async fn handle_baseline_arcs(
    ctx: &AppContext,
    entity_type: Option<&str>,
    mode: OutputMode,
) -> Result<()> {
    // Validate entity_type if provided
    if let Some(et) = entity_type {
        if !matches!(et, "character" | "knowledge") {
            anyhow::bail!(
                "Invalid entity type '{}'. Must be 'character' or 'knowledge'.",
                et
            );
        }
    }

    let types_to_process: Vec<&str> = match entity_type {
        Some(et) => vec![et],
        None => vec!["character", "knowledge"],
    };

    let mut total_created = 0usize;
    let mut total_skipped = 0usize;

    for etype in &types_to_process {
        // Fetch entities that have embeddings
        let fetch_query = format!(
            "SELECT id, embedding FROM {} WHERE embedding IS NOT NONE",
            etype
        );

        let mut response = ctx.db.query(&fetch_query).await?;

        #[derive(Deserialize)]
        struct EntityWithEmbedding {
            id: surrealdb::RecordId,
            embedding: Vec<f32>,
        }

        let entities: Vec<EntityWithEmbedding> = response.take(0).unwrap_or_default();

        for entity in &entities {
            let entity_id = entity.id.to_string();

            // Check if snapshot already exists
            let mut count_response = ctx
                .db
                .query("SELECT count() AS cnt FROM arc_snapshot WHERE entity_id = $eid GROUP ALL")
                .bind(("eid", entity.id.clone()))
                .await?;

            #[derive(Deserialize)]
            struct ArcCount {
                cnt: i64,
            }

            let count: Option<ArcCount> = count_response.take(0).unwrap_or(None);
            let existing = count.map(|c| c.cnt).unwrap_or(0);

            if existing > 0 {
                total_skipped += 1;
                continue;
            }

            // Create baseline snapshot
            if let Err(e) = ctx
                .db
                .query(
                    "CREATE arc_snapshot SET entity_id = $eid, entity_type = $etype, embedding = $embedding",
                )
                .bind(("eid", entity.id.clone()))
                .bind(("etype", etype.to_string()))
                .bind(("embedding", entity.embedding.clone()))
                .await
            {
                eprintln!("Warning: Failed to create snapshot for {}: {}", entity_id, e);
                continue;
            }

            total_created += 1;
        }
    }

    if mode == OutputMode::Json {
        output_json(&serde_json::json!({
            "created": total_created,
            "skipped": total_skipped,
            "entity_types": types_to_process,
        }));
    } else {
        print_header("Baseline Arc Snapshots");
        print_kv("Created", &total_created.to_string());
        print_kv(
            "Skipped (already had snapshots)",
            &total_skipped.to_string(),
        );
        if total_created > 0 {
            print_success(&format!("Created {} baseline snapshots", total_created));
        } else if total_skipped > 0 {
            println!("  All entities already have snapshots.");
        } else {
            println!("  No entities with embeddings found. Run 'narra world backfill' first.");
        }
    }

    Ok(())
}

// =============================================================================
// Benchmark — compare embedding model quality
// =============================================================================

/// Spearman rank correlation coefficient between two rank orderings.
fn spearman_correlation(ranks_a: &[usize], ranks_b: &[usize]) -> f64 {
    let n = ranks_a.len();
    if n < 2 {
        return 1.0;
    }
    let d_squared_sum: f64 = ranks_a
        .iter()
        .zip(ranks_b.iter())
        .map(|(a, b)| {
            let d = *a as f64 - *b as f64;
            d * d
        })
        .sum();
    let n_f = n as f64;
    1.0 - (6.0 * d_squared_sum) / (n_f * (n_f * n_f - 1.0))
}

pub async fn handle_benchmark(
    ctx: &AppContext,
    comparison_model: &str,
    queries: &[String],
    sample: usize,
    limit: usize,
    mode: OutputMode,
) -> Result<()> {
    use crate::embedding::provider::{create_embedding_service, EmbeddingProviderConfig};
    use crate::utils::math::cosine_similarity;

    let current_model = ctx.embedding_service.model_id().to_string();
    let current_dims = ctx.embedding_service.dimensions();

    if comparison_model == current_model {
        anyhow::bail!(
            "Comparison model '{}' is the same as current model. Choose a different model.",
            comparison_model
        );
    }

    if mode != OutputMode::Json {
        println!(
            "Benchmark: {} ({} dims) vs {} (comparison)\n",
            current_model, current_dims, comparison_model
        );
    }

    // Load comparison model
    let spinner = create_spinner(&format!(
        "Loading comparison model '{}'...",
        comparison_model
    ));
    let comp_config = EmbeddingProviderConfig::Candle {
        model: comparison_model.to_string(),
        cache_dir: None,
        show_download_progress: true,
    };
    let comp_service = create_embedding_service(&comp_config)
        .map_err(|e| anyhow::anyhow!("Failed to load comparison model: {}", e))?;
    let comp_dims = comp_service.dimensions();
    spinner.finish_and_clear();

    if mode != OutputMode::Json {
        println!(
            "  Loaded '{}' ({} dims)\n",
            comp_service.model_id(),
            comp_dims
        );
    }

    // Fetch sample entities with composite_text
    let tables = [
        "character",
        "location",
        "event",
        "scene",
        "knowledge",
        "note",
    ];
    let per_table = (sample / tables.len()).max(1);

    #[derive(Deserialize, Clone)]
    struct TextEntity {
        id: surrealdb::RecordId,
        composite_text: String,
        embedding: Option<Vec<f32>>,
    }

    let mut entities: Vec<TextEntity> = Vec::new();

    for table in &tables {
        let query = format!(
            "SELECT id, composite_text, embedding FROM {} WHERE composite_text IS NOT NONE LIMIT {}",
            table, per_table
        );
        let mut resp = ctx.db.query(&query).await?;
        let rows: Vec<TextEntity> = resp.take(0).unwrap_or_default();
        entities.extend(rows);
    }

    if entities.is_empty() {
        anyhow::bail!("No entities with composite_text found. Run 'narra world backfill' first.");
    }

    let entity_count = entities.len();

    if mode != OutputMode::Json {
        println!("  Sampled {} entities for comparison\n", entity_count);
    }

    // Batch-embed with comparison model
    let spinner = create_spinner("Embedding sample with comparison model...");
    let texts: Vec<String> = entities.iter().map(|e| e.composite_text.clone()).collect();
    let comp_start = std::time::Instant::now();
    let comp_embeddings = comp_service.embed_batch(&texts).await?;
    let comp_elapsed = comp_start.elapsed();
    spinner.finish_and_clear();

    // Also time current model on same texts for latency comparison
    let spinner = create_spinner("Embedding sample with current model...");
    let curr_start = std::time::Instant::now();
    let curr_embeddings = ctx.embedding_service.embed_batch(&texts).await?;
    let curr_elapsed = curr_start.elapsed();
    spinner.finish_and_clear();

    // Generate test queries
    let test_queries: Vec<String> = if queries.is_empty() {
        // Derive from entity IDs — extract the key portion as a natural query
        entities
            .iter()
            .take(10)
            .map(|e| {
                let key = e.id.key().to_string();
                key.replace(['⟨', '⟩'], "").replace(['_', '-'], " ")
            })
            .collect()
    } else {
        queries.to_vec()
    };

    if mode != OutputMode::Json {
        println!("  Running {} test queries...\n", test_queries.len());
    }

    // Per-query comparison
    #[derive(Serialize)]
    struct QueryResult {
        query: String,
        current_top_ids: Vec<String>,
        comparison_top_ids: Vec<String>,
        current_avg_score: f64,
        comparison_avg_score: f64,
        score_delta: f64,
        rank_correlation: f64,
    }

    let mut results: Vec<QueryResult> = Vec::new();

    for q in &test_queries {
        // Embed query with both models
        let curr_q_emb = ctx.embedding_service.embed_text(q).await?;
        let comp_q_emb = comp_service.embed_text(q).await?;

        // Score all entities with current model
        let mut curr_scores: Vec<(usize, f32)> = entities
            .iter()
            .enumerate()
            .map(|(i, e)| {
                let emb = e.embedding.as_ref().unwrap_or(&curr_embeddings[i]);
                if emb.len() == curr_q_emb.len() {
                    (i, cosine_similarity(&curr_q_emb, emb))
                } else {
                    (i, cosine_similarity(&curr_q_emb, &curr_embeddings[i]))
                }
            })
            .collect();
        curr_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Score all entities with comparison model
        let mut comp_scores: Vec<(usize, f32)> = comp_embeddings
            .iter()
            .enumerate()
            .map(|(i, emb)| (i, cosine_similarity(&comp_q_emb, emb)))
            .collect();
        comp_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let top_k = limit.min(entity_count);

        let curr_top: Vec<(usize, f32)> = curr_scores.iter().take(top_k).copied().collect();
        let comp_top: Vec<(usize, f32)> = comp_scores.iter().take(top_k).copied().collect();

        let curr_avg: f64 = curr_top.iter().map(|(_, s)| *s as f64).sum::<f64>() / top_k as f64;
        let comp_avg: f64 = comp_top.iter().map(|(_, s)| *s as f64).sum::<f64>() / top_k as f64;

        // Compute rank correlation on the union of top-k entity indices
        let curr_rank_map: std::collections::HashMap<usize, usize> = curr_scores
            .iter()
            .enumerate()
            .map(|(rank, (idx, _))| (*idx, rank))
            .collect();
        let comp_rank_map: std::collections::HashMap<usize, usize> = comp_scores
            .iter()
            .enumerate()
            .map(|(rank, (idx, _))| (*idx, rank))
            .collect();

        let mut union_indices: Vec<usize> = curr_top.iter().map(|(i, _)| *i).collect();
        for (i, _) in &comp_top {
            if !union_indices.contains(i) {
                union_indices.push(*i);
            }
        }

        let curr_ranks: Vec<usize> = union_indices
            .iter()
            .map(|i| *curr_rank_map.get(i).unwrap_or(&entity_count))
            .collect();
        let comp_ranks: Vec<usize> = union_indices
            .iter()
            .map(|i| *comp_rank_map.get(i).unwrap_or(&entity_count))
            .collect();

        let rank_corr = spearman_correlation(&curr_ranks, &comp_ranks);

        results.push(QueryResult {
            query: q.clone(),
            current_top_ids: curr_top
                .iter()
                .map(|(i, _)| entities[*i].id.to_string())
                .collect(),
            comparison_top_ids: comp_top
                .iter()
                .map(|(i, _)| entities[*i].id.to_string())
                .collect(),
            current_avg_score: curr_avg,
            comparison_avg_score: comp_avg,
            score_delta: comp_avg - curr_avg,
            rank_correlation: rank_corr,
        });
    }

    // Summary stats
    let avg_rank_corr: f64 =
        results.iter().map(|r| r.rank_correlation).sum::<f64>() / results.len() as f64;
    let avg_curr_score: f64 =
        results.iter().map(|r| r.current_avg_score).sum::<f64>() / results.len() as f64;
    let avg_comp_score: f64 =
        results.iter().map(|r| r.comparison_avg_score).sum::<f64>() / results.len() as f64;
    let avg_score_delta: f64 = avg_comp_score - avg_curr_score;

    let curr_latency_ms = curr_elapsed.as_millis() as f64 / entity_count as f64;
    let comp_latency_ms = comp_elapsed.as_millis() as f64 / entity_count as f64;

    if mode == OutputMode::Json {
        output_json(&serde_json::json!({
            "current_model": current_model,
            "current_dimensions": current_dims,
            "comparison_model": comparison_model,
            "comparison_dimensions": comp_dims,
            "entities_sampled": entity_count,
            "queries": results,
            "summary": {
                "avg_current_score": avg_curr_score,
                "avg_comparison_score": avg_comp_score,
                "avg_score_delta": avg_score_delta,
                "avg_rank_correlation": avg_rank_corr,
                "current_latency_ms_per_entity": curr_latency_ms,
                "comparison_latency_ms_per_entity": comp_latency_ms,
            }
        }));
        return Ok(());
    }

    // Model comparison table
    print_header("Model Comparison");
    print_table(
        &["Property", &current_model, comparison_model],
        vec![
            vec![
                "Dimensions".to_string(),
                current_dims.to_string(),
                comp_dims.to_string(),
            ],
            vec![
                "Avg latency/entity".to_string(),
                format!("{:.1}ms", curr_latency_ms),
                format!("{:.1}ms", comp_latency_ms),
            ],
            vec![
                "Avg top-k score".to_string(),
                format!("{:.4}", avg_curr_score),
                format!("{:.4}", avg_comp_score),
            ],
        ],
    );

    // Per-query results
    print_header("Per-Query Results");
    let query_rows: Vec<Vec<String>> = results
        .iter()
        .map(|r| {
            let q_display = if r.query.len() > 30 {
                format!("{}...", &r.query[..27])
            } else {
                r.query.clone()
            };
            vec![
                q_display,
                format!("{:.4}", r.current_avg_score),
                format!("{:.4}", r.comparison_avg_score),
                format!("{:+.4}", r.score_delta),
                format!("{:.3}", r.rank_correlation),
            ]
        })
        .collect();
    print_table(
        &["Query", "Current", "Comparison", "Delta", "Rank Corr"],
        query_rows,
    );

    // Summary
    print_header("Summary");
    print_kv("Entities sampled", &entity_count.to_string());
    print_kv("Queries tested", &results.len().to_string());
    print_kv("Avg score delta", &format!("{:+.4}", avg_score_delta));
    print_kv(
        "Avg rank agreement",
        &format!("{:.1}%", avg_rank_corr * 100.0),
    );
    print_kv(
        "Latency ratio",
        &format!("{:.1}x", comp_latency_ms / curr_latency_ms.max(0.01)),
    );

    if avg_score_delta > 0.01 {
        println!(
            "\n  {} Comparison model shows higher similarity scores. Consider switching.",
            "Recommendation:".green().bold()
        );
    } else if avg_score_delta < -0.01 {
        println!(
            "\n  {} Current model scores higher. No benefit to switching.",
            "Recommendation:".green().bold()
        );
    } else {
        println!(
            "\n  {} Scores are similar. Consider latency/size tradeoff.",
            "Recommendation:".yellow().bold()
        );
    }

    Ok(())
}

pub async fn handle_annotate(
    ctx: &crate::init::AppContext,
    entity_types: &[String],
    skip_emotions: bool,
    skip_themes: bool,
    skip_ner: bool,
    concurrency: usize,
    mode: crate::cli::OutputMode,
) -> Result<(), crate::NarraError> {
    use crate::services::{AnnotationPipeline, PipelineConfig};

    let types: Vec<String> = if entity_types.is_empty() {
        vec![
            "character".to_string(),
            "event".to_string(),
            "scene".to_string(),
        ]
    } else {
        entity_types.to_vec()
    };
    let type_refs: Vec<&str> = types.iter().map(|s| s.as_str()).collect();

    let config = PipelineConfig {
        run_emotions: !skip_emotions,
        run_themes: !skip_themes,
        run_ner: !skip_ner,
        concurrency,
    };

    let pipeline = AnnotationPipeline::new(
        ctx.db.clone(),
        ctx.emotion_service.clone(),
        ctx.theme_service.clone(),
        ctx.ner_service.clone(),
    );

    let progress = crate::services::noop_progress();

    println!("Running annotation pipeline on types: {}", types.join(", "));

    let report = pipeline.annotate_all(&type_refs, config, progress).await?;

    match mode {
        crate::cli::OutputMode::Json => {
            println!(
                "{}",
                serde_json::to_string_pretty(&report).unwrap_or_default()
            );
        }
        _ => {
            println!("\nAnnotation Pipeline Report");
            println!("==========================");
            println!("Total processed: {}", report.total_processed);
            println!("Emotion successes: {}", report.emotion_successes);
            println!("Theme successes: {}", report.theme_successes);
            println!("NER successes: {}", report.ner_successes);
            if report.errors > 0 {
                println!("Errors: {}", report.errors);
            }
        }
    }

    Ok(())
}
