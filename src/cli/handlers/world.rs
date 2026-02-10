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
    let mermaid = graph_service.generate_mermaid(graph_scope, options).await?;

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
