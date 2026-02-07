//! Utility command handlers: update, delete, health, backfill, export, validate, graph.

use std::path::Path;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::cli::output::{output_json, print_error, print_success, print_table, OutputMode};
use crate::init::AppContext;
use crate::repository::EntityRepository;

/// Strip a known table prefix from an entity ID, returning the bare key.
fn bare_key(id: &str, prefix: &str) -> String {
    id.strip_prefix(&format!("{}:", prefix))
        .unwrap_or(id)
        .to_string()
}

/// Extract entity type from a full entity ID (e.g., "character:alice" -> "character").
fn entity_type_from_id(entity_id: &str) -> Option<&str> {
    entity_id.split(':').next()
}

// =============================================================================
// Update
// =============================================================================

pub async fn handle_update(
    ctx: &AppContext,
    entity_id: &str,
    fields_json: Option<&str>,
    set_pairs: &[(String, String)],
    mode: OutputMode,
) -> Result<()> {
    let entity_type = entity_type_from_id(entity_id).ok_or_else(|| {
        anyhow::anyhow!(
            "Invalid entity ID '{}'. Expected format: type:key",
            entity_id
        )
    })?;
    let key = bare_key(entity_id, entity_type);

    // Build fields map from either --fields JSON or --set key=value pairs
    let fields: serde_json::Value = if let Some(json_str) = fields_json {
        serde_json::from_str(json_str)?
    } else if !set_pairs.is_empty() {
        let mut map = serde_json::Map::new();
        for (k, v) in set_pairs {
            // Try to parse value as JSON (number, bool, etc.), fall back to string
            let val = serde_json::from_str(v).unwrap_or(serde_json::Value::String(v.clone()));
            map.insert(k.clone(), val);
        }
        serde_json::Value::Object(map)
    } else {
        anyhow::bail!("Provide either --fields or --set to specify update data");
    };

    // Use a raw SurrealQL MERGE to update
    let record_ref = surrealdb::RecordId::from((entity_type, key.as_str()));
    let mut response = ctx
        .db
        .query("UPDATE ONLY $ref MERGE $fields RETURN AFTER")
        .bind(("ref", record_ref))
        .bind(("fields", fields))
        .await?;

    // Try to extract the result as JSON
    #[derive(Deserialize, Serialize)]
    struct IdName {
        id: surrealdb::RecordId,
        #[serde(default)]
        name: Option<String>,
        #[serde(default)]
        title: Option<String>,
    }

    let updated: Option<IdName> = response.take(0)?;

    match updated {
        Some(u) => {
            let display_name = u.name.or(u.title).unwrap_or_else(|| key.clone());
            if mode == OutputMode::Json {
                println!(
                    r#"{{"status": "ok", "id": "{}", "name": "{}"}}"#,
                    u.id, display_name
                );
            } else {
                print_success(&format!("Updated {} '{}'", entity_type, display_name));
            }
        }
        None => print_error(&format!("Entity '{}' not found", entity_id)),
    }

    Ok(())
}

// =============================================================================
// Delete
// =============================================================================

pub async fn handle_delete(
    ctx: &AppContext,
    entity_id: &str,
    _hard: bool,
    mode: OutputMode,
) -> Result<()> {
    let entity_type = entity_type_from_id(entity_id).ok_or_else(|| {
        anyhow::anyhow!(
            "Invalid entity ID '{}'. Expected format: type:key",
            entity_id
        )
    })?;
    let key = bare_key(entity_id, entity_type);

    let deleted_name: Option<String> = match entity_type {
        "character" => {
            let r = ctx.entity_repo.delete_character(&key).await?;
            r.map(|c| c.name)
        }
        "location" => {
            let r = ctx.entity_repo.delete_location(&key).await?;
            r.map(|l| l.name)
        }
        "event" => {
            let r = ctx.entity_repo.delete_event(&key).await?;
            r.map(|e| e.title)
        }
        "scene" => {
            let r = ctx.entity_repo.delete_scene(&key).await?;
            r.map(|s| s.title)
        }
        "universe_fact" => {
            let r = crate::models::fact::delete_fact(&ctx.db, &key).await?;
            r.map(|f| f.title)
        }
        "note" => {
            let r = crate::models::note::delete_note(&ctx.db, &key).await?;
            r.map(|n| n.title)
        }
        _ => anyhow::bail!("Unsupported entity type '{}' for delete", entity_type),
    };

    match deleted_name {
        Some(name) => {
            if mode == OutputMode::Json {
                println!(
                    r#"{{"status": "ok", "deleted": "{}", "name": "{}"}}"#,
                    entity_id, name
                );
            } else {
                print_success(&format!("Deleted {} '{}'", entity_type, name));
            }
        }
        None => print_error(&format!("Entity '{}' not found", entity_id)),
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

#[derive(Deserialize)]
struct CountResult {
    count: usize,
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
    mode: OutputMode,
) -> Result<()> {
    use crate::embedding::BackfillService;

    println!("Starting embedding backfill...");

    let backfill = BackfillService::new(ctx.db.clone(), ctx.embedding_service.clone());
    let stats = backfill.backfill_all().await?;

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

    let export_service = ExportService::new(ctx.db.clone());
    let import = export_service.export_world().await?;

    // Serialize to YAML
    let yaml = serde_yaml_ng::to_string(&import)?;

    // Prepend comment header
    let header = format!(
        "# Narra world export\n# Version: {}\n# Exported: {}\n",
        env!("CARGO_PKG_VERSION"),
        chrono::Utc::now().to_rfc3339()
    );
    let content = format!("{}{}", header, yaml);

    // Determine output path
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
// Validate
// =============================================================================

pub async fn handle_validate(
    ctx: &AppContext,
    entity_id: Option<&str>,
    mode: OutputMode,
) -> Result<()> {
    match entity_id {
        Some(eid) => {
            // Validate a specific entity
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
            // General consistency check - validate all characters
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
        // Parse "character:id" format
        let char_id = bare_key(scope, "character");
        GraphScope::CharacterCentered {
            character_id: char_id,
            depth,
        }
    };

    let options = GraphOptions::default();
    let mermaid = graph_service.generate_mermaid(graph_scope, options).await?;

    // Write to file if specified
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
        // Print to stdout if no output file
        println!("{}", mermaid);
    }

    Ok(())
}
