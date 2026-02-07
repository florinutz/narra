use std::path::Path;

use crate::cli::output::{output_json, print_error, print_success, print_table, OutputMode};
use crate::init::AppContext;
use crate::mcp::types::{ConflictMode, NarraImport};
use crate::services::import::ImportService;

pub async fn handle_import(
    ctx: &AppContext,
    file: &Path,
    on_conflict: &str,
    dry_run: bool,
    mode: OutputMode,
) -> anyhow::Result<()> {
    // Read and parse YAML
    let content = std::fs::read_to_string(file)
        .map_err(|e| anyhow::anyhow!("Failed to read file '{}': {}", file.display(), e))?;

    let import: NarraImport = serde_yaml_ng::from_str(&content)
        .map_err(|e| anyhow::anyhow!("Failed to parse YAML: {}", e))?;

    // Parse conflict mode
    let conflict_mode = match on_conflict.to_lowercase().as_str() {
        "skip" => ConflictMode::Skip,
        "update" => ConflictMode::Update,
        _ => ConflictMode::Error,
    };

    if dry_run {
        // Dry run: just show counts
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
            OutputMode::Human => {
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

    // Execute import
    let import_service = ImportService::new(ctx.db.clone(), ctx.staleness_manager.clone());
    let result = import_service
        .execute_import(import, conflict_mode)
        .await
        .map_err(|e| anyhow::anyhow!("Import failed: {}", e))?;

    match mode {
        OutputMode::Json => {
            output_json(&result);
        }
        OutputMode::Human => {
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

            // Print error details
            for type_result in &result.by_type {
                for err in &type_result.errors {
                    print_error(&format!("[{}] {}", type_result.entity_type, err));
                }
            }

            if result.total_created > 0 {
                println!("\nRun 'narra backfill' to generate embeddings for imported entities.");
            }
        }
    }

    Ok(())
}
