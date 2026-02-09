//! CLI handlers for connection paths and reverse reference queries.

use anyhow::Result;

use crate::cli::output::{output_json, output_json_list, print_table, OutputMode};
use crate::cli::resolve::resolve_single;
use crate::init::AppContext;
use crate::services::MermaidGraphService;

pub async fn handle_path(
    ctx: &AppContext,
    from: &str,
    to: &str,
    max_hops: usize,
    include_events: bool,
    mode: OutputMode,
    no_semantic: bool,
) -> Result<()> {
    let from_id = resolve_single(ctx, from, no_semantic).await?;
    let to_id = resolve_single(ctx, to, no_semantic).await?;

    let service = MermaidGraphService::new(ctx.db.clone());
    let paths = service
        .find_connection_paths(&from_id, &to_id, max_hops, include_events)
        .await
        .map_err(|e| anyhow::anyhow!("Connection path search failed: {}", e))?;

    if mode == OutputMode::Json {
        output_json(&paths);
    } else if paths.is_empty() {
        println!(
            "No connection paths found between {} and {} (max {} hops)",
            from_id, to_id, max_hops
        );
    } else {
        println!(
            "Connection paths: {} -> {} ({} found)",
            from_id,
            to_id,
            paths.len()
        );
        let rows: Vec<Vec<String>> = paths
            .iter()
            .enumerate()
            .map(|(i, path)| {
                let chain: String = path
                    .steps
                    .iter()
                    .map(|s| {
                        if s.connection_type.is_empty() {
                            s.entity_id.clone()
                        } else {
                            format!(" --{}--> {}", s.connection_type, s.entity_id)
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("");
                vec![format!("{}", i + 1), chain, format!("{}", path.total_hops)]
            })
            .collect();
        print_table(&["#", "Path", "Hops"], rows);
    }

    Ok(())
}

pub async fn handle_references(
    ctx: &AppContext,
    entity: &str,
    types: Option<Vec<String>>,
    limit: usize,
    mode: OutputMode,
    no_semantic: bool,
) -> Result<()> {
    let entity_id = resolve_single(ctx, entity, no_semantic).await?;

    let service = MermaidGraphService::new(ctx.db.clone());
    let refs = service
        .get_referencing_entities(&entity_id, types, limit)
        .await
        .map_err(|e| anyhow::anyhow!("Reverse reference query failed: {}", e))?;

    if mode == OutputMode::Json {
        output_json_list(&refs);
    } else if refs.is_empty() {
        println!("No entities reference {}", entity_id);
    } else {
        println!("Entities referencing {} ({} found):", entity_id, refs.len());
        let rows: Vec<Vec<String>> = refs
            .iter()
            .map(|r| {
                vec![
                    r.entity_type.clone(),
                    r.entity_id.clone(),
                    r.reference_field.clone(),
                ]
            })
            .collect();
        print_table(&["Type", "ID", "Via"], rows);
    }

    Ok(())
}
