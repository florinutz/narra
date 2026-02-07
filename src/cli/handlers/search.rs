//! Search handler for CLI.

use anyhow::Result;

use crate::cli::output::{output_json_list, print_table, OutputMode};
use crate::init::AppContext;
use crate::services::{EntityType, SearchFilter};

/// Parse an entity type string into an EntityType.
fn parse_entity_type(s: &str) -> Option<EntityType> {
    match s.to_lowercase().as_str() {
        "character" | "characters" => Some(EntityType::Character),
        "location" | "locations" => Some(EntityType::Location),
        "event" | "events" => Some(EntityType::Event),
        "scene" | "scenes" => Some(EntityType::Scene),
        "knowledge" => Some(EntityType::Knowledge),
        "note" | "notes" => Some(EntityType::Note),
        _ => None,
    }
}

pub async fn handle_search(
    ctx: &AppContext,
    query: &str,
    semantic: bool,
    hybrid: bool,
    entity_type: Option<&str>,
    limit: usize,
    mode: OutputMode,
) -> Result<()> {
    let entity_types = match entity_type {
        Some(t) => {
            if let Some(et) = parse_entity_type(t) {
                vec![et]
            } else {
                anyhow::bail!("Unknown entity type '{}'. Valid types: character, location, event, scene, knowledge, note", t);
            }
        }
        None => vec![],
    };

    let filter = SearchFilter {
        entity_types,
        limit: Some(limit),
        ..Default::default()
    };

    let results = if hybrid {
        ctx.search_service.hybrid_search(query, filter).await?
    } else if semantic {
        ctx.search_service.semantic_search(query, filter).await?
    } else {
        ctx.search_service.search(query, filter).await?
    };

    if mode == OutputMode::Json {
        output_json_list(&results);
        return Ok(());
    }

    let search_type = if hybrid {
        "hybrid"
    } else if semantic {
        "semantic"
    } else {
        "keyword"
    };

    println!(
        "Search ({}) for '{}': {} results\n",
        search_type,
        query,
        results.len()
    );

    let rows: Vec<Vec<String>> = results
        .iter()
        .map(|r| {
            vec![
                r.id.clone(),
                r.entity_type.clone(),
                r.name.clone(),
                format!("{:.4}", r.score),
            ]
        })
        .collect();

    print_table(&["ID", "Type", "Name", "Score"], rows);
    Ok(())
}
