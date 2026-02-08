//! Find handler for CLI — hybrid search by default.

use anyhow::Result;

use crate::cli::output::{output_json_list, print_hint, print_table, OutputMode};
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

#[allow(clippy::too_many_arguments)]
pub async fn handle_find(
    ctx: &AppContext,
    query: &str,
    keyword_only: bool,
    semantic_only: bool,
    no_semantic: bool,
    entity_type: Option<&str>,
    limit: usize,
    mode: OutputMode,
) -> Result<()> {
    let entity_types = match entity_type {
        Some(t) => {
            if let Some(et) = parse_entity_type(t) {
                vec![et]
            } else {
                anyhow::bail!(
                    "Unknown entity type '{}'. Valid types: character, location, event, scene, knowledge, note",
                    t
                );
            }
        }
        None => vec![],
    };

    let filter = SearchFilter {
        entity_types,
        limit: Some(limit),
        ..Default::default()
    };

    // Determine search strategy: hybrid by default, with graceful fallback
    let (results, search_mode) = if keyword_only || no_semantic {
        let r = ctx.search_service.search(query, filter).await?;
        (r, "keyword")
    } else if semantic_only {
        let r = ctx.search_service.semantic_search(query, filter).await?;
        (r, "semantic")
    } else {
        // Default: hybrid (auto-falls back to keyword if embeddings unavailable)
        let has_embeddings = ctx.embedding_service.is_available();
        let r = ctx.search_service.hybrid_search(query, filter).await?;
        let mode_label = if has_embeddings {
            "hybrid"
        } else {
            "keyword (semantic unavailable)"
        };
        (r, mode_label)
    };

    if mode == OutputMode::Json {
        output_json_list(&results);
        return Ok(());
    }

    println!(
        "Search ({}) for '{}': {} results\n",
        search_mode,
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

    if results.is_empty() {
        print_hint("Try broadening your query or removing the --type filter.");
    }

    Ok(())
}
