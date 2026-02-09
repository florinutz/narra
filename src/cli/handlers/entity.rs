//! CRUD handlers for character, location, event, and scene entities.

use anyhow::Result;

use crate::cli::output::{
    output_json, output_json_list, print_error, print_hint, print_success, print_table, OutputMode,
};
use crate::cli::resolve::{
    bare_key, entity_type_from_id, resolve_by_name, resolve_single, ResolutionMethod,
};
use crate::init::AppContext;
use crate::models::{CharacterCreate, EventCreate, LocationCreate, SceneCreate};
use crate::repository::EntityRepository;
use crate::services::SearchFilter;

// =============================================================================
// Unified Get — resolve by name or type:id
// =============================================================================

pub async fn handle_get(
    ctx: &AppContext,
    input: &str,
    mode: OutputMode,
    no_semantic: bool,
) -> Result<()> {
    if let Some(entity_type) = entity_type_from_id(input) {
        // Explicit type:id
        let key = bare_key(input, entity_type);
        dispatch_get(ctx, entity_type, &key, mode).await?;
        // Show similar entities if semantic available
        if !no_semantic && ctx.embedding_service.is_available() {
            show_similar_entities(ctx, input, mode).await;
        }
        return Ok(());
    }

    // Try name resolution with fuzzy/semantic fallback
    let search_svc = if no_semantic {
        None
    } else {
        Some(ctx.search_service.as_ref())
    };
    let matches = resolve_by_name(&ctx.db, input, search_svc).await?;

    match matches.len() {
        0 => {
            print_error(&format!("No entity named '{}'", input));
            print_hint(&format!("Try: narra find {}", input));
        }
        1 => {
            let m = &matches[0];
            // Indicate resolution method if not exact
            match &m.method {
                ResolutionMethod::Fuzzy { score } => {
                    print_hint(&format!(
                        "Fuzzy match: '{}' -> {} ({}) [score: {}%]",
                        input, m.id, m.name, score
                    ));
                }
                ResolutionMethod::Semantic => {
                    print_hint(&format!(
                        "Semantic match: '{}' -> {} ({})",
                        input, m.id, m.name
                    ));
                }
                ResolutionMethod::Exact => {}
            }
            let key = bare_key(&m.id, &m.entity_type);
            dispatch_get(ctx, &m.entity_type, &key, mode).await?;
            // Show similar entities
            if !no_semantic && ctx.embedding_service.is_available() {
                show_similar_entities(ctx, &m.id, mode).await;
            }
        }
        _ => {
            // Check if all non-exact (fuzzy/semantic suggestions)
            let any_exact = matches.iter().any(|m| m.method == ResolutionMethod::Exact);
            if any_exact {
                print_error(&format!("Multiple matches for '{}'. Use a full ID:", input));
            } else {
                print_hint(&format!("No exact match for '{}'. Did you mean:", input));
            }
            for m in &matches {
                let method_label = match &m.method {
                    ResolutionMethod::Exact => "".to_string(),
                    ResolutionMethod::Fuzzy { score } => format!(" [fuzzy: {}%]", score),
                    ResolutionMethod::Semantic => " [semantic]".to_string(),
                };
                println!("  {} ({}){}", m.id, m.name, method_label);
            }
        }
    }

    Ok(())
}

/// Show similar entities for a given entity ID using semantic search.
async fn show_similar_entities(ctx: &AppContext, entity_id: &str, mode: OutputMode) {
    if mode == OutputMode::Json {
        return; // Don't append human text to JSON output
    }

    // Get entity name to use as search query
    let entity_type = entity_type_from_id(entity_id).unwrap_or("character");
    let filter = SearchFilter {
        limit: Some(6), // Fetch one extra to filter out self
        ..Default::default()
    };

    let entity_name = get_entity_display_name(ctx, entity_id).await;
    let query = entity_name.as_deref().unwrap_or(entity_id);

    if let Ok(results) = ctx.search_service.semantic_search(query, filter).await {
        let similar: Vec<_> = results
            .into_iter()
            .filter(|r| r.id != entity_id && r.entity_type == entity_type)
            .take(5)
            .collect();
        if !similar.is_empty() {
            println!();
            crate::cli::output::print_section("Similar Entities", "");
            let rows: Vec<Vec<String>> = similar
                .iter()
                .map(|r| vec![r.id.clone(), format!("{:.4}", r.score), r.name.clone()])
                .collect();
            print_table(&["ID", "Score", "Name"], rows);
        }
    }
}

/// Get display name for an entity by looking it up.
async fn get_entity_display_name(ctx: &AppContext, entity_id: &str) -> Option<String> {
    let entity_type = entity_type_from_id(entity_id)?;
    let key = bare_key(entity_id, entity_type);
    match entity_type {
        "character" => ctx
            .entity_repo
            .get_character(&key)
            .await
            .ok()
            .flatten()
            .map(|c| c.name),
        "location" => ctx
            .entity_repo
            .get_location(&key)
            .await
            .ok()
            .flatten()
            .map(|l| l.name),
        "event" => ctx
            .entity_repo
            .get_event(&key)
            .await
            .ok()
            .flatten()
            .map(|e| e.title),
        "scene" => ctx
            .entity_repo
            .get_scene(&key)
            .await
            .ok()
            .flatten()
            .map(|s| s.title),
        _ => None,
    }
}

async fn dispatch_get(
    ctx: &AppContext,
    entity_type: &str,
    key: &str,
    mode: OutputMode,
) -> Result<()> {
    match entity_type {
        "character" => get_character(ctx, key, mode).await,
        "location" => get_location(ctx, key, mode).await,
        "event" => get_event(ctx, key, mode).await,
        "scene" => get_scene(ctx, key, mode).await,
        "universe_fact" => crate::cli::handlers::fact::get_fact(ctx, key, mode).await,
        "note" => {
            let notes = crate::models::note::get_note(&ctx.db, key).await?;
            match notes {
                Some(n) => output_json(&n),
                None => print_error(&format!("Note '{}' not found", key)),
            }
            Ok(())
        }
        other => {
            anyhow::bail!(
                "Unsupported entity type '{}'. Supported: character, location, event, scene, universe_fact, note",
                other
            );
        }
    }
}

// =============================================================================
// Unified List — any entity type by string
// =============================================================================

/// Normalize entity type string, accepting plural forms.
fn normalize_type(s: &str) -> String {
    match s.to_lowercase().as_str() {
        "character" | "characters" => "character".to_string(),
        "location" | "locations" => "location".to_string(),
        "event" | "events" => "event".to_string(),
        "scene" | "scenes" => "scene".to_string(),
        "knowledge" => "knowledge".to_string(),
        "relationship" | "relationships" => "relationship".to_string(),
        "fact" | "facts" => "fact".to_string(),
        "note" | "notes" => "note".to_string(),
        _ => s.to_string(),
    }
}

#[allow(clippy::too_many_arguments)]
pub async fn handle_list(
    ctx: &AppContext,
    entity_type_str: &str,
    character_filter: Option<&str>,
    category_filter: Option<&str>,
    enforcement_filter: Option<&str>,
    entity_filter: Option<&str>,
    limit: usize,
    mode: OutputMode,
) -> Result<()> {
    let entity_type = normalize_type(entity_type_str);

    match entity_type.as_str() {
        "character" => list_characters(ctx, mode).await,
        "location" => list_locations(ctx, mode).await,
        "event" => list_events(ctx, mode).await,
        "scene" => list_scenes(ctx, mode).await,
        "knowledge" => {
            crate::cli::handlers::knowledge::list_knowledge(ctx, character_filter, mode).await
        }
        "relationship" => {
            crate::cli::handlers::relationship::list_relationships(ctx, character_filter, mode)
                .await
        }
        "fact" => {
            crate::cli::handlers::fact::list_facts(
                ctx,
                entity_filter,
                category_filter,
                enforcement_filter,
                mode,
            )
            .await
        }
        "note" => crate::cli::handlers::note::list_notes(ctx, entity_filter, mode).await,
        other => {
            anyhow::bail!(
                "Unknown entity type '{}'. Valid types: character, location, event, scene, knowledge, relationship, fact, note (limit: {})",
                other,
                limit
            );
        }
    }
}

// =============================================================================
// Characters
// =============================================================================

pub async fn list_characters(ctx: &AppContext, mode: OutputMode) -> Result<()> {
    let chars = ctx.entity_repo.list_characters().await?;

    if mode == OutputMode::Json {
        output_json_list(&chars);
        return Ok(());
    }

    let rows: Vec<Vec<String>> = chars
        .iter()
        .map(|c| {
            vec![
                c.id.to_string(),
                c.name.clone(),
                c.roles.join(", "),
                c.aliases.join(", "),
            ]
        })
        .collect();

    print_table(&["ID", "Name", "Roles", "Aliases"], rows);
    Ok(())
}

pub async fn get_character(ctx: &AppContext, id: &str, _mode: OutputMode) -> Result<()> {
    let key = bare_key(id, "character");
    let character = ctx.entity_repo.get_character(&key).await?;

    match character {
        Some(c) => output_json(&c),
        None => print_error(&format!("Character '{}' not found", id)),
    }
    Ok(())
}

pub async fn create_character(
    ctx: &AppContext,
    name: &str,
    role: Option<&str>,
    description: Option<&str>,
    aliases: &[String],
    profile_json: Option<&str>,
    mode: OutputMode,
) -> Result<()> {
    let mut roles = Vec::new();
    if let Some(r) = role {
        roles.push(r.to_string());
    }

    let mut profile: std::collections::HashMap<String, Vec<String>> = match profile_json {
        Some(json) => serde_json::from_str(json)
            .map_err(|e| anyhow::anyhow!("Invalid profile JSON: {}", e))?,
        None => std::collections::HashMap::new(),
    };

    if let Some(desc) = description {
        profile
            .entry("description".to_string())
            .or_default()
            .push(desc.to_string());
    }

    let data = CharacterCreate {
        name: name.to_string(),
        aliases: aliases.to_vec(),
        roles,
        profile,
    };

    let character = ctx.entity_repo.create_character(data).await?;

    if mode == OutputMode::Json {
        output_json(&character);
    } else {
        print_success(&format!(
            "Created character '{}' ({})",
            character.name, character.id
        ));
    }
    Ok(())
}

// =============================================================================
// Locations
// =============================================================================

pub async fn list_locations(ctx: &AppContext, mode: OutputMode) -> Result<()> {
    let locs = ctx.entity_repo.list_locations().await?;

    if mode == OutputMode::Json {
        output_json_list(&locs);
        return Ok(());
    }

    let rows: Vec<Vec<String>> = locs
        .iter()
        .map(|l| {
            vec![
                l.id.to_string(),
                l.name.clone(),
                l.loc_type.clone(),
                l.description.clone().unwrap_or_default(),
            ]
        })
        .collect();

    print_table(&["ID", "Name", "Type", "Description"], rows);
    Ok(())
}

pub async fn get_location(ctx: &AppContext, id: &str, _mode: OutputMode) -> Result<()> {
    let key = bare_key(id, "location");
    let location = ctx.entity_repo.get_location(&key).await?;

    match location {
        Some(l) => {
            output_json(&l);
        }
        None => print_error(&format!("Location '{}' not found", id)),
    }
    Ok(())
}

pub async fn create_location(
    ctx: &AppContext,
    name: &str,
    description: Option<&str>,
    parent: Option<&str>,
    loc_type: Option<&str>,
    mode: OutputMode,
) -> Result<()> {
    let parent_id = parent.map(|p| {
        let key = bare_key(p, "location");
        surrealdb::RecordId::from(("location", key.as_str()))
    });

    let data = LocationCreate {
        name: name.to_string(),
        description: description.map(|s| s.to_string()),
        loc_type: loc_type.unwrap_or("general").to_string(),
        parent: parent_id,
    };

    let location = ctx.entity_repo.create_location(data).await?;

    if mode == OutputMode::Json {
        output_json(&location);
    } else {
        print_success(&format!(
            "Created location '{}' ({})",
            location.name, location.id
        ));
    }
    Ok(())
}

// =============================================================================
// Events
// =============================================================================

pub async fn list_events(ctx: &AppContext, mode: OutputMode) -> Result<()> {
    let events = ctx.entity_repo.list_events().await?;

    if mode == OutputMode::Json {
        output_json_list(&events);
        return Ok(());
    }

    let rows: Vec<Vec<String>> = events
        .iter()
        .map(|e| {
            vec![
                e.id.to_string(),
                e.title.clone(),
                e.sequence.to_string(),
                e.description.clone().unwrap_or_default(),
            ]
        })
        .collect();

    print_table(&["ID", "Title", "Seq", "Description"], rows);
    Ok(())
}

pub async fn get_event(ctx: &AppContext, id: &str, _mode: OutputMode) -> Result<()> {
    let key = bare_key(id, "event");
    let event = ctx.entity_repo.get_event(&key).await?;

    match event {
        Some(e) => {
            output_json(&e);
        }
        None => print_error(&format!("Event '{}' not found", id)),
    }
    Ok(())
}

pub async fn create_event(
    ctx: &AppContext,
    title: &str,
    description: Option<&str>,
    sequence: Option<i32>,
    date: Option<&str>,
    mode: OutputMode,
) -> Result<()> {
    // If no sequence provided, get the next one
    let seq = match sequence {
        Some(s) => s as i64,
        None => crate::models::event::get_next_sequence(&ctx.db).await?,
    };

    let date_val = date.map(|d| {
        let dt = chrono::NaiveDate::parse_from_str(d, "%Y-%m-%d")
            .or_else(|_| chrono::NaiveDate::parse_from_str(d, "%Y-%m-%dT%H:%M:%S"))
            .unwrap_or_else(|_| chrono::Utc::now().date_naive());
        let datetime = dt.and_hms_opt(0, 0, 0).unwrap();
        let utc = chrono::DateTime::<chrono::Utc>::from_naive_utc_and_offset(datetime, chrono::Utc);
        surrealdb::Datetime::from(utc)
    });

    let date_precision = date.map(|_| "day".to_string());

    let data = EventCreate {
        title: title.to_string(),
        description: description.map(|s| s.to_string()),
        sequence: seq,
        date: date_val,
        date_precision,
        duration_end: None,
    };

    let event = ctx.entity_repo.create_event(data).await?;

    if mode == OutputMode::Json {
        output_json(&event);
    } else {
        print_success(&format!(
            "Created event '{}' (seq={}, {})",
            event.title, event.sequence, event.id
        ));
    }
    Ok(())
}

// =============================================================================
// Scenes
// =============================================================================

pub async fn list_scenes(ctx: &AppContext, mode: OutputMode) -> Result<()> {
    let scenes = ctx.entity_repo.list_scenes().await?;

    if mode == OutputMode::Json {
        output_json_list(&scenes);
        return Ok(());
    }

    let rows: Vec<Vec<String>> = scenes
        .iter()
        .map(|s| {
            vec![
                s.id.to_string(),
                s.title.clone(),
                s.event.to_string(),
                s.primary_location.to_string(),
            ]
        })
        .collect();

    print_table(&["ID", "Title", "Event", "Location"], rows);
    Ok(())
}

pub async fn get_scene(ctx: &AppContext, id: &str, _mode: OutputMode) -> Result<()> {
    let key = bare_key(id, "scene");
    let scene = ctx.entity_repo.get_scene(&key).await?;

    match scene {
        Some(s) => {
            output_json(&s);
        }
        None => print_error(&format!("Scene '{}' not found", id)),
    }
    Ok(())
}

pub async fn create_scene(
    ctx: &AppContext,
    title: &str,
    event_id: &str,
    location_id: &str,
    summary: Option<&str>,
    mode: OutputMode,
) -> Result<()> {
    let event_key = bare_key(event_id, "event");
    let location_key = bare_key(location_id, "location");

    let data = SceneCreate {
        title: title.to_string(),
        summary: summary.map(|s| s.to_string()),
        event: surrealdb::RecordId::from(("event", event_key.as_str())),
        primary_location: surrealdb::RecordId::from(("location", location_key.as_str())),
        secondary_locations: vec![],
    };

    let scene = ctx.entity_repo.create_scene(data).await?;

    if mode == OutputMode::Json {
        output_json(&scene);
    } else {
        print_success(&format!("Created scene '{}' ({})", scene.title, scene.id));
    }
    Ok(())
}

// =============================================================================
// Protect / Unprotect
// =============================================================================

pub async fn handle_protect(
    ctx: &AppContext,
    entity: &str,
    mode: OutputMode,
    no_semantic: bool,
) -> Result<()> {
    let entity_id = resolve_single(ctx, entity, no_semantic).await?;
    ctx.impact_service.protect_entity(&entity_id).await;

    if mode == OutputMode::Json {
        output_json(&serde_json::json!({
            "status": "protected",
            "entity_id": entity_id,
        }));
    } else {
        print_success(&format!("Protected entity '{}'", entity_id));
        print_hint("Protection is in-memory only and will be lost on restart.");
    }
    Ok(())
}

pub async fn handle_unprotect(
    ctx: &AppContext,
    entity: &str,
    mode: OutputMode,
    no_semantic: bool,
) -> Result<()> {
    let entity_id = resolve_single(ctx, entity, no_semantic).await?;
    ctx.impact_service.unprotect_entity(&entity_id).await;

    if mode == OutputMode::Json {
        output_json(&serde_json::json!({
            "status": "unprotected",
            "entity_id": entity_id,
        }));
    } else {
        print_success(&format!("Removed protection from '{}'", entity_id));
    }
    Ok(())
}
