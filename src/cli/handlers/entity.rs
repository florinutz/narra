//! CRUD handlers for character, location, event, and scene entities.

use anyhow::Result;

use crate::cli::output::{
    output_json, output_json_list, print_error, print_success, print_table, OutputMode,
};
use crate::init::AppContext;
use crate::models::{CharacterCreate, EventCreate, LocationCreate, SceneCreate};
use crate::repository::EntityRepository;

// =============================================================================
// ID helpers
// =============================================================================

/// Strip a known table prefix from an entity ID, returning the bare key.
/// e.g. "character:alice" -> "alice", "alice" -> "alice"
fn bare_key(id: &str, prefix: &str) -> String {
    id.strip_prefix(&format!("{}:", prefix))
        .unwrap_or(id)
        .to_string()
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

pub async fn get_character(ctx: &AppContext, id: &str, mode: OutputMode) -> Result<()> {
    let key = bare_key(id, "character");
    let character = ctx.entity_repo.get_character(&key).await?;

    match character {
        Some(c) => {
            if mode == OutputMode::Json {
                output_json(&c);
            } else {
                output_json(&c); // Pretty-printed JSON for human mode too
            }
        }
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
        // Parse date string to SurrealDB Datetime
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
