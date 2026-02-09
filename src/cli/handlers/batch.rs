//! Batch create handler — YAML from stdin or file.

use anyhow::Result;
use colored::Colorize;

use crate::cli::output::{create_spinner, output_json, print_error, print_success, OutputMode};
use crate::init::AppContext;
use crate::mcp::types::{CharacterSpec, EventSpec, LocationSpec, RelationshipSpec};
use crate::models::{CharacterCreate, EventCreate, LocationCreate, RelationshipCreate};

pub async fn handle_batch_create(
    ctx: &AppContext,
    entity_type: &str,
    file: Option<&str>,
    mode: OutputMode,
) -> Result<()> {
    let yaml_content = if let Some(path) = file {
        std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("Failed to read file '{}': {}", path, e))?
    } else {
        use std::io::Read;
        let mut buf = String::new();
        std::io::stdin()
            .read_to_string(&mut buf)
            .map_err(|e| anyhow::anyhow!("Failed to read stdin: {}", e))?;
        buf
    };

    match entity_type.to_lowercase().as_str() {
        "character" | "characters" => batch_characters(ctx, &yaml_content, mode).await,
        "location" | "locations" => batch_locations(ctx, &yaml_content, mode).await,
        "event" | "events" => batch_events(ctx, &yaml_content, mode).await,
        "relationship" | "relationships" => batch_relationships(ctx, &yaml_content, mode).await,
        _ => anyhow::bail!(
            "Unknown entity type '{}'. Valid types: character, location, event, relationship",
            entity_type
        ),
    }
}

async fn batch_characters(ctx: &AppContext, yaml: &str, mode: OutputMode) -> Result<()> {
    use crate::models::character::{create_character, create_character_with_id};

    let specs: Vec<CharacterSpec> =
        serde_yaml_ng::from_str(yaml).map_err(|e| anyhow::anyhow!("Invalid YAML: {}", e))?;

    let total = specs.len();
    let spinner = create_spinner(&format!("Creating {} characters...", total));

    let mut created = Vec::new();
    let mut errors = Vec::new();

    for spec in specs {
        let data = CharacterCreate {
            name: spec.name.clone(),
            aliases: spec.aliases.unwrap_or_default(),
            roles: spec.role.map(|r| vec![r]).unwrap_or_default(),
            profile: spec.profile.unwrap_or_default(),
        };

        let result = if let Some(ref id) = spec.id {
            create_character_with_id(&ctx.db, id, data).await
        } else {
            create_character(&ctx.db, data).await
        };

        match result {
            Ok(character) => {
                let entity_id = character.id.to_string();
                ctx.staleness_manager.spawn_regeneration(
                    entity_id.clone(),
                    "character".to_string(),
                    None,
                );
                created.push((entity_id, character.name));
            }
            Err(e) => errors.push(format!("'{}': {}", spec.name, e)),
        }
    }

    spinner.finish_and_clear();
    print_batch_summary("character", total, &created, &errors, mode);
    Ok(())
}

async fn batch_locations(ctx: &AppContext, yaml: &str, mode: OutputMode) -> Result<()> {
    use crate::models::location::{create_location, create_location_with_id};
    use surrealdb::RecordId;

    let specs: Vec<LocationSpec> =
        serde_yaml_ng::from_str(yaml).map_err(|e| anyhow::anyhow!("Invalid YAML: {}", e))?;

    let total = specs.len();
    let spinner = create_spinner(&format!("Creating {} locations...", total));

    let mut created = Vec::new();
    let mut errors = Vec::new();

    for spec in specs {
        let parent_record_id = match spec.parent_id.as_ref() {
            Some(id) => match id.parse::<RecordId>() {
                Ok(rid) => Some(rid),
                Err(e) => {
                    errors.push(format!(
                        "Invalid parent_id '{}' for '{}': {}",
                        id, spec.name, e
                    ));
                    continue;
                }
            },
            None => None,
        };

        let data = LocationCreate {
            name: spec.name.clone(),
            description: spec.description,
            loc_type: spec.loc_type.unwrap_or_else(|| "place".to_string()),
            parent: parent_record_id,
        };

        let result = if let Some(ref id) = spec.id {
            create_location_with_id(&ctx.db, id, data).await
        } else {
            create_location(&ctx.db, data).await
        };

        match result {
            Ok(location) => {
                let entity_id = location.id.to_string();
                ctx.staleness_manager.spawn_regeneration(
                    entity_id.clone(),
                    "location".to_string(),
                    None,
                );
                created.push((entity_id, location.name));
            }
            Err(e) => errors.push(format!("'{}': {}", spec.name, e)),
        }
    }

    spinner.finish_and_clear();
    print_batch_summary("location", total, &created, &errors, mode);
    Ok(())
}

async fn batch_events(ctx: &AppContext, yaml: &str, mode: OutputMode) -> Result<()> {
    use crate::models::event::{create_event, create_event_with_id};

    let specs: Vec<EventSpec> =
        serde_yaml_ng::from_str(yaml).map_err(|e| anyhow::anyhow!("Invalid YAML: {}", e))?;

    let total = specs.len();
    let spinner = create_spinner(&format!("Creating {} events...", total));

    let mut created = Vec::new();
    let mut errors = Vec::new();

    for spec in specs {
        let parsed_date = match spec.date.as_ref() {
            Some(d) => match chrono::DateTime::parse_from_rfc3339(d) {
                Ok(dt) => Some(dt.with_timezone(&chrono::Utc).into()),
                Err(e) => {
                    errors.push(format!("Invalid date '{}' for '{}': {}", d, spec.title, e));
                    continue;
                }
            },
            None => None,
        };

        let data = EventCreate {
            title: spec.title.clone(),
            description: spec.description,
            sequence: spec.sequence.unwrap_or(0) as i64,
            date: parsed_date,
            date_precision: spec.date_precision,
            duration_end: None,
        };

        let result = if let Some(ref id) = spec.id {
            create_event_with_id(&ctx.db, id, data).await
        } else {
            create_event(&ctx.db, data).await
        };

        match result {
            Ok(event) => {
                let entity_id = event.id.to_string();
                ctx.staleness_manager.spawn_regeneration(
                    entity_id.clone(),
                    "event".to_string(),
                    None,
                );
                created.push((entity_id, event.title));
            }
            Err(e) => errors.push(format!("'{}': {}", spec.title, e)),
        }
    }

    spinner.finish_and_clear();
    print_batch_summary("event", total, &created, &errors, mode);
    Ok(())
}

async fn batch_relationships(ctx: &AppContext, yaml: &str, mode: OutputMode) -> Result<()> {
    use crate::models::relationship::create_relationship;

    let specs: Vec<RelationshipSpec> =
        serde_yaml_ng::from_str(yaml).map_err(|e| anyhow::anyhow!("Invalid YAML: {}", e))?;

    let total = specs.len();
    let spinner = create_spinner(&format!("Creating {} relationships...", total));

    let mut created = Vec::new();
    let mut errors = Vec::new();

    for spec in specs {
        let data = RelationshipCreate {
            from_character_id: spec.from_character_id.clone(),
            to_character_id: spec.to_character_id.clone(),
            rel_type: spec.rel_type.clone(),
            subtype: spec.subtype,
            label: spec.label,
        };

        match create_relationship(&ctx.db, data).await {
            Ok(rel) => {
                let entity_id = rel.id.to_string();
                ctx.staleness_manager.spawn_regeneration(
                    entity_id.clone(),
                    "relates_to".to_string(),
                    None,
                );
                created.push((
                    entity_id,
                    format!("{} -> {}", spec.from_character_id, spec.to_character_id),
                ));
            }
            Err(e) => errors.push(format!(
                "{} -> {}: {}",
                spec.from_character_id, spec.to_character_id, e
            )),
        }
    }

    spinner.finish_and_clear();
    print_batch_summary("relationship", total, &created, &errors, mode);
    Ok(())
}

fn print_batch_summary(
    entity_type: &str,
    total: usize,
    created: &[(String, String)],
    errors: &[String],
    mode: OutputMode,
) {
    if mode == OutputMode::Json {
        let created_json: Vec<serde_json::Value> = created
            .iter()
            .map(|(id, name)| serde_json::json!({"id": id, "name": name}))
            .collect();
        output_json(&serde_json::json!({
            "entity_type": entity_type,
            "total": total,
            "created": created.len(),
            "errors": errors,
            "entities": created_json,
        }));
        return;
    }

    if !created.is_empty() {
        print_success(&format!(
            "Created {}/{} {}s",
            created.len(),
            total,
            entity_type
        ));
        for (id, name) in created {
            println!("  {} {}", id.dimmed(), name);
        }
    }

    if !errors.is_empty() {
        println!();
        for err in errors {
            print_error(err);
        }
    }

    if created.is_empty() && errors.is_empty() {
        println!("No entities to create (empty input).");
    }
}
