//! Utility command handlers: update and delete.

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::cli::output::{print_error, print_success, OutputMode};
use crate::cli::resolve::{bare_key, entity_type_from_id};
use crate::init::AppContext;
use crate::repository::EntityRepository;

// =============================================================================
// Update (with optional --link / --unlink)
// =============================================================================

pub async fn handle_update(
    ctx: &AppContext,
    entity_id: &str,
    fields_json: Option<&str>,
    set_pairs: &[(String, String)],
    link: Option<&str>,
    unlink: Option<&str>,
    mode: OutputMode,
) -> Result<()> {
    let entity_type = entity_type_from_id(entity_id).ok_or_else(|| {
        anyhow::anyhow!(
            "Invalid entity ID '{}'. Expected format: type:key",
            entity_id
        )
    })?;
    let key = bare_key(entity_id, entity_type);

    // Handle field updates if provided
    let has_field_updates = fields_json.is_some() || !set_pairs.is_empty();
    if has_field_updates {
        let fields: serde_json::Value = if let Some(json_str) = fields_json {
            serde_json::from_str(json_str)?
        } else {
            let mut map = serde_json::Map::new();
            for (k, v) in set_pairs {
                let val = serde_json::from_str(v).unwrap_or(serde_json::Value::String(v.clone()));
                map.insert(k.clone(), val);
            }
            serde_json::Value::Object(map)
        };

        let record_ref = surrealdb::RecordId::from((entity_type, key.as_str()));
        let mut response = ctx
            .db
            .query("UPDATE ONLY $ref MERGE $fields RETURN AFTER")
            .bind(("ref", record_ref))
            .bind(("fields", fields))
            .await?;

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
            None => {
                print_error(&format!("Entity '{}' not found", entity_id));
                return Ok(());
            }
        }
    }

    // Handle --link
    if let Some(link_target) = link {
        match entity_type {
            "universe_fact" => {
                crate::models::fact::link_fact_to_entity(
                    &ctx.db,
                    &key,
                    link_target,
                    "manual",
                    None,
                )
                .await?;
                if mode != OutputMode::Json {
                    print_success(&format!("Linked {} to {}", entity_id, link_target));
                }
            }
            "note" => {
                crate::models::note::attach_note(&ctx.db, &key, link_target).await?;
                if mode != OutputMode::Json {
                    print_success(&format!("Attached {} to {}", entity_id, link_target));
                }
            }
            _ => {
                anyhow::bail!(
                    "--link is only supported for universe_fact and note entities, got '{}'",
                    entity_type
                );
            }
        }
    }

    // Handle --unlink
    if let Some(unlink_target) = unlink {
        match entity_type {
            "universe_fact" => {
                crate::models::fact::unlink_fact_from_entity(&ctx.db, &key, unlink_target).await?;
                if mode != OutputMode::Json {
                    print_success(&format!("Unlinked {} from {}", entity_id, unlink_target));
                }
            }
            "note" => {
                crate::models::note::detach_note(&ctx.db, &key, unlink_target).await?;
                if mode != OutputMode::Json {
                    print_success(&format!("Detached {} from {}", entity_id, unlink_target));
                }
            }
            _ => {
                anyhow::bail!(
                    "--unlink is only supported for universe_fact and note entities, got '{}'",
                    entity_type
                );
            }
        }
    }

    // Ensure at least one operation was requested
    if !has_field_updates && link.is_none() && unlink.is_none() {
        anyhow::bail!("Provide --fields, --set, --link, or --unlink to specify update operation");
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
