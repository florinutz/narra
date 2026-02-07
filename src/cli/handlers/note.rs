//! Note CRUD handlers for CLI.

use anyhow::Result;

use crate::cli::output::{
    output_json, output_json_list, print_error, print_success, print_table, OutputMode,
};
use crate::init::AppContext;
use crate::models::note;
use crate::models::NoteCreate;

/// Strip a known table prefix from an entity ID, returning the bare key.
fn bare_key(id: &str, prefix: &str) -> String {
    id.strip_prefix(&format!("{}:", prefix))
        .unwrap_or(id)
        .to_string()
}

pub async fn list_notes(ctx: &AppContext, entity: Option<&str>, mode: OutputMode) -> Result<()> {
    let notes = match entity {
        Some(entity_id) => note::get_entity_notes(&ctx.db, entity_id).await?,
        None => note::list_notes(&ctx.db, 100, 0).await?,
    };

    if mode == OutputMode::Json {
        output_json_list(&notes);
        return Ok(());
    }

    let rows: Vec<Vec<String>> = notes
        .iter()
        .map(|n| {
            let body_preview = if n.body.len() > 60 {
                format!("{}...", &n.body[..60])
            } else {
                n.body.clone()
            };
            vec![n.id.to_string(), n.title.clone(), body_preview]
        })
        .collect();

    print_table(&["ID", "Title", "Body"], rows);
    Ok(())
}

pub async fn create_note(
    ctx: &AppContext,
    title: &str,
    body: &str,
    attach_to: &[String],
    mode: OutputMode,
) -> Result<()> {
    let data = NoteCreate {
        title: title.to_string(),
        body: body.to_string(),
    };

    let created = note::create_note(&ctx.db, data).await?;
    let note_key = created.id.key().to_string();

    // Attach to entities if specified
    for entity_id in attach_to {
        match note::attach_note(&ctx.db, &note_key, entity_id).await {
            Ok(_) => {
                if mode != OutputMode::Json {
                    print_success(&format!("Attached to {}", entity_id));
                }
            }
            Err(e) => {
                print_error(&format!("Failed to attach to {}: {}", entity_id, e));
            }
        }
    }

    if mode == OutputMode::Json {
        output_json(&created);
    } else {
        print_success(&format!(
            "Created note '{}' ({})",
            created.title, created.id
        ));
    }
    Ok(())
}

pub async fn attach_note(
    ctx: &AppContext,
    note_id: &str,
    entity_id: &str,
    mode: OutputMode,
) -> Result<()> {
    let key = bare_key(note_id, "note");
    let attachment = note::attach_note(&ctx.db, &key, entity_id).await?;

    if mode == OutputMode::Json {
        output_json(&attachment);
    } else {
        print_success(&format!("Attached note {} to {}", key, entity_id));
    }
    Ok(())
}

pub async fn detach_note(
    ctx: &AppContext,
    note_id: &str,
    entity_id: &str,
    mode: OutputMode,
) -> Result<()> {
    let key = bare_key(note_id, "note");
    note::detach_note(&ctx.db, &key, entity_id).await?;

    if mode == OutputMode::Json {
        println!(
            r#"{{"status": "ok", "note": "{}", "entity": "{}"}}"#,
            key, entity_id
        );
    } else {
        print_success(&format!("Detached note {} from {}", key, entity_id));
    }
    Ok(())
}
