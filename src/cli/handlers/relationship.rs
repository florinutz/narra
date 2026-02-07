//! Relationship command handlers for CLI.

use anyhow::Result;

use crate::cli::output::{output_json, output_json_list, print_success, print_table, OutputMode};
use crate::init::AppContext;
use crate::models::RelationshipCreate;
use crate::repository::RelationshipRepository;

/// Strip a known table prefix from an entity ID, returning the bare key.
fn bare_key(id: &str, prefix: &str) -> String {
    id.strip_prefix(&format!("{}:", prefix))
        .unwrap_or(id)
        .to_string()
}

pub async fn list_relationships(
    ctx: &AppContext,
    character: Option<&str>,
    mode: OutputMode,
) -> Result<()> {
    match character {
        Some(char_id) => {
            let key = bare_key(char_id, "character");
            let rels = ctx
                .relationship_repo
                .get_character_relationships(&key)
                .await?;

            if mode == OutputMode::Json {
                output_json_list(&rels);
                return Ok(());
            }

            let rows: Vec<Vec<String>> = rels
                .iter()
                .map(|r| {
                    vec![
                        r.id.to_string(),
                        r.from_character.to_string(),
                        r.to_character.to_string(),
                        r.rel_type.clone(),
                        r.subtype.clone().unwrap_or_default(),
                        r.label.clone().unwrap_or_default(),
                    ]
                })
                .collect();

            print_table(&["ID", "From", "To", "Type", "Subtype", "Label"], rows);
        }
        None => {
            if mode == OutputMode::Json {
                println!("[]");
            } else {
                println!("Use --character <id> to list relationships for a specific character.");
            }
        }
    }

    Ok(())
}

pub async fn create_relationship(
    ctx: &AppContext,
    from: &str,
    to: &str,
    rel_type: &str,
    subtype: Option<&str>,
    label: Option<&str>,
    mode: OutputMode,
) -> Result<()> {
    // Repository expects BARE keys (not "character:alice")
    let from_key = bare_key(from, "character");
    let to_key = bare_key(to, "character");

    let data = RelationshipCreate {
        from_character_id: from_key.clone(),
        to_character_id: to_key.clone(),
        rel_type: rel_type.to_string(),
        subtype: subtype.map(|s| s.to_string()),
        label: label.map(|s| s.to_string()),
    };

    let rel = ctx
        .relationship_repo
        .create_relationship(&from_key, &to_key, data)
        .await?;

    if mode == OutputMode::Json {
        output_json(&rel);
    } else {
        print_success(&format!(
            "Created {} relationship: {} -> {} ({})",
            rel.rel_type, rel.from_character, rel.to_character, rel.id,
        ));
    }

    Ok(())
}
