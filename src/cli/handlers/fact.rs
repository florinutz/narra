//! Fact CRUD handlers for CLI.

use anyhow::Result;

use crate::cli::output::{
    output_json, output_json_list, print_error, print_success, print_table, OutputMode,
};
use crate::init::AppContext;
use crate::models::fact;
use crate::models::{EnforcementLevel, FactCategory, FactCreate, FactUpdate};

/// Strip a known table prefix from an entity ID, returning the bare key.
fn bare_key(id: &str, prefix: &str) -> String {
    id.strip_prefix(&format!("{}:", prefix))
        .unwrap_or(id)
        .to_string()
}

fn parse_category(s: &str) -> FactCategory {
    match s.to_lowercase().as_str() {
        "physics_magic" => FactCategory::PhysicsMagic,
        "social_cultural" => FactCategory::SocialCultural,
        "technology" => FactCategory::Technology,
        other => FactCategory::Custom(other.to_string()),
    }
}

fn parse_enforcement(s: &str) -> EnforcementLevel {
    match s.to_lowercase().as_str() {
        "informational" => EnforcementLevel::Informational,
        "warning" => EnforcementLevel::Warning,
        "strict" => EnforcementLevel::Strict,
        _ => EnforcementLevel::Warning,
    }
}

pub async fn list_facts(ctx: &AppContext, mode: OutputMode) -> Result<()> {
    let facts = fact::list_facts(&ctx.db).await?;

    if mode == OutputMode::Json {
        output_json_list(&facts);
        return Ok(());
    }

    let rows: Vec<Vec<String>> = facts
        .iter()
        .map(|f| {
            let cats: Vec<String> = f.categories.iter().map(|c| format!("{:?}", c)).collect();
            vec![
                f.id.to_string(),
                f.title.clone(),
                cats.join(", "),
                format!("{:?}", f.enforcement_level),
            ]
        })
        .collect();

    print_table(&["ID", "Title", "Categories", "Enforcement"], rows);
    Ok(())
}

pub async fn get_fact(ctx: &AppContext, id: &str, _mode: OutputMode) -> Result<()> {
    let key = bare_key(id, "universe_fact");
    let result = fact::get_fact(&ctx.db, &key).await?;

    match result {
        Some(f) => {
            output_json(&f);
        }
        None => print_error(&format!("Fact '{}' not found", id)),
    }
    Ok(())
}

pub async fn create_fact(
    ctx: &AppContext,
    title: &str,
    description: &str,
    categories: &[String],
    enforcement: Option<&str>,
    mode: OutputMode,
) -> Result<()> {
    let cats: Vec<FactCategory> = categories.iter().map(|c| parse_category(c)).collect();
    let enforcement_level = enforcement.map(parse_enforcement).unwrap_or_default();

    let data = FactCreate {
        title: title.to_string(),
        description: description.to_string(),
        categories: cats,
        enforcement_level,
        scope: None,
    };

    let created = fact::create_fact(&ctx.db, data).await?;

    if mode == OutputMode::Json {
        output_json(&created);
    } else {
        print_success(&format!(
            "Created fact '{}' ({})",
            created.title, created.id
        ));
    }
    Ok(())
}

pub async fn update_fact(
    ctx: &AppContext,
    id: &str,
    title: Option<&str>,
    description: Option<&str>,
    categories: &[String],
    enforcement: Option<&str>,
    mode: OutputMode,
) -> Result<()> {
    let key = bare_key(id, "universe_fact");

    let cats = if categories.is_empty() {
        None
    } else {
        Some(categories.iter().map(|c| parse_category(c)).collect())
    };

    let data = FactUpdate {
        title: title.map(|s| s.to_string()),
        description: description.map(|s| s.to_string()),
        categories: cats,
        enforcement_level: enforcement.map(parse_enforcement),
        scope: None,
        updated_at: surrealdb::Datetime::default(),
    };

    let updated = fact::update_fact(&ctx.db, &key, data).await?;

    match updated {
        Some(f) => {
            if mode == OutputMode::Json {
                output_json(&f);
            } else {
                print_success(&format!("Updated fact '{}' ({})", f.title, f.id));
            }
        }
        None => print_error(&format!("Fact '{}' not found", id)),
    }
    Ok(())
}

pub async fn delete_fact(ctx: &AppContext, id: &str, mode: OutputMode) -> Result<()> {
    let key = bare_key(id, "universe_fact");
    let deleted = fact::delete_fact(&ctx.db, &key).await?;

    match deleted {
        Some(f) => {
            if mode == OutputMode::Json {
                output_json(&f);
            } else {
                print_success(&format!("Deleted fact '{}' ({})", f.title, f.id));
            }
        }
        None => print_error(&format!("Fact '{}' not found", id)),
    }
    Ok(())
}

pub async fn link_fact(
    ctx: &AppContext,
    fact_id: &str,
    entity_id: &str,
    mode: OutputMode,
) -> Result<()> {
    let fact_key = bare_key(fact_id, "universe_fact");

    let app = fact::link_fact_to_entity(&ctx.db, &fact_key, entity_id, "manual", None).await?;

    if mode == OutputMode::Json {
        output_json(&app);
    } else {
        print_success(&format!("Linked fact {} to {}", fact_key, entity_id));
    }
    Ok(())
}

pub async fn unlink_fact(
    ctx: &AppContext,
    fact_id: &str,
    entity_id: &str,
    mode: OutputMode,
) -> Result<()> {
    let fact_key = bare_key(fact_id, "universe_fact");

    fact::unlink_fact_from_entity(&ctx.db, &fact_key, entity_id).await?;

    if mode == OutputMode::Json {
        println!(
            r#"{{"status": "ok", "fact": "{}", "entity": "{}"}}"#,
            fact_key, entity_id
        );
    } else {
        print_success(&format!("Unlinked fact {} from {}", fact_key, entity_id));
    }
    Ok(())
}
