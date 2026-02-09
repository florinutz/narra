//! Session management command handlers: context, pin, unpin.

use anyhow::Result;
use serde::Serialize;

use crate::cli::output::{
    output_json, print_header, print_hint, print_kv, print_success, print_table, OutputMode,
};
use crate::cli::resolve::resolve_single;
use crate::init::AppContext;
use crate::session::generate_startup_context;

pub async fn handle_context(ctx: &AppContext, mode: OutputMode) -> Result<()> {
    let info = generate_startup_context(&ctx.session_manager, &ctx.db)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to generate session context: {}", e))?;

    if mode == OutputMode::Json {
        output_json(&info);
        return Ok(());
    }

    print_header(&format!("Session Context ({})", info.verbosity));
    if let Some(ago) = &info.last_session_ago {
        print_kv("Last session", ago);
    }
    println!("  {}", info.summary);

    // Hot entities
    if !info.hot_entities.is_empty() {
        println!();
        println!("Hot Entities:");
        let rows: Vec<Vec<String>> = info
            .hot_entities
            .iter()
            .map(|e| vec![e.id.clone(), e.name.clone(), e.entity_type.clone()])
            .collect();
        print_table(&["ID", "Name", "Type"], rows);
    }

    // Pinned entities
    let pinned = ctx.session_manager.get_pinned().await;
    if !pinned.is_empty() {
        println!();
        println!("Pinned Entities:");
        for id in &pinned {
            println!("  - {}", id);
        }
    }

    // Pending decisions
    if !info.pending_decisions.is_empty() {
        println!();
        println!("Pending Decisions:");
        let rows: Vec<Vec<String>> = info
            .pending_decisions
            .iter()
            .map(|d| {
                vec![
                    d.id.clone(),
                    d.description.clone(),
                    d.age.clone(),
                    format!("{}", d.affected_count),
                ]
            })
            .collect();
        print_table(&["ID", "Description", "Age", "Affected"], rows);
    }

    // World overview
    if let Some(overview) = &info.world_overview {
        println!();
        print_kv("Characters", &overview.character_count.to_string());
        print_kv("Locations", &overview.location_count.to_string());
        print_kv("Events", &overview.event_count.to_string());
        print_kv("Scenes", &overview.scene_count.to_string());
        print_kv("Relationships", &overview.relationship_count.to_string());
    }

    Ok(())
}

pub async fn handle_pin(
    ctx: &AppContext,
    entity: &str,
    mode: OutputMode,
    no_semantic: bool,
) -> Result<()> {
    let entity_id = resolve_single(ctx, entity, no_semantic).await?;
    ctx.session_manager.pin_entity(&entity_id).await;
    ctx.session_manager
        .save()
        .await
        .map_err(|e| anyhow::anyhow!("Failed to save session: {}", e))?;

    let pinned = ctx.session_manager.get_pinned().await;

    if mode == OutputMode::Json {
        #[derive(Serialize)]
        struct PinResult {
            status: String,
            entity_id: String,
            total_pinned: usize,
        }
        output_json(&PinResult {
            status: "pinned".to_string(),
            entity_id: entity_id.clone(),
            total_pinned: pinned.len(),
        });
    } else {
        print_success(&format!(
            "Pinned '{}' ({} total pinned)",
            entity_id,
            pinned.len()
        ));
    }

    Ok(())
}

pub async fn handle_unpin(
    ctx: &AppContext,
    entity: &str,
    mode: OutputMode,
    no_semantic: bool,
) -> Result<()> {
    let entity_id = resolve_single(ctx, entity, no_semantic).await?;
    ctx.session_manager.unpin_entity(&entity_id).await;
    ctx.session_manager
        .save()
        .await
        .map_err(|e| anyhow::anyhow!("Failed to save session: {}", e))?;

    let pinned = ctx.session_manager.get_pinned().await;

    if mode == OutputMode::Json {
        #[derive(Serialize)]
        struct UnpinResult {
            status: String,
            entity_id: String,
            total_pinned: usize,
        }
        output_json(&UnpinResult {
            status: "unpinned".to_string(),
            entity_id: entity_id.clone(),
            total_pinned: pinned.len(),
        });
    } else {
        print_success(&format!(
            "Unpinned '{}' ({} remaining)",
            entity_id,
            pinned.len()
        ));
        if !pinned.is_empty() {
            print_hint(&format!("Still pinned: {}", pinned.join(", ")));
        }
    }

    Ok(())
}
