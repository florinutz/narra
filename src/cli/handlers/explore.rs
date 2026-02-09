//! Deep entity exploration handler.

use anyhow::Result;

use crate::cli::output::{
    output_json, print_error, print_header, print_hint, print_kv, print_section, print_table,
    OutputMode,
};
use crate::cli::resolve::{bare_key, entity_type_from_id, resolve_by_name, ResolutionMethod};
use crate::init::AppContext;
use crate::repository::{EntityRepository, KnowledgeRepository, RelationshipRepository};
use crate::services::{CompositeIntelligenceService, SearchFilter};

#[allow(clippy::too_many_arguments)]
pub async fn handle_explore(
    ctx: &AppContext,
    input: &str,
    no_similar: bool,
    depth: usize,
    mode: OutputMode,
    no_semantic: bool,
) -> Result<()> {
    // Resolve entity
    let (entity_id, entity_type, display_name) = resolve_entity(ctx, input, no_semantic).await?;

    match entity_type.as_str() {
        "character" => {
            explore_character(
                ctx,
                &entity_id,
                &display_name,
                no_similar,
                no_semantic,
                mode,
            )
            .await
        }
        _ => {
            explore_generic(
                ctx,
                &entity_id,
                &entity_type,
                &display_name,
                no_similar,
                no_semantic,
                depth,
                mode,
            )
            .await
        }
    }
}

/// Resolve an input string to an entity ID, type, and display name.
async fn resolve_entity(
    ctx: &AppContext,
    input: &str,
    no_semantic: bool,
) -> Result<(String, String, String)> {
    // If it looks like a full ID (has colon), use directly
    if let Some(entity_type) = entity_type_from_id(input) {
        let key = bare_key(input, entity_type);
        let name = get_display_name(ctx, entity_type, &key)
            .await
            .unwrap_or_else(|| input.to_string());
        return Ok((input.to_string(), entity_type.to_string(), name));
    }

    // Try name resolution
    let search_svc = if no_semantic {
        None
    } else {
        Some(ctx.search_service.as_ref())
    };
    let matches = resolve_by_name(&ctx.db, input, search_svc).await?;

    match matches.len() {
        0 => {
            print_error(&format!("No entity found for '{}'", input));
            print_hint(&format!("Try: narra find {}", input));
            anyhow::bail!("Entity not found: {}", input);
        }
        1 => {
            let m = &matches[0];
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
            Ok((m.id.clone(), m.entity_type.clone(), m.name.clone()))
        }
        _ => {
            print_error(&format!("Multiple matches for '{}'. Use a full ID:", input));
            for m in &matches {
                let method_label = match &m.method {
                    ResolutionMethod::Exact => "".to_string(),
                    ResolutionMethod::Fuzzy { score } => format!(" [fuzzy: {}%]", score),
                    ResolutionMethod::Semantic => " [semantic]".to_string(),
                };
                println!("  {} ({}){}", m.id, m.name, method_label);
            }
            anyhow::bail!("Ambiguous entity: {}", input);
        }
    }
}

/// Get display name for an entity.
async fn get_display_name(ctx: &AppContext, entity_type: &str, key: &str) -> Option<String> {
    match entity_type {
        "character" => ctx
            .entity_repo
            .get_character(key)
            .await
            .ok()
            .flatten()
            .map(|c| c.name),
        "location" => ctx
            .entity_repo
            .get_location(key)
            .await
            .ok()
            .flatten()
            .map(|l| l.name),
        "event" => ctx
            .entity_repo
            .get_event(key)
            .await
            .ok()
            .flatten()
            .map(|e| e.title),
        "scene" => ctx
            .entity_repo
            .get_scene(key)
            .await
            .ok()
            .flatten()
            .map(|s| s.title),
        _ => None,
    }
}

/// Deep exploration for a character entity.
async fn explore_character(
    ctx: &AppContext,
    entity_id: &str,
    display_name: &str,
    no_similar: bool,
    no_semantic: bool,
    mode: OutputMode,
) -> Result<()> {
    // Fetch dossier
    let service = CompositeIntelligenceService::new(ctx.db.clone());
    let dossier = service
        .character_dossier(entity_id)
        .await
        .map_err(|e| anyhow::anyhow!("Character dossier failed: {}", e))?;

    if mode == OutputMode::Json {
        // Build a combined JSON structure
        let relationships = ctx
            .relationship_repo
            .get_character_relationships(entity_id)
            .await
            .unwrap_or_default();
        let knowledge_states = ctx
            .knowledge_repo
            .get_character_knowledge_states(entity_id)
            .await
            .unwrap_or_default();
        let perceptions_of = ctx
            .relationship_repo
            .get_perceptions_of(entity_id)
            .await
            .unwrap_or_default();

        #[derive(serde::Serialize)]
        struct ExploreJson<D, R, K, P> {
            dossier: D,
            relationships: R,
            knowledge_states: K,
            perceptions_of: P,
        }

        output_json(&ExploreJson {
            dossier: &dossier,
            relationships: &relationships,
            knowledge_states: &knowledge_states,
            perceptions_of: &perceptions_of,
        });
        return Ok(());
    }

    // === Header ===
    print_header(&format!("Character: {}", display_name));

    // === Summary ===
    print_kv(
        "Roles",
        &if dossier.roles.is_empty() {
            "none".to_string()
        } else {
            dossier.roles.join(", ")
        },
    );
    print_kv(
        "Centrality",
        &dossier
            .centrality_rank
            .map(|r| format!("#{}", r))
            .unwrap_or_else(|| "unranked".to_string()),
    );
    print_kv(
        "Influence reach",
        &format!("{} characters", dossier.influence_reach),
    );
    print_kv(
        "Knowledge",
        &format!(
            "{} advantages, {} blind spots, {} false beliefs",
            dossier.knowledge_advantages, dossier.knowledge_blind_spots, dossier.false_beliefs
        ),
    );
    if let Some(avg) = dossier.avg_tension_toward_them {
        print_kv("Avg tension toward them", &format!("{:.1}", avg));
    }

    // === Relationships ===
    let relationships = ctx
        .relationship_repo
        .get_character_relationships(entity_id)
        .await
        .unwrap_or_default();

    if !relationships.is_empty() {
        print_section(&format!("Relationships ({})", relationships.len()), "");
        let rows: Vec<Vec<String>> = relationships
            .iter()
            .map(|r| {
                let other = if r.from_character.to_string() == entity_id {
                    r.to_character.to_string()
                } else {
                    r.from_character.to_string()
                };
                vec![
                    other,
                    r.rel_type.clone(),
                    r.label.as_deref().unwrap_or("-").to_string(),
                ]
            })
            .collect();
        print_table(&["Character", "Type", "Label"], rows);
    }

    // === Knowledge States ===
    let knowledge_states = ctx
        .knowledge_repo
        .get_character_knowledge_states(entity_id)
        .await
        .unwrap_or_default();

    if !knowledge_states.is_empty() {
        print_section(
            &format!("Knowledge States ({})", knowledge_states.len()),
            "",
        );
        let rows: Vec<Vec<String>> = knowledge_states
            .iter()
            .map(|k| {
                vec![
                    k.target.to_string(),
                    format!("{:?}", k.certainty),
                    format!("{:?}", k.learning_method),
                ]
            })
            .collect();
        print_table(&["Target", "Certainty", "Method"], rows);
    }

    // === Perceptions (how others see this character) ===
    if !dossier.key_perceptions.is_empty() {
        print_section("Perceptions (how others see them)", "");
        let rows: Vec<Vec<String>> = dossier
            .key_perceptions
            .iter()
            .map(|p| {
                vec![
                    p.observer.clone(),
                    p.tension_level
                        .map(|t| format!("{}", t))
                        .unwrap_or_else(|| "-".to_string()),
                    p.feelings.as_deref().unwrap_or("-").to_string(),
                ]
            })
            .collect();
        print_table(&["Observer", "Tension", "Feelings"], rows);
    }

    // === Similar Entities ===
    if !no_similar && !no_semantic && ctx.embedding_service.is_available() {
        show_similar(ctx, entity_id, display_name, "character").await;
    }

    // === Suggestions ===
    if !dossier.suggestions.is_empty() {
        print_section("Suggestions", "");
        for s in &dossier.suggestions {
            println!("  - {}", s);
        }
    }

    Ok(())
}

/// Deep exploration for non-character entities.
#[allow(clippy::too_many_arguments)]
async fn explore_generic(
    ctx: &AppContext,
    entity_id: &str,
    entity_type: &str,
    display_name: &str,
    no_similar: bool,
    no_semantic: bool,
    depth: usize,
    mode: OutputMode,
) -> Result<()> {
    // Get full entity content
    let full_content = ctx
        .context_service
        .get_entity_full_detail(entity_id)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to get entity detail: {}", e))?;

    // Get connected entities
    let connected = ctx
        .relationship_repo
        .get_connected_entities(entity_id, depth)
        .await
        .unwrap_or_default();

    if mode == OutputMode::Json {
        #[derive(serde::Serialize)]
        struct GenericExploreJson {
            entity: Option<crate::services::EntityFullContent>,
            connected_entities: Vec<String>,
        }
        output_json(&GenericExploreJson {
            entity: full_content,
            connected_entities: connected,
        });
        return Ok(());
    }

    // === Header ===
    let type_label = entity_type
        .chars()
        .next()
        .map(|c| c.to_uppercase().to_string() + &entity_type[1..])
        .unwrap_or_else(|| entity_type.to_string());
    print_header(&format!("{}: {}", type_label, display_name));

    // === Full Content ===
    if let Some(content) = &full_content {
        print_kv("ID", entity_id);
        print_kv("Tokens", &format!("~{}", content.estimated_tokens));
        println!();
        println!("{}", content.content);
    }

    // === Connected Entities ===
    if !connected.is_empty() {
        print_section(
            &format!("Connected Entities ({}, depth={})", connected.len(), depth),
            "",
        );
        for id in &connected {
            println!("  {}", id);
        }
    }

    // === Similar ===
    if !no_similar && !no_semantic && ctx.embedding_service.is_available() {
        show_similar(ctx, entity_id, display_name, entity_type).await;
    }

    Ok(())
}

/// Show similar entities section using semantic search.
async fn show_similar(ctx: &AppContext, entity_id: &str, display_name: &str, entity_type: &str) {
    let filter = SearchFilter {
        limit: Some(6),
        ..Default::default()
    };

    if let Ok(results) = ctx
        .search_service
        .semantic_search(display_name, filter)
        .await
    {
        let similar: Vec<_> = results
            .into_iter()
            .filter(|r| r.id != entity_id && r.entity_type == entity_type)
            .take(5)
            .collect();
        if !similar.is_empty() {
            print_section("Similar Entities", "");
            let rows: Vec<Vec<String>> = similar
                .iter()
                .map(|r| vec![r.id.clone(), format!("{:.4}", r.score), r.name.clone()])
                .collect();
            print_table(&["ID", "Score", "Name"], rows);
        }
    }
}
