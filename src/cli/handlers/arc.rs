//! CLI handlers for arc tracking and what-if analysis.

use anyhow::Result;
use serde::Serialize;

use crate::cli::output::{output_json, print_header, print_kv, print_table, OutputMode};
use crate::cli::resolve::resolve_single;
use crate::init::AppContext;
use crate::services::arc::ArcService;
use crate::utils::math::cosine_similarity;

/// Extract the bare name from a table:key ID.
fn name_from_id(entity_id: &str) -> String {
    entity_id
        .split(':')
        .next_back()
        .unwrap_or(entity_id)
        .to_string()
}

pub async fn handle_arc_history(
    ctx: &AppContext,
    entity: &str,
    limit: usize,
    mode: OutputMode,
    no_semantic: bool,
) -> Result<()> {
    let entity_id = resolve_single(ctx, entity, no_semantic).await?;

    let service = ArcService::new(ctx.db.clone());
    let result = service.analyze_history(&entity_id, limit).await?;

    if result.total_snapshots == 0 {
        println!("No arc snapshots found for {}", entity_id);
        return Ok(());
    }

    if mode == OutputMode::Json {
        output_json(&result);
    } else {
        let entity_name = name_from_id(&entity_id);
        print_header(&format!(
            "Arc History: {} ({} snapshots)",
            entity_name, result.total_snapshots
        ));
        print_kv(
            "Net displacement",
            &format!("{:.4} ({})", result.net_displacement, result.assessment),
        );

        let rows: Vec<Vec<String>> = result
            .snapshots
            .iter()
            .enumerate()
            .map(|(i, snap)| {
                vec![
                    format!("{}", i + 1),
                    snap.delta
                        .map(|d| format!("{:.4}", d))
                        .unwrap_or_else(|| "baseline".to_string()),
                    format!("{:.4}", snap.cumulative),
                    snap.event.as_deref().unwrap_or("-").to_string(),
                    snap.timestamp.clone(),
                ]
            })
            .collect();
        print_table(&["#", "Delta", "Cumulative", "Event", "Timestamp"], rows);
    }

    Ok(())
}

pub async fn handle_arc_compare(
    ctx: &AppContext,
    entity_a: &str,
    entity_b: &str,
    window: Option<String>,
    mode: OutputMode,
    no_semantic: bool,
) -> Result<()> {
    let id_a = resolve_single(ctx, entity_a, no_semantic).await?;
    let id_b = resolve_single(ctx, entity_b, no_semantic).await?;

    let service = ArcService::new(ctx.db.clone());
    let result = service
        .analyze_comparison(&id_a, &id_b, window.as_deref())
        .await?;

    if mode == OutputMode::Json {
        output_json(&result);
    } else {
        let name_a = name_from_id(&id_a);
        let name_b = name_from_id(&id_b);
        print_header(&format!("Arc Comparison: {} vs {}", name_a, name_b));
        print_kv(
            "Initial similarity",
            &format!("{:.4}", result.initial_similarity),
        );
        print_kv(
            "Current similarity",
            &format!("{:.4}", result.current_similarity),
        );
        print_kv(
            "Convergence",
            &format!("{:+.4} ({})", result.convergence_delta, result.convergence),
        );
        print_kv(
            "Trajectory",
            &format!(
                "{:.4} ({})",
                result.trajectory_similarity, result.trajectory
            ),
        );
        print_kv(
            "Snapshots analyzed",
            &format!("{} vs {}", result.snapshots_a, result.snapshots_b),
        );
    }

    Ok(())
}

pub async fn handle_arc_moment(
    ctx: &AppContext,
    entity: &str,
    event: Option<String>,
    mode: OutputMode,
    no_semantic: bool,
) -> Result<()> {
    let entity_id = resolve_single(ctx, entity, no_semantic).await?;

    let service = ArcService::new(ctx.db.clone());
    let result = service.analyze_moment(&entity_id, event.as_deref()).await?;

    if mode == OutputMode::Json {
        output_json(&result);
    } else {
        let entity_name = name_from_id(&entity_id);
        let moment_desc = if event.is_some() {
            "at event"
        } else {
            "latest"
        };
        print_header(&format!("{} ({} snapshot)", entity_name, moment_desc));
        print_kv("Entity type", &result.entity_type);
        print_kv(
            "Delta",
            &result
                .delta_magnitude
                .map(|d| format!("{:.4}", d))
                .unwrap_or_else(|| "baseline".to_string()),
        );
        print_kv("Timestamp", &result.created_at);
        if let Some(ref evt) = result.event_title {
            print_kv("Event", evt);
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub async fn handle_what_if(
    ctx: &AppContext,
    character: &str,
    fact: &str,
    certainty: Option<String>,
    mode: OutputMode,
    no_semantic: bool,
) -> Result<()> {
    use crate::embedding::composite::{character_composite, knowledge_composite};
    use crate::models::knowledge::get_current_knowledge;

    if !ctx.embedding_service.is_available() {
        anyhow::bail!("What-if analysis requires embedding service");
    }

    let char_id = resolve_single(ctx, character, no_semantic).await?;
    let char_ref = if char_id.starts_with("character:") {
        char_id.clone()
    } else {
        format!("character:{}", char_id)
    };
    let char_key = char_ref.strip_prefix("character:").unwrap_or(character);

    // Resolve fact — it's a knowledge ID
    let fact_ref = if fact.starts_with("knowledge:") {
        fact.to_string()
    } else {
        format!("knowledge:{}", fact)
    };

    let certainty = certainty.as_deref().unwrap_or("knows");

    // Fetch character
    let mut char_resp = ctx.db.query(format!("SELECT * FROM {}", char_ref)).await?;
    let char_record: Option<crate::models::Character> = char_resp.take(0)?;
    let char_record =
        char_record.ok_or_else(|| anyhow::anyhow!("Character not found: {}", char_ref))?;

    // Fetch fact
    #[derive(serde::Deserialize)]
    struct FactRecord {
        fact: String,
    }
    let fact_query = format!("SELECT fact FROM {}", fact_ref);
    let mut fact_resp = ctx.db.query(&fact_query).await?;
    let facts: Vec<FactRecord> = fact_resp.take(0)?;
    let fact_record = facts
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("Knowledge fact not found: {}", fact_ref))?;

    // Current knowledge state
    let current_knowledge = get_current_knowledge(&ctx.db, char_key, &fact_ref).await?;
    let already_known = if let Some(ref state) = current_knowledge {
        format!("Already known with certainty: {:?}", state.certainty)
    } else {
        "Not currently known".to_string()
    };

    // Current embedding
    let emb_query = format!("SELECT VALUE embedding FROM {}", char_ref);
    let mut emb_resp = ctx.db.query(&emb_query).await?;
    let embeddings: Vec<Option<Vec<f32>>> = emb_resp.take(0).unwrap_or_default();
    let current_embedding = embeddings
        .into_iter()
        .next()
        .flatten()
        .ok_or_else(|| anyhow::anyhow!("Character {} has no embedding", char_ref))?;

    // Fetch relationships
    let mut rel_result = ctx
        .db
        .query(format!(
            "SELECT ->relates_to->character.{{name, relationship_type}} AS relationships FROM {}",
            char_ref
        ))
        .await?;

    #[derive(serde::Deserialize)]
    struct RelResult {
        relationships: Option<Vec<serde_json::Value>>,
    }
    let rel_data: Option<RelResult> = rel_result.take(0).unwrap_or(None);
    let relationships: Vec<(String, String)> = rel_data
        .and_then(|r| r.relationships)
        .unwrap_or_default()
        .iter()
        .filter_map(|r| {
            let rel_type = r.get("relationship_type")?.as_str()?;
            let name = r.get("name")?.as_str()?;
            Some((rel_type.to_string(), name.to_string()))
        })
        .collect();

    // Fetch perceptions
    #[derive(serde::Deserialize)]
    struct PerceptionRecord {
        target_name: Option<String>,
        perception: Option<String>,
    }
    let mut perc_result = ctx
        .db
        .query(format!(
            "SELECT out.name AS target_name, perception FROM perceives WHERE in = {}",
            char_ref
        ))
        .await?;
    let perc_records: Vec<PerceptionRecord> = perc_result.take(0).unwrap_or_default();
    let perceptions: Vec<(String, String)> = perc_records
        .into_iter()
        .filter_map(|p| Some((p.target_name?, p.perception?)))
        .collect();

    // Build hypothetical composite
    let current_composite = character_composite(&char_record, &relationships, &perceptions);
    let knowledge_text = knowledge_composite(&fact_record.fact, &char_record.name, certainty, None);
    let hypothetical_composite = format!(
        "{}. Additionally, {}",
        current_composite.trim_end_matches('.'),
        knowledge_text
    );

    let hypothetical_embedding = ctx
        .embedding_service
        .embed_text(&hypothetical_composite)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to embed hypothetical composite: {}", e))?;

    let delta = 1.0 - cosine_similarity(&current_embedding, &hypothetical_embedding);

    let impact_label = if delta < 0.02 {
        "negligible"
    } else if delta < 0.05 {
        "minor"
    } else if delta < 0.10 {
        "moderate"
    } else if delta < 0.20 {
        "major"
    } else {
        "transformative"
    };

    // Conflict detection
    let fact_embedding = ctx
        .embedding_service
        .embed_text(&fact_record.fact)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to embed fact text: {}", e))?;

    let knowledge_query = format!(
        "SELECT id, fact, embedding FROM knowledge WHERE character = {} AND embedding IS NOT NONE",
        char_ref
    );
    let mut knowledge_resp = ctx.db.query(&knowledge_query).await?;

    #[derive(serde::Deserialize)]
    struct KnowledgeWithEmbedding {
        id: surrealdb::sql::Thing,
        fact: String,
        embedding: Vec<f32>,
    }
    let existing_knowledge: Vec<KnowledgeWithEmbedding> =
        knowledge_resp.take(0).unwrap_or_default();

    let mut conflicts: Vec<(String, String, f32)> = Vec::new();
    for k in &existing_knowledge {
        let sim = cosine_similarity(&fact_embedding, &k.embedding);
        if sim > 0.7 {
            conflicts.push((k.id.to_string(), k.fact.clone(), sim));
        }
    }

    // Cascade detection
    let cascade_query = format!(
        "SELECT type::string(in) AS observer_id, in.name AS observer_name \
         FROM perceives WHERE out = {} AND embedding IS NOT NONE",
        char_ref
    );
    let mut cascade_resp = ctx.db.query(&cascade_query).await?;

    #[derive(serde::Deserialize)]
    struct CascadeRecord {
        observer_name: Option<String>,
    }
    let cascade_records: Vec<CascadeRecord> = cascade_resp.take(0).unwrap_or_default();

    // Output
    if mode == OutputMode::Json {
        #[derive(Serialize)]
        struct WhatIfOutput {
            character: String,
            fact: String,
            certainty: String,
            current_status: String,
            delta: f32,
            impact: String,
            conflicts: usize,
            cascade_observers: usize,
        }
        output_json(&WhatIfOutput {
            character: char_record.name.clone(),
            fact: fact_record.fact.clone(),
            certainty: certainty.to_string(),
            current_status: already_known.clone(),
            delta,
            impact: impact_label.to_string(),
            conflicts: conflicts.len(),
            cascade_observers: cascade_records.len(),
        });
    } else {
        print_header(&format!(
            "What if {} learned \"{}\"?",
            char_record.name, fact_record.fact
        ));
        print_kv("Certainty", certainty);
        print_kv("Current status", &already_known);
        print_kv(
            "Embedding delta",
            &format!("{:.4} ({})", delta, impact_label),
        );
        print_kv("Conflicts detected", &format!("{}", conflicts.len()));
        print_kv(
            "Perspectives that would go stale",
            &format!("{}", cascade_records.len()),
        );

        if !conflicts.is_empty() {
            println!();
            let rows: Vec<Vec<String>> = conflicts
                .iter()
                .map(|(id, fact, sim)| vec![id.clone(), fact.clone(), format!("{:.3}", sim)])
                .collect();
            print_table(&["Knowledge ID", "Existing Fact", "Similarity"], rows);
        }

        if !cascade_records.is_empty() {
            let observer_names: Vec<String> = cascade_records
                .iter()
                .filter_map(|c| c.observer_name.clone())
                .collect();
            if !observer_names.is_empty() {
                println!("\nAffected observers: {}", observer_names.join(", "));
            }
        }
    }

    Ok(())
}
