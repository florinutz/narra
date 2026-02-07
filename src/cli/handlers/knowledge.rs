//! Knowledge command handlers for CLI.

use anyhow::Result;

use crate::cli::output::{
    output_json, output_json_list, print_error, print_success, print_table, OutputMode,
};
use crate::init::AppContext;
use crate::models::{CertaintyLevel, KnowledgeCreate, KnowledgeStateCreate, LearningMethod};
use crate::repository::KnowledgeRepository;

/// Strip a known table prefix from an entity ID, returning the bare key.
fn bare_key(id: &str, prefix: &str) -> String {
    id.strip_prefix(&format!("{}:", prefix))
        .unwrap_or(id)
        .to_string()
}

pub async fn list_knowledge(
    ctx: &AppContext,
    character: Option<&str>,
    mode: OutputMode,
) -> Result<()> {
    match character {
        Some(char_id) => {
            let key = bare_key(char_id, "character");
            let states = ctx
                .knowledge_repo
                .get_character_knowledge_states(&key)
                .await?;

            if mode == OutputMode::Json {
                output_json_list(&states);
                return Ok(());
            }

            let rows: Vec<Vec<String>> = states
                .iter()
                .map(|s| {
                    vec![
                        s.id.to_string(),
                        s.target.to_string(),
                        format!("{:?}", s.certainty),
                        format!("{:?}", s.learning_method),
                        s.event.as_ref().map(|e| e.to_string()).unwrap_or_default(),
                    ]
                })
                .collect();

            print_table(&["ID", "Target", "Certainty", "Method", "Event"], rows);
        }
        None => {
            // List all knowledge entries (Phase 1 style)
            let knowledge = ctx.knowledge_repo.get_character_knowledge("").await;
            match knowledge {
                Ok(items) => {
                    if mode == OutputMode::Json {
                        output_json_list(&items);
                    } else {
                        println!("Use --character to filter knowledge by character.");
                    }
                }
                Err(_) => {
                    if mode == OutputMode::Json {
                        println!("[]");
                    } else {
                        println!(
                            "Use --character <id> to list knowledge for a specific character."
                        );
                    }
                }
            }
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub async fn record_knowledge(
    ctx: &AppContext,
    character: &str,
    fact: &str,
    certainty: &str,
    method: Option<&str>,
    source: Option<&str>,
    event: Option<&str>,
    mode: OutputMode,
) -> Result<()> {
    let char_key = bare_key(character, "character");

    // First, create a knowledge record (Phase 1 fact)
    let knowledge = ctx
        .knowledge_repo
        .create_knowledge(KnowledgeCreate {
            character: surrealdb::RecordId::from(("character", char_key.as_str())),
            fact: fact.to_string(),
        })
        .await?;

    // Then create a knowledge state edge (Phase 3 provenance)
    let certainty_level = match certainty.to_lowercase().as_str() {
        "knows" => CertaintyLevel::Knows,
        "suspects" => CertaintyLevel::Suspects,
        "believes_wrongly" => CertaintyLevel::BelievesWrongly,
        "uncertain" => CertaintyLevel::Uncertain,
        "assumes" => CertaintyLevel::Assumes,
        "denies" => CertaintyLevel::Denies,
        "forgotten" => CertaintyLevel::Forgotten,
        _ => CertaintyLevel::Knows,
    };

    let learning_method = match method.unwrap_or("initial").to_lowercase().as_str() {
        "told" => LearningMethod::Told,
        "overheard" => LearningMethod::Overheard,
        "witnessed" => LearningMethod::Witnessed,
        "discovered" => LearningMethod::Discovered,
        "deduced" => LearningMethod::Deduced,
        "read" => LearningMethod::Read,
        "remembered" => LearningMethod::Remembered,
        "initial" => LearningMethod::Initial,
        _ => LearningMethod::Initial,
    };

    let target = knowledge.id.to_string();

    let state_data = KnowledgeStateCreate {
        certainty: certainty_level,
        learning_method,
        source_character: source.map(|s| bare_key(s, "character")),
        event: event.map(|e| bare_key(e, "event")),
        ..Default::default()
    };

    match ctx
        .knowledge_repo
        .create_knowledge_state(&char_key, &target, state_data)
        .await
    {
        Ok(state) => {
            if mode == OutputMode::Json {
                output_json(&state);
            } else {
                print_success(&format!(
                    "Recorded knowledge: {} {:?} '{}' ({})",
                    char_key, certainty_level, fact, state.id
                ));
            }
        }
        Err(e) => {
            // Knowledge was still created even if state edge failed
            if mode == OutputMode::Json {
                output_json(&knowledge);
            } else {
                print_error(&format!(
                    "Knowledge created ({}) but state edge failed: {}",
                    knowledge.id, e
                ));
            }
        }
    }

    Ok(())
}
