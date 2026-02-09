//! Natural language query handler — hybrid search + context enrichment.

use anyhow::Result;

use crate::cli::output::{output_json, print_hint, print_section, print_table, OutputMode};
use crate::init::AppContext;
use crate::services::{ContextConfig, SearchFilter};

#[allow(clippy::too_many_arguments)]
pub async fn handle_ask(
    ctx: &AppContext,
    question: &str,
    limit: usize,
    show_context: bool,
    budget: usize,
    mode: OutputMode,
    no_semantic: bool,
) -> Result<()> {
    let filter = SearchFilter {
        limit: Some(limit),
        ..Default::default()
    };

    // Run hybrid search (or keyword-only if no_semantic)
    let (results, search_mode) = if no_semantic {
        let r = ctx.search_service.search(question, filter).await?;
        (r, "keyword")
    } else {
        let has_embeddings = ctx.embedding_service.is_available();
        let r = ctx.search_service.hybrid_search(question, filter).await?;
        let label = if has_embeddings {
            "hybrid"
        } else {
            "keyword (semantic unavailable)"
        };
        (r, label)
    };

    if mode == OutputMode::Json {
        if show_context && !results.is_empty() {
            let entity_ids: Vec<String> = results.iter().take(10).map(|r| r.id.clone()).collect();
            let config = ContextConfig {
                token_budget: budget,
                max_entities: entity_ids.len(),
                ..Default::default()
            };
            let context = ctx
                .context_service
                .get_context(&entity_ids, config)
                .await
                .ok();

            #[derive(serde::Serialize)]
            struct AskJson<R, C> {
                question: String,
                search_mode: String,
                results: R,
                context: C,
            }
            output_json(&AskJson {
                question: question.to_string(),
                search_mode: search_mode.to_string(),
                results: &results,
                context: &context,
            });
        } else {
            #[derive(serde::Serialize)]
            struct AskJsonNoCtx<R> {
                question: String,
                search_mode: String,
                results: R,
            }
            output_json(&AskJsonNoCtx {
                question: question.to_string(),
                search_mode: search_mode.to_string(),
                results: &results,
            });
        }
        return Ok(());
    }

    // Human-readable output
    println!(
        "Question: {}\nSearch mode: {} | {} results\n",
        question,
        search_mode,
        results.len()
    );

    let rows: Vec<Vec<String>> = results
        .iter()
        .map(|r| {
            vec![
                r.id.clone(),
                r.entity_type.clone(),
                r.name.clone(),
                format!("{:.4}", r.score),
            ]
        })
        .collect();
    print_table(&["ID", "Type", "Name", "Score"], rows);

    if results.is_empty() {
        print_hint("No results found. Try rephrasing your question.");
        return Ok(());
    }

    // Context enrichment
    if show_context {
        let entity_ids: Vec<String> = results.iter().take(10).map(|r| r.id.clone()).collect();
        let config = ContextConfig {
            token_budget: budget,
            max_entities: entity_ids.len(),
            ..Default::default()
        };

        match ctx.context_service.get_context(&entity_ids, config).await {
            Ok(context) => {
                if !context.entities.is_empty() {
                    print_section(
                        &format!(
                            "Context: {} entities, ~{} tokens{}",
                            context.entities.len(),
                            context.estimated_tokens,
                            if context.truncated {
                                " (truncated)"
                            } else {
                                ""
                            }
                        ),
                        "",
                    );
                    for entity in &context.entities {
                        let content_preview = entity
                            .content
                            .as_deref()
                            .unwrap_or("")
                            .chars()
                            .take(120)
                            .collect::<String>();
                        println!(
                            "  {} (score: {:.1}) — {}{}",
                            entity.id,
                            entity.score,
                            content_preview,
                            if entity.content.as_ref().is_some_and(|c| c.len() > 120) {
                                "..."
                            } else {
                                ""
                            }
                        );
                    }
                }
            }
            Err(e) => {
                print_hint(&format!("Context enrichment failed: {}", e));
            }
        }
    }

    Ok(())
}
