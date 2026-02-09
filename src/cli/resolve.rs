//! Entity ID resolution and normalization utilities.

use anyhow::Result;
use serde::Deserialize;
use std::sync::Arc;
use surrealdb::{engine::local::Db, Surreal};

use crate::init::AppContext;
use crate::services::{SearchFilter, SearchService};

/// Strip a known table prefix from an entity ID, returning the bare key.
/// e.g. "character:alice" -> "alice", "alice" -> "alice"
pub fn bare_key(id: &str, prefix: &str) -> String {
    id.strip_prefix(&format!("{}:", prefix))
        .unwrap_or(id)
        .to_string()
}

/// Extract entity type from a full entity ID (e.g., "character:alice" -> "character").
pub fn entity_type_from_id(entity_id: &str) -> Option<&str> {
    if entity_id.contains(':') {
        entity_id.split(':').next()
    } else {
        None
    }
}

/// Ensure an entity ID has the proper table prefix.
/// If `input` already contains ':', return as-is.
/// If `expected_type` is provided, prepend it.
pub fn normalize_entity_id(input: &str, expected_type: Option<&str>) -> String {
    if input.contains(':') {
        input.to_string()
    } else if let Some(entity_type) = expected_type {
        format!("{}:{}", entity_type, input)
    } else {
        input.to_string()
    }
}

/// Resolve an entity name or ID to a single entity ID.
///
/// If `input` already contains ':', it's treated as a full ID and returned as-is.
/// Otherwise, resolves via name lookup (exact, fuzzy, semantic).
/// Errors if zero or multiple matches found.
pub async fn resolve_single(ctx: &AppContext, input: &str, no_semantic: bool) -> Result<String> {
    if input.contains(':') {
        return Ok(input.to_string());
    }
    let search_svc = if no_semantic {
        None
    } else {
        Some(ctx.search_service.as_ref())
    };
    let matches = resolve_by_name(&ctx.db, input, search_svc).await?;
    match matches.len() {
        0 => anyhow::bail!("No entity found for '{}'", input),
        1 => Ok(matches[0].id.clone()),
        _ => {
            eprintln!("Ambiguous name '{}'. Matches:", input);
            for m in &matches {
                eprintln!("  {} ({})", m.id, m.name);
            }
            anyhow::bail!("Use a full ID (type:key) to disambiguate");
        }
    }
}

/// How an entity was resolved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResolutionMethod {
    Exact,
    Fuzzy { score: u32 },
    Semantic,
}

/// A resolved entity with its type and display name.
#[derive(Debug, Clone)]
pub struct ResolvedEntity {
    pub entity_type: String,
    pub id: String,
    pub name: String,
    pub method: ResolutionMethod,
}

#[derive(Deserialize)]
struct NameResult {
    id: surrealdb::RecordId,
    name: String,
}

/// Resolve an input string to entity matches.
///
/// Fallback chain:
/// 1. Exact case-insensitive name/title match across character, location, event, scene
/// 2. Fuzzy search (Levenshtein) if exact returns 0 and search_service provided
/// 3. Semantic search if fuzzy also returns 0 and search_service provided
pub async fn resolve_by_name(
    db: &Arc<Surreal<Db>>,
    input: &str,
    search_service: Option<&(dyn SearchService + Send + Sync)>,
) -> Result<Vec<ResolvedEntity>> {
    // 1. Exact match
    let resolved = exact_name_match(db, input).await?;
    if !resolved.is_empty() {
        return Ok(resolved);
    }

    // 2. Fuzzy fallback
    if let Some(search) = search_service {
        let filter = SearchFilter {
            limit: Some(5),
            ..Default::default()
        };
        let fuzzy_results = search.fuzzy_search(input, 0.7, filter).await;
        if let Ok(results) = fuzzy_results {
            if !results.is_empty() {
                return Ok(results
                    .into_iter()
                    .map(|r| ResolvedEntity {
                        entity_type: r.entity_type.clone(),
                        id: r.id,
                        name: r.name,
                        method: ResolutionMethod::Fuzzy {
                            score: (r.score * 100.0) as u32,
                        },
                    })
                    .collect());
            }
        }

        // 3. Semantic fallback
        let filter = SearchFilter {
            limit: Some(3),
            ..Default::default()
        };
        let semantic_results = search.semantic_search(input, filter).await;
        if let Ok(results) = semantic_results {
            if !results.is_empty() {
                return Ok(results
                    .into_iter()
                    .map(|r| ResolvedEntity {
                        entity_type: r.entity_type.clone(),
                        id: r.id,
                        name: r.name,
                        method: ResolutionMethod::Semantic,
                    })
                    .collect());
            }
        }
    }

    Ok(Vec::new())
}

/// Exact case-insensitive name/title match across all entity tables.
async fn exact_name_match(db: &Arc<Surreal<Db>>, input: &str) -> Result<Vec<ResolvedEntity>> {
    let input_lower = input.to_lowercase();

    let (characters, locations, events, scenes) = tokio::join!(
        async {
            let mut resp = db
                .query("SELECT id, name FROM character WHERE string::lowercase(name) = $input")
                .bind(("input", input_lower.clone()))
                .await?;
            let results: Vec<NameResult> = resp.take(0).unwrap_or_default();
            Ok::<_, anyhow::Error>(results)
        },
        async {
            let mut resp = db
                .query("SELECT id, name FROM location WHERE string::lowercase(name) = $input")
                .bind(("input", input_lower.clone()))
                .await?;
            let results: Vec<NameResult> = resp.take(0).unwrap_or_default();
            Ok::<_, anyhow::Error>(results)
        },
        async {
            let mut resp = db
                .query(
                    "SELECT id, title AS name FROM event WHERE string::lowercase(title) = $input",
                )
                .bind(("input", input_lower.clone()))
                .await?;
            let results: Vec<NameResult> = resp.take(0).unwrap_or_default();
            Ok::<_, anyhow::Error>(results)
        },
        async {
            let mut resp = db
                .query(
                    "SELECT id, title AS name FROM scene WHERE string::lowercase(title) = $input",
                )
                .bind(("input", input_lower.clone()))
                .await?;
            let results: Vec<NameResult> = resp.take(0).unwrap_or_default();
            Ok::<_, anyhow::Error>(results)
        },
    );

    let mut resolved = Vec::new();

    for r in characters.unwrap_or_default() {
        resolved.push(ResolvedEntity {
            entity_type: "character".to_string(),
            id: r.id.to_string(),
            name: r.name,
            method: ResolutionMethod::Exact,
        });
    }
    for r in locations.unwrap_or_default() {
        resolved.push(ResolvedEntity {
            entity_type: "location".to_string(),
            id: r.id.to_string(),
            name: r.name,
            method: ResolutionMethod::Exact,
        });
    }
    for r in events.unwrap_or_default() {
        resolved.push(ResolvedEntity {
            entity_type: "event".to_string(),
            id: r.id.to_string(),
            name: r.name,
            method: ResolutionMethod::Exact,
        });
    }
    for r in scenes.unwrap_or_default() {
        resolved.push(ResolvedEntity {
            entity_type: "scene".to_string(),
            id: r.id.to_string(),
            name: r.name,
            method: ResolutionMethod::Exact,
        });
    }

    Ok(resolved)
}
