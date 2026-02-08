//! Entity ID resolution and normalization utilities.

use anyhow::Result;
use serde::Deserialize;
use std::sync::Arc;
use surrealdb::{engine::local::Db, Surreal};

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

/// A resolved entity with its type and display name.
#[derive(Debug, Clone)]
pub struct ResolvedEntity {
    pub entity_type: String,
    pub id: String,
    pub name: String,
}

#[derive(Deserialize)]
struct NameResult {
    id: surrealdb::RecordId,
    name: String,
}

/// Resolve an input string to entity matches by case-insensitive name/title search.
///
/// Searches character (name), location (name), event (title), scene (title).
/// Returns all matches across all entity tables.
pub async fn resolve_by_name(db: &Arc<Surreal<Db>>, input: &str) -> Result<Vec<ResolvedEntity>> {
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
        });
    }
    for r in locations.unwrap_or_default() {
        resolved.push(ResolvedEntity {
            entity_type: "location".to_string(),
            id: r.id.to_string(),
            name: r.name,
        });
    }
    for r in events.unwrap_or_default() {
        resolved.push(ResolvedEntity {
            entity_type: "event".to_string(),
            id: r.id.to_string(),
            name: r.name,
        });
    }
    for r in scenes.unwrap_or_default() {
        resolved.push(ResolvedEntity {
            entity_type: "scene".to_string(),
            id: r.id.to_string(),
            name: r.name,
        });
    }

    Ok(resolved)
}
