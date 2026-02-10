//! Shared DB queries for embedding composite text enrichment.
//!
//! Used by both `backfill.rs` and `staleness.rs` to fetch knowledge and
//! shared-scene data when building perspective composite texts.

use crate::db::connection::NarraDb;
use tracing::warn;

/// Fetch observer's knowledge about a target character (text-match on target name).
///
/// Returns (fact, certainty) tuples. Non-critical enrichment — returns empty vec on error.
pub async fn fetch_knowledge_about(
    db: &NarraDb,
    observer_id: &str,
    target_name: &str,
) -> Vec<(String, String)> {
    let query = format!(
        "SELECT out.fact AS fact, certainty FROM knows WHERE in = {} \
         AND out.fact IS NOT NONE AND string::contains(out.fact, $target_name) \
         ORDER BY learned_at DESC LIMIT 5",
        observer_id
    );

    #[derive(serde::Deserialize)]
    struct KnowledgeFact {
        fact: String,
        certainty: Option<String>,
    }

    let result = db
        .query(&query)
        .bind(("target_name", target_name.to_string()))
        .await;

    match result {
        Ok(mut resp) => {
            let facts: Vec<KnowledgeFact> = resp.take(0).unwrap_or_default();
            facts
                .into_iter()
                .map(|f| (f.fact, f.certainty.unwrap_or_else(|| "knows".to_string())))
                .collect()
        }
        Err(e) => {
            warn!("Failed to fetch knowledge about {}: {}", target_name, e);
            vec![]
        }
    }
}

/// Fetch scenes shared between two characters.
///
/// Uses two queries + Rust intersection. Returns (title, summary) tuples.
/// Non-critical enrichment — returns empty vec on error.
pub async fn fetch_shared_scenes(
    db: &NarraDb,
    observer_id: &str,
    target_id: &str,
) -> Vec<(String, Option<String>)> {
    let query_a = format!(
        "SELECT VALUE out FROM participates_in WHERE in = {}",
        observer_id
    );
    let query_b = format!(
        "SELECT VALUE out FROM participates_in WHERE in = {}",
        target_id
    );

    let (resp_a, resp_b) = match (db.query(&query_a).await, db.query(&query_b).await) {
        (Ok(a), Ok(b)) => (a, b),
        _ => return vec![],
    };

    let mut resp_a = resp_a;
    let mut resp_b = resp_b;

    let scenes_a: Vec<surrealdb::RecordId> = resp_a.take(0).unwrap_or_default();
    let scenes_b: Vec<surrealdb::RecordId> = resp_b.take(0).unwrap_or_default();

    let set_b: std::collections::HashSet<String> = scenes_b.iter().map(|r| r.to_string()).collect();
    let shared: Vec<String> = scenes_a
        .iter()
        .map(|r| r.to_string())
        .filter(|s| set_b.contains(s))
        .take(5)
        .collect();

    if shared.is_empty() {
        return vec![];
    }

    let scene_ids_str = shared.join(", ");
    let detail_query = format!(
        "SELECT title, summary FROM scene WHERE id IN [{}]",
        scene_ids_str
    );

    match db.query(&detail_query).await {
        Ok(mut resp) => {
            #[derive(serde::Deserialize)]
            struct SceneInfo {
                title: String,
                summary: Option<String>,
            }

            let scenes: Vec<SceneInfo> = resp.take(0).unwrap_or_default();
            scenes.into_iter().map(|s| (s.title, s.summary)).collect()
        }
        Err(e) => {
            warn!("Failed to fetch shared scenes: {}", e);
            vec![]
        }
    }
}
