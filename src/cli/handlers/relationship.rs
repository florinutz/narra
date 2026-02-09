//! Relationship command handlers for CLI.

use anyhow::Result;

use crate::cli::output::{
    create_spinner, output_json, output_json_list, print_hint, print_success, print_table,
    OutputMode,
};
use crate::cli::resolve::{bare_key, resolve_single};
use crate::init::AppContext;
use crate::models::RelationshipCreate;
use crate::repository::RelationshipRepository;

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

// =============================================================================
// Similar Relationships — find edges similar to a reference pair
// =============================================================================

#[allow(clippy::too_many_arguments)]
pub async fn handle_similar_relationships(
    ctx: &AppContext,
    observer: &str,
    target: &str,
    edge_type: Option<&str>,
    bias: Option<&str>,
    limit: usize,
    mode: OutputMode,
    no_semantic: bool,
) -> Result<()> {
    if !ctx.embedding_service.is_available() {
        anyhow::bail!(
            "Similar relationships requires embeddings. Run 'narra world backfill' first."
        );
    }

    let observer_id = resolve_single(ctx, observer, no_semantic).await?;
    let target_id = resolve_single(ctx, target, no_semantic).await?;

    // Normalize IDs
    let from_ref = if observer_id.contains(':') {
        observer_id.clone()
    } else {
        format!("character:{}", observer_id)
    };
    let to_ref = if target_id.contains(':') {
        target_id.clone()
    } else {
        format!("character:{}", target_id)
    };

    let spinner = create_spinner("Searching for similar relationships...");

    // Find source edge embedding
    let tables_to_try: Vec<&str> = match edge_type {
        Some("perceives") => vec!["perceives"],
        Some("relates_to") => vec!["relates_to"],
        _ => vec!["perceives", "relates_to"],
    };

    #[derive(serde::Deserialize)]
    struct EdgeEmbedding {
        embedding: Vec<f32>,
    }

    let mut source_embedding: Option<Vec<f32>> = None;
    let mut source_edge_type = "";

    // Convert to RecordId for proper binding
    let (from_table, from_key) = from_ref.split_once(':').unwrap_or(("character", &from_ref));
    let (to_table, to_key) = to_ref.split_once(':').unwrap_or(("character", &to_ref));
    let from_record = surrealdb::RecordId::from((from_table, from_key));
    let to_record = surrealdb::RecordId::from((to_table, to_key));

    for table in &tables_to_try {
        let query = format!(
            "SELECT embedding FROM {} WHERE in = $from_id AND out = $to_id AND embedding IS NOT NONE LIMIT 1",
            table
        );

        let mut resp = ctx
            .db
            .query(&query)
            .bind(("from_id", from_record.clone()))
            .bind(("to_id", to_record.clone()))
            .await?;
        let results: Vec<EdgeEmbedding> = resp.take(0).unwrap_or_default();
        if let Some(edge) = results.into_iter().next() {
            source_embedding = Some(edge.embedding);
            source_edge_type = table;
            break;
        }
    }

    let source_embedding = source_embedding.ok_or_else(|| {
        anyhow::anyhow!(
            "No edge with embedding found between {} and {}. Run 'narra world backfill'.",
            from_ref,
            to_ref
        )
    })?;

    // Build search vector (with optional bias)
    let search_vec = if let Some(bias_text) = bias {
        let bias_vec = ctx.embedding_service.embed_text(bias_text).await?;
        // Interpolate: 0.7 * source + 0.3 * bias, then normalize
        let combined: Vec<f32> = source_embedding
            .iter()
            .zip(bias_vec.iter())
            .map(|(s, b)| 0.7 * s + 0.3 * b)
            .collect();
        let norm: f32 = combined.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            combined.iter().map(|x| x / norm).collect()
        } else {
            combined
        }
    } else {
        source_embedding
    };

    // Search across both edge tables
    #[derive(serde::Deserialize, serde::Serialize)]
    struct SimilarEdge {
        from_name: Option<String>,
        to_name: Option<String>,
        edge_kind: Option<String>,
        score: f32,
        // perceives fields
        perception: Option<String>,
        feelings: Option<String>,
        tension_level: Option<i32>,
        // relates_to fields
        rel_type: Option<String>,
        subtype: Option<String>,
        label: Option<String>,
    }

    let mut all_results: Vec<SimilarEdge> = Vec::new();

    for table in &["perceives", "relates_to"] {
        let (extra_fields, edge_kind_str) = if *table == "perceives" {
            ("perception, feelings, tension_level,", "'perceives'")
        } else {
            ("rel_type, subtype, label,", "'relates_to'")
        };

        let query = format!(
            "SELECT in.name AS from_name, out.name AS to_name, \
             {extra_fields} {edge_kind_str} AS edge_kind, \
             vector::similarity::cosine(embedding, $search_vec) AS score \
             FROM {table} \
             WHERE embedding IS NOT NONE AND !(in = {from_ref} AND out = {to_ref}) \
             ORDER BY score DESC LIMIT {limit}",
        );

        let mut resp = ctx
            .db
            .query(&query)
            .bind(("search_vec", search_vec.clone()))
            .await?;

        let edges: Vec<SimilarEdge> = resp.take(0).unwrap_or_default();
        all_results.extend(edges);
    }

    // Sort by score descending
    all_results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    all_results.truncate(limit);

    spinner.finish_and_clear();

    if mode == OutputMode::Json {
        output_json(&all_results);
        return Ok(());
    }

    println!(
        "Similar relationships to {} -> {} ({}): {} results\n",
        from_ref,
        to_ref,
        source_edge_type,
        all_results.len()
    );
    if let Some(b) = bias {
        println!("  Biased toward: \"{}\"\n", b);
    }

    let rows: Vec<Vec<String>> = all_results
        .iter()
        .map(|e| {
            let from = e.from_name.as_deref().unwrap_or("?");
            let to = e.to_name.as_deref().unwrap_or("?");
            let kind = e.edge_kind.as_deref().unwrap_or("?");

            let detail = if kind == "perceives" {
                format!(
                    "{}{}",
                    e.perception.as_deref().unwrap_or("-"),
                    e.tension_level
                        .map(|t| format!(" (tension {}/10)", t))
                        .unwrap_or_default()
                )
            } else {
                format!(
                    "{}{}",
                    e.rel_type.as_deref().unwrap_or("-"),
                    e.subtype
                        .as_deref()
                        .map(|s| format!("/{}", s))
                        .unwrap_or_default()
                )
            };

            vec![
                format!("{} -> {}", from, to),
                kind.to_string(),
                format!("{:.4}", e.score),
                detail,
            ]
        })
        .collect();

    print_table(&["Pair", "Type", "Similarity", "Details"], rows);

    if all_results.is_empty() {
        print_hint(
            "No similar edges found. Ensure edge embeddings exist via 'narra world backfill'.",
        );
    }

    Ok(())
}
