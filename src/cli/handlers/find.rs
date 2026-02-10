//! Find handler for CLI — hybrid search by default, plus semantic subcommands.

use anyhow::Result;

use crate::cli::output::{
    create_spinner, output_json, output_json_list, print_hint, print_table, OutputMode,
};
use crate::cli::resolve::resolve_single;
use crate::init::AppContext;
use crate::repository::RelationshipRepository;
use crate::services::{EntityType, SearchFilter};
use crate::utils::math::cosine_similarity;

/// Parse an entity type string into an EntityType.
pub fn parse_entity_type(s: &str) -> Option<EntityType> {
    match s.to_lowercase().as_str() {
        "character" | "characters" => Some(EntityType::Character),
        "location" | "locations" => Some(EntityType::Location),
        "event" | "events" => Some(EntityType::Event),
        "scene" | "scenes" => Some(EntityType::Scene),
        "knowledge" => Some(EntityType::Knowledge),
        "note" | "notes" => Some(EntityType::Note),
        _ => None,
    }
}

#[allow(clippy::too_many_arguments)]
pub async fn handle_find(
    ctx: &AppContext,
    query: &str,
    keyword_only: bool,
    semantic_only: bool,
    rerank: bool,
    facet: Option<&str>,
    no_semantic: bool,
    entity_type: Option<&str>,
    limit: usize,
    mode: OutputMode,
) -> Result<()> {
    // Faceted search overrides other modes
    if let Some(facet_name) = facet {
        if !ctx.embedding_service.is_available() {
            anyhow::bail!("Faceted search unavailable — embedding model not loaded. Run 'narra world backfill' first.");
        }

        let filter = SearchFilter {
            limit: Some(limit),
            ..Default::default()
        };

        let results = ctx
            .search_service
            .faceted_search(query, facet_name, filter)
            .await?;

        if mode == OutputMode::Json {
            output_json_list(&results);
            return Ok(());
        }

        println!(
            "Faceted search ({} facet) for '{}': {} results\n",
            facet_name,
            query,
            results.len()
        );

        let rows: Vec<Vec<String>> = results
            .iter()
            .map(|r| vec![r.id.clone(), r.name.clone(), format!("{:.4}", r.score)])
            .collect();

        print_table(&["ID", "Name", "Score"], rows);

        if results.is_empty() {
            print_hint(&format!(
                "No characters found matching '{}' on {} facet. Try a different facet or query.",
                query, facet_name
            ));
        }

        return Ok(());
    }

    let entity_types = match entity_type {
        Some(t) => {
            if let Some(et) = parse_entity_type(t) {
                vec![et]
            } else {
                anyhow::bail!(
                    "Unknown entity type '{}'. Valid types: character, location, event, scene, knowledge, note",
                    t
                );
            }
        }
        None => vec![],
    };

    let filter = SearchFilter {
        entity_types,
        limit: Some(limit),
        ..Default::default()
    };

    // Determine search strategy: hybrid by default, with graceful fallback
    let (results, search_mode) = if keyword_only || no_semantic {
        let r = ctx.search_service.search(query, filter).await?;
        (r, "keyword")
    } else if semantic_only {
        let r = ctx.search_service.semantic_search(query, filter).await?;
        (r, "semantic")
    } else if rerank {
        let r = ctx.search_service.reranked_search(query, filter).await?;
        (r, "reranked")
    } else {
        // Default: hybrid (auto-falls back to keyword if embeddings unavailable)
        let has_embeddings = ctx.embedding_service.is_available();
        let r = ctx.search_service.hybrid_search(query, filter).await?;
        let mode_label = if has_embeddings {
            "hybrid"
        } else {
            "keyword (semantic unavailable)"
        };
        (r, mode_label)
    };

    if mode == OutputMode::Json {
        output_json_list(&results);
        return Ok(());
    }

    println!(
        "Search ({}) for '{}': {} results\n",
        search_mode,
        query,
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
        print_hint("Try broadening your query or removing the --type filter.");
    }

    Ok(())
}

// =============================================================================
// Semantic Join — cross-type vector search
// =============================================================================

pub async fn handle_semantic_join(
    ctx: &AppContext,
    query: &str,
    entity_type: Option<&str>,
    limit: usize,
    mode: OutputMode,
) -> Result<()> {
    if !ctx.embedding_service.is_available() {
        anyhow::bail!("Semantic search unavailable — embedding model not loaded. Run 'narra world backfill' first.");
    }

    let type_filter: Vec<EntityType> = if let Some(t) = entity_type {
        if let Some(et) = parse_entity_type(t) {
            vec![et]
        } else {
            anyhow::bail!(
                "Unknown entity type '{}'. Valid: character, location, event, scene, knowledge",
                t
            );
        }
    } else {
        EntityType::embeddable()
    };

    let spinner = create_spinner("Embedding query and searching...");

    let query_vector = ctx.embedding_service.embed_text(query).await?;

    #[derive(serde::Deserialize, serde::Serialize)]
    struct SearchResult {
        id: surrealdb::sql::Thing,
        entity_type: String,
        name: String,
        score: f32,
    }

    let mut all_results: Vec<SearchResult> = Vec::new();

    for et in &type_filter {
        let (table, name_field) = match et {
            EntityType::Character => ("character", "name"),
            EntityType::Location => ("location", "name"),
            EntityType::Event => ("event", "title"),
            EntityType::Scene => ("scene", "title"),
            EntityType::Knowledge => ("knowledge", "fact"),
            EntityType::Note => continue,
        };

        let k = limit * 2;
        let query_str = format!(
            "SELECT id, '{table}' AS entity_type, {name_field} AS name, \
             vector::similarity::cosine(embedding, $query_vector) AS score \
             FROM {table} \
             WHERE embedding IS NOT NONE \
             ORDER BY score DESC \
             LIMIT {k}",
        );

        let mut response = ctx
            .db
            .query(&query_str)
            .bind(("query_vector", query_vector.clone()))
            .await?;

        let table_results: Vec<SearchResult> = response.take(0).unwrap_or_default();
        all_results.extend(table_results);
    }

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
        "Semantic join for '{}': {} results (across {} types)\n",
        query,
        all_results.len(),
        type_filter.len()
    );

    let rows: Vec<Vec<String>> = all_results
        .iter()
        .map(|r| {
            vec![
                r.id.to_string(),
                r.entity_type.clone(),
                r.name.clone(),
                format!("{:.4}", r.score),
            ]
        })
        .collect();

    print_table(&["ID", "Type", "Name", "Score"], rows);

    if all_results.is_empty() {
        print_hint("No semantic matches found. Try broader terms or run 'narra world backfill'.");
    }

    Ok(())
}

// =============================================================================
// Semantic Knowledge — vector search within knowledge table
// =============================================================================

pub async fn handle_semantic_knowledge(
    ctx: &AppContext,
    query: &str,
    character: Option<&str>,
    limit: usize,
    mode: OutputMode,
    no_semantic: bool,
) -> Result<()> {
    if !ctx.embedding_service.is_available() {
        anyhow::bail!("Semantic search unavailable — embedding model not loaded. Run 'narra world backfill' first.");
    }

    // Resolve character if provided
    let character_id = if let Some(c) = character {
        Some(resolve_single(ctx, c, no_semantic).await?)
    } else {
        None
    };

    let spinner = create_spinner("Searching knowledge...");

    let query_vector = ctx.embedding_service.embed_text(query).await?;

    let query_str = if let Some(ref char_id) = character_id {
        let char_key = char_id.split(':').nth(1).unwrap_or(char_id);
        format!(
            "SELECT id, 'knowledge' AS entity_type, fact AS name, \
             vector::similarity::cosine(embedding, $query_vector) AS score, \
             character.name AS character_name \
             FROM knowledge \
             WHERE character = character:{char_key} AND embedding IS NOT NONE \
             ORDER BY score DESC \
             LIMIT {limit}",
            char_key = char_key,
            limit = limit * 2,
        )
    } else {
        format!(
            "SELECT id, 'knowledge' AS entity_type, fact AS name, \
             vector::similarity::cosine(embedding, $query_vector) AS score, \
             character.name AS character_name \
             FROM knowledge \
             WHERE embedding IS NOT NONE \
             ORDER BY score DESC \
             LIMIT {limit}",
            limit = limit * 2,
        )
    };

    let mut response = ctx
        .db
        .query(&query_str)
        .bind(("query_vector", query_vector))
        .await?;

    #[derive(serde::Deserialize, serde::Serialize)]
    struct KnowledgeResult {
        id: surrealdb::sql::Thing,
        entity_type: String,
        name: String,
        score: f32,
        character_name: Option<String>,
    }

    let results: Vec<KnowledgeResult> = response.take(0).unwrap_or_default();
    let results: Vec<KnowledgeResult> = results.into_iter().take(limit).collect();

    spinner.finish_and_clear();

    if mode == OutputMode::Json {
        output_json(&results);
        return Ok(());
    }

    let filter_label = character_id
        .as_deref()
        .map(|id| format!(" (filtered to {})", id))
        .unwrap_or_default();
    println!(
        "Semantic knowledge search for '{}'{}: {} results\n",
        query,
        filter_label,
        results.len()
    );

    let rows: Vec<Vec<String>> = results
        .iter()
        .map(|r| {
            vec![
                r.id.to_string(),
                r.name.clone(),
                r.character_name.as_deref().unwrap_or("-").to_string(),
                format!("{:.4}", r.score),
            ]
        })
        .collect();

    print_table(&["ID", "Fact", "Known By", "Score"], rows);

    if results.is_empty() {
        print_hint("No matches found. Try broader terms or run 'narra world backfill'.");
    }

    Ok(())
}

// =============================================================================
// Semantic Graph Search — graph proximity + vector similarity
// =============================================================================

#[allow(clippy::too_many_arguments)]
pub async fn handle_semantic_graph_search(
    ctx: &AppContext,
    entity: &str,
    query: &str,
    hops: usize,
    entity_type: Option<&str>,
    limit: usize,
    mode: OutputMode,
    no_semantic: bool,
) -> Result<()> {
    if !ctx.embedding_service.is_available() {
        anyhow::bail!("Semantic search unavailable — embedding model not loaded.");
    }

    let entity_id = resolve_single(ctx, entity, no_semantic).await?;

    let type_filter: Option<Vec<String>> = entity_type.map(|t| vec![t.to_lowercase()]);

    let spinner = create_spinner("Traversing graph and searching...");

    // Phase 1: graph traversal
    let connected = ctx
        .relationship_repo
        .get_connected_entities(&entity_id, hops)
        .await?;

    if connected.is_empty() {
        spinner.finish_and_clear();
        if mode == OutputMode::Json {
            output_json(&serde_json::json!({"results": [], "total": 0}));
        } else {
            println!(
                "No connected entities found within {} hops of {}",
                hops, entity_id
            );
        }
        return Ok(());
    }

    // Phase 2: embed query
    let query_vector = ctx.embedding_service.embed_text(query).await?;

    // Phase 3: group by table, batch fetch embeddings, compute cosine similarity
    let mut by_table: std::collections::HashMap<String, Vec<String>> =
        std::collections::HashMap::new();
    for id in &connected {
        let table = id.split(':').next().unwrap_or("unknown").to_string();
        if let Some(ref filter) = type_filter {
            if !filter.contains(&table) {
                continue;
            }
        }
        by_table.entry(table).or_default().push(id.clone());
    }

    #[derive(serde::Serialize)]
    struct ScoredResult {
        id: String,
        entity_type: String,
        name: String,
        score: f32,
    }

    let mut scored_results: Vec<ScoredResult> = Vec::new();

    for (table, ids) in &by_table {
        let name_field = match table.as_str() {
            "event" | "scene" => "title",
            "knowledge" => "fact",
            _ => "name",
        };

        let id_csv = ids.join(", ");
        let query_str = format!(
            "SELECT id, '{table}' AS entity_type, {name_field} AS name, embedding \
             FROM {table} WHERE id IN [{ids}] AND embedding IS NOT NONE",
            table = table,
            name_field = name_field,
            ids = id_csv,
        );

        let mut response = ctx.db.query(&query_str).await?;

        #[derive(serde::Deserialize)]
        struct EntityWithEmbedding {
            id: surrealdb::sql::Thing,
            entity_type: String,
            name: String,
            embedding: Vec<f32>,
        }

        let entities: Vec<EntityWithEmbedding> = response.take(0).unwrap_or_default();

        for e in entities {
            let similarity = cosine_similarity(&e.embedding, &query_vector);
            scored_results.push(ScoredResult {
                id: e.id.to_string(),
                entity_type: e.entity_type,
                name: e.name,
                score: similarity,
            });
        }
    }

    scored_results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    scored_results.truncate(limit);

    spinner.finish_and_clear();

    if mode == OutputMode::Json {
        output_json(&scored_results);
        return Ok(());
    }

    println!(
        "Graph search from '{}' ({} hops) for '{}': {} results (from {} candidates)\n",
        entity_id,
        hops,
        query,
        scored_results.len(),
        connected.len()
    );

    let rows: Vec<Vec<String>> = scored_results
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

    if scored_results.is_empty() {
        print_hint("Connected entities have no embeddings. Run 'narra world backfill'.");
    }

    Ok(())
}
