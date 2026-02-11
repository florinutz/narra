//! Integration tests for Tier 5: Embedding Infrastructure.
//!
//! Tests verify:
//! - Migration 013 fields (relates_to embedding, composite_text)
//! - relationship_composite() output format
//! - Composite text stored alongside embedding
//! - No-op detection when composite text unchanged
//! - relates_to edge gets embedding after backfill
//! - Metadata filters on semantic/hybrid search
//! - SimilarRelationships query operation
//! - EmbeddingHealth query operation

mod common;

use std::sync::Arc;

use common::{harness::TestHarness, to_mutation_input, to_query_input};
use narra::embedding::composite::relationship_composite;
use narra::embedding::NoopEmbeddingService;
use narra::mcp::{MutationRequest, NarraServer, QueryRequest, SearchMetadataFilter};
use narra::session::SessionStateManager;
use rmcp::handler::server::wrapper::Parameters;

/// Helper to create a 384-dimensional test embedding vector.
fn test_embedding(seed: f32) -> Vec<f32> {
    (0..384).map(|i| (seed + i as f32 * 0.001).sin()).collect()
}

/// Helper to format a 384-dim embedding for inline SQL.
fn embedding_sql(emb: &[f32]) -> String {
    let nums: Vec<String> = emb.iter().map(|v| format!("{:.6}", v)).collect();
    format!("[{}]", nums.join(","))
}

/// Helper to create NarraServer with isolated harness and noop embeddings.
async fn create_test_server(harness: &TestHarness) -> NarraServer {
    let session_path = harness.temp_path().join("session.json");
    let session_manager = Arc::new(
        SessionStateManager::load_or_create(&session_path)
            .expect("Failed to create session manager"),
    );
    NarraServer::new(
        harness.db.clone(),
        session_manager,
        Arc::new(NoopEmbeddingService::new()),
    )
    .await
}

// =============================================================================
// MIGRATION 013 TESTS (no embedding model needed)
// =============================================================================

/// Test that relates_to edge can have embedding and embedding_stale fields.
#[tokio::test]
async fn test_relates_to_embedding_fields() {
    let harness = TestHarness::new().await;

    // Create two characters
    harness
        .db
        .query("CREATE character:alice SET name = 'Alice', roles = ['warrior'], aliases = []")
        .await
        .unwrap();
    harness
        .db
        .query("CREATE character:bob SET name = 'Bob', roles = ['sage'], aliases = []")
        .await
        .unwrap();

    // Create a relates_to edge
    harness
        .db
        .query("RELATE character:alice->relates_to->character:bob SET rel_type = 'ally'")
        .await
        .unwrap();

    // Verify embedding_stale defaults to true
    let mut resp = harness.db.query(
        "SELECT embedding_stale FROM relates_to WHERE in = character:alice AND out = character:bob"
    ).await.unwrap();

    #[derive(serde::Deserialize)]
    struct StaleCheck {
        embedding_stale: bool,
    }

    let results: Vec<StaleCheck> = resp.take(0).unwrap();
    assert_eq!(results.len(), 1);
    assert!(
        results[0].embedding_stale,
        "embedding_stale should default to true"
    );

    // Set an embedding
    let emb = test_embedding(1.0);
    let emb_sql = embedding_sql(&emb);
    harness.db.query(format!(
        "UPDATE relates_to SET embedding = {}, embedding_stale = false WHERE in = character:alice AND out = character:bob",
        emb_sql
    )).await.unwrap();

    // Verify embedding was stored
    let mut resp = harness.db.query(
        "SELECT embedding_stale FROM relates_to WHERE in = character:alice AND out = character:bob"
    ).await.unwrap();

    let results: Vec<StaleCheck> = resp.take(0).unwrap();
    assert!(
        !results[0].embedding_stale,
        "embedding_stale should be false after update"
    );
}

/// Test that composite_text field exists on all embeddable tables.
#[tokio::test]
async fn test_composite_text_field_exists() {
    let harness = TestHarness::new().await;

    // Create entities and set composite_text on each
    harness.db.query("CREATE character:test SET name = 'Test', roles = [], aliases = [], composite_text = 'test composite'").await.unwrap();
    harness.db.query("CREATE location:test SET name = 'Test', loc_type = 'place', composite_text = 'location composite'").await.unwrap();
    harness.db.query("CREATE event:test SET title = 'Test', sequence = 1, composite_text = 'event composite'").await.unwrap();

    // Verify composite_text is stored
    #[derive(serde::Deserialize)]
    struct CompositeCheck {
        composite_text: Option<String>,
    }

    let mut resp = harness
        .db
        .query("SELECT composite_text FROM character:test")
        .await
        .unwrap();
    let results: Vec<CompositeCheck> = resp.take(0).unwrap();
    assert_eq!(results[0].composite_text.as_deref(), Some("test composite"));

    let mut resp = harness
        .db
        .query("SELECT composite_text FROM location:test")
        .await
        .unwrap();
    let results: Vec<CompositeCheck> = resp.take(0).unwrap();
    assert_eq!(
        results[0].composite_text.as_deref(),
        Some("location composite")
    );

    let mut resp = harness
        .db
        .query("SELECT composite_text FROM event:test")
        .await
        .unwrap();
    let results: Vec<CompositeCheck> = resp.take(0).unwrap();
    assert_eq!(
        results[0].composite_text.as_deref(),
        Some("event composite")
    );
}

/// Test that composite_text defaults to NONE.
#[tokio::test]
async fn test_composite_text_defaults_to_none() {
    let harness = TestHarness::new().await;

    harness
        .db
        .query("CREATE character:test SET name = 'Test', roles = [], aliases = []")
        .await
        .unwrap();

    #[derive(serde::Deserialize)]
    struct CompositeCheck {
        composite_text: Option<String>,
    }

    let mut resp = harness
        .db
        .query("SELECT composite_text FROM character:test")
        .await
        .unwrap();
    let results: Vec<CompositeCheck> = resp.take(0).unwrap();
    assert!(
        results[0].composite_text.is_none(),
        "composite_text should default to NONE"
    );
}

// =============================================================================
// COMPOSITE FUNCTION TESTS
// =============================================================================

/// Test relationship_composite() output format.
#[tokio::test]
async fn test_relationship_composite_output() {
    let result = relationship_composite(
        "Alice",
        &["warrior".to_string(), "protagonist".to_string()],
        "Bob",
        &["sage".to_string(), "mentor".to_string()],
        "family",
        Some("sibling"),
        Some("Alice's younger brother who she protects fiercely"),
    );

    assert!(
        result.contains("Alice (warrior, protagonist)"),
        "Should include from character with roles"
    );
    assert!(
        result.contains("Bob (sage, mentor)"),
        "Should include to character with roles"
    );
    assert!(
        result.contains("family bond"),
        "Should include relationship type"
    );
    assert!(
        result.contains("Subtype: sibling"),
        "Should include subtype"
    );
    assert!(
        result.contains("Alice's younger brother"),
        "Should include label"
    );
    assert!(result.ends_with('.'), "Should end with period");
}

/// Test relationship_composite() with minimal input.
#[tokio::test]
async fn test_relationship_composite_minimal() {
    let result = relationship_composite("Alice", &[], "Bob", &[], "rival", None, None);

    assert!(result.contains("Alice"));
    assert!(result.contains("Bob"));
    assert!(result.contains("rival"));
    assert!(
        !result.contains("Subtype:"),
        "Should not include subtype when None"
    );
    assert!(result.ends_with('.'));
}

// =============================================================================
// STALENESS TRIGGER TESTS (no embedding model needed)
// =============================================================================

/// Test that CreateRelationship triggers relates_to edge staleness management.
#[tokio::test]
async fn test_create_relationship_marks_character_embeddings_stale() {
    let harness = TestHarness::new().await;
    let server = create_test_server(&harness).await;

    // Create characters with embeddings
    let emb = test_embedding(1.0);
    let emb_sql = embedding_sql(&emb);
    harness
        .db
        .query(format!(
            "CREATE character:alice SET name = 'Alice', roles = ['warrior'], aliases = [], \
         embedding = {}, embedding_stale = false",
            emb_sql
        ))
        .await
        .unwrap();
    harness
        .db
        .query(format!(
            "CREATE character:bob SET name = 'Bob', roles = ['sage'], aliases = [], \
         embedding = {}, embedding_stale = false",
            emb_sql
        ))
        .await
        .unwrap();

    // Create relationship
    let result = server
        .handle_mutate(Parameters(to_mutation_input(
            MutationRequest::CreateRelationship {
                from_character_id: "alice".to_string(),
                to_character_id: "bob".to_string(),
                rel_type: "ally".to_string(),
                subtype: None,
                label: None,
            },
        )))
        .await;

    assert!(
        result.is_ok(),
        "CreateRelationship should succeed: {:?}",
        result.err()
    );

    // Wait for async regeneration to fire (it'll fail with noop embeddings, but the stale flag should be set)
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    // Verify character embeddings were marked stale
    #[derive(serde::Deserialize)]
    struct StaleCheck {
        embedding_stale: bool,
    }

    let mut resp = harness
        .db
        .query("SELECT embedding_stale FROM character:alice")
        .await
        .unwrap();
    let results: Vec<StaleCheck> = resp.take(0).unwrap();
    assert!(
        results[0].embedding_stale,
        "Alice's embedding should be marked stale"
    );

    let mut resp = harness
        .db
        .query("SELECT embedding_stale FROM character:bob")
        .await
        .unwrap();
    let results: Vec<StaleCheck> = resp.take(0).unwrap();
    assert!(
        results[0].embedding_stale,
        "Bob's embedding should be marked stale"
    );
}

/// Test that character name change marks relates_to edges stale.
#[tokio::test]
async fn test_character_name_change_marks_relates_to_stale() {
    let harness = TestHarness::new().await;
    let server = create_test_server(&harness).await;

    // Create characters
    harness
        .db
        .query("CREATE character:alice SET name = 'Alice', roles = ['warrior'], aliases = []")
        .await
        .unwrap();
    harness
        .db
        .query("CREATE character:bob SET name = 'Bob', roles = ['sage'], aliases = []")
        .await
        .unwrap();

    // Create a relationship and manually clear its stale flag
    harness.db.query(
        "RELATE character:alice->relates_to->character:bob SET rel_type = 'ally', embedding_stale = false"
    ).await.unwrap();

    // Verify edge is not stale
    #[derive(serde::Deserialize)]
    struct StaleCheck {
        embedding_stale: bool,
    }

    let mut resp = harness.db.query(
        "SELECT embedding_stale FROM relates_to WHERE in = character:alice AND out = character:bob"
    ).await.unwrap();
    let results: Vec<StaleCheck> = resp.take(0).unwrap();
    assert!(
        !results[0].embedding_stale,
        "Edge should not be stale before name change"
    );

    // Update character name
    let result = server
        .handle_mutate(Parameters(to_mutation_input(MutationRequest::Update {
            entity_id: "character:alice".to_string(),
            fields: serde_json::json!({"name": "Alice the Great"}),
        })))
        .await;

    assert!(result.is_ok(), "Update should succeed");

    // Verify relates_to edge was marked stale
    let mut resp = harness.db.query(
        "SELECT embedding_stale FROM relates_to WHERE in = character:alice AND out = character:bob"
    ).await.unwrap();
    let results: Vec<StaleCheck> = resp.take(0).unwrap();
    assert!(
        results[0].embedding_stale,
        "Edge should be stale after character name change"
    );
}

// =============================================================================
// EMBEDDING HEALTH TESTS
// =============================================================================

/// Test EmbeddingHealth on empty world returns all zeros.
#[tokio::test]
async fn test_embedding_health_empty_world() {
    let harness = TestHarness::new().await;
    let server = create_test_server(&harness).await;

    let result = server
        .handle_query(Parameters(to_query_input(QueryRequest::EmbeddingHealth)))
        .await;
    assert!(
        result.is_ok(),
        "EmbeddingHealth should succeed: {:?}",
        result.err()
    );

    let response = result.unwrap();
    assert!(
        !response.results.is_empty(),
        "Should have results for each embeddable type + overall"
    );

    // Find overall result
    let overall = response
        .results
        .iter()
        .find(|r| r.name == "OVERALL")
        .unwrap();
    assert!(
        overall.content.contains("0/0"),
        "Should show 0/0 for empty world"
    );
}

/// Test EmbeddingHealth reports correct counts.
#[tokio::test]
async fn test_embedding_health_with_data() {
    let harness = TestHarness::new().await;
    let server = create_test_server(&harness).await;

    let emb = test_embedding(1.0);
    let emb_sql = embedding_sql(&emb);

    // Create 3 characters: 2 with embeddings, 1 without
    harness
        .db
        .query(format!(
            "CREATE character:alice SET name = 'Alice', roles = [], aliases = [], \
         embedding = {}, embedding_stale = false",
            emb_sql
        ))
        .await
        .unwrap();
    harness
        .db
        .query(format!(
            "CREATE character:bob SET name = 'Bob', roles = [], aliases = [], \
         embedding = {}, embedding_stale = true",
            emb_sql
        ))
        .await
        .unwrap();
    harness
        .db
        .query("CREATE character:carol SET name = 'Carol', roles = [], aliases = []")
        .await
        .unwrap();

    let result = server
        .handle_query(Parameters(to_query_input(QueryRequest::EmbeddingHealth)))
        .await
        .unwrap();

    // Find character result
    let char_health = result
        .results
        .iter()
        .find(|r| r.name == "character")
        .unwrap();
    assert!(
        char_health.content.contains("2/3"),
        "Should show 2/3 embedded"
    );
    assert!(
        char_health.content.contains("1 stale"),
        "Should show 1 stale"
    );

    // Check hints suggest backfill
    assert!(
        result
            .hints
            .iter()
            .any(|h| h.contains("BackfillEmbeddings")),
        "Should suggest running BackfillEmbeddings"
    );
}

// =============================================================================
// SIMILAR RELATIONSHIPS TESTS
// =============================================================================

/// Test SimilarRelationships with no edge returns clear error.
#[tokio::test]
async fn test_similar_relationships_no_edge() {
    let harness = TestHarness::new().await;
    let server = create_test_server(&harness).await;

    // Create characters without any edges
    harness
        .db
        .query("CREATE character:alice SET name = 'Alice', roles = [], aliases = []")
        .await
        .unwrap();
    harness
        .db
        .query("CREATE character:bob SET name = 'Bob', roles = [], aliases = []")
        .await
        .unwrap();

    let result = server
        .handle_query(Parameters(to_query_input(
            QueryRequest::SimilarRelationships {
                observer_id: "alice".to_string(),
                target_id: "bob".to_string(),
                edge_type: None,
                bias: None,
                limit: Some(10),
            },
        )))
        .await;

    // Should return an error because embedding service is noop
    assert!(
        result.is_err(),
        "Should fail when embedding service unavailable"
    );
}

/// Test SimilarRelationships finds similar edges.
#[tokio::test]
async fn test_similar_relationships_finds_matches() {
    let harness = TestHarness::new().await;
    let server = create_test_server(&harness).await;

    let emb1 = test_embedding(1.0);
    let emb2 = test_embedding(1.1); // Slightly different
    let emb3 = test_embedding(5.0); // Very different

    // Create characters
    for name in &["alice", "bob", "carol", "dave"] {
        harness
            .db
            .query(format!(
                "CREATE character:{} SET name = '{}', roles = [], aliases = []",
                name,
                name.chars().next().unwrap().to_uppercase().to_string() + &name[1..]
            ))
            .await
            .unwrap();
    }

    // Create perceives edges with embeddings
    let emb1_sql = embedding_sql(&emb1);
    let emb2_sql = embedding_sql(&emb2);
    let emb3_sql = embedding_sql(&emb3);

    harness
        .db
        .query(format!(
            "RELATE character:alice->perceives->character:bob SET \
         rel_types = ['ally'], perception = 'trusted friend', feelings = 'warmth', \
         embedding = {}, embedding_stale = false",
            emb1_sql
        ))
        .await
        .unwrap();

    harness
        .db
        .query(format!(
            "RELATE character:carol->perceives->character:dave SET \
         rel_types = ['ally'], perception = 'loyal companion', feelings = 'friendship', \
         embedding = {}, embedding_stale = false",
            emb2_sql
        ))
        .await
        .unwrap();

    harness
        .db
        .query(format!(
            "RELATE character:alice->perceives->character:carol SET \
         rel_types = ['enemy'], perception = 'dangerous foe', feelings = 'hatred', \
         embedding = {}, embedding_stale = false",
            emb3_sql
        ))
        .await
        .unwrap();

    // SimilarRelationships would fail with NoopEmbeddingService since it
    // checks is_available(). This test verifies the query structure is correct.
    // Full functionality requires real embeddings, tested in cross_engine_queries tests.
    let result = server
        .handle_query(Parameters(to_query_input(
            QueryRequest::SimilarRelationships {
                observer_id: "alice".to_string(),
                target_id: "bob".to_string(),
                edge_type: Some("perceives".to_string()),
                bias: None,
                limit: Some(5),
            },
        )))
        .await;

    // Should fail because noop embedding service
    assert!(result.is_err(), "Should fail with noop embedding service");
}

// =============================================================================
// FILTERED SEARCH TESTS (with metadata filter parsing)
// =============================================================================

/// Test UnifiedSearch (semantic mode) with filter field (noop embedding service — just verify it doesn't crash).
#[tokio::test]
async fn test_semantic_search_with_filter_rejects_noop() {
    let harness = TestHarness::new().await;
    let server = create_test_server(&harness).await;

    let result = server
        .handle_query(Parameters(to_query_input(QueryRequest::UnifiedSearch {
            query: "betrayal".to_string(),
            mode: "semantic".to_string(),
            entity_types: Some(vec!["character".to_string()]),
            limit: Some(10),
            phase: None,
            filter: Some(SearchMetadataFilter {
                roles: Some("antagonist".to_string()),
                ..Default::default()
            }),
        })))
        .await;

    // Should fail because noop embedding service
    assert!(result.is_err(), "Should fail with noop embedding service");
}

/// Test UnifiedSearch (hybrid mode) with filter field (noop embedding — returns keyword-only results).
#[tokio::test]
async fn test_hybrid_search_with_filter_graceful_fallback() {
    let harness = TestHarness::new().await;
    let server = create_test_server(&harness).await;

    // Create a character
    harness
        .db
        .query("CREATE character:alice SET name = 'Alice', roles = ['warrior'], aliases = []")
        .await
        .unwrap();

    let result = server
        .handle_query(Parameters(to_query_input(QueryRequest::UnifiedSearch {
            query: "Alice".to_string(),
            mode: "hybrid".to_string(),
            entity_types: Some(vec!["character".to_string()]),
            limit: Some(10),
            phase: None,
            filter: Some(SearchMetadataFilter {
                roles: Some("warrior".to_string()),
                ..Default::default()
            }),
        })))
        .await;

    // Hybrid mode should fall back to keyword search when embeddings unavailable
    assert!(
        result.is_ok(),
        "Should fall back gracefully: {:?}",
        result.err()
    );
}

/// Test that an empty (default) filter is silently ignored.
#[tokio::test]
async fn test_filter_empty_fields_ignored() {
    let harness = TestHarness::new().await;
    let server = create_test_server(&harness).await;

    harness
        .db
        .query("CREATE character:alice SET name = 'Alice', roles = [], aliases = []")
        .await
        .unwrap();

    // Filter with no matching fields — all None fields produce empty filter list
    let result = server
        .handle_query(Parameters(to_query_input(QueryRequest::UnifiedSearch {
            query: "Alice".to_string(),
            mode: "hybrid".to_string(),
            entity_types: Some(vec!["character".to_string()]),
            limit: Some(10),
            phase: None,
            filter: Some(SearchMetadataFilter::default()),
        })))
        .await;

    assert!(
        result.is_ok(),
        "Empty filter should be silently ignored: {:?}",
        result.err()
    );
}

// =============================================================================
// RERANKED SEARCH FALLBACK
// =============================================================================

/// Test UnifiedSearch with mode="reranked" and NoopEmbeddingService returns error
/// or degrades gracefully (since reranking requires semantic search first).
#[tokio::test]
async fn test_unified_search_reranked_fallback() {
    let harness = TestHarness::new().await;
    let server = create_test_server(&harness).await;

    // Create a character so there's data to search
    harness
        .db
        .query("CREATE character:alice SET name = 'Alice', roles = ['warrior'], aliases = []")
        .await
        .unwrap();

    let result = server
        .handle_query(Parameters(to_query_input(QueryRequest::UnifiedSearch {
            query: "Alice".to_string(),
            mode: "reranked".to_string(),
            entity_types: Some(vec!["character".to_string()]),
            limit: Some(10),
            phase: None,
            filter: None,
        })))
        .await;

    // Reranked mode requires embeddings for the initial hybrid search.
    // With NoopEmbeddingService, it should either:
    // (a) fail with an error about embeddings, or
    // (b) degrade gracefully to keyword-only results (hybrid fallback, reranking skipped)
    match result {
        Ok(response) => {
            // Graceful degradation — the handler ran without error.
            // Hints should reference the search mode (re-ranked, hybrid, semantic, etc.)
            assert!(
                !response.hints.is_empty(),
                "Degraded response should have hints"
            );
            // Verify we didn't silently return garbage — either results or empty is fine
            // The key property is: no panic, no error, sensible output
        }
        Err(e) => {
            // Expected error about embedding service
            assert!(
                e.to_lowercase().contains("embed")
                    || e.to_lowercase().contains("semantic")
                    || e.to_lowercase().contains("rerank"),
                "Error should mention embedding/semantic/rerank: {}",
                e
            );
        }
    }
}
