//! Integration tests for Temporal Vectors / Arc Tracking (Phase 3).
//!
//! Tests verify:
//! - Arc snapshot table creation via schema migration
//! - Snapshot capture during embedding regeneration
//! - BaselineArcSnapshots mutation
//! - ArcHistory, ArcComparison, ArcDrift, ArcMoment query operations
//! - Delta magnitude computation
//! - Event ID linking on snapshots

mod common;

use std::collections::HashMap;
use std::sync::Arc;

use common::harness::TestHarness;
use narra::embedding::backfill::BackfillService;
use narra::embedding::{
    EmbeddingConfig, LocalEmbeddingService, NoopEmbeddingService, StalenessManager,
};
use narra::mcp::{MutationRequest, NarraServer, QueryRequest};
use narra::models::character::{
    create_character, update_character, CharacterCreate, CharacterUpdate,
};
use narra::models::event::{create_event, EventCreate};
use narra::session::SessionStateManager;
use rmcp::handler::server::wrapper::Parameters;
use surrealdb::Datetime;

/// Helper to create a 384-dimensional test embedding vector.
/// HNSW indexes on character/knowledge tables require 384-dim vectors.
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

/// Helper to create NarraServer with real embedding service.
async fn create_test_server_with_embeddings(
    harness: &TestHarness,
    embedding_service: Arc<LocalEmbeddingService>,
) -> NarraServer {
    let session_path = harness.temp_path().join("session.json");
    let session_manager = Arc::new(
        SessionStateManager::load_or_create(&session_path)
            .expect("Failed to create session manager"),
    );
    NarraServer::new(harness.db.clone(), session_manager, embedding_service).await
}

// =============================================================================
// SCHEMA / TABLE TESTS (no embedding model needed)
// =============================================================================

/// Test that arc_snapshot table is created by schema migration.
#[tokio::test]
async fn test_arc_snapshot_table_exists() {
    let harness = TestHarness::new().await;

    // Verify we can insert into arc_snapshot (table exists)
    let result = harness
        .db
        .query(
            "CREATE arc_snapshot SET entity_id = character:test, entity_type = 'character', \
             embedding = [1.0, 2.0, 3.0]",
        )
        .await;
    assert!(
        result.is_ok(),
        "Should be able to insert into arc_snapshot table"
    );

    // Query specific fields (avoid entity_id record type in serde_json::Value)
    let mut response = harness
        .db
        .query("SELECT entity_type, delta_magnitude, event_id, created_at FROM arc_snapshot")
        .await
        .expect("Query should succeed");
    let rows: Vec<serde_json::Value> = response.take(0).expect("Should parse results");
    assert_eq!(rows.len(), 1, "Should have one snapshot");

    let row = &rows[0];
    assert_eq!(
        row.get("entity_type").unwrap().as_str().unwrap(),
        "character"
    );
    assert!(
        row.get("created_at").is_some(),
        "created_at should be auto-set"
    );
    assert!(
        row.get("delta_magnitude").unwrap().is_null(),
        "delta_magnitude should default to NONE"
    );
    assert!(
        row.get("event_id").unwrap().is_null(),
        "event_id should default to NONE"
    );
}

/// Test that entity_type field validates correctly.
#[tokio::test]
async fn test_arc_snapshot_entity_type_validation() {
    let harness = TestHarness::new().await;

    // Valid types should work
    let result = harness
        .db
        .query(
            "CREATE arc_snapshot SET entity_id = character:test, entity_type = 'character', \
             embedding = [1.0]",
        )
        .await;
    assert!(result.is_ok(), "'character' should be valid entity_type");

    let result = harness
        .db
        .query(
            "CREATE arc_snapshot SET entity_id = knowledge:test, entity_type = 'knowledge', \
             embedding = [1.0]",
        )
        .await;
    assert!(result.is_ok(), "'knowledge' should be valid entity_type");

    // Invalid type should fail the ASSERT
    let mut response = harness
        .db
        .query(
            "CREATE arc_snapshot SET entity_id = location:test, entity_type = 'location', \
             embedding = [1.0]",
        )
        .await
        .expect("Query should execute");
    let result: Result<Vec<serde_json::Value>, _> = response.take(0);
    assert!(result.is_err(), "'location' should be rejected by ASSERT");
}

/// Test arc_snapshot indexes exist by verifying ordered queries work.
#[tokio::test]
async fn test_arc_snapshot_indexes() {
    let harness = TestHarness::new().await;

    // Insert multiple snapshots for the same entity
    for i in 0..5 {
        let emb: Vec<f32> = vec![i as f32; 3];
        harness
            .db
            .query("CREATE arc_snapshot SET entity_id = character:alice, entity_type = 'character', embedding = $emb")
            .bind(("emb", emb))
            .await
            .expect("Insert should succeed");
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }

    // Use count queries to avoid record deserialization issues
    #[derive(serde::Deserialize)]
    struct CountResult {
        cnt: i64,
    }

    // Query with entity_id + created_at ordering (uses idx_arc_entity_time)
    let mut response = harness
        .db
        .query(
            "SELECT count() AS cnt FROM arc_snapshot WHERE entity_id = character:alice GROUP ALL",
        )
        .await
        .expect("Query should succeed");
    let count: Option<CountResult> = response.take(0).unwrap_or(None);
    assert_eq!(count.unwrap().cnt, 5, "Should return all 5 snapshots");

    // Query with entity_type filter (uses idx_arc_entity_type)
    let mut response = harness
        .db
        .query("SELECT count() AS cnt FROM arc_snapshot WHERE entity_type = 'character' GROUP ALL")
        .await
        .expect("Query should succeed");
    let count: Option<CountResult> = response.take(0).unwrap_or(None);
    assert_eq!(
        count.unwrap().cnt,
        5,
        "Should return all character snapshots"
    );
}

// =============================================================================
// BASELINE MUTATION TESTS (no embedding model needed — uses manual embeddings)
// =============================================================================

/// Test BaselineArcSnapshots creates snapshots for entities with embeddings.
#[tokio::test]
async fn test_baseline_arc_snapshots_creates_snapshots() {
    let harness = TestHarness::new().await;
    let server = create_test_server(&harness).await;

    // Create characters
    let alice = create_character(
        &harness.db,
        CharacterCreate {
            name: "Alice".into(),
            aliases: vec![],
            roles: vec!["protagonist".into()],
            ..Default::default()
        },
    )
    .await
    .expect("Should create Alice");

    let bob = create_character(
        &harness.db,
        CharacterCreate {
            name: "Bob".into(),
            aliases: vec![],
            roles: vec!["antagonist".into()],
            ..Default::default()
        },
    )
    .await
    .expect("Should create Bob");

    // Manually set embeddings using inline SQL (avoids bind parameter deserialization issues)
    let alice_id = alice.id.to_string();
    let bob_id = bob.id.to_string();

    let alice_emb = test_embedding(1.0);
    let bob_emb = test_embedding(2.0);

    harness
        .db
        .query(format!(
            "UPDATE {} SET embedding = {}, embedding_stale = false",
            alice_id,
            embedding_sql(&alice_emb)
        ))
        .await
        .expect("Should set Alice embedding");

    harness
        .db
        .query(format!(
            "UPDATE {} SET embedding = {}, embedding_stale = false",
            bob_id,
            embedding_sql(&bob_emb)
        ))
        .await
        .expect("Should set Bob embedding");

    // Verify embeddings are actually set
    #[derive(serde::Deserialize)]
    struct EmbCheck {
        embedding: Option<Vec<f32>>,
    }

    let mut check = harness
        .db
        .query(format!("SELECT embedding FROM {}", alice_id))
        .await
        .unwrap();
    let result: Vec<EmbCheck> = check.take(0).unwrap();
    assert!(
        result[0].embedding.is_some(),
        "Alice embedding should be set"
    );

    // Run BaselineArcSnapshots
    let request = MutationRequest::BaselineArcSnapshots { entity_type: None };
    let response = server
        .handle_mutate(Parameters(request))
        .await
        .expect("BaselineArcSnapshots should succeed");

    assert!(
        response.entity.content.contains("2 created"),
        "Should create 2 snapshots, got: {}",
        response.entity.content
    );

    // Verify snapshots exist via count
    #[derive(serde::Deserialize)]
    struct CountResult {
        cnt: i64,
    }

    let mut resp = harness
        .db
        .query("SELECT count() AS cnt FROM arc_snapshot GROUP ALL")
        .await
        .expect("Query should succeed");
    let count: Option<CountResult> = resp.take(0).unwrap_or(None);
    assert_eq!(count.unwrap().cnt, 2, "Should have 2 baseline snapshots");
}

/// Test BaselineArcSnapshots is idempotent (skips entities with existing snapshots).
#[tokio::test]
async fn test_baseline_arc_snapshots_idempotent() {
    let harness = TestHarness::new().await;
    let server = create_test_server(&harness).await;

    // Create character with embedding
    let alice = create_character(
        &harness.db,
        CharacterCreate {
            name: "Alice".into(),
            ..Default::default()
        },
    )
    .await
    .expect("Should create Alice");

    let alice_id = alice.id.to_string();
    let emb = test_embedding(1.0);
    harness
        .db
        .query(format!(
            "UPDATE {} SET embedding = {}, embedding_stale = false",
            alice_id,
            embedding_sql(&emb)
        ))
        .await
        .expect("Should set embedding");

    // Run twice
    let request1 = MutationRequest::BaselineArcSnapshots { entity_type: None };
    let resp1 = server
        .handle_mutate(Parameters(request1))
        .await
        .expect("First run should succeed");
    assert!(
        resp1.entity.content.contains("1 created"),
        "First run: {}",
        resp1.entity.content
    );

    let request2 = MutationRequest::BaselineArcSnapshots { entity_type: None };
    let resp2 = server
        .handle_mutate(Parameters(request2))
        .await
        .expect("Second run should succeed");
    assert!(
        resp2.entity.content.contains("0 created"),
        "Second run should create 0, got: {}",
        resp2.entity.content
    );
    assert!(
        resp2.entity.content.contains("1 skipped"),
        "Second run should skip 1, got: {}",
        resp2.entity.content
    );

    // Only 1 snapshot total
    #[derive(serde::Deserialize)]
    struct CountResult {
        cnt: i64,
    }
    let mut resp = harness
        .db
        .query("SELECT count() AS cnt FROM arc_snapshot GROUP ALL")
        .await
        .expect("Query should succeed");
    let count: Option<CountResult> = resp.take(0).unwrap_or(None);
    assert_eq!(count.unwrap().cnt, 1, "Should still have only 1 snapshot");
}

/// Test BaselineArcSnapshots filters by entity_type.
#[tokio::test]
async fn test_baseline_arc_snapshots_type_filter() {
    let harness = TestHarness::new().await;
    let server = create_test_server(&harness).await;

    // Create character with embedding
    let alice = create_character(
        &harness.db,
        CharacterCreate {
            name: "Alice".into(),
            ..Default::default()
        },
    )
    .await
    .expect("Should create Alice");
    let emb = test_embedding(1.0);
    harness
        .db
        .query(format!(
            "UPDATE {} SET embedding = {}, embedding_stale = false",
            alice.id,
            embedding_sql(&emb)
        ))
        .await
        .expect("Set embedding");

    // Run with knowledge filter — should skip characters
    let request = MutationRequest::BaselineArcSnapshots {
        entity_type: Some("knowledge".to_string()),
    };
    let resp = server
        .handle_mutate(Parameters(request))
        .await
        .expect("Should succeed");
    assert!(
        resp.entity.content.contains("0 created"),
        "Knowledge-only filter should skip characters, got: {}",
        resp.entity.content
    );

    // Run with character filter — should create snapshot
    let request = MutationRequest::BaselineArcSnapshots {
        entity_type: Some("character".to_string()),
    };
    let resp = server
        .handle_mutate(Parameters(request))
        .await
        .expect("Should succeed");
    assert!(
        resp.entity.content.contains("1 created"),
        "Character filter should create 1, got: {}",
        resp.entity.content
    );
}

/// Test BaselineArcSnapshots rejects invalid entity_type.
#[tokio::test]
async fn test_baseline_arc_snapshots_invalid_type() {
    let harness = TestHarness::new().await;
    let server = create_test_server(&harness).await;

    let request = MutationRequest::BaselineArcSnapshots {
        entity_type: Some("location".to_string()),
    };
    let result = server.handle_mutate(Parameters(request)).await;
    assert!(result.is_err(), "Should reject 'location' as entity_type");
    assert!(
        result.unwrap_err().contains("Invalid entity_type"),
        "Error should mention invalid type"
    );
}

// =============================================================================
// QUERY OPERATION TESTS (manual snapshots, no embedding model needed)
// =============================================================================

/// Helper to insert a test snapshot directly.
async fn insert_snapshot(
    harness: &TestHarness,
    entity_id: &str,
    entity_type: &str,
    embedding: &[f32],
    delta_magnitude: Option<f32>,
) {
    let query = format!(
        "CREATE arc_snapshot SET entity_id = {}, entity_type = $etype, \
         embedding = $emb, delta_magnitude = $delta",
        entity_id
    );
    harness
        .db
        .query(&query)
        .bind(("etype", entity_type.to_string()))
        .bind(("emb", embedding.to_vec()))
        .bind(("delta", delta_magnitude))
        .await
        .expect("Should insert snapshot");
    // Small delay for distinct timestamps
    tokio::time::sleep(std::time::Duration::from_millis(15)).await;
}

/// Test ArcHistory returns snapshots in chronological order.
#[tokio::test]
async fn test_arc_history_returns_chronological_snapshots() {
    let harness = TestHarness::new().await;
    let server = create_test_server(&harness).await;

    let alice = create_character(
        &harness.db,
        CharacterCreate {
            name: "Alice".into(),
            ..Default::default()
        },
    )
    .await
    .expect("Should create Alice");
    let alice_id = alice.id.to_string();

    // Insert 3 snapshots with increasing change
    insert_snapshot(&harness, &alice_id, "character", &[1.0, 0.0, 0.0], None).await;
    insert_snapshot(
        &harness,
        &alice_id,
        "character",
        &[0.9, 0.1, 0.0],
        Some(0.05),
    )
    .await;
    insert_snapshot(
        &harness,
        &alice_id,
        "character",
        &[0.5, 0.5, 0.0],
        Some(0.15),
    )
    .await;

    let request = QueryRequest::ArcHistory {
        entity_id: alice_id.clone(),
        limit: Some(50),
    };
    let response = server
        .handle_query(Parameters(request))
        .await
        .expect("ArcHistory should succeed");

    assert_eq!(response.results.len(), 3, "Should return 3 snapshots");
    assert_eq!(response.total, 3);

    // First snapshot should say "baseline" (no delta)
    assert!(
        response.results[0].content.contains("baseline"),
        "First snapshot should be baseline, got: {}",
        response.results[0].content
    );

    // Second should have delta
    assert!(
        response.results[1].content.contains("0.05"),
        "Second snapshot should show delta, got: {}",
        response.results[1].content
    );

    // Hints should include net displacement and assessment
    let hints_joined = response.hints.join(" ");
    assert!(
        hints_joined.contains("Net displacement"),
        "Hints should include displacement"
    );
    assert!(
        hints_joined.contains("Cumulative drift"),
        "Hints should include cumulative drift"
    );
}

/// Test ArcHistory with empty results.
#[tokio::test]
async fn test_arc_history_empty() {
    let harness = TestHarness::new().await;
    let server = create_test_server(&harness).await;

    let request = QueryRequest::ArcHistory {
        entity_id: "character:nonexistent".to_string(),
        limit: None,
    };
    let response = server
        .handle_query(Parameters(request))
        .await
        .expect("Should return empty result, not error");

    assert_eq!(response.results.len(), 0);
    assert!(
        response
            .hints
            .iter()
            .any(|h| h.contains("No arc snapshots")),
        "Should hint about missing snapshots"
    );
}

/// Test ArcHistory respects limit.
#[tokio::test]
async fn test_arc_history_limit() {
    let harness = TestHarness::new().await;
    let server = create_test_server(&harness).await;

    let alice = create_character(
        &harness.db,
        CharacterCreate {
            name: "Alice".into(),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    let alice_id = alice.id.to_string();

    for i in 0..10 {
        insert_snapshot(
            &harness,
            &alice_id,
            "character",
            &[i as f32, 0.0],
            if i == 0 { None } else { Some(0.01) },
        )
        .await;
    }

    let request = QueryRequest::ArcHistory {
        entity_id: alice_id,
        limit: Some(3),
    };
    let response = server
        .handle_query(Parameters(request))
        .await
        .expect("Should succeed");
    assert_eq!(response.results.len(), 3, "Should respect limit of 3");
}

/// Test ArcComparison between two entities.
#[tokio::test]
async fn test_arc_comparison_converging() {
    let harness = TestHarness::new().await;
    let server = create_test_server(&harness).await;

    let alice = create_character(
        &harness.db,
        CharacterCreate {
            name: "Alice".into(),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    let bob = create_character(
        &harness.db,
        CharacterCreate {
            name: "Bob".into(),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    let alice_id = alice.id.to_string();
    let bob_id = bob.id.to_string();

    // Start far apart, end close together (converging)
    insert_snapshot(&harness, &alice_id, "character", &[1.0, 0.0, 0.0], None).await;
    insert_snapshot(&harness, &bob_id, "character", &[0.0, 1.0, 0.0], None).await;
    insert_snapshot(
        &harness,
        &alice_id,
        "character",
        &[0.7, 0.7, 0.0],
        Some(0.1),
    )
    .await;
    insert_snapshot(&harness, &bob_id, "character", &[0.7, 0.7, 0.0], Some(0.1)).await;

    let request = QueryRequest::ArcComparison {
        entity_id_a: alice_id,
        entity_id_b: bob_id,
        window: None,
    };
    let response = server
        .handle_query(Parameters(request))
        .await
        .expect("ArcComparison should succeed");

    assert_eq!(
        response.results.len(),
        1,
        "Should return single comparison result"
    );
    let content = &response.results[0].content;
    // extract_name_from_id returns the key part of the ID, not "Alice"
    // Just verify the comparison report structure
    assert!(
        content.contains("Convergence delta"),
        "Should discuss convergence, got: {}",
        content
    );
    assert!(
        content.contains("Trajectory similarity"),
        "Should discuss trajectory, got: {}",
        content
    );

    // Convergence should be positive (they moved toward each other)
    let hints_joined = response.hints.join(" ");
    assert!(
        hints_joined.contains("Convergence"),
        "Hints should mention convergence"
    );
}

/// Test ArcComparison with window parameter.
#[tokio::test]
async fn test_arc_comparison_with_window() {
    let harness = TestHarness::new().await;
    let server = create_test_server(&harness).await;

    let alice = create_character(
        &harness.db,
        CharacterCreate {
            name: "Alice".into(),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    let bob = create_character(
        &harness.db,
        CharacterCreate {
            name: "Bob".into(),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    let alice_id = alice.id.to_string();
    let bob_id = bob.id.to_string();

    for i in 0..5 {
        let emb = vec![i as f32, 0.0];
        insert_snapshot(
            &harness,
            &alice_id,
            "character",
            &emb,
            if i == 0 { None } else { Some(0.1) },
        )
        .await;
        insert_snapshot(
            &harness,
            &bob_id,
            "character",
            &emb,
            if i == 0 { None } else { Some(0.1) },
        )
        .await;
    }

    let request = QueryRequest::ArcComparison {
        entity_id_a: alice_id,
        entity_id_b: bob_id,
        window: Some("recent:2".to_string()),
    };
    let response = server
        .handle_query(Parameters(request))
        .await
        .expect("ArcComparison with window should succeed");
    assert_eq!(response.results.len(), 1);
    assert!(
        response.results[0].content.contains("2"),
        "Should reference window size"
    );
}

/// Test ArcComparison with missing snapshots returns error.
#[tokio::test]
async fn test_arc_comparison_missing_snapshots() {
    let harness = TestHarness::new().await;
    let server = create_test_server(&harness).await;

    let request = QueryRequest::ArcComparison {
        entity_id_a: "character:alice".to_string(),
        entity_id_b: "character:bob".to_string(),
        window: None,
    };
    let result = server.handle_query(Parameters(request)).await;
    assert!(result.is_err(), "Should error when no snapshots exist");
    assert!(
        result.unwrap_err().contains("BaselineArcSnapshots"),
        "Error should suggest running BaselineArcSnapshots"
    );
}

/// Test ArcDrift ranks entities by total drift.
#[tokio::test]
async fn test_arc_drift_ranks_by_drift() {
    let harness = TestHarness::new().await;
    let server = create_test_server(&harness).await;

    let alice = create_character(
        &harness.db,
        CharacterCreate {
            name: "Alice".into(),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    let bob = create_character(
        &harness.db,
        CharacterCreate {
            name: "Bob".into(),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    let alice_id = alice.id.to_string();
    let bob_id = bob.id.to_string();

    // Alice: lots of drift
    insert_snapshot(&harness, &alice_id, "character", &[1.0, 0.0], None).await;
    insert_snapshot(&harness, &alice_id, "character", &[0.0, 1.0], Some(0.5)).await;
    insert_snapshot(&harness, &alice_id, "character", &[-1.0, 0.0], Some(0.5)).await;

    // Bob: small drift
    insert_snapshot(&harness, &bob_id, "character", &[1.0, 0.0], None).await;
    insert_snapshot(&harness, &bob_id, "character", &[0.99, 0.01], Some(0.01)).await;

    let request = QueryRequest::ArcDrift {
        entity_type: None,
        limit: Some(10),
    };
    let response = server
        .handle_query(Parameters(request))
        .await
        .expect("ArcDrift should succeed");

    assert_eq!(response.results.len(), 2, "Should return 2 entities");

    // First result should have higher drift (Alice's total_drift = 1.0 vs Bob's 0.01)
    let first_confidence = response.results[0].confidence.unwrap();
    let second_confidence = response.results[1].confidence.unwrap();
    assert!(
        first_confidence > second_confidence,
        "First entity should have higher drift: {} vs {}",
        first_confidence,
        second_confidence
    );

    // Both should have drift content
    for result in &response.results {
        assert!(
            result.content.contains("Total drift"),
            "Should show drift metrics"
        );
        assert!(
            result.content.contains("Efficiency"),
            "Should show efficiency"
        );
    }
}

/// Test ArcDrift with entity_type filter.
#[tokio::test]
async fn test_arc_drift_type_filter() {
    let harness = TestHarness::new().await;
    let server = create_test_server(&harness).await;

    let alice = create_character(
        &harness.db,
        CharacterCreate {
            name: "Alice".into(),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    let alice_id = alice.id.to_string();

    // Character snapshot with drift
    insert_snapshot(&harness, &alice_id, "character", &[1.0, 0.0], None).await;
    insert_snapshot(&harness, &alice_id, "character", &[0.0, 1.0], Some(0.5)).await;

    // Knowledge snapshot with drift
    harness.db
        .query("CREATE knowledge:test_fact SET character = character:alice, fact = 'test fact', embedding = [1.0, 0.0], embedding_stale = false")
        .await
        .expect("Create knowledge");
    insert_snapshot(
        &harness,
        "knowledge:test_fact",
        "knowledge",
        &[1.0, 0.0],
        None,
    )
    .await;
    insert_snapshot(
        &harness,
        "knowledge:test_fact",
        "knowledge",
        &[0.9, 0.1],
        Some(0.02),
    )
    .await;

    // Filter to knowledge only
    let request = QueryRequest::ArcDrift {
        entity_type: Some("knowledge".to_string()),
        limit: None,
    };
    let response = server
        .handle_query(Parameters(request))
        .await
        .expect("Should succeed");
    assert_eq!(response.results.len(), 1, "Should only show knowledge");
    assert_eq!(response.results[0].entity_type, "knowledge");
}

/// Test ArcDrift with no data returns empty.
#[tokio::test]
async fn test_arc_drift_empty() {
    let harness = TestHarness::new().await;
    let server = create_test_server(&harness).await;

    let request = QueryRequest::ArcDrift {
        entity_type: None,
        limit: None,
    };
    let response = server
        .handle_query(Parameters(request))
        .await
        .expect("Should succeed with empty results");
    assert_eq!(response.results.len(), 0);
    assert!(
        response.hints.iter().any(|h| h.contains("No arc drift")),
        "Should hint about missing data"
    );
}

/// Test ArcMoment returns latest snapshot when no event specified.
#[tokio::test]
async fn test_arc_moment_latest() {
    let harness = TestHarness::new().await;
    let server = create_test_server(&harness).await;

    let alice = create_character(
        &harness.db,
        CharacterCreate {
            name: "Alice".into(),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    let alice_id = alice.id.to_string();

    insert_snapshot(&harness, &alice_id, "character", &[1.0, 0.0], None).await;
    insert_snapshot(&harness, &alice_id, "character", &[0.0, 1.0], Some(0.5)).await;

    let request = QueryRequest::ArcMoment {
        entity_id: alice_id,
        event_id: None,
    };
    let response = server
        .handle_query(Parameters(request))
        .await
        .expect("ArcMoment should succeed");

    assert_eq!(response.results.len(), 1, "Should return single snapshot");
    assert!(
        response.results[0].content.contains("latest"),
        "Should indicate this is the latest snapshot"
    );
    assert!(
        response.results[0].content.contains("0.5"),
        "Should show delta magnitude, got: {}",
        response.results[0].content
    );
}

/// Test ArcMoment with no snapshots returns error.
#[tokio::test]
async fn test_arc_moment_no_snapshots() {
    let harness = TestHarness::new().await;
    let server = create_test_server(&harness).await;

    let request = QueryRequest::ArcMoment {
        entity_id: "character:nonexistent".to_string(),
        event_id: None,
    };
    let result = server.handle_query(Parameters(request)).await;
    assert!(result.is_err(), "Should error with no snapshots");
    assert!(
        result.unwrap_err().contains("BaselineArcSnapshots"),
        "Should suggest running baseline"
    );
}

/// Test ArcMoment with event_id returns nearest-before snapshot.
#[tokio::test]
async fn test_arc_moment_at_event() {
    let harness = TestHarness::new().await;
    let server = create_test_server(&harness).await;

    let alice = create_character(
        &harness.db,
        CharacterCreate {
            name: "Alice".into(),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    let alice_id = alice.id.to_string();

    // Insert first snapshot
    insert_snapshot(&harness, &alice_id, "character", &[1.0, 0.0], None).await;

    // Wait, then create event
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    let event = create_event(
        &harness.db,
        EventCreate {
            title: "The Betrayal".into(),
            description: Some("A pivotal event".into()),
            sequence: 10,
            date: None,
            date_precision: None,
            duration_end: None,
        },
    )
    .await
    .expect("Should create event");
    let event_id = event.id.to_string();

    // Wait, then insert second snapshot
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    insert_snapshot(&harness, &alice_id, "character", &[0.0, 1.0], Some(0.5)).await;

    // Query for snapshot at the event (should get the first one, created before the event)
    let request = QueryRequest::ArcMoment {
        entity_id: alice_id,
        event_id: Some(event_id),
    };
    let response = server
        .handle_query(Parameters(request))
        .await
        .expect("ArcMoment at event should succeed");

    assert_eq!(response.results.len(), 1);
    assert!(
        response.results[0].content.contains("at event"),
        "Should indicate event context"
    );
    // Should be the baseline snapshot (before the event), not the 0.5 delta one
    assert!(
        response.results[0].content.contains("baseline"),
        "Should return pre-event snapshot (baseline), got: {}",
        response.results[0].content
    );
}

// =============================================================================
// SNAPSHOT CAPTURE INTEGRATION TESTS (require fastembed model)
// =============================================================================

/// Test that embedding regeneration creates arc snapshots automatically.
#[tokio::test]
#[ignore = "Requires fastembed model download (run manually with --ignored)"]
async fn test_regeneration_creates_arc_snapshot() {
    let harness = TestHarness::new().await;
    let embedding_service =
        Arc::new(LocalEmbeddingService::new(EmbeddingConfig::default()).unwrap());
    let staleness_manager = Arc::new(StalenessManager::new(
        harness.db.clone(),
        embedding_service.clone(),
    ));

    let alice = create_character(
        &harness.db,
        CharacterCreate {
            name: "Alice the Explorer".into(),
            aliases: vec![],
            roles: vec!["protagonist".into()],
            profile: HashMap::from([
                ("wound".into(), vec!["Lost in the wilderness at age 12. The world is dangerous. Always overprepares.".into()]),
            ])
        },
    )
    .await
    .expect("Should create Alice");
    let alice_id = alice.id.to_string();

    // Generate initial embedding (should create snapshot)
    staleness_manager
        .regenerate_embedding(&alice_id, None)
        .await
        .expect("Should regenerate embedding");

    #[derive(serde::Deserialize)]
    struct CountResult {
        cnt: i64,
    }

    let mut resp = harness
        .db
        .query(format!(
            "SELECT count() AS cnt FROM arc_snapshot WHERE entity_id = {} GROUP ALL",
            alice_id
        ))
        .await
        .expect("Query should succeed");
    let count: Option<CountResult> = resp.take(0).unwrap_or(None);
    assert_eq!(
        count.unwrap().cnt,
        1,
        "First regeneration should create 1 snapshot"
    );

    // Update character and regenerate (should create second snapshot WITH delta)
    update_character(
        &harness.db,
        &alice.id.key().to_string(),
        CharacterUpdate {
            profile: Some(HashMap::from([
                ("wound".into(), vec!["Found inner peace through meditation at age 30. The world is full of wonder. Approaches challenges with calm confidence.".into()]),
            ])),
            updated_at: Datetime::default(),
            ..Default::default()
        },
    )
    .await
    .expect("Should update Alice");

    staleness_manager
        .regenerate_embedding(&alice_id, None)
        .await
        .expect("Should regenerate again");

    let mut resp = harness
        .db
        .query(format!(
            "SELECT count() AS cnt FROM arc_snapshot WHERE entity_id = {} GROUP ALL",
            alice_id
        ))
        .await
        .expect("Query should succeed");
    let count: Option<CountResult> = resp.take(0).unwrap_or(None);
    assert_eq!(
        count.unwrap().cnt,
        2,
        "Second regeneration should create 2nd snapshot"
    );

    // Verify second snapshot has delta > 0
    let mut resp = harness.db
        .query(format!(
            "SELECT delta_magnitude FROM arc_snapshot WHERE entity_id = {} AND delta_magnitude IS NOT NONE",
            alice_id
        ))
        .await.expect("Query should succeed");

    #[derive(serde::Deserialize)]
    struct DeltaResult {
        delta_magnitude: f32,
    }
    let deltas: Vec<DeltaResult> = resp.take(0).unwrap_or_default();
    assert!(
        !deltas.is_empty(),
        "Should have at least one snapshot with delta"
    );
    assert!(
        deltas[0].delta_magnitude > 0.0,
        "Delta should be positive for changed character, got {}",
        deltas[0].delta_magnitude
    );
}

/// Test that event_id is captured on snapshots when provided via RecordKnowledge.
#[tokio::test]
#[ignore = "Requires fastembed model download (run manually with --ignored)"]
async fn test_snapshot_captures_event_id() {
    let harness = TestHarness::new().await;
    let embedding_service =
        Arc::new(LocalEmbeddingService::new(EmbeddingConfig::default()).unwrap());
    let staleness_manager = Arc::new(StalenessManager::new(
        harness.db.clone(),
        embedding_service.clone(),
    ));
    let server = create_test_server_with_embeddings(&harness, embedding_service.clone()).await;

    let alice = create_character(
        &harness.db,
        CharacterCreate {
            name: "Alice".into(),
            aliases: vec![],
            roles: vec!["protagonist".into()],
            ..Default::default()
        },
    )
    .await
    .expect("Should create Alice");

    let event = create_event(
        &harness.db,
        EventCreate {
            title: "The Discovery".into(),
            description: Some("Alice learns a secret".into()),
            sequence: 1,
            date: None,
            date_precision: None,
            duration_end: None,
        },
    )
    .await
    .expect("Should create event");

    // Record knowledge with event_id (creates the knowledge entity + knows edge)
    let request = MutationRequest::RecordKnowledge {
        character_id: alice.id.to_string(),
        target_id: "character:alice".to_string(),
        fact: "The treasure is hidden beneath the old oak tree".to_string(),
        certainty: "knows".to_string(),
        method: Some("discovered".to_string()),
        source_character_id: None,
        event_id: Some(event.id.to_string()),
    };

    let _resp = server
        .handle_mutate(Parameters(request))
        .await
        .expect("RecordKnowledge should succeed");

    // Find the knowledge entity that was created
    #[derive(serde::Deserialize)]
    struct KnowledgeId {
        id: surrealdb::RecordId,
    }

    let mut k_resp = harness
        .db
        .query("SELECT id FROM knowledge LIMIT 1")
        .await
        .expect("Query knowledge");
    let knowledge_ids: Vec<KnowledgeId> = k_resp.take(0).unwrap_or_default();
    assert!(!knowledge_ids.is_empty(), "Knowledge entity should exist");
    let knowledge_id = knowledge_ids[0].id.to_string();

    // Regenerate with event_id using synchronous path
    // (spawn_regeneration uses tokio::spawn which may not complete in single-threaded test runtime)
    staleness_manager
        .regenerate_embedding(&knowledge_id, Some(event.id.to_string()))
        .await
        .expect("Should regenerate knowledge embedding");

    // Verify snapshots were created — use COUNT to avoid deserializing RecordId fields
    #[derive(serde::Deserialize)]
    struct CountResult {
        cnt: i64,
    }

    let mut resp = harness
        .db
        .query(format!(
            "SELECT count() AS cnt FROM arc_snapshot WHERE entity_id = {} GROUP ALL",
            knowledge_id
        ))
        .await
        .expect("Query should succeed");
    let count: Option<CountResult> = resp.take(0).unwrap_or(None);
    assert!(
        count.map(|c| c.cnt > 0).unwrap_or(false),
        "Should have at least one snapshot"
    );

    // Verify snapshot with event_id exists (count-based to avoid RecordId deserialization)
    let mut resp = harness.db
        .query(format!(
            "SELECT count() AS cnt FROM arc_snapshot WHERE entity_id = {} AND event_id IS NOT NONE GROUP ALL",
            knowledge_id
        ))
        .await.expect("Query should succeed");
    let count: Option<CountResult> = resp.take(0).unwrap_or(None);
    assert!(
        count.map(|c| c.cnt > 0).unwrap_or(false),
        "Should have at least one snapshot with event_id"
    );

    // Verify entity_type is correct (string field, safe to deserialize)
    #[derive(serde::Deserialize)]
    struct TypeCheck {
        entity_type: String,
    }

    let mut resp = harness
        .db
        .query(format!(
            "SELECT entity_type FROM arc_snapshot WHERE entity_id = {} LIMIT 1",
            knowledge_id
        ))
        .await
        .expect("Query should succeed");
    let types: Vec<TypeCheck> = resp.take(0).unwrap_or_default();
    assert!(!types.is_empty());
    assert_eq!(types[0].entity_type, "knowledge");

    // Verify the embedding was stored on the knowledge entity
    #[derive(serde::Deserialize)]
    struct EmbeddingCheck {
        embedding: Option<Vec<f32>>,
    }
    let mut resp = harness
        .db
        .query(format!("SELECT embedding FROM {}", knowledge_id))
        .await
        .expect("Query embedding");
    let emb: Vec<EmbeddingCheck> = resp.take(0).unwrap_or_default();
    assert!(
        emb[0].embedding.is_some(),
        "Knowledge should have embedding after regeneration"
    );
}

/// Test end-to-end arc tracking workflow with real embeddings.
#[tokio::test]
#[ignore = "Requires fastembed model download (run manually with --ignored)"]
async fn test_full_arc_tracking_workflow() {
    let harness = TestHarness::new().await;
    let embedding_service =
        Arc::new(LocalEmbeddingService::new(EmbeddingConfig::default()).unwrap());
    let staleness_manager = Arc::new(StalenessManager::new(
        harness.db.clone(),
        embedding_service.clone(),
    ));
    let backfill_service = BackfillService::new(harness.db.clone(), embedding_service.clone());
    let server = create_test_server_with_embeddings(&harness, embedding_service.clone()).await;

    let alice = create_character(
        &harness.db,
        CharacterCreate {
            name: "Alice".into(),
            aliases: vec![],
            roles: vec!["protagonist".into()],
            profile: HashMap::from([
                ("wound".into(), vec!["Betrayed by her mentor at age 18. Authority figures cannot be trusted. Questions every order.".into()]),
            ]),
        },
    )
    .await
    .expect("Should create Alice");

    let bob = create_character(
        &harness.db,
        CharacterCreate {
            name: "Bob".into(),
            aliases: vec![],
            roles: vec!["companion".into()],
            profile: HashMap::from([
                ("wound".into(), vec!["Lost his family in a fire at age 10. You can rebuild anything. Never gives up.".into()]),
            ])
        },
    )
    .await
    .expect("Should create Bob");

    let alice_id = alice.id.to_string();
    let bob_id = bob.id.to_string();

    // Step 1: Backfill embeddings
    let stats = backfill_service
        .backfill_all()
        .await
        .expect("Backfill should succeed");
    assert!(stats.embedded >= 2, "Should embed at least 2 characters");

    // Step 2: Capture baseline snapshots
    let baseline_request = MutationRequest::BaselineArcSnapshots { entity_type: None };
    let baseline_resp = server
        .handle_mutate(Parameters(baseline_request))
        .await
        .expect("Baseline should succeed");
    assert!(
        baseline_resp.entity.content.contains("2 created"),
        "Should create 2 baselines, got: {}",
        baseline_resp.entity.content
    );

    // Step 3: Update Alice (character growth)
    update_character(
        &harness.db,
        &alice.id.key().to_string(),
        CharacterUpdate {
            profile: Some(HashMap::from([
                ("wound".into(), vec!["Learned to trust again at age 25. Some people are worthy of trust. Cautiously extends trust.".into()]),
            ])),
            updated_at: Datetime::default(),
            ..Default::default()
        },
    )
    .await
    .expect("Should update Alice");
    staleness_manager
        .regenerate_embedding(&alice_id, None)
        .await
        .expect("Should regenerate");

    // Step 4: Query all arc operations

    // ArcHistory
    let resp = server
        .handle_query(Parameters(QueryRequest::ArcHistory {
            entity_id: alice_id.clone(),
            limit: None,
        }))
        .await
        .expect("ArcHistory should succeed");
    assert_eq!(
        resp.results.len(),
        2,
        "Should have baseline + updated snapshot"
    );

    // ArcComparison
    let resp = server
        .handle_query(Parameters(QueryRequest::ArcComparison {
            entity_id_a: alice_id.clone(),
            entity_id_b: bob_id.clone(),
            window: None,
        }))
        .await
        .expect("ArcComparison should succeed");
    assert_eq!(resp.results.len(), 1);

    // ArcDrift
    let resp = server
        .handle_query(Parameters(QueryRequest::ArcDrift {
            entity_type: Some("character".to_string()),
            limit: None,
        }))
        .await
        .expect("ArcDrift should succeed");
    assert!(
        !resp.results.is_empty(),
        "Should have drift results for Alice"
    );

    // ArcMoment
    let resp = server
        .handle_query(Parameters(QueryRequest::ArcMoment {
            entity_id: alice_id.clone(),
            event_id: None,
        }))
        .await
        .expect("ArcMoment should succeed");
    assert_eq!(resp.results.len(), 1);

    println!("Full arc tracking workflow completed successfully");
}
