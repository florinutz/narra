//! Cross-cutting validation tests for MCP handlers.
//!
//! Tests handlers correctly reject invalid input across all operations.
//! Uses insta snapshots for error message stability.

use crate::common::builders::CharacterBuilder;
use crate::common::harness::TestHarness;

use insta::assert_snapshot;
use narra::embedding::{EmbeddingService, NoopEmbeddingService};
use narra::mcp::types::{MutationRequest, QueryRequest};
use narra::mcp::NarraServer;
use narra::repository::{EntityRepository, SurrealEntityRepository};
use narra::session::SessionStateManager;
use rmcp::handler::server::wrapper::Parameters;
use std::sync::Arc;
use tempfile::TempDir;

// ============================================================================
// MISSING REQUIRED FIELDS TESTS
// ============================================================================

/// Test QueryRequest::Lookup deserialization fails without entity_id.
#[test]
fn test_query_lookup_missing_entity_id() {
    let json_request = serde_json::json!({
        "operation": "lookup",
        // entity_id is required but missing
        "detail_level": "standard"
    });

    let result: Result<QueryRequest, _> = serde_json::from_value(json_request);

    assert!(
        result.is_err(),
        "Deserialization should fail for missing entity_id"
    );
    let error_msg = result.unwrap_err().to_string();
    assert_snapshot!("query_lookup_missing_entity_id", error_msg);
}

/// Test MutationRequest::CreateCharacter deserialization fails without name.
#[test]
fn test_mutation_create_character_missing_name() {
    let json_request = serde_json::json!({
        "operation": "create_character",
        // name is required but missing
        "role": "protagonist"
    });

    let result: Result<MutationRequest, _> = serde_json::from_value(json_request);

    assert!(
        result.is_err(),
        "Deserialization should fail for missing name"
    );
    let error_msg = result.unwrap_err().to_string();
    assert_snapshot!("mutation_create_character_missing_name", error_msg);
}

/// Test MutationRequest::CreateScene deserialization fails without event_id.
#[test]
fn test_mutation_create_scene_missing_event_id() {
    let json_request = serde_json::json!({
        "operation": "create_scene",
        "title": "Opening Scene",
        // event_id is required but missing
        "location_id": "location:abc123"
    });

    let result: Result<MutationRequest, _> = serde_json::from_value(json_request);

    assert!(
        result.is_err(),
        "Deserialization should fail for missing event_id"
    );
    let error_msg = result.unwrap_err().to_string();
    assert_snapshot!("mutation_create_scene_missing_event_id", error_msg);
}

// ============================================================================
// WRONG TYPE TESTS
// ============================================================================

/// Test QueryRequest::Lookup with entity_id as number instead of string.
#[test]
fn test_query_lookup_wrong_type_entity_id() {
    let json_request = serde_json::json!({
        "operation": "lookup",
        "entity_id": 12345,  // Should be string, not number
        "detail_level": "standard"
    });

    let result: Result<QueryRequest, _> = serde_json::from_value(json_request);

    assert!(
        result.is_err(),
        "Deserialization should fail for wrong type entity_id"
    );
    let error_msg = result.unwrap_err().to_string();
    assert_snapshot!("query_lookup_wrong_type_entity_id", error_msg);
}

/// Test MutationRequest::CreateEvent with sequence as string instead of number.
#[test]
fn test_mutation_create_event_wrong_type_sequence() {
    let json_request = serde_json::json!({
        "operation": "create_event",
        "title": "The Betrayal",
        "sequence": "not-a-number"  // Should be i32, not string
    });

    let result: Result<MutationRequest, _> = serde_json::from_value(json_request);

    assert!(
        result.is_err(),
        "Deserialization should fail for wrong type sequence"
    );
    let error_msg = result.unwrap_err().to_string();
    assert_snapshot!("mutation_create_event_wrong_type_sequence", error_msg);
}

// ============================================================================
// INVALID ID FORMAT TESTS
// ============================================================================

/// Test handler rejects malformed entity ID format.
#[tokio::test]
async fn test_query_lookup_malformed_id() {
    let harness = TestHarness::new().await;
    let temp_dir = TempDir::new().expect("Temp dir");
    let session_path = temp_dir.path().join("session.json");

    let session_manager =
        Arc::new(SessionStateManager::load_or_create(&session_path).expect("Session manager"));
    let server = NarraServer::new(
        harness.db.clone(),
        session_manager,
        Arc::new(NoopEmbeddingService::new()),
    )
    .await;

    // Malformed ID (not table:key format)
    let request = QueryRequest::Lookup {
        entity_id: "not-a-valid-record-id".to_string(),
        detail_level: None,
    };

    let response = server.handle_query(Parameters(request)).await;

    assert!(response.is_err(), "Should reject malformed entity_id");
    let error_msg = response.unwrap_err();
    assert_snapshot!("query_lookup_malformed_id", error_msg);
}

/// Test MutationRequest::Update with malformed entity_id.
#[tokio::test]
async fn test_mutation_update_malformed_entity_id() {
    let harness = TestHarness::new().await;
    let temp_dir = TempDir::new().expect("Temp dir");
    let session_path = temp_dir.path().join("session.json");

    let session_manager =
        Arc::new(SessionStateManager::load_or_create(&session_path).expect("Session manager"));
    let server = NarraServer::new(
        harness.db.clone(),
        session_manager,
        Arc::new(NoopEmbeddingService::new()),
    )
    .await;

    let request = MutationRequest::Update {
        entity_id: "###invalid###".to_string(),
        fields: serde_json::json!({"name": "New Name"}),
    };

    let response = server.handle_mutate(Parameters(request)).await;

    assert!(
        response.is_err(),
        "Should reject malformed entity_id in Update"
    );
    let error_msg = response.unwrap_err();
    assert_snapshot!("mutation_update_malformed_entity_id", error_msg);
}

/// Test MutationRequest::RecordKnowledge with malformed character_id.
#[tokio::test]
async fn test_mutation_record_knowledge_malformed_character_id() {
    let harness = TestHarness::new().await;
    let temp_dir = TempDir::new().expect("Temp dir");
    let session_path = temp_dir.path().join("session.json");

    let session_manager =
        Arc::new(SessionStateManager::load_or_create(&session_path).expect("Session manager"));
    let server = NarraServer::new(
        harness.db.clone(),
        session_manager,
        Arc::new(NoopEmbeddingService::new()),
    )
    .await;

    let request = MutationRequest::RecordKnowledge {
        character_id: "bad-id-format".to_string(),
        target_id: "character:bob".to_string(),
        fact: "Knows something".to_string(),
        certainty: "believes".to_string(),
        method: None,
        source_character_id: None,
        event_id: None,
    };

    let response = server.handle_mutate(Parameters(request)).await;

    assert!(
        response.is_err(),
        "Should reject malformed character_id in RecordKnowledge"
    );
    let error_msg = response.unwrap_err();
    assert_snapshot!(
        "mutation_record_knowledge_malformed_character_id",
        error_msg
    );
}

// ============================================================================
// VALID-FORMAT IDs THAT DON'T EXIST
// ============================================================================

/// Test lookup with valid-format ID that doesn't exist in database.
#[tokio::test]
async fn test_query_lookup_nonexistent_entity() {
    let harness = TestHarness::new().await;
    let temp_dir = TempDir::new().expect("Temp dir");
    let session_path = temp_dir.path().join("session.json");

    let session_manager =
        Arc::new(SessionStateManager::load_or_create(&session_path).expect("Session manager"));
    let server = NarraServer::new(
        harness.db.clone(),
        session_manager,
        Arc::new(NoopEmbeddingService::new()),
    )
    .await;

    // Valid format but doesn't exist
    let request = QueryRequest::Lookup {
        entity_id: "character:nonexistent_abc123".to_string(),
        detail_level: None,
    };

    let response = server.handle_query(Parameters(request)).await;

    assert!(response.is_err(), "Should reject non-existent entity");
    let error_msg = response.unwrap_err();
    assert_snapshot!("query_lookup_nonexistent_entity", error_msg);
}

/// Test delete with valid-format ID that doesn't exist.
#[tokio::test]
async fn test_mutation_delete_nonexistent_entity() {
    let harness = TestHarness::new().await;
    let temp_dir = TempDir::new().expect("Temp dir");
    let session_path = temp_dir.path().join("session.json");

    let session_manager =
        Arc::new(SessionStateManager::load_or_create(&session_path).expect("Session manager"));
    let server = NarraServer::new(
        harness.db.clone(),
        session_manager,
        Arc::new(NoopEmbeddingService::new()),
    )
    .await;

    // Valid format but doesn't exist
    let request = MutationRequest::Delete {
        entity_id: "character:nonexistent_xyz789".to_string(),
        hard: Some(false),
    };

    let response = server.handle_mutate(Parameters(request)).await;

    assert!(
        response.is_err(),
        "Should reject deletion of non-existent entity"
    );
    let error_msg = response.unwrap_err();
    assert_snapshot!("mutation_delete_nonexistent_entity", error_msg);
}

// ============================================================================
// OUT-OF-RANGE VALUE TESTS
// ============================================================================

/// Test search with negative limit value.
#[tokio::test]
async fn test_query_search_negative_limit() {
    let harness = TestHarness::new().await;
    let temp_dir = TempDir::new().expect("Temp dir");
    let session_path = temp_dir.path().join("session.json");

    let session_manager =
        Arc::new(SessionStateManager::load_or_create(&session_path).expect("Session manager"));
    let server = NarraServer::new(
        harness.db.clone(),
        session_manager,
        Arc::new(NoopEmbeddingService::new()),
    )
    .await;

    // Note: limit is usize, so negative values in JSON will fail deserialization.
    // This tests the handler's behavior with boundary values.
    let request = QueryRequest::Search {
        query: "test".to_string(),
        entity_types: None,
        limit: Some(0), // Edge case: zero limit
        cursor: None,
    };

    let response = server.handle_query(Parameters(request)).await;

    // Zero limit might be valid (return empty results) or error
    // This test documents actual behavior
    if response.is_err() {
        let error_msg = response.unwrap_err();
        assert_snapshot!("query_search_zero_limit_error", error_msg);
    } else {
        // If it succeeds, verify empty results
        let result = response.unwrap();
        assert!(
            result.results.is_empty(),
            "Zero limit should return empty results"
        );
    }
}

/// Test graph traversal with zero depth (edge case).
#[tokio::test]
async fn test_query_graph_traversal_zero_depth() {
    let harness = TestHarness::new().await;
    let temp_dir = TempDir::new().expect("Temp dir");
    let session_path = temp_dir.path().join("session.json");

    let entity_repo = SurrealEntityRepository::new(harness.db.clone());

    // Create a character to traverse from
    let alice = entity_repo
        .create_character(CharacterBuilder::new("Alice").build())
        .await
        .expect("Create Alice");
    let alice_id = format!("character:{}", alice.id.key());

    let session_manager =
        Arc::new(SessionStateManager::load_or_create(&session_path).expect("Session manager"));
    let server = NarraServer::new(
        harness.db.clone(),
        session_manager,
        Arc::new(NoopEmbeddingService::new()),
    )
    .await;

    // Zero depth - edge case
    let request = QueryRequest::GraphTraversal {
        entity_id: alice_id.clone(),
        depth: 0,
        format: None,
    };

    let response = server.handle_query(Parameters(request)).await;

    // Zero depth might return just the starting entity or error
    // This test documents actual behavior
    if response.is_err() {
        let error_msg = response.unwrap_err();
        assert_snapshot!("query_graph_traversal_zero_depth_error", error_msg);
    } else {
        let result = response.unwrap();
        // With zero depth, should only return the starting entity or empty
        assert!(
            result.results.len() <= 1,
            "Zero depth should return at most the starting entity"
        );
    }
}
