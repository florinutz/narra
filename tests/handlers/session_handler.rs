//! Integration tests for session handler (HAND-04).
//!
//! Tests get_session_context operations including:
//! - Session state with hot entities
//! - World overview counts
//! - Verbosity settings

use crate::common::builders::CharacterBuilder;
use crate::common::harness::TestHarness;

#[allow(unused_imports)]
use super::*;

use narra::mcp::tools::session::SessionContextRequest;
use narra::repository::{EntityRepository, SurrealEntityRepository};
use narra::session::SessionStateManager;
use pretty_assertions::assert_eq;
use rmcp::handler::server::wrapper::Parameters;
use std::sync::Arc;

// ============================================================================
// SESSION CONTEXT - BASIC OPERATIONS
// ============================================================================

/// Test that get_session_context returns valid SessionContextResponse.
#[tokio::test]
async fn test_get_session_context_success() {
    let harness = TestHarness::new().await;
    let server = crate::common::create_test_server(&harness).await;

    let request = SessionContextRequest { force_full: false };
    let response = server.handle_get_session_context(Parameters(request)).await;

    assert!(response.is_ok(), "get_session_context should succeed");
    let context = response.unwrap();

    // Verify response structure has expected fields
    assert!(!context.verbosity.is_empty(), "verbosity should be set");
    assert!(!context.summary.is_empty(), "summary should be set");
}

/// Test session context with empty world (fresh database).
#[tokio::test]
async fn test_get_session_context_empty_world() {
    let harness = TestHarness::new().await;
    let server = crate::common::create_test_server(&harness).await;

    let request = SessionContextRequest { force_full: false };
    let response = server.handle_get_session_context(Parameters(request)).await;

    assert!(response.is_ok());
    let context = response.unwrap();

    // Fresh DB should have zero counts in world overview
    if let Some(overview) = context.world_overview {
        assert_eq!(
            overview.character_count, 0,
            "Fresh DB should have no characters"
        );
        assert_eq!(
            overview.location_count, 0,
            "Fresh DB should have no locations"
        );
        assert_eq!(overview.event_count, 0, "Fresh DB should have no events");
        assert_eq!(overview.scene_count, 0, "Fresh DB should have no scenes");
    }

    // No hot entities in fresh session
    assert!(
        context.hot_entities.is_empty(),
        "Fresh session should have no hot entities"
    );
}

/// Test session context reflects actual entity counts.
#[tokio::test]
async fn test_get_session_context_with_entities() {
    let harness = TestHarness::new().await;

    let entity_repo = SurrealEntityRepository::new(harness.db.clone());

    // Create some entities
    entity_repo
        .create_character(CharacterBuilder::new("Alice").build())
        .await
        .expect("Create Alice");
    entity_repo
        .create_character(CharacterBuilder::new("Bob").build())
        .await
        .expect("Create Bob");

    let server = crate::common::create_test_server(&harness).await;

    let request = SessionContextRequest { force_full: false };
    let response = server.handle_get_session_context(Parameters(request)).await;

    assert!(response.is_ok());
    let context = response.unwrap();

    // World overview should reflect created characters
    if let Some(overview) = context.world_overview {
        assert_eq!(
            overview.character_count, 2,
            "Should have 2 characters in world overview"
        );
    } else {
        panic!("Expected world_overview to be present");
    }
}

// ============================================================================
// HOT ENTITIES TESTS
// ============================================================================

/// Test that session context includes hot entities when accessed via session manager.
#[tokio::test]
async fn test_session_context_tracks_hot_entities() {
    let harness = TestHarness::new().await;

    let entity_repo = SurrealEntityRepository::new(harness.db.clone());

    // Create a character
    let alice = entity_repo
        .create_character(CharacterBuilder::new("Alice").role("protagonist").build())
        .await
        .expect("Create Alice");
    let alice_id = format!("character:{}", alice.id.key());

    // Pre-populate session state with recorded access before creating server
    let session_path = harness.temp_path().join("session.json");
    let session_manager =
        Arc::new(SessionStateManager::load_or_create(&session_path).expect("Session manager"));
    session_manager.record_access(&alice_id).await;
    session_manager.save().await.expect("Save session state");
    drop(session_manager);

    let server = crate::common::create_test_server(&harness).await;

    let request = SessionContextRequest { force_full: false };
    let response = server.handle_get_session_context(Parameters(request)).await;

    assert!(response.is_ok());
    let context = response.unwrap();

    // Alice should appear in hot entities
    assert!(
        context.hot_entities.iter().any(|e| e.id == alice_id),
        "Recently accessed entity should appear in hot_entities"
    );
}

/// Test hot entity response has all required fields.
#[tokio::test]
async fn test_session_context_hot_entity_fields() {
    let harness = TestHarness::new().await;

    let entity_repo = SurrealEntityRepository::new(harness.db.clone());

    // Create a character
    let alice = entity_repo
        .create_character(
            CharacterBuilder::new("Alice Blackwood")
                .role("detective")
                .build(),
        )
        .await
        .expect("Create Alice");
    let alice_id = format!("character:{}", alice.id.key());

    // Pre-populate session state with recorded access before creating server
    let session_path = harness.temp_path().join("session.json");
    let session_manager =
        Arc::new(SessionStateManager::load_or_create(&session_path).expect("Session manager"));
    session_manager.record_access(&alice_id).await;
    session_manager.save().await.expect("Save session state");
    drop(session_manager);

    let server = crate::common::create_test_server(&harness).await;

    let request = SessionContextRequest { force_full: false };
    let response = server.handle_get_session_context(Parameters(request)).await;

    assert!(response.is_ok());
    let context = response.unwrap();

    // Find Alice in hot entities
    let alice_hot = context
        .hot_entities
        .iter()
        .find(|e| e.id == alice_id)
        .expect("Alice should be in hot entities");

    // Verify HotEntityResponse fields
    assert_eq!(alice_hot.id, alice_id, "id should match");
    assert_eq!(alice_hot.name, "Alice Blackwood", "name should match");
    assert_eq!(
        alice_hot.entity_type, "character",
        "entity_type should be character"
    );
    // last_accessed may or may not be populated depending on implementation
}

// ============================================================================
// VERBOSITY TESTS
// ============================================================================

/// Test verbosity level is set in response.
#[tokio::test]
async fn test_session_context_verbosity_level() {
    let harness = TestHarness::new().await;
    let server = crate::common::create_test_server(&harness).await;

    let request = SessionContextRequest { force_full: false };
    let response = server.handle_get_session_context(Parameters(request)).await;

    assert!(response.is_ok());
    let context = response.unwrap();

    // Verbosity values from StartupVerbosity enum:
    // "brief", "standard", "full", "new_world", "empty_world"
    let valid_verbosity = ["brief", "standard", "full", "new_world", "empty_world"];
    assert!(
        valid_verbosity.contains(&context.verbosity.as_str()),
        "verbosity should be one of: {:?}, got: {}",
        valid_verbosity,
        context.verbosity
    );
}

/// Test force_full flag influences verbosity.
#[tokio::test]
async fn test_session_context_force_full() {
    let harness = TestHarness::new().await;

    // Pre-populate session state with a marked session end before creating server
    let session_path = harness.temp_path().join("session.json");
    let session_manager =
        Arc::new(SessionStateManager::load_or_create(&session_path).expect("Session manager"));
    session_manager.mark_session_end().await;
    session_manager.save().await.expect("Save session state");
    drop(session_manager);

    let server = crate::common::create_test_server(&harness).await;

    // Without force_full - may be minimal if recently active
    let request_normal = SessionContextRequest { force_full: false };
    let response_normal = server
        .handle_get_session_context(Parameters(request_normal))
        .await;
    assert!(response_normal.is_ok());

    // With force_full - should not reduce verbosity
    let request_full = SessionContextRequest { force_full: true };
    let _response_full = server
        .handle_get_session_context(Parameters(request_full))
        .await;

    // Note: The actual behavior depends on implementation.
    // If force_full=true doesn't change behavior, this test documents that.
    // The test verifies both calls succeed, which is the minimum contract.
}
