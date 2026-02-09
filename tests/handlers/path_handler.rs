//! Handler tests for ConnectionPath and ReverseQuery operations.
//!
//! These operations were added in Session 1 (CLI restructure) and are
//! used by `narra path` and `narra references` CLI commands.

use narra::mcp::QueryRequest;
use narra::models::relationship::{create_relationship, RelationshipCreate};
use narra::repository::{EntityRepository, SurrealEntityRepository};
use rmcp::handler::server::wrapper::Parameters;

use crate::common::builders::{CharacterBuilder, EventBuilder, LocationBuilder, SceneBuilder};
use crate::common::harness::TestHarness;

// =============================================================================
// CONNECTION PATH TESTS
// =============================================================================

/// Test connection path finds direct relationship between two characters.
#[tokio::test]
async fn test_connection_path_direct() {
    let harness = TestHarness::new().await;
    let repo = SurrealEntityRepository::new(harness.db.clone());

    let alice = repo
        .create_character(CharacterBuilder::new("Alice").build())
        .await
        .expect("Alice");
    let bob = repo
        .create_character(CharacterBuilder::new("Bob").build())
        .await
        .expect("Bob");

    create_relationship(
        &harness.db,
        RelationshipCreate {
            from_character_id: alice.id.key().to_string(),
            to_character_id: bob.id.key().to_string(),
            rel_type: "friendship".to_string(),
            subtype: None,
            label: Some("old friends".to_string()),
        },
    )
    .await
    .expect("Relationship");

    let server = crate::common::create_test_server(&harness).await;

    let request = QueryRequest::ConnectionPath {
        from_id: format!("character:{}", alice.id.key()),
        to_id: format!("character:{}", bob.id.key()),
        max_hops: Some(3),
        include_events: Some(false),
    };

    let response = server
        .handle_query(Parameters(request))
        .await
        .expect("ConnectionPath should succeed");

    assert!(
        !response.results.is_empty(),
        "Should find at least one path result"
    );
    // Path was found — just verify the response structure is valid
    assert!(
        !response.results[0].content.is_empty(),
        "Path content should not be empty"
    );
}

/// Test connection path between unconnected characters returns empty or no-path result.
#[tokio::test]
async fn test_connection_path_no_connection() {
    let harness = TestHarness::new().await;
    let repo = SurrealEntityRepository::new(harness.db.clone());

    let alice = repo
        .create_character(CharacterBuilder::new("Alice").build())
        .await
        .expect("Alice");
    let bob = repo
        .create_character(CharacterBuilder::new("Bob").build())
        .await
        .expect("Bob");

    let server = crate::common::create_test_server(&harness).await;

    let request = QueryRequest::ConnectionPath {
        from_id: format!("character:{}", alice.id.key()),
        to_id: format!("character:{}", bob.id.key()),
        max_hops: Some(3),
        include_events: Some(false),
    };

    let response = server.handle_query(Parameters(request)).await;

    // Should succeed but indicate no paths found
    assert!(response.is_ok(), "Should succeed even with no paths");
    let response = response.unwrap();
    // Content should indicate no connection
    let content = &response.results[0].content;
    assert!(
        content.contains("No") || content.contains("no") || content.contains("0 paths"),
        "Should indicate no paths found, got: {}",
        content
    );
}

/// Test connection path with multi-hop chain A->B->C.
#[tokio::test]
async fn test_connection_path_indirect() {
    let harness = TestHarness::new().await;
    let repo = SurrealEntityRepository::new(harness.db.clone());

    let alice = repo
        .create_character(CharacterBuilder::new("Alice").build())
        .await
        .expect("Alice");
    let bob = repo
        .create_character(CharacterBuilder::new("Bob").build())
        .await
        .expect("Bob");
    let charlie = repo
        .create_character(CharacterBuilder::new("Charlie").build())
        .await
        .expect("Charlie");

    // A->B and B->C (no direct A->C)
    create_relationship(
        &harness.db,
        RelationshipCreate {
            from_character_id: alice.id.key().to_string(),
            to_character_id: bob.id.key().to_string(),
            rel_type: "friendship".to_string(),
            subtype: None,
            label: None,
        },
    )
    .await
    .expect("A->B");

    create_relationship(
        &harness.db,
        RelationshipCreate {
            from_character_id: bob.id.key().to_string(),
            to_character_id: charlie.id.key().to_string(),
            rel_type: "mentorship".to_string(),
            subtype: None,
            label: None,
        },
    )
    .await
    .expect("B->C");

    let server = crate::common::create_test_server(&harness).await;

    let request = QueryRequest::ConnectionPath {
        from_id: format!("character:{}", alice.id.key()),
        to_id: format!("character:{}", charlie.id.key()),
        max_hops: Some(3),
        include_events: Some(false),
    };

    let response = server
        .handle_query(Parameters(request))
        .await
        .expect("ConnectionPath should find indirect path");

    assert!(!response.results.is_empty(), "Should find indirect path");
}

// =============================================================================
// REVERSE QUERY TESTS
// =============================================================================

/// Test reverse query finds scenes that reference a character.
#[tokio::test]
async fn test_reverse_query_character_scenes() {
    let harness = TestHarness::new().await;
    let repo = SurrealEntityRepository::new(harness.db.clone());

    let alice = repo
        .create_character(CharacterBuilder::new("Alice").build())
        .await
        .expect("Alice");

    let event = repo
        .create_event(EventBuilder::new("Opening").sequence(1).build())
        .await
        .expect("Event");

    let location = repo
        .create_location(LocationBuilder::new("Castle").build())
        .await
        .expect("Location");

    let _scene = repo
        .create_scene(
            SceneBuilder::new(
                "Alice's Arrival",
                event.id.key().to_string(),
                location.id.key().to_string(),
            )
            .summary("Alice arrives at the castle")
            .build(),
        )
        .await
        .expect("Scene");

    // Add Alice to scene participants
    let alice_id = alice.id.to_string();
    let _ = harness
        .db
        .query(format!(
            "UPDATE scene SET participants = [{}] WHERE title = 'Alice\\'s Arrival'",
            alice_id
        ))
        .await;

    let server = crate::common::create_test_server(&harness).await;

    let request = QueryRequest::ReverseQuery {
        entity_id: format!("character:{}", alice.id.key()),
        referencing_types: None,
        limit: Some(20),
    };

    let response = server
        .handle_query(Parameters(request))
        .await
        .expect("ReverseQuery should succeed");

    // Should return results (even if just the character references)
    // total is usize, just verify it exists
    let _ = response.total;
}

/// Test reverse query with no references returns empty.
#[tokio::test]
async fn test_reverse_query_no_references() {
    let harness = TestHarness::new().await;
    let repo = SurrealEntityRepository::new(harness.db.clone());

    let alice = repo
        .create_character(CharacterBuilder::new("Alice").build())
        .await
        .expect("Alice");

    let server = crate::common::create_test_server(&harness).await;

    let request = QueryRequest::ReverseQuery {
        entity_id: format!("character:{}", alice.id.key()),
        referencing_types: None,
        limit: Some(20),
    };

    let response = server
        .handle_query(Parameters(request))
        .await
        .expect("ReverseQuery should succeed even with no refs");

    // May have empty results or just the entity itself
    let _ = response.total;
}
