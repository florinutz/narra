//! Handler tests for perception analytics operations.
//!
//! Tests PerceptionGap, PerceptionMatrix, and PerceptionShift queries
//! added in Session 3 (perception/arc CLI commands).

use narra::mcp::QueryRequest;
use narra::models::perception::{create_perception, PerceptionCreate};
use narra::repository::{EntityRepository, SurrealEntityRepository};
use rmcp::handler::server::wrapper::Parameters;

use crate::common::builders::CharacterBuilder;
use crate::common::harness::TestHarness;

/// Helper to create a character pair with perceptions.
async fn setup_perception_pair(harness: &TestHarness) -> (String, String) {
    let repo = SurrealEntityRepository::new(harness.db.clone());

    let alice = repo
        .create_character(CharacterBuilder::new("Alice").role("protagonist").build())
        .await
        .expect("Alice");
    let bob = repo
        .create_character(CharacterBuilder::new("Bob").role("antagonist").build())
        .await
        .expect("Bob");

    // Alice perceives Bob
    create_perception(
        &harness.db,
        &alice.id.key().to_string(),
        &bob.id.key().to_string(),
        PerceptionCreate {
            rel_types: vec!["rival".to_string()],
            subtype: None,
            feelings: Some("distrustful and wary".to_string()),
            perception: Some("sees Bob as dangerous and cunning".to_string()),
            tension_level: Some(7),
            history_notes: None,
        },
    )
    .await
    .expect("Alice->Bob perception");

    // Bob perceives Alice
    create_perception(
        &harness.db,
        &bob.id.key().to_string(),
        &alice.id.key().to_string(),
        PerceptionCreate {
            rel_types: vec!["rival".to_string()],
            subtype: None,
            feelings: Some("respectful but threatened".to_string()),
            perception: Some("sees Alice as noble but naive".to_string()),
            tension_level: Some(5),
            history_notes: None,
        },
    )
    .await
    .expect("Bob->Alice perception");

    let alice_id = format!("character:{}", alice.id.key());
    let bob_id = format!("character:{}", bob.id.key());

    (alice_id, bob_id)
}

// =============================================================================
// PERCEPTION GAP TESTS
// =============================================================================

/// Test perception gap requires perception edges to exist.
///
/// Without perspective embeddings, the handler returns a not-found error.
#[tokio::test]
async fn test_perception_gap_needs_embeddings() {
    let harness = TestHarness::new().await;
    let (alice_id, bob_id) = setup_perception_pair(&harness).await;

    let server = crate::common::create_test_server(&harness).await;

    let request = QueryRequest::PerceptionGap {
        observer_id: alice_id.clone(),
        target_id: bob_id.clone(),
    };

    let response = server.handle_query(Parameters(request)).await;

    // With NoopEmbedding, no perspective embeddings exist, so we get an error
    assert!(
        response.is_err(),
        "PerceptionGap should error without perspective embeddings"
    );
    let err = response.unwrap_err();
    assert!(
        err.contains("perceives") || err.contains("Not found"),
        "Error should reference missing perception, got: {}",
        err
    );
}

/// Test perception gap when no perception edge exists at all.
#[tokio::test]
async fn test_perception_gap_no_perception() {
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

    let request = QueryRequest::PerceptionGap {
        observer_id: format!("character:{}", alice.id.key()),
        target_id: format!("character:{}", bob.id.key()),
    };

    let response = server.handle_query(Parameters(request)).await;

    // Should error with no perception
    assert!(
        response.is_err(),
        "Should error without any perception data"
    );
}

// =============================================================================
// PERCEPTION MATRIX TESTS
// =============================================================================

/// Test perception matrix requires perspective embeddings.
///
/// Without embeddings, PerceptionMatrix errors because it needs perspective
/// vectors to compute gap metrics.
#[tokio::test]
async fn test_perception_matrix_needs_perspectives() {
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

    // Bob perceives Alice (creates edge, but no perspective embedding with Noop)
    create_perception(
        &harness.db,
        &bob.id.key().to_string(),
        &alice.id.key().to_string(),
        PerceptionCreate {
            rel_types: vec!["ally".to_string()],
            subtype: None,
            feelings: Some("admiring".to_string()),
            perception: Some("sees Alice as a leader".to_string()),
            tension_level: Some(2),
            history_notes: None,
        },
    )
    .await
    .expect("Bob->Alice");

    let server = crate::common::create_test_server(&harness).await;

    let request = QueryRequest::PerceptionMatrix {
        target_id: format!("character:{}", alice.id.key()),
        limit: Some(20),
    };

    let response = server.handle_query(Parameters(request)).await;

    // Without perspective embeddings, this errors
    assert!(
        response.is_err(),
        "PerceptionMatrix should error without perspective embeddings"
    );
}

/// Test perception matrix with no observers at all.
#[tokio::test]
async fn test_perception_matrix_no_observers() {
    let harness = TestHarness::new().await;
    let repo = SurrealEntityRepository::new(harness.db.clone());

    let alice = repo
        .create_character(CharacterBuilder::new("Alice").build())
        .await
        .expect("Alice");

    let server = crate::common::create_test_server(&harness).await;

    let request = QueryRequest::PerceptionMatrix {
        target_id: format!("character:{}", alice.id.key()),
        limit: Some(20),
    };

    let response = server.handle_query(Parameters(request)).await;

    // No observers = error about missing perspectives
    assert!(response.is_err(), "Should error with no observers");
}

// =============================================================================
// PERCEPTION SHIFT TESTS
// =============================================================================

/// Test perception shift requires perspective snapshot data.
#[tokio::test]
async fn test_perception_shift_needs_snapshots() {
    let harness = TestHarness::new().await;
    let (alice_id, bob_id) = setup_perception_pair(&harness).await;

    let server = crate::common::create_test_server(&harness).await;

    let request = QueryRequest::PerceptionShift {
        observer_id: alice_id,
        target_id: bob_id,
    };

    let response = server.handle_query(Parameters(request)).await;

    // Without perspective snapshots, should error
    assert!(
        response.is_err(),
        "PerceptionShift should error without perspective snapshots"
    );
}
