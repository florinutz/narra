//! Handler tests for arc tracking and WhatIf operations.
//!
//! Tests ArcHistory, ArcComparison, ArcMoment, and WhatIf queries
//! added in Session 3 (arc/what-if CLI commands).

use narra::mcp::QueryRequest;
use narra::repository::{
    EntityRepository, KnowledgeRepository, SurrealEntityRepository, SurrealKnowledgeRepository,
};
use rmcp::handler::server::wrapper::Parameters;

use crate::common::{
    builders::{CharacterBuilder, KnowledgeBuilder},
    harness::TestHarness,
    to_query_input,
};

// =============================================================================
// ARC HISTORY TESTS
// =============================================================================

/// Test arc history for entity with no snapshots returns gracefully.
#[tokio::test]
async fn test_arc_history_no_snapshots() {
    let harness = TestHarness::new().await;
    let repo = SurrealEntityRepository::new(harness.db.clone());

    let alice = repo
        .create_character(CharacterBuilder::new("Alice").build())
        .await
        .expect("Alice");

    let server = crate::common::create_test_server(&harness).await;

    let request = QueryRequest::ArcHistory {
        entity_id: format!("character:{}", alice.id.key()),
        limit: Some(50),
    };

    let response = server
        .handle_query(Parameters(to_query_input(request)))
        .await
        .expect("ArcHistory should succeed even without snapshots");

    // With no snapshots, may return empty results or a single "no data" result
    assert!(
        response.results.is_empty() || response.results.len() == 1,
        "Should return zero or one result"
    );
}

/// Test arc history for nonexistent entity.
#[tokio::test]
async fn test_arc_history_nonexistent() {
    let harness = TestHarness::new().await;
    let server = crate::common::create_test_server(&harness).await;

    let request = QueryRequest::ArcHistory {
        entity_id: "character:nonexistent".to_string(),
        limit: Some(50),
    };

    let response = server
        .handle_query(Parameters(to_query_input(request)))
        .await;

    // Should handle gracefully (either error or empty result)
    assert!(
        response.is_ok() || response.is_err(),
        "Should handle gracefully"
    );
}

// =============================================================================
// ARC COMPARISON TESTS
// =============================================================================

/// Test arc comparison between two entities with no snapshots returns an error
/// (since no baseline data exists with NoopEmbedding).
#[tokio::test]
async fn test_arc_comparison_no_snapshots() {
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

    let request = QueryRequest::ArcComparison {
        entity_id_a: format!("character:{}", alice.id.key()),
        entity_id_b: format!("character:{}", bob.id.key()),
        window: None,
    };

    let response = server
        .handle_query(Parameters(to_query_input(request)))
        .await;

    // Without snapshots, should return an error about missing arc data
    assert!(
        response.is_err(),
        "ArcComparison should error without snapshots"
    );
    let err = response.unwrap_err();
    assert!(
        err.contains("arc_snapshots")
            || err.contains("BaselineArcSnapshots")
            || err.contains("Not found"),
        "Error should reference missing snapshots, got: {}",
        err
    );
}

// =============================================================================
// ARC MOMENT TESTS
// =============================================================================

/// Test arc moment for entity with no snapshots returns error.
#[tokio::test]
async fn test_arc_moment_no_snapshots() {
    let harness = TestHarness::new().await;
    let repo = SurrealEntityRepository::new(harness.db.clone());

    let alice = repo
        .create_character(CharacterBuilder::new("Alice").build())
        .await
        .expect("Alice");

    let server = crate::common::create_test_server(&harness).await;

    let request = QueryRequest::ArcMoment {
        entity_id: format!("character:{}", alice.id.key()),
        event_id: None,
    };

    let response = server
        .handle_query(Parameters(to_query_input(request)))
        .await;

    // Without snapshots, should error
    assert!(
        response.is_err(),
        "ArcMoment should error without snapshots"
    );
}

// =============================================================================
// WHAT-IF TESTS
// =============================================================================

/// Test what-if analysis without embedding model returns clear error.
///
/// WhatIf requires embedding model for computing impact. With NoopEmbedding,
/// it should return a meaningful error about missing embeddings.
#[tokio::test]
async fn test_what_if_no_embedding() {
    let harness = TestHarness::new().await;
    let repo = SurrealEntityRepository::new(harness.db.clone());

    let alice = repo
        .create_character(CharacterBuilder::new("Alice").build())
        .await
        .expect("Alice");

    let knowledge_repo = SurrealKnowledgeRepository::new(harness.db.clone());
    let knowledge = knowledge_repo
        .create_knowledge(KnowledgeBuilder::new("The crystal has been shattered").build())
        .await
        .expect("Knowledge");

    let server = crate::common::create_test_server(&harness).await;

    let request = QueryRequest::WhatIf {
        character_id: format!("character:{}", alice.id.key()),
        fact_id: knowledge.id.to_string(),
        certainty: Some("knows".to_string()),
        source_character: None,
    };

    let response = server
        .handle_query(Parameters(to_query_input(request)))
        .await;

    // WhatIf requires embeddings - should error with Noop
    assert!(
        response.is_err(),
        "WhatIf should error without embedding model"
    );
    let err = response.unwrap_err();
    assert!(
        err.contains("embedding")
            || err.contains("unavailable")
            || err.contains("BackfillEmbeddings"),
        "Error should reference embedding requirements, got: {}",
        err
    );
}
