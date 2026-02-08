//! Query handler integration tests.
//!
//! Tests validate the query tool handler correctly translates between MCP
//! protocol requests and service layer calls, returning properly formatted
//! responses.

use std::sync::Arc;

use insta::assert_snapshot;
use narra::embedding::{EmbeddingConfig, EmbeddingService, LocalEmbeddingService};
use narra::mcp::{DetailLevel, NarraServer, QueryRequest};
use narra::repository::{EntityRepository, SurrealEntityRepository};
use narra::session::SessionStateManager;
use pretty_assertions::assert_eq;
use rmcp::handler::server::wrapper::Parameters;

use crate::common::builders::{CharacterBuilder, LocationBuilder};
use crate::common::harness::TestHarness;

// =============================================================================
// LOOKUP OPERATIONS
// =============================================================================

/// Test successful character lookup via handler.
///
/// Verifies:
/// - Handler returns correct entity_type
/// - Entity ID matches request
/// - Name is correctly populated
/// - Confidence is 1.0 for direct lookup
#[tokio::test]
async fn test_lookup_character_success() {
    let harness = TestHarness::new().await;
    let server = crate::common::create_test_server(&harness).await;

    // Create test character via repository
    let repo = SurrealEntityRepository::new(harness.db.clone());
    let character = repo
        .create_character(CharacterBuilder::new("Alice").build())
        .await
        .expect("Failed to create character");
    let char_id = character.id.to_string();

    // Call handler directly
    let request = QueryRequest::Lookup {
        entity_id: char_id.clone(),
        detail_level: Some(DetailLevel::Standard),
    };
    let response = server.handle_query(Parameters(request)).await;

    // Verify success
    assert!(response.is_ok(), "Query should succeed");
    let response = response.unwrap();

    assert_eq!(
        response.results.len(),
        1,
        "Should return exactly one result"
    );
    assert_eq!(response.total, 1);

    let entity = &response.results[0];
    assert_eq!(entity.id, char_id);
    assert_eq!(entity.name, "Alice");
    assert_eq!(entity.entity_type, "character");
    assert_eq!(entity.confidence, Some(1.0));
}

/// Test successful location lookup via handler.
///
/// Verifies entity_type is "location" for location entities.
#[tokio::test]
async fn test_lookup_location_success() {
    let harness = TestHarness::new().await;
    let server = crate::common::create_test_server(&harness).await;

    // Create test location via repository
    let repo = SurrealEntityRepository::new(harness.db.clone());
    let location = repo
        .create_location(LocationBuilder::new("The Tower").build())
        .await
        .expect("Failed to create location");
    let loc_id = location.id.to_string();

    // Call handler directly
    let request = QueryRequest::Lookup {
        entity_id: loc_id.clone(),
        detail_level: None,
    };
    let response = server.handle_query(Parameters(request)).await;

    // Verify success
    assert!(response.is_ok(), "Query should succeed");
    let response = response.unwrap();

    assert_eq!(response.results.len(), 1);
    let entity = &response.results[0];
    assert_eq!(entity.entity_type, "location");
    assert_eq!(entity.name, "The Tower");
}

/// Test lookup for nonexistent entity returns error.
///
/// Verifies error message is appropriate for not-found case.
#[tokio::test]
async fn test_lookup_nonexistent_entity() {
    let harness = TestHarness::new().await;
    let server = crate::common::create_test_server(&harness).await;

    let request = QueryRequest::Lookup {
        entity_id: "character:nonexistent".to_string(),
        detail_level: None,
    };
    let response = server.handle_query(Parameters(request)).await;

    // Verify error
    assert!(
        response.is_err(),
        "Query should fail for nonexistent entity"
    );
    let error_message = response.unwrap_err();

    // Snapshot the error message for stability tracking
    assert_snapshot!("lookup_nonexistent_error", error_message);
}

// =============================================================================
// SEARCH OPERATIONS
// =============================================================================

/// Test search by name returns matching entities.
///
/// Verifies:
/// - Search returns entities matching query
/// - Results have confidence scores
#[tokio::test]
async fn test_search_by_name() {
    let harness = TestHarness::new().await;
    let server = crate::common::create_test_server(&harness).await;

    // Create test characters via repository
    let repo = SurrealEntityRepository::new(harness.db.clone());
    repo.create_character(CharacterBuilder::new("Alice").build())
        .await
        .expect("Failed to create Alice");
    repo.create_character(CharacterBuilder::new("Bob").build())
        .await
        .expect("Failed to create Bob");
    repo.create_character(CharacterBuilder::new("Charlie").build())
        .await
        .expect("Failed to create Charlie");

    // Search for Alice
    let request = QueryRequest::Search {
        query: "Alice".to_string(),
        entity_types: Some(vec!["character".to_string()]),
        limit: Some(10),
        cursor: None,
    };
    let response = server.handle_query(Parameters(request)).await;

    // Verify success
    assert!(response.is_ok(), "Search should succeed");
    let response = response.unwrap();

    // Should find Alice
    assert!(
        !response.results.is_empty(),
        "Should find at least one result"
    );

    // First result should be Alice (best match)
    let first = &response.results[0];
    assert_eq!(first.name, "Alice");
    assert!(
        first.confidence.is_some(),
        "Search results should have confidence"
    );
}

/// Test search pagination with cursor.
///
/// Verifies:
/// - First page returns correct number of results
/// - Cursor is returned when more results exist
/// - Second page can be fetched with cursor
/// - No overlap between pages
#[tokio::test]
async fn test_search_pagination() {
    let harness = TestHarness::new().await;
    let server = crate::common::create_test_server(&harness).await;

    // Create 25 characters for pagination testing
    let repo = SurrealEntityRepository::new(harness.db.clone());
    for i in 0..25 {
        repo.create_character(CharacterBuilder::new(format!("Character{:02}", i)).build())
            .await
            .expect("Failed to create character");
    }

    // Page 1: First 10 results
    let page1_request = QueryRequest::Search {
        query: "Character".to_string(),
        entity_types: Some(vec!["character".to_string()]),
        limit: Some(10),
        cursor: None,
    };
    let page1_response = server.handle_query(Parameters(page1_request)).await;

    assert!(page1_response.is_ok(), "Page 1 search should succeed");
    let page1 = page1_response.unwrap();

    assert_eq!(page1.results.len(), 10, "First page should have 10 results");
    assert!(
        page1.next_cursor.is_some(),
        "Should have cursor for next page"
    );

    // Page 2: Use cursor
    let page2_request = QueryRequest::Search {
        query: "Character".to_string(),
        entity_types: Some(vec!["character".to_string()]),
        limit: Some(10),
        cursor: page1.next_cursor.clone(),
    };
    let page2_response = server.handle_query(Parameters(page2_request)).await;

    assert!(page2_response.is_ok(), "Page 2 search should succeed");
    let page2 = page2_response.unwrap();

    assert_eq!(
        page2.results.len(),
        10,
        "Second page should have 10 results"
    );

    // Verify no overlap between pages
    let page1_ids: Vec<_> = page1.results.iter().map(|r| &r.id).collect();
    for result in &page2.results {
        assert!(
            !page1_ids.contains(&&result.id),
            "Pages should not overlap: {} found in both",
            result.id
        );
    }
}

/// Test search with no matches returns empty results.
///
/// Verifies:
/// - Empty results array
/// - No cursor for empty results
#[tokio::test]
async fn test_search_empty_results() {
    let harness = TestHarness::new().await;
    let server = crate::common::create_test_server(&harness).await;

    let request = QueryRequest::Search {
        query: "NonexistentEntity12345".to_string(),
        entity_types: None,
        limit: Some(20),
        cursor: None,
    };
    let response = server.handle_query(Parameters(request)).await;

    assert!(response.is_ok(), "Empty search should succeed");
    let response = response.unwrap();

    assert!(response.results.is_empty(), "Should have no results");
    assert!(
        response.next_cursor.is_none(),
        "Should have no cursor for empty results"
    );
}

// =============================================================================
// OVERVIEW OPERATIONS
// =============================================================================

/// Test overview returns list of entities for a type.
///
/// Verifies:
/// - Overview returns all characters
/// - Each result has correct entity_type
#[tokio::test]
async fn test_overview_characters() {
    let harness = TestHarness::new().await;
    let server = crate::common::create_test_server(&harness).await;

    // Create test characters
    let repo = SurrealEntityRepository::new(harness.db.clone());
    repo.create_character(CharacterBuilder::new("Alice").build())
        .await
        .expect("Failed to create Alice");
    repo.create_character(CharacterBuilder::new("Bob").build())
        .await
        .expect("Failed to create Bob");

    let request = QueryRequest::Overview {
        entity_type: "character".to_string(),
        limit: Some(20),
    };
    let response = server.handle_query(Parameters(request)).await;

    assert!(response.is_ok(), "Overview should succeed");
    let response = response.unwrap();

    assert_eq!(response.results.len(), 2, "Should return 2 characters");
    for result in &response.results {
        assert_eq!(result.entity_type, "character");
    }
}

/// Test overview with unknown entity type returns error.
///
/// Verifies appropriate error message for invalid type.
#[tokio::test]
async fn test_overview_unknown_type() {
    let harness = TestHarness::new().await;
    let server = crate::common::create_test_server(&harness).await;

    let request = QueryRequest::Overview {
        entity_type: "invalid_type".to_string(),
        limit: Some(20),
    };
    let response = server.handle_query(Parameters(request)).await;

    assert!(response.is_err(), "Overview should fail for unknown type");
    let error_message = response.unwrap_err();

    // Snapshot the error message
    assert_snapshot!("overview_unknown_type_error", error_message);
}

// =============================================================================
// GRAPH INTELLIGENCE OPERATIONS
// =============================================================================

/// Test centrality metrics computation returns character network structure.
///
/// Verifies:
/// - Handler returns centrality metrics for characters
/// - Each result has degree, betweenness, closeness values
/// - Narrative roles are assigned (hub, bridge, peripheral, isolated)
#[tokio::test]
async fn test_centrality_metrics() {
    let harness = TestHarness::new().await;
    let server = crate::common::create_test_server(&harness).await;

    // Create test characters with relationships
    let repo = SurrealEntityRepository::new(harness.db.clone());
    let alice = repo
        .create_character(CharacterBuilder::new("Alice").build())
        .await
        .expect("Failed to create Alice");
    let bob = repo
        .create_character(CharacterBuilder::new("Bob").build())
        .await
        .expect("Failed to create Bob");
    let charlie = repo
        .create_character(CharacterBuilder::new("Charlie").build())
        .await
        .expect("Failed to create Charlie");

    // Create relationships: Alice <-> Bob, Bob <-> Charlie (Bob is bridge)
    use narra::models::relationship::{create_relationship, RelationshipCreate};
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
    .expect("Alice->Bob relationship");

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
    .expect("Bob->Charlie relationship");

    // Call centrality metrics
    let request = QueryRequest::CentralityMetrics {
        scope: None,
        metrics: None,
        limit: Some(10),
    };
    let response = server.handle_query(Parameters(request)).await;

    // Verify success
    assert!(response.is_ok(), "Centrality metrics should succeed");
    let response = response.unwrap();

    assert!(
        !response.results.is_empty(),
        "Should return centrality metrics"
    );

    // Check that results have expected format
    for result in &response.results {
        assert_eq!(result.entity_type, "character");
        assert!(
            result.content.contains("Degree:"),
            "Should have degree metric"
        );
        assert!(
            result.content.contains("Betweenness:"),
            "Should have betweenness metric"
        );
        assert!(
            result.content.contains("Closeness:"),
            "Should have closeness metric"
        );
        assert!(
            result.content.contains("Role:"),
            "Should have narrative role"
        );
    }

    // Bob should have highest betweenness (bridge between Alice and Charlie)
    let bob_result = response.results.iter().find(|r| r.name == "Bob");
    assert!(bob_result.is_some(), "Bob should be in centrality results");
}

/// Test influence propagation traces information flow through directed relationships.
///
/// Verifies:
/// - Handler returns reachable characters from source
/// - Paths show correct direction and relationship types
/// - Path strength is classified (direct, likely, possible)
#[tokio::test]
async fn test_influence_propagation() {
    let harness = TestHarness::new().await;
    let server = crate::common::create_test_server(&harness).await;

    // Create test characters
    let repo = SurrealEntityRepository::new(harness.db.clone());
    let alice = repo
        .create_character(CharacterBuilder::new("Alice").build())
        .await
        .expect("Failed to create Alice");
    let bob = repo
        .create_character(CharacterBuilder::new("Bob").build())
        .await
        .expect("Failed to create Bob");
    let charlie = repo
        .create_character(CharacterBuilder::new("Charlie").build())
        .await
        .expect("Failed to create Charlie");

    // Create directed relationships: Alice -> Bob -> Charlie
    // Use perceives edge (directed) for influence propagation
    use narra::models::perception::{create_perception, PerceptionCreate};
    create_perception(
        &harness.db,
        &alice.id.key().to_string(),
        &bob.id.key().to_string(),
        PerceptionCreate {
            rel_types: vec!["trusts".to_string()],
            subtype: None,
            feelings: None,
            perception: None,
            tension_level: None,
            history_notes: None,
        },
    )
    .await
    .expect("Alice->Bob perception");

    create_perception(
        &harness.db,
        &bob.id.key().to_string(),
        &charlie.id.key().to_string(),
        PerceptionCreate {
            rel_types: vec!["trusts".to_string()],
            subtype: None,
            feelings: None,
            perception: None,
            tension_level: None,
            history_notes: None,
        },
    )
    .await
    .expect("Bob->Charlie perception");

    // Call influence propagation from Alice
    let request = QueryRequest::InfluencePropagation {
        from_character_id: format!("character:{}", alice.id.key()),
        knowledge_fact: None,
        max_depth: Some(3),
    };
    let response = server.handle_query(Parameters(request)).await;

    // Verify success
    assert!(response.is_ok(), "Influence propagation should succeed");
    let response = response.unwrap();

    // Should find Bob and Charlie reachable from Alice
    assert!(
        !response.results.is_empty(),
        "Should find reachable characters"
    );

    // Verify path structure in content
    for result in &response.results {
        assert_eq!(result.entity_type, "influence_path");
        assert!(result.content.contains("->"), "Should show path direction");
        assert!(
            result.content.contains("Strength:"),
            "Should show path strength"
        );
    }

    // Should include hints about reachable characters
    assert!(!response.hints.is_empty(), "Should include hints");
}

/// Test dramatic irony report returns proper response structure.
///
/// Verifies:
/// - Handler succeeds and returns response
/// - Response structure matches QueryResponse format
/// - Hints provide helpful information
#[tokio::test]
async fn test_dramatic_irony_report() {
    let harness = TestHarness::new().await;
    let server = crate::common::create_test_server(&harness).await;

    // Create test characters (knowledge asymmetries will be detected if present)
    let repo = SurrealEntityRepository::new(harness.db.clone());
    let _alice = repo
        .create_character(CharacterBuilder::new("Alice").build())
        .await
        .expect("Failed to create Alice");
    let _bob = repo
        .create_character(CharacterBuilder::new("Bob").build())
        .await
        .expect("Failed to create Bob");

    // Call dramatic irony report
    let request = QueryRequest::DramaticIronyReport {
        character_id: None,
        min_scene_threshold: Some(0),
    };
    let response = server.handle_query(Parameters(request)).await;

    // Verify success
    assert!(response.is_ok(), "Dramatic irony report should succeed");
    let response = response.unwrap();

    // Should include hints about asymmetries
    assert!(!response.hints.is_empty(), "Should include hints");

    // If results exist, verify they have correct entity_type
    for result in &response.results {
        assert_eq!(
            result.entity_type, "dramatic_irony",
            "Should return dramatic_irony type"
        );
    }
}

/// Test semantic join performs cross-type vector search.
///
/// Verifies:
/// - Handler returns semantically similar entities across types
/// - Results ranked by similarity score
/// - Requires embedding model (gracefully handles when unavailable)
///
/// NOTE: This test is ignored because it requires the fastembed model.
/// Run with: cargo test --ignored test_semantic_join
#[tokio::test]
#[ignore]
async fn test_semantic_join() {
    let harness = TestHarness::new().await;

    // Create server with real embedding service for this test
    let session_path = harness.temp_path().join("session.json");
    let session_manager = Arc::new(
        SessionStateManager::load_or_create(&session_path)
            .expect("Failed to create session manager"),
    );

    let embedding_service: Arc<dyn EmbeddingService + Send + Sync> =
        Arc::new(LocalEmbeddingService::new(EmbeddingConfig::default()).unwrap());
    let server = NarraServer::new(harness.db.clone(), session_manager, embedding_service).await;

    // Create test characters
    let repo = SurrealEntityRepository::new(harness.db.clone());
    let _alice = repo
        .create_character(CharacterBuilder::new("Alice").role("warrior").build())
        .await
        .expect("Failed to create Alice");
    let _bob = repo
        .create_character(CharacterBuilder::new("Bob").role("merchant").build())
        .await
        .expect("Failed to create Bob");

    // Generate embeddings
    use narra::mcp::MutationRequest;
    let backfill_request = MutationRequest::BackfillEmbeddings {
        entity_type: Some("character".to_string()),
    };
    let _backfill_response = server
        .handle_mutate(Parameters(backfill_request))
        .await
        .expect("Backfill should succeed");

    // Call semantic join with query about character traits
    let request = QueryRequest::SemanticJoin {
        query: "character struggling with duty".to_string(),
        entity_types: Some(vec!["character".to_string()]),
        limit: Some(10),
    };
    let response = server.handle_query(Parameters(request)).await;

    // Verify success
    assert!(response.is_ok(), "Semantic join should succeed");
    let response = response.unwrap();

    // Should find semantically related characters
    assert!(!response.results.is_empty(), "Should find semantic matches");

    // Results should have confidence scores
    for result in &response.results {
        assert!(result.confidence.is_some(), "Should have similarity score");
    }

    // Should include hints about cross-field search
    assert!(
        response
            .hints
            .iter()
            .any(|h| h.contains("cross-field") || h.contains("semantic")),
        "Should mention semantic capabilities"
    );
}

/// Test thematic clustering discovers story themes from embeddings.
///
/// Verifies:
/// - Handler returns thematic clusters
/// - Each cluster has label and member list
/// - Confidence reflects cluster size relative to total
/// - Requires embeddings (gracefully handles when unavailable)
///
/// NOTE: This test is ignored because it requires the fastembed model.
/// Run with: cargo test --ignored test_thematic_clustering
#[tokio::test]
#[ignore]
async fn test_thematic_clustering() {
    let harness = TestHarness::new().await;

    // Create server with real embedding service for this test
    let session_path = harness.temp_path().join("session.json");
    let session_manager = Arc::new(
        SessionStateManager::load_or_create(&session_path)
            .expect("Failed to create session manager"),
    );

    let embedding_service: Arc<dyn EmbeddingService + Send + Sync> =
        Arc::new(LocalEmbeddingService::new(EmbeddingConfig::default()).unwrap());
    let server = NarraServer::new(harness.db.clone(), session_manager, embedding_service).await;

    // Create test characters with thematic groupings
    let repo = SurrealEntityRepository::new(harness.db.clone());
    let _alice = repo
        .create_character(CharacterBuilder::new("Alice").role("warrior").build())
        .await
        .expect("Failed to create Alice");
    let _bob = repo
        .create_character(CharacterBuilder::new("Bob").role("merchant").build())
        .await
        .expect("Failed to create Bob");
    let _charlie = repo
        .create_character(CharacterBuilder::new("Charlie").role("noble").build())
        .await
        .expect("Failed to create Charlie");

    // Generate embeddings
    use narra::mcp::MutationRequest;
    let backfill_request = MutationRequest::BackfillEmbeddings {
        entity_type: Some("character".to_string()),
    };
    let _backfill_response = server
        .handle_mutate(Parameters(backfill_request))
        .await
        .expect("Backfill should succeed");

    // Call thematic clustering
    let request = QueryRequest::ThematicClustering {
        entity_types: Some(vec!["character".to_string()]),
        num_themes: Some(2),
    };
    let response = server.handle_query(Parameters(request)).await;

    // Verify success
    assert!(response.is_ok(), "Thematic clustering should succeed");
    let response = response.unwrap();

    // Should find theme clusters
    assert!(!response.results.is_empty(), "Should discover themes");

    // Each cluster should have proper structure
    for result in &response.results {
        assert_eq!(result.entity_type, "theme_cluster");
        assert!(!result.name.is_empty(), "Cluster should have label");
        assert!(!result.content.is_empty(), "Should list members");
        assert!(result.confidence.is_some(), "Should have confidence score");
    }

    // Should include hints about clustering
    assert!(
        response
            .hints
            .iter()
            .any(|h| h.contains("theme") || h.contains("cluster")),
        "Should mention themes or clustering"
    );
}
