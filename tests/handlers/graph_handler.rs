//! Integration tests for graph handler operations (HAND-03).
//!
//! Tests GraphTraversal queries and generate_graph operations including:
//! - Graph traversal at different depths
//! - Mermaid diagram generation with various scopes
//! - Scope parsing for "full" and "character:ID" formats
//! - Error handling for invalid inputs

use std::sync::Arc;

use narra::embedding::{EmbeddingService, NoopEmbeddingService};
use narra::mcp::tools::graph::GraphRequest;
use narra::mcp::{NarraServer, QueryRequest};
use narra::models::perception::{create_perception, PerceptionCreate};
use narra::models::relationship::{create_relationship, RelationshipCreate};
use narra::repository::{EntityRepository, SurrealEntityRepository};
use narra::session::SessionStateManager;
use rmcp::handler::server::wrapper::Parameters;

use crate::common::builders::CharacterBuilder;
use crate::common::harness::TestHarness;

/// Helper to create NarraServer with isolated harness.
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

// ============================================================================
// GRAPH TRAVERSAL TESTS
// ============================================================================

/// Test successful graph traversal returns connected entities.
#[tokio::test]
async fn test_graph_traversal_success() {
    let harness = TestHarness::new().await;
    let entity_repo = SurrealEntityRepository::new(harness.db.clone());

    // Create characters
    let alice = entity_repo
        .create_character(CharacterBuilder::new("Alice").build())
        .await
        .expect("Alice");
    let bob = entity_repo
        .create_character(CharacterBuilder::new("Bob").build())
        .await
        .expect("Bob");

    // Create relationship using relates_to edge (which get_connected_entities traverses)
    create_relationship(
        &harness.db,
        RelationshipCreate {
            from_character_id: alice.id.key().to_string(),
            to_character_id: bob.id.key().to_string(),
            rel_type: "friendship".to_string(),
            subtype: None,
            label: Some("best friends".to_string()),
        },
    )
    .await
    .expect("Relationship created");

    // Create server
    let session_path = harness.temp_path().join("session.json");
    let session_manager = Arc::new(SessionStateManager::load_or_create(&session_path).unwrap());
    let server = NarraServer::new(
        harness.db.clone(),
        session_manager,
        Arc::new(NoopEmbeddingService::new()),
    )
    .await;

    // Traverse graph from Alice
    let request = QueryRequest::GraphTraversal {
        entity_id: format!("character:{}", alice.id.key()),
        depth: 2,
        format: None,
    };

    let response = server
        .handle_query(Parameters(request))
        .await
        .expect("Graph traversal should succeed");

    // Verify connected entities returned
    assert!(
        !response.results.is_empty(),
        "Should find connected entities"
    );
    assert!(response.hints.len() >= 1, "Should include helpful hints");

    // Check that Bob is in results
    let has_bob = response
        .results
        .iter()
        .any(|r| r.id.contains(&bob.id.key().to_string()));
    assert!(has_bob, "Should find Bob connected to Alice");
}

/// Test graph traversal with depth=1 returns only direct connections.
#[tokio::test]
async fn test_graph_traversal_depth_1() {
    let harness = TestHarness::new().await;
    let entity_repo = SurrealEntityRepository::new(harness.db.clone());

    // Create chain: Alice -> Bob -> Charlie
    let alice = entity_repo
        .create_character(CharacterBuilder::new("Alice").build())
        .await
        .expect("Alice");
    let bob = entity_repo
        .create_character(CharacterBuilder::new("Bob").build())
        .await
        .expect("Bob");
    let charlie = entity_repo
        .create_character(CharacterBuilder::new("Charlie").build())
        .await
        .expect("Charlie");

    // Alice -> Bob
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
    .expect("Alice->Bob");

    // Bob -> Charlie
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
    .expect("Bob->Charlie");

    // Create server
    let session_path = harness.temp_path().join("session.json");
    let session_manager = Arc::new(SessionStateManager::load_or_create(&session_path).unwrap());
    let server = NarraServer::new(
        harness.db.clone(),
        session_manager,
        Arc::new(NoopEmbeddingService::new()),
    )
    .await;

    // Traverse from Alice with depth=1
    let request = QueryRequest::GraphTraversal {
        entity_id: format!("character:{}", alice.id.key()),
        depth: 1,
        format: None,
    };

    let response = server
        .handle_query(Parameters(request))
        .await
        .expect("Depth-1 traversal should succeed");

    // Should find Bob (depth 1) but not Charlie (depth 2)
    let has_bob = response
        .results
        .iter()
        .any(|r| r.id.contains(&bob.id.key().to_string()));
    let has_charlie = response
        .results
        .iter()
        .any(|r| r.id.contains(&charlie.id.key().to_string()));

    assert!(has_bob, "Should find Bob at depth 1");
    assert!(!has_charlie, "Should NOT find Charlie at depth 1");
}

/// Test graph traversal with depth=2 returns transitive connections.
#[tokio::test]
async fn test_graph_traversal_depth_2() {
    let harness = TestHarness::new().await;
    let entity_repo = SurrealEntityRepository::new(harness.db.clone());

    // Create chain: Alice -> Bob -> Charlie -> David
    let alice = entity_repo
        .create_character(CharacterBuilder::new("Alice").build())
        .await
        .expect("Alice");
    let bob = entity_repo
        .create_character(CharacterBuilder::new("Bob").build())
        .await
        .expect("Bob");
    let charlie = entity_repo
        .create_character(CharacterBuilder::new("Charlie").build())
        .await
        .expect("Charlie");
    let david = entity_repo
        .create_character(CharacterBuilder::new("David").build())
        .await
        .expect("David");

    // Alice -> Bob -> Charlie -> David
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
    .expect("Alice->Bob");

    create_relationship(
        &harness.db,
        RelationshipCreate {
            from_character_id: bob.id.key().to_string(),
            to_character_id: charlie.id.key().to_string(),
            rel_type: "friendship".to_string(),
            subtype: None,
            label: None,
        },
    )
    .await
    .expect("Bob->Charlie");

    create_relationship(
        &harness.db,
        RelationshipCreate {
            from_character_id: charlie.id.key().to_string(),
            to_character_id: david.id.key().to_string(),
            rel_type: "friendship".to_string(),
            subtype: None,
            label: None,
        },
    )
    .await
    .expect("Charlie->David");

    // Create server
    let session_path = harness.temp_path().join("session.json");
    let session_manager = Arc::new(SessionStateManager::load_or_create(&session_path).unwrap());
    let server = NarraServer::new(
        harness.db.clone(),
        session_manager,
        Arc::new(NoopEmbeddingService::new()),
    )
    .await;

    // Traverse from Alice with depth=2
    let request = QueryRequest::GraphTraversal {
        entity_id: format!("character:{}", alice.id.key()),
        depth: 2,
        format: None,
    };

    let response = server
        .handle_query(Parameters(request))
        .await
        .expect("Depth-2 traversal should succeed");

    // Should find Bob (depth 1) and Charlie (depth 2), but not David (depth 3)
    let has_bob = response
        .results
        .iter()
        .any(|r| r.id.contains(&bob.id.key().to_string()));
    let has_charlie = response
        .results
        .iter()
        .any(|r| r.id.contains(&charlie.id.key().to_string()));
    let has_david = response
        .results
        .iter()
        .any(|r| r.id.contains(&david.id.key().to_string()));

    assert!(has_bob, "Should find Bob at depth 1");
    assert!(has_charlie, "Should find Charlie at depth 2");
    assert!(
        !has_david,
        "Should NOT find David at depth 2 (he's at depth 3)"
    );
}

/// Test graph traversal from isolated entity returns empty results.
#[tokio::test]
async fn test_graph_traversal_no_connections() {
    let harness = TestHarness::new().await;
    let entity_repo = SurrealEntityRepository::new(harness.db.clone());

    // Create isolated character
    let lonely = entity_repo
        .create_character(CharacterBuilder::new("Lonely").build())
        .await
        .expect("Lonely");

    // Create server
    let session_path = harness.temp_path().join("session.json");
    let session_manager = Arc::new(SessionStateManager::load_or_create(&session_path).unwrap());
    let server = NarraServer::new(
        harness.db.clone(),
        session_manager,
        Arc::new(NoopEmbeddingService::new()),
    )
    .await;

    // Traverse from isolated character
    let request = QueryRequest::GraphTraversal {
        entity_id: format!("character:{}", lonely.id.key()),
        depth: 2,
        format: None,
    };

    let response = server
        .handle_query(Parameters(request))
        .await
        .expect("Traversal should succeed even with no connections");

    // Should have empty results
    assert!(
        response.results.is_empty(),
        "Isolated entity should have no connected entities"
    );
    assert!(!response.hints.is_empty(), "Should still include hints");
}

/// Test graph traversal from nonexistent entity returns error.
#[tokio::test]
async fn test_graph_traversal_nonexistent_entity() {
    let harness = TestHarness::new().await;

    // Create server (no entities created)
    let session_path = harness.temp_path().join("session.json");
    let session_manager = Arc::new(SessionStateManager::load_or_create(&session_path).unwrap());
    let server = NarraServer::new(
        harness.db.clone(),
        session_manager,
        Arc::new(NoopEmbeddingService::new()),
    )
    .await;

    // Traverse from nonexistent entity
    let request = QueryRequest::GraphTraversal {
        entity_id: "character:nonexistent_id_12345".to_string(),
        depth: 2,
        format: None,
    };

    // Should not error - graph traversal returns empty for non-existent nodes
    // (as the BFS simply finds no neighbors from a node that doesn't exist)
    let response = server.handle_query(Parameters(request)).await;

    // The handler returns success with empty results rather than an error
    // because the BFS algorithm gracefully handles missing starting nodes
    assert!(
        response.is_ok(),
        "Traversal from non-existent entity should succeed with empty results"
    );
    let resp = response.unwrap();
    assert!(
        resp.results.is_empty(),
        "Should have no results for non-existent entity"
    );
}

// ============================================================================
// GENERATE_GRAPH TESTS - SCOPE HANDLING
// ============================================================================

/// Test generate_graph with scope="full" generates network of all characters.
#[tokio::test]
async fn test_generate_graph_full_network() {
    let harness = TestHarness::new().await;
    let entity_repo = SurrealEntityRepository::new(harness.db.clone());

    // Create characters with relationships
    let alice = entity_repo
        .create_character(CharacterBuilder::new("Alice").role("protagonist").build())
        .await
        .expect("Alice");
    let bob = entity_repo
        .create_character(CharacterBuilder::new("Bob").role("antagonist").build())
        .await
        .expect("Bob");

    // Create perception edge (used by MermaidGraphService)
    create_perception(
        &harness.db,
        &alice.id.key().to_string(),
        &bob.id.key().to_string(),
        PerceptionCreate {
            rel_types: vec!["rivalry".to_string()],
            subtype: None,
            feelings: None,
            perception: None,
            tension_level: None,
            history_notes: None,
        },
    )
    .await
    .expect("Alice->Bob perception");

    // Create server
    let session_path = harness.temp_path().join("session.json");
    let session_manager = Arc::new(SessionStateManager::load_or_create(&session_path).unwrap());
    let server = NarraServer::new(
        harness.db.clone(),
        session_manager,
        Arc::new(NoopEmbeddingService::new()),
    )
    .await;

    // Generate full network graph with unique filename
    let request = GraphRequest {
        scope: "full".to_string(),
        depth: None,
        include_roles: None,
        filename: Some("test_full_network_graph.md".to_string()),
    };

    let response = server
        .handle_generate_graph(Parameters(request))
        .await
        .expect("Full network graph should generate");

    // Verify response structure
    assert!(
        response.file_path.contains("test_full_network_graph.md"),
        "File path should contain provided filename"
    );
    assert!(
        response.character_count >= 2,
        "Should include at least 2 characters"
    );
    assert!(!response.hints.is_empty(), "Should include hints");

    // Verify file was created
    assert!(
        std::path::Path::new(&response.file_path).exists(),
        "Graph file should exist at {}",
        response.file_path
    );

    // Read and verify file contents
    let content = std::fs::read_to_string(&response.file_path).expect("Read graph file");
    assert!(content.contains("```mermaid"), "Should have mermaid fence");
    assert!(content.contains("Alice"), "Should contain Alice");
    assert!(content.contains("Bob"), "Should contain Bob");

    // Cleanup test file
    let _ = std::fs::remove_file(&response.file_path);
}

/// Test generate_graph with scope="character:ID" generates centered view.
#[tokio::test]
async fn test_generate_graph_character_centered() {
    let harness = TestHarness::new().await;
    let entity_repo = SurrealEntityRepository::new(harness.db.clone());

    // Create characters with chain: Center -> Near -> Far
    let center = entity_repo
        .create_character(CharacterBuilder::new("Center").build())
        .await
        .expect("Center");
    let near = entity_repo
        .create_character(CharacterBuilder::new("Near").build())
        .await
        .expect("Near");
    let far = entity_repo
        .create_character(CharacterBuilder::new("Far").build())
        .await
        .expect("Far");

    // Center -> Near
    create_perception(
        &harness.db,
        &center.id.key().to_string(),
        &near.id.key().to_string(),
        PerceptionCreate {
            rel_types: vec!["friendship".to_string()],
            subtype: None,
            feelings: None,
            perception: None,
            tension_level: None,
            history_notes: None,
        },
    )
    .await
    .expect("Center->Near");

    // Near -> Far (depth 2 from center)
    create_perception(
        &harness.db,
        &near.id.key().to_string(),
        &far.id.key().to_string(),
        PerceptionCreate {
            rel_types: vec!["professional".to_string()],
            subtype: None,
            feelings: None,
            perception: None,
            tension_level: None,
            history_notes: None,
        },
    )
    .await
    .expect("Near->Far");

    // Create server
    let session_path = harness.temp_path().join("session.json");
    let session_manager = Arc::new(SessionStateManager::load_or_create(&session_path).unwrap());
    let server = NarraServer::new(
        harness.db.clone(),
        session_manager,
        Arc::new(NoopEmbeddingService::new()),
    )
    .await;

    // Generate character-centered graph with depth=1
    let request = GraphRequest {
        scope: format!("character:{}", center.id.key()),
        depth: Some(1),
        include_roles: None,
        filename: Some("test_centered_graph.md".to_string()),
    };

    let response = server
        .handle_generate_graph(Parameters(request))
        .await
        .expect("Character-centered graph should generate");

    // Verify file was created
    assert!(
        std::path::Path::new(&response.file_path).exists(),
        "Graph file should exist"
    );

    // Read and verify content includes Center and Near but not Far (depth 2)
    let content = std::fs::read_to_string(&response.file_path).expect("Read graph file");
    assert!(content.contains("Center"), "Should contain Center");
    assert!(content.contains("Near"), "Should contain Near at depth 1");
    // Far is at depth 2, should NOT be in depth-1 graph
    assert!(
        !content.contains("Far"),
        "Should NOT contain Far in depth-1 graph"
    );

    // Cleanup
    let _ = std::fs::remove_file(&response.file_path);
}

/// Test generate_graph with invalid scope returns descriptive error.
#[tokio::test]
async fn test_generate_graph_invalid_scope() {
    let harness = TestHarness::new().await;

    // Create server
    let session_path = harness.temp_path().join("session.json");
    let session_manager = Arc::new(SessionStateManager::load_or_create(&session_path).unwrap());
    let server = NarraServer::new(
        harness.db.clone(),
        session_manager,
        Arc::new(NoopEmbeddingService::new()),
    )
    .await;

    // Try invalid scope format
    let request = GraphRequest {
        scope: "invalid_scope_format".to_string(),
        depth: None,
        include_roles: None,
        filename: None,
    };

    let result = server.handle_generate_graph(Parameters(request)).await;

    // Should return error
    assert!(result.is_err(), "Invalid scope should return error");
    let error = result.unwrap_err();
    assert!(
        error.contains("Invalid scope"),
        "Error should mention 'Invalid scope', got: {}",
        error
    );
    assert!(
        error.contains("full") || error.contains("character:"),
        "Error should mention valid formats, got: {}",
        error
    );
}

// ============================================================================
// GENERATE_GRAPH TESTS - OPTIONS
// ============================================================================

/// Test generate_graph with include_roles=true includes roles in output.
#[tokio::test]
async fn test_generate_graph_with_roles() {
    let harness = TestHarness::new().await;
    let entity_repo = SurrealEntityRepository::new(harness.db.clone());

    // Create character with specific roles
    let _character = entity_repo
        .create_character(
            CharacterBuilder::new("Detective")
                .role("protagonist")
                .role("investigator")
                .build(),
        )
        .await
        .expect("Detective");

    // Create server
    let session_path = harness.temp_path().join("session.json");
    let session_manager = Arc::new(SessionStateManager::load_or_create(&session_path).unwrap());
    let server = NarraServer::new(
        harness.db.clone(),
        session_manager,
        Arc::new(NoopEmbeddingService::new()),
    )
    .await;

    // Generate graph with roles
    let request = GraphRequest {
        scope: "full".to_string(),
        depth: None,
        include_roles: Some(true),
        filename: Some("test_roles_graph.md".to_string()),
    };

    let response = server
        .handle_generate_graph(Parameters(request))
        .await
        .expect("Graph with roles should generate");

    // Verify file contents include role information
    let content = std::fs::read_to_string(&response.file_path).expect("Read graph file");
    assert!(
        content.contains("Detective"),
        "Should contain character name"
    );
    assert!(
        content.contains("protagonist") || content.contains("investigator"),
        "Should contain role information when include_roles=true"
    );

    // Cleanup
    let _ = std::fs::remove_file(&response.file_path);
}

/// Test generate_graph uses custom filename when provided.
#[tokio::test]
async fn test_generate_graph_custom_filename() {
    let harness = TestHarness::new().await;
    let entity_repo = SurrealEntityRepository::new(harness.db.clone());

    // Create a character
    let _alice = entity_repo
        .create_character(CharacterBuilder::new("Alice").build())
        .await
        .expect("Alice");

    // Create server
    let session_path = harness.temp_path().join("session.json");
    let session_manager = Arc::new(SessionStateManager::load_or_create(&session_path).unwrap());
    let server = NarraServer::new(
        harness.db.clone(),
        session_manager,
        Arc::new(NoopEmbeddingService::new()),
    )
    .await;

    let custom_name = "my_custom_graph_filename.md";

    // Generate graph with custom filename
    let request = GraphRequest {
        scope: "full".to_string(),
        depth: None,
        include_roles: None,
        filename: Some(custom_name.to_string()),
    };

    let response = server
        .handle_generate_graph(Parameters(request))
        .await
        .expect("Graph should generate with custom filename");

    // Verify custom filename was used
    assert!(
        response.file_path.contains(custom_name),
        "File path '{}' should contain custom filename '{}'",
        response.file_path,
        custom_name
    );

    // Verify file exists at expected location
    assert!(
        std::path::Path::new(&response.file_path).exists(),
        "File should exist at custom path"
    );

    // Cleanup
    let _ = std::fs::remove_file(&response.file_path);
}

// ============================================================================
// RESPONSE STRUCTURE TESTS
// ============================================================================

/// Test generate_graph response includes all expected fields.
#[tokio::test]
async fn test_generate_graph_response_fields() {
    let harness = TestHarness::new().await;
    let entity_repo = SurrealEntityRepository::new(harness.db.clone());

    // Create characters with relationship
    let alice = entity_repo
        .create_character(CharacterBuilder::new("Alice").build())
        .await
        .expect("Alice");
    let bob = entity_repo
        .create_character(CharacterBuilder::new("Bob").build())
        .await
        .expect("Bob");

    create_perception(
        &harness.db,
        &alice.id.key().to_string(),
        &bob.id.key().to_string(),
        PerceptionCreate {
            rel_types: vec!["friendship".to_string()],
            subtype: None,
            feelings: None,
            perception: None,
            tension_level: None,
            history_notes: None,
        },
    )
    .await
    .expect("Alice->Bob");

    // Create server
    let session_path = harness.temp_path().join("session.json");
    let session_manager = Arc::new(SessionStateManager::load_or_create(&session_path).unwrap());
    let server = NarraServer::new(
        harness.db.clone(),
        session_manager,
        Arc::new(NoopEmbeddingService::new()),
    )
    .await;

    let request = GraphRequest {
        scope: "full".to_string(),
        depth: None,
        include_roles: None,
        filename: Some("test_response_fields.md".to_string()),
    };

    let response = server
        .handle_generate_graph(Parameters(request))
        .await
        .expect("Graph should generate");

    // Verify all GraphResponse fields
    assert!(
        !response.file_path.is_empty(),
        "file_path should be populated"
    );
    assert!(
        response.character_count >= 2,
        "character_count should reflect entities in graph"
    );
    assert!(
        response.relationship_count >= 1,
        "relationship_count should reflect edges in graph"
    );
    assert!(
        !response.hints.is_empty(),
        "hints should provide helpful guidance"
    );

    // Cleanup
    let _ = std::fs::remove_file(&response.file_path);
}

/// Test GraphTraversal response structure.
#[tokio::test]
async fn test_graph_traversal_response_structure() {
    let harness = TestHarness::new().await;
    let entity_repo = SurrealEntityRepository::new(harness.db.clone());

    // Create connected characters
    let alice = entity_repo
        .create_character(CharacterBuilder::new("Alice").build())
        .await
        .expect("Alice");
    let bob = entity_repo
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
            label: None,
        },
    )
    .await
    .expect("Relationship");

    // Create server
    let session_path = harness.temp_path().join("session.json");
    let session_manager = Arc::new(SessionStateManager::load_or_create(&session_path).unwrap());
    let server = NarraServer::new(
        harness.db.clone(),
        session_manager,
        Arc::new(NoopEmbeddingService::new()),
    )
    .await;

    let request = QueryRequest::GraphTraversal {
        entity_id: format!("character:{}", alice.id.key()),
        depth: 2,
        format: None,
    };

    let response = server
        .handle_query(Parameters(request))
        .await
        .expect("Traversal should succeed");

    // Verify QueryResponse structure for graph traversal
    assert!(response.total >= 1, "total should count connected entities");
    assert!(
        response.next_cursor.is_none(),
        "graph traversal should not paginate"
    );
    assert!(!response.hints.is_empty(), "hints should provide guidance");
    assert!(
        response.token_estimate > 0,
        "token_estimate should be calculated"
    );

    // Verify entity results have expected fields
    for result in &response.results {
        assert!(!result.id.is_empty(), "Entity result should have id");
        assert!(
            !result.entity_type.is_empty(),
            "Entity result should have entity_type"
        );
    }
}
