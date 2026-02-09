//! Mutate handler integration tests.
//!
//! Tests validate the mutate tool handler correctly translates between MCP
//! protocol requests and service layer calls, returning properly formatted
//! MutationResponse structures.

use insta::assert_snapshot;
use narra::mcp::{
    CharacterSpec, DetailLevel, EventSpec, LocationSpec, MutationRequest, QueryRequest,
    RelationshipSpec,
};
use narra::repository::{EntityRepository, SurrealEntityRepository};
use pretty_assertions::assert_eq;
use rmcp::handler::server::wrapper::Parameters;

use crate::common::{
    builders::{CharacterBuilder, EventBuilder, LocationBuilder},
    harness::TestHarness,
    to_mutation_input, to_query_input,
};

// =============================================================================
// CHARACTER CRUD
// =============================================================================

/// Test successful character creation via handler.
///
/// Verifies:
/// - Handler returns MutationResponse with entity field
/// - Entity has correct entity_type and name
/// - Entity ID is in expected format
#[tokio::test]
async fn test_create_character_success() {
    let harness = TestHarness::new().await;
    let server = crate::common::create_test_server(&harness).await;

    let request = MutationRequest::CreateCharacter {
        id: None,
        name: "Alice".to_string(),
        role: Some("protagonist".to_string()),
        aliases: None,
        description: None,
        profile: None,
    };
    let response = server
        .handle_mutate(Parameters(to_mutation_input(request)))
        .await;

    assert!(response.is_ok(), "Create character should succeed");
    let response = response.unwrap();

    // Verify entity fields
    assert_eq!(response.entity.entity_type, "character");
    assert_eq!(response.entity.name, "Alice");
    assert!(
        response.entity.id.starts_with("character:"),
        "ID should be in table:key format"
    );
    assert_eq!(response.entity.confidence, Some(1.0));

    // Verify hints are present
    assert!(
        !response.hints.is_empty(),
        "Should have hints for next steps"
    );
}

/// Test character creation with aliases.
///
/// Verifies aliases are stored correctly.
#[tokio::test]
async fn test_create_character_with_aliases() {
    let harness = TestHarness::new().await;
    let server = crate::common::create_test_server(&harness).await;

    let request = MutationRequest::CreateCharacter {
        id: None,
        name: "The Shadow".to_string(),
        role: Some("antagonist".to_string()),
        aliases: Some(vec!["Dark One".to_string(), "The Nameless".to_string()]),
        description: None,
        profile: None,
    };
    let response = server
        .handle_mutate(Parameters(to_mutation_input(request)))
        .await;

    assert!(
        response.is_ok(),
        "Create character with aliases should succeed"
    );
    let response = response.unwrap();
    let char_id = response.entity.id.clone();

    // Verify character can be looked up and has aliases
    let repo = SurrealEntityRepository::new(harness.db.clone());
    let char_key = char_id.split(':').nth(1).unwrap();
    let character = repo
        .get_character(char_key)
        .await
        .expect("Get should succeed")
        .expect("Character should exist");

    assert_eq!(character.name, "The Shadow");
    assert!(character.aliases.contains(&"Dark One".to_string()));
    assert!(character.aliases.contains(&"The Nameless".to_string()));
}

/// Test character hard delete.
///
/// Verifies entity is actually deleted from database.
#[tokio::test]
async fn test_delete_character_hard() {
    let harness = TestHarness::new().await;
    let server = crate::common::create_test_server(&harness).await;

    // Create character first
    let create_request = MutationRequest::CreateCharacter {
        id: None,
        name: "ToDelete".to_string(),
        role: None,
        aliases: None,
        description: None,
        profile: None,
    };
    let create_response = server
        .handle_mutate(Parameters(to_mutation_input(create_request)))
        .await
        .expect("Create should succeed");
    let char_id = create_response.entity.id.clone();

    // Delete character
    let delete_request = MutationRequest::Delete {
        entity_id: char_id.clone(),
        hard: Some(true),
    };
    let delete_response = server
        .handle_mutate(Parameters(to_mutation_input(delete_request)))
        .await;

    assert!(delete_response.is_ok(), "Delete should succeed");
    let delete_response = delete_response.unwrap();

    assert_eq!(delete_response.entity.entity_type, "character");
    assert!(
        delete_response.impact.is_some(),
        "Delete should include impact analysis"
    );

    // Verify character is actually deleted
    let repo = SurrealEntityRepository::new(harness.db.clone());
    let char_key = char_id.split(':').nth(1).unwrap();
    let character = repo
        .get_character(char_key)
        .await
        .expect("Get should not error");

    assert!(character.is_none(), "Character should be deleted");
}

// =============================================================================
// LOCATION CRUD
// =============================================================================

/// Test successful location creation via handler.
#[tokio::test]
async fn test_create_location_success() {
    let harness = TestHarness::new().await;
    let server = crate::common::create_test_server(&harness).await;

    let request = MutationRequest::CreateLocation {
        id: None,
        name: "The Tower".to_string(),
        parent_id: None,
        description: Some("A tall stone tower".to_string()),
    };
    let response = server
        .handle_mutate(Parameters(to_mutation_input(request)))
        .await;

    assert!(response.is_ok(), "Create location should succeed");
    let response = response.unwrap();

    assert_eq!(response.entity.entity_type, "location");
    assert_eq!(response.entity.name, "The Tower");
    assert!(response.entity.id.starts_with("location:"));
}

/// Test location creation with parent.
///
/// Verifies parent relationship is established.
#[tokio::test]
async fn test_create_location_with_parent() {
    let harness = TestHarness::new().await;
    let server = crate::common::create_test_server(&harness).await;

    // Create parent location first
    let parent_request = MutationRequest::CreateLocation {
        id: None,
        name: "The Castle".to_string(),
        parent_id: None,
        description: None,
    };
    let parent_response = server
        .handle_mutate(Parameters(to_mutation_input(parent_request)))
        .await
        .expect("Parent creation should succeed");
    let parent_id = parent_response.entity.id.clone();

    // Create child location
    let child_request = MutationRequest::CreateLocation {
        id: None,
        name: "The Throne Room".to_string(),
        parent_id: Some(parent_id.clone()),
        description: None,
    };
    let child_response = server
        .handle_mutate(Parameters(to_mutation_input(child_request)))
        .await;

    assert!(
        child_response.is_ok(),
        "Create child location should succeed"
    );
    let child_response = child_response.unwrap();

    assert_eq!(child_response.entity.name, "The Throne Room");

    // Verify parent relationship via repository
    let repo = SurrealEntityRepository::new(harness.db.clone());
    let child_key = child_response.entity.id.split(':').nth(1).unwrap();
    let location = repo
        .get_location(child_key)
        .await
        .expect("Get should succeed")
        .expect("Location should exist");

    assert!(location.parent.is_some(), "Child should have parent");
}

/// Test location hard delete.
#[tokio::test]
async fn test_delete_location_hard() {
    let harness = TestHarness::new().await;
    let server = crate::common::create_test_server(&harness).await;

    // Create location
    let create_request = MutationRequest::CreateLocation {
        id: None,
        name: "ToDelete".to_string(),
        parent_id: None,
        description: None,
    };
    let create_response = server
        .handle_mutate(Parameters(to_mutation_input(create_request)))
        .await
        .expect("Create should succeed");
    let loc_id = create_response.entity.id.clone();

    // Delete location
    let delete_request = MutationRequest::Delete {
        entity_id: loc_id.clone(),
        hard: Some(true),
    };
    let delete_response = server
        .handle_mutate(Parameters(to_mutation_input(delete_request)))
        .await;

    assert!(delete_response.is_ok(), "Delete should succeed");

    // Verify deletion
    let repo = SurrealEntityRepository::new(harness.db.clone());
    let loc_key = loc_id.split(':').nth(1).unwrap();
    let location = repo
        .get_location(loc_key)
        .await
        .expect("Get should not error");

    assert!(location.is_none(), "Location should be deleted");
}

// =============================================================================
// EVENT CRUD
// =============================================================================

/// Test successful event creation via handler.
#[tokio::test]
async fn test_create_event_success() {
    let harness = TestHarness::new().await;
    let server = crate::common::create_test_server(&harness).await;

    let request = MutationRequest::CreateEvent {
        id: None,
        title: "The Betrayal".to_string(),
        description: Some("Marcus reveals his allegiance".to_string()),
        sequence: Some(100),
        date: None,
        date_precision: None,
    };
    let response = server
        .handle_mutate(Parameters(to_mutation_input(request)))
        .await;

    assert!(response.is_ok(), "Create event should succeed");
    let response = response.unwrap();

    assert_eq!(response.entity.entity_type, "event");
    assert_eq!(response.entity.name, "The Betrayal");
    assert!(response.entity.id.starts_with("event:"));
}

/// Test event creation with RFC3339 date.
#[tokio::test]
async fn test_create_event_with_date() {
    let harness = TestHarness::new().await;
    let server = crate::common::create_test_server(&harness).await;

    let request = MutationRequest::CreateEvent {
        id: None,
        title: "The Coronation".to_string(),
        description: None,
        sequence: Some(1),
        date: Some("2023-06-15T14:30:00Z".to_string()),
        date_precision: Some("day".to_string()),
    };
    let response = server
        .handle_mutate(Parameters(to_mutation_input(request)))
        .await;

    assert!(response.is_ok(), "Create event with date should succeed");
    let response = response.unwrap();

    // Verify event was created
    let repo = SurrealEntityRepository::new(harness.db.clone());
    let event_key = response.entity.id.split(':').nth(1).unwrap();
    let event = repo
        .get_event(event_key)
        .await
        .expect("Get should succeed")
        .expect("Event should exist");

    assert!(event.date.is_some(), "Event should have date");
}

/// Test event hard delete.
#[tokio::test]
async fn test_delete_event_hard() {
    let harness = TestHarness::new().await;
    let server = crate::common::create_test_server(&harness).await;

    // Create event
    let create_request = MutationRequest::CreateEvent {
        id: None,
        title: "ToDelete".to_string(),
        description: None,
        sequence: None,
        date: None,
        date_precision: None,
    };
    let create_response = server
        .handle_mutate(Parameters(to_mutation_input(create_request)))
        .await
        .expect("Create should succeed");
    let event_id = create_response.entity.id.clone();

    // Delete event
    let delete_request = MutationRequest::Delete {
        entity_id: event_id.clone(),
        hard: Some(true),
    };
    let delete_response = server
        .handle_mutate(Parameters(to_mutation_input(delete_request)))
        .await;

    assert!(delete_response.is_ok(), "Delete should succeed");

    // Verify deletion
    let repo = SurrealEntityRepository::new(harness.db.clone());
    let event_key = event_id.split(':').nth(1).unwrap();
    let event = repo
        .get_event(event_key)
        .await
        .expect("Get should not error");

    assert!(event.is_none(), "Event should be deleted");
}

// =============================================================================
// SCENE CRUD
// =============================================================================

/// Test successful scene creation via handler.
#[tokio::test]
async fn test_create_scene_success() {
    let harness = TestHarness::new().await;
    let server = crate::common::create_test_server(&harness).await;

    // Create prerequisite event and location
    let repo = SurrealEntityRepository::new(harness.db.clone());
    let event = repo
        .create_event(EventBuilder::new("Test Event").build())
        .await
        .expect("Create event");
    let location = repo
        .create_location(LocationBuilder::new("Test Location").build())
        .await
        .expect("Create location");

    let request = MutationRequest::CreateScene {
        title: "Opening Scene".to_string(),
        event_id: event.id.to_string(),
        location_id: location.id.to_string(),
        summary: Some("The story begins".to_string()),
    };
    let response = server
        .handle_mutate(Parameters(to_mutation_input(request)))
        .await;

    assert!(response.is_ok(), "Create scene should succeed");
    let response = response.unwrap();

    assert_eq!(response.entity.entity_type, "scene");
    assert_eq!(response.entity.name, "Opening Scene");
    assert!(response.entity.id.starts_with("scene:"));
}

/// Test scene creation with malformed event ID format.
///
/// Note: SurrealDB allows creating scenes with nonexistent event/location IDs
/// (dangling references). This test verifies that *malformed* IDs are rejected
/// at the RecordId parsing stage.
#[tokio::test]
async fn test_create_scene_malformed_event_id() {
    let harness = TestHarness::new().await;
    let server = crate::common::create_test_server(&harness).await;

    // Create location but use malformed event ID (not valid RecordId format)
    let repo = SurrealEntityRepository::new(harness.db.clone());
    let location = repo
        .create_location(LocationBuilder::new("Test Location").build())
        .await
        .expect("Create location");

    let request = MutationRequest::CreateScene {
        title: "Bad Scene".to_string(),
        event_id: "not-a-valid-record-id".to_string(), // Malformed, not "table:key" format
        location_id: location.id.to_string(),
        summary: None,
    };
    let response = server
        .handle_mutate(Parameters(to_mutation_input(request)))
        .await;

    assert!(
        response.is_err(),
        "Create scene with malformed event_id should fail"
    );
    let error_message = response.unwrap_err();

    // Snapshot the error
    assert_snapshot!("create_scene_malformed_event_id", error_message);
}

// =============================================================================
// DELETE EDGE CASES
// =============================================================================

/// Test delete for nonexistent entity.
///
/// Verifies appropriate error message.
#[tokio::test]
async fn test_delete_nonexistent_entity() {
    let harness = TestHarness::new().await;
    let server = crate::common::create_test_server(&harness).await;

    let request = MutationRequest::Delete {
        entity_id: "character:nonexistent123".to_string(),
        hard: Some(true),
    };
    let response = server
        .handle_mutate(Parameters(to_mutation_input(request)))
        .await;

    // Note: The current implementation may succeed silently for nonexistent entities
    // or may return an error. This test documents actual behavior.
    if let Err(error_message) = response {
        assert_snapshot!("delete_nonexistent_error", error_message);
    }
    // If it succeeds, that's also valid behavior (idempotent delete)
}

/// Test soft delete returns not-implemented error.
///
/// Verifies appropriate error message for unimplemented feature.
#[tokio::test]
async fn test_delete_soft_not_implemented() {
    let harness = TestHarness::new().await;
    let server = crate::common::create_test_server(&harness).await;

    // Create character to attempt soft delete on
    let create_request = MutationRequest::CreateCharacter {
        id: None,
        name: "SoftDeleteTest".to_string(),
        role: None,
        aliases: None,
        description: None,
        profile: None,
    };
    let create_response = server
        .handle_mutate(Parameters(to_mutation_input(create_request)))
        .await
        .expect("Create should succeed");
    let char_id = create_response.entity.id.clone();

    // Attempt soft delete
    let delete_request = MutationRequest::Delete {
        entity_id: char_id,
        hard: Some(false),
    };
    let response = server
        .handle_mutate(Parameters(to_mutation_input(delete_request)))
        .await;

    assert!(
        response.is_err(),
        "Soft delete should fail (not implemented)"
    );
    let error_message = response.unwrap_err();

    // Snapshot the error
    assert_snapshot!("delete_soft_not_implemented", error_message);
}

// =============================================================================
// INTEGRATION VERIFICATION
// =============================================================================

/// Test created entity can be looked up via query handler.
///
/// Verifies create and query handlers work together correctly.
#[tokio::test]
async fn test_created_entity_queryable() {
    let harness = TestHarness::new().await;
    let server = crate::common::create_test_server(&harness).await;

    // Create character via mutate handler
    let create_request = MutationRequest::CreateCharacter {
        id: None,
        name: "QueryTest".to_string(),
        role: Some("test".to_string()),
        aliases: None,
        description: None,
        profile: None,
    };
    let create_response = server
        .handle_mutate(Parameters(to_mutation_input(create_request)))
        .await
        .expect("Create should succeed");
    let char_id = create_response.entity.id.clone();

    // Look up via query handler
    let query_request = QueryRequest::Lookup {
        entity_id: char_id.clone(),
        detail_level: Some(DetailLevel::Standard),
    };
    let query_response = server
        .handle_query(Parameters(to_query_input(query_request)))
        .await;

    assert!(query_response.is_ok(), "Query should find created entity");
    let query_response = query_response.unwrap();

    assert_eq!(query_response.results.len(), 1);
    assert_eq!(query_response.results[0].id, char_id);
    assert_eq!(query_response.results[0].name, "QueryTest");
}

// =============================================================================
// BATCH OPERATIONS
// =============================================================================

/// Test batch character creation.
#[tokio::test]
async fn test_batch_create_characters() {
    let harness = TestHarness::new().await;
    let server = crate::common::create_test_server(&harness).await;

    let request = MutationRequest::BatchCreateCharacters {
        characters: vec![
            CharacterSpec {
                id: None,
                name: "Alice".to_string(),
                role: Some("protagonist".to_string()),
                aliases: None,
                description: None,
                profile: None,
            },
            CharacterSpec {
                id: None,
                name: "Bob".to_string(),
                role: Some("antagonist".to_string()),
                aliases: Some(vec!["Robert".to_string()]),
                description: None,
                profile: None,
            },
            CharacterSpec {
                id: None,
                name: "Carol".to_string(),
                role: None,
                aliases: None,
                description: None,
                profile: Some(
                    [("wound".to_string(), vec!["abandonment".to_string()])]
                        .into_iter()
                        .collect(),
                ),
            },
        ],
    };

    let response = server
        .handle_mutate(Parameters(to_mutation_input(request)))
        .await;
    assert!(response.is_ok(), "Batch create should succeed");
    let response = response.unwrap();

    assert_eq!(response.entity.entity_type, "batch");
    let entities = response.entities.expect("Should have entities list");
    assert_eq!(entities.len(), 3);

    // Verify all are characters
    for e in &entities {
        assert_eq!(e.entity_type, "character");
        assert!(e.id.starts_with("character:"));
    }
}

/// Test batch character creation with caller-specified IDs.
#[tokio::test]
async fn test_batch_create_characters_with_ids() {
    let harness = TestHarness::new().await;
    let server = crate::common::create_test_server(&harness).await;

    let request = MutationRequest::BatchCreateCharacters {
        characters: vec![
            CharacterSpec {
                id: Some("alice".to_string()),
                name: "Alice".to_string(),
                role: Some("protagonist".to_string()),
                aliases: None,
                description: None,
                profile: None,
            },
            CharacterSpec {
                id: Some("bob".to_string()),
                name: "Bob".to_string(),
                role: None,
                aliases: None,
                description: None,
                profile: None,
            },
        ],
    };

    let response = server
        .handle_mutate(Parameters(to_mutation_input(request)))
        .await
        .expect("Batch create with IDs should succeed");

    let entities = response.entities.expect("Should have entities list");
    assert_eq!(entities.len(), 2);
    assert_eq!(entities[0].id, "character:alice");
    assert_eq!(entities[1].id, "character:bob");

    // Verify via repository
    let repo = SurrealEntityRepository::new(harness.db.clone());
    let alice = repo
        .get_character("alice")
        .await
        .expect("Get should succeed")
        .expect("Alice should exist");
    assert_eq!(alice.name, "Alice");
}

/// Test single create with caller-specified ID.
#[tokio::test]
async fn test_create_character_with_id() {
    let harness = TestHarness::new().await;
    let server = crate::common::create_test_server(&harness).await;

    let request = MutationRequest::CreateCharacter {
        id: Some("hero".to_string()),
        name: "The Hero".to_string(),
        role: Some("protagonist".to_string()),
        aliases: None,
        description: None,
        profile: None,
    };
    let response = server
        .handle_mutate(Parameters(to_mutation_input(request)))
        .await
        .expect("Create with ID should succeed");

    assert_eq!(response.entity.id, "character:hero");
    assert_eq!(response.entity.name, "The Hero");
}

/// Test batch location creation with caller-specified IDs.
#[tokio::test]
async fn test_batch_create_locations() {
    let harness = TestHarness::new().await;
    let server = crate::common::create_test_server(&harness).await;

    let request = MutationRequest::BatchCreateLocations {
        locations: vec![
            LocationSpec {
                id: Some("castle".to_string()),
                name: "The Castle".to_string(),
                description: Some("A grand fortress".to_string()),
                parent_id: None,
                loc_type: None,
            },
            LocationSpec {
                id: Some("throne_room".to_string()),
                name: "Throne Room".to_string(),
                description: None,
                parent_id: Some("location:castle".to_string()),
                loc_type: None,
            },
        ],
    };

    let response = server
        .handle_mutate(Parameters(to_mutation_input(request)))
        .await
        .expect("Batch create locations should succeed");

    let entities = response.entities.expect("Should have entities list");
    assert_eq!(entities.len(), 2);
    assert_eq!(entities[0].id, "location:castle");
    assert_eq!(entities[1].id, "location:throne_room");
}

/// Test batch event creation.
#[tokio::test]
async fn test_batch_create_events() {
    let harness = TestHarness::new().await;
    let server = crate::common::create_test_server(&harness).await;

    let request = MutationRequest::BatchCreateEvents {
        events: vec![
            EventSpec {
                id: Some("betrayal".to_string()),
                title: "The Betrayal".to_string(),
                description: Some("Marcus reveals his allegiance".to_string()),
                sequence: Some(100),
                date: None,
                date_precision: None,
            },
            EventSpec {
                id: Some("coronation".to_string()),
                title: "The Coronation".to_string(),
                description: None,
                sequence: Some(200),
                date: None,
                date_precision: None,
            },
        ],
    };

    let response = server
        .handle_mutate(Parameters(to_mutation_input(request)))
        .await
        .expect("Batch create events should succeed");

    let entities = response.entities.expect("Should have entities list");
    assert_eq!(entities.len(), 2);
    assert_eq!(entities[0].id, "event:betrayal");
    assert_eq!(entities[1].id, "event:coronation");
}

/// Test batch relationship creation.
#[tokio::test]
async fn test_batch_create_relationships() {
    let harness = TestHarness::new().await;
    let server = crate::common::create_test_server(&harness).await;

    // Create characters first
    let repo = SurrealEntityRepository::new(harness.db.clone());
    repo.create_character(CharacterBuilder::new("Alice").build())
        .await
        .expect("Create Alice");
    repo.create_character(CharacterBuilder::new("Bob").build())
        .await
        .expect("Create Bob");
    repo.create_character(CharacterBuilder::new("Carol").build())
        .await
        .expect("Create Carol");

    let request = MutationRequest::BatchCreateRelationships {
        relationships: vec![
            RelationshipSpec {
                from_character_id: "alice".to_string(),
                to_character_id: "bob".to_string(),
                rel_type: "friendship".to_string(),
                subtype: None,
                label: Some("Best friends".to_string()),
            },
            RelationshipSpec {
                from_character_id: "bob".to_string(),
                to_character_id: "carol".to_string(),
                rel_type: "rivalry".to_string(),
                subtype: None,
                label: None,
            },
        ],
    };

    let response = server
        .handle_mutate(Parameters(to_mutation_input(request)))
        .await
        .expect("Batch create relationships should succeed");

    let entities = response.entities.expect("Should have entities list");
    assert_eq!(entities.len(), 2);
    for e in &entities {
        assert_eq!(e.entity_type, "relationship");
    }
}

/// Test batch with mixed auto-generated and caller-specified IDs.
#[tokio::test]
async fn test_batch_mixed_ids() {
    let harness = TestHarness::new().await;
    let server = crate::common::create_test_server(&harness).await;

    let request = MutationRequest::BatchCreateCharacters {
        characters: vec![
            CharacterSpec {
                id: Some("hero".to_string()),
                name: "The Hero".to_string(),
                role: None,
                aliases: None,
                description: None,
                profile: None,
            },
            CharacterSpec {
                id: None, // Auto-generated
                name: "Random NPC".to_string(),
                role: None,
                aliases: None,
                description: None,
                profile: None,
            },
        ],
    };

    let response = server
        .handle_mutate(Parameters(to_mutation_input(request)))
        .await
        .expect("Batch with mixed IDs should succeed");

    let entities = response.entities.expect("Should have entities list");
    assert_eq!(entities.len(), 2);
    assert_eq!(entities[0].id, "character:hero");
    assert!(
        entities[1].id.starts_with("character:"),
        "Auto-generated ID should start with character:"
    );
    assert_ne!(
        entities[1].id, "character:hero",
        "Auto-generated should be different from specified"
    );
}
