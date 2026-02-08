//! Integration tests for Phase 17 referential integrity features.
//!
//! Tests verify:
//! - CASCADE deletion behavior (owned data automatically deleted)
//! - REJECT deletion behavior (protected references prevent deletion)
//! - UNSET behavior (arrays cleaned up when items deleted)
//! - Reverse queries (finding entities that reference a target)
//! - Connection path finding (BFS graph traversal)

mod common;

use common::builders::{CharacterBuilder, EventBuilder, LocationBuilder, SceneBuilder};
use common::harness::TestHarness;
use narra::models::character::{create_character, delete_character};
use narra::models::event::create_event;
use narra::models::knowledge::{create_knowledge, KnowledgeCreate};
use narra::models::location::create_location;
#[cfg(feature = "surrealdb3")]
use narra::models::location::delete_location;
use narra::models::perception::{create_perception, PerceptionCreate};
use narra::models::scene::create_scene;
#[cfg(feature = "surrealdb3")]
use narra::models::scene::get_scene;
use narra::services::graph::MermaidGraphService;
#[cfg(feature = "surrealdb3")]
use narra::NarraError;

/// Test cascade delete: deleting character removes their knowledge.
///
/// Verifies: DATA-01 CASCADE behavior for knowledge.character field
#[tokio::test]
async fn test_cascade_delete_character_removes_knowledge() {
    let harness = TestHarness::new().await;

    // Create character with knowledge
    let alice = create_character(&harness.db, CharacterBuilder::new("Alice").build())
        .await
        .expect("Should create Alice");

    let knowledge = create_knowledge(
        &harness.db,
        KnowledgeCreate {
            character: alice.id.clone(),
            fact: "The sky is blue".to_string(),
        },
    )
    .await
    .expect("Should create knowledge");

    println!("Created Alice and knowledge fact");

    // Delete the character
    let alice_key = alice.id.key().to_string();
    delete_character(&harness.db, &alice_key)
        .await
        .expect("Should delete Alice");

    // Verify knowledge is also deleted (CASCADE behavior)
    let query = format!("SELECT * FROM knowledge:{}", knowledge.id.key());
    let mut result = harness
        .db
        .query(&query)
        .await
        .expect("Query should succeed");
    let remaining: Vec<surrealdb::sql::Value> = result.take(0).unwrap_or_default();

    assert!(
        remaining.is_empty(),
        "Knowledge should be cascade-deleted when character is deleted"
    );

    println!("✓ Knowledge was cascade-deleted with character");
}

/// Test cascade delete: deleting character removes their relationships.
///
/// Verifies: DATA-01 CASCADE behavior for perceives edges
#[tokio::test]
async fn test_cascade_delete_character_removes_relationships() {
    let harness = TestHarness::new().await;

    // Create two characters with a perception relationship
    let alice = create_character(&harness.db, CharacterBuilder::new("Alice").build())
        .await
        .expect("Should create Alice");

    let bob = create_character(&harness.db, CharacterBuilder::new("Bob").build())
        .await
        .expect("Should create Bob");

    let alice_key = alice.id.key().to_string();
    let bob_key = bob.id.key().to_string();

    let perception = create_perception(
        &harness.db,
        &alice_key,
        &bob_key,
        PerceptionCreate {
            rel_types: vec!["friendship".to_string()],
            subtype: None,
            feelings: Some("warm".to_string()),
            perception: None,
            tension_level: Some(5),
            history_notes: None,
        },
    )
    .await
    .expect("Should create perception");

    println!("Created Alice -> Bob friendship perception");

    // Delete Alice
    delete_character(&harness.db, &alice_key)
        .await
        .expect("Should delete Alice");

    // Verify the perceives edge is also deleted (CASCADE behavior)
    let query = format!("SELECT * FROM perceives:{}", perception.id.key());
    let mut result = harness
        .db
        .query(&query)
        .await
        .expect("Query should succeed");
    let remaining: Vec<surrealdb::sql::Value> = result.take(0).unwrap_or_default();

    assert!(
        remaining.is_empty(),
        "Perceives edge should be cascade-deleted when character is deleted"
    );

    println!("✓ Perception edge was cascade-deleted with character");
}

/// Test cascade delete: deleting character removes their participations.
///
/// Verifies: DATA-01 CASCADE behavior for participates_in edges
#[tokio::test]
async fn test_cascade_delete_character_removes_participations() {
    let harness = TestHarness::new().await;

    // Create character, location, event, and scene
    let alice = create_character(&harness.db, CharacterBuilder::new("Alice").build())
        .await
        .expect("Should create Alice");

    let location = create_location(&harness.db, LocationBuilder::new("The Plaza").build())
        .await
        .expect("Should create location");

    let event = create_event(
        &harness.db,
        EventBuilder::new("The Meeting").sequence(100).build(),
    )
    .await
    .expect("Should create event");

    let alice_key = alice.id.key().to_string();
    let event_key = event.id.key().to_string();
    let location_key = location.id.key().to_string();

    let scene = create_scene(
        &harness.db,
        SceneBuilder::new("Opening Scene", &event_key, &location_key).build(),
    )
    .await
    .expect("Should create scene");

    let scene_key = scene.id.key().to_string();

    // Create participates_in edge (SCHEMAFULL table requires 'role' field)
    let query = format!(
        "RELATE character:{} -> participates_in -> scene:{} SET role = 'participant'",
        alice_key, scene_key
    );
    harness
        .db
        .query(&query)
        .await
        .expect("Should create participation edge");

    println!("Created character participation in scene");

    // Delete the character
    delete_character(&harness.db, &alice_key)
        .await
        .expect("Should delete Alice");

    // Verify the participates_in edge is deleted (CASCADE behavior)
    let query = format!(
        "SELECT * FROM participates_in WHERE in = character:{} AND out = scene:{}",
        alice_key, scene_key
    );
    let mut result = harness
        .db
        .query(&query)
        .await
        .expect("Query should succeed");
    let edges: Vec<surrealdb::sql::Value> = result.take(0).unwrap_or_default();

    assert!(
        edges.is_empty(),
        "Participates_in edge should be cascade-deleted when character is deleted"
    );

    println!("✓ Participation edge was cascade-deleted with character");
}

/// Test reject delete: deleting location with scenes returns error.
///
/// Verifies: DATA-03 REJECT behavior for scene.primary_location field
#[tokio::test]
#[cfg(feature = "surrealdb3")]
async fn test_reject_delete_location_with_scenes() {
    let harness = TestHarness::new().await;

    // Create location, event, and scene using the location
    let location = create_location(&harness.db, LocationBuilder::new("The Tower").build())
        .await
        .expect("Should create location");

    let event = create_event(
        &harness.db,
        EventBuilder::new("The Battle").sequence(100).build(),
    )
    .await
    .expect("Should create event");

    let location_key = location.id.key().to_string();
    let event_key = event.id.key().to_string();

    let _scene = create_scene(
        &harness.db,
        SceneBuilder::new("The Siege", &event_key, &location_key).build(),
    )
    .await
    .expect("Should create scene");

    println!("Created location with scene referencing it");

    // Attempt to delete the location - should fail with ReferentialIntegrityViolation
    let result = delete_location(&harness.db, &location_key).await;

    assert!(
        result.is_err(),
        "Deleting location referenced in scenes should fail"
    );

    match result {
        Err(NarraError::ReferentialIntegrityViolation {
            entity_type,
            entity_id,
            message,
        }) => {
            assert_eq!(entity_type, "location");
            assert_eq!(entity_id, location_key);
            println!("✓ Got expected ReferentialIntegrityViolation: {}", message);
        }
        _ => panic!(
            "Expected ReferentialIntegrityViolation error, got: {:?}",
            result
        ),
    }
}

/// Test reject delete: deleting parent location returns error.
///
/// Verifies: DATA-03 REJECT behavior for location.parent field
#[tokio::test]
#[cfg(feature = "surrealdb3")]
async fn test_reject_delete_location_parent() {
    let harness = TestHarness::new().await;

    // Create parent and child locations
    let parent = create_location(&harness.db, LocationBuilder::new("The Kingdom").build())
        .await
        .expect("Should create parent location");

    let parent_key = parent.id.key().to_string();

    let _child = create_location(
        &harness.db,
        LocationBuilder::new("The Capital")
            .parent(&parent_key)
            .build(),
    )
    .await
    .expect("Should create child location");

    println!("Created parent location with child");

    // Attempt to delete the parent - should fail with ReferentialIntegrityViolation
    let result = delete_location(&harness.db, &parent_key).await;

    assert!(
        result.is_err(),
        "Deleting parent location should fail when children exist"
    );

    match result {
        Err(NarraError::ReferentialIntegrityViolation {
            entity_type,
            entity_id,
            message,
        }) => {
            assert_eq!(entity_type, "location");
            assert_eq!(entity_id, parent_key);
            println!("✓ Got expected ReferentialIntegrityViolation: {}", message);
        }
        _ => panic!(
            "Expected ReferentialIntegrityViolation error, got: {:?}",
            result
        ),
    }
}

/// Test unset: deleting location removes it from secondary_locations.
///
/// Verifies: DATA-02 UNSET behavior for scene.secondary_locations array
#[tokio::test]
#[cfg(feature = "surrealdb3")]
async fn test_unset_location_from_secondary_locations() {
    let harness = TestHarness::new().await;

    // Create two locations, event, and scene with secondary location
    let primary_loc = create_location(&harness.db, LocationBuilder::new("The Courtyard").build())
        .await
        .expect("Should create primary location");

    let secondary_loc = create_location(&harness.db, LocationBuilder::new("The Garden").build())
        .await
        .expect("Should create secondary location");

    let event = create_event(
        &harness.db,
        EventBuilder::new("The Festival").sequence(100).build(),
    )
    .await
    .expect("Should create event");

    let primary_key = primary_loc.id.key().to_string();
    let secondary_key = secondary_loc.id.key().to_string();
    let event_key = event.id.key().to_string();

    let scene = create_scene(
        &harness.db,
        SceneBuilder::new("The Celebration", &event_key, &primary_key)
            .secondary_location(&secondary_key)
            .build(),
    )
    .await
    .expect("Should create scene");

    println!("Created scene with primary and secondary locations");

    let scene_key = scene.id.key().to_string();

    // Verify secondary location is in the array
    let scene_data = get_scene(&harness.db, &scene_key)
        .await
        .expect("Should fetch scene")
        .expect("Scene should exist");
    assert_eq!(scene_data.secondary_locations.len(), 1);
    assert_eq!(scene_data.secondary_locations[0], secondary_loc.id);

    // Delete the secondary location
    delete_location(&harness.db, &secondary_key)
        .await
        .expect("Should delete secondary location (not referenced as primary)");

    // Verify secondary_locations array no longer contains the deleted location (UNSET behavior)
    let scene_data = get_scene(&harness.db, &scene_key)
        .await
        .expect("Should fetch scene")
        .expect("Scene should still exist");

    assert!(
        scene_data.secondary_locations.is_empty(),
        "Secondary location should be removed from array (UNSET behavior)"
    );

    println!("✓ Secondary location was removed from array via UNSET");
}

/// Test reverse query: find scenes for a character.
///
/// Verifies: DATA-04 reverse query capability via graph service
#[tokio::test]
async fn test_reverse_query_finds_scenes_for_character() {
    let harness = TestHarness::new().await;
    let graph_service = MermaidGraphService::new(harness.db.clone());

    // Create character, location, event, scene, and participation
    let alice = create_character(&harness.db, CharacterBuilder::new("Alice").build())
        .await
        .expect("Should create Alice");

    let location = create_location(&harness.db, LocationBuilder::new("The Plaza").build())
        .await
        .expect("Should create location");

    let event = create_event(
        &harness.db,
        EventBuilder::new("The Meeting").sequence(100).build(),
    )
    .await
    .expect("Should create event");

    let alice_key = alice.id.key().to_string();
    let event_key = event.id.key().to_string();
    let location_key = location.id.key().to_string();

    let scene = create_scene(
        &harness.db,
        SceneBuilder::new("Opening Scene", &event_key, &location_key).build(),
    )
    .await
    .expect("Should create scene");

    let scene_key = scene.id.key().to_string();

    // Create participates_in edge (SCHEMAFULL table requires 'role' field)
    let query = format!(
        "RELATE character:{} -> participates_in -> scene:{} SET role = 'participant'",
        alice_key, scene_key
    );
    harness
        .db
        .query(&query)
        .await
        .expect("Should create participation edge");

    println!("Created character with scene participation");

    // Use graph service to find scenes for the character
    let results = graph_service
        .get_referencing_entities(
            &format!("character:{}", alice_key),
            Some(vec!["scene".to_string()]),
            10,
        )
        .await
        .expect("Reverse query should succeed");

    // Verify the scene is found
    assert!(!results.is_empty(), "Should find at least one scene");

    let scene_result = results
        .iter()
        .find(|r| r.entity_type == "scene" && r.entity_id == format!("scene:{}", scene_key));

    assert!(
        scene_result.is_some(),
        "Should find the scene referencing the character"
    );

    println!("✓ Reverse query found scene for character");
}

/// Test reverse query: find events for a character.
///
/// Verifies: DATA-04 reverse query capability for events
#[tokio::test]
async fn test_reverse_query_finds_events_for_character() {
    let harness = TestHarness::new().await;
    let graph_service = MermaidGraphService::new(harness.db.clone());

    // Create character, event, and involvement edge
    let alice = create_character(&harness.db, CharacterBuilder::new("Alice").build())
        .await
        .expect("Should create Alice");

    let event = create_event(
        &harness.db,
        EventBuilder::new("The Battle").sequence(100).build(),
    )
    .await
    .expect("Should create event");

    let alice_key = alice.id.key().to_string();
    let event_key = event.id.key().to_string();

    // Create involved_in edge
    let query = format!(
        "RELATE character:{} -> involved_in -> event:{}",
        alice_key, event_key
    );
    harness
        .db
        .query(&query)
        .await
        .expect("Should create involvement edge");

    println!("Created character with event involvement");

    // Use graph service to find events for the character
    let results = graph_service
        .get_referencing_entities(
            &format!("character:{}", alice_key),
            Some(vec!["event".to_string()]),
            10,
        )
        .await
        .expect("Reverse query should succeed");

    // Verify the event is found
    assert!(!results.is_empty(), "Should find at least one event");

    let event_result = results
        .iter()
        .find(|r| r.entity_type == "event" && r.entity_id == format!("event:{}", event_key));

    assert!(
        event_result.is_some(),
        "Should find the event referencing the character"
    );

    println!("✓ Reverse query found event for character");
}

/// Test connection path: find path via relationship.
///
/// Verifies: GRAPH-01 BFS-based connection path finding
#[tokio::test]
async fn test_connection_path_via_relationship() {
    let harness = TestHarness::new().await;
    let graph_service = MermaidGraphService::new(harness.db.clone());

    // Create three characters: A -> B -> C
    let alice = create_character(&harness.db, CharacterBuilder::new("Alice").build())
        .await
        .expect("Should create Alice");

    let bob = create_character(&harness.db, CharacterBuilder::new("Bob").build())
        .await
        .expect("Should create Bob");

    let carol = create_character(&harness.db, CharacterBuilder::new("Carol").build())
        .await
        .expect("Should create Carol");

    let alice_key = alice.id.key().to_string();
    let bob_key = bob.id.key().to_string();
    let carol_key = carol.id.key().to_string();

    // Create perceptions: Alice -> Bob
    create_perception(
        &harness.db,
        &alice_key,
        &bob_key,
        PerceptionCreate {
            rel_types: vec!["friendship".to_string()],
            subtype: None,
            feelings: Some("warm".to_string()),
            perception: None,
            tension_level: Some(5),
            history_notes: None,
        },
    )
    .await
    .expect("Should create Alice -> Bob perception");

    // Create perceptions: Bob -> Carol
    create_perception(
        &harness.db,
        &bob_key,
        &carol_key,
        PerceptionCreate {
            rel_types: vec!["friendship".to_string()],
            subtype: None,
            feelings: Some("warm".to_string()),
            perception: None,
            tension_level: Some(5),
            history_notes: None,
        },
    )
    .await
    .expect("Should create Bob -> Carol perception");

    println!("Created character chain: Alice -> Bob -> Carol");

    // Find connection path from Alice to Carol
    let paths = graph_service
        .find_connection_paths(
            &format!("character:{}", alice_key),
            &format!("character:{}", carol_key),
            5,
            false, // Don't include events
        )
        .await
        .expect("Should find connection path");

    // Verify path exists
    assert!(
        !paths.is_empty(),
        "Should find at least one connection path"
    );

    let shortest_path = &paths[0];
    assert_eq!(
        shortest_path.total_hops, 2,
        "Path should be 2 hops: Alice -> Bob -> Carol"
    );

    // Verify path contains Alice, Bob, and Carol
    let path_ids: Vec<String> = shortest_path
        .steps
        .iter()
        .map(|step| step.entity_id.clone())
        .collect();

    assert!(path_ids.contains(&format!("character:{}", alice_key)));
    assert!(path_ids.contains(&format!("character:{}", bob_key)));
    assert!(path_ids.contains(&format!("character:{}", carol_key)));

    println!(
        "✓ Found connection path: A -> B -> C ({} hops)",
        shortest_path.total_hops
    );
}

/// Test connection path: no connection returns empty.
///
/// Verifies: GRAPH-01 handles unconnected characters gracefully
#[tokio::test]
async fn test_connection_path_no_connection() {
    let harness = TestHarness::new().await;
    let graph_service = MermaidGraphService::new(harness.db.clone());

    // Create two unconnected characters
    let alice = create_character(&harness.db, CharacterBuilder::new("Alice").build())
        .await
        .expect("Should create Alice");

    let bob = create_character(&harness.db, CharacterBuilder::new("Bob").build())
        .await
        .expect("Should create Bob");

    let alice_key = alice.id.key().to_string();
    let bob_key = bob.id.key().to_string();

    println!("Created two unconnected characters");

    // Try to find connection path - should return empty
    let paths = graph_service
        .find_connection_paths(
            &format!("character:{}", alice_key),
            &format!("character:{}", bob_key),
            5,
            false,
        )
        .await
        .expect("Should succeed even with no path");

    assert!(
        paths.is_empty(),
        "Should return empty result when no connection exists"
    );

    println!("✓ No connection path found (as expected)");
}
