//! Integration tests for entity CRUD operations via the repository layer.
//!
//! These tests verify that Character, Location, Event, and Scene CRUD
//! operations work correctly through the SurrealEntityRepository.

mod common;

use narra::models::{CharacterUpdate, EventUpdate, LocationUpdate, SceneUpdate};
use narra::repository::{EntityRepository, SurrealEntityRepository};
use pretty_assertions::assert_eq;
use surrealdb::Datetime;

use common::builders::{CharacterBuilder, EventBuilder, LocationBuilder, SceneBuilder};
use common::harness::TestHarness;

/// Test complete character CRUD workflow: create, read, update, delete, list.
///
/// Verifies:
/// - Characters can be created with all fields
/// - Characters can be retrieved by ID
/// - Characters can be updated (partial updates work)
/// - Characters can be deleted
/// - Deleted characters are excluded from queries
#[tokio::test]
async fn test_character_crud_workflow() {
    let harness = TestHarness::new().await;
    let repo = SurrealEntityRepository::new(harness.db.clone());

    // CREATE
    let create_data = CharacterBuilder::new("Alice")
        .alias("The Shadow")
        .role("protagonist")
        .build();

    let character = repo
        .create_character(create_data)
        .await
        .expect("Failed to create character");

    assert_eq!(character.name, "Alice");
    assert!(character.aliases.contains(&"The Shadow".to_string()));
    assert!(character.roles.contains(&"protagonist".to_string()));

    // READ
    let char_id = character.id.key().to_string();
    let fetched = repo
        .get_character(&char_id)
        .await
        .expect("Failed to get character")
        .expect("Character should exist");

    assert_eq!(fetched.name, "Alice");
    assert_eq!(fetched.id, character.id);

    // LIST
    let all_characters = repo
        .list_characters()
        .await
        .expect("Failed to list characters");
    assert_eq!(all_characters.len(), 1);
    assert_eq!(all_characters[0].name, "Alice");

    // UPDATE
    let update_data = CharacterUpdate {
        name: Some("Alice Wonderland".to_string()),
        roles: Some(vec!["protagonist".to_string(), "adventurer".to_string()]),
        updated_at: Datetime::default(),
        ..Default::default()
    };

    let updated = repo
        .update_character(&char_id, update_data)
        .await
        .expect("Failed to update character")
        .expect("Character should exist for update");

    assert_eq!(updated.name, "Alice Wonderland");
    assert!(updated.roles.contains(&"adventurer".to_string()));
    // Aliases should be unchanged (partial update)
    assert!(updated.aliases.contains(&"The Shadow".to_string()));

    // Verify update persists
    let refetched = repo
        .get_character(&char_id)
        .await
        .expect("Failed to refetch character")
        .expect("Character should still exist");
    assert_eq!(refetched.name, "Alice Wonderland");

    // DELETE
    let deleted = repo
        .delete_character(&char_id)
        .await
        .expect("Failed to delete character")
        .expect("Character should exist for deletion");
    assert_eq!(deleted.name, "Alice Wonderland");

    // Verify deletion - character should not be retrievable
    let after_delete = repo
        .get_character(&char_id)
        .await
        .expect("Get should not error");
    assert!(
        after_delete.is_none(),
        "Deleted character should not be retrievable"
    );

    // Verify deletion - character should not appear in list
    let final_list = repo
        .list_characters()
        .await
        .expect("Failed to list characters");
    assert!(
        final_list.is_empty(),
        "Deleted character should not appear in list"
    );
}

/// Test that multiple characters can be created and listed.
#[tokio::test]
async fn test_character_list_multiple() {
    let harness = TestHarness::new().await;
    let repo = SurrealEntityRepository::new(harness.db.clone());

    // Create multiple characters
    let alice = repo
        .create_character(CharacterBuilder::new("Alice").build())
        .await
        .expect("Failed to create Alice");

    let bob = repo
        .create_character(CharacterBuilder::new("Bob").role("antagonist").build())
        .await
        .expect("Failed to create Bob");

    let charlie = repo
        .create_character(CharacterBuilder::new("Charlie").alias("Chuck").build())
        .await
        .expect("Failed to create Charlie");

    // List should return all three
    let all = repo
        .list_characters()
        .await
        .expect("Failed to list characters");
    assert_eq!(all.len(), 3);

    // Verify all names are present
    let names: Vec<&str> = all.iter().map(|c| c.name.as_str()).collect();
    assert!(names.contains(&"Alice"));
    assert!(names.contains(&"Bob"));
    assert!(names.contains(&"Charlie"));

    // Delete one, verify list updates
    repo.delete_character(&bob.id.key().to_string())
        .await
        .expect("Failed to delete Bob");

    let after_delete = repo
        .list_characters()
        .await
        .expect("Failed to list characters");
    assert_eq!(after_delete.len(), 2);

    let names_after: Vec<&str> = after_delete.iter().map(|c| c.name.as_str()).collect();
    assert!(names_after.contains(&"Alice"));
    assert!(!names_after.contains(&"Bob"));
    assert!(names_after.contains(&"Charlie"));

    // Clean up
    repo.delete_character(&alice.id.key().to_string())
        .await
        .ok();
    repo.delete_character(&charlie.id.key().to_string())
        .await
        .ok();
}

// =============================================================================
// LOCATION CRUD TESTS
// =============================================================================

/// Test complete location CRUD workflow: create, read, update, delete, list.
///
/// Verifies:
/// - Locations can be created with all fields
/// - Locations can be retrieved by ID
/// - Locations can be updated (partial updates work)
/// - Locations can be deleted
/// - Deleted locations are excluded from queries
#[tokio::test]
async fn test_location_crud_workflow() {
    let harness = TestHarness::new().await;
    let repo = SurrealEntityRepository::new(harness.db.clone());

    // CREATE
    let create_data = LocationBuilder::new("The Tower")
        .description("A tall stone tower on the hill")
        .loc_type("building")
        .build();

    let location = repo
        .create_location(create_data)
        .await
        .expect("Failed to create location");

    assert_eq!(location.name, "The Tower");
    assert_eq!(
        location.description,
        Some("A tall stone tower on the hill".to_string())
    );
    assert_eq!(location.loc_type, "building");

    // READ
    let loc_id = location.id.key().to_string();
    let fetched = repo
        .get_location(&loc_id)
        .await
        .expect("Failed to get location")
        .expect("Location should exist");

    assert_eq!(fetched.name, "The Tower");
    assert_eq!(fetched.id, location.id);

    // LIST
    let all_locations = repo
        .list_locations()
        .await
        .expect("Failed to list locations");
    assert_eq!(all_locations.len(), 1);
    assert_eq!(all_locations[0].name, "The Tower");

    // UPDATE
    let update_data = LocationUpdate {
        name: Some("The Dark Tower".to_string()),
        loc_type: Some("fortress".to_string()),
        description: None, // Keep existing
        parent: None,
        updated_at: Datetime::default(),
    };

    let updated = repo
        .update_location(&loc_id, update_data)
        .await
        .expect("Failed to update location")
        .expect("Location should exist for update");

    assert_eq!(updated.name, "The Dark Tower");
    assert_eq!(updated.loc_type, "fortress");
    // Description should be unchanged (partial update)
    assert_eq!(
        updated.description,
        Some("A tall stone tower on the hill".to_string())
    );

    // Verify update persists
    let refetched = repo
        .get_location(&loc_id)
        .await
        .expect("Failed to refetch location")
        .expect("Location should still exist");
    assert_eq!(refetched.name, "The Dark Tower");

    // DELETE
    let deleted = repo
        .delete_location(&loc_id)
        .await
        .expect("Failed to delete location")
        .expect("Location should exist for deletion");
    assert_eq!(deleted.name, "The Dark Tower");

    // Verify deletion - location should not be retrievable
    let after_delete = repo
        .get_location(&loc_id)
        .await
        .expect("Get should not error");
    assert!(
        after_delete.is_none(),
        "Deleted location should not be retrievable"
    );

    // Verify deletion - location should not appear in list
    let final_list = repo
        .list_locations()
        .await
        .expect("Failed to list locations");
    assert!(
        final_list.is_empty(),
        "Deleted location should not appear in list"
    );
}

// =============================================================================
// EVENT CRUD TESTS
// =============================================================================

/// Test complete event CRUD workflow: create, read, update, delete, list.
///
/// Verifies:
/// - Events can be created with all fields
/// - Events can be retrieved by ID
/// - Events can be updated (partial updates work)
/// - Events can be deleted
/// - Deleted events are excluded from queries
/// - Events are ordered by sequence in list
#[tokio::test]
async fn test_event_crud_workflow() {
    let harness = TestHarness::new().await;
    let repo = SurrealEntityRepository::new(harness.db.clone());

    // CREATE
    let create_data = EventBuilder::new("The Betrayal")
        .description("Marcus reveals his true allegiance")
        .sequence(100)
        .build();

    let event = repo
        .create_event(create_data)
        .await
        .expect("Failed to create event");

    assert_eq!(event.title, "The Betrayal");
    assert_eq!(
        event.description,
        Some("Marcus reveals his true allegiance".to_string())
    );
    assert_eq!(event.sequence, 100);

    // READ
    let event_id = event.id.key().to_string();
    let fetched = repo
        .get_event(&event_id)
        .await
        .expect("Failed to get event")
        .expect("Event should exist");

    assert_eq!(fetched.title, "The Betrayal");
    assert_eq!(fetched.id, event.id);

    // LIST
    let all_events = repo.list_events().await.expect("Failed to list events");
    assert_eq!(all_events.len(), 1);
    assert_eq!(all_events[0].title, "The Betrayal");

    // UPDATE
    let update_data = EventUpdate {
        title: Some("The Great Betrayal".to_string()),
        sequence: Some(150),
        description: None,
        date: None,
        date_precision: None,
        duration_end: None,
        updated_at: Datetime::default(),
    };

    let updated = repo
        .update_event(&event_id, update_data)
        .await
        .expect("Failed to update event")
        .expect("Event should exist for update");

    assert_eq!(updated.title, "The Great Betrayal");
    assert_eq!(updated.sequence, 150);
    // Description should be unchanged (partial update)
    assert_eq!(
        updated.description,
        Some("Marcus reveals his true allegiance".to_string())
    );

    // Verify update persists
    let refetched = repo
        .get_event(&event_id)
        .await
        .expect("Failed to refetch event")
        .expect("Event should still exist");
    assert_eq!(refetched.title, "The Great Betrayal");

    // DELETE
    let deleted = repo
        .delete_event(&event_id)
        .await
        .expect("Failed to delete event")
        .expect("Event should exist for deletion");
    assert_eq!(deleted.title, "The Great Betrayal");

    // Verify deletion - event should not be retrievable
    let after_delete = repo
        .get_event(&event_id)
        .await
        .expect("Get should not error");
    assert!(
        after_delete.is_none(),
        "Deleted event should not be retrievable"
    );

    // Verify deletion - event should not appear in list
    let final_list = repo.list_events().await.expect("Failed to list events");
    assert!(
        final_list.is_empty(),
        "Deleted event should not appear in list"
    );
}

/// Test that events are listed in sequence order.
#[tokio::test]
async fn test_event_list_ordered_by_sequence() {
    let harness = TestHarness::new().await;
    let repo = SurrealEntityRepository::new(harness.db.clone());

    // Create events out of order
    let event_c = repo
        .create_event(EventBuilder::new("Event C").sequence(300).build())
        .await
        .expect("Failed to create event C");

    let event_a = repo
        .create_event(EventBuilder::new("Event A").sequence(100).build())
        .await
        .expect("Failed to create event A");

    let event_b = repo
        .create_event(EventBuilder::new("Event B").sequence(200).build())
        .await
        .expect("Failed to create event B");

    // List should return in sequence order (A, B, C)
    let events = repo.list_events().await.expect("Failed to list events");
    assert_eq!(events.len(), 3);
    assert_eq!(events[0].title, "Event A");
    assert_eq!(events[1].title, "Event B");
    assert_eq!(events[2].title, "Event C");

    // Clean up
    repo.delete_event(&event_a.id.key().to_string()).await.ok();
    repo.delete_event(&event_b.id.key().to_string()).await.ok();
    repo.delete_event(&event_c.id.key().to_string()).await.ok();
}

// =============================================================================
// SCENE CRUD TESTS
// =============================================================================

/// Test complete scene CRUD workflow: create, read, update, delete, list.
///
/// Verifies:
/// - Scenes can be created with required dependencies (event, location)
/// - Scenes can be retrieved by ID
/// - Scenes can be updated (partial updates work)
/// - Scenes can be deleted
/// - Deleted scenes are excluded from queries
#[tokio::test]
async fn test_scene_crud_workflow() {
    let harness = TestHarness::new().await;
    let repo = SurrealEntityRepository::new(harness.db.clone());

    // First create required dependencies: event and location
    let event = repo
        .create_event(EventBuilder::new("Chapter One").sequence(10).build())
        .await
        .expect("Failed to create event");
    let event_id = event.id.key().to_string();

    let location = repo
        .create_location(
            LocationBuilder::new("The Tavern")
                .loc_type("building")
                .build(),
        )
        .await
        .expect("Failed to create location");
    let location_id = location.id.key().to_string();

    // CREATE
    let create_data = SceneBuilder::new("Opening Scene", &event_id, &location_id)
        .summary("Our heroes meet for the first time")
        .build();

    let scene = repo
        .create_scene(create_data)
        .await
        .expect("Failed to create scene");

    assert_eq!(scene.title, "Opening Scene");
    assert_eq!(
        scene.summary,
        Some("Our heroes meet for the first time".to_string())
    );
    // Verify references are stored
    assert!(scene.event.to_string().contains(&event_id));
    assert!(scene.primary_location.to_string().contains(&location_id));

    // READ
    let scene_id = scene.id.key().to_string();
    let fetched = repo
        .get_scene(&scene_id)
        .await
        .expect("Failed to get scene")
        .expect("Scene should exist");

    assert_eq!(fetched.title, "Opening Scene");
    assert_eq!(fetched.id, scene.id);

    // LIST
    let all_scenes = repo.list_scenes().await.expect("Failed to list scenes");
    assert_eq!(all_scenes.len(), 1);
    assert_eq!(all_scenes[0].title, "Opening Scene");

    // UPDATE
    let update_data = SceneUpdate {
        title: Some("The Grand Opening".to_string()),
        summary: Some(Some("A fateful meeting at the tavern".to_string())),
        event: None,
        primary_location: None,
        secondary_locations: None,
        updated_at: Datetime::default(),
    };

    let updated = repo
        .update_scene(&scene_id, update_data)
        .await
        .expect("Failed to update scene")
        .expect("Scene should exist for update");

    assert_eq!(updated.title, "The Grand Opening");
    assert_eq!(
        updated.summary,
        Some("A fateful meeting at the tavern".to_string())
    );
    // Event and location should be unchanged (partial update)
    assert!(updated.event.to_string().contains(&event_id));

    // Verify update persists
    let refetched = repo
        .get_scene(&scene_id)
        .await
        .expect("Failed to refetch scene")
        .expect("Scene should still exist");
    assert_eq!(refetched.title, "The Grand Opening");

    // DELETE
    let deleted = repo
        .delete_scene(&scene_id)
        .await
        .expect("Failed to delete scene")
        .expect("Scene should exist for deletion");
    assert_eq!(deleted.title, "The Grand Opening");

    // Verify deletion - scene should not be retrievable
    let after_delete = repo
        .get_scene(&scene_id)
        .await
        .expect("Get should not error");
    assert!(
        after_delete.is_none(),
        "Deleted scene should not be retrievable"
    );

    // Verify deletion - scene should not appear in list
    let final_list = repo.list_scenes().await.expect("Failed to list scenes");
    assert!(
        final_list.is_empty(),
        "Deleted scene should not appear in list"
    );

    // Clean up dependencies
    repo.delete_event(&event_id).await.ok();
    repo.delete_location(&location_id).await.ok();
}

/// Test that multiple scenes can be listed for a given event.
#[tokio::test]
async fn test_scene_list_multiple_at_event() {
    let harness = TestHarness::new().await;
    let repo = SurrealEntityRepository::new(harness.db.clone());

    // Create one event and location
    let event = repo
        .create_event(EventBuilder::new("The Festival").sequence(50).build())
        .await
        .expect("Failed to create event");
    let event_id = event.id.key().to_string();

    let location1 = repo
        .create_location(LocationBuilder::new("Main Stage").loc_type("venue").build())
        .await
        .expect("Failed to create location 1");
    let location1_id = location1.id.key().to_string();

    let location2 = repo
        .create_location(
            LocationBuilder::new("Back Alley")
                .loc_type("outdoor")
                .build(),
        )
        .await
        .expect("Failed to create location 2");
    let location2_id = location2.id.key().to_string();

    // Create multiple scenes at the same event
    let scene1 = repo
        .create_scene(
            SceneBuilder::new("The Performance", &event_id, &location1_id)
                .summary("A dazzling show")
                .build(),
        )
        .await
        .expect("Failed to create scene 1");

    let scene2 = repo
        .create_scene(
            SceneBuilder::new("The Secret Meeting", &event_id, &location2_id)
                .summary("Conspirators gather")
                .build(),
        )
        .await
        .expect("Failed to create scene 2");

    // List should return both scenes
    let scenes = repo.list_scenes().await.expect("Failed to list scenes");
    assert_eq!(scenes.len(), 2);

    let titles: Vec<&str> = scenes.iter().map(|s| s.title.as_str()).collect();
    assert!(titles.contains(&"The Performance"));
    assert!(titles.contains(&"The Secret Meeting"));

    // Delete one scene, verify list updates
    repo.delete_scene(&scene1.id.key().to_string())
        .await
        .expect("Failed to delete scene 1");

    let after_delete = repo.list_scenes().await.expect("Failed to list scenes");
    assert_eq!(after_delete.len(), 1);
    assert_eq!(after_delete[0].title, "The Secret Meeting");

    // Clean up
    repo.delete_scene(&scene2.id.key().to_string()).await.ok();
    repo.delete_event(&event_id).await.ok();
    repo.delete_location(&location1_id).await.ok();
    repo.delete_location(&location2_id).await.ok();
}
