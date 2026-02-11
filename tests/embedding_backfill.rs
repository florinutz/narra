//! Integration tests for embedding backfill and staleness management.
//!
//! These tests verify Phase 16 requirements for embedding lifecycle:
//! - Backfill generates embeddings for all entity types
//! - Backfill is idempotent (skips already-embedded entities)
//! - Staleness marking triggers re-embedding
//! - Single-type backfill works correctly
//! - Composite text generation produces natural language

mod common;

use common::harness::TestHarness;
use narra::embedding::backfill::BackfillService;
use narra::embedding::composite::character_composite;
use narra::embedding::StalenessManager;
use narra::embedding::{EmbeddingConfig, LocalEmbeddingService};
use narra::models::character::{
    create_character, get_character, update_character, CharacterCreate, CharacterUpdate,
};
use narra::models::event::{create_event, EventCreate};
use narra::models::location::{create_location, LocationCreate};
use narra::models::perception::{create_perception, PerceptionCreate};
use narra::models::scene::{create_scene, SceneCreate};
use std::collections::HashMap;
use std::sync::Arc;
use surrealdb::Datetime;

/// Test backfill generates embeddings for all entity types.
///
/// Verifies: Backfill processes character, location, event, scene with correct dimensions
///
/// Note: Requires real embedding model (~50MB download on first run).
#[tokio::test]
#[ignore = "Requires model download (run manually with --ignored)"]
async fn test_backfill_generates_embeddings_for_all_types() {
    let harness = TestHarness::new().await;

    let embedding_service =
        Arc::new(LocalEmbeddingService::new(EmbeddingConfig::default()).unwrap());
    let _staleness_manager = Arc::new(StalenessManager::new(
        harness.db.clone(),
        embedding_service.clone(),
    ));
    let backfill_service = BackfillService::new(harness.db.clone(), embedding_service.clone());

    // Create one entity of each embeddable type
    let character = create_character(
        &harness.db,
        CharacterCreate {
            name: "Backfill Test Character".into(),
            aliases: vec![],
            roles: vec!["protagonist".into()],
            profile: HashMap::from([(
                "wound".into(),
                vec!["Test wound at age 20. Test belief. Test pattern.".into()],
            )]),
        },
    )
    .await
    .expect("Should create character");

    let location = create_location(
        &harness.db,
        LocationCreate {
            name: "Backfill Test Location".into(),
            description: Some("A test location for backfill".into()),
            loc_type: "building".into(),
            parent: None,
        },
    )
    .await
    .expect("Should create location");

    let event = create_event(
        &harness.db,
        EventCreate {
            title: "Backfill Test Event".into(),
            description: Some("A test event for backfill".into()),
            sequence: 100,
            date: None,
            date_precision: None,
            duration_end: None,
        },
    )
    .await
    .expect("Should create event");

    let scene = create_scene(
        &harness.db,
        SceneCreate {
            title: "Backfill Test Scene".into(),
            summary: Some("A test scene for backfill with meaningful content".into()),
            event: event.id.clone(),
            primary_location: location.id.clone(),
            secondary_locations: vec![],
        },
    )
    .await
    .expect("Should create scene");

    let char_id = character.id.key().to_string();
    let loc_id = location.id.key().to_string();
    let event_id = event.id.key().to_string();
    let scene_id = scene.id.key().to_string();

    println!("Created entities: character, location, event, scene");

    // Verify all start with embedding = NONE and embedding_stale = true
    // (Not strictly necessary for test, just informational)
    println!("Entities created, ready for backfill");

    // Run backfill
    let stats = backfill_service
        .backfill_all()
        .await
        .expect("Backfill should succeed");
    println!("Backfill stats: {:?}", stats);

    assert_eq!(stats.total_entities, 4, "Should process 4 entities");
    assert_eq!(stats.embedded, 4, "Should embed all 4 entities");
    assert_eq!(stats.failed, 0, "Should have no failures");

    // Verify all entities now have embeddings with correct dimension (384 for BGE-small)
    let mut response = harness
        .db
        .query(format!(
            "SELECT embedding, embedding_stale FROM character:{}",
            char_id
        ))
        .await
        .unwrap();
    let char_after: Vec<serde_json::Value> = response.take(0).unwrap();
    let char_data = &char_after[0];
    let char_embedding = char_data.get("embedding").expect("Should have embedding");
    let char_stale = char_data
        .get("embedding_stale")
        .expect("Should have embedding_stale");

    assert!(
        !char_embedding.is_null(),
        "Character should have non-null embedding"
    );
    if let Some(arr) = char_embedding.as_array() {
        assert_eq!(
            arr.len(),
            384,
            "Embedding should be 384-dimensional (BGE-small)"
        );
    } else {
        panic!("Embedding should be an array");
    }
    assert_eq!(
        char_stale,
        &serde_json::json!(false),
        "Character should not be stale"
    );

    // Check location
    let mut response = harness
        .db
        .query(format!(
            "SELECT embedding, embedding_stale FROM location:{}",
            loc_id
        ))
        .await
        .unwrap();
    let loc_after: Vec<serde_json::Value> = response.take(0).unwrap();
    let loc_embedding = loc_after[0].get("embedding").unwrap();
    assert!(!loc_embedding.is_null(), "Location should have embedding");

    // Check event
    let mut response = harness
        .db
        .query(format!(
            "SELECT embedding, embedding_stale FROM event:{}",
            event_id
        ))
        .await
        .unwrap();
    let event_after: Vec<serde_json::Value> = response.take(0).unwrap();
    let event_embedding = event_after[0].get("embedding").unwrap();
    assert!(!event_embedding.is_null(), "Event should have embedding");

    // Check scene
    let mut response = harness
        .db
        .query(format!(
            "SELECT embedding, embedding_stale FROM scene:{}",
            scene_id
        ))
        .await
        .unwrap();
    let scene_after: Vec<serde_json::Value> = response.take(0).unwrap();
    let scene_embedding = scene_after[0].get("embedding").unwrap();
    assert!(!scene_embedding.is_null(), "Scene should have embedding");

    // Verify BackfillStats shows correct counts by type
    assert_eq!(
        stats.entity_type_stats.get("character"),
        Some(&1),
        "Should embed 1 character"
    );
    assert_eq!(
        stats.entity_type_stats.get("location"),
        Some(&1),
        "Should embed 1 location"
    );
    assert_eq!(
        stats.entity_type_stats.get("event"),
        Some(&1),
        "Should embed 1 event"
    );
    assert_eq!(
        stats.entity_type_stats.get("scene"),
        Some(&1),
        "Should embed 1 scene"
    );

    println!("✓ Backfill generated embeddings for all entity types with correct dimensions");
}

/// Test backfill is idempotent (doesn't re-embed already-embedded entities).
///
/// Verifies: Backfill skips entities with valid embeddings
///
/// Note: Requires real embedding model (~50MB download on first run).
#[tokio::test]
#[ignore = "Requires model download (run manually with --ignored)"]
async fn test_backfill_is_idempotent() {
    let harness = TestHarness::new().await;

    let embedding_service =
        Arc::new(LocalEmbeddingService::new(EmbeddingConfig::default()).unwrap());
    let _staleness_manager = Arc::new(StalenessManager::new(
        harness.db.clone(),
        embedding_service.clone(),
    ));
    let backfill_service = BackfillService::new(harness.db.clone(), embedding_service.clone());

    // Create entities
    let character = create_character(
        &harness.db,
        CharacterCreate {
            name: "Idempotent Test".into(),
            aliases: vec![],
            roles: vec!["test".into()],
            ..Default::default()
        },
    )
    .await
    .expect("Should create character");

    let char_id = character.id.key().to_string();

    // Run backfill first time
    let stats1 = backfill_service
        .backfill_all()
        .await
        .expect("First backfill should succeed");
    println!("First backfill: {:?}", stats1);
    assert!(stats1.embedded >= 1, "Should embed at least 1 entity");

    // Record the embedding
    let mut response = harness
        .db
        .query(format!("SELECT embedding FROM character:{}", char_id))
        .await
        .unwrap();
    let result1: Vec<serde_json::Value> = response.take(0).unwrap();
    let embedding1 = result1[0].get("embedding").unwrap().clone();

    // Run backfill again
    let stats2 = backfill_service
        .backfill_all()
        .await
        .expect("Second backfill should succeed");
    println!("Second backfill: {:?}", stats2);

    // Already-embedded entities are filtered out at query level (WHERE embedding IS NONE OR embedding_stale = true)
    // so they don't appear in total_entities at all
    assert_eq!(
        stats2.embedded, 0,
        "Should not re-embed entities with valid embeddings"
    );
    assert_eq!(
        stats2.total_entities, 0,
        "Already-embedded entities should be filtered out by query"
    );

    // Verify embedding hasn't changed
    let mut response = harness
        .db
        .query(format!("SELECT embedding FROM character:{}", char_id))
        .await
        .unwrap();
    let result2: Vec<serde_json::Value> = response.take(0).unwrap();
    let embedding2 = result2[0].get("embedding").unwrap();

    assert_eq!(
        &embedding1, embedding2,
        "Embedding should remain unchanged on second backfill"
    );

    println!("✓ Backfill is idempotent (skips already-embedded entities)");
}

/// Test staleness marking triggers re-embedding on update.
///
/// Verifies: Mutation marks embeddings stale, backfill regenerates them
///
/// Note: Requires real embedding model (~50MB download on first run).
#[tokio::test]
#[ignore = "Requires model download (run manually with --ignored)"]
async fn test_staleness_marking_on_update() {
    let harness = TestHarness::new().await;

    let embedding_service =
        Arc::new(LocalEmbeddingService::new(EmbeddingConfig::default()).unwrap());
    let staleness_manager = Arc::new(StalenessManager::new(
        harness.db.clone(),
        embedding_service.clone(),
    ));
    let backfill_service = BackfillService::new(harness.db.clone(), embedding_service.clone());

    // Create character
    let character = create_character(
        &harness.db,
        CharacterCreate {
            name: "Staleness Test".into(),
            aliases: vec![],
            roles: vec!["protagonist".into()],
            profile: HashMap::from([(
                "wound".into(),
                vec![
                    "Original wound at age 20. Original belief about the world. Original pattern."
                        .into(),
                ],
            )]),
        },
    )
    .await
    .expect("Should create character");

    let char_id = character.id.key().to_string();

    // Run backfill (embedding generated)
    let stats1 = backfill_service
        .backfill_all()
        .await
        .expect("Backfill should succeed");
    println!("Initial backfill: {:?}", stats1);
    assert!(stats1.embedded >= 1);

    // Verify embedding_stale = false
    let mut response = harness
        .db
        .query(format!(
            "SELECT embedding, embedding_stale FROM character:{}",
            char_id
        ))
        .await
        .unwrap();
    let result: Vec<serde_json::Value> = response.take(0).unwrap();
    let stale_before = result[0].get("embedding_stale").unwrap();
    assert_eq!(
        stale_before,
        &serde_json::json!(false),
        "Should not be stale after backfill"
    );

    // Record old embedding
    let embedding_before = result[0].get("embedding").unwrap().clone();

    // Update character name via DB and mark stale
    let updated = update_character(
        &harness.db,
        &char_id,
        CharacterUpdate {
            name: Some("Staleness Test UPDATED".into()),
            profile: Some(HashMap::from([
                ("wound".into(), vec!["Completely different traumatic event at age 25. Completely different worldview now. Different pattern.".into()]),
            ])),
            updated_at: Datetime::default(),
            ..Default::default()
        },
    )
    .await
    .expect("Update should succeed");

    assert!(updated.is_some(), "Character should still exist");

    // Manually mark stale (since mutation handler isn't in this test path)
    staleness_manager
        .mark_stale(&format!("character:{}", char_id))
        .await
        .expect("Should mark stale");

    // Verify embedding_stale = true
    let mut response = harness
        .db
        .query(format!("SELECT embedding_stale FROM character:{}", char_id))
        .await
        .unwrap();
    let result: Vec<serde_json::Value> = response.take(0).unwrap();
    let stale_after_update = result[0].get("embedding_stale").unwrap();
    assert_eq!(
        stale_after_update,
        &serde_json::json!(true),
        "Should be stale after update"
    );

    // Run backfill again
    let stats2 = backfill_service
        .backfill_all()
        .await
        .expect("Second backfill should succeed");
    println!("Second backfill: {:?}", stats2);
    assert!(stats2.embedded >= 1, "Should re-embed stale entity");

    // Verify new embedding generated
    let mut response = harness
        .db
        .query(format!(
            "SELECT embedding, embedding_stale FROM character:{}",
            char_id
        ))
        .await
        .unwrap();
    let result: Vec<serde_json::Value> = response.take(0).unwrap();
    let embedding_after = result[0].get("embedding").unwrap();
    let stale_after_backfill = result[0].get("embedding_stale").unwrap();

    assert_eq!(
        stale_after_backfill,
        &serde_json::json!(false),
        "Should not be stale after re-embedding"
    );

    // Verify embedding differs (text changed, so embedding should change)
    // Note: We can't compare exact vectors since the embedding model is deterministic but the text is different
    assert_ne!(
        &embedding_before, embedding_after,
        "Embedding should differ after text change and re-embedding"
    );

    println!("✓ Staleness marking triggers re-embedding on update");
}

/// Test backfill single entity type.
///
/// Verifies: backfill_type processes only the specified type
///
/// Note: Requires real embedding model (~50MB download on first run).
#[tokio::test]
#[ignore = "Requires model download (run manually with --ignored)"]
async fn test_backfill_single_type() {
    let harness = TestHarness::new().await;

    let embedding_service =
        Arc::new(LocalEmbeddingService::new(EmbeddingConfig::default()).unwrap());
    let _staleness_manager = Arc::new(StalenessManager::new(
        harness.db.clone(),
        embedding_service.clone(),
    ));
    let backfill_service = BackfillService::new(harness.db.clone(), embedding_service.clone());

    // Create multiple entity types
    let character = create_character(
        &harness.db,
        CharacterCreate {
            name: "Single Type Test Char".into(),
            aliases: vec![],
            roles: vec![],
            ..Default::default()
        },
    )
    .await
    .expect("Should create character");

    let location = create_location(
        &harness.db,
        LocationCreate {
            name: "Single Type Test Loc".into(),
            description: Some("Test location".into()),
            loc_type: "building".into(),
            parent: None,
        },
    )
    .await
    .expect("Should create location");

    let event = create_event(
        &harness.db,
        EventCreate {
            title: "Single Type Test Event".into(),
            description: Some("Test event".into()),
            sequence: 100,
            date: None,
            date_precision: None,
            duration_end: None,
        },
    )
    .await
    .expect("Should create event");

    let char_id = character.id.key().to_string();
    let loc_id = location.id.key().to_string();
    let event_id = event.id.key().to_string();

    // Run backfill_type("character") only
    let stats = backfill_service
        .backfill_type("character")
        .await
        .expect("Character backfill should succeed");
    println!("Character-only backfill: {:?}", stats);

    assert!(stats.embedded >= 1, "Should embed at least 1 character");

    // Verify only characters have embeddings
    let mut response = harness
        .db
        .query(format!("SELECT embedding FROM character:{}", char_id))
        .await
        .unwrap();
    let char_result: Vec<serde_json::Value> = response.take(0).unwrap();
    let char_embedding = char_result[0].get("embedding").unwrap();
    assert!(!char_embedding.is_null(), "Character should have embedding");

    // Other types should still have NONE
    let mut response = harness
        .db
        .query(format!("SELECT embedding FROM location:{}", loc_id))
        .await
        .unwrap();
    let loc_result: Vec<serde_json::Value> = response.take(0).unwrap();
    let loc_embedding = loc_result[0].get("embedding").unwrap();
    assert!(
        loc_embedding.is_null(),
        "Location should NOT have embedding yet"
    );

    let mut response = harness
        .db
        .query(format!("SELECT embedding FROM event:{}", event_id))
        .await
        .unwrap();
    let event_result: Vec<serde_json::Value> = response.take(0).unwrap();
    let event_embedding = event_result[0].get("embedding").unwrap();
    assert!(
        event_embedding.is_null(),
        "Event should NOT have embedding yet"
    );

    println!("✓ Single-type backfill works correctly");
}

/// Test composite text generation produces natural language.
///
/// Verifies: Composite text is readable and includes key character attributes
#[tokio::test]
async fn test_composite_text_generation() {
    let harness = TestHarness::new().await;

    // Create a character with rich attributes
    let character = create_character(
        &harness.db,
        CharacterCreate {
            name: "Elena Vasquez".into(),
            aliases: vec!["The Shadow".into()],
            roles: vec!["protagonist".into(), "detective".into()],
            profile: HashMap::from([
                ("wound".into(), vec![
                    "Father abandoned the family when she was eight. People you love will always leave you. Keeps everyone at emotional distance.".into(),
                    "Betrayed by her partner Marcus at 28. Trust is weakness. Never reveals vulnerabilities.".into(),
                ]),
                ("desire_conscious".into(), vec!["Bring Marcus to justice".into()]),
                ("desire_unconscious".into(), vec!["Prove she is worthy of love despite her wounds".into()]),
                ("contradiction".into(), vec!["Claims not to care about Marcus but obsessively tracks him".into()]),
            ]),
        },
    )
    .await
    .expect("Should create character");

    let char_id = character.id.key().to_string();

    // Create a relationship for relationship info
    let marcus = create_character(
        &harness.db,
        CharacterCreate {
            name: "Marcus Chen".into(),
            aliases: vec![],
            roles: vec!["antagonist".into()],
            ..Default::default()
        },
    )
    .await
    .expect("Should create Marcus");

    let marcus_id = marcus.id.key().to_string();

    // Create perception (Elena sees Marcus)
    create_perception(
        &harness.db,
        &char_id,
        &marcus_id,
        PerceptionCreate {
            rel_types: vec!["professional".into(), "rivalry".into()],
            subtype: None,
            feelings: Some("Bitter hatred".into()),
            perception: Some("The man who destroyed everything".into()),
            tension_level: Some(10),
            history_notes: Some("Former partners, he betrayed her".into()),
        },
    )
    .await
    .expect("Should create perception");

    // Fetch the full character with relationships
    let full_character = get_character(&harness.db, &char_id)
        .await
        .expect("Should fetch character")
        .expect("Character should exist");

    // Generate composite text
    // For this test, we don't have relationships loaded, so pass empty slice
    let composite = character_composite(&full_character, &[], &[]);
    println!("Generated composite text:\n{}", composite);

    // Verify text contains key elements
    assert!(
        composite.contains("Elena Vasquez"),
        "Should contain character name"
    );
    assert!(
        composite.contains("protagonist") || composite.contains("detective"),
        "Should contain role"
    );
    // Composite focuses on beliefs and patterns, not event details
    assert!(
        composite.contains("People you love")
            || composite.contains("leave")
            || composite.contains("emotional distance"),
        "Should contain wound belief or pattern"
    );
    assert!(
        composite.contains("Marcus")
            || composite.contains("justice")
            || composite.contains("contradiction"),
        "Should contain desire or contradiction info"
    );

    // Verify text reads naturally (not key-value dump)
    assert!(
        !composite.contains("wounds:"),
        "Should not be a key-value dump"
    );
    assert!(
        !composite.contains("desires:"),
        "Should not be a key-value dump"
    );
    assert!(
        !composite.contains("{"),
        "Should not contain JSON-like syntax"
    );

    // Verify reasonable length (not too short, not absurdly long)
    assert!(
        composite.len() > 100,
        "Composite text should be substantial"
    );
    assert!(
        composite.len() < 5000,
        "Composite text should be reasonably concise"
    );

    println!("✓ Composite text generation produces natural language");
}
