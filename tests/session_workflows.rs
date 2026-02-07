//! Integration tests for session continuity workflows (SERV-04).
//!
//! Tests hot entity tracking, context restoration, and startup summaries.

mod common;

use narra::repository::{EntityRepository, SurrealEntityRepository};
use narra::services::{CachedContextService, ContextConfig, ContextService};
use narra::session::{PendingDecision, SessionStateManager};
use pretty_assertions::assert_eq;
use std::sync::Arc;
use tempfile::TempDir;

use common::builders::CharacterBuilder;
use common::harness::TestHarness;

// ============================================================================
// SESSION STATE PERSISTENCE TESTS
// ============================================================================

/// Test session state persists across process boundaries (save/load).
#[tokio::test]
async fn test_session_persistence() {
    let temp_dir = TempDir::new().expect("Temp dir");
    let session_path = temp_dir.path().join("session.json");

    // First "process": create and save state
    {
        let manager = SessionStateManager::load_or_create(&session_path)
            .expect("Should create session manager");

        manager.record_access("character:alice").await;
        manager.record_access("character:bob").await;
        manager.pin_entity("character:alice").await;
        manager.mark_session_end().await;

        manager.save().await.expect("Should save session");
    }

    // Second "process": load and verify state
    {
        let manager = SessionStateManager::load_or_create(&session_path)
            .expect("Should load session manager");

        let pinned = manager.get_pinned().await;
        assert!(
            pinned.contains(&"character:alice".to_string()),
            "Pinned entity should persist"
        );

        let recent = manager.get_recent(10).await;
        assert!(
            recent.contains(&"character:alice".to_string()),
            "Recent accesses should persist"
        );
        assert!(
            recent.contains(&"character:bob".to_string()),
            "Recent accesses should persist"
        );

        assert!(
            manager.get_last_session().await.is_some(),
            "Last session timestamp should persist"
        );
    }
}

// ============================================================================
// HOT ENTITY TRACKING TESTS
// ============================================================================

/// Test that recent entity accesses are tracked correctly.
#[tokio::test]
async fn test_hot_entity_tracking() {
    let temp_dir = TempDir::new().expect("Temp dir");
    let session_path = temp_dir.path().join("session.json");

    let manager =
        SessionStateManager::load_or_create(&session_path).expect("Should create session manager");

    // Access entities in order
    manager.record_access("character:first").await;
    manager.record_access("character:second").await;
    manager.record_access("character:third").await;

    // Most recent should be first in the list
    let recent = manager.get_recent(10).await;
    assert_eq!(
        recent[0], "character:third",
        "Most recent access should be first"
    );
    assert_eq!(recent[1], "character:second");
    assert_eq!(recent[2], "character:first", "Oldest access should be last");

    // Re-accessing an entity should move it to the front
    manager.record_access("character:first").await;
    let recent_after = manager.get_recent(10).await;
    assert_eq!(
        recent_after[0], "character:first",
        "Re-accessed entity should move to front"
    );
}

/// Test that hot entity list is limited to 100 entries.
#[tokio::test]
async fn test_hot_entity_limit() {
    let temp_dir = TempDir::new().expect("Temp dir");
    let session_path = temp_dir.path().join("session.json");

    let manager =
        SessionStateManager::load_or_create(&session_path).expect("Should create session manager");

    // Access more than 100 entities
    for i in 0..150 {
        manager.record_access(&format!("character:{}", i)).await;
    }

    let all_recent = manager.get_recent(200).await;
    assert_eq!(all_recent.len(), 100, "Should be limited to 100 entries");

    // Most recent should be character:149
    assert_eq!(all_recent[0], "character:149");
}

// ============================================================================
// PIN/UNPIN TESTS
// ============================================================================

/// Test pinning and unpinning entities.
#[tokio::test]
async fn test_pin_unpin_entity() {
    let temp_dir = TempDir::new().expect("Temp dir");
    let session_path = temp_dir.path().join("session.json");

    let manager =
        SessionStateManager::load_or_create(&session_path).expect("Should create session manager");

    // Pin an entity
    manager.pin_entity("character:important").await;

    let pinned = manager.get_pinned().await;
    assert!(pinned.contains(&"character:important".to_string()));

    // Pinning again should not duplicate
    manager.pin_entity("character:important").await;
    let pinned_after = manager.get_pinned().await;
    assert_eq!(
        pinned_after
            .iter()
            .filter(|e| *e == "character:important")
            .count(),
        1,
        "Should not have duplicates"
    );

    // Unpin
    manager.unpin_entity("character:important").await;
    let pinned_after_unpin = manager.get_pinned().await;
    assert!(
        !pinned_after_unpin.contains(&"character:important".to_string()),
        "Entity should be unpinned"
    );
}

// ============================================================================
// CONTEXT RESTORATION TESTS
// ============================================================================

/// Test full session lifecycle: create entities, track access, restore context.
///
/// This test demonstrates the full lifecycle within a single database connection,
/// simulating session save/load with the SessionStateManager. RocksDB embedded
/// mode doesn't support multiple connections to the same path from one process.
#[tokio::test]
async fn test_session_full_lifecycle() {
    let harness = TestHarness::new().await;
    let temp_dir = TempDir::new().expect("Temp dir");
    let session_path = temp_dir.path().join("session.json");

    let entity_repo = SurrealEntityRepository::new(harness.db.clone());

    // Phase 1: Create entities and track access, then save session
    let character_id;
    {
        let session_manager =
            Arc::new(SessionStateManager::load_or_create(&session_path).expect("Session manager"));

        // Create a character
        let character = entity_repo
            .create_character(
                CharacterBuilder::new("Alice Blackwood")
                    .role("protagonist")
                    .build(),
            )
            .await
            .expect("Character");
        character_id = format!("character:{}", character.id.key());

        // Track access and pin
        session_manager.record_access(&character_id).await;
        session_manager.pin_entity(&character_id).await;

        // End session and save to disk
        session_manager.mark_session_end().await;
        session_manager.save().await.expect("Save");
    }

    // Phase 2: Load session from disk and verify state restored
    {
        let session_manager = Arc::new(
            SessionStateManager::load_or_create(&session_path).expect("Session manager 2"),
        );

        // Create context service with restored session
        let context_service =
            CachedContextService::new(harness.db.clone(), session_manager.clone());

        // Verify pinned entity is restored from saved session
        let pinned = session_manager.get_pinned().await;
        assert!(
            pinned.contains(&character_id),
            "Pinned entity should be restored after session load"
        );

        // Verify hot entities are tracked (restored from saved session)
        let hot = context_service.get_hot_entities(10).await;
        assert!(
            hot.contains(&character_id),
            "Hot entity should be in recent list after session load"
        );

        // Get context - should include pinned entity
        let context = context_service
            .get_context(
                &[], // No explicit mentions
                ContextConfig {
                    max_entities: 10,
                    token_budget: 2000,
                    graph_depth: 1,
                    pinned_entities: vec![character_id.clone()],
                    ..Default::default()
                },
            )
            .await
            .expect("Should get context");

        assert!(
            !context.entities.is_empty(),
            "Context should include entities"
        );
        assert!(
            context.entities.iter().any(|e| e.id == character_id),
            "Context should include pinned character"
        );
    }
}

// ============================================================================
// PENDING DECISIONS TESTS
// ============================================================================

/// Test pending decision tracking.
#[tokio::test]
async fn test_pending_decisions() {
    let temp_dir = TempDir::new().expect("Temp dir");
    let session_path = temp_dir.path().join("session.json");

    let manager =
        SessionStateManager::load_or_create(&session_path).expect("Should create session manager");

    // Add a pending decision
    let decision = PendingDecision {
        id: "decision-1".to_string(),
        description: "Should Alice confront Bob now or wait?".to_string(),
        created_at: chrono::Utc::now(),
        entity_ids: vec!["character:alice".to_string(), "character:bob".to_string()],
    };

    manager.add_pending_decision(decision).await;

    // Verify it's tracked
    let pending = manager.get_pending_decisions().await;
    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0].id, "decision-1");

    // Resolve the decision
    manager.resolve_pending_decision("decision-1").await;

    let pending_after = manager.get_pending_decisions().await;
    assert_eq!(pending_after.len(), 0, "Decision should be resolved");
}

/// Test pending decisions persist across sessions.
#[tokio::test]
async fn test_pending_decisions_persistence() {
    let temp_dir = TempDir::new().expect("Temp dir");
    let session_path = temp_dir.path().join("session.json");

    // Create decision in first session
    {
        let manager = SessionStateManager::load_or_create(&session_path).expect("Manager 1");

        manager
            .add_pending_decision(PendingDecision {
                id: "persistent-decision".to_string(),
                description: "This should persist".to_string(),
                created_at: chrono::Utc::now(),
                entity_ids: vec![],
            })
            .await;

        manager.save().await.expect("Save");
    }

    // Verify in second session
    {
        let manager = SessionStateManager::load_or_create(&session_path).expect("Manager 2");

        let pending = manager.get_pending_decisions().await;
        assert_eq!(pending.len(), 1, "Decision should persist");
        assert_eq!(pending[0].id, "persistent-decision");
    }
}

// ============================================================================
// CONTEXT SERVICE INTEGRATION TESTS
// ============================================================================

/// Test context scoring considers pins, recency, and mentions.
#[tokio::test]
async fn test_context_scoring() {
    let harness = TestHarness::new().await;
    let temp_dir = TempDir::new().expect("Temp dir");
    let session_path = temp_dir.path().join("session.json");

    let entity_repo = SurrealEntityRepository::new(harness.db.clone());
    let session_manager =
        Arc::new(SessionStateManager::load_or_create(&session_path).expect("Session"));

    // Create three characters
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

    let alice_id = format!("character:{}", alice.id.key());
    let bob_id = format!("character:{}", bob.id.key());
    let charlie_id = format!("character:{}", charlie.id.key());

    // Pin Alice
    session_manager.pin_entity(&alice_id).await;

    // Access Bob recently
    session_manager.record_access(&bob_id).await;

    // Create context service
    let context_service = CachedContextService::new(harness.db.clone(), session_manager);

    // Get context mentioning Charlie
    let context = context_service
        .get_context(
            &[charlie_id.clone()], // Mention Charlie
            ContextConfig::default(),
        )
        .await
        .expect("Context");

    // All three should be in context
    assert!(
        context.entities.iter().any(|e| e.id == alice_id),
        "Pinned Alice in context"
    );
    assert!(
        context.entities.iter().any(|e| e.id == bob_id),
        "Recent Bob in context"
    );
    assert!(
        context.entities.iter().any(|e| e.id == charlie_id),
        "Mentioned Charlie in context"
    );

    // Charlie (mentioned) should have highest score
    let charlie_entity = context
        .entities
        .iter()
        .find(|e| e.id == charlie_id)
        .unwrap();
    let alice_entity = context.entities.iter().find(|e| e.id == alice_id).unwrap();

    assert!(
        charlie_entity.score > alice_entity.score,
        "Mentioned entity should score higher than pinned (10.0 vs 2.0)"
    );
}

/// Test that context includes hot entities.
#[tokio::test]
async fn test_context_hot_entities() {
    let harness = TestHarness::new().await;
    let temp_dir = TempDir::new().expect("Temp dir");
    let session_path = temp_dir.path().join("session.json");

    let entity_repo = SurrealEntityRepository::new(harness.db.clone());
    let session_manager =
        Arc::new(SessionStateManager::load_or_create(&session_path).expect("Session"));

    // Create a character
    let alice = entity_repo
        .create_character(CharacterBuilder::new("Alice").role("protagonist").build())
        .await
        .expect("Alice");

    let alice_id = format!("character:{}", alice.id.key());

    // Record access to make it a "hot" entity
    session_manager.record_access(&alice_id).await;

    // Create context service and verify hot entities includes Alice
    let context_service = CachedContextService::new(harness.db.clone(), session_manager);

    let hot = context_service.get_hot_entities(10).await;
    assert!(
        hot.contains(&alice_id),
        "Recently accessed entity should be in hot list"
    );
}
