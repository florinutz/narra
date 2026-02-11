//! Phase handler integration tests.
//!
//! Tests validate the MCP handler layer for narrative phase detection,
//! persistence, and temporal queries:
//! - SavePhases / LoadPhases roundtrip
//! - ClearPhases empties saved state
//! - DetectPhases with save flag
//! - QueryAround via MCP
//! - DetectTransitions via MCP
//! - UnifiedSearch with phase filter
//! - Error paths: insufficient entities, nonexistent anchor, clear on empty DB

use narra::mcp::{MutationRequest, QueryRequest};
use narra::models::character::create_character;
use narra::models::event::create_event;
use narra::models::location::create_location;
use narra::models::scene::create_scene;
use rmcp::handler::server::wrapper::Parameters;

use crate::common::{
    builders::{CharacterBuilder, EventBuilder, LocationBuilder, SceneBuilder},
    harness::{create_test_server, TestHarness},
    to_mutation_input, to_query_input,
};

/// IDs returned from the shared world setup.
#[allow(dead_code)]
struct PhaseWorld {
    alice_id: String,
    bob_id: String,
    villain_id: String,
}

/// Build a two-phase entity world with embeddings + scene edges.
///
/// Early phase (sequence 10): Alice + Bob in scene, embeddings [0.5; 384]
/// Late phase (sequence 90): Alice + Villain in scene, embeddings [-0.5; 384]
async fn create_phase_world(harness: &TestHarness) -> PhaseWorld {
    // Events
    let event_early = create_event(
        &harness.db,
        EventBuilder::new("The Beginning")
            .description("Early in the story")
            .sequence(10)
            .build(),
    )
    .await
    .unwrap();

    let event_late = create_event(
        &harness.db,
        EventBuilder::new("The Climax")
            .description("The final confrontation")
            .sequence(90)
            .build(),
    )
    .await
    .unwrap();

    // Location
    let location = create_location(
        &harness.db,
        LocationBuilder::new("The Arena")
            .description("A neutral ground")
            .loc_type("building")
            .build(),
    )
    .await
    .unwrap();

    // Characters
    let alice = create_character(
        &harness.db,
        CharacterBuilder::new("Alice").role("protagonist").build(),
    )
    .await
    .unwrap();

    let bob = create_character(
        &harness.db,
        CharacterBuilder::new("Bob").role("ally").build(),
    )
    .await
    .unwrap();

    let villain = create_character(
        &harness.db,
        CharacterBuilder::new("Villain").role("antagonist").build(),
    )
    .await
    .unwrap();

    // Scenes
    let scene_early = create_scene(
        &harness.db,
        SceneBuilder::new(
            "First Meeting",
            event_early.id.key().to_string(),
            location.id.key().to_string(),
        )
        .summary("Alice and Bob meet")
        .build(),
    )
    .await
    .unwrap();

    let scene_late = create_scene(
        &harness.db,
        SceneBuilder::new(
            "Final Battle",
            event_late.id.key().to_string(),
            location.id.key().to_string(),
        )
        .summary("Alice confronts the villain")
        .build(),
    )
    .await
    .unwrap();

    // Set embeddings (early cluster: 0.5, late cluster: -0.5)
    for (id, val) in [
        (event_early.id.to_string(), 0.5_f32),
        (location.id.to_string(), 0.0),
        (alice.id.to_string(), 0.3),
        (bob.id.to_string(), 0.5),
        (scene_early.id.to_string(), 0.5),
        (event_late.id.to_string(), -0.5),
        (villain.id.to_string(), -0.5),
        (scene_late.id.to_string(), -0.5),
    ] {
        let embedding = vec![val; 384];
        harness
            .db
            .query(format!(
                "UPDATE {} SET embedding = $emb, embedding_stale = false",
                id
            ))
            .bind(("emb", embedding))
            .await
            .unwrap();
    }

    // participates_in edges
    for (char_id, scene_id) in [
        (&alice.id, &scene_early.id),
        (&bob.id, &scene_early.id),
        (&alice.id, &scene_late.id),
        (&villain.id, &scene_late.id),
    ] {
        harness
            .db
            .query(format!(
                "RELATE {}->participates_in->{} SET role = 'participant'",
                char_id, scene_id
            ))
            .await
            .unwrap();
    }

    PhaseWorld {
        alice_id: alice.id.to_string(),
        bob_id: bob.id.to_string(),
        villain_id: villain.id.to_string(),
    }
}

// =============================================================================
// SAVE / LOAD / CLEAR ROUNDTRIP
// =============================================================================

/// SavePhases → LoadPhases roundtrip: 2 phases persisted and loaded correctly.
#[tokio::test]
async fn test_save_phases_then_load_roundtrip() {
    let harness = TestHarness::new().await;
    let _world = create_phase_world(&harness).await;
    let server = create_test_server(&harness).await;

    // SavePhases (mutation)
    let save_response = server
        .handle_mutate(Parameters(to_mutation_input(MutationRequest::SavePhases {
            entity_types: None,
            num_phases: Some(2),
            content_weight: None,
            neighborhood_weight: None,
            temporal_weight: None,
        })))
        .await
        .expect("SavePhases should succeed");

    assert_eq!(save_response.entity.entity_type, "phase_batch");
    assert!(
        save_response.entity.name.contains("2 phases"),
        "Should save 2 phases, got: {}",
        save_response.entity.name
    );

    // LoadPhases (query)
    let load_response = server
        .handle_query(Parameters(to_query_input(QueryRequest::LoadPhases)))
        .await
        .expect("LoadPhases should succeed");

    assert_eq!(load_response.total, 2, "Should load 2 saved phases");
    assert_eq!(load_response.results.len(), 2);
    for result in &load_response.results {
        assert_eq!(result.entity_type, "narrative_phase");
        assert!(result.content.contains("members"));
    }
}

/// Save → ClearPhases → Load returns empty.
#[tokio::test]
async fn test_clear_phases_empties_saved() {
    let harness = TestHarness::new().await;
    let _world = create_phase_world(&harness).await;
    let server = create_test_server(&harness).await;

    // Save first
    server
        .handle_mutate(Parameters(to_mutation_input(MutationRequest::SavePhases {
            entity_types: None,
            num_phases: Some(2),
            content_weight: None,
            neighborhood_weight: None,
            temporal_weight: None,
        })))
        .await
        .expect("SavePhases should succeed");

    // Clear
    let clear_response = server
        .handle_mutate(Parameters(to_mutation_input(MutationRequest::ClearPhases)))
        .await
        .expect("ClearPhases should succeed");

    assert_eq!(clear_response.entity.entity_type, "phase_batch");
    assert!(
        clear_response.entity.content.contains("Cleared"),
        "Should confirm clearing: {}",
        clear_response.entity.content
    );

    // Load should return empty
    let load_response = server
        .handle_query(Parameters(to_query_input(QueryRequest::LoadPhases)))
        .await
        .expect("LoadPhases should succeed");

    assert_eq!(
        load_response.total, 0,
        "Should have no saved phases after clear"
    );
    assert!(load_response.results.is_empty());
}

/// DetectPhases with save=false doesn't persist; save=true does.
#[tokio::test]
async fn test_detect_phases_save_flag() {
    let harness = TestHarness::new().await;
    let _world = create_phase_world(&harness).await;
    let server = create_test_server(&harness).await;

    // DetectPhases with save=false
    let detect_response = server
        .handle_query(Parameters(to_query_input(QueryRequest::DetectPhases {
            entity_types: None,
            num_phases: Some(2),
            content_weight: None,
            neighborhood_weight: None,
            temporal_weight: None,
            save: Some(false),
        })))
        .await
        .expect("DetectPhases should succeed");

    assert_eq!(detect_response.total, 2, "Should detect 2 phases");

    // LoadPhases should return empty (not saved)
    let load_response = server
        .handle_query(Parameters(to_query_input(QueryRequest::LoadPhases)))
        .await
        .expect("LoadPhases should succeed");
    assert_eq!(
        load_response.total, 0,
        "Should have no saved phases when save=false"
    );

    // DetectPhases with save=true
    let detect_saved = server
        .handle_query(Parameters(to_query_input(QueryRequest::DetectPhases {
            entity_types: None,
            num_phases: Some(2),
            content_weight: None,
            neighborhood_weight: None,
            temporal_weight: None,
            save: Some(true),
        })))
        .await
        .expect("DetectPhases with save should succeed");

    assert_eq!(detect_saved.total, 2);

    // LoadPhases should now return phases
    let load_response = server
        .handle_query(Parameters(to_query_input(QueryRequest::LoadPhases)))
        .await
        .expect("LoadPhases should succeed");
    assert_eq!(
        load_response.total, 2,
        "Should have 2 saved phases when save=true"
    );
}

// =============================================================================
// QUERY AROUND
// =============================================================================

/// QueryAround returns neighbors as EntityResults with similarity scores.
#[tokio::test]
async fn test_query_around_via_mcp() {
    let harness = TestHarness::new().await;
    let world = create_phase_world(&harness).await;
    let server = create_test_server(&harness).await;

    let response = server
        .handle_query(Parameters(to_query_input(QueryRequest::QueryAround {
            anchor_id: world.alice_id.clone(),
            entity_types: Some(vec!["character".to_string()]),
            limit: Some(10),
        })))
        .await
        .expect("QueryAround should succeed");

    // Should find Bob and Villain as neighbors
    assert!(
        response.total >= 2,
        "Should find at least Bob and Villain, got {}",
        response.total
    );

    // All results should be entity type "character"
    for result in &response.results {
        assert_eq!(result.entity_type, "character");
        assert!(
            result.confidence.is_some(),
            "Results should have similarity score"
        );
    }

    // Hints should mention anchor
    assert!(
        response.hints.iter().any(|h| h.contains("Anchor")),
        "Hints should mention anchor entity"
    );
}

// =============================================================================
// DETECT TRANSITIONS
// =============================================================================

/// DetectTransitions returns bridge entities spanning phases.
#[tokio::test]
async fn test_detect_transitions_via_mcp() {
    let harness = TestHarness::new().await;
    let _world = create_phase_world(&harness).await;
    let server = create_test_server(&harness).await;

    let response = server
        .handle_query(Parameters(to_query_input(
            QueryRequest::DetectTransitions {
                entity_types: None,
                num_phases: Some(2),
                content_weight: None,
                neighborhood_weight: None,
                temporal_weight: None,
            },
        )))
        .await
        .expect("DetectTransitions should succeed");

    // Hints should report number of bridge entities and phases
    assert!(
        response.hints.iter().any(|h| h.contains("phases")),
        "Hints should mention phases analyzed"
    );

    // If bridges found, they should have proper structure
    for result in &response.results {
        assert!(
            result.content.contains("Bridges"),
            "Bridge entity content should describe phase bridging: {}",
            result.content
        );
        assert!(result.confidence.is_some(), "Should have bridge_strength");
    }
}

// =============================================================================
// UNIFIED SEARCH WITH PHASE FILTER
// =============================================================================

/// Save phases, then UnifiedSearch with phase filter limits results to that phase.
#[tokio::test]
async fn test_unified_search_phase_filter() {
    let harness = TestHarness::new().await;
    let world = create_phase_world(&harness).await;
    let server = create_test_server(&harness).await;

    // Save phases first
    server
        .handle_mutate(Parameters(to_mutation_input(MutationRequest::SavePhases {
            entity_types: None,
            num_phases: Some(2),
            content_weight: None,
            neighborhood_weight: None,
            temporal_weight: None,
        })))
        .await
        .expect("SavePhases should succeed");

    // Load phases to find which phase Bob is in (early phase)
    let load_response = server
        .handle_query(Parameters(to_query_input(QueryRequest::LoadPhases)))
        .await
        .expect("LoadPhases should succeed");

    // Identify Bob's phase by checking member content
    let bob_phase_id = load_response
        .results
        .iter()
        .find(|r| r.content.contains("Bob"))
        .map(|r| {
            // Extract phase ID from the result id like "phase:phase_0"
            r.name
                .split(':')
                .next()
                .unwrap_or("")
                .replace("Phase ", "")
                .trim()
                .parse::<usize>()
                .unwrap_or(0)
        })
        .expect("Bob should be in a phase");

    // Search with that phase filter — should include Bob but not Villain
    let search_response = server
        .handle_query(Parameters(to_query_input(QueryRequest::UnifiedSearch {
            query: "character".to_string(),
            mode: "hybrid".to_string(),
            entity_types: Some(vec!["character".to_string()]),
            limit: Some(20),
            phase: Some(bob_phase_id),
            filter: None,
        })))
        .await
        .expect("UnifiedSearch with phase filter should succeed");

    // Check that the phase filter hint is present
    let has_phase_hint = search_response
        .hints
        .iter()
        .any(|h| h.contains("phase") || h.contains("Phase"));
    assert!(
        has_phase_hint,
        "Should have phase filtering hint in: {:?}",
        search_response.hints
    );

    // If results returned, Villain should not be in Bob's phase
    let villain_in_results = search_response
        .results
        .iter()
        .any(|r| r.id == world.villain_id);
    if !search_response.results.is_empty() {
        assert!(
            !villain_in_results,
            "Villain should not appear in Bob's early phase"
        );
    }
}

// =============================================================================
// ERROR PATHS
// =============================================================================

/// DetectPhases with fewer than 3 entities returns a graceful error.
#[tokio::test]
async fn test_detect_phases_insufficient_entities() {
    let harness = TestHarness::new().await;
    let server = create_test_server(&harness).await;

    // Create only 1 character with embedding (need ≥ num_phases for k-means)
    let alice = create_character(
        &harness.db,
        CharacterBuilder::new("Alice").role("protagonist").build(),
    )
    .await
    .unwrap();
    let emb = vec![0.5_f32; 384];
    harness
        .db
        .query(format!(
            "UPDATE {} SET embedding = $emb, embedding_stale = false",
            alice.id
        ))
        .bind(("emb", emb))
        .await
        .unwrap();

    // Request 3 phases with only 1 entity — should fail
    let result = server
        .handle_query(Parameters(to_query_input(QueryRequest::DetectPhases {
            entity_types: Some(vec!["character".to_string()]),
            num_phases: Some(3),
            content_weight: None,
            neighborhood_weight: None,
            temporal_weight: None,
            save: None,
        })))
        .await;

    assert!(
        result.is_err(),
        "DetectPhases should fail with insufficient entities"
    );
}

/// QueryAround with nonexistent anchor returns error.
#[tokio::test]
async fn test_query_around_nonexistent_anchor() {
    let harness = TestHarness::new().await;
    let server = create_test_server(&harness).await;

    let result = server
        .handle_query(Parameters(to_query_input(QueryRequest::QueryAround {
            anchor_id: "character:nonexistent_ghost".to_string(),
            entity_types: None,
            limit: Some(10),
        })))
        .await;

    assert!(
        result.is_err(),
        "QueryAround should fail for nonexistent anchor"
    );
}

/// ClearPhases on empty DB returns 0 count, no error.
#[tokio::test]
async fn test_clear_phases_on_empty_db() {
    let harness = TestHarness::new().await;
    let server = create_test_server(&harness).await;

    let response = server
        .handle_mutate(Parameters(to_mutation_input(MutationRequest::ClearPhases)))
        .await
        .expect("ClearPhases on empty DB should succeed");

    assert_eq!(response.entity.entity_type, "phase_batch");
    assert!(
        response.entity.content.contains("Cleared 0"),
        "Should report 0 phases cleared: {}",
        response.entity.content
    );
}
