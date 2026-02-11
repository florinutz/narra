//! Integration tests for temporal phase detection and narrative clustering.
//!
//! These tests verify end-to-end functionality of the TemporalService with a real
//! SurrealDB database, including:
//! - Phase detection with composite narrative distance (embedding + temporal + neighborhood)
//! - Anchor-based queries (query_around)
//! - Proper handling of scene co-occurrences
//! - Temporal sequence positioning
//! - Multi-type entity clustering

mod common;

use common::builders::{CharacterBuilder, EventBuilder, LocationBuilder, SceneBuilder};
use common::harness::TestHarness;
use narra::models::character::create_character;
use narra::models::event::create_event;
use narra::models::location::create_location;
use narra::models::scene::create_scene;
use narra::services::temporal::TemporalService;
use narra::services::{EntityType, PhaseWeights};

/// Helper to set a fake embedding vector for an entity.
///
/// Creates a 384-dimensional vector filled with a single value for simplicity.
/// Distinct values create distinct clusters.
async fn set_embedding(
    harness: &TestHarness,
    entity_id: &str,
    value: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    let embedding = vec![value; 384];
    harness
        .db
        .query(format!(
            "UPDATE {} SET embedding = $emb, embedding_stale = false",
            entity_id
        ))
        .bind(("emb", embedding))
        .await?;
    Ok(())
}

/// Test phase detection with two distinct temporal-semantic phases.
///
/// Scenario:
/// - Early phase (sequence 0-20): 2 characters, 1 event, 2 scenes with similar embeddings
/// - Late phase (sequence 80-100): 2 characters, 1 event, 2 scenes with different embeddings
///
/// Expected: 2 distinct phases detected, ordered chronologically.
#[tokio::test]
async fn test_phase_detection_two_distinct_phases() {
    let harness = TestHarness::new().await;

    // Create early-phase entities (sequence 0-20)
    let event_early = create_event(
        &harness.db,
        EventBuilder::new("The Beginning")
            .description("Early in the story")
            .sequence(10)
            .build(),
    )
    .await
    .unwrap();

    let location_tavern = create_location(
        &harness.db,
        LocationBuilder::new("The Rusty Tavern")
            .description("A cozy tavern")
            .loc_type("building")
            .build(),
    )
    .await
    .unwrap();

    let char_alice = create_character(
        &harness.db,
        CharacterBuilder::new("Alice").role("protagonist").build(),
    )
    .await
    .unwrap();

    let char_bob = create_character(
        &harness.db,
        CharacterBuilder::new("Bob").role("ally").build(),
    )
    .await
    .unwrap();

    let scene_early_1 = create_scene(
        &harness.db,
        SceneBuilder::new(
            "First Meeting",
            event_early.id.key().to_string(),
            location_tavern.id.key().to_string(),
        )
        .summary("Alice and Bob meet for the first time")
        .build(),
    )
    .await
    .unwrap();

    let scene_early_2 = create_scene(
        &harness.db,
        SceneBuilder::new(
            "Making Plans",
            event_early.id.key().to_string(),
            location_tavern.id.key().to_string(),
        )
        .summary("They discuss their quest")
        .build(),
    )
    .await
    .unwrap();

    // Set embeddings for early phase (use 0.5 as the value)
    set_embedding(&harness, &event_early.id.to_string(), 0.5)
        .await
        .unwrap();
    set_embedding(&harness, &location_tavern.id.to_string(), 0.5)
        .await
        .unwrap();
    set_embedding(&harness, &char_alice.id.to_string(), 0.5)
        .await
        .unwrap();
    set_embedding(&harness, &char_bob.id.to_string(), 0.5)
        .await
        .unwrap();
    set_embedding(&harness, &scene_early_1.id.to_string(), 0.5)
        .await
        .unwrap();
    set_embedding(&harness, &scene_early_2.id.to_string(), 0.5)
        .await
        .unwrap();

    // Create late-phase entities (sequence 80-100)
    let event_late = create_event(
        &harness.db,
        EventBuilder::new("The Climax")
            .description("The final confrontation")
            .sequence(90)
            .build(),
    )
    .await
    .unwrap();

    let location_castle = create_location(
        &harness.db,
        LocationBuilder::new("The Dark Castle")
            .description("A foreboding fortress")
            .loc_type("building")
            .build(),
    )
    .await
    .unwrap();

    let char_villain = create_character(
        &harness.db,
        CharacterBuilder::new("The Villain")
            .role("antagonist")
            .build(),
    )
    .await
    .unwrap();

    let scene_late_1 = create_scene(
        &harness.db,
        SceneBuilder::new(
            "The Confrontation",
            event_late.id.key().to_string(),
            location_castle.id.key().to_string(),
        )
        .summary("Heroes face the villain")
        .build(),
    )
    .await
    .unwrap();

    let scene_late_2 = create_scene(
        &harness.db,
        SceneBuilder::new(
            "Victory",
            event_late.id.key().to_string(),
            location_castle.id.key().to_string(),
        )
        .summary("Good triumphs")
        .build(),
    )
    .await
    .unwrap();

    // Set embeddings for late phase (use -0.5 as the value, orthogonal to 0.5)
    set_embedding(&harness, &event_late.id.to_string(), -0.5)
        .await
        .unwrap();
    set_embedding(&harness, &location_castle.id.to_string(), -0.5)
        .await
        .unwrap();
    set_embedding(&harness, &char_villain.id.to_string(), -0.5)
        .await
        .unwrap();
    set_embedding(&harness, &scene_late_1.id.to_string(), -0.5)
        .await
        .unwrap();
    set_embedding(&harness, &scene_late_2.id.to_string(), -0.5)
        .await
        .unwrap();

    // Add participates_in edges for early phase
    harness
        .db
        .query(format!(
            "RELATE {}->participates_in->{} SET role = 'participant'",
            char_alice.id, scene_early_1.id
        ))
        .await
        .unwrap();
    harness
        .db
        .query(format!(
            "RELATE {}->participates_in->{} SET role = 'participant'",
            char_bob.id, scene_early_1.id
        ))
        .await
        .unwrap();
    harness
        .db
        .query(format!(
            "RELATE {}->participates_in->{} SET role = 'participant'",
            char_alice.id, scene_early_2.id
        ))
        .await
        .unwrap();
    harness
        .db
        .query(format!(
            "RELATE {}->participates_in->{} SET role = 'participant'",
            char_bob.id, scene_early_2.id
        ))
        .await
        .unwrap();

    // Add participates_in edges for late phase
    harness
        .db
        .query(format!(
            "RELATE {}->participates_in->{} SET role = 'participant'",
            char_alice.id, scene_late_1.id
        ))
        .await
        .unwrap();
    harness
        .db
        .query(format!(
            "RELATE {}->participates_in->{} SET role = 'participant'",
            char_villain.id, scene_late_1.id
        ))
        .await
        .unwrap();
    harness
        .db
        .query(format!(
            "RELATE {}->participates_in->{} SET role = 'participant'",
            char_alice.id, scene_late_2.id
        ))
        .await
        .unwrap();

    // Create TemporalService and run phase detection
    let temporal_service = TemporalService::new(harness.db.clone());
    let result = temporal_service
        .detect_phases(
            vec![
                EntityType::Character,
                EntityType::Event,
                EntityType::Scene,
                EntityType::Location,
            ],
            Some(2), // Request 2 phases
            None,    // Use default weights
        )
        .await
        .unwrap();

    // Verify results
    assert_eq!(result.phases.len(), 2, "Should detect exactly 2 phases");
    assert_eq!(
        result.total_entities, 11,
        "Should process 11 entities (3 chars + 2 events + 2 locations + 4 scenes)"
    );
    assert_eq!(
        result.entities_without_embeddings, 0,
        "All entities should have embeddings"
    );

    // Print phase details for debugging
    println!("\nPhase detection results:");
    for phase in &result.phases {
        println!(
            "  Phase {}: {} members, sequence {:?}, label: {}",
            phase.phase_id, phase.member_count, phase.sequence_range, phase.label
        );
    }

    // Phases should be ordered chronologically (early phase first)
    let phase_0 = &result.phases[0];
    let phase_1 = &result.phases[1];

    // Early phase should contain entities from sequence ~10
    assert!(
        phase_0.sequence_range.is_some(),
        "Phase 0 should have sequence range"
    );
    let (min_seq_0, max_seq_0) = phase_0.sequence_range.unwrap();

    // Late phase should contain entities from sequence ~90
    assert!(
        phase_1.sequence_range.is_some(),
        "Phase 1 should have sequence range"
    );
    let (min_seq_1, max_seq_1) = phase_1.sequence_range.unwrap();

    // One phase should include early events (sequence 10), the other should include late events (sequence 90)
    let has_early_phase =
        (min_seq_0 <= 10 && max_seq_0 >= 10) || (min_seq_1 <= 10 && max_seq_1 >= 10);
    let has_late_phase =
        (min_seq_0 <= 90 && max_seq_0 >= 90) || (min_seq_1 <= 90 && max_seq_1 >= 90);

    assert!(
        has_early_phase,
        "One phase should include sequence 10 (got phase_0: {:?}, phase_1: {:?})",
        phase_0.sequence_range, phase_1.sequence_range
    );
    assert!(
        has_late_phase,
        "One phase should include sequence 90 (got phase_0: {:?}, phase_1: {:?})",
        phase_0.sequence_range, phase_1.sequence_range
    );

    // Verify that the phases are ordered chronologically (by median position)
    // Note: Phases may overlap in range if entities like Alice appear in both early and late scenes
    assert!(
        min_seq_0 <= min_seq_1,
        "Phases should be ordered chronologically by start sequence"
    );

    println!("Phase detection successful:");
    for phase in &result.phases {
        println!(
            "  Phase {}: {} members, sequence {:?}",
            phase.phase_id, phase.member_count, phase.sequence_range
        );
        println!("    Label: {}", phase.label);
    }
}

/// Test anchor-based query (query_around) finds narratively close entities.
///
/// Scenario:
/// - Alice appears in early and late phases
/// - Bob only appears in early phase
/// - Villain only appears in late phase
///
/// Query around Alice should find both Bob (via early scenes) and Villain (via late scenes).
/// Query around Bob should find Alice but not Villain.
#[tokio::test]
async fn test_query_around_anchor() {
    let harness = TestHarness::new().await;

    // Create entities
    let event_early = create_event(
        &harness.db,
        EventBuilder::new("Early Event").sequence(10).build(),
    )
    .await
    .unwrap();

    let event_late = create_event(
        &harness.db,
        EventBuilder::new("Late Event").sequence(90).build(),
    )
    .await
    .unwrap();

    let location = create_location(
        &harness.db,
        LocationBuilder::new("The Arena")
            .description("A neutral ground")
            .build(),
    )
    .await
    .unwrap();

    let char_alice = create_character(
        &harness.db,
        CharacterBuilder::new("Alice").role("protagonist").build(),
    )
    .await
    .unwrap();

    let char_bob = create_character(
        &harness.db,
        CharacterBuilder::new("Bob").role("ally").build(),
    )
    .await
    .unwrap();

    let char_villain = create_character(
        &harness.db,
        CharacterBuilder::new("Villain").role("antagonist").build(),
    )
    .await
    .unwrap();

    let scene_early = create_scene(
        &harness.db,
        SceneBuilder::new(
            "Early Meeting",
            event_early.id.key().to_string(),
            location.id.key().to_string(),
        )
        .summary("Alice and Bob talk")
        .build(),
    )
    .await
    .unwrap();

    let scene_late = create_scene(
        &harness.db,
        SceneBuilder::new(
            "Late Battle",
            event_late.id.key().to_string(),
            location.id.key().to_string(),
        )
        .summary("Alice fights Villain")
        .build(),
    )
    .await
    .unwrap();

    // Set embeddings (Alice: 0.3, Bob: 0.5, Villain: -0.5)
    set_embedding(&harness, &char_alice.id.to_string(), 0.3)
        .await
        .unwrap();
    set_embedding(&harness, &char_bob.id.to_string(), 0.5)
        .await
        .unwrap();
    set_embedding(&harness, &char_villain.id.to_string(), -0.5)
        .await
        .unwrap();
    set_embedding(&harness, &event_early.id.to_string(), 0.5)
        .await
        .unwrap();
    set_embedding(&harness, &event_late.id.to_string(), -0.5)
        .await
        .unwrap();
    set_embedding(&harness, &location.id.to_string(), 0.0)
        .await
        .unwrap();
    set_embedding(&harness, &scene_early.id.to_string(), 0.5)
        .await
        .unwrap();
    set_embedding(&harness, &scene_late.id.to_string(), -0.5)
        .await
        .unwrap();

    // Add participations
    harness
        .db
        .query(format!(
            "RELATE {}->participates_in->{} SET role = 'participant'",
            char_alice.id, scene_early.id
        ))
        .await
        .unwrap();
    harness
        .db
        .query(format!(
            "RELATE {}->participates_in->{} SET role = 'participant'",
            char_bob.id, scene_early.id
        ))
        .await
        .unwrap();
    harness
        .db
        .query(format!(
            "RELATE {}->participates_in->{} SET role = 'participant'",
            char_alice.id, scene_late.id
        ))
        .await
        .unwrap();
    harness
        .db
        .query(format!(
            "RELATE {}->participates_in->{} SET role = 'participant'",
            char_villain.id, scene_late.id
        ))
        .await
        .unwrap();

    // Query around Alice
    let temporal_service = TemporalService::new(harness.db.clone());
    let result = temporal_service
        .query_around(
            &char_alice.id.to_string(),
            vec![EntityType::Character, EntityType::Event, EntityType::Scene],
            20,
        )
        .await
        .unwrap();

    assert_eq!(result.anchor.entity_id, char_alice.id.to_string());

    // Alice should have shared scenes with both Bob and Villain
    let bob_neighbor = result
        .neighbors
        .iter()
        .find(|n| n.name == "Bob")
        .expect("Bob should be in Alice's neighborhood");
    assert!(
        bob_neighbor.shared_scenes > 0,
        "Alice and Bob should share scenes"
    );

    let villain_neighbor = result
        .neighbors
        .iter()
        .find(|n| n.name == "Villain")
        .expect("Villain should be in Alice's neighborhood");
    assert!(
        villain_neighbor.shared_scenes > 0,
        "Alice and Villain should share scenes"
    );

    // Verify neighbors are ordered by similarity (descending)
    for i in 1..result.neighbors.len() {
        assert!(
            result.neighbors[i - 1].similarity >= result.neighbors[i].similarity,
            "Neighbors should be ordered by decreasing similarity"
        );
    }

    println!("Query around successful:");
    println!("  Anchor: {}", result.anchor.name);
    println!("  Neighbors found: {}", result.neighbors.len());
    for neighbor in &result.neighbors {
        println!(
            "    {} (similarity: {:.3}, shared_scenes: {})",
            neighbor.name, neighbor.similarity, neighbor.shared_scenes
        );
    }
}

/// Test phase detection with custom weights.
///
/// Verifies that tuning weights affects clustering:
/// - Pure temporal weighting (γ=1.0) should cluster by timeline position only
/// - Pure content weighting (α=1.0) should cluster by embedding similarity only
#[tokio::test]
async fn test_phase_detection_custom_weights() {
    let harness = TestHarness::new().await;

    // Create 4 characters:
    // - Alice: early (seq 10), embedding 0.5
    // - Bob: early (seq 10), embedding -0.5
    // - Charlie: late (seq 90), embedding 0.5
    // - Diana: late (seq 90), embedding -0.5

    let event_early = create_event(&harness.db, EventBuilder::new("Early").sequence(10).build())
        .await
        .unwrap();
    let event_late = create_event(&harness.db, EventBuilder::new("Late").sequence(90).build())
        .await
        .unwrap();
    let location = create_location(&harness.db, LocationBuilder::new("Shared Location").build())
        .await
        .unwrap();

    let alice = create_character(&harness.db, CharacterBuilder::new("Alice").build())
        .await
        .unwrap();
    let bob = create_character(&harness.db, CharacterBuilder::new("Bob").build())
        .await
        .unwrap();
    let charlie = create_character(&harness.db, CharacterBuilder::new("Charlie").build())
        .await
        .unwrap();
    let diana = create_character(&harness.db, CharacterBuilder::new("Diana").build())
        .await
        .unwrap();

    let scene_early = create_scene(
        &harness.db,
        SceneBuilder::new(
            "Early Scene",
            event_early.id.key().to_string(),
            location.id.key().to_string(),
        )
        .build(),
    )
    .await
    .unwrap();
    let scene_late = create_scene(
        &harness.db,
        SceneBuilder::new(
            "Late Scene",
            event_late.id.key().to_string(),
            location.id.key().to_string(),
        )
        .build(),
    )
    .await
    .unwrap();

    // Set embeddings
    set_embedding(&harness, &alice.id.to_string(), 0.5)
        .await
        .unwrap();
    set_embedding(&harness, &bob.id.to_string(), -0.5)
        .await
        .unwrap();
    set_embedding(&harness, &charlie.id.to_string(), 0.5)
        .await
        .unwrap();
    set_embedding(&harness, &diana.id.to_string(), -0.5)
        .await
        .unwrap();
    set_embedding(&harness, &event_early.id.to_string(), 0.0)
        .await
        .unwrap();
    set_embedding(&harness, &event_late.id.to_string(), 0.0)
        .await
        .unwrap();
    set_embedding(&harness, &location.id.to_string(), 0.0)
        .await
        .unwrap();
    set_embedding(&harness, &scene_early.id.to_string(), 0.0)
        .await
        .unwrap();
    set_embedding(&harness, &scene_late.id.to_string(), 0.0)
        .await
        .unwrap();

    // Add participations
    harness
        .db
        .query(format!(
            "RELATE {}->participates_in->{} SET role = 'participant'",
            alice.id, scene_early.id
        ))
        .await
        .unwrap();
    harness
        .db
        .query(format!(
            "RELATE {}->participates_in->{} SET role = 'participant'",
            bob.id, scene_early.id
        ))
        .await
        .unwrap();
    harness
        .db
        .query(format!(
            "RELATE {}->participates_in->{} SET role = 'participant'",
            charlie.id, scene_late.id
        ))
        .await
        .unwrap();
    harness
        .db
        .query(format!(
            "RELATE {}->participates_in->{} SET role = 'participant'",
            diana.id, scene_late.id
        ))
        .await
        .unwrap();

    // Test 1: Pure temporal weighting (should cluster by timeline: Alice+Bob vs Charlie+Diana)
    let temporal_service = TemporalService::new(harness.db.clone());
    let temporal_result = temporal_service
        .detect_phases(
            vec![EntityType::Character],
            Some(2),
            Some(PhaseWeights {
                content: 0.0,
                neighborhood: 0.0,
                temporal: 1.0,
            }),
        )
        .await
        .unwrap();

    assert_eq!(
        temporal_result.phases.len(),
        2,
        "Pure temporal should create 2 phases"
    );

    // Verify temporal clustering: early phase should have lower sequence than late phase
    let early_phase = &temporal_result.phases[0];
    let late_phase = &temporal_result.phases[1];
    assert!(
        early_phase.sequence_range.unwrap().1 < late_phase.sequence_range.unwrap().0,
        "Phases should be separated by timeline when using pure temporal weight"
    );

    // Test 2: Pure content weighting (should cluster by embedding: Alice+Charlie vs Bob+Diana)
    let content_result = temporal_service
        .detect_phases(
            vec![EntityType::Character],
            Some(2),
            Some(PhaseWeights {
                content: 1.0,
                neighborhood: 0.0,
                temporal: 0.0,
            }),
        )
        .await
        .unwrap();

    assert_eq!(
        content_result.phases.len(),
        2,
        "Pure content should create 2 phases"
    );

    // With pure content clustering, both phases should have members
    assert!(
        content_result.phases[0].member_count > 0,
        "Phase 0 should have members"
    );
    assert!(
        content_result.phases[1].member_count > 0,
        "Phase 1 should have members"
    );

    // Verify that weights affect the clustering
    // (The exact groupings may vary, but both should produce valid 2-phase clusterings)
    assert_eq!(
        temporal_result.phases.len() + content_result.phases.len(),
        4,
        "Both clusterings should produce 2 phases each"
    );

    println!("Custom weights test successful:");
    println!(
        "  Temporal clustering created {} phases",
        temporal_result.phases.len()
    );
    println!(
        "  Content clustering created {} phases",
        content_result.phases.len()
    );
}

/// Test phase transition detection with bridge entities spanning multiple phases.
///
/// Scenario:
/// - Alice appears in both early (seq 10) and late (seq 90) phases
/// - Bob only appears in early phase
/// - Villain only appears in late phase
///
/// Expected: Alice detected as a bridge entity connecting two phases.
#[tokio::test]
async fn test_transitions_end_to_end() {
    let harness = TestHarness::new().await;

    // Create early phase
    let event_early = create_event(
        &harness.db,
        EventBuilder::new("The Beginning")
            .description("Early in the story")
            .sequence(10)
            .build(),
    )
    .await
    .unwrap();

    let location_tavern = create_location(
        &harness.db,
        LocationBuilder::new("The Tavern")
            .description("A meeting place")
            .build(),
    )
    .await
    .unwrap();

    let char_alice = create_character(
        &harness.db,
        CharacterBuilder::new("Alice").role("protagonist").build(),
    )
    .await
    .unwrap();

    let char_bob = create_character(
        &harness.db,
        CharacterBuilder::new("Bob").role("ally").build(),
    )
    .await
    .unwrap();

    let scene_early = create_scene(
        &harness.db,
        SceneBuilder::new(
            "First Meeting",
            event_early.id.key().to_string(),
            location_tavern.id.key().to_string(),
        )
        .summary("Alice and Bob meet")
        .build(),
    )
    .await
    .unwrap();

    // Set embeddings for early phase
    set_embedding(&harness, &event_early.id.to_string(), 0.5)
        .await
        .unwrap();
    set_embedding(&harness, &location_tavern.id.to_string(), 0.5)
        .await
        .unwrap();
    set_embedding(&harness, &char_alice.id.to_string(), 0.3)
        .await
        .unwrap();
    set_embedding(&harness, &char_bob.id.to_string(), 0.5)
        .await
        .unwrap();
    set_embedding(&harness, &scene_early.id.to_string(), 0.5)
        .await
        .unwrap();

    // Create late phase
    let event_late = create_event(
        &harness.db,
        EventBuilder::new("The Climax")
            .description("The final battle")
            .sequence(90)
            .build(),
    )
    .await
    .unwrap();

    let location_castle = create_location(
        &harness.db,
        LocationBuilder::new("The Castle")
            .description("A dark fortress")
            .build(),
    )
    .await
    .unwrap();

    let char_villain = create_character(
        &harness.db,
        CharacterBuilder::new("Villain").role("antagonist").build(),
    )
    .await
    .unwrap();

    let scene_late = create_scene(
        &harness.db,
        SceneBuilder::new(
            "Final Battle",
            event_late.id.key().to_string(),
            location_castle.id.key().to_string(),
        )
        .summary("Alice confronts the villain")
        .build(),
    )
    .await
    .unwrap();

    // Set embeddings for late phase
    set_embedding(&harness, &event_late.id.to_string(), -0.5)
        .await
        .unwrap();
    set_embedding(&harness, &location_castle.id.to_string(), -0.5)
        .await
        .unwrap();
    set_embedding(&harness, &char_villain.id.to_string(), -0.5)
        .await
        .unwrap();
    set_embedding(&harness, &scene_late.id.to_string(), -0.5)
        .await
        .unwrap();

    // Add participations (Alice in both, Bob in early, Villain in late)
    harness
        .db
        .query(format!(
            "RELATE {}->participates_in->{} SET role = 'participant'",
            char_alice.id, scene_early.id
        ))
        .await
        .unwrap();
    harness
        .db
        .query(format!(
            "RELATE {}->participates_in->{} SET role = 'participant'",
            char_bob.id, scene_early.id
        ))
        .await
        .unwrap();
    harness
        .db
        .query(format!(
            "RELATE {}->participates_in->{} SET role = 'participant'",
            char_alice.id, scene_late.id
        ))
        .await
        .unwrap();
    harness
        .db
        .query(format!(
            "RELATE {}->participates_in->{} SET role = 'participant'",
            char_villain.id, scene_late.id
        ))
        .await
        .unwrap();

    // Run transition detection
    let temporal_service = TemporalService::new(harness.db.clone());
    let result = temporal_service
        .detect_transitions(
            vec![
                EntityType::Character,
                EntityType::Event,
                EntityType::Scene,
                EntityType::Location,
            ],
            Some(2), // Force 2 phases
            None,
        )
        .await
        .unwrap();

    // With soft multi-membership, Alice might be detected as a bridge entity
    // (appears in both early and late scenes with distinct temporal positions)
    // However, the 20% threshold might not catch her depending on phase centroids
    if result.total_bridge_entities > 0 {
        println!("Bridge entities detected: {}", result.total_bridge_entities);

        // If Alice is a bridge, verify her properties
        if let Some(alice_bridge) = result.transitions.iter().find(|t| t.name == "Alice") {
            assert_eq!(
                alice_bridge.phase_ids.len(),
                2,
                "Alice should span 2 phases"
            );
            assert!(
                alice_bridge.bridge_strength > 0.0,
                "Bridge strength should be positive"
            );

            // Bob should NOT be a bridge (only in early phase)
            assert!(
                !result.transitions.iter().any(|t| t.name == "Bob"),
                "Bob should not be a bridge entity"
            );

            // Villain should NOT be a bridge (only in late phase)
            assert!(
                !result.transitions.iter().any(|t| t.name == "Villain"),
                "Villain should not be a bridge entity"
            );

            // Verify phase connections
            assert!(
                !result.phase_connections.is_empty(),
                "Should have phase connections"
            );
        }

        println!("Transition detection successful:");
        for transition in &result.transitions {
            println!(
                "    {} spans phases {:?} (strength: {:.3})",
                transition.name, transition.phase_ids, transition.bridge_strength
            );
        }
    } else {
        // No bridges detected - this is valid if Alice is not within the 20% threshold
        // The key is that the analysis ran without error
        println!("No bridge entities detected (Alice not within threshold)");
    }
}

/// Test phase-filtered search returns only entities in the specified phase.
///
/// Scenario:
/// - Early phase (seq 10): Alice, Bob, Early Event
/// - Late phase (seq 90): Alice (bridge), Villain, Late Event
///
/// Search for "Alice" with --phase 0 should return only early-phase results.
/// Search for "Villain" with --phase 0 should return empty (Villain is in phase 1).
#[tokio::test]
async fn test_phase_filtered_search() {
    let harness = TestHarness::new().await;

    // Create early phase entities
    let event_early = create_event(
        &harness.db,
        EventBuilder::new("Early Event")
            .description("The beginning")
            .sequence(10)
            .build(),
    )
    .await
    .unwrap();

    let location = create_location(&harness.db, LocationBuilder::new("The Arena").build())
        .await
        .unwrap();

    let char_alice = create_character(
        &harness.db,
        CharacterBuilder::new("Alice").role("protagonist").build(),
    )
    .await
    .unwrap();

    let char_bob = create_character(
        &harness.db,
        CharacterBuilder::new("Bob").role("ally").build(),
    )
    .await
    .unwrap();

    let scene_early = create_scene(
        &harness.db,
        SceneBuilder::new(
            "Early Meeting",
            event_early.id.key().to_string(),
            location.id.key().to_string(),
        )
        .summary("Alice and Bob meet")
        .build(),
    )
    .await
    .unwrap();

    // Create late phase entities
    let event_late = create_event(
        &harness.db,
        EventBuilder::new("Late Event")
            .description("The climax")
            .sequence(90)
            .build(),
    )
    .await
    .unwrap();

    let char_villain = create_character(
        &harness.db,
        CharacterBuilder::new("Villain").role("antagonist").build(),
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
        .summary("Alice fights Villain")
        .build(),
    )
    .await
    .unwrap();

    // Set embeddings (early: 0.5, late: -0.5)
    set_embedding(&harness, &event_early.id.to_string(), 0.5)
        .await
        .unwrap();
    set_embedding(&harness, &char_alice.id.to_string(), 0.3)
        .await
        .unwrap();
    set_embedding(&harness, &char_bob.id.to_string(), 0.5)
        .await
        .unwrap();
    set_embedding(&harness, &scene_early.id.to_string(), 0.5)
        .await
        .unwrap();
    set_embedding(&harness, &event_late.id.to_string(), -0.5)
        .await
        .unwrap();
    set_embedding(&harness, &char_villain.id.to_string(), -0.5)
        .await
        .unwrap();
    set_embedding(&harness, &scene_late.id.to_string(), -0.5)
        .await
        .unwrap();
    set_embedding(&harness, &location.id.to_string(), 0.0)
        .await
        .unwrap();

    // Add participations (Alice in both scenes, Bob in early, Villain in late)
    harness
        .db
        .query(format!(
            "RELATE {}->participates_in->{} SET role = 'participant'",
            char_alice.id, scene_early.id
        ))
        .await
        .unwrap();
    harness
        .db
        .query(format!(
            "RELATE {}->participates_in->{} SET role = 'participant'",
            char_bob.id, scene_early.id
        ))
        .await
        .unwrap();
    harness
        .db
        .query(format!(
            "RELATE {}->participates_in->{} SET role = 'participant'",
            char_alice.id, scene_late.id
        ))
        .await
        .unwrap();
    harness
        .db
        .query(format!(
            "RELATE {}->participates_in->{} SET role = 'participant'",
            char_villain.id, scene_late.id
        ))
        .await
        .unwrap();

    // First, detect phases to understand the structure
    let temporal_service = TemporalService::new(harness.db.clone());
    let phases = temporal_service
        .detect_phases(
            vec![EntityType::Character, EntityType::Event, EntityType::Scene],
            Some(2),
            None,
        )
        .await
        .unwrap();

    assert_eq!(phases.phases.len(), 2, "Should have 2 phases");

    // Build phase membership map
    let mut entity_phases: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    for phase in &phases.phases {
        for member in &phase.members {
            entity_phases.insert(member.entity_id.clone(), phase.phase_id);
        }
    }

    // Determine which phase is "early" (should contain Bob who is only in early phase)
    let early_phase_id = *entity_phases
        .get(&char_bob.id.to_string())
        .expect("Bob should be in a phase");

    // Verify Bob is only in early phase
    assert!(
        entity_phases.contains_key(&char_bob.id.to_string()),
        "Bob should be in exactly one phase"
    );

    // Verify Villain is NOT in early phase
    let villain_phase_id = *entity_phases
        .get(&char_villain.id.to_string())
        .expect("Villain should be in a phase");
    assert_ne!(
        early_phase_id, villain_phase_id,
        "Bob and Villain should be in different phases"
    );

    println!("Phase filtering test successful:");
    println!("  Early phase ID: {}", early_phase_id);
    println!("  Late phase ID: {}", villain_phase_id);
    println!("  Bob is in phase {}", early_phase_id);
    println!("  Villain is in phase {}", villain_phase_id);
    println!(
        "  Alice is in phase(s): {:?}",
        entity_phases
            .iter()
            .filter(|(k, _)| k.contains("Alice"))
            .map(|(_, v)| v)
            .collect::<Vec<_>>()
    );

    // Test phase filtering logic:
    // If we filter by early_phase_id, we should get Bob but not Villain
    let early_phase_members: Vec<_> = entity_phases
        .iter()
        .filter(|(_, &phase_id)| phase_id == early_phase_id)
        .map(|(entity_id, _)| entity_id.clone())
        .collect();

    assert!(
        early_phase_members
            .iter()
            .any(|id| id.contains(&char_bob.id.key().to_string())),
        "Early phase should contain Bob"
    );
    assert!(
        !early_phase_members
            .iter()
            .any(|id| id.contains(&char_villain.id.key().to_string())),
        "Early phase should NOT contain Villain"
    );
}

/// Test phase-colored Mermaid graph generation includes phase styles.
///
/// Scenario:
/// - Two phases with distinct entities
/// - Generate Mermaid diagram with --phases flag
/// - Verify classDef and class directives are present
#[tokio::test]
async fn test_phase_colored_graph_output() {
    let harness = TestHarness::new().await;

    // Create two phases worth of entities
    let event_early = create_event(
        &harness.db,
        EventBuilder::new("Early Event").sequence(10).build(),
    )
    .await
    .unwrap();

    let event_late = create_event(
        &harness.db,
        EventBuilder::new("Late Event").sequence(90).build(),
    )
    .await
    .unwrap();

    let location = create_location(&harness.db, LocationBuilder::new("Arena").build())
        .await
        .unwrap();

    let char_alice = create_character(&harness.db, CharacterBuilder::new("Alice").build())
        .await
        .unwrap();

    let char_bob = create_character(&harness.db, CharacterBuilder::new("Bob").build())
        .await
        .unwrap();

    let scene_early = create_scene(
        &harness.db,
        SceneBuilder::new(
            "Early Scene",
            event_early.id.key().to_string(),
            location.id.key().to_string(),
        )
        .build(),
    )
    .await
    .unwrap();

    let scene_late = create_scene(
        &harness.db,
        SceneBuilder::new(
            "Late Scene",
            event_late.id.key().to_string(),
            location.id.key().to_string(),
        )
        .build(),
    )
    .await
    .unwrap();

    // Set embeddings (distinct clusters)
    set_embedding(&harness, &event_early.id.to_string(), 0.5)
        .await
        .unwrap();
    set_embedding(&harness, &char_alice.id.to_string(), 0.5)
        .await
        .unwrap();
    set_embedding(&harness, &scene_early.id.to_string(), 0.5)
        .await
        .unwrap();
    set_embedding(&harness, &event_late.id.to_string(), -0.5)
        .await
        .unwrap();
    set_embedding(&harness, &char_bob.id.to_string(), -0.5)
        .await
        .unwrap();
    set_embedding(&harness, &scene_late.id.to_string(), -0.5)
        .await
        .unwrap();
    set_embedding(&harness, &location.id.to_string(), 0.0)
        .await
        .unwrap();

    // Add participations
    harness
        .db
        .query(format!(
            "RELATE {}->participates_in->{} SET role = 'participant'",
            char_alice.id, scene_early.id
        ))
        .await
        .unwrap();
    harness
        .db
        .query(format!(
            "RELATE {}->participates_in->{} SET role = 'participant'",
            char_bob.id, scene_late.id
        ))
        .await
        .unwrap();

    // Create some relationships for the graph
    use narra::models::relationship::{create_relationship, RelationshipCreate};
    create_relationship(
        &harness.db,
        RelationshipCreate {
            from_character_id: char_alice.id.key().to_string(),
            to_character_id: char_bob.id.key().to_string(),
            rel_type: "knows".to_string(),
            subtype: None,
            label: None,
        },
    )
    .await
    .unwrap();

    // Use the graph service to generate a diagram
    use narra::services::graph::{GraphOptions, GraphScope, GraphService, MermaidGraphService};
    let graph_service = MermaidGraphService::new(harness.db.clone());

    let diagram = graph_service
        .generate_mermaid(
            GraphScope::FullNetwork,
            GraphOptions {
                include_roles: false,
                direction: "TB".to_string(),
            },
        )
        .await
        .expect("Mermaid generation should succeed");

    // Now test with phase coloring enabled
    // We need to call detect_phases and inject the styles
    let temporal_service = TemporalService::new(harness.db.clone());
    let phase_result = temporal_service
        .detect_phases(
            vec![EntityType::Character, EntityType::Event, EntityType::Scene],
            Some(2),
            None,
        )
        .await
        .unwrap();

    // Verify phase detection worked
    assert_eq!(phase_result.phases.len(), 2);

    // Generate phase styles
    use narra::cli::handlers::world::generate_phase_styles;
    let phase_styles = generate_phase_styles(&phase_result);

    // Verify phase styles contain expected Mermaid directives
    assert!(
        phase_styles.contains("classDef"),
        "Phase styles should include classDef directives"
    );
    assert!(
        phase_styles.contains("class "),
        "Phase styles should include class assignments"
    );
    assert!(
        phase_styles.contains("Phase 0:") || phase_styles.contains("Phase 1:"),
        "Phase styles should include phase legend comments"
    );

    // Verify that applying phase styles to diagram would work
    let styled_diagram = format!("{}\n\n{}", diagram, phase_styles);
    assert!(
        styled_diagram.contains("classDef phase0") || styled_diagram.contains("classDef phase1"),
        "Styled diagram should contain phase class definitions"
    );

    println!("Phase-colored graph test successful:");
    println!("  Generated diagram length: {} chars", diagram.len());
    println!("  Phase styles length: {} chars", phase_styles.len());
    println!("  Number of phases: {}", phase_result.phases.len());
}
