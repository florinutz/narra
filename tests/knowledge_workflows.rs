//! Integration tests for knowledge system workflows (SERV-02).
//!
//! Tests knowledge facts, certainty levels, temporal queries, and provenance.
//!
//! The knowledge system uses an append-only pattern where certainty changes
//! create new edges rather than updating existing ones. This preserves the
//! complete history of what characters knew and when.

mod common;

use pretty_assertions::assert_eq;

use narra::models::{CertaintyLevel, KnowledgeStateCreate, LearningMethod};
use narra::repository::{
    EntityRepository, KnowledgeRepository, SurrealEntityRepository, SurrealKnowledgeRepository,
};

use common::builders::{CharacterBuilder, EventBuilder, KnowledgeBuilder};
use common::harness::TestHarness;

// ============================================================================
// KNOWLEDGE FACT TESTS
// ============================================================================

/// Test creating and retrieving knowledge facts.
#[tokio::test]
async fn test_knowledge_fact_crud() {
    let harness = TestHarness::new().await;
    let entity_repo = SurrealEntityRepository::new(harness.db.clone());
    let knowledge_repo = SurrealKnowledgeRepository::new(harness.db.clone());

    // Create a character to own the knowledge fact
    let narrator = entity_repo
        .create_character(CharacterBuilder::new("Narrator").role("narrator").build())
        .await
        .expect("Should create narrator");
    let narrator_id = narrator.id.key().to_string();

    // Create a knowledge fact
    let fact = knowledge_repo
        .create_knowledge(
            KnowledgeBuilder::new("The butler has a key to the study")
                .for_character(&narrator_id)
                .build(),
        )
        .await
        .expect("Should create knowledge");

    assert_eq!(fact.fact, "The butler has a key to the study");

    // Retrieve by ID
    let fetched = knowledge_repo
        .get_knowledge(&fact.id.key().to_string())
        .await
        .expect("Should fetch knowledge")
        .expect("Knowledge should exist");

    assert_eq!(fetched.fact, fact.fact);
}

/// Test searching knowledge by fact content.
#[tokio::test]
async fn test_knowledge_search() {
    let harness = TestHarness::new().await;
    let entity_repo = SurrealEntityRepository::new(harness.db.clone());
    let knowledge_repo = SurrealKnowledgeRepository::new(harness.db.clone());

    // Create narrator to own facts
    let narrator = entity_repo
        .create_character(CharacterBuilder::new("Narrator").build())
        .await
        .expect("Narrator");
    let narrator_id = narrator.id.key().to_string();

    // Create multiple facts
    knowledge_repo
        .create_knowledge(
            KnowledgeBuilder::new("The butler has a key")
                .for_character(&narrator_id)
                .build(),
        )
        .await
        .expect("Fact 1");
    knowledge_repo
        .create_knowledge(
            KnowledgeBuilder::new("The cook was asleep")
                .for_character(&narrator_id)
                .build(),
        )
        .await
        .expect("Fact 2");
    knowledge_repo
        .create_knowledge(
            KnowledgeBuilder::new("The butler was seen near the study")
                .for_character(&narrator_id)
                .build(),
        )
        .await
        .expect("Fact 3");

    // Search for butler-related facts
    let results = knowledge_repo
        .search_knowledge("butler")
        .await
        .expect("Should search knowledge");

    assert_eq!(results.len(), 2, "Should find 2 butler-related facts");
    assert!(results.iter().all(|k| k.fact.contains("butler")));
}

// ============================================================================
// CERTAINTY LEVEL TESTS
// ============================================================================

/// Test knowledge state creation with different certainty levels.
#[tokio::test]
async fn test_knowledge_certainty_levels() {
    let harness = TestHarness::new().await;
    let entity_repo = SurrealEntityRepository::new(harness.db.clone());
    let knowledge_repo = SurrealKnowledgeRepository::new(harness.db.clone());

    // Setup: create character and fact
    let character = entity_repo
        .create_character(CharacterBuilder::new("Detective Alice").build())
        .await
        .expect("Character");
    let character_id = character.id.key().to_string();

    // Create narrator to own the fact
    let narrator = entity_repo
        .create_character(CharacterBuilder::new("Narrator").build())
        .await
        .expect("Narrator");
    let narrator_id = narrator.id.key().to_string();

    let fact = knowledge_repo
        .create_knowledge(
            KnowledgeBuilder::new("The murder weapon was a candlestick")
                .for_character(&narrator_id)
                .build(),
        )
        .await
        .expect("Fact");
    let fact_id = format!("knowledge:{}", fact.id.key());

    // Create event for provenance
    let event = entity_repo
        .create_event(EventBuilder::new("Discovery").sequence(1).build())
        .await
        .expect("Event");
    let event_id = event.id.key().to_string();

    // Test: create knowledge state with SUSPECTS certainty
    let state = knowledge_repo
        .create_knowledge_state(
            &character_id,
            &fact_id,
            KnowledgeStateCreate {
                certainty: CertaintyLevel::Suspects,
                learning_method: LearningMethod::Discovered,
                event: Some(event_id.clone()),
                ..Default::default()
            },
        )
        .await
        .expect("Should create knowledge state");

    assert_eq!(state.certainty, CertaintyLevel::Suspects);

    // Verify character has this knowledge state
    let states = knowledge_repo
        .get_character_knowledge_states(&character_id)
        .await
        .expect("Should get states");

    assert_eq!(states.len(), 1);
    assert_eq!(states[0].certainty, CertaintyLevel::Suspects);
}

/// Test certainty progression: Suspects -> Believes -> Knows.
///
/// This test verifies the append-only pattern where certainty changes
/// create new knowledge state edges rather than updating existing ones.
#[tokio::test]
async fn test_knowledge_certainty_workflow() {
    let harness = TestHarness::new().await;
    let entity_repo = SurrealEntityRepository::new(harness.db.clone());
    let knowledge_repo = SurrealKnowledgeRepository::new(harness.db.clone());

    // Setup character
    let character = entity_repo
        .create_character(CharacterBuilder::new("Alice").build())
        .await
        .expect("Character");
    let char_id = character.id.key().to_string();

    // Create narrator to own the fact
    let narrator = entity_repo
        .create_character(CharacterBuilder::new("Narrator").build())
        .await
        .expect("Narrator");
    let narrator_id = narrator.id.key().to_string();

    let fact = knowledge_repo
        .create_knowledge(
            KnowledgeBuilder::new("The gardener is the killer")
                .for_character(&narrator_id)
                .build(),
        )
        .await
        .expect("Fact");
    let fact_id = format!("knowledge:{}", fact.id.key());

    // Event 1: Initial suspicion
    let event1 = entity_repo
        .create_event(EventBuilder::new("Finds muddy boots").sequence(1).build())
        .await
        .expect("Event 1");
    let event1_id = event1.id.key().to_string();

    // Event 2: Growing evidence
    let event2 = entity_repo
        .create_event(
            EventBuilder::new("Witnesses alibi collapse")
                .sequence(2)
                .build(),
        )
        .await
        .expect("Event 2");
    let event2_id = event2.id.key().to_string();

    // Event 3: Confession
    let event3 = entity_repo
        .create_event(EventBuilder::new("Confession obtained").sequence(3).build())
        .await
        .expect("Event 3");
    let event3_id = event3.id.key().to_string();

    // Initial state: Suspects
    knowledge_repo
        .create_knowledge_state(
            &char_id,
            &fact_id,
            KnowledgeStateCreate {
                certainty: CertaintyLevel::Suspects,
                learning_method: LearningMethod::Discovered,
                event: Some(event1_id.clone()),
                ..Default::default()
            },
        )
        .await
        .expect("Initial state");

    // Update to Believes (using the repository's update method)
    knowledge_repo
        .update_certainty(
            &char_id,
            &fact_id,
            CertaintyLevel::Uncertain, // Using Uncertain as proxy for "believes"
            &event2_id,
        )
        .await
        .expect("Update to Believes");

    // Update to Knows
    knowledge_repo
        .update_certainty(&char_id, &fact_id, CertaintyLevel::Knows, &event3_id)
        .await
        .expect("Update to Knows");

    // Verify current state is Knows
    let current = knowledge_repo
        .get_current_knowledge(&char_id, &fact_id)
        .await
        .expect("Should get current")
        .expect("Should exist");

    assert_eq!(
        current.certainty,
        CertaintyLevel::Knows,
        "Current certainty should be Knows after progression"
    );
}

// ============================================================================
// TEMPORAL QUERY TESTS
// ============================================================================

/// Test point-in-time knowledge query (what did character know at event X?).
///
/// This verifies that temporal queries correctly return the knowledge state
/// as it existed at the time of a specific event.
#[tokio::test]
async fn test_knowledge_at_event() {
    let harness = TestHarness::new().await;
    let entity_repo = SurrealEntityRepository::new(harness.db.clone());
    let knowledge_repo = SurrealKnowledgeRepository::new(harness.db.clone());

    // Setup character
    let character = entity_repo
        .create_character(CharacterBuilder::new("Witness").build())
        .await
        .expect("Character");
    let char_id = character.id.key().to_string();

    // Create narrator to own facts
    let narrator = entity_repo
        .create_character(CharacterBuilder::new("Narrator").build())
        .await
        .expect("Narrator");
    let narrator_id = narrator.id.key().to_string();

    // Create sequence of events
    let event1 = entity_repo
        .create_event(EventBuilder::new("Sees suspect").sequence(1).build())
        .await
        .expect("Event 1");
    let event1_id = event1.id.key().to_string();

    let event2 = entity_repo
        .create_event(EventBuilder::new("Hears alibi").sequence(2).build())
        .await
        .expect("Event 2");
    let event2_id = event2.id.key().to_string();

    // Create two facts
    let fact1 = knowledge_repo
        .create_knowledge(
            KnowledgeBuilder::new("Suspect was at the scene")
                .for_character(&narrator_id)
                .build(),
        )
        .await
        .expect("Fact 1");
    let fact1_id = format!("knowledge:{}", fact1.id.key());

    let fact2 = knowledge_repo
        .create_knowledge(
            KnowledgeBuilder::new("Suspect has an alibi")
                .for_character(&narrator_id)
                .build(),
        )
        .await
        .expect("Fact 2");
    let fact2_id = format!("knowledge:{}", fact2.id.key());

    // Character learns fact1 at event1
    knowledge_repo
        .create_knowledge_state(
            &char_id,
            &fact1_id,
            KnowledgeStateCreate {
                certainty: CertaintyLevel::Knows,
                learning_method: LearningMethod::Witnessed,
                event: Some(event1_id.clone()),
                ..Default::default()
            },
        )
        .await
        .expect("State 1");

    // Character learns fact2 at event2
    knowledge_repo
        .create_knowledge_state(
            &char_id,
            &fact2_id,
            KnowledgeStateCreate {
                certainty: CertaintyLevel::Knows,
                learning_method: LearningMethod::Told,
                event: Some(event2_id.clone()),
                ..Default::default()
            },
        )
        .await
        .expect("State 2");

    // Query knowledge at event1 (should only have fact1)
    let knowledge_at_event1 = knowledge_repo
        .get_knowledge_at_event(&char_id, &event1_id)
        .await
        .expect("Should query at event1");

    assert_eq!(
        knowledge_at_event1.len(),
        1,
        "At event1, character should know only 1 fact"
    );

    // Query knowledge at event2 (should have both facts)
    let knowledge_at_event2 = knowledge_repo
        .get_knowledge_at_event(&char_id, &event2_id)
        .await
        .expect("Should query at event2");

    assert_eq!(
        knowledge_at_event2.len(),
        2,
        "At event2, character should know 2 facts"
    );
}

// ============================================================================
// APPEND-ONLY HISTORY TESTS
// ============================================================================

/// Test that knowledge history preserves all state changes (append-only).
///
/// This is a critical test for the knowledge system's append-only design.
/// When certainty changes, we create new edges rather than updating existing
/// ones, preserving the complete history of what a character believed over time.
#[tokio::test]
async fn test_knowledge_history_append_only() {
    let harness = TestHarness::new().await;
    let entity_repo = SurrealEntityRepository::new(harness.db.clone());
    let knowledge_repo = SurrealKnowledgeRepository::new(harness.db.clone());

    // Setup
    let character = entity_repo
        .create_character(CharacterBuilder::new("Investigator").build())
        .await
        .expect("Character");
    let char_id = character.id.key().to_string();

    // Create narrator to own fact
    let narrator = entity_repo
        .create_character(CharacterBuilder::new("Narrator").build())
        .await
        .expect("Narrator");
    let narrator_id = narrator.id.key().to_string();

    let fact = knowledge_repo
        .create_knowledge(
            KnowledgeBuilder::new("The safe was opened at midnight")
                .for_character(&narrator_id)
                .build(),
        )
        .await
        .expect("Fact");
    let fact_id = format!("knowledge:{}", fact.id.key());

    // Create three events for the progression
    let event1 = entity_repo
        .create_event(EventBuilder::new("E1").sequence(1).build())
        .await
        .expect("E1");
    let event2 = entity_repo
        .create_event(EventBuilder::new("E2").sequence(2).build())
        .await
        .expect("E2");
    let event3 = entity_repo
        .create_event(EventBuilder::new("E3").sequence(3).build())
        .await
        .expect("E3");
    let e1_id = event1.id.key().to_string();
    let e2_id = event2.id.key().to_string();
    let e3_id = event3.id.key().to_string();

    // Initial: Suspects
    knowledge_repo
        .create_knowledge_state(
            &char_id,
            &fact_id,
            KnowledgeStateCreate {
                certainty: CertaintyLevel::Suspects,
                learning_method: LearningMethod::Discovered,
                event: Some(e1_id),
                ..Default::default()
            },
        )
        .await
        .expect("Initial");

    // Update 1: Uncertain (more evidence)
    knowledge_repo
        .update_certainty(&char_id, &fact_id, CertaintyLevel::Uncertain, &e2_id)
        .await
        .expect("Update 1");

    // Update 2: Knows (confirmed)
    knowledge_repo
        .update_certainty(&char_id, &fact_id, CertaintyLevel::Knows, &e3_id)
        .await
        .expect("Update 2");

    // Get full history
    let history = knowledge_repo
        .get_knowledge_history(&char_id, &fact_id)
        .await
        .expect("Should get history");

    // CRITICAL: Append-only means ALL three states should exist
    assert_eq!(
        history.len(),
        3,
        "Append-only history should preserve all 3 states"
    );

    // Verify progression is captured
    let certainties: Vec<CertaintyLevel> = history.iter().map(|s| s.certainty).collect();

    assert!(certainties.contains(&CertaintyLevel::Suspects));
    assert!(certainties.contains(&CertaintyLevel::Uncertain));
    assert!(certainties.contains(&CertaintyLevel::Knows));
}

// ============================================================================
// PROVENANCE TESTS
// ============================================================================

/// Test that we can track who told whom about a fact.
///
/// Provenance tracking enables writers to understand how information
/// spread through their story - who knew what, and who told whom.
#[tokio::test]
async fn test_knowledge_provenance_source() {
    let harness = TestHarness::new().await;
    let entity_repo = SurrealEntityRepository::new(harness.db.clone());
    let knowledge_repo = SurrealKnowledgeRepository::new(harness.db.clone());

    // Create two characters
    let alice = entity_repo
        .create_character(CharacterBuilder::new("Alice").role("detective").build())
        .await
        .expect("Alice");
    let alice_id = alice.id.key().to_string();

    let bob = entity_repo
        .create_character(CharacterBuilder::new("Bob").role("witness").build())
        .await
        .expect("Bob");
    let bob_id = bob.id.key().to_string();

    // Create narrator to own fact
    let narrator = entity_repo
        .create_character(CharacterBuilder::new("Narrator").build())
        .await
        .expect("Narrator");
    let narrator_id = narrator.id.key().to_string();

    // Create a fact and event
    let fact = knowledge_repo
        .create_knowledge(
            KnowledgeBuilder::new("The victim had enemies")
                .for_character(&narrator_id)
                .build(),
        )
        .await
        .expect("Fact");
    let fact_id = format!("knowledge:{}", fact.id.key());

    let event = entity_repo
        .create_event(EventBuilder::new("Bob's testimony").sequence(1).build())
        .await
        .expect("Event");
    let event_id = event.id.key().to_string();

    // Bob knows the fact first (direct knowledge)
    knowledge_repo
        .create_knowledge_state(
            &bob_id,
            &fact_id,
            KnowledgeStateCreate {
                certainty: CertaintyLevel::Knows,
                learning_method: LearningMethod::Witnessed,
                event: Some(event_id.clone()),
                source_character: None, // Direct knowledge
                ..Default::default()
            },
        )
        .await
        .expect("Bob's state");

    // Alice learns from Bob (provenance tracking)
    knowledge_repo
        .create_knowledge_state(
            &alice_id,
            &fact_id,
            KnowledgeStateCreate {
                certainty: CertaintyLevel::Uncertain, // Hearsay - less certain
                learning_method: LearningMethod::Told,
                event: Some(event_id.clone()),
                source_character: Some(bob_id.clone()), // Source is Bob
                ..Default::default()
            },
        )
        .await
        .expect("Alice's state");

    // Query who knows this fact
    let knowers = knowledge_repo
        .get_fact_knowers(&fact_id)
        .await
        .expect("Should get knowers");

    assert_eq!(knowers.len(), 2, "Both Alice and Bob should know the fact");

    // Alice's knowledge should have Bob as source
    let alice_state = knowers
        .iter()
        .find(|s| s.character.key().to_string() == alice_id)
        .expect("Alice should be a knower");

    assert!(
        alice_state.source_character.is_some(),
        "Alice's knowledge should have a source"
    );
    let source_id = alice_state
        .source_character
        .as_ref()
        .map(|r| r.key().to_string());
    assert_eq!(
        source_id,
        Some(bob_id.clone()),
        "Alice's source should be Bob"
    );
}

/// Test multi-character perception filtering (different characters know different things).
///
/// This verifies that the knowledge system correctly isolates what each
/// character knows - a critical feature for narrative consistency.
#[tokio::test]
async fn test_multi_character_knowledge_isolation() {
    let harness = TestHarness::new().await;
    let entity_repo = SurrealEntityRepository::new(harness.db.clone());
    let knowledge_repo = SurrealKnowledgeRepository::new(harness.db.clone());

    // Three characters
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

    // Create narrator to own facts
    let narrator = entity_repo
        .create_character(CharacterBuilder::new("Narrator").build())
        .await
        .expect("Narrator");
    let narrator_id = narrator.id.key().to_string();

    // Three facts
    let fact1 = knowledge_repo
        .create_knowledge(
            KnowledgeBuilder::new("Fact 1: Secret")
                .for_character(&narrator_id)
                .build(),
        )
        .await
        .expect("F1");
    let fact2 = knowledge_repo
        .create_knowledge(
            KnowledgeBuilder::new("Fact 2: Clue")
                .for_character(&narrator_id)
                .build(),
        )
        .await
        .expect("F2");
    let fact3 = knowledge_repo
        .create_knowledge(
            KnowledgeBuilder::new("Fact 3: Red herring")
                .for_character(&narrator_id)
                .build(),
        )
        .await
        .expect("F3");

    let event = entity_repo
        .create_event(EventBuilder::new("E").sequence(1).build())
        .await
        .expect("E");
    let event_id = event.id.key().to_string();

    // Alice knows fact1 and fact2
    knowledge_repo
        .create_knowledge_state(
            &alice.id.key().to_string(),
            &format!("knowledge:{}", fact1.id.key()),
            KnowledgeStateCreate {
                certainty: CertaintyLevel::Knows,
                learning_method: LearningMethod::Discovered,
                event: Some(event_id.clone()),
                ..Default::default()
            },
        )
        .await
        .expect("Alice-F1");
    knowledge_repo
        .create_knowledge_state(
            &alice.id.key().to_string(),
            &format!("knowledge:{}", fact2.id.key()),
            KnowledgeStateCreate {
                certainty: CertaintyLevel::Knows,
                learning_method: LearningMethod::Discovered,
                event: Some(event_id.clone()),
                ..Default::default()
            },
        )
        .await
        .expect("Alice-F2");

    // Bob knows fact2 and fact3
    knowledge_repo
        .create_knowledge_state(
            &bob.id.key().to_string(),
            &format!("knowledge:{}", fact2.id.key()),
            KnowledgeStateCreate {
                certainty: CertaintyLevel::Knows,
                learning_method: LearningMethod::Discovered,
                event: Some(event_id.clone()),
                ..Default::default()
            },
        )
        .await
        .expect("Bob-F2");
    knowledge_repo
        .create_knowledge_state(
            &bob.id.key().to_string(),
            &format!("knowledge:{}", fact3.id.key()),
            KnowledgeStateCreate {
                certainty: CertaintyLevel::Knows,
                learning_method: LearningMethod::Discovered,
                event: Some(event_id.clone()),
                ..Default::default()
            },
        )
        .await
        .expect("Bob-F3");

    // Charlie knows nothing

    // Verify isolation
    let alice_knowledge = knowledge_repo
        .get_character_knowledge_states(&alice.id.key().to_string())
        .await
        .expect("Alice's knowledge");
    let bob_knowledge = knowledge_repo
        .get_character_knowledge_states(&bob.id.key().to_string())
        .await
        .expect("Bob's knowledge");
    let charlie_knowledge = knowledge_repo
        .get_character_knowledge_states(&charlie.id.key().to_string())
        .await
        .expect("Charlie's knowledge");

    assert_eq!(alice_knowledge.len(), 2, "Alice should know 2 facts");
    assert_eq!(bob_knowledge.len(), 2, "Bob should know 2 facts");
    assert_eq!(charlie_knowledge.len(), 0, "Charlie should know 0 facts");
}
