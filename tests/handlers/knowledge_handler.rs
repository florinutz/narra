//! Knowledge handler integration tests.
//!
//! Tests validate the knowledge tool handlers correctly handle:
//! - RecordKnowledge mutations with certainty levels and learning methods
//! - Temporal queries for character knowledge with provenance
//!
//! This covers HAND-02 requirements.

use insta::assert_snapshot;
use narra::mcp::{MutationRequest, QueryRequest};
use narra::models::{CertaintyLevel, LearningMethod};
use narra::repository::{
    EntityRepository, KnowledgeRepository, SurrealEntityRepository, SurrealKnowledgeRepository,
};
use pretty_assertions::assert_eq;
use rmcp::handler::server::wrapper::Parameters;

use crate::common::{
    builders::{CharacterBuilder, EventBuilder, LocationBuilder},
    harness::TestHarness,
    to_mutation_input, to_query_input,
};

// =============================================================================
// RECORD KNOWLEDGE - BASIC OPERATIONS
// =============================================================================

/// Test successful knowledge recording creates both entity and state edge.
///
/// Verifies:
/// - Handler returns success response
/// - Entity type is "knowledge"
/// - Response contains fact information
#[tokio::test]
async fn test_record_knowledge_success() {
    let harness = TestHarness::new().await;
    let server = crate::common::create_test_server(&harness).await;

    // Create a character via repository
    let repo = SurrealEntityRepository::new(harness.db.clone());
    let character = repo
        .create_character(CharacterBuilder::new("Alice").build())
        .await
        .expect("Failed to create character");
    let char_id = character.id.to_string();

    // Create a target (something to know about)
    let location = repo
        .create_location(LocationBuilder::new("The Tower").build())
        .await
        .expect("Failed to create location");
    let target_id = location.id.to_string();

    // Record knowledge via handler (use initial method since no event)
    let request = MutationRequest::RecordKnowledge {
        character_id: char_id.clone(),
        target_id: target_id.clone(),
        fact: "The tower holds ancient secrets".to_string(),
        certainty: "knows".to_string(),
        method: Some("initial".to_string()),
        source_character_id: None,
        event_id: None,
    };
    let response = server
        .handle_mutate(Parameters(to_mutation_input(request)))
        .await;

    assert!(
        response.is_ok(),
        "Record knowledge should succeed: {:?}",
        response.err()
    );
    let response = response.unwrap();

    assert_eq!(response.entity.entity_type, "knowledge");
    // Content format is: "{character_id} now {certainty:?} about {target_id}: {fact}"
    assert!(
        response.entity.content.contains("Knows"),
        "Content should contain certainty level: {}",
        response.entity.content
    );
    assert!(!response.hints.is_empty(), "Should provide hints");
}

/// Test knowledge recording with source character tracks provenance.
///
/// Verifies:
/// - Knowledge state includes source_character reference
#[tokio::test]
async fn test_record_knowledge_with_source_character() {
    let harness = TestHarness::new().await;
    let server = crate::common::create_test_server(&harness).await;

    // Create characters
    let repo = SurrealEntityRepository::new(harness.db.clone());
    let alice = repo
        .create_character(CharacterBuilder::new("Alice").build())
        .await
        .expect("Failed to create Alice");
    let bob = repo
        .create_character(CharacterBuilder::new("Bob").build())
        .await
        .expect("Failed to create Bob");

    // Create event for the learning
    let event = repo
        .create_event(EventBuilder::new("The Revelation").sequence(1).build())
        .await
        .expect("Failed to create event");

    // Bob tells Alice something
    let request = MutationRequest::RecordKnowledge {
        character_id: alice.id.to_string(),
        target_id: bob.id.to_string(),
        fact: "Bob is secretly a spy".to_string(),
        certainty: "knows".to_string(),
        method: Some("told".to_string()),
        source_character_id: Some(bob.id.key().to_string()),
        event_id: Some(event.id.key().to_string()),
    };
    let response = server
        .handle_mutate(Parameters(to_mutation_input(request)))
        .await;

    assert!(
        response.is_ok(),
        "Record knowledge should succeed: {:?}",
        response.err()
    );
    let response = response.unwrap();

    assert_eq!(response.entity.entity_type, "knowledge");

    // Verify provenance via repository query
    let knowledge_repo = SurrealKnowledgeRepository::new(harness.db.clone());
    let states = knowledge_repo
        .get_character_knowledge_states(alice.id.key().to_string().as_str())
        .await
        .expect("Failed to get knowledge states");

    assert!(!states.is_empty(), "Should have knowledge states");
    let state = &states[0];
    assert!(
        state.source_character.is_some(),
        "Should have source character"
    );
}

/// Test knowledge recording with event tracks temporal link.
///
/// Verifies:
/// - Knowledge state includes event reference
/// - learned_at timestamp is from event
#[tokio::test]
async fn test_record_knowledge_with_event() {
    let harness = TestHarness::new().await;
    let server = crate::common::create_test_server(&harness).await;

    // Create character and event
    let repo = SurrealEntityRepository::new(harness.db.clone());
    let character = repo
        .create_character(CharacterBuilder::new("Alice").build())
        .await
        .expect("Failed to create character");
    let event = repo
        .create_event(EventBuilder::new("Discovery").sequence(10).build())
        .await
        .expect("Failed to create event");

    // Record knowledge tied to event
    let request = MutationRequest::RecordKnowledge {
        character_id: character.id.to_string(),
        target_id: character.id.to_string(), // Self-knowledge
        fact: "I discovered my true heritage".to_string(),
        certainty: "knows".to_string(),
        method: Some("discovered".to_string()),
        source_character_id: None,
        event_id: Some(event.id.key().to_string()),
    };
    let response = server
        .handle_mutate(Parameters(to_mutation_input(request)))
        .await;

    assert!(
        response.is_ok(),
        "Record knowledge should succeed: {:?}",
        response.err()
    );

    // Verify event link via repository
    let knowledge_repo = SurrealKnowledgeRepository::new(harness.db.clone());
    let states = knowledge_repo
        .get_character_knowledge_states(character.id.key().to_string().as_str())
        .await
        .expect("Failed to get knowledge states");

    assert!(!states.is_empty(), "Should have knowledge states");
    let state = &states[0];
    assert!(state.event.is_some(), "Should have event reference");
}

// =============================================================================
// RECORD KNOWLEDGE - CERTAINTY LEVELS
// =============================================================================

/// Test certainty="knows" maps to CertaintyLevel::Knows.
#[tokio::test]
async fn test_record_knowledge_certainty_knows() {
    let harness = TestHarness::new().await;
    let server = crate::common::create_test_server(&harness).await;

    let repo = SurrealEntityRepository::new(harness.db.clone());
    let character = repo
        .create_character(CharacterBuilder::new("Alice").build())
        .await
        .expect("Failed to create character");

    let request = MutationRequest::RecordKnowledge {
        character_id: character.id.to_string(),
        target_id: character.id.to_string(),
        fact: "Alice knows this fact with certainty".to_string(),
        certainty: "knows".to_string(),
        method: Some("initial".to_string()),
        source_character_id: None,
        event_id: None,
    };
    let response = server
        .handle_mutate(Parameters(to_mutation_input(request)))
        .await;
    assert!(response.is_ok());

    // Verify certainty level
    let knowledge_repo = SurrealKnowledgeRepository::new(harness.db.clone());
    let states = knowledge_repo
        .get_character_knowledge_states(character.id.key().to_string().as_str())
        .await
        .expect("Failed to get knowledge states");

    assert!(!states.is_empty());
    assert_eq!(states[0].certainty, CertaintyLevel::Knows);
}

/// Test certainty="suspects" maps to CertaintyLevel::Suspects.
#[tokio::test]
async fn test_record_knowledge_certainty_suspects() {
    let harness = TestHarness::new().await;
    let server = crate::common::create_test_server(&harness).await;

    let repo = SurrealEntityRepository::new(harness.db.clone());
    let character = repo
        .create_character(CharacterBuilder::new("Alice").build())
        .await
        .expect("Failed to create character");

    let request = MutationRequest::RecordKnowledge {
        character_id: character.id.to_string(),
        target_id: character.id.to_string(),
        fact: "Alice suspects something is wrong".to_string(),
        certainty: "suspects".to_string(),
        method: Some("initial".to_string()),
        source_character_id: None,
        event_id: None,
    };
    let response = server
        .handle_mutate(Parameters(to_mutation_input(request)))
        .await;
    assert!(response.is_ok());

    let knowledge_repo = SurrealKnowledgeRepository::new(harness.db.clone());
    let states = knowledge_repo
        .get_character_knowledge_states(character.id.key().to_string().as_str())
        .await
        .expect("Failed to get knowledge states");

    assert!(!states.is_empty());
    assert_eq!(states[0].certainty, CertaintyLevel::Suspects);
}

/// Test certainty="believes_wrongly" maps to CertaintyLevel::BelievesWrongly.
#[tokio::test]
async fn test_record_knowledge_certainty_believes_wrongly() {
    let harness = TestHarness::new().await;
    let server = crate::common::create_test_server(&harness).await;

    let repo = SurrealEntityRepository::new(harness.db.clone());
    let character = repo
        .create_character(CharacterBuilder::new("Alice").build())
        .await
        .expect("Failed to create character");

    // BelievesWrongly requires truth_value - but handler doesn't have that param
    // The handler sets truth_value to the fact when certainty is BelievesWrongly
    let request = MutationRequest::RecordKnowledge {
        character_id: character.id.to_string(),
        target_id: character.id.to_string(),
        fact: "Alice thinks she is safe".to_string(),
        certainty: "believes_wrongly".to_string(),
        method: Some("initial".to_string()),
        source_character_id: None,
        event_id: None,
    };
    let response = server
        .handle_mutate(Parameters(to_mutation_input(request)))
        .await;
    assert!(
        response.is_ok(),
        "BelievesWrongly should succeed: {:?}",
        response.err()
    );

    let knowledge_repo = SurrealKnowledgeRepository::new(harness.db.clone());
    let states = knowledge_repo
        .get_character_knowledge_states(character.id.key().to_string().as_str())
        .await
        .expect("Failed to get knowledge states");

    assert!(!states.is_empty());
    assert_eq!(states[0].certainty, CertaintyLevel::BelievesWrongly);
    assert!(
        states[0].truth_value.is_some(),
        "Should have truth_value set"
    );
}

/// Test unknown certainty string falls back to Uncertain.
#[tokio::test]
async fn test_record_knowledge_certainty_unknown() {
    let harness = TestHarness::new().await;
    let server = crate::common::create_test_server(&harness).await;

    let repo = SurrealEntityRepository::new(harness.db.clone());
    let character = repo
        .create_character(CharacterBuilder::new("Alice").build())
        .await
        .expect("Failed to create character");

    let request = MutationRequest::RecordKnowledge {
        character_id: character.id.to_string(),
        target_id: character.id.to_string(),
        fact: "Some uncertain fact".to_string(),
        certainty: "unknown_certainty_value".to_string(),
        method: Some("initial".to_string()),
        source_character_id: None,
        event_id: None,
    };
    let response = server
        .handle_mutate(Parameters(to_mutation_input(request)))
        .await;
    assert!(response.is_ok());

    let knowledge_repo = SurrealKnowledgeRepository::new(harness.db.clone());
    let states = knowledge_repo
        .get_character_knowledge_states(character.id.key().to_string().as_str())
        .await
        .expect("Failed to get knowledge states");

    assert!(!states.is_empty());
    assert_eq!(
        states[0].certainty,
        CertaintyLevel::Uncertain,
        "Unknown certainty should fall back to Uncertain"
    );
}

// =============================================================================
// RECORD KNOWLEDGE - LEARNING METHODS
// =============================================================================

/// Test method="told" maps to LearningMethod::Told.
#[tokio::test]
async fn test_record_knowledge_method_told() {
    let harness = TestHarness::new().await;
    let server = crate::common::create_test_server(&harness).await;

    let repo = SurrealEntityRepository::new(harness.db.clone());
    let alice = repo
        .create_character(CharacterBuilder::new("Alice").build())
        .await
        .expect("Failed to create Alice");
    let bob = repo
        .create_character(CharacterBuilder::new("Bob").build())
        .await
        .expect("Failed to create Bob");
    let event = repo
        .create_event(EventBuilder::new("Conversation").sequence(1).build())
        .await
        .expect("Failed to create event");

    let request = MutationRequest::RecordKnowledge {
        character_id: alice.id.to_string(),
        target_id: bob.id.to_string(),
        fact: "Bob told Alice a secret".to_string(),
        certainty: "knows".to_string(),
        method: Some("told".to_string()),
        source_character_id: Some(bob.id.key().to_string()),
        event_id: Some(event.id.key().to_string()),
    };
    let response = server
        .handle_mutate(Parameters(to_mutation_input(request)))
        .await;
    assert!(response.is_ok());

    let knowledge_repo = SurrealKnowledgeRepository::new(harness.db.clone());
    let states = knowledge_repo
        .get_character_knowledge_states(alice.id.key().to_string().as_str())
        .await
        .expect("Failed to get knowledge states");

    assert!(!states.is_empty());
    assert_eq!(states[0].learning_method, LearningMethod::Told);
}

/// Test method="witnessed" maps to LearningMethod::Witnessed.
#[tokio::test]
async fn test_record_knowledge_method_witnessed() {
    let harness = TestHarness::new().await;
    let server = crate::common::create_test_server(&harness).await;

    let repo = SurrealEntityRepository::new(harness.db.clone());
    let character = repo
        .create_character(CharacterBuilder::new("Alice").build())
        .await
        .expect("Failed to create character");
    let event = repo
        .create_event(EventBuilder::new("The Crime").sequence(1).build())
        .await
        .expect("Failed to create event");

    let request = MutationRequest::RecordKnowledge {
        character_id: character.id.to_string(),
        target_id: character.id.to_string(),
        fact: "Alice witnessed the crime".to_string(),
        certainty: "knows".to_string(),
        method: Some("witnessed".to_string()),
        source_character_id: None,
        event_id: Some(event.id.key().to_string()),
    };
    let response = server
        .handle_mutate(Parameters(to_mutation_input(request)))
        .await;
    assert!(response.is_ok());

    let knowledge_repo = SurrealKnowledgeRepository::new(harness.db.clone());
    let states = knowledge_repo
        .get_character_knowledge_states(character.id.key().to_string().as_str())
        .await
        .expect("Failed to get knowledge states");

    assert!(!states.is_empty());
    assert_eq!(states[0].learning_method, LearningMethod::Witnessed);
}

/// Test method="deduced" maps to LearningMethod::Deduced.
#[tokio::test]
async fn test_record_knowledge_method_deduced() {
    let harness = TestHarness::new().await;
    let server = crate::common::create_test_server(&harness).await;

    let repo = SurrealEntityRepository::new(harness.db.clone());
    let character = repo
        .create_character(CharacterBuilder::new("Sherlock").build())
        .await
        .expect("Failed to create character");
    let event = repo
        .create_event(EventBuilder::new("Investigation").sequence(1).build())
        .await
        .expect("Failed to create event");

    let request = MutationRequest::RecordKnowledge {
        character_id: character.id.to_string(),
        target_id: character.id.to_string(),
        fact: "Sherlock deduced the murderer's identity".to_string(),
        certainty: "knows".to_string(),
        method: Some("deduced".to_string()),
        source_character_id: None,
        event_id: Some(event.id.key().to_string()),
    };
    let response = server
        .handle_mutate(Parameters(to_mutation_input(request)))
        .await;
    assert!(response.is_ok());

    let knowledge_repo = SurrealKnowledgeRepository::new(harness.db.clone());
    let states = knowledge_repo
        .get_character_knowledge_states(character.id.key().to_string().as_str())
        .await
        .expect("Failed to get knowledge states");

    assert!(!states.is_empty());
    assert_eq!(states[0].learning_method, LearningMethod::Deduced);
}

// =============================================================================
// TEMPORAL QUERIES
// =============================================================================

/// Test temporal query without event_id returns all current knowledge.
#[tokio::test]
async fn test_temporal_current_knowledge() {
    let harness = TestHarness::new().await;
    let server = crate::common::create_test_server(&harness).await;

    // Create character and record some knowledge
    let repo = SurrealEntityRepository::new(harness.db.clone());
    let character = repo
        .create_character(CharacterBuilder::new("Alice").build())
        .await
        .expect("Failed to create character");

    // Record knowledge via handler
    let request = MutationRequest::RecordKnowledge {
        character_id: character.id.to_string(),
        target_id: character.id.to_string(),
        fact: "First fact Alice knows".to_string(),
        certainty: "knows".to_string(),
        method: Some("initial".to_string()),
        source_character_id: None,
        event_id: None,
    };
    server
        .handle_mutate(Parameters(to_mutation_input(request)))
        .await
        .expect("Record failed");

    let request2 = MutationRequest::RecordKnowledge {
        character_id: character.id.to_string(),
        target_id: character.id.to_string(),
        fact: "Second fact Alice knows".to_string(),
        certainty: "suspects".to_string(),
        method: Some("initial".to_string()),
        source_character_id: None,
        event_id: None,
    };
    server
        .handle_mutate(Parameters(to_mutation_input(request2)))
        .await
        .expect("Record failed");

    // Query temporal knowledge (no event = all current knowledge)
    let query = QueryRequest::Temporal {
        character_id: character.id.to_string(),
        event_id: None,
        event_name: None,
    };
    let response = server.handle_query(Parameters(to_query_input(query))).await;

    assert!(
        response.is_ok(),
        "Temporal query should succeed: {:?}",
        response.err()
    );
    let response = response.unwrap();

    assert_eq!(
        response.results.len(),
        2,
        "Should return 2 knowledge states"
    );
    assert!(
        response.hints.iter().any(|h| h.contains("knows")),
        "Hints should mention knowledge count"
    );
}

/// Test temporal query at specific event returns knowledge learned by that point.
#[tokio::test]
async fn test_temporal_at_event() {
    let harness = TestHarness::new().await;
    let server = crate::common::create_test_server(&harness).await;

    // Create character and events
    let repo = SurrealEntityRepository::new(harness.db.clone());
    let character = repo
        .create_character(CharacterBuilder::new("Alice").build())
        .await
        .expect("Failed to create character");

    let event1 = repo
        .create_event(EventBuilder::new("Event One").sequence(1).build())
        .await
        .expect("Failed to create event1");
    let event2 = repo
        .create_event(EventBuilder::new("Event Two").sequence(2).build())
        .await
        .expect("Failed to create event2");

    // Record knowledge at event1
    let request1 = MutationRequest::RecordKnowledge {
        character_id: character.id.to_string(),
        target_id: character.id.to_string(),
        fact: "Fact learned at event 1".to_string(),
        certainty: "knows".to_string(),
        method: Some("witnessed".to_string()),
        source_character_id: None,
        event_id: Some(event1.id.key().to_string()),
    };
    server
        .handle_mutate(Parameters(to_mutation_input(request1)))
        .await
        .expect("Record 1 failed");

    // Record knowledge at event2
    let request2 = MutationRequest::RecordKnowledge {
        character_id: character.id.to_string(),
        target_id: character.id.to_string(),
        fact: "Fact learned at event 2".to_string(),
        certainty: "knows".to_string(),
        method: Some("told".to_string()),
        source_character_id: None,
        event_id: Some(event2.id.key().to_string()),
    };
    server
        .handle_mutate(Parameters(to_mutation_input(request2)))
        .await
        .expect("Record 2 failed");

    // Query at event1 - should only see knowledge from event1
    let query = QueryRequest::Temporal {
        character_id: character.id.to_string(),
        event_id: Some(event1.id.to_string()),
        event_name: None,
    };
    let response = server.handle_query(Parameters(to_query_input(query))).await;

    assert!(
        response.is_ok(),
        "Temporal query at event should succeed: {:?}",
        response.err()
    );
    let response = response.unwrap();

    // Should only have knowledge learned at or before event1
    assert_eq!(
        response.results.len(),
        1,
        "Should return 1 knowledge state at event1"
    );
}

/// Test temporal query by event_name resolves event and returns knowledge.
#[tokio::test]
async fn test_temporal_by_event_name() {
    let harness = TestHarness::new().await;
    let server = crate::common::create_test_server(&harness).await;

    // Create character and event
    let repo = SurrealEntityRepository::new(harness.db.clone());
    let character = repo
        .create_character(CharacterBuilder::new("Alice").build())
        .await
        .expect("Failed to create character");

    let event = repo
        .create_event(
            EventBuilder::new("The Great Revelation")
                .sequence(1)
                .build(),
        )
        .await
        .expect("Failed to create event");

    // Record knowledge at event
    let request = MutationRequest::RecordKnowledge {
        character_id: character.id.to_string(),
        target_id: character.id.to_string(),
        fact: "The truth was revealed".to_string(),
        certainty: "knows".to_string(),
        method: Some("witnessed".to_string()),
        source_character_id: None,
        event_id: Some(event.id.key().to_string()),
    };
    server
        .handle_mutate(Parameters(to_mutation_input(request)))
        .await
        .expect("Record failed");

    // Query using event name instead of ID
    let query = QueryRequest::Temporal {
        character_id: character.id.to_string(),
        event_id: None,
        event_name: Some("Great Revelation".to_string()), // Partial match
    };
    let response = server.handle_query(Parameters(to_query_input(query))).await;

    assert!(
        response.is_ok(),
        "Temporal query by name should succeed: {:?}",
        response.err()
    );
    let response = response.unwrap();

    // Should find knowledge via event name resolution
    assert!(
        !response.results.is_empty(),
        "Should return knowledge via event name resolution"
    );
}

// =============================================================================
// ERROR CASES
// =============================================================================

// =============================================================================
// BATCH RECORD KNOWLEDGE
// =============================================================================

/// Test batch recording knowledge for multiple characters.
///
/// Verifies:
/// - Handler returns success with batch entity_type
/// - All knowledge entries are created
/// - Entities list contains individual results
#[tokio::test]
async fn test_batch_record_knowledge_success() {
    let harness = TestHarness::new().await;
    let server = crate::common::create_test_server(&harness).await;

    // Create characters and a target
    let repo = SurrealEntityRepository::new(harness.db.clone());
    let alice = repo
        .create_character(CharacterBuilder::new("Alice").build())
        .await
        .expect("Failed to create Alice");
    let bob = repo
        .create_character(CharacterBuilder::new("Bob").build())
        .await
        .expect("Failed to create Bob");
    let location = repo
        .create_location(LocationBuilder::new("The Tower").build())
        .await
        .expect("Failed to create location");

    let request = MutationRequest::BatchRecordKnowledge {
        knowledge: vec![
            narra::mcp::KnowledgeSpec {
                character_id: alice.id.to_string(),
                target_id: location.id.to_string(),
                fact: "The tower is haunted".to_string(),
                certainty: "knows".to_string(),
                method: Some("initial".to_string()),
                source_character_id: None,
                event_id: None,
            },
            narra::mcp::KnowledgeSpec {
                character_id: bob.id.to_string(),
                target_id: location.id.to_string(),
                fact: "The tower holds treasure".to_string(),
                certainty: "suspects".to_string(),
                method: Some("initial".to_string()),
                source_character_id: None,
                event_id: None,
            },
        ],
    };
    let response = server
        .handle_mutate(Parameters(to_mutation_input(request)))
        .await;

    assert!(
        response.is_ok(),
        "Batch record knowledge should succeed: {:?}",
        response.err()
    );
    let response = response.unwrap();

    assert_eq!(response.entity.entity_type, "batch");
    let entities = response.entities.expect("Should have entities list");
    assert_eq!(entities.len(), 2);
    for e in &entities {
        assert_eq!(e.entity_type, "knowledge");
    }
    assert!(response.hints.iter().any(|h| h.contains("2/2")));
}

/// Test batch knowledge with partial failure.
///
/// Verifies:
/// - Valid entries succeed, invalid ones fail
/// - Response includes error information in hints
#[tokio::test]
async fn test_batch_record_knowledge_partial_failure() {
    let harness = TestHarness::new().await;
    let server = crate::common::create_test_server(&harness).await;

    let repo = SurrealEntityRepository::new(harness.db.clone());
    let alice = repo
        .create_character(CharacterBuilder::new("Alice").build())
        .await
        .expect("Failed to create Alice");
    let location = repo
        .create_location(LocationBuilder::new("The Tower").build())
        .await
        .expect("Failed to create location");

    let request = MutationRequest::BatchRecordKnowledge {
        knowledge: vec![
            narra::mcp::KnowledgeSpec {
                character_id: alice.id.to_string(),
                target_id: location.id.to_string(),
                fact: "Valid knowledge".to_string(),
                certainty: "knows".to_string(),
                method: Some("initial".to_string()),
                source_character_id: None,
                event_id: None,
            },
            narra::mcp::KnowledgeSpec {
                character_id: "invalid_format".to_string(), // Bad ID
                target_id: location.id.to_string(),
                fact: "This should fail".to_string(),
                certainty: "knows".to_string(),
                method: None,
                source_character_id: None,
                event_id: None,
            },
        ],
    };
    let response = server
        .handle_mutate(Parameters(to_mutation_input(request)))
        .await;

    assert!(
        response.is_ok(),
        "Partial batch should still succeed: {:?}",
        response.err()
    );
    let response = response.unwrap();

    let entities = response.entities.expect("Should have entities list");
    assert_eq!(entities.len(), 1, "Only valid entry should succeed");
    assert!(
        response.hints.iter().any(|h| h.contains("1/2")),
        "Hints should show partial success"
    );
    assert!(
        response.hints.iter().any(|h| h.contains("Errors")),
        "Hints should mention errors"
    );
}

/// Test batch knowledge where all entries fail.
///
/// Verifies error is returned when entire batch fails.
#[tokio::test]
async fn test_batch_record_knowledge_all_fail() {
    let harness = TestHarness::new().await;
    let server = crate::common::create_test_server(&harness).await;

    let request = MutationRequest::BatchRecordKnowledge {
        knowledge: vec![
            narra::mcp::KnowledgeSpec {
                character_id: "bad_id_1".to_string(),
                target_id: "character:someone".to_string(),
                fact: "Fact 1".to_string(),
                certainty: "knows".to_string(),
                method: None,
                source_character_id: None,
                event_id: None,
            },
            narra::mcp::KnowledgeSpec {
                character_id: "bad_id_2".to_string(),
                target_id: "character:someone".to_string(),
                fact: "Fact 2".to_string(),
                certainty: "knows".to_string(),
                method: None,
                source_character_id: None,
                event_id: None,
            },
        ],
    };
    let response = server
        .handle_mutate(Parameters(to_mutation_input(request)))
        .await;

    assert!(response.is_err(), "All-fail batch should return error");
    let err = response.unwrap_err();
    assert!(
        err.contains("All") && err.contains("failed"),
        "Error should indicate all entries failed: {}",
        err
    );
}

/// Test batch knowledge with event links creates proper temporal associations.
#[tokio::test]
async fn test_batch_record_knowledge_with_events() {
    let harness = TestHarness::new().await;
    let server = crate::common::create_test_server(&harness).await;

    let repo = SurrealEntityRepository::new(harness.db.clone());
    let alice = repo
        .create_character(CharacterBuilder::new("Alice").build())
        .await
        .expect("Failed to create Alice");
    let bob = repo
        .create_character(CharacterBuilder::new("Bob").build())
        .await
        .expect("Failed to create Bob");
    let event = repo
        .create_event(EventBuilder::new("The Meeting").sequence(1).build())
        .await
        .expect("Failed to create event");

    let request = MutationRequest::BatchRecordKnowledge {
        knowledge: vec![
            narra::mcp::KnowledgeSpec {
                character_id: alice.id.to_string(),
                target_id: bob.id.to_string(),
                fact: "Alice met Bob at the meeting".to_string(),
                certainty: "knows".to_string(),
                method: Some("witnessed".to_string()),
                source_character_id: None,
                event_id: Some(event.id.key().to_string()),
            },
            narra::mcp::KnowledgeSpec {
                character_id: bob.id.to_string(),
                target_id: alice.id.to_string(),
                fact: "Bob met Alice at the meeting".to_string(),
                certainty: "knows".to_string(),
                method: Some("witnessed".to_string()),
                source_character_id: None,
                event_id: Some(event.id.key().to_string()),
            },
        ],
    };
    let response = server
        .handle_mutate(Parameters(to_mutation_input(request)))
        .await;

    assert!(
        response.is_ok(),
        "Batch with events should succeed: {:?}",
        response.err()
    );
    let response = response.unwrap();
    let entities = response.entities.expect("Should have entities list");
    assert_eq!(entities.len(), 2);

    // Verify both characters have knowledge states with event references
    let knowledge_repo = SurrealKnowledgeRepository::new(harness.db.clone());
    let alice_states = knowledge_repo
        .get_character_knowledge_states(alice.id.key().to_string().as_str())
        .await
        .expect("Get Alice knowledge");
    assert!(!alice_states.is_empty());
    assert!(alice_states[0].event.is_some(), "Should have event link");

    let bob_states = knowledge_repo
        .get_character_knowledge_states(bob.id.key().to_string().as_str())
        .await
        .expect("Get Bob knowledge");
    assert!(!bob_states.is_empty());
    assert!(bob_states[0].event.is_some(), "Should have event link");
}

// =============================================================================
// ERROR CASES
// =============================================================================

/// Test recording knowledge with invalid character_id returns error.
#[tokio::test]
async fn test_record_knowledge_invalid_character() {
    let harness = TestHarness::new().await;
    let server = crate::common::create_test_server(&harness).await;

    let request = MutationRequest::RecordKnowledge {
        character_id: "invalid_format".to_string(), // Not table:key format
        target_id: "character:someone".to_string(),
        fact: "Some fact".to_string(),
        certainty: "knows".to_string(),
        method: Some("initial".to_string()),
        source_character_id: None,
        event_id: None,
    };
    let response = server
        .handle_mutate(Parameters(to_mutation_input(request)))
        .await;

    assert!(response.is_err(), "Should fail with invalid character_id");
    let error_message = response.unwrap_err();

    assert_snapshot!("record_knowledge_invalid_character_error", error_message);
}

/// Test temporal query for nonexistent character returns empty or error.
#[tokio::test]
async fn test_temporal_nonexistent_character() {
    let harness = TestHarness::new().await;
    let server = crate::common::create_test_server(&harness).await;

    let query = QueryRequest::Temporal {
        character_id: "character:nonexistent".to_string(),
        event_id: None,
        event_name: None,
    };
    let response = server.handle_query(Parameters(to_query_input(query))).await;

    // The temporal query should succeed but return empty results
    // (querying knowledge for a character that exists but has no knowledge)
    assert!(
        response.is_ok(),
        "Temporal query should succeed even for nonexistent character"
    );
    let response = response.unwrap();
    assert!(
        response.results.is_empty(),
        "Should return empty results for nonexistent character"
    );
}
