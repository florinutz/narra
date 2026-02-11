//! Handler tests for Session 4 features.
//!
//! Tests for InvestigateContradictions, ListFacts with filters,
//! BaselineArcSnapshots, and Temporal queries.

use narra::mcp::{MutationRequest, QueryRequest};
use narra::models::fact::{self, FactCreate};
use narra::models::knowledge::{create_knowledge_state, KnowledgeStateCreate, LearningMethod};
use narra::models::{EnforcementLevel, FactCategory};
use narra::repository::{
    EntityRepository, KnowledgeRepository, SurrealEntityRepository, SurrealKnowledgeRepository,
};
use rmcp::handler::server::wrapper::Parameters;

use crate::common::{
    builders::{CharacterBuilder, EventBuilder, KnowledgeBuilder},
    harness::TestHarness,
    to_mutation_input, to_query_input,
};

// =============================================================================
// INVESTIGATE CONTRADICTIONS TESTS
// =============================================================================

/// Test investigate_contradictions on a character with no violations.
#[tokio::test]
async fn test_investigate_contradictions_clean() {
    let harness = TestHarness::new().await;
    let repo = SurrealEntityRepository::new(harness.db.clone());

    let alice = repo
        .create_character(CharacterBuilder::new("Alice").build())
        .await
        .expect("Alice");

    let server = crate::common::create_test_server(&harness).await;

    let request = QueryRequest::InvestigateContradictions {
        entity_id: format!("character:{}", alice.id.key()),
        max_depth: 3,
    };

    let response = server
        .handle_query(Parameters(to_query_input(request)))
        .await
        .expect("InvestigateContradictions should succeed");

    assert_eq!(response.results.len(), 1, "Should return one result");
    let content = &response.results[0].content;
    assert!(
        content.contains("No contradictions") || content.contains("0 issue"),
        "Should report no contradictions, got: {}",
        content
    );
}

/// Test investigate_contradictions with depth limit.
#[tokio::test]
async fn test_investigate_contradictions_depth_1() {
    let harness = TestHarness::new().await;
    let repo = SurrealEntityRepository::new(harness.db.clone());

    let alice = repo
        .create_character(CharacterBuilder::new("Alice").build())
        .await
        .expect("Alice");

    let server = crate::common::create_test_server(&harness).await;

    let request = QueryRequest::InvestigateContradictions {
        entity_id: format!("character:{}", alice.id.key()),
        max_depth: 1,
    };

    let response = server
        .handle_query(Parameters(to_query_input(request)))
        .await
        .expect("Should succeed with depth 1");

    assert_eq!(response.results.len(), 1);
    // With depth 1, only immediate connections checked
    assert!(
        response.total >= 1,
        "Should check at least the entity itself"
    );
}

// =============================================================================
// LIST FACTS WITH FILTERS TESTS
// =============================================================================

/// Test listing facts with category filter.
#[tokio::test]
async fn test_list_facts_category_filter() {
    let harness = TestHarness::new().await;

    // Create facts with different categories
    fact::create_fact(
        &harness.db,
        FactCreate {
            title: "Gravity works normally".to_string(),
            description: "Standard gravitational constant".to_string(),
            categories: vec![FactCategory::PhysicsMagic],
            enforcement_level: EnforcementLevel::Strict,
            scope: None,
        },
    )
    .await
    .expect("Physics fact");

    fact::create_fact(
        &harness.db,
        FactCreate {
            title: "Elders are respected".to_string(),
            description: "Social hierarchy based on age".to_string(),
            categories: vec![FactCategory::SocialCultural],
            enforcement_level: EnforcementLevel::Warning,
            scope: None,
        },
    )
    .await
    .expect("Social fact");

    let server = crate::common::create_test_server(&harness).await;

    let request = QueryRequest::ListFacts {
        category: Some("physics_magic".to_string()),
        enforcement_level: None,
        search: None,
        entity_id: None,
        limit: None,
        cursor: None,
    };

    let response = server
        .handle_query(Parameters(to_query_input(request)))
        .await
        .expect("ListFacts with category should succeed");

    // Should return at least the physics fact
    assert!(
        !response.results.is_empty(),
        "Should find physics_magic facts"
    );
    // Verify all returned facts match the category
    for result in &response.results {
        assert!(
            result.content.contains("PhysicsMagic") || result.content.contains("physics"),
            "Filtered result should be physics category, got: {}",
            result.content
        );
    }
}

/// Test listing facts with enforcement level filter.
#[tokio::test]
async fn test_list_facts_enforcement_filter() {
    let harness = TestHarness::new().await;

    fact::create_fact(
        &harness.db,
        FactCreate {
            title: "Hard rule".to_string(),
            description: "This must not be violated".to_string(),
            categories: vec![FactCategory::PhysicsMagic],
            enforcement_level: EnforcementLevel::Strict,
            scope: None,
        },
    )
    .await
    .expect("Strict fact");

    fact::create_fact(
        &harness.db,
        FactCreate {
            title: "Soft guideline".to_string(),
            description: "This is just informational".to_string(),
            categories: vec![FactCategory::SocialCultural],
            enforcement_level: EnforcementLevel::Informational,
            scope: None,
        },
    )
    .await
    .expect("Info fact");

    let server = crate::common::create_test_server(&harness).await;

    let request = QueryRequest::ListFacts {
        category: None,
        enforcement_level: Some("strict".to_string()),
        search: None,
        entity_id: None,
        limit: None,
        cursor: None,
    };

    let response = server
        .handle_query(Parameters(to_query_input(request)))
        .await
        .expect("ListFacts with enforcement should succeed");

    assert!(!response.results.is_empty(), "Should find strict facts");
}

/// Test listing all facts without filters.
#[tokio::test]
async fn test_list_facts_unfiltered() {
    let harness = TestHarness::new().await;

    fact::create_fact(
        &harness.db,
        FactCreate {
            title: "Fact A".to_string(),
            description: "Description A".to_string(),
            categories: vec![],
            enforcement_level: EnforcementLevel::Warning,
            scope: None,
        },
    )
    .await
    .expect("Fact A");

    fact::create_fact(
        &harness.db,
        FactCreate {
            title: "Fact B".to_string(),
            description: "Description B".to_string(),
            categories: vec![],
            enforcement_level: EnforcementLevel::Warning,
            scope: None,
        },
    )
    .await
    .expect("Fact B");

    let server = crate::common::create_test_server(&harness).await;

    let request = QueryRequest::ListFacts {
        category: None,
        enforcement_level: None,
        search: None,
        entity_id: None,
        limit: None,
        cursor: None,
    };

    let response = server
        .handle_query(Parameters(to_query_input(request)))
        .await
        .expect("ListFacts unfiltered should succeed");

    assert_eq!(response.results.len(), 2, "Should return all 2 facts");
}

/// Test ListFacts abbreviates descriptions when more than 5 results.
#[tokio::test]
async fn test_list_facts_abbreviation() {
    let harness = TestHarness::new().await;

    // Create 7 facts with long descriptions to trigger abbreviation (threshold is >5)
    for i in 0..7 {
        fact::create_fact(
            &harness.db,
            FactCreate {
                title: format!("Fact {}", i),
                description: format!(
                    "This is a very detailed description for fact number {}. It has multiple sentences to test abbreviation behavior. The third sentence adds more length.",
                    i
                ),
                categories: vec![],
                enforcement_level: EnforcementLevel::Warning,
                scope: None,
            },
        )
        .await
        .expect("Create fact");
    }

    let server = crate::common::create_test_server(&harness).await;

    let request = QueryRequest::ListFacts {
        category: None,
        enforcement_level: None,
        search: None,
        entity_id: None,
        limit: None,
        cursor: None,
    };

    let response = server
        .handle_query(Parameters(to_query_input(request)))
        .await
        .expect("ListFacts should succeed");

    assert_eq!(response.results.len(), 7, "Should return all 7 facts");

    // With >5 results, descriptions should be abbreviated to first sentence
    for result in &response.results {
        // Abbreviated content should NOT contain "multiple sentences" (second sentence)
        assert!(
            !result.content.contains("multiple sentences"),
            "Content should be abbreviated (truncated to first sentence), got: {}",
            result.content
        );
    }
}

/// Test ListFacts does NOT abbreviate when 5 or fewer results.
#[tokio::test]
async fn test_list_facts_no_abbreviation_small_set() {
    let harness = TestHarness::new().await;

    // Create exactly 3 facts
    for i in 0..3 {
        fact::create_fact(
            &harness.db,
            FactCreate {
                title: format!("Fact {}", i),
                description: format!(
                    "Full description for fact {}. Second sentence with more details.",
                    i
                ),
                categories: vec![],
                enforcement_level: EnforcementLevel::Warning,
                scope: None,
            },
        )
        .await
        .expect("Create fact");
    }

    let server = crate::common::create_test_server(&harness).await;

    let request = QueryRequest::ListFacts {
        category: None,
        enforcement_level: None,
        search: None,
        entity_id: None,
        limit: None,
        cursor: None,
    };

    let response = server
        .handle_query(Parameters(to_query_input(request)))
        .await
        .expect("ListFacts should succeed");

    assert_eq!(response.results.len(), 3);

    // With <=5 results, descriptions should NOT be abbreviated
    for result in &response.results {
        assert!(
            result.content.contains("Second sentence"),
            "Content should NOT be abbreviated for small sets, got: {}",
            result.content
        );
    }
}

// =============================================================================
// BASELINE ARC SNAPSHOTS TESTS
// =============================================================================

/// Test baseline arc snapshots on empty world.
#[tokio::test]
async fn test_baseline_arc_snapshots_empty() {
    let harness = TestHarness::new().await;
    let server = crate::common::create_test_server(&harness).await;

    let request = MutationRequest::BaselineArcSnapshots { entity_type: None };

    let response = server
        .handle_mutate(Parameters(to_mutation_input(request)))
        .await
        .expect("BaselineArcSnapshots should succeed on empty world");

    // Content should indicate 0 created, 0 skipped
    assert!(
        response.entity.content.contains("0 created"),
        "Should report 0 created on empty world, got: {}",
        response.entity.content
    );
}

/// Test baseline arc snapshots with entity type filter.
#[tokio::test]
async fn test_baseline_arc_snapshots_filtered() {
    let harness = TestHarness::new().await;
    let server = crate::common::create_test_server(&harness).await;

    let request = MutationRequest::BaselineArcSnapshots {
        entity_type: Some("character".to_string()),
    };

    let response = server
        .handle_mutate(Parameters(to_mutation_input(request)))
        .await
        .expect("BaselineArcSnapshots with type filter should succeed");

    assert_eq!(response.entity.entity_type, "arc_operation");
}

/// Test baseline arc snapshots rejects invalid entity type.
#[tokio::test]
async fn test_baseline_arc_snapshots_invalid_type() {
    let harness = TestHarness::new().await;
    let server = crate::common::create_test_server(&harness).await;

    let request = MutationRequest::BaselineArcSnapshots {
        entity_type: Some("invalid_type".to_string()),
    };

    let response = server
        .handle_mutate(Parameters(to_mutation_input(request)))
        .await;

    assert!(response.is_err(), "Should reject invalid entity type");
}

// =============================================================================
// TEMPORAL QUERY TESTS
// =============================================================================

/// Test temporal query for character knowledge (all knowledge, no event anchor).
#[tokio::test]
async fn test_temporal_all_knowledge() {
    let harness = TestHarness::new().await;
    let repo = SurrealEntityRepository::new(harness.db.clone());

    let alice = repo
        .create_character(CharacterBuilder::new("Alice").build())
        .await
        .expect("Alice");

    // Create a knowledge fact and record Alice knowing it
    let knowledge_repo = SurrealKnowledgeRepository::new(harness.db.clone());
    let knowledge = knowledge_repo
        .create_knowledge(KnowledgeBuilder::new("The king is dead").build())
        .await
        .expect("Knowledge");

    create_knowledge_state(
        &harness.db,
        &alice.id.key().to_string(),
        &knowledge.id.to_string(),
        KnowledgeStateCreate {
            certainty: narra::models::knowledge::CertaintyLevel::Knows,
            learning_method: LearningMethod::Initial,
            source_character: None,
            event: None,
            premises: None,
            truth_value: None,
        },
    )
    .await
    .expect("Record knowledge");

    let server = crate::common::create_test_server(&harness).await;

    let request = QueryRequest::Temporal {
        character_id: format!("character:{}", alice.id.key()),
        event_id: None,
        event_name: None,
    };

    let response = server
        .handle_query(Parameters(to_query_input(request)))
        .await
        .expect("Temporal query should succeed");

    assert!(
        !response.results.is_empty(),
        "Should return knowledge states"
    );
}

/// Test temporal query with event anchor.
#[tokio::test]
async fn test_temporal_at_event() {
    let harness = TestHarness::new().await;
    let repo = SurrealEntityRepository::new(harness.db.clone());

    let alice = repo
        .create_character(CharacterBuilder::new("Alice").build())
        .await
        .expect("Alice");

    let event = repo
        .create_event(EventBuilder::new("The Revelation").sequence(1).build())
        .await
        .expect("Event");

    // Create knowledge and record Alice learning it at the event
    let knowledge_repo = SurrealKnowledgeRepository::new(harness.db.clone());
    let knowledge = knowledge_repo
        .create_knowledge(KnowledgeBuilder::new("Bob is a spy").build())
        .await
        .expect("Knowledge");

    create_knowledge_state(
        &harness.db,
        &alice.id.key().to_string(),
        &knowledge.id.to_string(),
        KnowledgeStateCreate {
            certainty: narra::models::knowledge::CertaintyLevel::Suspects,
            learning_method: LearningMethod::Overheard,
            source_character: None,
            event: Some(event.id.key().to_string()),
            premises: None,
            truth_value: None,
        },
    )
    .await
    .expect("Record knowledge at event");

    let server = crate::common::create_test_server(&harness).await;

    let request = QueryRequest::Temporal {
        character_id: format!("character:{}", alice.id.key()),
        event_id: Some(format!("event:{}", event.id.key())),
        event_name: None,
    };

    let response = server
        .handle_query(Parameters(to_query_input(request)))
        .await
        .expect("Temporal query at event should succeed");

    assert!(
        !response.results.is_empty(),
        "Should return knowledge states at event"
    );
}

/// Test temporal query for character with no knowledge.
#[tokio::test]
async fn test_temporal_no_knowledge() {
    let harness = TestHarness::new().await;
    let repo = SurrealEntityRepository::new(harness.db.clone());

    let alice = repo
        .create_character(CharacterBuilder::new("Alice").build())
        .await
        .expect("Alice");

    let server = crate::common::create_test_server(&harness).await;

    let request = QueryRequest::Temporal {
        character_id: format!("character:{}", alice.id.key()),
        event_id: None,
        event_name: None,
    };

    let response = server
        .handle_query(Parameters(to_query_input(request)))
        .await
        .expect("Temporal query should succeed even with no knowledge");

    // May return empty results or one result with "no knowledge" content
    assert!(
        response.results.is_empty() || response.results.len() == 1,
        "Should return zero or one result"
    );
}
