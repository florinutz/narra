//! Integration tests for Composite Intelligence operations.
//!
//! Tests verify:
//! - SituationReport: aggregated irony, conflicts, tensions, themes
//! - CharacterDossier: centrality, influence, knowledge, perceptions
//! - ScenePlanning: pairwise dynamics, irony, tensions, facts
//!
//! Uses raw DB queries to set up test data without embedding dependencies.

mod common;

use std::sync::Arc;

use common::{harness::TestHarness, to_query_input};
use narra::embedding::NoopEmbeddingService;
use narra::mcp::{NarraServer, QueryRequest};
use narra::models::character::{create_character, CharacterCreate};
use narra::models::perception::{create_perception_pair, PerceptionCreate};
use narra::services::CompositeIntelligenceService;
use narra::session::SessionStateManager;
use rmcp::handler::server::wrapper::Parameters;

/// Helper to create NarraServer with noop embeddings.
async fn create_test_server(harness: &TestHarness) -> NarraServer {
    let session_path = harness.temp_path().join("session.json");
    let session_manager = Arc::new(
        SessionStateManager::load_or_create(&session_path)
            .expect("Failed to create session manager"),
    );
    NarraServer::new(
        harness.db.clone(),
        session_manager,
        Arc::new(NoopEmbeddingService::new()),
    )
    .await
}

// =============================================================================
// SITUATION REPORT
// =============================================================================

/// Test SituationReport with an empty world returns structured output with suggestions.
#[tokio::test]
async fn test_situation_report_empty_world() {
    let harness = TestHarness::new().await;
    let service = CompositeIntelligenceService::new(harness.db.clone());

    let report = service
        .situation_report()
        .await
        .expect("Situation report should succeed on empty DB");

    assert!(report.irony_highlights.is_empty());
    assert!(report.knowledge_conflicts.is_empty());
    assert!(report.high_tension_pairs.is_empty());
    assert!(
        !report.suggestions.is_empty(),
        "Should have at least one suggestion"
    );
    assert!(
        report.suggestions[0].contains("stable"),
        "Empty world suggestion should mention stability"
    );
}

/// Test SituationReport with data populates all sections.
#[tokio::test]
async fn test_situation_report_structured_output() {
    let harness = TestHarness::new().await;

    // Create characters
    let alice = create_character(
        &harness.db,
        CharacterCreate {
            name: "Alice".into(),
            roles: vec!["protagonist".into()],
            ..Default::default()
        },
    )
    .await
    .unwrap();

    let bob = create_character(
        &harness.db,
        CharacterCreate {
            name: "Bob".into(),
            roles: vec!["antagonist".into()],
            ..Default::default()
        },
    )
    .await
    .unwrap();

    let alice_key = alice.id.key().to_string();
    let bob_key = bob.id.key().to_string();

    // Create perception with high tension
    create_perception_pair(
        &harness.db,
        &alice_key,
        &bob_key,
        PerceptionCreate {
            rel_types: vec!["rival".to_string()],
            subtype: None,
            feelings: Some("hostile".to_string()),
            perception: Some("untrustworthy".to_string()),
            tension_level: Some(9),
            history_notes: None,
        },
        PerceptionCreate {
            rel_types: vec!["ally".to_string()],
            subtype: None,
            feelings: Some("trusting".to_string()),
            perception: Some("reliable".to_string()),
            tension_level: Some(2),
            history_notes: None,
        },
    )
    .await
    .unwrap();

    // Create a knowledge asymmetry: Alice knows a secret about Bob, Bob doesn't know
    harness
        .db
        .query(format!(
            "RELATE character:{}->knows->character:{} SET \
             fact = 'Bobs true identity', \
             certainty = 'knows', \
             learning_method = 'witnessed', \
             learned_at = time::now(), \
             created_at = time::now(), \
             updated_at = time::now()",
            alice_key, bob_key
        ))
        .await
        .unwrap();

    let service = CompositeIntelligenceService::new(harness.db.clone());
    let report = service.situation_report().await.expect("Should succeed");

    // High-tension pair should be found (alice->bob tension 9 >= 7)
    assert!(
        !report.high_tension_pairs.is_empty(),
        "Should find high-tension pair"
    );
    assert!(
        report.high_tension_pairs[0].tension_level >= 7,
        "Tension should be >= 7"
    );

    // Should have suggestions
    assert!(!report.suggestions.is_empty());
}

/// Test SituationReport via MCP.
#[tokio::test]
async fn test_situation_report_mcp() {
    let harness = TestHarness::new().await;
    let server = create_test_server(&harness).await;

    let request = QueryRequest::SituationReport { detail_level: None };
    let response = server
        .handle_query(Parameters(to_query_input(request)))
        .await
        .expect("MCP situation_report should succeed");

    assert_eq!(response.results.len(), 1);
    assert_eq!(response.results[0].entity_type, "report");
    assert!(response.results[0].name.contains("Situation"));
}

// =============================================================================
// CHARACTER DOSSIER
// =============================================================================

/// Test CharacterDossier for a character with relationships and knowledge.
#[tokio::test]
async fn test_character_dossier_with_knowledge() {
    let harness = TestHarness::new().await;

    let alice = create_character(
        &harness.db,
        CharacterCreate {
            name: "Alice".into(),
            roles: vec!["warrior".into(), "protagonist".into()],
            ..Default::default()
        },
    )
    .await
    .unwrap();

    let bob = create_character(
        &harness.db,
        CharacterCreate {
            name: "Bob".into(),
            roles: vec!["healer".into()],
            ..Default::default()
        },
    )
    .await
    .unwrap();

    let alice_key = alice.id.key().to_string();
    let bob_key = bob.id.key().to_string();

    // Create relationship
    harness
        .db
        .query(format!(
            "RELATE character:{}->relates_to->character:{} SET \
             rel_type = 'ally', created_at = time::now(), updated_at = time::now()",
            alice_key, bob_key
        ))
        .await
        .unwrap();

    // Bob perceives Alice with tension
    create_perception_pair(
        &harness.db,
        &bob_key,
        &alice_key,
        PerceptionCreate {
            rel_types: vec!["leader".to_string()],
            subtype: None,
            feelings: Some("respectful but wary".to_string()),
            perception: Some("strong but reckless".to_string()),
            tension_level: Some(6),
            history_notes: None,
        },
        PerceptionCreate {
            rel_types: vec!["team member".to_string()],
            subtype: None,
            feelings: Some("trusts".to_string()),
            perception: Some("reliable healer".to_string()),
            tension_level: Some(1),
            history_notes: None,
        },
    )
    .await
    .unwrap();

    let service = CompositeIntelligenceService::new(harness.db.clone());
    let dossier = service
        .character_dossier(&format!("character:{}", alice_key))
        .await
        .expect("Dossier should succeed");

    assert_eq!(dossier.name, "Alice");
    assert!(dossier.roles.contains(&"warrior".to_string()));
    assert!(dossier.roles.contains(&"protagonist".to_string()));

    // Should have perception data (Bob perceives Alice)
    // Note: perception pair creates edges in both directions
    assert!(
        !dossier.key_perceptions.is_empty(),
        "Should have perceptions"
    );
}

/// Test CharacterDossier for nonexistent character returns error.
#[tokio::test]
async fn test_character_dossier_not_found() {
    let harness = TestHarness::new().await;
    let service = CompositeIntelligenceService::new(harness.db.clone());

    let result = service.character_dossier("character:nonexistent").await;
    assert!(result.is_err(), "Should fail for nonexistent character");
}

/// Test CharacterDossier via MCP.
#[tokio::test]
async fn test_character_dossier_mcp() {
    let harness = TestHarness::new().await;
    let server = create_test_server(&harness).await;

    // Create a character first
    let alice = create_character(
        &harness.db,
        CharacterCreate {
            name: "Alice".into(),
            ..Default::default()
        },
    )
    .await
    .unwrap();

    let request = QueryRequest::CharacterDossier {
        character_id: alice.id.to_string(),
        detail_level: None,
    };
    let response = server
        .handle_query(Parameters(to_query_input(request)))
        .await
        .expect("MCP character_dossier should succeed");

    assert_eq!(response.results.len(), 1);
    assert!(response.results[0].name.contains("Alice"));
}

// =============================================================================
// SCENE PLANNING
// =============================================================================

/// Test ScenePlanning with two characters.
#[tokio::test]
async fn test_scene_prep_two_characters() {
    let harness = TestHarness::new().await;

    let alice = create_character(
        &harness.db,
        CharacterCreate {
            name: "Alice".into(),
            ..Default::default()
        },
    )
    .await
    .unwrap();

    let bob = create_character(
        &harness.db,
        CharacterCreate {
            name: "Bob".into(),
            ..Default::default()
        },
    )
    .await
    .unwrap();

    let alice_key = alice.id.key().to_string();
    let bob_key = bob.id.key().to_string();

    // High-tension perception
    create_perception_pair(
        &harness.db,
        &alice_key,
        &bob_key,
        PerceptionCreate {
            rel_types: vec!["enemy".to_string()],
            subtype: None,
            feelings: Some("hatred".to_string()),
            perception: None,
            tension_level: Some(10),
            history_notes: None,
        },
        PerceptionCreate {
            rel_types: vec!["friend".to_string()],
            subtype: None,
            feelings: Some("love".to_string()),
            perception: None,
            tension_level: Some(1),
            history_notes: None,
        },
    )
    .await
    .unwrap();

    let service = CompositeIntelligenceService::new(harness.db.clone());
    let plan = service
        .scene_prep(&[
            format!("character:{}", alice_key),
            format!("character:{}", bob_key),
        ])
        .await
        .expect("Scene prep should succeed");

    assert_eq!(plan.characters.len(), 2);
    assert_eq!(plan.pair_dynamics.len(), 1, "2 characters = 1 pair");

    // Should find the high-tension pair
    assert!(
        plan.highest_tension_pair.is_some(),
        "Should detect tension pair"
    );
    let (_, _, tension) = plan.highest_tension_pair.unwrap();
    assert!(tension >= 7, "Tension should be high");
}

/// Test ScenePlanning with empty world doesn't panic.
#[tokio::test]
async fn test_scene_prep_empty_world() {
    let harness = TestHarness::new().await;
    let service = CompositeIntelligenceService::new(harness.db.clone());

    // Characters don't exist in DB but that's OK — service should handle gracefully
    let plan = service
        .scene_prep(&[
            "character:ghost_a".to_string(),
            "character:ghost_b".to_string(),
        ])
        .await
        .expect("Scene prep should not panic on missing characters");

    assert_eq!(plan.pair_dynamics.len(), 1);
    assert_eq!(plan.total_irony_opportunities, 0);
}

/// Test ScenePlanning via MCP requires at least 2 characters.
#[tokio::test]
async fn test_scene_planning_mcp_validation() {
    let harness = TestHarness::new().await;
    let server = create_test_server(&harness).await;

    // Too few characters
    let request = QueryRequest::ScenePlanning {
        character_ids: vec!["alice".to_string()],
        detail_level: None,
    };
    let result = server
        .handle_query(Parameters(to_query_input(request)))
        .await;
    assert!(result.is_err(), "Should reject < 2 characters");
}

/// Test ScenePlanning via MCP with valid input.
#[tokio::test]
async fn test_scene_planning_mcp() {
    let harness = TestHarness::new().await;
    let server = create_test_server(&harness).await;

    create_character(
        &harness.db,
        CharacterCreate {
            name: "Alice".into(),
            ..Default::default()
        },
    )
    .await
    .unwrap();

    create_character(
        &harness.db,
        CharacterCreate {
            name: "Bob".into(),
            ..Default::default()
        },
    )
    .await
    .unwrap();

    let request = QueryRequest::ScenePlanning {
        character_ids: vec!["character:alice".to_string(), "character:bob".to_string()],
        detail_level: None,
    };
    let response = server
        .handle_query(Parameters(to_query_input(request)))
        .await
        .expect("Scene planning should succeed");

    assert_eq!(response.results.len(), 1);
    assert!(response.results[0].name.contains("Scene Plan"));
}

// =============================================================================
// DETAIL LEVEL TESTS
// =============================================================================

/// Test SituationReport with detail_level parameter accepts all modes.
#[tokio::test]
async fn test_situation_report_detail_levels() {
    let harness = TestHarness::new().await;
    let server = create_test_server(&harness).await;

    // Create enough data for truncation to matter
    let names = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace"];
    let mut keys = Vec::new();
    for name in &names {
        let c = create_character(
            &harness.db,
            CharacterCreate {
                name: (*name).into(),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        keys.push(c.id.key().to_string());
    }

    // Create high-tension perceptions between many pairs
    for i in 0..keys.len() - 1 {
        create_perception_pair(
            &harness.db,
            &keys[i],
            &keys[i + 1],
            PerceptionCreate {
                rel_types: vec!["rival".to_string()],
                subtype: None,
                feelings: Some("hostile".to_string()),
                perception: None,
                tension_level: Some(8),
                history_notes: None,
            },
            PerceptionCreate {
                rel_types: vec!["ally".to_string()],
                subtype: None,
                feelings: Some("neutral".to_string()),
                perception: None,
                tension_level: Some(3),
                history_notes: None,
            },
        )
        .await
        .unwrap();
    }

    // All three detail levels should succeed
    for level in &["summary", "standard", "full"] {
        let request = QueryRequest::SituationReport {
            detail_level: Some(level.to_string()),
        };
        let response = server
            .handle_query(Parameters(to_query_input(request)))
            .await
            .unwrap_or_else(|_| panic!("SituationReport detail_level='{}' should succeed", level));

        assert_eq!(response.results.len(), 1);
        assert_eq!(response.results[0].entity_type, "report");
    }

    // Summary with enough data should contain truncation hint
    let summary_request = QueryRequest::SituationReport {
        detail_level: Some("summary".to_string()),
    };
    let summary_response = server
        .handle_query(Parameters(to_query_input(summary_request)))
        .await
        .expect("Summary should succeed");

    let summary_content = &summary_response.results[0].content;
    // With 6 high-tension pairs (>5 summary limit), should see truncation hint
    if summary_content.contains("High Tension Pairs") {
        assert!(
            summary_content.contains("not shown") || summary_content.contains("detail_level"),
            "Summary should indicate truncation for tension pairs when > 5 pairs"
        );
    }
}

/// Test CharacterDossier with detail_level="summary" limits relationships shown.
#[tokio::test]
async fn test_character_dossier_detail_level_summary() {
    let harness = TestHarness::new().await;
    let server = create_test_server(&harness).await;

    let alice = create_character(
        &harness.db,
        CharacterCreate {
            name: "Alice".into(),
            ..Default::default()
        },
    )
    .await
    .unwrap();

    // Create many relationships so truncation kicks in (summary=3, standard=10)
    for name in ["Bob", "Carol", "Dave", "Eve", "Frank"] {
        let other = create_character(
            &harness.db,
            CharacterCreate {
                name: name.into(),
                ..Default::default()
            },
        )
        .await
        .unwrap();

        let alice_key = alice.id.key().to_string();
        let other_key = other.id.key().to_string();
        create_perception_pair(
            &harness.db,
            &alice_key,
            &other_key,
            PerceptionCreate {
                rel_types: vec!["ally".to_string()],
                subtype: None,
                feelings: Some("friendly".to_string()),
                perception: None,
                tension_level: Some(3),
                history_notes: None,
            },
            PerceptionCreate {
                rel_types: vec!["ally".to_string()],
                subtype: None,
                feelings: Some("friendly".to_string()),
                perception: None,
                tension_level: Some(2),
                history_notes: None,
            },
        )
        .await
        .unwrap();
    }

    // Summary mode
    let summary_request = QueryRequest::CharacterDossier {
        character_id: alice.id.to_string(),
        detail_level: Some("summary".to_string()),
    };
    let summary_response = server
        .handle_query(Parameters(to_query_input(summary_request)))
        .await
        .expect("Summary dossier should succeed");

    // Full mode
    let full_request = QueryRequest::CharacterDossier {
        character_id: alice.id.to_string(),
        detail_level: Some("full".to_string()),
    };
    let full_response = server
        .handle_query(Parameters(to_query_input(full_request)))
        .await
        .expect("Full dossier should succeed");

    // Summary should be shorter
    let summary_len = summary_response.results[0].content.len();
    let full_len = full_response.results[0].content.len();
    assert!(
        summary_len <= full_len,
        "Summary ({} chars) should be <= full ({} chars)",
        summary_len,
        full_len
    );

    // Summary with many relationships should contain truncation hint
    let summary_content = &summary_response.results[0].content;
    if summary_content.contains("Relationships") {
        // If there are relationships shown, summary should hint at more
        assert!(
            summary_content.contains("not shown") || summary_content.contains("Relationships ("),
            "Summary should indicate truncated relationships"
        );
    }
}

/// Test ScenePlanning with detail_level="summary" limits pair dynamics shown.
#[tokio::test]
async fn test_scene_planning_detail_level_summary() {
    let harness = TestHarness::new().await;
    let server = create_test_server(&harness).await;

    let alice = create_character(
        &harness.db,
        CharacterCreate {
            name: "Alice".into(),
            ..Default::default()
        },
    )
    .await
    .unwrap();

    let bob = create_character(
        &harness.db,
        CharacterCreate {
            name: "Bob".into(),
            ..Default::default()
        },
    )
    .await
    .unwrap();

    // Summary mode
    let summary_request = QueryRequest::ScenePlanning {
        character_ids: vec![alice.id.to_string(), bob.id.to_string()],
        detail_level: Some("summary".to_string()),
    };
    let summary_response = server
        .handle_query(Parameters(to_query_input(summary_request)))
        .await
        .expect("Summary scene planning should succeed");

    // Full mode
    let full_request = QueryRequest::ScenePlanning {
        character_ids: vec![alice.id.to_string(), bob.id.to_string()],
        detail_level: Some("full".to_string()),
    };
    let full_response = server
        .handle_query(Parameters(to_query_input(full_request)))
        .await
        .expect("Full scene planning should succeed");

    let summary_len = summary_response.results[0].content.len();
    let full_len = full_response.results[0].content.len();
    assert!(
        summary_len <= full_len,
        "Summary ({} chars) should be <= full ({} chars)",
        summary_len,
        full_len
    );
}

// =============================================================================
// TRUNCATED FIELD & TOKEN BUDGET TESTS
// =============================================================================

/// Test that QueryResponse.truncated is None when response fits budget.
#[tokio::test]
async fn test_truncated_field_none_when_fits_budget() {
    let harness = TestHarness::new().await;
    let server = create_test_server(&harness).await;

    // Small query — should fit default budget
    let request = QueryRequest::SituationReport {
        detail_level: Some("summary".to_string()),
    };
    let response = server
        .handle_query(Parameters(to_query_input(request)))
        .await
        .expect("Should succeed");

    // Empty world situation report is small, should not be truncated
    assert!(
        response.truncated.is_none(),
        "Small response should not be truncated"
    );
}

/// Test that hints are truncated to max 3.
#[tokio::test]
async fn test_hints_max_three() {
    let harness = TestHarness::new().await;
    let server = create_test_server(&harness).await;

    let alice = create_character(
        &harness.db,
        CharacterCreate {
            name: "Alice".into(),
            ..Default::default()
        },
    )
    .await
    .unwrap();

    // Lookup — generates hints
    let request = QueryRequest::Lookup {
        entity_id: alice.id.to_string(),
        detail_level: Some(narra::mcp::DetailLevel::Standard),
    };
    let response = server
        .handle_query(Parameters(to_query_input(request)))
        .await
        .expect("Lookup should succeed");

    assert!(
        response.hints.len() <= 3,
        "Hints should be capped at 3, got {}",
        response.hints.len()
    );
}

/// Test that search results also have hints capped at 3.
#[tokio::test]
async fn test_search_hints_max_three() {
    let harness = TestHarness::new().await;
    let server = create_test_server(&harness).await;

    create_character(
        &harness.db,
        CharacterCreate {
            name: "Alice".into(),
            ..Default::default()
        },
    )
    .await
    .unwrap();

    let request = QueryRequest::Search {
        query: "Alice".to_string(),
        entity_types: None,
        limit: None,
        cursor: None,
    };
    let response = server
        .handle_query(Parameters(to_query_input(request)))
        .await
        .expect("Search should succeed");

    assert!(
        response.hints.len() <= 3,
        "Search hints should be capped at 3, got {}",
        response.hints.len()
    );
}

// =============================================================================
// ENRICHMENT TESTS — narrative tensions and character roles in composites
// =============================================================================

use common::builders::{CharacterBuilder, EventBuilder, KnowledgeBuilder};
use narra::models::knowledge::{
    create_knowledge_state, CertaintyLevel, KnowledgeStateCreate, LearningMethod,
};
use narra::models::perception::create_perception;
use narra::models::relationship::{create_relationship, RelationshipCreate};
use narra::repository::{
    EntityRepository, KnowledgeRepository, SurrealEntityRepository, SurrealKnowledgeRepository,
};

/// Test ScenePlan includes narrative tensions when characters have contradictory knowledge.
#[tokio::test]
async fn test_scene_prep_with_narrative_tensions() {
    let harness = TestHarness::new().await;
    let entity_repo = SurrealEntityRepository::new(harness.db.clone());
    let knowledge_repo = SurrealKnowledgeRepository::new(harness.db.clone());

    let alice = entity_repo
        .create_character(CharacterBuilder::new("Alice").build())
        .await
        .unwrap();
    let bob = entity_repo
        .create_character(CharacterBuilder::new("Bob").build())
        .await
        .unwrap();
    let alice_key = alice.id.key().to_string();
    let bob_key = bob.id.key().to_string();

    let event = entity_repo
        .create_event(EventBuilder::new("Discovery").sequence(1).build())
        .await
        .unwrap();
    let event_key = event.id.key().to_string();

    // Create a knowledge entity (target for knowledge states)
    let fact = knowledge_repo
        .create_knowledge(
            KnowledgeBuilder::new("The butler did it")
                .for_character(&alice_key)
                .build(),
        )
        .await
        .unwrap();
    let fact_id = format!("knowledge:{}", fact.id.key());

    // Alice knows the fact, Bob believes wrongly
    create_knowledge_state(
        &harness.db,
        &alice_key,
        &fact_id,
        KnowledgeStateCreate {
            certainty: CertaintyLevel::Knows,
            learning_method: LearningMethod::Initial,
            event: Some(event_key.clone()),
            truth_value: Some("true".to_string()),
            ..Default::default()
        },
    )
    .await
    .unwrap();

    create_knowledge_state(
        &harness.db,
        &bob_key,
        &fact_id,
        KnowledgeStateCreate {
            certainty: CertaintyLevel::BelievesWrongly,
            learning_method: LearningMethod::Initial,
            event: Some(event_key.clone()),
            truth_value: Some("false".to_string()),
            ..Default::default()
        },
    )
    .await
    .unwrap();

    let service = CompositeIntelligenceService::new(harness.db.clone());
    let plan = service
        .scene_prep(&[
            format!("character:{}", alice_key),
            format!("character:{}", bob_key),
        ])
        .await
        .expect("Scene prep should succeed");

    // Should detect contradictory knowledge tension
    assert!(
        !plan.narrative_tensions.is_empty(),
        "Should detect narrative tensions from contradictory knowledge"
    );

    let has_contradiction = plan
        .narrative_tensions
        .iter()
        .any(|t| t.tension_type == "contradictory_knowledge");
    assert!(
        has_contradiction,
        "Should have a contradictory_knowledge tension type, got: {:?}",
        plan.narrative_tensions
            .iter()
            .map(|t| &t.tension_type)
            .collect::<Vec<_>>()
    );
}

/// Test ScenePlan includes character roles when graph has enough structure.
#[tokio::test]
async fn test_scene_prep_with_character_roles() {
    let harness = TestHarness::new().await;
    let entity_repo = SurrealEntityRepository::new(harness.db.clone());

    // Create a hub character with many connections
    let hub = entity_repo
        .create_character(CharacterBuilder::new("Hub").build())
        .await
        .unwrap();
    let hub_key = hub.id.key().to_string();

    let mut spoke_keys = vec![];
    for i in 0..5 {
        let spoke = entity_repo
            .create_character(CharacterBuilder::new(&format!("Spoke{}", i)).build())
            .await
            .unwrap();
        let spoke_key = spoke.id.key().to_string();

        // Both relationship AND perception (social_hub needs perception edges)
        create_relationship(
            &harness.db,
            RelationshipCreate {
                from_character_id: hub_key.clone(),
                to_character_id: spoke_key.clone(),
                rel_type: "ally".to_string(),
                subtype: None,
                label: None,
            },
        )
        .await
        .unwrap();

        create_perception(
            &harness.db,
            &hub_key,
            &spoke_key,
            PerceptionCreate {
                rel_types: vec!["ally".to_string()],
                subtype: None,
                feelings: None,
                perception: None,
                tension_level: None,
                history_notes: None,
            },
        )
        .await
        .unwrap();

        spoke_keys.push(spoke_key);
    }

    let service = CompositeIntelligenceService::new(harness.db.clone());
    let plan = service
        .scene_prep(&[
            format!("character:{}", hub_key),
            format!("character:{}", spoke_keys[0]),
            format!("character:{}", spoke_keys[1]),
        ])
        .await
        .expect("Scene prep should succeed");

    // The hub character should be detected as having a role
    let hub_role = plan
        .character_roles
        .iter()
        .find(|r| r.character_id == format!("character:{}", hub_key));
    assert!(
        hub_role.is_some(),
        "Hub character should have an inferred role, got: {:?}",
        plan.character_roles
    );

    if let Some(role) = hub_role {
        let is_hub_like = role.primary_role == "social_hub"
            || role.primary_role == "connector"
            || role.secondary_roles.contains(&"social_hub".to_string())
            || role.secondary_roles.contains(&"connector".to_string());
        assert!(
            is_hub_like,
            "Hub with 5 connections should be social_hub or connector, got: {} (secondary: {:?})",
            role.primary_role, role.secondary_roles
        );
    }
}

/// Test CharacterDossier includes narrative tensions for the target character.
#[tokio::test]
async fn test_character_dossier_with_narrative_tensions() {
    let harness = TestHarness::new().await;

    let alice = create_character(
        &harness.db,
        CharacterCreate {
            name: "Alice".into(),
            roles: vec!["protagonist".to_string()],
            ..Default::default()
        },
    )
    .await
    .unwrap();
    let bob = create_character(
        &harness.db,
        CharacterCreate {
            name: "Bob".into(),
            ..Default::default()
        },
    )
    .await
    .unwrap();

    let alice_key = alice.id.key().to_string();
    let bob_key = bob.id.key().to_string();

    // Create conflicting loyalties: Alice is allied with Bob, but also rivals
    create_relationship(
        &harness.db,
        RelationshipCreate {
            from_character_id: alice_key.clone(),
            to_character_id: bob_key.clone(),
            rel_type: "ally".to_string(),
            subtype: None,
            label: Some("allied".to_string()),
        },
    )
    .await
    .unwrap();

    // Add a high-tension perception (creates edge-based tension)
    create_perception_pair(
        &harness.db,
        &alice_key,
        &bob_key,
        PerceptionCreate {
            rel_types: vec!["rival".to_string()],
            subtype: None,
            feelings: Some("distrustful".to_string()),
            perception: None,
            tension_level: Some(9),
            history_notes: None,
        },
        PerceptionCreate {
            rel_types: vec!["rival".to_string()],
            subtype: None,
            feelings: Some("hostile".to_string()),
            perception: None,
            tension_level: Some(8),
            history_notes: None,
        },
    )
    .await
    .unwrap();

    let service = CompositeIntelligenceService::new(harness.db.clone());
    let dossier = service
        .character_dossier(&format!("character:{}", alice_key))
        .await
        .expect("Dossier should succeed");

    // Narrative tensions should be filtered to those involving Alice
    for t in &dossier.narrative_tensions {
        let alice_full = format!("character:{}", alice_key);
        assert!(
            t.character_a_id == alice_full || t.character_b_id == alice_full,
            "Tension should involve Alice: {:?}",
            t
        );
    }

    // The high-tension perception (>=7) should generate an edge tension
    let has_edge_tension = dossier
        .narrative_tensions
        .iter()
        .any(|t| t.tension_type == "high_edge_tension");
    assert!(
        has_edge_tension,
        "Should detect high_edge_tension from tension_level=9, got: {:?}",
        dossier
            .narrative_tensions
            .iter()
            .map(|t| &t.tension_type)
            .collect::<Vec<_>>()
    );
}

/// Test CharacterDossier annotation fields are None when services unavailable (default state in tests).
#[tokio::test]
async fn test_character_dossier_annotation_fields_none() {
    let harness = TestHarness::new().await;

    let alice = create_character(
        &harness.db,
        CharacterCreate {
            name: "Alice".into(),
            ..Default::default()
        },
    )
    .await
    .unwrap();

    let service = CompositeIntelligenceService::new(harness.db.clone());
    let dossier = service
        .character_dossier(&alice.id.to_string())
        .await
        .expect("Dossier should succeed");

    // Annotation fields should be None (populated at handler level, not in service)
    assert!(
        dossier.emotion_profile.is_none(),
        "emotion_profile should be None when populated at handler level"
    );
    assert!(
        dossier.theme_tags.is_none(),
        "theme_tags should be None when populated at handler level"
    );
}

/// Test ScenePlan annotation fields are None by default.
#[tokio::test]
async fn test_scene_plan_annotation_fields_none() {
    let harness = TestHarness::new().await;

    let alice = create_character(
        &harness.db,
        CharacterCreate {
            name: "Alice".into(),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    let bob = create_character(
        &harness.db,
        CharacterCreate {
            name: "Bob".into(),
            ..Default::default()
        },
    )
    .await
    .unwrap();

    let service = CompositeIntelligenceService::new(harness.db.clone());
    let plan = service
        .scene_prep(&[alice.id.to_string(), bob.id.to_string()])
        .await
        .expect("Scene prep should succeed");

    assert!(
        plan.scene_themes.is_none(),
        "scene_themes should be None when populated at handler level"
    );
}
