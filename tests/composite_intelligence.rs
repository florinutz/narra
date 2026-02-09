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

    let request = QueryRequest::SituationReport;
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
    };
    let response = server
        .handle_query(Parameters(to_query_input(request)))
        .await
        .expect("Scene planning should succeed");

    assert_eq!(response.results.len(), 1);
    assert!(response.results[0].name.contains("Scene Plan"));
}
