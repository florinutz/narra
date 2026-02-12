//! Integration tests for TensionService and RoleInferenceService.
//!
//! Tests both services against real SurrealDB with crafted narrative topologies.

mod common;

use common::builders::{CharacterBuilder, EventBuilder, KnowledgeBuilder};
use common::harness::TestHarness;
use narra::models::knowledge::{
    create_knowledge_state, CertaintyLevel, KnowledgeStateCreate, LearningMethod,
};
use narra::models::perception::{create_perception, create_perception_pair, PerceptionCreate};
use narra::models::relationship::{create_relationship, RelationshipCreate};
use narra::repository::{
    EntityRepository, KnowledgeRepository, SurrealEntityRepository, SurrealKnowledgeRepository,
};
use narra::services::{RoleInferenceService, TensionService};

/// Helper: create a character and return (full_id, key)
async fn create_char(harness: &TestHarness, name: &str) -> (String, String) {
    let repo = SurrealEntityRepository::new(harness.db.clone());
    let c = repo
        .create_character(CharacterBuilder::new(name).build())
        .await
        .expect("create character");
    let key = c.id.key().to_string();
    let full = format!("character:{}", key);
    (full, key)
}

/// Helper: create an event and return its key
async fn create_evt(harness: &TestHarness, title: &str, seq: i64) -> String {
    let repo = SurrealEntityRepository::new(harness.db.clone());
    let e = repo
        .create_event(EventBuilder::new(title).sequence(seq).build())
        .await
        .expect("create event");
    e.id.key().to_string()
}

/// Helper: create knowledge state with all required fields
async fn add_knowledge_state(
    harness: &TestHarness,
    char_key: &str,
    fact_id: &str,
    certainty: CertaintyLevel,
    event_key: &str,
) {
    let mut state = KnowledgeStateCreate {
        certainty,
        learning_method: LearningMethod::Initial,
        event: Some(event_key.to_string()),
        ..Default::default()
    };
    if certainty == CertaintyLevel::BelievesWrongly {
        state.truth_value = Some("false".to_string());
    }
    create_knowledge_state(&harness.db, char_key, fact_id, state)
        .await
        .unwrap();
}

// ---------------------------------------------------------------------------
// TensionService integration tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_tension_empty_world() {
    let harness = TestHarness::new().await;
    let service = TensionService::new(harness.db.clone());
    let report = service.detect_tensions(20, 0.0).await.unwrap();

    assert_eq!(report.total_count, 0);
    assert!(report.tensions.is_empty());
}

#[tokio::test]
async fn test_tension_contradictory_knowledge() {
    let harness = TestHarness::new().await;
    let knowledge_repo = SurrealKnowledgeRepository::new(harness.db.clone());

    let (_alice_id, alice_key) = create_char(&harness, "Alice").await;
    let (_bob_id, bob_key) = create_char(&harness, "Bob").await;
    let event_key = create_evt(&harness, "Discovery", 1).await;

    let fact = knowledge_repo
        .create_knowledge(
            KnowledgeBuilder::new("The butler did it")
                .for_character(&alice_key)
                .build(),
        )
        .await
        .unwrap();
    let fact_id = format!("knowledge:{}", fact.id.key());

    add_knowledge_state(
        &harness,
        &alice_key,
        &fact_id,
        CertaintyLevel::Knows,
        &event_key,
    )
    .await;
    add_knowledge_state(
        &harness,
        &bob_key,
        &fact_id,
        CertaintyLevel::BelievesWrongly,
        &event_key,
    )
    .await;

    let service = TensionService::new(harness.db.clone());
    let report = service.detect_tensions(20, 0.0).await.unwrap();

    assert!(
        report.total_count > 0,
        "Expected at least one tension from contradictory knowledge, got 0"
    );

    let has_contradictory = report.tensions.iter().any(|t| {
        t.tension_type == "contradictory_knowledge"
            && ((t.character_a_name == "Alice" && t.character_b_name == "Bob")
                || (t.character_a_name == "Bob" && t.character_b_name == "Alice"))
    });
    assert!(
        has_contradictory,
        "Expected contradictory_knowledge tension between Alice and Bob. Got: {:?}",
        report
            .tensions
            .iter()
            .map(|t| (&t.tension_type, &t.character_a_name, &t.character_b_name))
            .collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn test_tension_conflicting_loyalties() {
    let harness = TestHarness::new().await;

    let (_alice_id, alice_key) = create_char(&harness, "Alice").await;
    let (_bob_id, bob_key) = create_char(&harness, "Bob").await;
    let (_charlie_id, charlie_key) = create_char(&harness, "Charlie").await;

    // Alice is allies with Bob
    create_relationship(
        &harness.db,
        RelationshipCreate {
            from_character_id: alice_key.clone(),
            to_character_id: bob_key.clone(),
            rel_type: "ally".to_string(),
            subtype: None,
            label: Some("close allies".to_string()),
        },
    )
    .await
    .unwrap();

    // Alice is also allies with Charlie
    create_relationship(
        &harness.db,
        RelationshipCreate {
            from_character_id: alice_key.clone(),
            to_character_id: charlie_key.clone(),
            rel_type: "ally".to_string(),
            subtype: None,
            label: Some("political allies".to_string()),
        },
    )
    .await
    .unwrap();

    // But Bob and Charlie are rivals
    create_relationship(
        &harness.db,
        RelationshipCreate {
            from_character_id: bob_key.clone(),
            to_character_id: charlie_key.clone(),
            rel_type: "rival".to_string(),
            subtype: None,
            label: Some("bitter enemies".to_string()),
        },
    )
    .await
    .unwrap();

    let service = TensionService::new(harness.db.clone());
    let report = service.detect_tensions(20, 0.0).await.unwrap();

    // "conflicting_loyalty" (singular) — the signal type from detect_conflicting_loyalties
    let has_conflict = report
        .tensions
        .iter()
        .any(|t| t.tension_type == "conflicting_loyalty");
    assert!(
        has_conflict,
        "Expected conflicting_loyalty tension. Got types: {:?}",
        report
            .tensions
            .iter()
            .map(|t| &t.tension_type)
            .collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn test_tension_high_edge_tension_from_perceptions() {
    let harness = TestHarness::new().await;

    let (_alice_id, alice_key) = create_char(&harness, "Alice").await;
    let (_bob_id, bob_key) = create_char(&harness, "Bob").await;

    // High tension perception: Alice → Bob with tension_level=9
    create_perception(
        &harness.db,
        &alice_key,
        &bob_key,
        PerceptionCreate {
            rel_types: vec!["rival".to_string()],
            subtype: None,
            feelings: Some("deep resentment".to_string()),
            perception: Some("Alice sees Bob as a threat".to_string()),
            tension_level: Some(9),
            history_notes: None,
        },
    )
    .await
    .unwrap();

    let service = TensionService::new(harness.db.clone());
    let report = service.detect_tensions(20, 0.0).await.unwrap();

    assert!(
        report.total_count > 0,
        "Expected at least one tension from high edge tension (perception >= 7)"
    );

    let has_edge_tension = report
        .tensions
        .iter()
        .any(|t| t.tension_type == "high_edge_tension");
    assert!(
        has_edge_tension,
        "Expected high_edge_tension type. Got: {:?}",
        report
            .tensions
            .iter()
            .map(|t| &t.tension_type)
            .collect::<Vec<_>>()
    );

    // Severity should be non-trivial
    let top = &report.tensions[0];
    assert!(top.severity > 0.0, "Expected non-zero severity");
}

#[tokio::test]
async fn test_tension_severity_ordering() {
    let harness = TestHarness::new().await;
    let knowledge_repo = SurrealKnowledgeRepository::new(harness.db.clone());

    let (_alice_id, alice_key) = create_char(&harness, "Alice").await;
    let (_bob_id, bob_key) = create_char(&harness, "Bob").await;
    let (_charlie_id, charlie_key) = create_char(&harness, "Charlie").await;
    let (_diana_id, diana_key) = create_char(&harness, "Diana").await;
    let event_key = create_evt(&harness, "The event", 1).await;

    // Strong tension: Alice-Bob (contradictory knowledge + rivalry + high perception tension)
    let fact = knowledge_repo
        .create_knowledge(
            KnowledgeBuilder::new("The treasure location")
                .for_character(&alice_key)
                .build(),
        )
        .await
        .unwrap();
    let fact_id = format!("knowledge:{}", fact.id.key());

    add_knowledge_state(
        &harness,
        &alice_key,
        &fact_id,
        CertaintyLevel::Knows,
        &event_key,
    )
    .await;
    add_knowledge_state(
        &harness,
        &bob_key,
        &fact_id,
        CertaintyLevel::BelievesWrongly,
        &event_key,
    )
    .await;

    create_relationship(
        &harness.db,
        RelationshipCreate {
            from_character_id: alice_key.clone(),
            to_character_id: bob_key.clone(),
            rel_type: "rival".to_string(),
            subtype: None,
            label: None,
        },
    )
    .await
    .unwrap();

    create_perception(
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
    )
    .await
    .unwrap();

    // Weaker tension: Charlie-Diana (just allies, no conflict)
    create_relationship(
        &harness.db,
        RelationshipCreate {
            from_character_id: charlie_key.clone(),
            to_character_id: diana_key.clone(),
            rel_type: "ally".to_string(),
            subtype: None,
            label: None,
        },
    )
    .await
    .unwrap();

    let service = TensionService::new(harness.db.clone());
    let report = service.detect_tensions(20, 0.0).await.unwrap();

    // Report should have tensions sorted by severity descending
    for w in report.tensions.windows(2) {
        assert!(
            w[0].severity >= w[1].severity,
            "Expected severity descending, got {} before {}",
            w[0].severity,
            w[1].severity
        );
    }
}

#[tokio::test]
async fn test_tension_min_severity_filter() {
    let harness = TestHarness::new().await;
    let knowledge_repo = SurrealKnowledgeRepository::new(harness.db.clone());

    let (_alice_id, alice_key) = create_char(&harness, "Alice").await;
    let (_bob_id, bob_key) = create_char(&harness, "Bob").await;
    let event_key = create_evt(&harness, "Learning event", 1).await;

    let fact = knowledge_repo
        .create_knowledge(
            KnowledgeBuilder::new("A secret")
                .for_character(&alice_key)
                .build(),
        )
        .await
        .unwrap();
    let fact_id = format!("knowledge:{}", fact.id.key());

    add_knowledge_state(
        &harness,
        &alice_key,
        &fact_id,
        CertaintyLevel::Knows,
        &event_key,
    )
    .await;
    add_knowledge_state(
        &harness,
        &bob_key,
        &fact_id,
        CertaintyLevel::BelievesWrongly,
        &event_key,
    )
    .await;

    let service = TensionService::new(harness.db.clone());

    let report_low = service.detect_tensions(20, 0.0).await.unwrap();
    let report_high = service.detect_tensions(20, 0.99).await.unwrap();

    assert!(
        report_high.total_count <= report_low.total_count,
        "Higher min_severity should never return more results"
    );
}

// ---------------------------------------------------------------------------
// RoleInferenceService integration tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_role_inference_empty_world() {
    let harness = TestHarness::new().await;
    let service = RoleInferenceService::new(harness.db.clone());
    let report = service.infer_roles(20).await.unwrap();

    assert_eq!(report.total_characters, 0);
    assert!(report.roles.is_empty());
}

#[tokio::test]
async fn test_role_inference_social_hub() {
    let harness = TestHarness::new().await;

    let (_hub_id, hub_key) = create_char(&harness, "Hub").await;

    let mut spoke_keys = Vec::new();
    for i in 0..5 {
        let (_id, key) = create_char(&harness, &format!("Spoke{}", i)).await;
        spoke_keys.push(key);
    }

    for spoke_key in &spoke_keys {
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
            spoke_key,
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
    }

    let service = RoleInferenceService::new(harness.db.clone());
    let report = service.infer_roles(20).await.unwrap();

    assert_eq!(report.total_characters, 6);

    let hub_role = report.roles.iter().find(|r| r.character_name == "Hub");
    assert!(hub_role.is_some(), "Hub should appear in role results");
    let hub_role = hub_role.unwrap();

    // Hub should have high confidence and be social_hub or connector
    let is_hub_like = hub_role.primary_role == "social_hub"
        || hub_role.primary_role == "connector"
        || hub_role.secondary_roles.contains(&"social_hub".to_string())
        || hub_role.secondary_roles.contains(&"connector".to_string());
    assert!(
        is_hub_like,
        "Hub should be inferred as social_hub or connector, got primary={}, secondary={:?}",
        hub_role.primary_role, hub_role.secondary_roles
    );
}

#[tokio::test]
async fn test_role_inference_deceived_character() {
    let harness = TestHarness::new().await;
    let knowledge_repo = SurrealKnowledgeRepository::new(harness.db.clone());

    let (_alice_id, alice_key) = create_char(&harness, "Alice").await;
    let (_narrator_id, narrator_key) = create_char(&harness, "Narrator").await;
    let (_bob_id, bob_key) = create_char(&harness, "Bob").await;
    let event_key = create_evt(&harness, "Deception event", 1).await;

    // Give Alice connections so she's not an outsider (which outranks deceived)
    create_perception(
        &harness.db,
        &alice_key,
        &bob_key,
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
    create_perception(
        &harness.db,
        &bob_key,
        &alice_key,
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

    for i in 0..3 {
        let fact = knowledge_repo
            .create_knowledge(
                KnowledgeBuilder::new(format!("False fact {}", i))
                    .for_character(&narrator_key)
                    .build(),
            )
            .await
            .unwrap();
        let fact_id = format!("knowledge:{}", fact.id.key());

        add_knowledge_state(
            &harness,
            &alice_key,
            &fact_id,
            CertaintyLevel::BelievesWrongly,
            &event_key,
        )
        .await;
    }

    let service = RoleInferenceService::new(harness.db.clone());
    let report = service.infer_roles(20).await.unwrap();

    let alice_role = report.roles.iter().find(|r| r.character_name == "Alice");
    assert!(alice_role.is_some(), "Alice should appear in role results");
    let alice_role = alice_role.unwrap();

    let is_deceived = alice_role.primary_role == "deceived"
        || alice_role.secondary_roles.contains(&"deceived".to_string());
    assert!(
        is_deceived,
        "Alice with 3 false beliefs should be inferred as deceived, got primary={}, secondary={:?}",
        alice_role.primary_role, alice_role.secondary_roles
    );
}

#[tokio::test]
async fn test_role_inference_bridge_character() {
    let harness = TestHarness::new().await;

    // Bridge connects two clusters via perceives edges
    let (_bridge_id, bridge_key) = create_char(&harness, "Bridge").await;
    let (_a1_id, a1_key) = create_char(&harness, "ClusterA1").await;
    let (_a2_id, a2_key) = create_char(&harness, "ClusterA2").await;
    let (_b1_id, b1_key) = create_char(&harness, "ClusterB1").await;
    let (_b2_id, b2_key) = create_char(&harness, "ClusterB2").await;

    // Intra-cluster A edges
    create_perception(
        &harness.db,
        &a1_key,
        &a2_key,
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
    create_perception(
        &harness.db,
        &a2_key,
        &a1_key,
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

    // Intra-cluster B edges
    create_perception(
        &harness.db,
        &b1_key,
        &b2_key,
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
    create_perception(
        &harness.db,
        &b2_key,
        &b1_key,
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

    // Bridge connects both clusters (bidirectional)
    for key in [&a1_key, &a2_key, &b1_key, &b2_key] {
        create_perception(
            &harness.db,
            &bridge_key,
            key,
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
        create_perception(
            &harness.db,
            key,
            &bridge_key,
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
    }

    let service = RoleInferenceService::new(harness.db.clone());
    let report = service.infer_roles(20).await.unwrap();

    let bridge_role = report.roles.iter().find(|r| r.character_name == "Bridge");
    assert!(
        bridge_role.is_some(),
        "Bridge should appear in role results"
    );
    let bridge_role = bridge_role.unwrap();

    // Bridge has highest degree centrality (4 connections vs 2 for others)
    // Should be social_hub or connector (betweenness is always 0 in current impl)
    let is_central = bridge_role.primary_role == "social_hub"
        || bridge_role.primary_role == "connector"
        || bridge_role.primary_role == "bridge";
    assert!(
        is_central,
        "Bridge character should be inferred as social_hub or connector (highest degree), got primary={}, secondary={:?}",
        bridge_role.primary_role, bridge_role.secondary_roles
    );

    // Bridge should be ranked first by confidence (highest centrality)
    assert_eq!(
        report.roles[0].character_name, "Bridge",
        "Bridge should be ranked first by confidence"
    );
}

#[tokio::test]
async fn test_role_inference_mentor() {
    let harness = TestHarness::new().await;

    let (_mentor_id, mentor_key) = create_char(&harness, "Mentor").await;
    let (_student1_id, student1_key) = create_char(&harness, "Student1").await;
    let (_student2_id, student2_key) = create_char(&harness, "Student2").await;
    let (_student3_id, student3_key) = create_char(&harness, "Student3").await;

    for student_key in [&student1_key, &student2_key, &student3_key] {
        create_relationship(
            &harness.db,
            RelationshipCreate {
                from_character_id: mentor_key.clone(),
                to_character_id: student_key.clone(),
                rel_type: "mentor".to_string(),
                subtype: None,
                label: None,
            },
        )
        .await
        .unwrap();
    }

    let service = RoleInferenceService::new(harness.db.clone());
    let report = service.infer_roles(20).await.unwrap();

    let mentor_role = report.roles.iter().find(|r| r.character_name == "Mentor");
    assert!(
        mentor_role.is_some(),
        "Mentor should appear in role results"
    );
    let mentor_role = mentor_role.unwrap();

    let is_mentor = mentor_role.primary_role == "mentor"
        || mentor_role.secondary_roles.contains(&"mentor".to_string());
    assert!(
        is_mentor,
        "Character with 3 mentorship edges should be inferred as mentor, got primary={}, secondary={:?}",
        mentor_role.primary_role, mentor_role.secondary_roles
    );
}

#[tokio::test]
async fn test_role_inference_confidence_ordering() {
    let harness = TestHarness::new().await;

    let (_alice_id, alice_key) = create_char(&harness, "Alice").await;
    let (_bob_id, bob_key) = create_char(&harness, "Bob").await;
    let (_charlie_id, _charlie_key) = create_char(&harness, "Charlie").await;

    create_perception(
        &harness.db,
        &alice_key,
        &bob_key,
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

    let service = RoleInferenceService::new(harness.db.clone());
    let report = service.infer_roles(20).await.unwrap();

    for w in report.roles.windows(2) {
        assert!(
            w[0].confidence >= w[1].confidence,
            "Expected confidence descending, got {} ({}) before {} ({})",
            w[0].character_name,
            w[0].confidence,
            w[1].character_name,
            w[1].confidence
        );
    }
}

#[tokio::test]
async fn test_role_inference_limit_parameter() {
    let harness = TestHarness::new().await;

    for i in 0..5 {
        create_char(&harness, &format!("Char{}", i)).await;
    }

    let service = RoleInferenceService::new(harness.db.clone());

    let full = service.infer_roles(100).await.unwrap();
    assert_eq!(full.total_characters, 5);
    assert_eq!(full.roles.len(), 5);

    let limited = service.infer_roles(2).await.unwrap();
    assert_eq!(limited.total_characters, 5);
    assert_eq!(limited.roles.len(), 2);
}

// ---------------------------------------------------------------------------
// Combined: tension + role inference on the same world
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_combined_rich_narrative_world() {
    let harness = TestHarness::new().await;
    let knowledge_repo = SurrealKnowledgeRepository::new(harness.db.clone());

    let (_alice_id, alice_key) = create_char(&harness, "Alice").await;
    let (_bob_id, bob_key) = create_char(&harness, "Bob").await;
    let (_charlie_id, charlie_key) = create_char(&harness, "Charlie").await;
    let (_diana_id, diana_key) = create_char(&harness, "Diana").await;
    let (_eve_id, _eve_key) = create_char(&harness, "Eve").await;
    let event_key = create_evt(&harness, "The revelation", 1).await;

    // Knowledge: Alice knows, Bob believes wrongly
    let fact = knowledge_repo
        .create_knowledge(
            KnowledgeBuilder::new("The crown's secret")
                .for_character(&alice_key)
                .build(),
        )
        .await
        .unwrap();
    let fact_id = format!("knowledge:{}", fact.id.key());

    add_knowledge_state(
        &harness,
        &alice_key,
        &fact_id,
        CertaintyLevel::Knows,
        &event_key,
    )
    .await;
    add_knowledge_state(
        &harness,
        &bob_key,
        &fact_id,
        CertaintyLevel::BelievesWrongly,
        &event_key,
    )
    .await;

    // Relationships
    create_relationship(
        &harness.db,
        RelationshipCreate {
            from_character_id: alice_key.clone(),
            to_character_id: charlie_key.clone(),
            rel_type: "ally".to_string(),
            subtype: None,
            label: None,
        },
    )
    .await
    .unwrap();

    create_relationship(
        &harness.db,
        RelationshipCreate {
            from_character_id: charlie_key.clone(),
            to_character_id: bob_key.clone(),
            rel_type: "rival".to_string(),
            subtype: None,
            label: None,
        },
    )
    .await
    .unwrap();

    // Perceptions
    create_perception_pair(
        &harness.db,
        &alice_key,
        &charlie_key,
        PerceptionCreate {
            rel_types: vec!["ally".to_string()],
            subtype: None,
            feelings: Some("trusts".to_string()),
            perception: None,
            tension_level: Some(1),
            history_notes: None,
        },
        PerceptionCreate {
            rel_types: vec!["ally".to_string()],
            subtype: None,
            feelings: Some("admires".to_string()),
            perception: None,
            tension_level: Some(2),
            history_notes: None,
        },
    )
    .await
    .unwrap();

    create_perception_pair(
        &harness.db,
        &charlie_key,
        &bob_key,
        PerceptionCreate {
            rel_types: vec!["rival".to_string()],
            subtype: None,
            feelings: Some("contempt".to_string()),
            perception: None,
            tension_level: Some(8),
            history_notes: None,
        },
        PerceptionCreate {
            rel_types: vec!["rival".to_string()],
            subtype: None,
            feelings: Some("fear".to_string()),
            perception: None,
            tension_level: Some(7),
            history_notes: None,
        },
    )
    .await
    .unwrap();

    create_perception(
        &harness.db,
        &diana_key,
        &alice_key,
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

    // Run both services
    let tension_service = TensionService::new(harness.db.clone());
    let tension_report = tension_service.detect_tensions(20, 0.0).await.unwrap();

    let role_service = RoleInferenceService::new(harness.db.clone());
    let role_report = role_service.infer_roles(20).await.unwrap();

    // Tensions should exist (contradictory knowledge + high perception tensions)
    assert!(
        tension_report.total_count > 0,
        "Rich world should produce at least one tension"
    );

    // All 5 characters should have roles
    assert_eq!(role_report.total_characters, 5);
    assert_eq!(role_report.roles.len(), 5);

    // Eve (isolated) should be outsider
    let eve_role = role_report.roles.iter().find(|r| r.character_name == "Eve");
    assert!(eve_role.is_some());
    let eve_role = eve_role.unwrap();
    let is_outsider = eve_role.primary_role == "outsider"
        || eve_role.secondary_roles.contains(&"outsider".to_string());
    assert!(
        is_outsider,
        "Eve with no connections should be outsider, got primary={}, secondary={:?}",
        eve_role.primary_role, eve_role.secondary_roles
    );

    // Bob (BelievesWrongly) should appear
    let bob_role = role_report.roles.iter().find(|r| r.character_name == "Bob");
    assert!(bob_role.is_some());
}
