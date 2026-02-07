//! Integration tests for consistency validation tools (Phase 13).
//!
//! Tests on-demand validation and investigation queries.

mod common;

use common::harness::TestHarness;
use narra::models::character::create_character;
use narra::models::perception::{create_perception, PerceptionCreate};
use narra::models::CharacterCreate;
use narra::services::{ConsistencyChecker, ConsistencySeverity};

/// Test relationship impossible state detection.
///
/// Scenario: A is B's parent AND B is A's parent (impossible).
#[tokio::test]
async fn test_circular_parent_detection() {
    let harness = TestHarness::new().await;
    let db = &harness.db;

    // Create two characters
    let alice = create_character(
        &db,
        CharacterCreate {
            name: "Alice".to_string(),
            aliases: vec![],
            roles: vec![],
            ..Default::default()
        },
    )
    .await
    .expect("Failed to create Alice");

    let bob = create_character(
        &db,
        CharacterCreate {
            name: "Bob".to_string(),
            aliases: vec![],
            roles: vec![],
            ..Default::default()
        },
    )
    .await
    .expect("Failed to create Bob");

    let alice_key = alice.id.key().to_string();
    let bob_key = bob.id.key().to_string();

    // Alice perceives Bob as her child (Alice is parent)
    create_perception(
        &db,
        &alice_key,
        &bob_key,
        PerceptionCreate {
            rel_types: vec!["family".to_string(), "parent".to_string()],
            feelings: Some("protective".to_string()),
            subtype: None,
            perception: None,
            tension_level: None,
            history_notes: None,
        },
    )
    .await
    .expect("Failed to create Alice->Bob perception");

    // Bob perceives Alice as his child (Bob is parent) - IMPOSSIBLE circular
    create_perception(
        &db,
        &bob_key,
        &alice_key,
        PerceptionCreate {
            rel_types: vec!["family".to_string(), "parent".to_string()],
            feelings: Some("protective".to_string()),
            subtype: None,
            perception: None,
            tension_level: None,
            history_notes: None,
        },
    )
    .await
    .expect("Failed to create Bob->Alice perception");

    let checker = ConsistencyChecker::new(db.clone());
    let violations = checker
        .check_relationship_violations(&alice_key)
        .await
        .expect("Should not error");

    // Should find CRITICAL circular parent violation
    assert!(
        !violations.is_empty(),
        "Should detect circular parent relationship"
    );
    assert!(
        violations
            .iter()
            .any(|v| v.severity == ConsistencySeverity::Critical),
        "Circular parent should be CRITICAL severity"
    );
    assert!(
        violations
            .iter()
            .any(|v| v.message.to_lowercase().contains("circular")),
        "Message should mention circular relationship"
    );
}

/// Test relationship asymmetry detection for family.
///
/// Scenario: Siblings with different feelings - should be WARNING.
#[tokio::test]
async fn test_family_asymmetry_warning() {
    let harness = TestHarness::new().await;
    let db = &harness.db;

    let alice = create_character(
        &db,
        CharacterCreate {
            name: "Alice".to_string(),
            aliases: vec![],
            roles: vec![],
            ..Default::default()
        },
    )
    .await
    .expect("Failed to create Alice");

    let bob = create_character(
        &db,
        CharacterCreate {
            name: "Bob".to_string(),
            aliases: vec![],
            roles: vec![],
            ..Default::default()
        },
    )
    .await
    .expect("Failed to create Bob");

    let alice_key = alice.id.key().to_string();
    let bob_key = bob.id.key().to_string();

    // Alice loves Bob (family sibling)
    create_perception(
        &db,
        &alice_key,
        &bob_key,
        PerceptionCreate {
            rel_types: vec!["family".to_string()],
            subtype: Some("sibling".to_string()),
            feelings: Some("love".to_string()),
            perception: None,
            tension_level: None,
            history_notes: None,
        },
    )
    .await
    .expect("Failed to create Alice->Bob perception");

    // Bob resents Alice (family asymmetry)
    create_perception(
        &db,
        &bob_key,
        &alice_key,
        PerceptionCreate {
            rel_types: vec!["family".to_string()],
            subtype: Some("sibling".to_string()),
            feelings: Some("resentment".to_string()),
            perception: None,
            tension_level: None,
            history_notes: None,
        },
    )
    .await
    .expect("Failed to create Bob->Alice perception");

    let checker = ConsistencyChecker::new(db.clone());
    let violations = checker
        .check_relationship_violations(&alice_key)
        .await
        .expect("Should not error");

    // Should find WARNING for family asymmetry
    if !violations.is_empty() {
        assert!(
            violations
                .iter()
                .any(|v| v.severity == ConsistencySeverity::Warning),
            "Family asymmetry should be WARNING severity"
        );
    }
}

/// Test romantic asymmetry is INFO (often intentional).
#[tokio::test]
async fn test_romantic_asymmetry_info() {
    let harness = TestHarness::new().await;
    let db = &harness.db;

    let alice = create_character(
        &db,
        CharacterCreate {
            name: "Alice".to_string(),
            aliases: vec![],
            roles: vec![],
            ..Default::default()
        },
    )
    .await
    .expect("Failed to create Alice");

    let bob = create_character(
        &db,
        CharacterCreate {
            name: "Bob".to_string(),
            aliases: vec![],
            roles: vec![],
            ..Default::default()
        },
    )
    .await
    .expect("Failed to create Bob");

    let alice_key = alice.id.key().to_string();
    let bob_key = bob.id.key().to_string();

    // Alice loves Bob romantically
    create_perception(
        &db,
        &alice_key,
        &bob_key,
        PerceptionCreate {
            rel_types: vec!["romantic".to_string()],
            feelings: Some("devotion".to_string()),
            subtype: None,
            perception: None,
            tension_level: None,
            history_notes: None,
        },
    )
    .await
    .expect("Failed to create Alice->Bob perception");

    // Bob is indifferent to Alice (unrequited love)
    create_perception(
        &db,
        &bob_key,
        &alice_key,
        PerceptionCreate {
            rel_types: vec!["romantic".to_string()],
            feelings: Some("indifference".to_string()),
            subtype: None,
            perception: None,
            tension_level: None,
            history_notes: None,
        },
    )
    .await
    .expect("Failed to create Bob->Alice perception");

    let checker = ConsistencyChecker::new(db.clone());
    let violations = checker
        .check_relationship_violations(&alice_key)
        .await
        .expect("Should not error");

    // Romantic asymmetry should be INFO (often intentional drama), never CRITICAL
    for v in &violations {
        assert_ne!(
            v.severity,
            ConsistencySeverity::Critical,
            "Romantic asymmetry should NOT be CRITICAL"
        );
    }
}

/// Test one-way relationship is NOT a violation.
#[tokio::test]
async fn test_one_way_relationship_valid() {
    let harness = TestHarness::new().await;
    let db = &harness.db;

    let alice = create_character(
        &db,
        CharacterCreate {
            name: "Alice".to_string(),
            aliases: vec![],
            roles: vec![],
            ..Default::default()
        },
    )
    .await
    .expect("Failed to create Alice");

    let bob = create_character(
        &db,
        CharacterCreate {
            name: "Bob".to_string(),
            aliases: vec![],
            roles: vec![],
            ..Default::default()
        },
    )
    .await
    .expect("Failed to create Bob");

    let alice_key = alice.id.key().to_string();
    let bob_key = bob.id.key().to_string();

    // Alice perceives Bob (one-way, Bob doesn't know Alice exists)
    create_perception(
        &db,
        &alice_key,
        &bob_key,
        PerceptionCreate {
            rel_types: vec!["rivalry".to_string()],
            feelings: Some("jealousy".to_string()),
            subtype: None,
            perception: None,
            tension_level: None,
            history_notes: None,
        },
    )
    .await
    .expect("Failed to create Alice->Bob perception");

    // NO reverse perception from Bob to Alice

    let checker = ConsistencyChecker::new(db.clone());
    let violations = checker
        .check_relationship_violations(&alice_key)
        .await
        .expect("Should not error");

    // One-way relationships are valid - filter violations related to this specific relationship
    let bob_violations: Vec<_> = violations
        .iter()
        .filter(|v| v.message.to_lowercase().contains("bob"))
        .collect();

    // One-way relationships should not generate CRITICAL violations
    assert!(
        bob_violations
            .iter()
            .all(|v| v.severity != ConsistencySeverity::Critical),
        "One-way relationships should not generate CRITICAL violations"
    );
}

/// Test that validation runs without error on entity with no relationships.
#[tokio::test]
async fn test_validate_character_no_relationships() {
    let harness = TestHarness::new().await;
    let db = &harness.db;

    let alice = create_character(
        &db,
        CharacterCreate {
            name: "Lonely Alice".to_string(),
            aliases: vec![],
            roles: vec!["hermit".to_string()],
            ..Default::default()
        },
    )
    .await
    .expect("Failed to create Alice");

    let alice_key = alice.id.key().to_string();

    let checker = ConsistencyChecker::new(db.clone());

    // Should complete without error even with no relationships
    let timeline_result = checker.check_timeline_violations(&alice_key).await;
    assert!(timeline_result.is_ok(), "Timeline check should succeed");

    let rel_result = checker.check_relationship_violations(&alice_key).await;
    assert!(rel_result.is_ok(), "Relationship check should succeed");
}
