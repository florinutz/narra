//! Integration tests for consistency checking during mutations.

mod common;

use common::harness::TestHarness;
use narra::models::character::create_character;
use narra::models::fact::{create_fact, link_fact_to_entity, EnforcementLevel, FactCreate};
use narra::models::CharacterCreate;
use narra::services::{ConsistencyChecker, ConsistencyService};

#[tokio::test]
async fn test_strict_fact_creates_critical_violation() {
    let harness = TestHarness::new().await;
    let db = &harness.db;

    // Create a Strict fact: "No magic in this world"
    let fact = create_fact(
        &db,
        FactCreate {
            title: "No magic".to_string(),
            description:
                "Magic does not exist in this world. No characters can have magical abilities."
                    .to_string(),
            categories: vec![],
            enforcement_level: EnforcementLevel::Strict,
            scope: None,
        },
    )
    .await
    .expect("Failed to create fact");

    // Create a character to link the fact to
    let character = create_character(
        &db,
        CharacterCreate {
            name: "Test Wizard".to_string(),
            aliases: vec![],
            roles: vec!["wizard".to_string()],
            ..Default::default()
        },
    )
    .await
    .expect("Failed to create character");

    // Link fact to character
    let entity_id = character.id.to_string();
    let fact_key = fact.id.key().to_string();
    link_fact_to_entity(&db, &fact_key, &entity_id, "manual", None)
        .await
        .expect("Failed to link fact");

    // Check consistency with mutation that violates the fact
    let checker = ConsistencyChecker::new(db.clone());
    let mutation_data = serde_json::json!({
        "name": "Test Wizard",
        "abilities": ["magical powers", "can cast spells"]
    });

    let result = checker
        .check_entity_mutation(&entity_id, &mutation_data)
        .await
        .expect("Consistency check should not error");

    // Strict fact linked to an entity with violating mutation data
    // The service should detect at least one violation
    assert!(
        result.total_violations > 0,
        "Strict fact should produce violations when mutation conflicts with fact"
    );
}

#[tokio::test]
async fn test_warning_fact_allows_mutation() {
    let harness = TestHarness::new().await;
    let db = &harness.db;

    // Create a Warning-level fact
    let fact = create_fact(
        &db,
        FactCreate {
            title: "Names follow culture pattern".to_string(),
            description:
                "Character names should follow the established cultural naming conventions."
                    .to_string(),
            categories: vec![],
            enforcement_level: EnforcementLevel::Warning,
            scope: None,
        },
    )
    .await
    .expect("Failed to create fact");

    let checker = ConsistencyChecker::new(db.clone());

    // Check creation that might violate naming convention
    let creation_data = serde_json::json!({
        "name": "X123_Invalid",
        "role": "peasant"
    });

    let result = checker
        .check_entity_creation("character", &creation_data)
        .await
        .expect("Consistency check should not error");

    // Warning-level facts should never have blocking violations
    assert!(
        !result.has_blocking_violations,
        "Warning-level facts should not block mutations"
    );
}

#[tokio::test]
async fn test_no_violations_when_no_facts() {
    let harness = TestHarness::new().await;
    let db = &harness.db;

    let checker = ConsistencyChecker::new(db.clone());

    // Check mutation with no facts defined
    let mutation_data = serde_json::json!({
        "name": "Normal Character",
        "abilities": ["sword fighting"]
    });

    let result = checker
        .check_entity_creation("character", &mutation_data)
        .await
        .expect("Consistency check should not error");

    // No facts = no violations
    assert_eq!(
        result.total_violations, 0,
        "Should have no violations when no facts are defined"
    );
    assert!(result.is_valid, "Should be valid with no facts");
}

#[tokio::test]
async fn test_informational_fact_never_blocks() {
    let harness = TestHarness::new().await;
    let db = &harness.db;

    // Create an Informational-level fact
    create_fact(
        &db,
        FactCreate {
            title: "Common greetings".to_string(),
            description: "People commonly greet each other with a nod.".to_string(),
            categories: vec![],
            enforcement_level: EnforcementLevel::Informational,
            scope: None,
        },
    )
    .await
    .expect("Failed to create fact");

    let checker = ConsistencyChecker::new(db.clone());

    let mutation_data = serde_json::json!({
        "greeting_style": "handshake",
    });

    let result = checker
        .check_entity_mutation("character:test", &mutation_data)
        .await
        .expect("Consistency check should not error");

    // Informational facts should never block
    assert!(
        !result.has_blocking_violations,
        "Informational facts should never block"
    );
}

#[tokio::test]
async fn test_timeout_prevents_long_running_check() {
    let harness = TestHarness::new().await;
    let db = &harness.db;

    let checker = ConsistencyChecker::new(db.clone());

    // Even with complex data, check should complete within timeout
    let large_data = serde_json::json!({
        "description": "A ".repeat(10000),
        "attributes": (0..100).map(|i| format!("attr_{}", i)).collect::<Vec<_>>(),
    });

    let start = std::time::Instant::now();
    let result = checker
        .check_entity_mutation("character:test", &large_data)
        .await;
    let elapsed = start.elapsed();

    // Should complete or timeout within 3 seconds (2s timeout + overhead)
    assert!(
        elapsed.as_secs() < 3,
        "Check should complete within timeout"
    );
    assert!(
        result.is_ok(),
        "Check should complete successfully or timeout gracefully"
    );
}
