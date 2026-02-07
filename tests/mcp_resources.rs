//! Integration tests for MCP Resources.

mod common;

use std::sync::Arc;

use common::harness::TestHarness;
use narra::models::character::create_character;
use narra::models::CharacterCreate;
use narra::session::SessionStateManager;

#[tokio::test]
async fn test_session_context_resource_empty_world() {
    let harness = TestHarness::new().await;
    let session_path = harness.temp_path().join("session.json");
    let session_manager = Arc::new(
        SessionStateManager::load_or_create(&session_path)
            .expect("Failed to create session manager"),
    );

    let result = narra::mcp::resources::get_session_context_resource(&session_manager, &harness.db)
        .await
        .expect("Should return session context");

    // Parse and verify structure
    let info: narra::session::SessionStartupInfo =
        serde_json::from_str(&result).expect("Should be valid SessionStartupInfo JSON");

    // Empty world should have EmptyWorld verbosity
    assert_eq!(info.verbosity, narra::session::StartupVerbosity::EmptyWorld);
    assert!(info.hot_entities.is_empty());
}

#[tokio::test]
async fn test_entity_resource_not_found() {
    let harness = TestHarness::new().await;
    let session_path = harness.temp_path().join("session.json");
    let session_manager = Arc::new(
        SessionStateManager::load_or_create(&session_path)
            .expect("Failed to create session manager"),
    );

    // Create context service
    let context_service: Arc<dyn narra::services::ContextService + Send + Sync> = Arc::new(
        narra::services::CachedContextService::new(harness.db.clone(), session_manager),
    );

    let result =
        narra::mcp::resources::get_entity_resource("character:nonexistent", &context_service).await;

    assert!(result.is_err());
    assert!(result.unwrap_err().contains("not found"));
}

#[tokio::test]
async fn test_entity_resource_with_character() {
    let harness = TestHarness::new().await;
    let session_path = harness.temp_path().join("session.json");
    let session_manager = Arc::new(
        SessionStateManager::load_or_create(&session_path)
            .expect("Failed to create session manager"),
    );

    // Create a test character
    let character = create_character(
        &harness.db,
        CharacterCreate {
            name: "Alice".to_string(),
            aliases: vec![],
            roles: vec![],
            ..Default::default()
        },
    )
    .await
    .expect("Should create character");

    let char_id = character.id.to_string();

    // Create context service
    let context_service: Arc<dyn narra::services::ContextService + Send + Sync> = Arc::new(
        narra::services::CachedContextService::new(harness.db.clone(), session_manager),
    );

    let result = narra::mcp::resources::get_entity_resource(&char_id, &context_service)
        .await
        .expect("Should return entity");

    // Verify it's valid JSON with expected fields
    let entity: serde_json::Value = serde_json::from_str(&result).expect("Should be valid JSON");

    assert_eq!(entity["id"], char_id);
    assert_eq!(entity["name"], "Alice");
}

#[tokio::test]
async fn test_consistency_issues_resource_empty() {
    let harness = TestHarness::new().await;

    let consistency_service: Arc<dyn narra::services::ConsistencyService> =
        Arc::new(narra::services::ConsistencyChecker::new(harness.db.clone()));

    let result =
        narra::mcp::resources::get_consistency_issues_resource(&harness.db, &consistency_service)
            .await
            .expect("Should return consistency report");

    // Parse and verify structure
    let report: serde_json::Value = serde_json::from_str(&result).expect("Should be valid JSON");

    assert_eq!(report["total_issues"], 0);
    assert_eq!(report["critical_count"], 0);
}
