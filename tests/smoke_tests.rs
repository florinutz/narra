//! Smoke tests for all 5 MCP tools (consolidated from original 11).
//!
//! Purpose: Regression prevention - verify each tool can be invoked without error
//! and returns structurally valid responses.

mod common;

use common::harness::TestHarness;
use narra::mcp::tools::export::ExportRequest;
use narra::mcp::tools::graph::GraphRequest;
use narra::mcp::{MutationRequest, NarraServer, QueryRequest, SessionRequest};
use rmcp::handler::server::wrapper::Parameters;

/// Helper to create a character and return its ID.
async fn create_test_character(server: &NarraServer, name: &str) -> String {
    let request = MutationRequest::CreateCharacter {
        id: None,
        name: name.to_string(),
        role: Some("Test Role".to_string()),
        aliases: None,
        description: Some("Test character for smoke tests".to_string()),
        profile: None,
    };
    let result = server
        .mutate(Parameters(request))
        .await
        .expect("create character should succeed");
    result.0.entity.id
}

// =============================================================================
// SMOKE TESTS - One per MCP tool + operations that moved between tools
// =============================================================================

#[tokio::test]
async fn smoke_test_query() {
    let harness = TestHarness::new().await;
    let server = common::create_test_server(&harness).await;

    // Create test data so the overview has something to return
    create_test_character(&server, "Query Test Character").await;

    let request = QueryRequest::Overview {
        entity_type: "character".to_string(),
        limit: Some(10),
    };

    let result = server
        .query(Parameters(request))
        .await
        .expect("query should succeed");

    assert!(
        result.0.total > 0,
        "Overview should find at least one entity"
    );
    assert!(
        !result.0.results.is_empty(),
        "Overview should return results"
    );
    assert_eq!(result.0.results[0].entity_type, "character");
}

#[tokio::test]
async fn smoke_test_mutate() {
    let harness = TestHarness::new().await;
    let server = common::create_test_server(&harness).await;

    let request = MutationRequest::CreateCharacter {
        id: None,
        name: "Smoke Test Character".to_string(),
        role: Some("Protagonist".to_string()),
        aliases: None,
        description: Some("Created in smoke test".to_string()),
        profile: None,
    };

    let result = server
        .mutate(Parameters(request))
        .await
        .expect("mutate should succeed");

    assert_eq!(result.0.entity.entity_type, "character");
    assert!(
        result.0.entity.id.starts_with("character:"),
        "ID should have table prefix"
    );
    assert!(
        result.0.entity.content.contains("Smoke Test Character"),
        "Content should mention the character name"
    );
}

#[tokio::test]
async fn smoke_test_analyze_impact_via_query() {
    let harness = TestHarness::new().await;
    let server = common::create_test_server(&harness).await;

    let character_id = create_test_character(&server, "Impact Test Character").await;

    let request = QueryRequest::AnalyzeImpact {
        entity_id: character_id,
        proposed_change: Some("Delete character".to_string()),
        include_details: Some(true),
    };

    let result = server
        .query(Parameters(request))
        .await
        .expect("analyze_impact via query should succeed");

    assert!(
        result.0.token_estimate > 0,
        "Response should have a token estimate"
    );
    assert!(!result.0.results.is_empty(), "Should return impact results");
}

#[tokio::test]
async fn smoke_test_protect_entity_via_mutate() {
    let harness = TestHarness::new().await;
    let server = common::create_test_server(&harness).await;

    let character_id = create_test_character(&server, "Protect Test Character").await;

    let request = MutationRequest::ProtectEntity {
        entity_id: character_id.clone(),
    };

    let result = server
        .mutate(Parameters(request))
        .await
        .expect("protect_entity via mutate should succeed");

    assert!(
        result.0.entity.content.contains("protected"),
        "Response should confirm protection"
    );
}

#[tokio::test]
async fn smoke_test_unprotect_entity_via_mutate() {
    let harness = TestHarness::new().await;
    let server = common::create_test_server(&harness).await;

    let character_id = create_test_character(&server, "Unprotect Test Character").await;

    server
        .mutate(Parameters(MutationRequest::ProtectEntity {
            entity_id: character_id.clone(),
        }))
        .await
        .expect("protect should succeed");

    let result = server
        .mutate(Parameters(MutationRequest::UnprotectEntity {
            entity_id: character_id,
        }))
        .await
        .expect("unprotect via mutate should succeed");

    assert!(
        result.0.entity.content.contains("Protection removed")
            || result.0.entity.content.contains("unprotected"),
        "Response should confirm unprotection, got: {}",
        result.0.entity.content
    );
}

#[tokio::test]
async fn smoke_test_pin_entity_via_session() {
    let harness = TestHarness::new().await;
    let server = common::create_test_server(&harness).await;

    let character_id = create_test_character(&server, "Pin Test Character").await;

    let request = SessionRequest::PinEntity {
        entity_id: character_id.clone(),
    };

    let result = server
        .session(Parameters(request))
        .await
        .expect("pin_entity via session should succeed");

    assert_eq!(result.0.operation, "pin_entity");
    let pin_result = result.0.pin_result.expect("Should have pin_result");
    assert!(pin_result.success, "Pin should succeed");
    assert_eq!(pin_result.entity_id, character_id);
}

#[tokio::test]
async fn smoke_test_unpin_entity_via_session() {
    let harness = TestHarness::new().await;
    let server = common::create_test_server(&harness).await;

    let character_id = create_test_character(&server, "Unpin Test Character").await;

    server
        .session(Parameters(SessionRequest::PinEntity {
            entity_id: character_id.clone(),
        }))
        .await
        .expect("pin should succeed");

    let result = server
        .session(Parameters(SessionRequest::UnpinEntity {
            entity_id: character_id.clone(),
        }))
        .await
        .expect("unpin via session should succeed");

    assert_eq!(result.0.operation, "unpin_entity");
    let pin_result = result.0.pin_result.expect("Should have pin_result");
    assert!(pin_result.success, "Unpin should succeed");
    assert_eq!(pin_result.entity_id, character_id);
}

#[tokio::test]
async fn smoke_test_get_session_context_via_session() {
    let harness = TestHarness::new().await;
    let server = common::create_test_server(&harness).await;

    let request = SessionRequest::GetContext { force_full: false };

    let result = server
        .session(Parameters(request))
        .await
        .expect("get_session_context via session should succeed");

    assert_eq!(result.0.operation, "get_context");
    let context = result.0.context.expect("Should have context data");
    assert!(
        !context.verbosity.is_empty(),
        "Context should have a verbosity level"
    );
}

#[tokio::test]
async fn smoke_test_export_world() {
    let harness = TestHarness::new().await;
    let server = common::create_test_server(&harness).await;

    let export_path = harness.temp_path().join("export.yaml");

    let request = ExportRequest {
        output_path: Some(export_path.to_string_lossy().to_string()),
    };

    let result = server
        .export_world(Parameters(request))
        .await
        .expect("export_world should succeed");

    assert!(
        !result.0.output_path.is_empty(),
        "Should return the output path"
    );
    assert!(
        std::path::Path::new(&result.0.output_path).exists(),
        "Export file should exist on disk"
    );
}

#[tokio::test]
async fn smoke_test_generate_graph() {
    let harness = TestHarness::new().await;
    let server = common::create_test_server(&harness).await;

    let request = GraphRequest {
        scope: "full".to_string(),
        depth: None,
        include_roles: None,
        filename: Some("smoke-test-graph.md".to_string()),
    };

    let result = server
        .generate_graph(Parameters(request))
        .await
        .expect("generate_graph should succeed");

    assert!(
        !result.0.file_path.is_empty(),
        "Should return a file path for the graph"
    );
}

#[tokio::test]
async fn smoke_test_validate_entity_via_query() {
    let harness = TestHarness::new().await;
    let server = common::create_test_server(&harness).await;

    let character_id = create_test_character(&server, "Validate Test Character").await;

    let request = QueryRequest::ValidateEntity {
        entity_id: character_id,
    };

    let result = server
        .query(Parameters(request))
        .await
        .expect("validate_entity via query should succeed");

    assert!(
        !result.0.results.is_empty(),
        "Validation should return results"
    );
    assert!(
        result.0.token_estimate > 0,
        "Response should have a token estimate"
    );
}

#[tokio::test]
async fn smoke_test_investigate_contradictions_via_query() {
    let harness = TestHarness::new().await;
    let server = common::create_test_server(&harness).await;

    let character_id = create_test_character(&server, "Investigate Test Character").await;

    let request = QueryRequest::InvestigateContradictions {
        entity_id: character_id,
        max_depth: 2,
    };

    let result = server
        .query(Parameters(request))
        .await
        .expect("investigate_contradictions via query should succeed");

    assert!(
        result.0.token_estimate > 0,
        "Response should have a token estimate"
    );
}
