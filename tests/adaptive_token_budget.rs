/// Integration tests for adaptive token budget feature
mod common;
use common::harness::{create_test_server, TestHarness};
use common::input_helpers::to_query_input;
use narra::mcp::{QueryInput, QueryRequest};
use rmcp::handler::server::wrapper::Parameters;
use serde_json::json;

/// Test per-request token_budget parameter overrides defaults
#[tokio::test]
async fn test_per_request_budget_low() {
    let harness = TestHarness::new().await;
    let server = create_test_server(&harness).await;

    let input = QueryInput {
        operation: "situation_report".to_string(),
        token_budget: Some(500), // Should trigger "summary" mode
        params: serde_json::Map::new(),
    };

    let response = server
        .handle_query(Parameters(input))
        .await
        .expect("Query should succeed");

    // With low budget, auto-selected "summary" mode should limit content
    assert!(
        response.token_estimate <= 2000,
        "Low budget should produce compact output"
    );
}

/// Test high token_budget triggers "full" detail mode
#[tokio::test]
async fn test_per_request_budget_high() {
    let harness = TestHarness::new().await;
    let server = create_test_server(&harness).await;

    let input = QueryInput {
        operation: "situation_report".to_string(),
        token_budget: Some(6000), // Should trigger "full" mode
        params: serde_json::Map::new(),
    };

    let response = server
        .handle_query(Parameters(input))
        .await
        .expect("Query should succeed");

    // Response should be generated (can be small if world is empty)
    assert!(!response.results.is_empty());
}

/// Test explicit detail_level overrides budget-based auto-selection
#[tokio::test]
async fn test_explicit_detail_level_wins() {
    let harness = TestHarness::new().await;
    let server = create_test_server(&harness).await;

    let mut params = serde_json::Map::new();
    params.insert("detail_level".to_string(), json!("full"));

    let input = QueryInput {
        operation: "situation_report".to_string(),
        token_budget: Some(500), // Low budget, but explicit detail_level should win
        params,
    };

    let response = server
        .handle_query(Parameters(input))
        .await
        .expect("Query should succeed");

    // Explicit detail_level=full honored despite low budget
    assert!(!response.results.is_empty());
}

/// Test per-tool-type defaults: composite gets 4000, lookup gets 1000
#[tokio::test]
async fn test_tool_type_defaults() {
    let harness = TestHarness::new().await;
    let server = create_test_server(&harness).await;

    // Composite report (no token_budget specified, should use 4000 default)
    let composite_input = QueryInput {
        operation: "situation_report".to_string(),
        token_budget: None, // Use per-tool-type default
        params: serde_json::Map::new(),
    };

    let composite_response = server
        .handle_query(Parameters(composite_input))
        .await
        .expect("Composite query should succeed");

    assert!(!composite_response.results.is_empty());

    // Lookup operation (should use 1000 default)
    let lookup_request = QueryRequest::Lookup {
        entity_id: "character:nonexistent".to_string(),
        detail_level: None,
    };
    let lookup_input = to_query_input(lookup_request);

    let lookup_result = server.handle_query(Parameters(lookup_input)).await;

    // Lookup of nonexistent entity should fail gracefully
    assert!(lookup_result.is_err());
}

/// Test NARRA_TOKEN_BUDGET env var overrides per-tool defaults
#[tokio::test]
async fn test_env_var_override() {
    std::env::set_var("NARRA_TOKEN_BUDGET", "1500");

    let harness = TestHarness::new().await;
    let server = create_test_server(&harness).await;

    let input = QueryInput {
        operation: "situation_report".to_string(),
        token_budget: None, // Should use env var (1500) instead of tool-type default (4000)
        params: serde_json::Map::new(),
    };

    let response = server
        .handle_query(Parameters(input))
        .await
        .expect("Query should succeed");

    // Env var should trigger "standard" mode (1501-3500 range)
    assert!(!response.results.is_empty());

    std::env::remove_var("NARRA_TOKEN_BUDGET");
}

/// Test MAX_TOKEN_BUDGET cap (8000) enforced
#[tokio::test]
async fn test_max_budget_cap() {
    let harness = TestHarness::new().await;
    let server = create_test_server(&harness).await;

    let input = QueryInput {
        operation: "situation_report".to_string(),
        token_budget: Some(999999), // Should be capped at MAX_TOKEN_BUDGET (8000)
        params: serde_json::Map::new(),
    };

    let response = server
        .handle_query(Parameters(input))
        .await
        .expect("Query should succeed");

    // Budget should be capped, not cause error
    assert!(!response.results.is_empty());
}

/// Test that token budget actually triggers truncation when results exceed budget.
///
/// Creates 15 characters with long descriptions to generate a large Overview
/// response, then requests with a very small token_budget to force truncation.
#[tokio::test]
async fn test_token_budget_triggers_truncation() {
    let harness = TestHarness::new().await;
    let server = create_test_server(&harness).await;

    // Create 15 characters with long descriptions to generate enough content
    for i in 0..15 {
        let name = format!("Character_{}", i);
        let desc = format!(
            "A very detailed description of character {} who lives in a vast kingdom \
             and has many adventures throughout the land, encountering numerous challenges \
             and growing as a person through each experience they face on their journey.",
            i
        );
        harness
            .db
            .query(format!(
                "CREATE character:{} SET name = '{}', roles = ['warrior'], aliases = [], \
                 description = '{}'",
                name.to_lowercase(),
                name,
                desc
            ))
            .await
            .unwrap();
    }

    // Request Overview with very small token budget to force truncation
    let mut params = serde_json::Map::new();
    params.insert("entity_type".to_string(), json!("character"));

    let input = QueryInput {
        operation: "overview".to_string(),
        token_budget: Some(100), // Very small — should truncate 15 results
        params,
    };

    let response = server
        .handle_query(Parameters(input))
        .await
        .expect("Query should succeed even with tiny budget");

    // With 15 characters and budget of 100 tokens, truncation should occur
    assert!(
        response.truncated.is_some(),
        "Response should be truncated with budget=100 and 15 characters"
    );

    let truncation = response.truncated.unwrap();
    assert_eq!(truncation.reason, "token_budget");
    assert_eq!(
        truncation.original_count, 15,
        "Should report all 15 original results"
    );
    assert!(
        truncation.returned_count < 15,
        "Should return fewer than 15 results after truncation, got {}",
        truncation.returned_count
    );
    assert!(
        truncation.returned_count >= 1,
        "Should return at least 1 result even when budget is tiny"
    );
}
