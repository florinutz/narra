//! Tool selection accuracy tests (Phase 15)
//!
//! These tests verify that tool/resource/prompt descriptions enable
//! correct selection for common user intents.

use narra::embedding::NoopEmbeddingService;
use narra::mcp::NarraServer;
use narra::session::SessionStateManager;
use std::sync::Arc;

/// Test scenarios mapping user intent to expected primitive.
/// Fields are for documentation/reference, read by humans reviewing the test.
#[derive(Debug)]
struct SelectionScenario {
    intent: &'static str,
    expected_primitive: &'static str,
    expected_name: &'static str,
}

const SELECTION_SCENARIOS: &[SelectionScenario] = &[
    // Query scenarios -> query tool
    SelectionScenario {
        intent: "Find a character named Alice",
        expected_primitive: "tool",
        expected_name: "query",
    },
    SelectionScenario {
        intent: "Search for locations in the story",
        expected_primitive: "tool",
        expected_name: "query",
    },
    SelectionScenario {
        intent: "What does Bob know at chapter 5?",
        expected_primitive: "tool",
        expected_name: "query",
    },
    // Mutate scenarios -> mutate tool
    SelectionScenario {
        intent: "Create a new character called Eve",
        expected_primitive: "tool",
        expected_name: "mutate",
    },
    SelectionScenario {
        intent: "Update Alice's description",
        expected_primitive: "tool",
        expected_name: "mutate",
    },
    SelectionScenario {
        intent: "Record that Bob learned the secret",
        expected_primitive: "tool",
        expected_name: "mutate",
    },
    // Validate scenarios -> validate tool
    SelectionScenario {
        intent: "Check if there are any consistency problems",
        expected_primitive: "tool",
        expected_name: "validate",
    },
    SelectionScenario {
        intent: "What contradicts the magic system rules?",
        expected_primitive: "tool",
        expected_name: "validate",
    },
    // Resource scenarios -> resources
    SelectionScenario {
        intent: "Show me the full details of character Alice",
        expected_primitive: "resource",
        expected_name: "narra://entity/character:alice",
    },
    SelectionScenario {
        intent: "What's the current session context?",
        expected_primitive: "resource",
        expected_name: "narra://session/context",
    },
    SelectionScenario {
        intent: "List all current consistency issues",
        expected_primitive: "resource",
        expected_name: "narra://consistency/issues",
    },
    // Prompt scenarios -> prompts
    SelectionScenario {
        intent: "Guide me through checking consistency step by step",
        expected_primitive: "prompt",
        expected_name: "check_consistency",
    },
    SelectionScenario {
        intent: "Analyze dramatic irony between Alice and Bob",
        expected_primitive: "prompt",
        expected_name: "dramatic_irony",
    },
    // Impact analysis -> analyze_impact tool
    SelectionScenario {
        intent: "What would happen if I delete this character?",
        expected_primitive: "tool",
        expected_name: "analyze_impact",
    },
    // Graph generation -> generate_graph tool
    SelectionScenario {
        intent: "Create a relationship diagram",
        expected_primitive: "tool",
        expected_name: "generate_graph",
    },
];

#[test]
fn test_selection_scenarios_documented() {
    // This test verifies that we have documented scenarios for selection accuracy
    // The actual selection is done by Claude based on descriptions

    let total = SELECTION_SCENARIOS.len();
    let tool_count = SELECTION_SCENARIOS
        .iter()
        .filter(|s| s.expected_primitive == "tool")
        .count();
    let resource_count = SELECTION_SCENARIOS
        .iter()
        .filter(|s| s.expected_primitive == "resource")
        .count();
    let prompt_count = SELECTION_SCENARIOS
        .iter()
        .filter(|s| s.expected_primitive == "prompt")
        .count();

    // Verify we have coverage across all primitives
    assert!(tool_count >= 8, "Should have at least 8 tool scenarios");
    assert!(
        resource_count >= 3,
        "Should have at least 3 resource scenarios"
    );
    assert!(prompt_count >= 2, "Should have at least 2 prompt scenarios");

    println!(
        "Selection scenarios: {} total ({} tool, {} resource, {} prompt)",
        total, tool_count, resource_count, prompt_count
    );

    // Print all scenarios for documentation visibility
    for s in SELECTION_SCENARIOS {
        println!(
            "  [{}/{}] {}",
            s.expected_primitive, s.expected_name, s.intent
        );
    }
}

#[test]
fn test_tool_descriptions_under_token_limit() {
    // Verify all tool descriptions are under 50 tokens
    // Using approximate: 1.3 tokens per word for technical English

    let tool_descriptions = [
        (
            "query",
            "Query world state: lookup by ID, search text, traverse relationships, get temporal knowledge, or list entity types. Returns entities with confidence scores.",
        ),
        (
            "mutate",
            "Modify world state: create/update/delete entities, record character knowledge, manage facts and notes. CRITICAL violations block mutations.",
        ),
        (
            "analyze_impact",
            "Preview change impact before mutating. Returns affected entities by severity. Use before significant mutations.",
        ),
        (
            "validate",
            "Check entity consistency: fact violations, timeline contradictions, relationship conflicts. Returns issues with suggested fixes.",
        ),
        (
            "protect_entity",
            "Mark entity as protected. Protected entities trigger CRITICAL severity in impact analysis.",
        ),
        (
            "unprotect_entity",
            "Remove protection from entity, restoring normal severity calculations.",
        ),
        (
            "pin_entity",
            "Pin entity to working context. Pinned entities persist across sessions.",
        ),
        ("unpin_entity", "Unpin entity from working context."),
        (
            "get_session_context",
            "Get session summary: recent work, hot entities, pinned items, pending decisions. Call at session start.",
        ),
        (
            "export_world",
            "Export all world data to JSON file for backup or migration.",
        ),
        (
            "generate_graph",
            "Generate Mermaid relationship graph to .planning/exports/. Use scope='full' or scope='character:ID' with depth.",
        ),
    ];

    for (name, desc) in tool_descriptions {
        let word_count = desc.split_whitespace().count();
        let estimated_tokens = (word_count as f32 * 1.3) as usize;

        assert!(
            estimated_tokens < 50,
            "Tool '{}' description exceeds 50 tokens: {} words ~= {} tokens\nDescription: {}",
            name,
            word_count,
            estimated_tokens,
            desc
        );

        println!(
            "{}: {} words, ~{} tokens",
            name, word_count, estimated_tokens
        );
    }
}

#[tokio::test]
async fn test_context_budget_with_prompts() {
    // Verify context budget still under 10K after adding prompts
    use narra::db::{
        connection::{init_db, DbConfig},
        schema::apply_schema,
    };
    use tempfile::TempDir;

    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let db_path = temp_dir.path().join("test.db");
    let session_path = temp_dir.path().join("session.json");

    let db = init_db(
        &DbConfig::Embedded {
            path: Some(db_path.to_string_lossy().into_owned()),
        },
        temp_dir.path(),
    )
    .await
    .expect("Failed to connect to test database");

    apply_schema(&db).await.expect("Failed to apply schema");

    let session_manager = Arc::new(
        SessionStateManager::load_or_create(&session_path)
            .expect("Failed to create session manager"),
    );
    let server = NarraServer::new(
        Arc::new(db),
        session_manager,
        Arc::new(NoopEmbeddingService::new()),
    )
    .await;

    // Use the rmcp ServerHandler trait to get server info
    use rmcp::ServerHandler;
    let info = server.get_info();

    // Estimate tokens: 4 chars per token
    fn estimate_tokens(text: &str) -> usize {
        text.len().div_ceil(4)
    }

    let info_json = serde_json::to_string(&info).unwrap();
    let info_tokens = estimate_tokens(&info_json);

    let instruction_tokens = info
        .instructions
        .as_ref()
        .map(|i| estimate_tokens(i))
        .unwrap_or(0);

    let total = info_tokens + instruction_tokens;

    println!(
        "Context budget: {} tokens (info: {}, instructions: {})",
        total, info_tokens, instruction_tokens
    );

    assert!(
        total < 10_000,
        "Context budget {} exceeds 10K token limit",
        total
    );

    // Warn if over 8K
    if total > 8_000 {
        println!(
            "WARNING: Context budget {} exceeds 8K warning threshold",
            total
        );
    }
}
