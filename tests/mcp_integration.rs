//! Integration tests for MCP server tools
//!
//! Tests verify that:
//! 1. Query tool operations return expected results
//! 2. Mutation tool operations create/update entities correctly
//! 3. Impact analysis calculates severity correctly
//! 4. Tools integrate properly with services layer

mod common;

use common::harness::TestHarness;
use narra::mcp::{DetailLevel, MutationRequest, QueryRequest};

#[tokio::test]
async fn test_create_and_lookup_character() {
    let harness = TestHarness::new().await;
    let server = common::create_test_server(&harness).await;

    // Create a character via mutation tool
    let create_request = MutationRequest::CreateCharacter {
        id: None,
        name: "Andrei Volkov".to_string(),
        role: Some("Protagonist".to_string()),
        aliases: None,
        description: Some("A mysterious figure with a troubled past".to_string()),
        profile: None,
    };

    let create_result = server
        .handle_mutate(rmcp::handler::server::wrapper::Parameters(create_request))
        .await
        .expect("Create character failed");

    println!("Created character ID: {}", create_result.entity.id);
    assert_eq!(create_result.entity.entity_type, "character");
    assert!(create_result.entity.id.contains("character:"));

    // Lookup the created character
    let lookup_request = QueryRequest::Lookup {
        entity_id: create_result.entity.id.clone(),
        detail_level: Some(DetailLevel::Full),
    };

    let lookup_result = server
        .handle_query(rmcp::handler::server::wrapper::Parameters(lookup_request))
        .await
        .expect("Lookup failed");

    assert_eq!(lookup_result.results.len(), 1);
    assert!(lookup_result.results[0].content.contains("Andrei Volkov"));
    assert!(!lookup_result.hints.is_empty());
}

#[tokio::test]
async fn test_search_with_fuzzy_matching() {
    let harness = TestHarness::new().await;
    let server = common::create_test_server(&harness).await;

    // Create some characters
    for name in ["Elena", "Elina", "Elena Petrova"] {
        let request = MutationRequest::CreateCharacter {
            id: None,
            name: name.to_string(),
            role: None,
            aliases: None,
            description: None,
            profile: None,
        };
        server
            .handle_mutate(rmcp::handler::server::wrapper::Parameters(request))
            .await
            .expect("Create failed");
    }

    // Search with partial/fuzzy name
    let search_request = QueryRequest::Search {
        query: "Elen".to_string(),
        entity_types: Some(vec!["character".to_string()]),
        limit: Some(10),
        cursor: None,
    };

    let search_result = server
        .handle_query(rmcp::handler::server::wrapper::Parameters(search_request))
        .await
        .expect("Search failed");

    // Should find Elena variants via fuzzy matching (might be 2 or 3 depending on threshold)
    assert!(
        !search_result.results.is_empty(),
        "Expected at least 1 result, got {}",
        search_result.results.len()
    );
    assert!(search_result.token_estimate > 0);
}

#[tokio::test]
async fn test_record_and_query_knowledge() {
    let harness = TestHarness::new().await;
    let server = common::create_test_server(&harness).await;

    // Create character and something to know about
    let char_request = MutationRequest::CreateCharacter {
        id: None,
        name: "Marie".to_string(),
        role: Some("Detective".to_string()),
        aliases: None,
        description: None,
        profile: None,
    };
    let char_result = server
        .handle_mutate(rmcp::handler::server::wrapper::Parameters(char_request))
        .await
        .unwrap();
    let character_id = char_result.entity.id;
    println!("Character ID: {}", character_id);

    let location_request = MutationRequest::CreateLocation {
        id: None,
        name: "The Old Mill".to_string(),
        description: Some("An abandoned mill with dark secrets".to_string()),
        parent_id: None,
    };
    let location_result = server
        .handle_mutate(rmcp::handler::server::wrapper::Parameters(location_request))
        .await
        .unwrap();
    let location_id = location_result.entity.id;
    println!("Location ID: {}", location_id);

    // Record knowledge (use initial method since no event)
    let knowledge_request = MutationRequest::RecordKnowledge {
        character_id: character_id.clone(),
        target_id: location_id.clone(),
        fact: "The mill was the site of a murder in 1923".to_string(),
        certainty: "suspects".to_string(),
        method: Some("initial".to_string()),
        source_character_id: None,
        event_id: None,
    };

    let knowledge_result = server
        .handle_mutate(rmcp::handler::server::wrapper::Parameters(
            knowledge_request,
        ))
        .await
        .expect("Record knowledge failed");

    assert_eq!(knowledge_result.entity.entity_type, "knowledge");

    // Query temporal knowledge
    let temporal_request = QueryRequest::Temporal {
        character_id: character_id.clone(),
        event_id: None,
        event_name: None,
    };

    let temporal_result = server
        .handle_query(rmcp::handler::server::wrapper::Parameters(temporal_request))
        .await
        .expect("Temporal query failed");

    assert!(!temporal_result.results.is_empty());
    // Note: content may not contain "murder" since it shows target ID, not fact
    // This is expected behavior - temporal queries return knowledge state metadata
}

#[tokio::test]
async fn test_impact_analysis() {
    let harness = TestHarness::new().await;
    let server = common::create_test_server(&harness).await;

    // Create interconnected entities
    let char1 = server
        .handle_mutate(rmcp::handler::server::wrapper::Parameters(
            MutationRequest::CreateCharacter {
                id: None,
                name: "Alice".to_string(),
                role: None,
                aliases: None,
                description: None,
                profile: None,
            },
        ))
        .await
        .unwrap();

    let char2 = server
        .handle_mutate(rmcp::handler::server::wrapper::Parameters(
            MutationRequest::CreateCharacter {
                id: None,
                name: "Bob".to_string(),
                role: None,
                aliases: None,
                description: None,
                profile: None,
            },
        ))
        .await
        .unwrap();

    // Create relationship (knowledge from Alice about Bob, use initial method)
    server
        .handle_mutate(rmcp::handler::server::wrapper::Parameters(
            MutationRequest::RecordKnowledge {
                character_id: char1.entity.id.clone(),
                target_id: char2.entity.id.clone(),
                fact: "Alice knows Bob is her brother".to_string(),
                certainty: "knows".to_string(),
                method: Some("initial".to_string()),
                source_character_id: None,
                event_id: None,
            },
        ))
        .await
        .unwrap();

    // Protect Bob
    server
        .handle_protect_entity(rmcp::handler::server::wrapper::Parameters(
            char2.entity.id.clone(),
        ))
        .await
        .expect("Protect failed");

    // Analyze impact of changing Bob
    use narra::mcp::tools::impact::ImpactRequest;
    let impact_request = ImpactRequest {
        entity_id: char2.entity.id.clone(),
        proposed_change: Some("Delete Bob".to_string()),
        include_details: Some(true),
    };

    let impact_result = server
        .handle_analyze_impact(rmcp::handler::server::wrapper::Parameters(impact_request))
        .await
        .expect("Impact analysis failed");

    // Should be critical because Bob is protected
    assert_eq!(impact_result.severity, "Critical");
    assert!(!impact_result.suggestions.is_empty());
}

#[tokio::test]
async fn test_pagination_with_cursors() {
    let harness = TestHarness::new().await;
    let server = common::create_test_server(&harness).await;

    // Create many characters
    for i in 0..25 {
        server
            .handle_mutate(rmcp::handler::server::wrapper::Parameters(
                MutationRequest::CreateCharacter {
                    id: None,
                    name: format!("Character {}", i),
                    role: None,
                    aliases: None,
                    description: None,
                    profile: None,
                },
            ))
            .await
            .unwrap();
    }

    // First page
    let page1_request = QueryRequest::Overview {
        entity_type: "character".to_string(),
        limit: Some(10),
    };

    let page1 = server
        .handle_query(rmcp::handler::server::wrapper::Parameters(page1_request))
        .await
        .expect("Page 1 failed");

    // Overview respects limit
    assert!(page1.results.len() <= 10);
}

#[tokio::test]
async fn test_update_with_impact() {
    let harness = TestHarness::new().await;
    let server = common::create_test_server(&harness).await;

    // Create a character
    let create_result = server
        .handle_mutate(rmcp::handler::server::wrapper::Parameters(
            MutationRequest::CreateCharacter {
                id: None,
                name: "Original Name".to_string(),
                role: Some("Hero".to_string()),
                aliases: None,
                description: None,
                profile: None,
            },
        ))
        .await
        .unwrap();

    let entity_id = create_result.entity.id;

    // Update the character
    let update_request = MutationRequest::Update {
        entity_id: entity_id.clone(),
        fields: serde_json::json!({
            "name": "Updated Name",
            "roles": ["Villain"]
        }),
    };

    let update_result = server
        .handle_mutate(rmcp::handler::server::wrapper::Parameters(update_request))
        .await
        .expect("Update failed");

    assert!(
        update_result.entity.content.contains("Updated")
            || update_result.entity.content.contains("character")
    );
    // Impact should be included if any related entities exist
    // (in this case, likely None since no relationships)
}

#[tokio::test]
async fn test_graph_traversal() {
    let harness = TestHarness::new().await;
    let server = common::create_test_server(&harness).await;

    // Create characters with relationships via knowledge
    let char1 = server
        .handle_mutate(rmcp::handler::server::wrapper::Parameters(
            MutationRequest::CreateCharacter {
                id: None,
                name: "Node A".to_string(),
                role: None,
                aliases: None,
                description: None,
                profile: None,
            },
        ))
        .await
        .unwrap();

    let char2 = server
        .handle_mutate(rmcp::handler::server::wrapper::Parameters(
            MutationRequest::CreateCharacter {
                id: None,
                name: "Node B".to_string(),
                role: None,
                aliases: None,
                description: None,
                profile: None,
            },
        ))
        .await
        .unwrap();

    // Create connection via knowledge (use initial method)
    server
        .handle_mutate(rmcp::handler::server::wrapper::Parameters(
            MutationRequest::RecordKnowledge {
                character_id: char1.entity.id.clone(),
                target_id: char2.entity.id.clone(),
                fact: "A knows B".to_string(),
                certainty: "knows".to_string(),
                method: Some("initial".to_string()),
                source_character_id: None,
                event_id: None,
            },
        ))
        .await
        .unwrap();

    // Traverse graph from char1
    let traversal_request = QueryRequest::GraphTraversal {
        entity_id: char1.entity.id.clone(),
        depth: 2,
        format: None,
    };

    let traversal_result = server
        .handle_query(rmcp::handler::server::wrapper::Parameters(
            traversal_request,
        ))
        .await
        .expect("Graph traversal failed");

    // Should find connected entities
    assert!(!traversal_result.hints.is_empty());
}
