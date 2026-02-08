//! Integration tests for Cross-Engine Queries (Phase 2).
//!
//! Tests verify:
//! - SemanticKnowledge: KNN search on knowledge embeddings with optional character filter
//! - SemanticGraphSearch: graph traversal + cosine similarity ranking
//! - Knowledge composite text generation (certainty-aware)
//! - Error paths when embedding service unavailable

mod common;

use std::sync::Arc;

use common::harness::TestHarness;
use narra::embedding::backfill::BackfillService;
use narra::embedding::composite::knowledge_composite;
use narra::embedding::{EmbeddingConfig, LocalEmbeddingService, NoopEmbeddingService};
use narra::mcp::{MutationRequest, NarraServer, QueryRequest};
use narra::models::character::{create_character, CharacterCreate};
use narra::models::event::{create_event, EventCreate};
use narra::session::SessionStateManager;
use rmcp::handler::server::wrapper::Parameters;
use std::collections::HashMap;

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

/// Helper to create NarraServer with real embedding service.
async fn create_test_server_with_embeddings(
    harness: &TestHarness,
    embedding_service: Arc<LocalEmbeddingService>,
) -> NarraServer {
    let session_path = harness.temp_path().join("session.json");
    let session_manager = Arc::new(
        SessionStateManager::load_or_create(&session_path)
            .expect("Failed to create session manager"),
    );
    NarraServer::new(harness.db.clone(), session_manager, embedding_service).await
}

// =============================================================================
// KNOWLEDGE COMPOSITE TEXT TESTS (no embedding model needed)
// =============================================================================

/// Test knowledge_composite generates certainty-aware text.
#[test]
fn test_knowledge_composite_knows() {
    let text = knowledge_composite(
        "the treasure is hidden under the oak tree",
        "Alice",
        "knows",
        None,
    );
    assert!(
        text.contains("Alice") && text.contains("knows"),
        "Should include character name and certainty: {}",
        text
    );
    assert!(
        text.contains("treasure") || text.contains("oak tree"),
        "Should include the fact content: {}",
        text
    );
}

#[test]
fn test_knowledge_composite_suspects() {
    let text = knowledge_composite("Bob is the traitor", "Alice", "suspects", None);
    assert!(
        text.contains("suspects"),
        "Should reflect 'suspects' certainty: {}",
        text
    );
}

#[test]
fn test_knowledge_composite_suspects_with_method() {
    let text = knowledge_composite(
        "the curse can be broken",
        "Elena",
        "suspects",
        Some("overheard"),
    );
    assert!(
        text.contains("suspects"),
        "Should reflect 'suspects' certainty: {}",
        text
    );
    assert!(
        text.contains("overheard"),
        "Should include learning method: {}",
        text
    );
}

#[test]
fn test_knowledge_composite_unknown_certainty_defaults_to_knows() {
    // "believes" is not an explicit match arm, falls through to default "knows"
    let text = knowledge_composite("the curse can be broken", "Elena", "believes", None);
    assert!(
        text.contains("knows"),
        "Unknown certainty should default to 'knows': {}",
        text
    );
}

// =============================================================================
// SEMANTIC KNOWLEDGE — ERROR PATHS (no embedding model needed)
// =============================================================================

/// Test SemanticKnowledge errors when embedding service unavailable.
#[tokio::test]
async fn test_semantic_knowledge_no_embedding_service() {
    let harness = TestHarness::new().await;
    let server = create_test_server(&harness).await;

    let request = QueryRequest::SemanticKnowledge {
        query: "the royal succession".to_string(),
        character_id: None,
        limit: None,
    };
    let result = server.handle_query(Parameters(request)).await;
    assert!(result.is_err(), "Should error without embedding service");
    assert!(
        result.unwrap_err().contains("unavailable"),
        "Error should mention unavailability"
    );
}

/// Test SemanticGraphSearch errors when embedding service unavailable.
#[tokio::test]
async fn test_semantic_graph_search_no_embedding_service() {
    let harness = TestHarness::new().await;
    let server = create_test_server(&harness).await;

    let request = QueryRequest::SemanticGraphSearch {
        entity_id: "character:alice".to_string(),
        max_hops: None,
        query: "betrayal".to_string(),
        entity_types: None,
        limit: None,
    };
    let result = server.handle_query(Parameters(request)).await;
    assert!(result.is_err(), "Should error without embedding service");
    assert!(
        result.unwrap_err().contains("unavailable"),
        "Error should mention unavailability"
    );
}

// =============================================================================
// SEMANTIC KNOWLEDGE — INTEGRATION (require fastembed)
// =============================================================================

/// Test SemanticKnowledge finds knowledge by meaning.
#[tokio::test]
#[ignore = "Requires fastembed model download (run manually with --ignored)"]
async fn test_semantic_knowledge_finds_by_meaning() {
    let harness = TestHarness::new().await;
    let embedding_service =
        Arc::new(LocalEmbeddingService::new(EmbeddingConfig::default()).unwrap());
    let backfill_service = BackfillService::new(harness.db.clone(), embedding_service.clone());
    let server = create_test_server_with_embeddings(&harness, embedding_service.clone()).await;

    // Create characters
    let alice = create_character(
        &harness.db,
        CharacterCreate {
            name: "Alice".into(),
            aliases: vec![],
            roles: vec!["protagonist".into()],
            ..Default::default()
        },
    )
    .await
    .expect("Should create Alice");

    let bob = create_character(
        &harness.db,
        CharacterCreate {
            name: "Bob".into(),
            aliases: vec![],
            roles: vec!["companion".into()],
            ..Default::default()
        },
    )
    .await
    .expect("Should create Bob");

    // Record knowledge facts via mutations
    let facts = vec![
        (
            alice.id.to_string(),
            "character:alice",
            "The ancient sword Excalibur is hidden in the lake of Avalon",
        ),
        (
            alice.id.to_string(),
            "character:alice",
            "The dragon guarding the northern pass is actually a polymorphed wizard",
        ),
        (
            bob.id.to_string(),
            "character:bob",
            "The royal succession passes through the female line in this kingdom",
        ),
        (
            bob.id.to_string(),
            "character:bob",
            "Trade routes through the mountain pass are controlled by bandits",
        ),
    ];

    for (char_id, target_id, fact) in &facts {
        let request = MutationRequest::RecordKnowledge {
            character_id: char_id.clone(),
            target_id: target_id.to_string(),
            fact: fact.to_string(),
            certainty: "knows".to_string(),
            method: Some("initial".to_string()),
            source_character_id: None,
            event_id: None,
        };
        server
            .handle_mutate(Parameters(request))
            .await
            .expect("RecordKnowledge should succeed");
    }

    // Backfill embeddings synchronously (tokio::spawn doesn't reliably complete in test runtime)
    backfill_service
        .backfill_all()
        .await
        .expect("Backfill should succeed");

    // Search for sword/weapon-related knowledge
    let request = QueryRequest::SemanticKnowledge {
        query: "magical weapons and legendary swords".to_string(),
        character_id: None,
        limit: Some(5),
    };
    let response = server
        .handle_query(Parameters(request))
        .await
        .expect("SemanticKnowledge should succeed");

    assert!(
        !response.results.is_empty(),
        "Should find knowledge about weapons/swords"
    );

    // The Excalibur fact should rank high
    let top_result = &response.results[0];
    assert!(
        top_result.content.contains("Excalibur") || top_result.content.contains("sword"),
        "Top result should be about Excalibur, got: {}",
        top_result.content
    );
    assert!(
        top_result.confidence.unwrap() > 0.0,
        "Should have positive similarity score"
    );
}

/// Test SemanticKnowledge filters by character.
#[tokio::test]
#[ignore = "Requires fastembed model download (run manually with --ignored)"]
async fn test_semantic_knowledge_character_filter() {
    let harness = TestHarness::new().await;
    let embedding_service =
        Arc::new(LocalEmbeddingService::new(EmbeddingConfig::default()).unwrap());
    let backfill_service = BackfillService::new(harness.db.clone(), embedding_service.clone());
    let server = create_test_server_with_embeddings(&harness, embedding_service.clone()).await;

    let alice = create_character(
        &harness.db,
        CharacterCreate {
            name: "Alice".into(),
            aliases: vec![],
            roles: vec!["protagonist".into()],
            ..Default::default()
        },
    )
    .await
    .expect("Should create Alice");

    let bob = create_character(
        &harness.db,
        CharacterCreate {
            name: "Bob".into(),
            aliases: vec![],
            roles: vec!["companion".into()],
            ..Default::default()
        },
    )
    .await
    .expect("Should create Bob");

    // Give Alice and Bob different knowledge
    let alice_request = MutationRequest::RecordKnowledge {
        character_id: alice.id.to_string(),
        target_id: "character:alice".to_string(),
        fact: "The dungeon entrance is concealed behind the waterfall".to_string(),
        certainty: "knows".to_string(),
        method: Some("initial".to_string()),
        source_character_id: None,
        event_id: None,
    };
    server
        .handle_mutate(Parameters(alice_request))
        .await
        .expect("Should record Alice's knowledge");

    let bob_request = MutationRequest::RecordKnowledge {
        character_id: bob.id.to_string(),
        target_id: "character:bob".to_string(),
        fact: "The secret passage leads underground to the dungeon".to_string(),
        certainty: "knows".to_string(),
        method: Some("initial".to_string()),
        source_character_id: None,
        event_id: None,
    };
    server
        .handle_mutate(Parameters(bob_request))
        .await
        .expect("Should record Bob's knowledge");

    // Clear any embeddings set by async spawn_regeneration — these may not be indexed
    // in the HNSW index. Force synchronous re-backfill for reliable HNSW indexing.
    harness
        .db
        .query("UPDATE knowledge SET embedding = NONE, embedding_stale = true")
        .await
        .expect("Clear knowledge embeddings");
    backfill_service
        .backfill_all()
        .await
        .expect("Backfill should succeed");

    // Search filtered to Alice only
    let request = QueryRequest::SemanticKnowledge {
        query: "hidden dungeon entrances".to_string(),
        character_id: Some(alice.id.to_string()),
        limit: Some(10),
    };
    let response = server
        .handle_query(Parameters(request))
        .await
        .expect("Filtered SemanticKnowledge should succeed");

    // Should only return Alice's knowledge
    for result in &response.results {
        assert!(
            result.content.contains("Alice") || result.content.contains("waterfall"),
            "Filtered results should only include Alice's knowledge, got: {}",
            result.content
        );
    }

    // Hints should mention the filter
    let hints_joined = response.hints.join(" ");
    assert!(
        hints_joined.contains("character") || hints_joined.contains("filter"),
        "Hints should mention character filter"
    );
}

/// Test SemanticKnowledge with no knowledge returns empty results.
#[tokio::test]
#[ignore = "Requires fastembed model download (run manually with --ignored)"]
async fn test_semantic_knowledge_empty_results() {
    let harness = TestHarness::new().await;
    let embedding_service =
        Arc::new(LocalEmbeddingService::new(EmbeddingConfig::default()).unwrap());
    let server = create_test_server_with_embeddings(&harness, embedding_service.clone()).await;

    let request = QueryRequest::SemanticKnowledge {
        query: "anything at all".to_string(),
        character_id: None,
        limit: None,
    };
    let response = server
        .handle_query(Parameters(request))
        .await
        .expect("Should return empty, not error");

    assert_eq!(response.results.len(), 0, "Should have no results");
    assert!(
        response.hints.iter().any(|h| h.contains("No knowledge")),
        "Should hint about no matches"
    );
}

// =============================================================================
// SEMANTIC GRAPH SEARCH — INTEGRATION (require fastembed)
// =============================================================================

/// Test SemanticGraphSearch finds connected entities ranked by meaning.
#[tokio::test]
#[ignore = "Requires fastembed model download (run manually with --ignored)"]
async fn test_semantic_graph_search_ranks_connected_entities() {
    let harness = TestHarness::new().await;
    let embedding_service =
        Arc::new(LocalEmbeddingService::new(EmbeddingConfig::default()).unwrap());
    let backfill_service = BackfillService::new(harness.db.clone(), embedding_service.clone());
    let server = create_test_server_with_embeddings(&harness, embedding_service.clone()).await;

    // Create a network of characters
    let alice = create_character(
        &harness.db,
        CharacterCreate {
            name: "Alice the Brave Knight".into(),
            aliases: vec![],
            roles: vec!["knight".into(), "protagonist".into()],
            profile: HashMap::from([
                ("wound".into(), vec!["Watched her commander fall in battle against the dark army at age 20. A true warrior never abandons their comrades. Charges headfirst into danger to protect others.".into()]),
            ]),
        },
    )
    .await
    .expect("Should create Alice");

    let bob = create_character(
        &harness.db,
        CharacterCreate {
            name: "Bob the Cunning Spy".into(),
            aliases: vec!["The Shadow".into()],
            roles: vec!["spy".into(), "informant".into()],
            profile: HashMap::from([
                ("wound".into(), vec!["Discovered his entire identity was fabricated by the guild at age 15. Deception is the only real power. Never reveals his true intentions.".into()]),
            ]),
        },
    )
    .await
    .expect("Should create Bob");

    let carol = create_character(
        &harness.db,
        CharacterCreate {
            name: "Carol the Healer Priestess".into(),
            aliases: vec![],
            roles: vec!["healer".into(), "priestess".into()],
            profile: HashMap::from([
                ("wound".into(), vec!["Failed to save a dying child despite her powers at age 25. Healing requires sacrifice. Overextends herself trying to save everyone.".into()]),
            ]),
        },
    )
    .await
    .expect("Should create Carol");

    // Backfill embeddings
    let stats = backfill_service
        .backfill_all()
        .await
        .expect("Backfill should succeed");
    assert!(stats.embedded >= 3, "Should embed at least 3 characters");

    // Create relationships: Alice <-> Bob, Alice <-> Carol
    // CreateRelationship expects bare keys (without "character:" prefix)
    let create_rel_ab = MutationRequest::CreateRelationship {
        from_character_id: alice.id.key().to_string(),
        to_character_id: bob.id.key().to_string(),
        rel_type: "professional".to_string(),
        subtype: Some("alliance".to_string()),
        label: Some("spy network".to_string()),
    };
    server
        .handle_mutate(Parameters(create_rel_ab))
        .await
        .expect("Should create Alice-Bob relationship");

    let create_rel_ac = MutationRequest::CreateRelationship {
        from_character_id: alice.id.key().to_string(),
        to_character_id: carol.id.key().to_string(),
        rel_type: "professional".to_string(),
        subtype: Some("comrades".to_string()),
        label: Some("battle companions".to_string()),
    };
    server
        .handle_mutate(Parameters(create_rel_ac))
        .await
        .expect("Should create Alice-Carol relationship");

    // SemanticGraphSearch from Alice about espionage/deception
    let request = QueryRequest::SemanticGraphSearch {
        entity_id: alice.id.to_string(),
        max_hops: Some(2),
        query: "espionage deception and covert operations".to_string(),
        entity_types: None,
        limit: Some(10),
    };
    let response = server
        .handle_query(Parameters(request))
        .await
        .expect("SemanticGraphSearch should succeed");

    assert!(
        !response.results.is_empty(),
        "Should find connected entities"
    );

    // Bob (spy/deception) should rank higher than Carol (healer) for this query
    let bob_id = bob.id.to_string();
    let carol_id = carol.id.to_string();

    let bob_result = response.results.iter().find(|r| r.id == bob_id);
    let carol_result = response.results.iter().find(|r| r.id == carol_id);

    assert!(
        bob_result.is_some(),
        "Bob the spy should be found for espionage query"
    );

    if let (Some(bob_res), Some(carol_res)) = (bob_result, carol_result) {
        assert!(
            bob_res.confidence.unwrap() > carol_res.confidence.unwrap(),
            "Bob (spy) should rank higher than Carol (healer) for espionage query: {} vs {}",
            bob_res.confidence.unwrap(),
            carol_res.confidence.unwrap()
        );
    }
}

/// Test SemanticGraphSearch with entity_types filter.
#[tokio::test]
#[ignore = "Requires fastembed model download (run manually with --ignored)"]
async fn test_semantic_graph_search_type_filter() {
    let harness = TestHarness::new().await;
    let embedding_service =
        Arc::new(LocalEmbeddingService::new(EmbeddingConfig::default()).unwrap());
    let backfill_service = BackfillService::new(harness.db.clone(), embedding_service.clone());
    let server = create_test_server_with_embeddings(&harness, embedding_service.clone()).await;

    let alice = create_character(
        &harness.db,
        CharacterCreate {
            name: "Alice".into(),
            aliases: vec![],
            roles: vec!["protagonist".into()],
            ..Default::default()
        },
    )
    .await
    .expect("Should create Alice");

    let bob = create_character(
        &harness.db,
        CharacterCreate {
            name: "Bob".into(),
            aliases: vec![],
            roles: vec!["companion".into()],
            ..Default::default()
        },
    )
    .await
    .expect("Should create Bob");

    // Backfill
    backfill_service
        .backfill_all()
        .await
        .expect("Backfill should succeed");

    // Create relationship (expects bare keys without "character:" prefix)
    let create_rel = MutationRequest::CreateRelationship {
        from_character_id: alice.id.key().to_string(),
        to_character_id: bob.id.key().to_string(),
        rel_type: "friendship".to_string(),
        subtype: None,
        label: None,
    };
    server
        .handle_mutate(Parameters(create_rel))
        .await
        .expect("Should create relationship");

    // Search with type filter for "character" only
    let request = QueryRequest::SemanticGraphSearch {
        entity_id: alice.id.to_string(),
        max_hops: Some(2),
        query: "companion".to_string(),
        entity_types: Some(vec!["character".to_string()]),
        limit: Some(10),
    };
    let response = server
        .handle_query(Parameters(request))
        .await
        .expect("Should succeed with type filter");

    // All results should be characters
    for result in &response.results {
        assert_eq!(
            result.entity_type, "character",
            "All results should be characters when filtered"
        );
    }
}

/// Test SemanticGraphSearch with no connections returns empty.
#[tokio::test]
#[ignore = "Requires fastembed model download (run manually with --ignored)"]
async fn test_semantic_graph_search_no_connections() {
    let harness = TestHarness::new().await;
    let embedding_service =
        Arc::new(LocalEmbeddingService::new(EmbeddingConfig::default()).unwrap());
    let server = create_test_server_with_embeddings(&harness, embedding_service.clone()).await;

    let alice = create_character(
        &harness.db,
        CharacterCreate {
            name: "Alice".into(),
            aliases: vec![],
            roles: vec!["loner".into()],
            ..Default::default()
        },
    )
    .await
    .expect("Should create Alice");

    let request = QueryRequest::SemanticGraphSearch {
        entity_id: alice.id.to_string(),
        max_hops: Some(2),
        query: "anything".to_string(),
        entity_types: None,
        limit: None,
    };
    let response = server
        .handle_query(Parameters(request))
        .await
        .expect("Should return empty, not error");

    assert_eq!(
        response.results.len(),
        0,
        "Isolated entity should have no graph results"
    );
    assert!(
        response.hints.iter().any(|h| h.contains("No connected")),
        "Should hint about no connections"
    );
}

// =============================================================================
// KNOWLEDGE EMBEDDING PIPELINE (require fastembed)
// =============================================================================

/// Test that RecordKnowledge generates embeddings for knowledge entities.
#[tokio::test]
#[ignore = "Requires fastembed model download (run manually with --ignored)"]
async fn test_record_knowledge_generates_embedding() {
    let harness = TestHarness::new().await;
    let embedding_service =
        Arc::new(LocalEmbeddingService::new(EmbeddingConfig::default()).unwrap());
    let server = create_test_server_with_embeddings(&harness, embedding_service.clone()).await;

    let alice = create_character(
        &harness.db,
        CharacterCreate {
            name: "Alice".into(),
            aliases: vec![],
            roles: vec!["protagonist".into()],
            ..Default::default()
        },
    )
    .await
    .expect("Should create Alice");

    let event = create_event(
        &harness.db,
        EventCreate {
            title: "The Prophecy Revealed".into(),
            description: Some("Alice hears the ancient prophecy".into()),
            sequence: 1,
            date: None,
            date_precision: None,
            duration_end: None,
        },
    )
    .await
    .expect("Should create event");

    // Record knowledge with event (non-initial methods require an event)
    let request = MutationRequest::RecordKnowledge {
        character_id: alice.id.to_string(),
        target_id: "character:alice".to_string(),
        fact: "The ancient prophecy speaks of a chosen one who will unite the kingdoms".to_string(),
        certainty: "believes".to_string(),
        method: Some("heard".to_string()),
        source_character_id: None,
        event_id: Some(event.id.to_string()),
    };
    server
        .handle_mutate(Parameters(request))
        .await
        .expect("RecordKnowledge should succeed");

    // Wait for background embedding generation
    tokio::time::sleep(std::time::Duration::from_secs(3)).await;

    // Verify knowledge has embedding
    #[derive(serde::Deserialize)]
    struct EmbCheck {
        embedding: Option<Vec<f32>>,
        embedding_stale: Option<bool>,
    }

    let mut resp = harness
        .db
        .query("SELECT embedding, embedding_stale FROM knowledge LIMIT 1")
        .await
        .expect("Query should succeed");
    let results: Vec<EmbCheck> = resp.take(0).unwrap_or_default();

    assert!(
        !results.is_empty(),
        "Should have at least one knowledge entry"
    );
    let knowledge = &results[0];
    assert!(
        knowledge.embedding.is_some(),
        "Knowledge should have embedding after RecordKnowledge"
    );
    assert_eq!(
        knowledge.embedding.as_ref().unwrap().len(),
        384,
        "Knowledge embedding should be 384-dimensional"
    );
    assert_eq!(
        knowledge.embedding_stale,
        Some(false),
        "Knowledge should not be stale after embedding"
    );
}
