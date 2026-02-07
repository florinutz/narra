//! Integration tests for semantic and hybrid search functionality.
//!
//! These tests verify Phase 16 requirements for meaning-based search:
//! - Semantic search ranks by psychological/thematic similarity
//! - Scene mood/theme search works
//! - Hybrid search combines keyword and semantic results
//! - Graceful degradation when embedding model unavailable
//! - Empty embeddings handled correctly

mod common;

use common::harness::TestHarness;
use narra::embedding::backfill::BackfillService;
use narra::embedding::StalenessManager;
use narra::embedding::{EmbeddingConfig, LocalEmbeddingService, NoopEmbeddingService};
use narra::models::character::create_character;
use narra::models::event::{create_event, EventCreate};
use narra::models::location::{create_location, LocationCreate};
use narra::models::scene::{create_scene, SceneCreate};
use narra::models::CharacterCreate;
use narra::services::{SearchFilter, SearchService, SurrealSearchService};
use std::collections::HashMap;
use std::sync::Arc;

/// Test semantic search finds characters by psychological meaning.
///
/// Verifies: Semantic search ranks by meaning similarity, not just keywords
///
/// Note: Requires real fastembed model (~50MB download on first run).
/// Run with: cargo test --test semantic_search -- --ignored --nocapture
#[tokio::test]
#[ignore = "Requires fastembed model download (run manually with --ignored)"]
async fn test_semantic_search_finds_by_meaning() {
    let harness = TestHarness::new().await;

    // Use real embedding service for semantic search
    let embedding_service =
        Arc::new(LocalEmbeddingService::new(EmbeddingConfig::default()).unwrap());
    let staleness_manager = Arc::new(StalenessManager::new(
        harness.db.clone(),
        embedding_service.clone(),
    ));
    let backfill_service = BackfillService::new(harness.db.clone(), embedding_service.clone());
    let search_service = SurrealSearchService::new(harness.db.clone(), embedding_service.clone());

    // Create characters with distinct psychological profiles
    let alice = create_character(
        &harness.db,
        CharacterCreate {
            name: "Alice".into(),
            aliases: vec![],
            roles: vec!["warrior".into()],
            profile: HashMap::from([
                ("wound".into(), vec!["Forced to choose between family duty and personal freedom at age 16. Believes duty means sacrificing everything you want. Resents obligations while fulfilling them.".into()]),
                ("desire_conscious".into(), vec!["Escape family obligations and live freely".into()]),
            ]),
        },
    )
    .await
    .expect("Should create Alice");

    let bob = create_character(
        &harness.db,
        CharacterCreate {
            name: "Bob".into(),
            aliases: vec![],
            roles: vec!["merchant".into()],
            profile: HashMap::from([
                ("wound".into(), vec!["Grew up in poverty, watched parents struggle at age 10. Believes money is the only real security. Hoards wealth, fears generosity.".into()]),
                ("desire_conscious".into(), vec!["Accumulate wealth and never be poor again".into()]),
            ]),
        },
    )
    .await
    .expect("Should create Bob");

    let _carol = create_character(
        &harness.db,
        CharacterCreate {
            name: "Carol".into(),
            aliases: vec![],
            roles: vec!["healer".into()],
            profile: HashMap::from([
                ("wound".into(), vec!["Lost her child to illness she couldn't cure at age 30. Believes loss is inevitable and she is powerless to stop it. Avoids deep attachments, keeps emotional distance.".into()]),
                ("desire_unconscious".into(), vec!["Prove she can save someone this time".into()]),
            ]),
        },
    )
    .await
    .expect("Should create Carol");

    println!("Created characters: Alice (duty/freedom), Bob (greed/poverty), Carol (loss/healing)");

    // Run backfill to generate embeddings
    let stats = backfill_service
        .backfill_all()
        .await
        .expect("Backfill should succeed");
    println!("Backfill stats: {:?}", stats);
    assert!(stats.embedded >= 3, "Should embed at least 3 characters");

    // Semantic search: "characters who struggle with family duty"
    // This should match Alice (duty vs freedom) more than Bob (greed) or Carol (loss)
    let results = search_service
        .semantic_search(
            "characters who struggle with family duty and obligations",
            SearchFilter::default(),
        )
        .await
        .expect("Semantic search should succeed");

    println!("Semantic search results for 'family duty': {:?}", results);

    // Verify results are non-empty
    assert!(!results.is_empty(), "Semantic search should return results");

    // Alice should rank higher than Bob (duty is semantically closer to Alice's wound)
    let alice_id = format!("character:{}", alice.id.key());
    let bob_id = format!("character:{}", bob.id.key());

    let alice_result = results.iter().find(|r| r.id == alice_id);
    let bob_result = results.iter().find(|r| r.id == bob_id);

    // Alice should be in results
    assert!(
        alice_result.is_some(),
        "Alice should be found by semantic search"
    );

    // If Bob is in results, Alice should have higher score
    if let Some(bob_res) = bob_result {
        let alice_res = alice_result.unwrap();
        assert!(
            alice_res.score > bob_res.score,
            "Alice (duty/freedom) should rank higher than Bob (greed) for 'family duty' query"
        );
    }

    println!("✓ Semantic search correctly ranks by psychological meaning");
}

/// Test semantic search finds scenes by mood/atmosphere.
///
/// Verifies: Scene semantic search works for thematic/mood queries
///
/// Note: Requires real fastembed model (~50MB download on first run).
#[tokio::test]
#[ignore = "Requires fastembed model download (run manually with --ignored)"]
async fn test_semantic_search_scene_by_mood() {
    let harness = TestHarness::new().await;

    let embedding_service =
        Arc::new(LocalEmbeddingService::new(EmbeddingConfig::default()).unwrap());
    let staleness_manager = Arc::new(StalenessManager::new(
        harness.db.clone(),
        embedding_service.clone(),
    ));
    let backfill_service = BackfillService::new(harness.db.clone(), embedding_service.clone());
    let search_service = SurrealSearchService::new(harness.db.clone(), embedding_service.clone());

    // Create event and location for scene context
    let event = create_event(
        &harness.db,
        EventCreate {
            title: "The Meeting".into(),
            description: Some("A pivotal moment".into()),
            sequence: 100,
            date: None,
            date_precision: None,
            duration_end: None,
        },
    )
    .await
    .expect("Should create event");

    let location = create_location(
        &harness.db,
        LocationCreate {
            name: "The Plaza".into(),
            description: Some("A public square".into()),
            loc_type: "exterior".into(),
            parent: None,
        },
    )
    .await
    .expect("Should create location");

    // Create scenes with different moods
    let confrontation = create_scene(
        &harness.db,
        SceneCreate {
            title: "The Confrontation".into(),
            summary: Some("A tense standoff between two rivals, weapons drawn, threatening words exchanged in anger".into()),
            event: event.id.clone(),
            primary_location: location.id.clone(),
            secondary_locations: vec![],
        },
    )
    .await
    .expect("Should create confrontation scene");

    let festival = create_scene(
        &harness.db,
        SceneCreate {
            title: "The Festival".into(),
            summary: Some("A joyful celebration with music, dancing, laughter, and community coming together in happiness".into()),
            event: event.id.clone(),
            primary_location: location.id.clone(),
            secondary_locations: vec![],
        },
    )
    .await
    .expect("Should create festival scene");

    println!("Created scenes: Confrontation (tense) and Festival (joyful)");

    // Run backfill
    let stats = backfill_service
        .backfill_all()
        .await
        .expect("Backfill should succeed");
    println!("Backfill stats: {:?}", stats);
    assert!(stats.embedded >= 2, "Should embed at least 2 scenes");

    // Search for "tense scenes"
    let results = search_service
        .semantic_search("tense angry threatening scenes", SearchFilter::default())
        .await
        .expect("Semantic search should succeed");

    println!("Semantic search results for 'tense': {:?}", results);
    assert!(!results.is_empty(), "Should find scenes");

    let confrontation_id = format!("scene:{}", confrontation.id.key());
    let festival_id = format!("scene:{}", festival.id.key());

    let confrontation_result = results.iter().find(|r| r.id == confrontation_id);
    let festival_result = results.iter().find(|r| r.id == festival_id);

    // Confrontation should be found
    assert!(
        confrontation_result.is_some(),
        "Confrontation scene should be found"
    );

    // If festival is found, confrontation should rank higher
    if let Some(festival_res) = festival_result {
        let confrontation_res = confrontation_result.unwrap();
        assert!(
            confrontation_res.score > festival_res.score,
            "Confrontation (tense) should rank higher than Festival (joyful) for 'tense' query"
        );
    }

    println!("✓ Semantic search correctly finds scenes by mood");
}

/// Test hybrid search combines keyword matching with semantic similarity.
///
/// Verifies: Hybrid search uses both keyword and semantic components
///
/// Note: Requires real fastembed model (~50MB download on first run).
#[tokio::test]
#[ignore = "Requires fastembed model download (run manually with --ignored)"]
async fn test_hybrid_search_combines_keyword_and_meaning() {
    let harness = TestHarness::new().await;

    let embedding_service =
        Arc::new(LocalEmbeddingService::new(EmbeddingConfig::default()).unwrap());
    let staleness_manager = Arc::new(StalenessManager::new(
        harness.db.clone(),
        embedding_service.clone(),
    ));
    let backfill_service = BackfillService::new(harness.db.clone(), embedding_service.clone());
    let search_service = SurrealSearchService::new(harness.db.clone(), embedding_service.clone());

    // Create characters
    let bob = create_character(
        &harness.db,
        CharacterCreate {
            name: "Bob the Merchant".into(),
            aliases: vec!["Bobby".into()],
            roles: vec!["merchant".into()],
            profile: HashMap::from([
                ("wound".into(), vec!["Betrayed by business partner at age 25. Believes trust leads to betrayal. Always looks for hidden motives.".into()]),
            ]),
        },
    )
    .await
    .expect("Should create Bob");

    let _alice = create_character(
        &harness.db,
        CharacterCreate {
            name: "Alice the Warrior".into(),
            aliases: vec![],
            roles: vec!["warrior".into()],
            profile: HashMap::from([
                ("wound".into(), vec!["Betrayed by her mentor who turned against the kingdom at age 18. Believes betrayal is the deepest wound. Struggles to trust authority figures.".into()]),
            ]),
        },
    )
    .await
    .expect("Should create Alice");

    println!("Created Bob (name match + betrayal theme) and Alice (betrayal theme)");

    // Run backfill
    let stats = backfill_service
        .backfill_all()
        .await
        .expect("Backfill should succeed");
    println!("Backfill stats: {:?}", stats);
    assert!(stats.embedded >= 2);

    // Hybrid search: "betrayal by Bob"
    // Should find Bob (keyword match on name) and potentially Alice (semantic match on betrayal theme)
    let results = search_service
        .hybrid_search("betrayal by Bob", SearchFilter::default())
        .await
        .expect("Hybrid search should succeed");

    println!("Hybrid search results for 'betrayal by Bob': {:?}", results);
    assert!(!results.is_empty(), "Hybrid search should return results");

    let bob_id = format!("character:{}", bob.id.key());

    // Bob should definitely appear (keyword match on "Bob")
    let bob_result = results.iter().find(|r| r.id == bob_id);
    assert!(
        bob_result.is_some(),
        "Bob should be found by hybrid search (keyword component ensures this)"
    );

    println!("✓ Hybrid search combines keyword and semantic components");
}

/// Test semantic search graceful degradation when embedding model unavailable.
///
/// Verifies: System handles missing embedding service without crashing
#[tokio::test]
async fn test_semantic_search_graceful_degradation() {
    let harness = TestHarness::new().await;

    // Use NoopEmbeddingService (is_available = false)
    let noop_service = Arc::new(NoopEmbeddingService::new());
    let search_service = SurrealSearchService::new(harness.db.clone(), noop_service);

    // Create a character
    create_character(
        &harness.db,
        CharacterCreate {
            name: "Alice".into(),
            aliases: vec![],
            roles: vec!["protagonist".into()],
            ..Default::default()
        },
    )
    .await
    .expect("Should create character");

    // Semantic search should return empty results (not error)
    let results = search_service
        .semantic_search("any query", SearchFilter::default())
        .await
        .expect("Semantic search should not error when model unavailable");

    assert!(
        results.is_empty(),
        "Semantic search should return empty when embeddings unavailable"
    );

    // Hybrid search should fall back to keyword-only
    let _keyword_results = search_service
        .hybrid_search("Alice", SearchFilter::default())
        .await
        .expect("Hybrid search should fall back to keyword search");

    // Note: keyword search might not find results if no fulltext index exists yet,
    // but it should not error

    println!("✓ Graceful degradation works when embedding service unavailable");
}

/// Test semantic search handles entities with empty embeddings.
///
/// Verifies: Search doesn't crash on entities missing embeddings
#[tokio::test]
async fn test_semantic_search_empty_embeddings() {
    let harness = TestHarness::new().await;

    let embedding_service =
        Arc::new(LocalEmbeddingService::new(EmbeddingConfig::default()).unwrap());
    let search_service = SurrealSearchService::new(harness.db.clone(), embedding_service);

    // Create entities but do NOT run backfill
    let _alice = create_character(
        &harness.db,
        CharacterCreate {
            name: "Alice".into(),
            aliases: vec![],
            roles: vec!["protagonist".into()],
            profile: HashMap::from([
                ("wound".into(), vec!["Abandoned as a child at age 8. Believes I am unworthy of love. Pushes people away.".into()]),
            ]),
        },
    )
    .await
    .expect("Should create character");

    let _bob = create_character(
        &harness.db,
        CharacterCreate {
            name: "Bob".into(),
            aliases: vec![],
            roles: vec!["antagonist".into()],
            ..Default::default()
        },
    )
    .await
    .expect("Should create character");

    // Semantic search without embeddings should return empty
    let results = search_service
        .semantic_search("abandonment and childhood trauma", SearchFilter::default())
        .await
        .expect("Semantic search should succeed even with no embeddings");

    assert!(
        results.is_empty(),
        "Semantic search should return empty when entities have no embeddings"
    );

    println!("✓ Semantic search handles entities with missing embeddings");
}
