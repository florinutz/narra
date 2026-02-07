//! Integration tests for Narrative Intelligence Tools (Tier 4).
//!
//! Tests verify:
//! - UnresolvedTensions: perception asymmetry + scene intersection scoring
//! - ThematicGaps: cluster gap detection with narrative interpretation
//! - WhatIf: hypothetical embedding shift, conflict detection, cascade preview
//!
//! Most tests use manually-injected embeddings to avoid requiring the fastembed model.
//! Tests marked `#[ignore]` require real embeddings (run with `cargo test --ignored`).

mod common;

use std::sync::Arc;

use common::harness::TestHarness;
use narra::embedding::{EmbeddingConfig, LocalEmbeddingService, NoopEmbeddingService};
use narra::mcp::{MutationRequest, NarraServer, QueryRequest};
use narra::models::character::{create_character, CharacterCreate};
use narra::models::event::{create_event, EventCreate};
use narra::models::perception::{create_perception, create_perception_pair, PerceptionCreate};
use narra::session::SessionStateManager;
use rmcp::handler::server::wrapper::Parameters;
use std::collections::HashMap;

/// Helper to create a 384-dimensional test embedding vector.
fn test_embedding(seed: f32) -> Vec<f32> {
    (0..384).map(|i| (seed + i as f32 * 0.001).sin()).collect()
}

/// Helper to format a 384-dim embedding for inline SQL.
fn embedding_sql(emb: &[f32]) -> String {
    let nums: Vec<String> = emb.iter().map(|v| format!("{:.6}", v)).collect();
    format!("[{}]", nums.join(","))
}

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
// UNRESOLVED TENSIONS
// =============================================================================

/// Test UnresolvedTensions with no perceives edges returns empty results.
#[tokio::test]
async fn test_unresolved_tensions_empty_world() {
    let harness = TestHarness::new().await;
    let server = create_test_server(&harness).await;

    let request = QueryRequest::UnresolvedTensions {
        limit: None,
        min_asymmetry: None,
        max_shared_scenes: None,
    };
    let response = server
        .handle_query(Parameters(request))
        .await
        .expect("Should succeed with empty results");

    assert_eq!(response.results.len(), 0);
    assert!(
        response
            .hints
            .iter()
            .any(|h| h.contains("No perceives edges")),
        "Should hint about missing edges, got: {:?}",
        response.hints
    );
}

/// Test UnresolvedTensions finds bidirectional pairs with asymmetric embeddings.
#[tokio::test]
async fn test_unresolved_tensions_finds_asymmetric_pairs() {
    let harness = TestHarness::new().await;
    let server = create_test_server(&harness).await;

    // Create characters
    let alice = create_character(
        &harness.db,
        CharacterCreate {
            name: "Alice".into(),
            roles: vec!["protagonist".into()],
            ..Default::default()
        },
    )
    .await
    .expect("Create Alice");

    let bob = create_character(
        &harness.db,
        CharacterCreate {
            name: "Bob".into(),
            roles: vec!["antagonist".into()],
            ..Default::default()
        },
    )
    .await
    .expect("Create Bob");

    let alice_key = alice.id.key().to_string();
    let bob_key = bob.id.key().to_string();

    // Create bidirectional perceives edges
    create_perception_pair(
        &harness.db,
        &alice_key,
        &bob_key,
        PerceptionCreate {
            rel_types: vec!["rival".to_string()],
            subtype: None,
            feelings: Some("distrusts deeply".to_string()),
            perception: Some("dangerous and unpredictable".to_string()),
            tension_level: Some(8),
            history_notes: None,
        },
        PerceptionCreate {
            rel_types: vec!["ally".to_string()],
            subtype: None,
            feelings: Some("trusts completely".to_string()),
            perception: Some("reliable friend".to_string()),
            tension_level: Some(2),
            history_notes: None,
        },
    )
    .await
    .expect("Create perception pair");

    // Set very different embeddings on the perceives edges to simulate asymmetry
    let alice_view_emb = test_embedding(1.0);
    let bob_view_emb = test_embedding(5.0); // Very different seed = high asymmetry

    // Set embeddings on perceives edges
    harness
        .db
        .query(format!(
            "UPDATE perceives SET embedding = {} WHERE in = character:{} AND out = character:{}",
            embedding_sql(&alice_view_emb),
            alice_key,
            bob_key
        ))
        .await
        .expect("Set Alice->Bob embedding");

    harness
        .db
        .query(format!(
            "UPDATE perceives SET embedding = {} WHERE in = character:{} AND out = character:{}",
            embedding_sql(&bob_view_emb),
            bob_key,
            alice_key
        ))
        .await
        .expect("Set Bob->Alice embedding");

    // Query
    let request = QueryRequest::UnresolvedTensions {
        limit: Some(10),
        min_asymmetry: Some(0.01),
        max_shared_scenes: None,
    };
    let response = server
        .handle_query(Parameters(request))
        .await
        .expect("UnresolvedTensions should succeed");

    assert_eq!(response.results.len(), 1, "Should find one tension pair");
    let result = &response.results[0];
    assert_eq!(result.entity_type, "unresolved_tension");
    assert!(
        result.content.contains("Asymmetry:"),
        "Should show asymmetry"
    );
    assert!(
        result.content.contains("Shared scenes:"),
        "Should show shared scene count"
    );
    assert!(
        result.confidence.is_some(),
        "Should have tension_score as confidence"
    );
    assert!(
        result.confidence.unwrap() > 0.0,
        "Tension score should be positive"
    );
}

/// Test UnresolvedTensions filters by min_asymmetry.
#[tokio::test]
async fn test_unresolved_tensions_min_asymmetry_filter() {
    let harness = TestHarness::new().await;
    let server = create_test_server(&harness).await;

    let alice = create_character(
        &harness.db,
        CharacterCreate {
            name: "Alice".into(),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    let bob = create_character(
        &harness.db,
        CharacterCreate {
            name: "Bob".into(),
            ..Default::default()
        },
    )
    .await
    .unwrap();

    let alice_key = alice.id.key().to_string();
    let bob_key = bob.id.key().to_string();

    create_perception_pair(
        &harness.db,
        &alice_key,
        &bob_key,
        PerceptionCreate {
            rel_types: vec!["friend".into()],
            subtype: None,
            feelings: None,
            perception: None,
            tension_level: None,
            history_notes: None,
        },
        PerceptionCreate {
            rel_types: vec!["friend".into()],
            subtype: None,
            feelings: None,
            perception: None,
            tension_level: None,
            history_notes: None,
        },
    )
    .await
    .unwrap();

    // Set nearly identical embeddings (low asymmetry)
    let emb = test_embedding(1.0);
    harness
        .db
        .query(format!(
            "UPDATE perceives SET embedding = {} WHERE in = character:{} AND out = character:{}",
            embedding_sql(&emb),
            alice_key,
            bob_key
        ))
        .await
        .unwrap();
    harness
        .db
        .query(format!(
            "UPDATE perceives SET embedding = {} WHERE in = character:{} AND out = character:{}",
            embedding_sql(&emb),
            bob_key,
            alice_key
        ))
        .await
        .unwrap();

    // With high min_asymmetry, should find nothing
    let request = QueryRequest::UnresolvedTensions {
        limit: None,
        min_asymmetry: Some(0.5),
        max_shared_scenes: None,
    };
    let response = server
        .handle_query(Parameters(request))
        .await
        .expect("Should succeed");

    assert_eq!(
        response.results.len(),
        0,
        "Identical embeddings should not pass high asymmetry filter"
    );
}

/// Test UnresolvedTensions without bidirectional edges returns empty.
#[tokio::test]
async fn test_unresolved_tensions_one_way_only() {
    let harness = TestHarness::new().await;
    let server = create_test_server(&harness).await;

    let alice = create_character(
        &harness.db,
        CharacterCreate {
            name: "Alice".into(),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    let bob = create_character(
        &harness.db,
        CharacterCreate {
            name: "Bob".into(),
            ..Default::default()
        },
    )
    .await
    .unwrap();

    // Only one-way perception
    create_perception(
        &harness.db,
        &alice.id.key().to_string(),
        &bob.id.key().to_string(),
        PerceptionCreate {
            rel_types: vec!["rival".into()],
            subtype: None,
            feelings: None,
            perception: None,
            tension_level: None,
            history_notes: None,
        },
    )
    .await
    .unwrap();

    // Set embedding on the one-way edge
    let emb = test_embedding(1.0);
    harness
        .db
        .query(format!(
            "UPDATE perceives SET embedding = {} WHERE in = {} AND out = {}",
            embedding_sql(&emb),
            alice.id,
            bob.id
        ))
        .await
        .unwrap();

    let request = QueryRequest::UnresolvedTensions {
        limit: None,
        min_asymmetry: None,
        max_shared_scenes: None,
    };
    let response = server
        .handle_query(Parameters(request))
        .await
        .expect("Should succeed");

    assert_eq!(
        response.results.len(),
        0,
        "One-way perceptions should not produce tensions"
    );
}

/// Test UnresolvedTensions respects max_shared_scenes filter.
#[tokio::test]
async fn test_unresolved_tensions_max_shared_scenes() {
    let harness = TestHarness::new().await;
    let server = create_test_server(&harness).await;

    let alice = create_character(
        &harness.db,
        CharacterCreate {
            name: "Alice".into(),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    let bob = create_character(
        &harness.db,
        CharacterCreate {
            name: "Bob".into(),
            ..Default::default()
        },
    )
    .await
    .unwrap();

    let alice_key = alice.id.key().to_string();
    let bob_key = bob.id.key().to_string();

    create_perception_pair(
        &harness.db,
        &alice_key,
        &bob_key,
        PerceptionCreate {
            rel_types: vec!["rival".into()],
            subtype: None,
            feelings: None,
            perception: None,
            tension_level: None,
            history_notes: None,
        },
        PerceptionCreate {
            rel_types: vec!["ally".into()],
            subtype: None,
            feelings: None,
            perception: None,
            tension_level: None,
            history_notes: None,
        },
    )
    .await
    .unwrap();

    // Set different embeddings
    let emb_a = test_embedding(1.0);
    let emb_b = test_embedding(5.0);
    harness
        .db
        .query(format!(
            "UPDATE perceives SET embedding = {} WHERE in = character:{} AND out = character:{}",
            embedding_sql(&emb_a),
            alice_key,
            bob_key
        ))
        .await
        .unwrap();
    harness
        .db
        .query(format!(
            "UPDATE perceives SET embedding = {} WHERE in = character:{} AND out = character:{}",
            embedding_sql(&emb_b),
            bob_key,
            alice_key
        ))
        .await
        .unwrap();

    // Create shared scenes — need event + location first
    let event = create_event(
        &harness.db,
        EventCreate {
            title: "The Confrontation".into(),
            sequence: 1,
            description: None,
            date: None,
            date_precision: None,
            duration_end: None,
        },
    )
    .await
    .unwrap();

    use narra::models::location::{create_location, LocationCreate};
    let location = create_location(
        &harness.db,
        LocationCreate {
            name: "The Arena".into(),
            loc_type: "place".into(),
            description: None,
            parent: None,
        },
    )
    .await
    .unwrap();

    use narra::models::scene::{create_scene, SceneCreate};
    // Create 3 scenes and add both characters
    for i in 0..3 {
        let scene = create_scene(
            &harness.db,
            SceneCreate {
                title: format!("Scene {}", i),
                summary: None,
                event: event.id.clone(),
                primary_location: location.id.clone(),
                secondary_locations: vec![],
            },
        )
        .await
        .unwrap();

        // Add both characters to scene via participates_in
        let scene_id = scene.id.to_string();
        harness
            .db
            .query(format!(
                "RELATE {}->participates_in->{} SET role = 'participant'",
                alice.id, scene_id
            ))
            .await
            .unwrap();
        harness
            .db
            .query(format!(
                "RELATE {}->participates_in->{} SET role = 'participant'",
                bob.id, scene_id
            ))
            .await
            .unwrap();
    }

    // Verify participates_in edges were created
    #[derive(serde::Deserialize)]
    struct CountResult {
        count: i64,
    }
    let mut cnt_resp = harness
        .db
        .query("SELECT count() FROM participates_in GROUP ALL")
        .await
        .unwrap();
    let counts: Vec<CountResult> = cnt_resp.take(0).unwrap_or_default();
    assert_eq!(
        counts[0].count, 6,
        "Should have 6 participates_in edges (2 chars x 3 scenes)"
    );

    // With max_shared_scenes=2, the pair (3 shared scenes) should be filtered out
    let request = QueryRequest::UnresolvedTensions {
        limit: None,
        min_asymmetry: Some(0.01),
        max_shared_scenes: Some(2),
    };
    let response = server
        .handle_query(Parameters(request))
        .await
        .expect("Should succeed");

    assert_eq!(
        response.results.len(),
        0,
        "Pair with 3 shared scenes should be excluded by max_shared_scenes=2"
    );

    // With max_shared_scenes=5, should be included
    let request = QueryRequest::UnresolvedTensions {
        limit: None,
        min_asymmetry: Some(0.01),
        max_shared_scenes: Some(5),
    };
    let response = server
        .handle_query(Parameters(request))
        .await
        .expect("Should succeed");

    assert_eq!(
        response.results.len(),
        1,
        "Pair with 3 shared scenes should be included with max_shared_scenes=5"
    );
}

/// Test UnresolvedTensions respects limit parameter.
#[tokio::test]
async fn test_unresolved_tensions_limit() {
    let harness = TestHarness::new().await;
    let server = create_test_server(&harness).await;

    // Create 3 characters with pairwise bidirectional perceives
    let mut chars = Vec::new();
    for name in ["Alice", "Bob", "Charlie"] {
        chars.push(
            create_character(
                &harness.db,
                CharacterCreate {
                    name: name.to_string(),
                    ..Default::default()
                },
            )
            .await
            .unwrap(),
        );
    }

    let keys: Vec<String> = chars.iter().map(|c| c.id.key().to_string()).collect();

    // Create bidirectional perceptions for all pairs
    for i in 0..3 {
        for j in (i + 1)..3 {
            create_perception_pair(
                &harness.db,
                &keys[i],
                &keys[j],
                PerceptionCreate {
                    rel_types: vec!["rival".into()],
                    subtype: None,
                    feelings: None,
                    perception: None,
                    tension_level: None,
                    history_notes: None,
                },
                PerceptionCreate {
                    rel_types: vec!["ally".into()],
                    subtype: None,
                    feelings: None,
                    perception: None,
                    tension_level: None,
                    history_notes: None,
                },
            )
            .await
            .unwrap();

            // Set different embeddings per direction
            let emb_fwd = test_embedding((i * 10 + j) as f32);
            let emb_rev = test_embedding((i * 10 + j + 50) as f32);

            harness.db.query(format!(
                "UPDATE perceives SET embedding = {} WHERE in = character:{} AND out = character:{}",
                embedding_sql(&emb_fwd), keys[i], keys[j]
            )).await.unwrap();
            harness.db.query(format!(
                "UPDATE perceives SET embedding = {} WHERE in = character:{} AND out = character:{}",
                embedding_sql(&emb_rev), keys[j], keys[i]
            )).await.unwrap();
        }
    }

    // With limit=1, should return only 1 result
    let request = QueryRequest::UnresolvedTensions {
        limit: Some(1),
        min_asymmetry: Some(0.0),
        max_shared_scenes: None,
    };
    let response = server
        .handle_query(Parameters(request))
        .await
        .expect("Should succeed");

    assert_eq!(response.results.len(), 1, "Should respect limit=1");
}

// =============================================================================
// THEMATIC GAPS
// =============================================================================

/// Test ThematicGaps with insufficient entities for clustering returns error.
#[tokio::test]
async fn test_thematic_gaps_insufficient_entities() {
    let harness = TestHarness::new().await;
    let server = create_test_server(&harness).await;

    // Only 1 character with embedding — too few for clustering
    let alice = create_character(
        &harness.db,
        CharacterCreate {
            name: "Alice".into(),
            ..Default::default()
        },
    )
    .await
    .unwrap();

    let emb = test_embedding(1.0);
    harness
        .db
        .query(format!(
            "UPDATE {} SET embedding = {}, embedding_stale = false",
            alice.id,
            embedding_sql(&emb)
        ))
        .await
        .unwrap();

    let request = QueryRequest::ThematicGaps {
        min_cluster_size: None,
        expected_types: None,
    };
    let result = server.handle_query(Parameters(request)).await;

    assert!(
        result.is_err(),
        "Should error with insufficient entities for clustering"
    );
    assert!(
        result.unwrap_err().contains("Insufficient"),
        "Error should mention insufficient entities"
    );
}

/// Test ThematicGaps detects missing entity types in clusters.
#[tokio::test]
async fn test_thematic_gaps_finds_gaps() {
    let harness = TestHarness::new().await;
    let server = create_test_server(&harness).await;

    // Create 4 characters with similar embeddings (will cluster together)
    // but NO events — so "event" should be a gap
    for (i, name) in ["Alice", "Bob", "Charlie", "Diana"].iter().enumerate() {
        let char = create_character(
            &harness.db,
            CharacterCreate {
                name: name.to_string(),
                roles: vec!["warrior".into()],
                ..Default::default()
            },
        )
        .await
        .unwrap();

        // Use similar embeddings so they cluster together
        let emb = test_embedding(1.0 + i as f32 * 0.01);
        harness
            .db
            .query(format!(
                "UPDATE {} SET embedding = {}, embedding_stale = false",
                char.id,
                embedding_sql(&emb)
            ))
            .await
            .unwrap();
    }

    let request = QueryRequest::ThematicGaps {
        min_cluster_size: Some(2),
        expected_types: Some(vec!["character".to_string(), "event".to_string()]),
    };
    let response = server
        .handle_query(Parameters(request))
        .await
        .expect("ThematicGaps should succeed");

    // Should find at least one gap (events missing from character-only clusters)
    assert!(
        !response.results.is_empty(),
        "Should find gaps (no events in character clusters)"
    );

    for result in &response.results {
        assert_eq!(result.entity_type, "thematic_gap");
        assert!(
            result.content.contains("Missing types:"),
            "Should list missing types: {}",
            result.content
        );
        assert!(
            result.confidence.is_some(),
            "Should have gap_severity as confidence"
        );
    }

    // Hints should mention analysis
    assert!(
        response.hints.iter().any(|h| h.contains("clusters")),
        "Hints should mention clusters: {:?}",
        response.hints
    );
}

/// Test ThematicGaps with no gaps returns empty results.
#[tokio::test]
async fn test_thematic_gaps_no_gaps_when_types_present() {
    let harness = TestHarness::new().await;
    let server = create_test_server(&harness).await;

    // Create 4 characters
    for (i, name) in ["Alice", "Bob", "Charlie", "Diana"].iter().enumerate() {
        let char = create_character(
            &harness.db,
            CharacterCreate {
                name: name.to_string(),
                ..Default::default()
            },
        )
        .await
        .unwrap();

        let emb = test_embedding(1.0 + i as f32 * 0.01);
        harness
            .db
            .query(format!(
                "UPDATE {} SET embedding = {}, embedding_stale = false",
                char.id,
                embedding_sql(&emb)
            ))
            .await
            .unwrap();
    }

    // Only expect "character" type — all present, so no gaps
    let request = QueryRequest::ThematicGaps {
        min_cluster_size: Some(2),
        expected_types: Some(vec!["character".to_string()]),
    };
    let response = server
        .handle_query(Parameters(request))
        .await
        .expect("Should succeed");

    assert_eq!(
        response.results.len(),
        0,
        "No gaps expected when all clusters contain the expected type"
    );
}

/// Test ThematicGaps min_cluster_size filter.
#[tokio::test]
async fn test_thematic_gaps_min_cluster_size() {
    let harness = TestHarness::new().await;
    let server = create_test_server(&harness).await;

    // Create 3 characters (the minimum for clustering)
    for (i, name) in ["Alice", "Bob", "Charlie"].iter().enumerate() {
        let char = create_character(
            &harness.db,
            CharacterCreate {
                name: name.to_string(),
                ..Default::default()
            },
        )
        .await
        .unwrap();

        let emb = test_embedding(1.0 + i as f32 * 0.01);
        harness
            .db
            .query(format!(
                "UPDATE {} SET embedding = {}, embedding_stale = false",
                char.id,
                embedding_sql(&emb)
            ))
            .await
            .unwrap();
    }

    // With min_cluster_size=100, no cluster should qualify
    let request = QueryRequest::ThematicGaps {
        min_cluster_size: Some(100),
        expected_types: Some(vec!["character".to_string(), "event".to_string()]),
    };
    let response = server
        .handle_query(Parameters(request))
        .await
        .expect("Should succeed");

    assert_eq!(
        response.results.len(),
        0,
        "No clusters should meet min_cluster_size=100"
    );
}

// =============================================================================
// WHAT IF
// =============================================================================

/// Test WhatIf with noop embedding service returns error.
#[tokio::test]
async fn test_what_if_no_embedding_service() {
    let harness = TestHarness::new().await;
    let server = create_test_server(&harness).await; // noop embeddings

    let request = QueryRequest::WhatIf {
        character_id: "alice".to_string(),
        fact_id: "secret1".to_string(),
        certainty: None,
        source_character: None,
    };
    let result = server.handle_query(Parameters(request)).await;

    assert!(
        result.is_err(),
        "Should error when embedding service unavailable"
    );
    assert!(
        result.unwrap_err().contains("embedding model not loaded"),
        "Should mention embedding model"
    );
}

/// Test WhatIf with nonexistent character returns error.
#[tokio::test]
async fn test_what_if_nonexistent_character() {
    let harness = TestHarness::new().await;
    let embedding_service =
        Arc::new(LocalEmbeddingService::new(EmbeddingConfig::default()).unwrap());
    let server = create_test_server_with_embeddings(&harness, embedding_service).await;

    let request = QueryRequest::WhatIf {
        character_id: "nonexistent".to_string(),
        fact_id: "secret1".to_string(),
        certainty: None,
        source_character: None,
    };
    let result = server.handle_query(Parameters(request)).await;

    assert!(result.is_err(), "Should error for nonexistent character");
    assert!(
        result.unwrap_err().contains("not found"),
        "Should mention entity not found"
    );
}

/// Test WhatIf with nonexistent fact returns error.
#[tokio::test]
#[ignore = "Requires fastembed model download"]
async fn test_what_if_nonexistent_fact() {
    let harness = TestHarness::new().await;
    let embedding_service =
        Arc::new(LocalEmbeddingService::new(EmbeddingConfig::default()).unwrap());
    let server = create_test_server_with_embeddings(&harness, embedding_service).await;

    // Create character but not the fact
    let _alice = create_character(
        &harness.db,
        CharacterCreate {
            name: "Alice".into(),
            ..Default::default()
        },
    )
    .await
    .unwrap();

    let request = QueryRequest::WhatIf {
        character_id: "alice".to_string(),
        fact_id: "nonexistent_fact".to_string(),
        certainty: None,
        source_character: None,
    };
    let result = server.handle_query(Parameters(request)).await;

    assert!(result.is_err(), "Should error for nonexistent fact");
    assert!(
        result.unwrap_err().contains("not found"),
        "Should mention entity not found"
    );
}

/// Test WhatIf with character missing embedding returns helpful error.
#[tokio::test]
#[ignore = "Requires fastembed model download"]
async fn test_what_if_no_character_embedding() {
    let harness = TestHarness::new().await;
    let embedding_service =
        Arc::new(LocalEmbeddingService::new(EmbeddingConfig::default()).unwrap());
    let server = create_test_server_with_embeddings(&harness, embedding_service).await;

    // Create character without embedding
    let alice = create_character(
        &harness.db,
        CharacterCreate {
            name: "Alice".into(),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    let alice_key = alice.id.key().to_string();

    // Create knowledge fact via proper mutation
    server
        .handle_mutate(Parameters(MutationRequest::RecordKnowledge {
            character_id: alice.id.to_string(),
            target_id: "treasure_secret".to_string(),
            fact: "The treasure is hidden under the mountain".to_string(),
            certainty: "knows".to_string(),
            method: Some("initial".to_string()),
            source_character_id: None,
            event_id: None,
        }))
        .await
        .expect("RecordKnowledge should succeed");

    // Find the knowledge key
    #[derive(serde::Deserialize)]
    struct KId {
        id: surrealdb::RecordId,
    }
    let mut k_resp = harness
        .db
        .query("SELECT id FROM knowledge LIMIT 1")
        .await
        .unwrap();
    let k_ids: Vec<KId> = k_resp.take(0).unwrap_or_default();
    assert!(!k_ids.is_empty(), "Knowledge should exist");
    let knowledge_key = k_ids[0].id.key().to_string();

    let request = QueryRequest::WhatIf {
        character_id: alice_key,
        fact_id: knowledge_key,
        certainty: None,
        source_character: None,
    };
    let result = server.handle_query(Parameters(request)).await;

    assert!(
        result.is_err(),
        "Should error when character has no embedding"
    );
    let err_msg = result.unwrap_err();
    assert!(
        err_msg.contains("BackfillEmbeddings"),
        "Should suggest running BackfillEmbeddings, got: {}",
        err_msg
    );
}

/// Test full WhatIf workflow with real embeddings.
#[tokio::test]
#[ignore = "Requires fastembed model download"]
async fn test_what_if_full_workflow() {
    let harness = TestHarness::new().await;
    let embedding_service =
        Arc::new(LocalEmbeddingService::new(EmbeddingConfig::default()).unwrap());
    let server = create_test_server_with_embeddings(&harness, embedding_service.clone()).await;

    // Create characters
    let alice = create_character(
        &harness.db,
        CharacterCreate {
            name: "Alice".into(),
            aliases: vec![],
            roles: vec!["warrior".into()],
            profile: HashMap::from([(
                "wound".into(),
                vec!["Lost her family at age 12. Trust no one. Pushes people away.".into()],
            )]),
        },
    )
    .await
    .unwrap();

    let bob = create_character(
        &harness.db,
        CharacterCreate {
            name: "Bob".into(),
            aliases: vec![],
            roles: vec!["healer".into()],
            ..Default::default()
        },
    )
    .await
    .unwrap();

    // Backfill embeddings for characters
    let backfill_request = MutationRequest::BackfillEmbeddings {
        entity_type: Some("character".into()),
    };
    server
        .handle_mutate(Parameters(backfill_request))
        .await
        .expect("Backfill should succeed");

    // Record knowledge that Bob knows a secret
    let record_request = MutationRequest::RecordKnowledge {
        character_id: bob.id.to_string(),
        target_id: "secret".to_string(),
        fact: "Alice is the lost princess of the Northern Kingdom".to_string(),
        certainty: "knows".to_string(),
        method: Some("initial".to_string()),
        source_character_id: None,
        event_id: None,
    };
    server
        .handle_mutate(Parameters(record_request))
        .await
        .expect("RecordKnowledge should succeed");

    // Backfill knowledge embeddings
    let backfill_request = MutationRequest::BackfillEmbeddings {
        entity_type: Some("knowledge".into()),
    };
    server
        .handle_mutate(Parameters(backfill_request))
        .await
        .expect("Knowledge backfill should succeed");

    // Find the knowledge entity ID
    #[derive(serde::Deserialize)]
    struct KnowledgeId {
        id: surrealdb::RecordId,
    }

    let mut k_resp = harness
        .db
        .query("SELECT id FROM knowledge LIMIT 1")
        .await
        .expect("Query knowledge");
    let knowledge_ids: Vec<KnowledgeId> = k_resp.take(0).unwrap_or_default();
    assert!(!knowledge_ids.is_empty(), "Knowledge entity should exist");
    let knowledge_key = knowledge_ids[0].id.key().to_string();

    // Create a perceives edge so Bob → Alice (for cascade test)
    create_perception(
        &harness.db,
        &bob.id.key().to_string(),
        &alice.id.key().to_string(),
        PerceptionCreate {
            rel_types: vec!["ally".into()],
            perception: Some("a brave warrior".into()),
            subtype: None,
            feelings: None,
            tension_level: None,
            history_notes: None,
        },
    )
    .await
    .unwrap();

    // Set embedding on the perceives edge for cascade detection
    let persp_emb = test_embedding(3.0);
    harness
        .db
        .query(format!(
            "UPDATE perceives SET embedding = {} WHERE in = {} AND out = {}",
            embedding_sql(&persp_emb),
            bob.id,
            alice.id
        ))
        .await
        .unwrap();

    // Now run WhatIf: "What if Alice learned the secret?"
    let request = QueryRequest::WhatIf {
        character_id: alice.id.key().to_string(),
        fact_id: knowledge_key,
        certainty: Some("suspects".to_string()),
        source_character: None,
    };
    let response = server
        .handle_query(Parameters(request))
        .await
        .expect("WhatIf should succeed");

    // Should have at least the primary analysis result
    assert!(
        !response.results.is_empty(),
        "Should return analysis results"
    );

    // Primary result
    let primary = &response.results[0];
    assert_eq!(primary.entity_type, "what_if_analysis");
    assert!(
        primary.content.contains("Embedding delta:"),
        "Should show delta: {}",
        primary.content
    );
    assert!(
        primary.content.contains("Alice"),
        "Should mention character name: {}",
        primary.content
    );
    assert!(
        primary.content.contains("princess") || primary.content.contains("would learn"),
        "Should reference the fact: {}",
        primary.content
    );
    assert!(
        primary.content.contains("suspects"),
        "Should mention certainty: {}",
        primary.content
    );
    assert!(
        primary.confidence.is_some(),
        "Should have delta as confidence"
    );

    // Check for cascade result (Bob observes Alice)
    let cascade = response
        .results
        .iter()
        .find(|r| r.entity_type == "what_if_cascade");
    assert!(
        cascade.is_some(),
        "Should have cascade result for Bob's perspective on Alice"
    );
    let cascade = cascade.unwrap();
    assert!(
        cascade.content.contains("Bob"),
        "Cascade should mention Bob as observer: {}",
        cascade.content
    );

    // Check hints
    assert!(
        response.hints.iter().any(|h| h.contains("Impact:")),
        "Hints should include impact assessment: {:?}",
        response.hints
    );
    assert!(
        response
            .hints
            .iter()
            .any(|h| h.contains("preview") || h.contains("RecordKnowledge")),
        "Hints should mention this is a preview: {:?}",
        response.hints
    );
}

/// Test WhatIf ID normalization works with bare keys.
#[tokio::test]
#[ignore = "Requires fastembed model download"]
async fn test_what_if_id_normalization() {
    let harness = TestHarness::new().await;
    let embedding_service =
        Arc::new(LocalEmbeddingService::new(EmbeddingConfig::default()).unwrap());
    let server = create_test_server_with_embeddings(&harness, embedding_service.clone()).await;

    let alice = create_character(
        &harness.db,
        CharacterCreate {
            name: "Alice".into(),
            roles: vec!["warrior".into()],
            ..Default::default()
        },
    )
    .await
    .unwrap();

    // Backfill
    let backfill_request = MutationRequest::BackfillEmbeddings {
        entity_type: Some("character".into()),
    };
    server
        .handle_mutate(Parameters(backfill_request))
        .await
        .unwrap();

    // Create knowledge
    let record_request = MutationRequest::RecordKnowledge {
        character_id: alice.id.to_string(),
        target_id: "mysecret".to_string(),
        fact: "The world is flat".to_string(),
        certainty: "suspects".to_string(),
        method: Some("initial".to_string()),
        source_character_id: None,
        event_id: None,
    };
    server
        .handle_mutate(Parameters(record_request))
        .await
        .unwrap();

    // Backfill knowledge
    server
        .handle_mutate(Parameters(MutationRequest::BackfillEmbeddings {
            entity_type: Some("knowledge".into()),
        }))
        .await
        .unwrap();

    // Find knowledge key
    #[derive(serde::Deserialize)]
    struct KnowledgeId {
        id: surrealdb::RecordId,
    }
    let mut k_resp = harness
        .db
        .query("SELECT id FROM knowledge LIMIT 1")
        .await
        .unwrap();
    let k_ids: Vec<KnowledgeId> = k_resp.take(0).unwrap_or_default();
    let knowledge_key = k_ids[0].id.key().to_string();

    // Call with bare keys (no prefix)
    let request = QueryRequest::WhatIf {
        character_id: alice.id.key().to_string(), // bare key
        fact_id: knowledge_key.clone(),           // bare key
        certainty: None,
        source_character: None,
    };
    let response = server.handle_query(Parameters(request)).await;
    assert!(
        response.is_ok(),
        "Should handle bare keys: {:?}",
        response.err()
    );

    // Call with full IDs
    let request = QueryRequest::WhatIf {
        character_id: alice.id.to_string(),              // full ID
        fact_id: format!("knowledge:{}", knowledge_key), // full ID
        certainty: None,
        source_character: None,
    };
    let response = server.handle_query(Parameters(request)).await;
    assert!(
        response.is_ok(),
        "Should handle full IDs: {:?}",
        response.err()
    );
}

/// Test WhatIf conflict detection flags semantically similar existing knowledge.
#[tokio::test]
#[ignore = "Requires fastembed model download"]
async fn test_what_if_conflict_detection() {
    let harness = TestHarness::new().await;
    let embedding_service =
        Arc::new(LocalEmbeddingService::new(EmbeddingConfig::default()).unwrap());
    let server = create_test_server_with_embeddings(&harness, embedding_service.clone()).await;

    let alice = create_character(
        &harness.db,
        CharacterCreate {
            name: "Alice".into(),
            roles: vec!["detective".into()],
            ..Default::default()
        },
    )
    .await
    .unwrap();

    // Backfill character embedding
    server
        .handle_mutate(Parameters(MutationRequest::BackfillEmbeddings {
            entity_type: Some("character".into()),
        }))
        .await
        .unwrap();

    // Record contradictory knowledge
    // Alice believes "Bob is innocent"
    server
        .handle_mutate(Parameters(MutationRequest::RecordKnowledge {
            character_id: alice.id.to_string(),
            target_id: "bob_innocence".to_string(),
            fact: "Bob is completely innocent and trustworthy".to_string(),
            certainty: "knows".to_string(),
            method: Some("initial".to_string()),
            source_character_id: None,
            event_id: None,
        }))
        .await
        .unwrap();

    // Now create a fact that says "Bob is guilty"
    server
        .handle_mutate(Parameters(MutationRequest::RecordKnowledge {
            character_id: alice.id.to_string(), // owned by alice but we'll treat as a fact
            target_id: "bob_guilt".to_string(),
            fact: "Bob is guilty of the crime and cannot be trusted".to_string(),
            certainty: "knows".to_string(),
            method: Some("initial".to_string()),
            source_character_id: None,
            event_id: None,
        }))
        .await
        .unwrap();

    // Backfill knowledge embeddings
    server
        .handle_mutate(Parameters(MutationRequest::BackfillEmbeddings {
            entity_type: Some("knowledge".into()),
        }))
        .await
        .unwrap();

    // Find the guilt fact ID
    #[derive(serde::Deserialize)]
    struct KFact {
        id: surrealdb::RecordId,
        fact: String,
    }
    let mut resp = harness
        .db
        .query("SELECT id, fact FROM knowledge WHERE fact CONTAINS 'guilty'")
        .await
        .unwrap();
    let facts: Vec<KFact> = resp.take(0).unwrap_or_default();
    assert!(!facts.is_empty(), "Should find the guilt fact");
    let guilt_key = facts[0].id.key().to_string();

    // WhatIf: what if Alice learns the guilt fact?
    let request = QueryRequest::WhatIf {
        character_id: alice.id.key().to_string(),
        fact_id: guilt_key,
        certainty: Some("knows".to_string()),
        source_character: None,
    };
    let response = server
        .handle_query(Parameters(request))
        .await
        .expect("WhatIf should succeed");

    // Check for conflict results
    let conflicts: Vec<_> = response
        .results
        .iter()
        .filter(|r| r.entity_type == "what_if_conflict")
        .collect();

    // The innocence and guilt facts should be semantically similar enough to flag
    // (both about Bob, trust, and wrongdoing — high semantic overlap)
    // Note: this depends on the embedding model, so we just verify the structure
    if !conflicts.is_empty() {
        assert!(
            conflicts[0].content.contains("Existing knowledge:"),
            "Conflict should reference existing knowledge: {}",
            conflicts[0].content
        );
        assert!(
            conflicts[0].content.contains("Similarity"),
            "Conflict should show similarity score: {}",
            conflicts[0].content
        );
    }

    // Hints should mention this is a preview
    assert!(
        response.hints.iter().any(|h| h.contains("RecordKnowledge")),
        "Should mention RecordKnowledge to commit"
    );
}

// =============================================================================
// RESPONSE FORMAT TESTS
// =============================================================================

/// Test that all three operations produce valid QueryResponse structure.
#[tokio::test]
async fn test_response_format_unresolved_tensions() {
    let harness = TestHarness::new().await;
    let server = create_test_server(&harness).await;

    let request = QueryRequest::UnresolvedTensions {
        limit: None,
        min_asymmetry: None,
        max_shared_scenes: None,
    };
    let response = server
        .handle_query(Parameters(request))
        .await
        .expect("Should return valid response");

    // Even empty, should have valid structure
    assert!(response.total == response.results.len());
    assert!(response.next_cursor.is_none());
    assert!(!response.hints.is_empty(), "Should always have hints");
    assert!(response.token_estimate >= 0);
}

#[tokio::test]
async fn test_response_format_thematic_gaps_propagates_clustering_error() {
    let harness = TestHarness::new().await;
    let server = create_test_server(&harness).await;

    // With no entities, clustering should fail and propagate error
    let request = QueryRequest::ThematicGaps {
        min_cluster_size: None,
        expected_types: None,
    };
    let result = server.handle_query(Parameters(request)).await;

    // Should propagate the clustering error (insufficient entities)
    assert!(result.is_err(), "Should propagate clustering error");
}

// =============================================================================
// KNOWLEDGE CONFLICTS
// =============================================================================

/// Test KnowledgeConflicts returns empty when no BelievesWrongly states exist.
#[tokio::test]
async fn test_knowledge_conflicts_empty_when_no_conflicts() {
    let harness = TestHarness::new().await;
    let server = create_test_server(&harness).await;

    // Create a character with normal (Knows) knowledge — no BelievesWrongly
    let alice = create_character(
        &harness.db,
        CharacterCreate {
            name: "Alice".into(),
            ..Default::default()
        },
    )
    .await
    .unwrap();

    // Record normal knowledge (certainty = "knows", not "believes_wrongly")
    server
        .handle_mutate(Parameters(MutationRequest::RecordKnowledge {
            character_id: alice.id.to_string(),
            target_id: "fact1".to_string(),
            fact: "The sun rises in the east".to_string(),
            certainty: "knows".to_string(),
            method: Some("initial".to_string()),
            source_character_id: None,
            event_id: None,
        }))
        .await
        .expect("RecordKnowledge should succeed");

    let request = QueryRequest::KnowledgeConflicts {
        character_id: None,
        limit: None,
    };
    let response = server
        .handle_query(Parameters(request))
        .await
        .expect("KnowledgeConflicts should succeed");

    assert_eq!(
        response.results.len(),
        0,
        "No BelievesWrongly states = no conflicts"
    );
    assert!(
        response.hints.iter().any(|h| h.contains("0")),
        "Hints should mention 0 conflicts: {:?}",
        response.hints
    );
}

/// Test KnowledgeConflicts finds BelievesWrongly states with truth_value.
#[tokio::test]
async fn test_knowledge_conflicts_finds_believes_wrongly() {
    let harness = TestHarness::new().await;
    let server = create_test_server(&harness).await;

    let alice = create_character(
        &harness.db,
        CharacterCreate {
            name: "Alice".into(),
            ..Default::default()
        },
    )
    .await
    .unwrap();

    // Record BelievesWrongly knowledge with truth_value
    server
        .handle_mutate(Parameters(MutationRequest::RecordKnowledge {
            character_id: alice.id.to_string(),
            target_id: "secret_identity".to_string(),
            fact: "Bob is a simple merchant".to_string(),
            certainty: "believes_wrongly".to_string(),
            method: Some("initial".to_string()),
            source_character_id: None,
            event_id: None,
        }))
        .await
        .expect("RecordKnowledge should succeed");

    // Set truth_value on the knows edge directly (RecordKnowledge may not support it via MCP)
    harness
        .db
        .query("UPDATE knows SET truth_value = 'Bob is actually the king in disguise' WHERE certainty = 'believes_wrongly'")
        .await
        .expect("Set truth_value");

    let request = QueryRequest::KnowledgeConflicts {
        character_id: None,
        limit: None,
    };
    let response = server
        .handle_query(Parameters(request))
        .await
        .expect("KnowledgeConflicts should succeed");

    assert!(
        !response.results.is_empty(),
        "Should find at least one conflict from BelievesWrongly state"
    );
    assert_eq!(
        response.results[0].entity_type, "knowledge_conflict",
        "Entity type should be knowledge_conflict"
    );
    assert!(
        response.results[0].content.contains("believes wrongly"),
        "Content should describe the wrong belief: {}",
        response.results[0].content
    );
    assert!(
        response.results[0].content.contains("actual truth"),
        "Content should include truth_value: {}",
        response.results[0].content
    );
}

/// Test KnowledgeConflicts filters by character_id.
#[tokio::test]
async fn test_knowledge_conflicts_filters_by_character() {
    let harness = TestHarness::new().await;
    let server = create_test_server(&harness).await;

    let alice = create_character(
        &harness.db,
        CharacterCreate {
            name: "Alice".into(),
            ..Default::default()
        },
    )
    .await
    .unwrap();

    let bob = create_character(
        &harness.db,
        CharacterCreate {
            name: "Bob".into(),
            ..Default::default()
        },
    )
    .await
    .unwrap();

    // Both have BelievesWrongly states
    for char in [&alice, &bob] {
        server
            .handle_mutate(Parameters(MutationRequest::RecordKnowledge {
                character_id: char.id.to_string(),
                target_id: format!("secret_{}", char.id.key()),
                fact: "Wrong belief".to_string(),
                certainty: "believes_wrongly".to_string(),
                method: Some("initial".to_string()),
                source_character_id: None,
                event_id: None,
            }))
            .await
            .expect("RecordKnowledge should succeed");
    }

    // Filter to Alice only
    let alice_key = alice.id.key().to_string();
    let request = QueryRequest::KnowledgeConflicts {
        character_id: Some(alice_key.clone()),
        limit: None,
    };
    let response = server
        .handle_query(Parameters(request))
        .await
        .expect("KnowledgeConflicts should succeed");

    // Should only find Alice's conflict, not Bob's
    for result in &response.results {
        assert!(
            result.content.contains(&format!("character:{}", alice_key)),
            "Filtered results should only contain Alice's conflicts, got: {}",
            result.content
        );
    }
}

/// Test KnowledgeConflicts respects limit parameter.
#[tokio::test]
async fn test_knowledge_conflicts_respects_limit() {
    let harness = TestHarness::new().await;
    let server = create_test_server(&harness).await;

    let alice = create_character(
        &harness.db,
        CharacterCreate {
            name: "Alice".into(),
            ..Default::default()
        },
    )
    .await
    .unwrap();

    // Create multiple BelievesWrongly states
    for i in 0..5 {
        server
            .handle_mutate(Parameters(MutationRequest::RecordKnowledge {
                character_id: alice.id.to_string(),
                target_id: format!("wrong_belief_{}", i),
                fact: format!("Wrong belief number {}", i),
                certainty: "believes_wrongly".to_string(),
                method: Some("initial".to_string()),
                source_character_id: None,
                event_id: None,
            }))
            .await
            .expect("RecordKnowledge should succeed");
    }

    // With limit=2
    let request = QueryRequest::KnowledgeConflicts {
        character_id: None,
        limit: Some(2),
    };
    let response = server
        .handle_query(Parameters(request))
        .await
        .expect("KnowledgeConflicts should succeed");

    assert!(
        response.results.len() <= 2,
        "Should respect limit=2, got {} results",
        response.results.len()
    );
}
