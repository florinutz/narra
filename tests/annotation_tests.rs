//! Integration tests for annotation CRUD, staleness propagation, and emotion queries.

mod common;

use common::harness::TestHarness;
use common::{to_mutation_input, to_query_input};
use narra::models::annotation::{
    delete_entity_annotations, get_annotation, get_entity_annotations, get_stale_annotations,
    mark_annotations_stale, upsert_annotation, AnnotationCreate,
};
use narra::mcp::{MutationRequest, NarraServer, QueryRequest};
use rmcp::handler::server::wrapper::Parameters;

/// Helper to create a character and return its ID.
async fn create_test_character(server: &NarraServer, name: &str) -> String {
    let request = MutationRequest::CreateCharacter {
        id: None,
        name: name.to_string(),
        role: Some("Test Role".to_string()),
        aliases: None,
        description: Some("A test character".to_string()),
        profile: None,
    };
    let result = server
        .mutate(Parameters(to_mutation_input(request)))
        .await
        .expect("create character should succeed");
    result.0.entity.id
}

/// Helper to create a sample annotation for testing.
fn sample_annotation(entity_id: &str) -> AnnotationCreate {
    AnnotationCreate {
        entity_id: entity_id.to_string(),
        model_type: "emotion".to_string(),
        model_version: "test-v1".to_string(),
        output: serde_json::json!({
            "scores": [
                {"label": "joy", "score": 0.9},
                {"label": "sadness", "score": 0.1}
            ],
            "dominant": "joy",
            "active_count": 1
        }),
    }
}

// =============================================================================
// Annotation CRUD
// =============================================================================

#[tokio::test]
async fn test_upsert_and_get_annotation() {
    let harness = TestHarness::new().await;
    let data = sample_annotation("character:alice");

    let created = upsert_annotation(&harness.db, data)
        .await
        .expect("upsert should succeed");

    assert_eq!(created.entity_id, "character:alice");
    assert_eq!(created.model_type, "emotion");
    assert_eq!(created.model_version, "test-v1");
    assert!(!created.stale);

    // Retrieve it back
    let fetched = get_annotation(&harness.db, "character:alice", "emotion")
        .await
        .expect("get should succeed");

    let fetched = fetched.expect("annotation should exist");
    assert_eq!(fetched.entity_id, "character:alice");
    assert_eq!(fetched.model_type, "emotion");
    assert_eq!(fetched.output["dominant"], "joy");
}

#[tokio::test]
async fn test_upsert_replaces_existing() {
    let harness = TestHarness::new().await;

    // Create first version
    let v1 = sample_annotation("character:alice");
    upsert_annotation(&harness.db, v1)
        .await
        .expect("upsert v1");

    // Upsert second version — same entity_id + model_type
    let v2 = AnnotationCreate {
        entity_id: "character:alice".to_string(),
        model_type: "emotion".to_string(),
        model_version: "test-v2".to_string(),
        output: serde_json::json!({"dominant": "anger", "scores": [], "active_count": 0}),
    };
    upsert_annotation(&harness.db, v2)
        .await
        .expect("upsert v2");

    // Should only have one annotation
    let all = get_entity_annotations(&harness.db, "character:alice")
        .await
        .expect("get all");
    assert_eq!(all.len(), 1);
    assert_eq!(all[0].model_version, "test-v2");
    assert_eq!(all[0].output["dominant"], "anger");
}

#[tokio::test]
async fn test_get_nonexistent_annotation() {
    let harness = TestHarness::new().await;

    let result = get_annotation(&harness.db, "character:nobody", "emotion")
        .await
        .expect("get should succeed");
    assert!(result.is_none());
}

#[tokio::test]
async fn test_get_entity_annotations_multiple_model_types() {
    let harness = TestHarness::new().await;

    // Create annotations with different model types for the same entity
    let emotion = sample_annotation("character:alice");
    upsert_annotation(&harness.db, emotion)
        .await
        .expect("upsert emotion");

    let theme = AnnotationCreate {
        entity_id: "character:alice".to_string(),
        model_type: "theme".to_string(),
        model_version: "test-v1".to_string(),
        output: serde_json::json!({"themes": ["betrayal", "loyalty"]}),
    };
    upsert_annotation(&harness.db, theme)
        .await
        .expect("upsert theme");

    let all = get_entity_annotations(&harness.db, "character:alice")
        .await
        .expect("get all");
    assert_eq!(all.len(), 2);

    let model_types: Vec<&str> = all.iter().map(|a| a.model_type.as_str()).collect();
    assert!(model_types.contains(&"emotion"));
    assert!(model_types.contains(&"theme"));
}

#[tokio::test]
async fn test_mark_annotations_stale() {
    let harness = TestHarness::new().await;

    // Create fresh annotation
    let data = sample_annotation("character:alice");
    upsert_annotation(&harness.db, data)
        .await
        .expect("upsert");

    // Verify it's fresh
    let fresh = get_annotation(&harness.db, "character:alice", "emotion")
        .await
        .expect("get")
        .expect("exists");
    assert!(!fresh.stale);

    // Mark stale
    let count = mark_annotations_stale(&harness.db, "character:alice")
        .await
        .expect("mark stale");
    assert_eq!(count, 1);

    // Verify it's stale now
    let stale = get_annotation(&harness.db, "character:alice", "emotion")
        .await
        .expect("get")
        .expect("exists");
    assert!(stale.stale);
}

#[tokio::test]
async fn test_mark_stale_idempotent() {
    let harness = TestHarness::new().await;

    let data = sample_annotation("character:alice");
    upsert_annotation(&harness.db, data)
        .await
        .expect("upsert");

    // Mark stale twice
    let count1 = mark_annotations_stale(&harness.db, "character:alice")
        .await
        .expect("mark stale 1");
    assert_eq!(count1, 1);

    let count2 = mark_annotations_stale(&harness.db, "character:alice")
        .await
        .expect("mark stale 2");
    assert_eq!(count2, 0, "already stale — should not re-mark");
}

#[tokio::test]
async fn test_mark_stale_no_annotations() {
    let harness = TestHarness::new().await;

    // No annotations exist for this entity
    let count = mark_annotations_stale(&harness.db, "character:nobody")
        .await
        .expect("mark stale");
    assert_eq!(count, 0);
}

#[tokio::test]
async fn test_get_stale_annotations() {
    let harness = TestHarness::new().await;

    // Create two annotations for different entities
    let a1 = sample_annotation("character:alice");
    upsert_annotation(&harness.db, a1).await.expect("upsert a1");

    let a2 = AnnotationCreate {
        entity_id: "character:bob".to_string(),
        model_type: "emotion".to_string(),
        model_version: "test-v1".to_string(),
        output: serde_json::json!({"dominant": "neutral", "scores": [], "active_count": 0}),
    };
    upsert_annotation(&harness.db, a2).await.expect("upsert a2");

    // Mark only alice stale
    mark_annotations_stale(&harness.db, "character:alice")
        .await
        .expect("mark stale");

    // Get all stale
    let stale = get_stale_annotations(&harness.db, None, 100)
        .await
        .expect("get stale");
    assert_eq!(stale.len(), 1);
    assert_eq!(stale[0].entity_id, "character:alice");

    // Filter by model type
    let stale_emotions = get_stale_annotations(&harness.db, Some("emotion"), 100)
        .await
        .expect("get stale emotions");
    assert_eq!(stale_emotions.len(), 1);

    let stale_themes = get_stale_annotations(&harness.db, Some("theme"), 100)
        .await
        .expect("get stale themes");
    assert_eq!(stale_themes.len(), 0);
}

#[tokio::test]
async fn test_get_stale_annotations_respects_limit() {
    let harness = TestHarness::new().await;

    // Create 3 annotations and mark them all stale
    for name in &["alice", "bob", "carol"] {
        let data = AnnotationCreate {
            entity_id: format!("character:{}", name),
            model_type: "emotion".to_string(),
            model_version: "test-v1".to_string(),
            output: serde_json::json!({"dominant": "neutral", "scores": [], "active_count": 0}),
        };
        upsert_annotation(&harness.db, data).await.expect("upsert");
        mark_annotations_stale(&harness.db, &format!("character:{}", name))
            .await
            .expect("mark stale");
    }

    let limited = get_stale_annotations(&harness.db, None, 2)
        .await
        .expect("get stale limited");
    assert_eq!(limited.len(), 2);
}

#[tokio::test]
async fn test_delete_entity_annotations() {
    let harness = TestHarness::new().await;

    // Create annotations for alice
    let emotion = sample_annotation("character:alice");
    upsert_annotation(&harness.db, emotion).await.expect("upsert emotion");

    let theme = AnnotationCreate {
        entity_id: "character:alice".to_string(),
        model_type: "theme".to_string(),
        model_version: "test-v1".to_string(),
        output: serde_json::json!({"themes": ["identity"]}),
    };
    upsert_annotation(&harness.db, theme).await.expect("upsert theme");

    // Delete all annotations for alice
    let deleted = delete_entity_annotations(&harness.db, "character:alice")
        .await
        .expect("delete");
    assert_eq!(deleted, 2);

    // Verify gone
    let remaining = get_entity_annotations(&harness.db, "character:alice")
        .await
        .expect("get all");
    assert!(remaining.is_empty());
}

#[tokio::test]
async fn test_delete_nonexistent_annotations() {
    let harness = TestHarness::new().await;

    let deleted = delete_entity_annotations(&harness.db, "character:nobody")
        .await
        .expect("delete");
    assert_eq!(deleted, 0);
}

// =============================================================================
// Staleness propagation through mutations
// =============================================================================

#[tokio::test]
async fn test_entity_update_marks_annotations_stale() {
    let harness = TestHarness::new().await;
    let server = common::create_test_server(&harness).await;

    // Create a character
    let char_id = create_test_character(&server, "Staleness Test").await;

    // Manually insert an annotation for this character
    let data = sample_annotation(&char_id);
    upsert_annotation(&harness.db, data)
        .await
        .expect("upsert annotation");

    // Verify fresh
    let fresh = get_annotation(&harness.db, &char_id, "emotion")
        .await
        .expect("get")
        .expect("exists");
    assert!(!fresh.stale);

    // Update the character (triggers staleness hook)
    let update = MutationRequest::Update {
        entity_id: char_id.clone(),
        fields: serde_json::json!({"description": "Updated description"}),
    };
    server
        .mutate(Parameters(to_mutation_input(update)))
        .await
        .expect("update should succeed");

    // Annotation should now be stale
    let stale = get_annotation(&harness.db, &char_id, "emotion")
        .await
        .expect("get")
        .expect("exists");
    assert!(stale.stale, "annotation should be stale after entity update");
}

#[tokio::test]
async fn test_entity_delete_removes_annotations() {
    let harness = TestHarness::new().await;
    let server = common::create_test_server(&harness).await;

    // Create a character
    let char_id = create_test_character(&server, "Delete Test").await;

    // Insert annotation
    let data = sample_annotation(&char_id);
    upsert_annotation(&harness.db, data)
        .await
        .expect("upsert annotation");

    // Delete the character
    let delete = MutationRequest::Delete {
        entity_id: char_id.clone(),
        hard: None,
    };
    server
        .mutate(Parameters(to_mutation_input(delete)))
        .await
        .expect("delete should succeed");

    // Annotation should be gone
    let remaining = get_entity_annotations(&harness.db, &char_id)
        .await
        .expect("get all");
    assert!(
        remaining.is_empty(),
        "annotations should be deleted when entity is deleted"
    );
}

// =============================================================================
// Emotion query handler
// =============================================================================

#[tokio::test]
async fn test_emotions_query_nonexistent_entity() {
    let harness = TestHarness::new().await;
    let server = common::create_test_server(&harness).await;

    // Query emotions for an entity that doesn't exist.
    // If model is available: should fail with "Entity not found"
    // If model is unavailable: should fail with "not available"
    let request = QueryRequest::Emotions {
        entity_id: "character:nonexistent".to_string(),
    };

    let result = server.query(Parameters(to_query_input(request))).await;
    assert!(result.is_err(), "should error for nonexistent entity");
}

#[tokio::test]
async fn test_emotions_query_serialization() {
    // Verify the Emotions variant round-trips through serde correctly
    let request = QueryRequest::Emotions {
        entity_id: "character:alice".to_string(),
    };
    let json = serde_json::to_value(&request).expect("serialize");
    assert_eq!(json["operation"], "emotions");
    assert_eq!(json["entity_id"], "character:alice");

    let deserialized: QueryRequest = serde_json::from_value(json).expect("deserialize");
    match deserialized {
        QueryRequest::Emotions { entity_id } => assert_eq!(entity_id, "character:alice"),
        _ => panic!("expected Emotions variant"),
    }
}

// =============================================================================
// Annotation data integrity
// =============================================================================

#[tokio::test]
async fn test_annotation_upsert_resets_stale_flag() {
    let harness = TestHarness::new().await;

    // Create and mark stale
    let data = sample_annotation("character:alice");
    upsert_annotation(&harness.db, data)
        .await
        .expect("upsert");
    mark_annotations_stale(&harness.db, "character:alice")
        .await
        .expect("mark stale");

    // Verify stale
    let stale = get_annotation(&harness.db, "character:alice", "emotion")
        .await
        .expect("get")
        .expect("exists");
    assert!(stale.stale);

    // Upsert fresh annotation — should reset stale flag
    let fresh = sample_annotation("character:alice");
    let created = upsert_annotation(&harness.db, fresh)
        .await
        .expect("upsert fresh");
    assert!(!created.stale, "upsert should reset stale flag to false");
}

#[tokio::test]
async fn test_annotation_computed_at_is_set() {
    let harness = TestHarness::new().await;

    let data = sample_annotation("character:alice");
    let created = upsert_annotation(&harness.db, data)
        .await
        .expect("upsert");

    // computed_at should be non-default (SurrealDB sets it to time::now())
    // Just verify the field is populated and serializable
    let json = serde_json::to_value(&created.computed_at).expect("serialize computed_at");
    assert!(json.is_string(), "computed_at should serialize as a datetime string");
}

#[tokio::test]
async fn test_annotations_isolated_between_entities() {
    let harness = TestHarness::new().await;

    // Create annotations for two different entities
    let a1 = sample_annotation("character:alice");
    upsert_annotation(&harness.db, a1).await.expect("upsert a1");

    let a2 = AnnotationCreate {
        entity_id: "character:bob".to_string(),
        model_type: "emotion".to_string(),
        model_version: "test-v1".to_string(),
        output: serde_json::json!({"dominant": "anger", "scores": [], "active_count": 0}),
    };
    upsert_annotation(&harness.db, a2).await.expect("upsert a2");

    // Mark alice stale — should not affect bob
    mark_annotations_stale(&harness.db, "character:alice")
        .await
        .expect("mark stale");

    let alice = get_annotation(&harness.db, "character:alice", "emotion")
        .await
        .expect("get alice")
        .expect("alice exists");
    assert!(alice.stale);

    let bob = get_annotation(&harness.db, "character:bob", "emotion")
        .await
        .expect("get bob")
        .expect("bob exists");
    assert!(!bob.stale, "bob's annotation should remain fresh");

    // Delete alice — should not affect bob
    delete_entity_annotations(&harness.db, "character:alice")
        .await
        .expect("delete alice");

    let bob_still = get_annotation(&harness.db, "character:bob", "emotion")
        .await
        .expect("get bob")
        .expect("bob still exists");
    assert_eq!(bob_still.output["dominant"], "anger");
}
