//! Integration tests for annotation CRUD, staleness propagation, emotion, and theme queries.

mod common;

use common::harness::TestHarness;
use common::{to_mutation_input, to_query_input};
use narra::mcp::{MutationRequest, NarraServer, QueryRequest};
use narra::models::annotation::{
    delete_entity_annotations, get_annotation, get_entity_annotations, get_stale_annotations,
    mark_annotations_stale, upsert_annotation, AnnotationCreate, NerEntity, NerOutput, ThemeOutput,
    ThemeScore,
};
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
        .handle_mutate(Parameters(to_mutation_input(request)))
        .await
        .expect("create character should succeed");
    result.entity.id
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
    upsert_annotation(&harness.db, v1).await.expect("upsert v1");

    // Upsert second version — same entity_id + model_type
    let v2 = AnnotationCreate {
        entity_id: "character:alice".to_string(),
        model_type: "emotion".to_string(),
        model_version: "test-v2".to_string(),
        output: serde_json::json!({"dominant": "anger", "scores": [], "active_count": 0}),
    };
    upsert_annotation(&harness.db, v2).await.expect("upsert v2");

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
    upsert_annotation(&harness.db, data).await.expect("upsert");

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
    upsert_annotation(&harness.db, data).await.expect("upsert");

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
    upsert_annotation(&harness.db, emotion)
        .await
        .expect("upsert emotion");

    let theme = AnnotationCreate {
        entity_id: "character:alice".to_string(),
        model_type: "theme".to_string(),
        model_version: "test-v1".to_string(),
        output: serde_json::json!({"themes": ["identity"]}),
    };
    upsert_annotation(&harness.db, theme)
        .await
        .expect("upsert theme");

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
        .handle_mutate(Parameters(to_mutation_input(update)))
        .await
        .expect("update should succeed");

    // Annotation should now be stale
    let stale = get_annotation(&harness.db, &char_id, "emotion")
        .await
        .expect("get")
        .expect("exists");
    assert!(
        stale.stale,
        "annotation should be stale after entity update"
    );
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
        .handle_mutate(Parameters(to_mutation_input(delete)))
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

    let result = server
        .handle_query(Parameters(to_query_input(request)))
        .await;
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
    upsert_annotation(&harness.db, data).await.expect("upsert");
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
    let created = upsert_annotation(&harness.db, data).await.expect("upsert");

    // computed_at should be non-default (SurrealDB sets it to time::now())
    // Just verify the field is populated and serializable
    let json = serde_json::to_value(&created.computed_at).expect("serialize computed_at");
    assert!(
        json.is_string(),
        "computed_at should serialize as a datetime string"
    );
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

// =============================================================================
// Theme query handler
// =============================================================================

#[tokio::test]
async fn test_themes_query_serialization() {
    // Verify the Themes variant round-trips through serde correctly (no custom themes)
    let request = QueryRequest::Themes {
        entity_id: "character:alice".to_string(),
        themes: None,
    };
    let json = serde_json::to_value(&request).expect("serialize");
    assert_eq!(json["operation"], "themes");
    assert_eq!(json["entity_id"], "character:alice");
    assert!(json.get("themes").is_none() || json["themes"].is_null());

    let deserialized: QueryRequest = serde_json::from_value(json).expect("deserialize");
    match deserialized {
        QueryRequest::Themes { entity_id, themes } => {
            assert_eq!(entity_id, "character:alice");
            assert!(themes.is_none());
        }
        _ => panic!("expected Themes variant"),
    }
}

#[tokio::test]
async fn test_themes_query_serialization_with_custom_themes() {
    // Verify the Themes variant with custom themes
    let request = QueryRequest::Themes {
        entity_id: "scene:battle".to_string(),
        themes: Some(vec!["honor".to_string(), "sacrifice".to_string()]),
    };
    let json = serde_json::to_value(&request).expect("serialize");
    assert_eq!(json["operation"], "themes");
    assert_eq!(json["entity_id"], "scene:battle");
    assert_eq!(json["themes"][0], "honor");
    assert_eq!(json["themes"][1], "sacrifice");

    let deserialized: QueryRequest = serde_json::from_value(json).expect("deserialize");
    match deserialized {
        QueryRequest::Themes { entity_id, themes } => {
            assert_eq!(entity_id, "scene:battle");
            let themes = themes.expect("custom themes should be present");
            assert_eq!(themes, vec!["honor", "sacrifice"]);
        }
        _ => panic!("expected Themes variant"),
    }
}

#[tokio::test]
async fn test_themes_query_nonexistent_entity() {
    let harness = TestHarness::new().await;
    let server = common::create_test_server(&harness).await;

    // Query themes for an entity that doesn't exist.
    // If model is available: should fail with "Entity not found"
    // If model is unavailable: should fail with "not available"
    let request = QueryRequest::Themes {
        entity_id: "character:nonexistent".to_string(),
        themes: None,
    };

    let result = server
        .handle_query(Parameters(to_query_input(request)))
        .await;
    assert!(result.is_err(), "should error for nonexistent entity");
}

// =============================================================================
// Theme annotation caching
// =============================================================================

#[tokio::test]
async fn test_theme_annotation_upsert_and_get() {
    let harness = TestHarness::new().await;

    let theme_output = ThemeOutput {
        themes: vec![
            ThemeScore {
                label: "betrayal".to_string(),
                score: 0.85,
            },
            ThemeScore {
                label: "love".to_string(),
                score: 0.3,
            },
        ],
        dominant: "betrayal".to_string(),
        active_count: 1,
    };

    let data = AnnotationCreate {
        entity_id: "character:alice".to_string(),
        model_type: "theme".to_string(),
        model_version: "nli-roberta-base-v1".to_string(),
        output: serde_json::to_value(&theme_output).expect("serialize theme output"),
    };

    let created = upsert_annotation(&harness.db, data)
        .await
        .expect("upsert should succeed");

    assert_eq!(created.entity_id, "character:alice");
    assert_eq!(created.model_type, "theme");
    assert!(!created.stale);

    // Retrieve and deserialize
    let fetched = get_annotation(&harness.db, "character:alice", "theme")
        .await
        .expect("get should succeed")
        .expect("annotation should exist");

    let parsed: ThemeOutput =
        serde_json::from_value(fetched.output).expect("deserialize theme output");
    assert_eq!(parsed.dominant, "betrayal");
    assert_eq!(parsed.themes.len(), 2);
    assert_eq!(parsed.active_count, 1);
}

#[tokio::test]
async fn test_theme_and_emotion_annotations_coexist() {
    let harness = TestHarness::new().await;

    // Create emotion annotation
    let emotion = sample_annotation("character:alice");
    upsert_annotation(&harness.db, emotion)
        .await
        .expect("upsert emotion");

    // Create theme annotation for same entity
    let theme = AnnotationCreate {
        entity_id: "character:alice".to_string(),
        model_type: "theme".to_string(),
        model_version: "nli-roberta-base-v1".to_string(),
        output: serde_json::json!({
            "themes": [{"label": "betrayal", "score": 0.8}],
            "dominant": "betrayal",
            "active_count": 1
        }),
    };
    upsert_annotation(&harness.db, theme)
        .await
        .expect("upsert theme");

    // Both should exist
    let all = get_entity_annotations(&harness.db, "character:alice")
        .await
        .expect("get all");
    assert_eq!(all.len(), 2);

    // Retrieve each independently
    let emotion = get_annotation(&harness.db, "character:alice", "emotion")
        .await
        .expect("get emotion")
        .expect("emotion exists");
    assert_eq!(emotion.output["dominant"], "joy");

    let theme = get_annotation(&harness.db, "character:alice", "theme")
        .await
        .expect("get theme")
        .expect("theme exists");
    assert_eq!(theme.output["dominant"], "betrayal");
}

#[tokio::test]
async fn test_entity_update_marks_both_annotation_types_stale() {
    let harness = TestHarness::new().await;
    let server = common::create_test_server(&harness).await;

    // Create a character
    let char_id = create_test_character(&server, "Multi Annotation").await;

    // Insert both emotion and theme annotations
    let emotion = sample_annotation(&char_id);
    upsert_annotation(&harness.db, emotion)
        .await
        .expect("upsert emotion");

    let theme = AnnotationCreate {
        entity_id: char_id.clone(),
        model_type: "theme".to_string(),
        model_version: "nli-roberta-base-v1".to_string(),
        output: serde_json::json!({"themes": [], "dominant": "none", "active_count": 0}),
    };
    upsert_annotation(&harness.db, theme)
        .await
        .expect("upsert theme");

    // Update the character (triggers staleness hook)
    let update = MutationRequest::Update {
        entity_id: char_id.clone(),
        fields: serde_json::json!({"description": "Changed description"}),
    };
    server
        .handle_mutate(Parameters(to_mutation_input(update)))
        .await
        .expect("update should succeed");

    // Both annotations should be stale
    let emotion = get_annotation(&harness.db, &char_id, "emotion")
        .await
        .expect("get emotion")
        .expect("emotion exists");
    assert!(emotion.stale, "emotion annotation should be stale");

    let theme = get_annotation(&harness.db, &char_id, "theme")
        .await
        .expect("get theme")
        .expect("theme exists");
    assert!(theme.stale, "theme annotation should be stale");
}

#[tokio::test]
async fn test_entity_delete_removes_all_annotation_types() {
    let harness = TestHarness::new().await;
    let server = common::create_test_server(&harness).await;

    // Create a character
    let char_id = create_test_character(&server, "Delete Both").await;

    // Insert both annotation types
    let emotion = sample_annotation(&char_id);
    upsert_annotation(&harness.db, emotion)
        .await
        .expect("upsert emotion");

    let theme = AnnotationCreate {
        entity_id: char_id.clone(),
        model_type: "theme".to_string(),
        model_version: "nli-roberta-base-v1".to_string(),
        output: serde_json::json!({"themes": [], "dominant": "none", "active_count": 0}),
    };
    upsert_annotation(&harness.db, theme)
        .await
        .expect("upsert theme");

    // Verify both exist
    let all = get_entity_annotations(&harness.db, &char_id)
        .await
        .expect("get all");
    assert_eq!(all.len(), 2);

    // Delete the character
    let delete = MutationRequest::Delete {
        entity_id: char_id.clone(),
        hard: None,
    };
    server
        .handle_mutate(Parameters(to_mutation_input(delete)))
        .await
        .expect("delete should succeed");

    // All annotations should be gone
    let remaining = get_entity_annotations(&harness.db, &char_id)
        .await
        .expect("get all");
    assert!(
        remaining.is_empty(),
        "both annotation types should be deleted when entity is deleted"
    );
}

#[tokio::test]
async fn test_get_stale_annotations_filters_by_theme_model_type() {
    let harness = TestHarness::new().await;

    // Create emotion and theme annotations
    let emotion = sample_annotation("character:alice");
    upsert_annotation(&harness.db, emotion)
        .await
        .expect("upsert emotion");

    let theme = AnnotationCreate {
        entity_id: "character:alice".to_string(),
        model_type: "theme".to_string(),
        model_version: "nli-roberta-base-v1".to_string(),
        output: serde_json::json!({"themes": [], "dominant": "none", "active_count": 0}),
    };
    upsert_annotation(&harness.db, theme)
        .await
        .expect("upsert theme");

    // Mark all stale
    mark_annotations_stale(&harness.db, "character:alice")
        .await
        .expect("mark stale");

    // Filter by theme
    let stale_themes = get_stale_annotations(&harness.db, Some("theme"), 100)
        .await
        .expect("get stale themes");
    assert_eq!(stale_themes.len(), 1);
    assert_eq!(stale_themes[0].model_type, "theme");

    // Filter by emotion
    let stale_emotions = get_stale_annotations(&harness.db, Some("emotion"), 100)
        .await
        .expect("get stale emotions");
    assert_eq!(stale_emotions.len(), 1);
    assert_eq!(stale_emotions[0].model_type, "emotion");

    // All stale
    let all_stale = get_stale_annotations(&harness.db, None, 100)
        .await
        .expect("get all stale");
    assert_eq!(all_stale.len(), 2);
}

#[tokio::test]
async fn test_theme_annotation_output_preserves_full_structure() {
    let harness = TestHarness::new().await;

    // Create a rich theme output with multiple themes
    let theme_output = ThemeOutput {
        themes: vec![
            ThemeScore {
                label: "betrayal".to_string(),
                score: 0.92,
            },
            ThemeScore {
                label: "power".to_string(),
                score: 0.71,
            },
            ThemeScore {
                label: "deception".to_string(),
                score: 0.55,
            },
            ThemeScore {
                label: "love".to_string(),
                score: 0.12,
            },
        ],
        dominant: "betrayal".to_string(),
        active_count: 3,
    };

    let data = AnnotationCreate {
        entity_id: "scene:confrontation".to_string(),
        model_type: "theme".to_string(),
        model_version: "nli-roberta-base-v1".to_string(),
        output: serde_json::to_value(&theme_output).expect("serialize"),
    };

    upsert_annotation(&harness.db, data).await.expect("upsert");

    // Retrieve and verify full structure survives round-trip through SurrealDB
    let fetched = get_annotation(&harness.db, "scene:confrontation", "theme")
        .await
        .expect("get")
        .expect("exists");

    let parsed: ThemeOutput = serde_json::from_value(fetched.output).expect("deserialize");

    assert_eq!(parsed.dominant, "betrayal");
    assert_eq!(parsed.active_count, 3);
    assert_eq!(parsed.themes.len(), 4);

    // Verify individual scores survived
    assert_eq!(parsed.themes[0].label, "betrayal");
    assert!((parsed.themes[0].score - 0.92).abs() < 0.001);
    assert_eq!(parsed.themes[3].label, "love");
    assert!((parsed.themes[3].score - 0.12).abs() < 0.001);
}

// =============================================================================
// NER query handler
// =============================================================================

#[tokio::test]
async fn test_extract_entities_query_serialization() {
    // Verify the ExtractEntities variant round-trips through serde correctly
    let request = QueryRequest::ExtractEntities {
        entity_id: "character:alice".to_string(),
    };
    let json = serde_json::to_value(&request).expect("serialize");
    assert_eq!(json["operation"], "extract_entities");
    assert_eq!(json["entity_id"], "character:alice");

    let deserialized: QueryRequest = serde_json::from_value(json).expect("deserialize");
    match deserialized {
        QueryRequest::ExtractEntities { entity_id } => {
            assert_eq!(entity_id, "character:alice");
        }
        _ => panic!("expected ExtractEntities variant"),
    }
}

#[tokio::test]
async fn test_extract_entities_query_nonexistent_entity() {
    let harness = TestHarness::new().await;
    let server = common::create_test_server(&harness).await;

    // Query entities for a character that doesn't exist.
    // If model is available: should fail with "Entity not found"
    // If model is unavailable: should fail with "not available"
    let request = QueryRequest::ExtractEntities {
        entity_id: "character:nonexistent".to_string(),
    };

    let result = server
        .handle_query(Parameters(to_query_input(request)))
        .await;
    assert!(result.is_err(), "should error for nonexistent entity");
}

// =============================================================================
// NER annotation caching
// =============================================================================

#[tokio::test]
async fn test_ner_annotation_upsert_and_get() {
    let harness = TestHarness::new().await;

    let ner_output = NerOutput {
        entities: vec![
            NerEntity {
                text: "Alice".to_string(),
                label: "PER".to_string(),
                score: 0.98,
                start: 0,
                end: 5,
            },
            NerEntity {
                text: "Winterfell".to_string(),
                label: "LOC".to_string(),
                score: 0.91,
                start: 20,
                end: 30,
            },
        ],
        entity_count: 2,
    };

    let data = AnnotationCreate {
        entity_id: "scene:opening".to_string(),
        model_type: "ner".to_string(),
        model_version: "bert-base-ner-v1".to_string(),
        output: serde_json::to_value(&ner_output).expect("serialize ner output"),
    };

    let created = upsert_annotation(&harness.db, data)
        .await
        .expect("upsert should succeed");

    assert_eq!(created.entity_id, "scene:opening");
    assert_eq!(created.model_type, "ner");
    assert!(!created.stale);

    // Retrieve and deserialize
    let fetched = get_annotation(&harness.db, "scene:opening", "ner")
        .await
        .expect("get should succeed")
        .expect("annotation should exist");

    let parsed: NerOutput = serde_json::from_value(fetched.output).expect("deserialize ner output");
    assert_eq!(parsed.entity_count, 2);
    assert_eq!(parsed.entities[0].text, "Alice");
    assert_eq!(parsed.entities[0].label, "PER");
    assert_eq!(parsed.entities[1].text, "Winterfell");
    assert_eq!(parsed.entities[1].label, "LOC");
}

#[tokio::test]
async fn test_all_three_annotation_types_coexist() {
    let harness = TestHarness::new().await;

    // Create emotion annotation
    let emotion = sample_annotation("character:alice");
    upsert_annotation(&harness.db, emotion)
        .await
        .expect("upsert emotion");

    // Create theme annotation
    let theme = AnnotationCreate {
        entity_id: "character:alice".to_string(),
        model_type: "theme".to_string(),
        model_version: "nli-roberta-base-v1".to_string(),
        output: serde_json::json!({
            "themes": [{"label": "betrayal", "score": 0.8}],
            "dominant": "betrayal",
            "active_count": 1
        }),
    };
    upsert_annotation(&harness.db, theme)
        .await
        .expect("upsert theme");

    // Create NER annotation
    let ner = AnnotationCreate {
        entity_id: "character:alice".to_string(),
        model_type: "ner".to_string(),
        model_version: "bert-base-ner-v1".to_string(),
        output: serde_json::json!({
            "entities": [{"text": "Bob", "label": "PER", "score": 0.95, "start": 0, "end": 3}],
            "entity_count": 1
        }),
    };
    upsert_annotation(&harness.db, ner)
        .await
        .expect("upsert ner");

    // All three should exist
    let all = get_entity_annotations(&harness.db, "character:alice")
        .await
        .expect("get all");
    assert_eq!(all.len(), 3);

    let model_types: Vec<&str> = all.iter().map(|a| a.model_type.as_str()).collect();
    assert!(model_types.contains(&"emotion"));
    assert!(model_types.contains(&"theme"));
    assert!(model_types.contains(&"ner"));

    // Retrieve each independently
    let emotion = get_annotation(&harness.db, "character:alice", "emotion")
        .await
        .expect("get emotion")
        .expect("emotion exists");
    assert_eq!(emotion.output["dominant"], "joy");

    let theme = get_annotation(&harness.db, "character:alice", "theme")
        .await
        .expect("get theme")
        .expect("theme exists");
    assert_eq!(theme.output["dominant"], "betrayal");

    let ner = get_annotation(&harness.db, "character:alice", "ner")
        .await
        .expect("get ner")
        .expect("ner exists");
    assert_eq!(ner.output["entity_count"], 1);
}

#[tokio::test]
async fn test_entity_update_marks_all_three_annotation_types_stale() {
    let harness = TestHarness::new().await;
    let server = common::create_test_server(&harness).await;

    let char_id = create_test_character(&server, "Triple Annotation").await;

    // Insert all three annotation types
    let emotion = sample_annotation(&char_id);
    upsert_annotation(&harness.db, emotion)
        .await
        .expect("upsert emotion");

    let theme = AnnotationCreate {
        entity_id: char_id.clone(),
        model_type: "theme".to_string(),
        model_version: "nli-roberta-base-v1".to_string(),
        output: serde_json::json!({"themes": [], "dominant": "none", "active_count": 0}),
    };
    upsert_annotation(&harness.db, theme)
        .await
        .expect("upsert theme");

    let ner = AnnotationCreate {
        entity_id: char_id.clone(),
        model_type: "ner".to_string(),
        model_version: "bert-base-ner-v1".to_string(),
        output: serde_json::json!({"entities": [], "entity_count": 0}),
    };
    upsert_annotation(&harness.db, ner)
        .await
        .expect("upsert ner");

    // Update the character
    let update = MutationRequest::Update {
        entity_id: char_id.clone(),
        fields: serde_json::json!({"description": "Changed again"}),
    };
    server
        .handle_mutate(Parameters(to_mutation_input(update)))
        .await
        .expect("update should succeed");

    // All three annotations should be stale
    for model_type in &["emotion", "theme", "ner"] {
        let ann = get_annotation(&harness.db, &char_id, model_type)
            .await
            .expect(&format!("get {}", model_type))
            .expect(&format!("{} exists", model_type));
        assert!(
            ann.stale,
            "{} annotation should be stale after entity update",
            model_type
        );
    }
}

#[tokio::test]
async fn test_entity_delete_removes_all_three_annotation_types() {
    let harness = TestHarness::new().await;
    let server = common::create_test_server(&harness).await;

    let char_id = create_test_character(&server, "Delete All Three").await;

    // Insert all three annotation types
    let emotion = sample_annotation(&char_id);
    upsert_annotation(&harness.db, emotion)
        .await
        .expect("upsert emotion");

    let theme = AnnotationCreate {
        entity_id: char_id.clone(),
        model_type: "theme".to_string(),
        model_version: "nli-roberta-base-v1".to_string(),
        output: serde_json::json!({"themes": [], "dominant": "none", "active_count": 0}),
    };
    upsert_annotation(&harness.db, theme)
        .await
        .expect("upsert theme");

    let ner = AnnotationCreate {
        entity_id: char_id.clone(),
        model_type: "ner".to_string(),
        model_version: "bert-base-ner-v1".to_string(),
        output: serde_json::json!({"entities": [], "entity_count": 0}),
    };
    upsert_annotation(&harness.db, ner)
        .await
        .expect("upsert ner");

    // Verify all three exist
    let all = get_entity_annotations(&harness.db, &char_id)
        .await
        .expect("get all");
    assert_eq!(all.len(), 3);

    // Delete the character
    let delete = MutationRequest::Delete {
        entity_id: char_id.clone(),
        hard: None,
    };
    server
        .handle_mutate(Parameters(to_mutation_input(delete)))
        .await
        .expect("delete should succeed");

    // All annotations should be gone
    let remaining = get_entity_annotations(&harness.db, &char_id)
        .await
        .expect("get all");
    assert!(
        remaining.is_empty(),
        "all three annotation types should be deleted when entity is deleted"
    );
}

#[tokio::test]
async fn test_get_stale_annotations_filters_by_ner_model_type() {
    let harness = TestHarness::new().await;

    // Create all three annotation types
    let emotion = sample_annotation("character:alice");
    upsert_annotation(&harness.db, emotion)
        .await
        .expect("upsert emotion");

    let theme = AnnotationCreate {
        entity_id: "character:alice".to_string(),
        model_type: "theme".to_string(),
        model_version: "nli-roberta-base-v1".to_string(),
        output: serde_json::json!({"themes": [], "dominant": "none", "active_count": 0}),
    };
    upsert_annotation(&harness.db, theme)
        .await
        .expect("upsert theme");

    let ner = AnnotationCreate {
        entity_id: "character:alice".to_string(),
        model_type: "ner".to_string(),
        model_version: "bert-base-ner-v1".to_string(),
        output: serde_json::json!({"entities": [], "entity_count": 0}),
    };
    upsert_annotation(&harness.db, ner)
        .await
        .expect("upsert ner");

    // Mark all stale
    mark_annotations_stale(&harness.db, "character:alice")
        .await
        .expect("mark stale");

    // Filter by ner
    let stale_ner = get_stale_annotations(&harness.db, Some("ner"), 100)
        .await
        .expect("get stale ner");
    assert_eq!(stale_ner.len(), 1);
    assert_eq!(stale_ner[0].model_type, "ner");

    // All stale should return 3
    let all_stale = get_stale_annotations(&harness.db, None, 100)
        .await
        .expect("get all stale");
    assert_eq!(all_stale.len(), 3);
}

#[tokio::test]
async fn test_ner_annotation_output_preserves_full_structure() {
    let harness = TestHarness::new().await;

    let ner_output = NerOutput {
        entities: vec![
            NerEntity {
                text: "Elena".to_string(),
                label: "PER".to_string(),
                score: 0.99,
                start: 0,
                end: 5,
            },
            NerEntity {
                text: "Matei".to_string(),
                label: "PER".to_string(),
                score: 0.97,
                start: 15,
                end: 20,
            },
            NerEntity {
                text: "Blackwood Manor".to_string(),
                label: "LOC".to_string(),
                score: 0.88,
                start: 30,
                end: 45,
            },
            NerEntity {
                text: "The Order".to_string(),
                label: "ORG".to_string(),
                score: 0.72,
                start: 50,
                end: 59,
            },
        ],
        entity_count: 4,
    };

    let data = AnnotationCreate {
        entity_id: "scene:confrontation".to_string(),
        model_type: "ner".to_string(),
        model_version: "bert-base-ner-v1".to_string(),
        output: serde_json::to_value(&ner_output).expect("serialize"),
    };

    upsert_annotation(&harness.db, data).await.expect("upsert");

    // Retrieve and verify full structure survives round-trip through SurrealDB
    let fetched = get_annotation(&harness.db, "scene:confrontation", "ner")
        .await
        .expect("get")
        .expect("exists");

    let parsed: NerOutput = serde_json::from_value(fetched.output).expect("deserialize");

    assert_eq!(parsed.entity_count, 4);
    assert_eq!(parsed.entities.len(), 4);

    // Verify individual entities survived
    assert_eq!(parsed.entities[0].text, "Elena");
    assert_eq!(parsed.entities[0].label, "PER");
    assert!((parsed.entities[0].score - 0.99).abs() < 0.001);
    assert_eq!(parsed.entities[0].start, 0);
    assert_eq!(parsed.entities[0].end, 5);

    assert_eq!(parsed.entities[2].text, "Blackwood Manor");
    assert_eq!(parsed.entities[2].label, "LOC");

    assert_eq!(parsed.entities[3].text, "The Order");
    assert_eq!(parsed.entities[3].label, "ORG");
    assert!((parsed.entities[3].score - 0.72).abs() < 0.001);
}

// =============================================================================
// MCP endpoint integration: narrative_tensions
// =============================================================================

#[tokio::test]
async fn test_narrative_tensions_query_serialization() {
    // Round-trip with defaults
    let request = QueryRequest::NarrativeTensions {
        limit: None,
        min_severity: None,
    };
    let json = serde_json::to_value(&request).expect("serialize");
    assert_eq!(json["operation"], "narrative_tensions");

    let deserialized: QueryRequest = serde_json::from_value(json).expect("deserialize");
    match deserialized {
        QueryRequest::NarrativeTensions {
            limit,
            min_severity,
        } => {
            assert!(limit.is_none());
            assert!(min_severity.is_none());
        }
        _ => panic!("expected NarrativeTensions variant"),
    }
}

#[tokio::test]
async fn test_narrative_tensions_query_serialization_with_params() {
    let request = QueryRequest::NarrativeTensions {
        limit: Some(5),
        min_severity: Some(0.3),
    };
    let json = serde_json::to_value(&request).expect("serialize");
    assert_eq!(json["operation"], "narrative_tensions");
    assert_eq!(json["limit"], 5);

    let deserialized: QueryRequest = serde_json::from_value(json).expect("deserialize");
    match deserialized {
        QueryRequest::NarrativeTensions {
            limit,
            min_severity,
        } => {
            assert_eq!(limit, Some(5));
            assert!((min_severity.unwrap() - 0.3).abs() < 0.001);
        }
        _ => panic!("expected NarrativeTensions variant"),
    }
}

#[tokio::test]
async fn test_narrative_tensions_via_mcp_empty_world() {
    let harness = TestHarness::new().await;
    let server = common::create_test_server(&harness).await;

    let request = QueryRequest::NarrativeTensions {
        limit: Some(10),
        min_severity: None,
    };
    let result = server
        .handle_query(Parameters(to_query_input(request)))
        .await;

    // Should succeed even with empty world (returns empty results)
    assert!(
        result.is_ok(),
        "Narrative tensions should handle empty world"
    );
    let response = result.unwrap();
    assert_eq!(response.total, 1, "Should have report entity");
}

#[tokio::test]
async fn test_narrative_tensions_via_mcp_with_conflicting_loyalties() {
    use narra::models::relationship::{create_relationship, RelationshipCreate};
    use narra::repository::{EntityRepository, SurrealEntityRepository};

    let harness = TestHarness::new().await;
    let server = common::create_test_server(&harness).await;
    let repo = SurrealEntityRepository::new(harness.db.clone());

    // Build conflicting loyalty triangle: Alice allies with both Bob and Charlie, but Bob and Charlie are rivals
    let alice = repo
        .create_character(common::builders::CharacterBuilder::new("Alice").build())
        .await
        .unwrap();
    let alice_key = alice.id.key().to_string();

    let bob = repo
        .create_character(common::builders::CharacterBuilder::new("Bob").build())
        .await
        .unwrap();
    let bob_key = bob.id.key().to_string();

    let charlie = repo
        .create_character(common::builders::CharacterBuilder::new("Charlie").build())
        .await
        .unwrap();
    let charlie_key = charlie.id.key().to_string();

    // Alice <-> Bob: allies
    create_relationship(
        &harness.db,
        RelationshipCreate {
            from_character_id: alice_key.clone(),
            to_character_id: bob_key.clone(),
            rel_type: "ally".to_string(),
            subtype: None,
            label: Some("close allies".to_string()),
        },
    )
    .await
    .unwrap();

    // Alice <-> Charlie: allies
    create_relationship(
        &harness.db,
        RelationshipCreate {
            from_character_id: alice_key.clone(),
            to_character_id: charlie_key.clone(),
            rel_type: "ally".to_string(),
            subtype: None,
            label: Some("political allies".to_string()),
        },
    )
    .await
    .unwrap();

    // Bob <-> Charlie: rivals
    create_relationship(
        &harness.db,
        RelationshipCreate {
            from_character_id: bob_key.clone(),
            to_character_id: charlie_key.clone(),
            rel_type: "rival".to_string(),
            subtype: None,
            label: Some("bitter enemies".to_string()),
        },
    )
    .await
    .unwrap();

    let request = QueryRequest::NarrativeTensions {
        limit: Some(20),
        min_severity: Some(0.0),
    };
    let result = server
        .handle_query(Parameters(to_query_input(request)))
        .await;

    assert!(result.is_ok(), "Narrative tensions query should succeed");
    let response = result.unwrap();
    assert_eq!(response.total, 1, "Should have report entity");
    // The content should mention tensions found
    let content = &response.results[0].content;
    assert!(
        content.contains("tension") || content.contains("Tension") || content.contains("conflict"),
        "Report should mention tensions: {}",
        content
    );
}

// =============================================================================
// MCP endpoint integration: infer_roles
// =============================================================================

#[tokio::test]
async fn test_infer_roles_query_serialization() {
    let request = QueryRequest::InferRoles { limit: None };
    let json = serde_json::to_value(&request).expect("serialize");
    assert_eq!(json["operation"], "infer_roles");

    let deserialized: QueryRequest = serde_json::from_value(json).expect("deserialize");
    match deserialized {
        QueryRequest::InferRoles { limit } => assert!(limit.is_none()),
        _ => panic!("expected InferRoles variant"),
    }
}

#[tokio::test]
async fn test_infer_roles_query_serialization_with_limit() {
    let request = QueryRequest::InferRoles { limit: Some(10) };
    let json = serde_json::to_value(&request).expect("serialize");
    assert_eq!(json["operation"], "infer_roles");
    assert_eq!(json["limit"], 10);

    let deserialized: QueryRequest = serde_json::from_value(json).expect("deserialize");
    match deserialized {
        QueryRequest::InferRoles { limit } => assert_eq!(limit, Some(10)),
        _ => panic!("expected InferRoles variant"),
    }
}

#[tokio::test]
async fn test_infer_roles_via_mcp_empty_world() {
    let harness = TestHarness::new().await;
    let server = common::create_test_server(&harness).await;

    let request = QueryRequest::InferRoles { limit: Some(10) };
    let result = server
        .handle_query(Parameters(to_query_input(request)))
        .await;

    assert!(result.is_ok(), "Role inference should handle empty world");
    let response = result.unwrap();
    assert_eq!(response.total, 1, "Should have report entity");
}

#[tokio::test]
async fn test_infer_roles_via_mcp_with_social_hub() {
    use narra::models::relationship::{create_relationship, RelationshipCreate};
    use narra::repository::{EntityRepository, SurrealEntityRepository};

    let harness = TestHarness::new().await;
    let server = common::create_test_server(&harness).await;
    let repo = SurrealEntityRepository::new(harness.db.clone());

    // Create a hub character with many connections
    let hub = repo
        .create_character(common::builders::CharacterBuilder::new("Hub").build())
        .await
        .unwrap();
    let hub_key = hub.id.key().to_string();

    for i in 0..5 {
        let spoke = repo
            .create_character(
                common::builders::CharacterBuilder::new(&format!("Spoke{}", i)).build(),
            )
            .await
            .unwrap();
        let spoke_key = spoke.id.key().to_string();

        create_relationship(
            &harness.db,
            RelationshipCreate {
                from_character_id: hub_key.clone(),
                to_character_id: spoke_key,
                rel_type: "ally".to_string(),
                subtype: None,
                label: None,
            },
        )
        .await
        .unwrap();
    }

    let request = QueryRequest::InferRoles { limit: Some(20) };
    let result = server
        .handle_query(Parameters(to_query_input(request)))
        .await;

    assert!(result.is_ok(), "Role inference query should succeed");
    let response = result.unwrap();
    assert_eq!(response.total, 1, "Should have report entity");
    let content = &response.results[0].content;
    // Hub should be identified with a role
    assert!(
        content.contains("Hub"),
        "Report should mention Hub character: {}",
        content
    );
}

// =============================================================================
// MCP endpoint integration: annotate_entities mutation
// =============================================================================

#[tokio::test]
async fn test_annotate_entities_mutation_serialization() {
    let request = MutationRequest::AnnotateEntities {
        entity_types: None,
        run_emotions: true,
        run_themes: true,
        run_ner: true,
        concurrency: None,
    };
    let json = serde_json::to_value(&request).expect("serialize");
    assert_eq!(json["operation"], "annotate_entities");
    assert_eq!(json["run_emotions"], true);
    assert_eq!(json["run_themes"], true);
    assert_eq!(json["run_ner"], true);

    let deserialized: MutationRequest = serde_json::from_value(json).expect("deserialize");
    match deserialized {
        MutationRequest::AnnotateEntities {
            entity_types,
            run_emotions,
            run_themes,
            run_ner,
            concurrency,
        } => {
            assert!(entity_types.is_none());
            assert!(run_emotions);
            assert!(run_themes);
            assert!(run_ner);
            assert!(concurrency.is_none());
        }
        _ => panic!("expected AnnotateEntities variant"),
    }
}

#[tokio::test]
async fn test_annotate_entities_mutation_serialization_with_params() {
    let request = MutationRequest::AnnotateEntities {
        entity_types: Some(vec!["character".to_string(), "event".to_string()]),
        run_emotions: true,
        run_themes: false,
        run_ner: true,
        concurrency: Some(2),
    };
    let json = serde_json::to_value(&request).expect("serialize");
    assert_eq!(json["operation"], "annotate_entities");
    assert_eq!(json["entity_types"][0], "character");
    assert_eq!(json["entity_types"][1], "event");
    assert!(!json["run_themes"].as_bool().unwrap());
    assert_eq!(json["concurrency"], 2);

    let deserialized: MutationRequest = serde_json::from_value(json).expect("deserialize");
    match deserialized {
        MutationRequest::AnnotateEntities {
            entity_types,
            run_emotions,
            run_themes,
            run_ner,
            concurrency,
        } => {
            assert_eq!(
                entity_types,
                Some(vec!["character".to_string(), "event".to_string()])
            );
            assert!(run_emotions);
            assert!(!run_themes);
            assert!(run_ner);
            assert_eq!(concurrency, Some(2));
        }
        _ => panic!("expected AnnotateEntities variant"),
    }
}

#[tokio::test]
async fn test_annotate_entities_via_mcp_empty_world() {
    let harness = TestHarness::new().await;
    let server = common::create_test_server(&harness).await;

    // Run annotation pipeline on empty world — should succeed with 0 processed
    let request = MutationRequest::AnnotateEntities {
        entity_types: None,
        run_emotions: true,
        run_themes: true,
        run_ner: true,
        concurrency: None,
    };
    let result = server
        .handle_mutate(Parameters(to_mutation_input(request)))
        .await;

    assert!(
        result.is_ok(),
        "Annotate should handle empty world: {:?}",
        result.err()
    );
    let response = result.unwrap();
    assert!(
        response.entity.content.contains("0")
            || response.entity.content.to_lowercase().contains("processed"),
        "Should report 0 entities processed: {}",
        response.entity.content
    );
}

#[tokio::test]
async fn test_annotate_entities_via_mcp_with_characters() {
    let harness = TestHarness::new().await;
    let server = common::create_test_server(&harness).await;

    // Create some characters to annotate
    create_test_character(&server, "Alice").await;
    create_test_character(&server, "Bob").await;

    // Run annotation pipeline — ML models are noop in test, so classifiers won't produce
    // results, but the pipeline should still succeed without errors
    let request = MutationRequest::AnnotateEntities {
        entity_types: Some(vec!["character".to_string()]),
        run_emotions: true,
        run_themes: true,
        run_ner: true,
        concurrency: Some(1),
    };
    let result = server
        .handle_mutate(Parameters(to_mutation_input(request)))
        .await;

    assert!(
        result.is_ok(),
        "Annotate should succeed with noop services: {:?}",
        result.err()
    );
}

#[tokio::test]
async fn test_annotate_entities_via_mcp_custom_types() {
    let harness = TestHarness::new().await;
    let server = common::create_test_server(&harness).await;

    // Annotate only events (none exist, but should not error)
    let request = MutationRequest::AnnotateEntities {
        entity_types: Some(vec!["event".to_string()]),
        run_emotions: false,
        run_themes: true,
        run_ner: false,
        concurrency: Some(2),
    };
    let result = server
        .handle_mutate(Parameters(to_mutation_input(request)))
        .await;

    assert!(
        result.is_ok(),
        "Annotate with custom types should succeed: {:?}",
        result.err()
    );
}
