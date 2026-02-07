mod common;

use common::harness::TestHarness;
use narra::mcp::tools::ExportRequest;
use narra::mcp::types::{ConflictMode, NarraImport};
use narra::models::scene::SceneParticipantCreate;
use narra::models::{CharacterCreate, EventCreate, LocationCreate, SceneCreate};
use narra::repository::{EntityRepository, SurrealEntityRepository};
use narra::services::export::ExportService;
use std::sync::Arc;

#[tokio::test]
async fn test_export_empty_world() {
    let harness = TestHarness::new().await;
    let server = common::create_test_server(&harness).await;

    let export_path = harness.temp_path().join("export.yaml");
    let request = ExportRequest {
        output_path: Some(export_path.to_str().unwrap().to_string()),
    };

    let response = server
        .handle_export_world(rmcp::handler::server::wrapper::Parameters(request))
        .await
        .unwrap();

    assert!(std::path::Path::new(&response.output_path).exists());
    assert_eq!(response.summary.character_count, 0);
    assert_eq!(response.summary.location_count, 0);
    assert_eq!(response.summary.event_count, 0);
    assert_eq!(response.summary.scene_count, 0);
    assert_eq!(response.summary.relationship_count, 0);
    assert_eq!(response.summary.knowledge_count, 0);
    assert_eq!(response.summary.note_count, 0);
    assert_eq!(response.summary.fact_count, 0);

    // Verify output is valid YAML parseable as NarraImport
    let content = std::fs::read_to_string(&response.output_path).unwrap();
    assert!(content.starts_with("# Narra world export"));
    let parsed: NarraImport = serde_yaml_ng::from_str(&content).unwrap();
    assert!(parsed.characters.is_empty());
}

#[tokio::test]
async fn test_export_with_data() {
    let harness = TestHarness::new().await;
    let entity_repo = SurrealEntityRepository::new(harness.db.clone());

    entity_repo
        .create_character(CharacterCreate {
            name: "Alice".to_string(),
            aliases: vec!["A".to_string()],
            roles: vec!["protagonist".to_string()],
            ..Default::default()
        })
        .await
        .unwrap();

    entity_repo
        .create_character(CharacterCreate {
            name: "Bob".to_string(),
            aliases: vec![],
            roles: vec!["supporting".to_string()],
            ..Default::default()
        })
        .await
        .unwrap();

    entity_repo
        .create_location(LocationCreate {
            name: "The Manor".to_string(),
            description: Some("Old Victorian mansion".to_string()),
            loc_type: "building".to_string(),
            parent: None,
        })
        .await
        .unwrap();

    entity_repo
        .create_event(EventCreate {
            title: "First Meeting".to_string(),
            description: Some("Alice meets Bob".to_string()),
            sequence: 1,
            date: None,
            date_precision: None,
            duration_end: None,
        })
        .await
        .unwrap();

    let server = common::create_test_server(&harness).await;

    let export_path = harness.temp_path().join("export.yaml");
    let request = ExportRequest {
        output_path: Some(export_path.to_str().unwrap().to_string()),
    };

    let response = server
        .handle_export_world(rmcp::handler::server::wrapper::Parameters(request))
        .await
        .unwrap();

    assert_eq!(response.summary.character_count, 2);
    assert_eq!(response.summary.location_count, 1);
    assert_eq!(response.summary.event_count, 1);

    // Parse exported YAML as NarraImport
    let content = std::fs::read_to_string(&response.output_path).unwrap();
    let parsed: NarraImport = serde_yaml_ng::from_str(&content).unwrap();
    assert_eq!(parsed.characters.len(), 2);
    assert_eq!(parsed.locations.len(), 1);
    assert_eq!(parsed.events.len(), 1);

    // Verify character fields
    let alice = parsed
        .characters
        .iter()
        .find(|c| c.name == "Alice")
        .unwrap();
    assert_eq!(alice.role.as_deref(), Some("protagonist"));
    assert_eq!(alice.aliases.as_ref().unwrap(), &["A"]);

    // Verify location fields
    let manor = &parsed.locations[0];
    assert_eq!(manor.name, "The Manor");
    assert_eq!(manor.loc_type.as_deref(), Some("building"));
}

#[tokio::test]
async fn test_export_round_trip() {
    let harness = TestHarness::new().await;
    let entity_repo = SurrealEntityRepository::new(harness.db.clone());

    // Create test data
    entity_repo
        .create_character(CharacterCreate {
            name: "Alice".to_string(),
            aliases: vec!["A".to_string()],
            roles: vec!["protagonist".to_string()],
            ..Default::default()
        })
        .await
        .unwrap();

    entity_repo
        .create_location(LocationCreate {
            name: "The Manor".to_string(),
            description: Some("Old Victorian mansion".to_string()),
            loc_type: "building".to_string(),
            parent: None,
        })
        .await
        .unwrap();

    entity_repo
        .create_event(EventCreate {
            title: "First Meeting".to_string(),
            description: Some("Alice meets Bob".to_string()),
            sequence: 1,
            date: None,
            date_precision: None,
            duration_end: None,
        })
        .await
        .unwrap();

    // Export
    let export_service = ExportService::new(harness.db.clone());
    let exported = export_service.export_world().await.unwrap();
    let yaml = serde_yaml_ng::to_string(&exported).unwrap();

    // Parse back as NarraImport
    let reimported: NarraImport = serde_yaml_ng::from_str(&yaml).unwrap();

    // Re-import into a fresh database with on_conflict: skip (entities already exist)
    let staleness_mgr = Arc::new(narra::embedding::StalenessManager::new(
        harness.db.clone(),
        common::test_embedding_service(),
    ));
    let import_service =
        narra::services::import::ImportService::new(harness.db.clone(), staleness_mgr);
    let result = import_service
        .execute_import(reimported, ConflictMode::Skip)
        .await
        .unwrap();

    // All entities already exist, so they should all be skipped with 0 errors
    assert_eq!(
        result.total_errors, 0,
        "round-trip produced errors: {:?}",
        result.by_type
    );
    assert_eq!(
        result.total_created, 0,
        "unexpected creations on round-trip"
    );
    assert!(
        result.total_skipped > 0,
        "nothing was skipped on round-trip"
    );
}

#[tokio::test]
async fn test_export_all_entity_types() {
    let harness = TestHarness::new().await;
    let entity_repo = SurrealEntityRepository::new(harness.db.clone());

    // Characters
    let alice = entity_repo
        .create_character(CharacterCreate {
            name: "Alice".to_string(),
            aliases: vec![],
            roles: vec!["protagonist".to_string()],
            ..Default::default()
        })
        .await
        .unwrap();

    let bob = entity_repo
        .create_character(CharacterCreate {
            name: "Bob".to_string(),
            aliases: vec![],
            roles: vec!["supporting".to_string()],
            ..Default::default()
        })
        .await
        .unwrap();

    // Location
    let loc = entity_repo
        .create_location(LocationCreate {
            name: "Castle".to_string(),
            description: Some("A grand castle".to_string()),
            loc_type: "building".to_string(),
            parent: None,
        })
        .await
        .unwrap();

    // Event
    let ev = entity_repo
        .create_event(EventCreate {
            title: "Battle".to_string(),
            description: Some("The great battle".to_string()),
            sequence: 1,
            date: None,
            date_precision: None,
            duration_end: None,
        })
        .await
        .unwrap();

    // Scene
    let scene = narra::models::scene::create_scene_with_id(
        &harness.db,
        "battle_scene",
        SceneCreate {
            title: "Battle Scene".to_string(),
            summary: Some("The climactic battle".to_string()),
            event: ev.id.clone(),
            primary_location: loc.id.clone(),
            secondary_locations: vec![],
        },
    )
    .await
    .unwrap();

    // Add participant
    narra::models::scene::add_scene_participant(
        &harness.db,
        SceneParticipantCreate {
            character_id: alice.id.key().to_string(),
            scene_id: scene.id.key().to_string(),
            role: "protagonist".to_string(),
            notes: Some("Leading the charge".to_string()),
        },
    )
    .await
    .unwrap();

    // Relationship
    narra::models::relationship::create_relationship(
        &harness.db,
        narra::models::RelationshipCreate {
            from_character_id: alice.id.key().to_string(),
            to_character_id: bob.id.key().to_string(),
            rel_type: "rivalry".to_string(),
            subtype: None,
            label: Some("bitter rivals".to_string()),
        },
    )
    .await
    .unwrap();

    // Knowledge
    let knowledge = narra::models::knowledge::create_knowledge(
        &harness.db,
        narra::models::knowledge::KnowledgeCreate {
            character: alice.id.clone(),
            fact: "Bob is secretly a prince".to_string(),
        },
    )
    .await
    .unwrap();
    narra::models::knowledge::create_knowledge_state(
        &harness.db,
        &alice.id.key().to_string(),
        &format!("knowledge:{}", knowledge.id.key()),
        narra::models::knowledge::KnowledgeStateCreate::default(),
    )
    .await
    .unwrap();

    // Note
    let note = narra::models::note::create_note_with_id(
        &harness.db,
        "plot_notes",
        narra::models::note::NoteCreate {
            title: "Plot Notes".to_string(),
            body: "Remember to foreshadow the betrayal".to_string(),
        },
    )
    .await
    .unwrap();
    narra::models::note::attach_note(
        &harness.db,
        &note.id.key().to_string(),
        &format!("character:{}", alice.id.key()),
    )
    .await
    .unwrap();

    // Fact
    let fact = narra::models::fact::create_fact_with_id(
        &harness.db,
        "no_magic",
        narra::models::fact::FactCreate {
            title: "No Magic".to_string(),
            description: "Magic does not exist in this world".to_string(),
            categories: vec![narra::models::fact::FactCategory::PhysicsMagic],
            enforcement_level: narra::models::fact::EnforcementLevel::Strict,
            scope: None,
        },
    )
    .await
    .unwrap();
    narra::models::fact::link_fact_to_entity(
        &harness.db,
        &fact.id.key().to_string(),
        &format!("character:{}", alice.id.key()),
        "manual",
        None,
    )
    .await
    .unwrap();

    // Export
    let export_service = ExportService::new(harness.db.clone());
    let exported = export_service.export_world().await.unwrap();

    // Verify all 8 entity types populated
    assert!(!exported.characters.is_empty(), "characters missing");
    assert!(!exported.locations.is_empty(), "locations missing");
    assert!(!exported.events.is_empty(), "events missing");
    assert!(!exported.scenes.is_empty(), "scenes missing");
    assert!(!exported.relationships.is_empty(), "relationships missing");
    assert!(!exported.knowledge.is_empty(), "knowledge missing");
    assert!(!exported.notes.is_empty(), "notes missing");
    assert!(!exported.facts.is_empty(), "facts missing");
}

#[tokio::test]
async fn test_export_scene_with_participants() {
    let harness = TestHarness::new().await;
    let entity_repo = SurrealEntityRepository::new(harness.db.clone());

    let alice = entity_repo
        .create_character(CharacterCreate {
            name: "Alice".to_string(),
            aliases: vec![],
            roles: vec!["protagonist".to_string()],
            ..Default::default()
        })
        .await
        .unwrap();

    let loc = entity_repo
        .create_location(LocationCreate {
            name: "Garden".to_string(),
            description: None,
            loc_type: "place".to_string(),
            parent: None,
        })
        .await
        .unwrap();

    let ev = entity_repo
        .create_event(EventCreate {
            title: "Discovery".to_string(),
            description: None,
            sequence: 1,
            date: None,
            date_precision: None,
            duration_end: None,
        })
        .await
        .unwrap();

    let scene = narra::models::scene::create_scene_with_id(
        &harness.db,
        "discovery_scene",
        SceneCreate {
            title: "Discovery Scene".to_string(),
            summary: None,
            event: ev.id.clone(),
            primary_location: loc.id.clone(),
            secondary_locations: vec![],
        },
    )
    .await
    .unwrap();

    narra::models::scene::add_scene_participant(
        &harness.db,
        SceneParticipantCreate {
            character_id: alice.id.key().to_string(),
            scene_id: scene.id.key().to_string(),
            role: "pov".to_string(),
            notes: Some("Discovering the secret".to_string()),
        },
    )
    .await
    .unwrap();

    let export_service = ExportService::new(harness.db.clone());
    let exported = export_service.export_world().await.unwrap();

    assert_eq!(exported.scenes.len(), 1);
    let scene_spec = &exported.scenes[0];
    assert_eq!(scene_spec.participants.len(), 1);
    assert_eq!(scene_spec.participants[0].role, "pov");
    assert_eq!(
        scene_spec.participants[0].notes.as_deref(),
        Some("Discovering the secret")
    );
}

#[tokio::test]
async fn test_export_knowledge_with_facts() {
    let harness = TestHarness::new().await;
    let entity_repo = SurrealEntityRepository::new(harness.db.clone());

    let alice = entity_repo
        .create_character(CharacterCreate {
            name: "Alice".to_string(),
            aliases: vec![],
            roles: vec!["protagonist".to_string()],
            ..Default::default()
        })
        .await
        .unwrap();

    let knowledge = narra::models::knowledge::create_knowledge(
        &harness.db,
        narra::models::knowledge::KnowledgeCreate {
            character: alice.id.clone(),
            fact: "The sky is blue".to_string(),
        },
    )
    .await
    .unwrap();

    narra::models::knowledge::create_knowledge_state(
        &harness.db,
        &alice.id.key().to_string(),
        &format!("knowledge:{}", knowledge.id.key()),
        narra::models::knowledge::KnowledgeStateCreate {
            certainty: narra::models::knowledge::CertaintyLevel::Suspects,
            learning_method: narra::models::knowledge::LearningMethod::Initial,
            ..Default::default()
        },
    )
    .await
    .unwrap();

    let export_service = ExportService::new(harness.db.clone());
    let exported = export_service.export_world().await.unwrap();

    assert_eq!(exported.knowledge.len(), 1);
    let k = &exported.knowledge[0];
    assert_eq!(k.fact, "The sky is blue");
    assert_eq!(k.certainty, "suspects");
}

#[tokio::test]
async fn test_export_notes_with_attachments() {
    let harness = TestHarness::new().await;
    let entity_repo = SurrealEntityRepository::new(harness.db.clone());

    let alice = entity_repo
        .create_character(CharacterCreate {
            name: "Alice".to_string(),
            aliases: vec![],
            roles: vec![],
            ..Default::default()
        })
        .await
        .unwrap();

    let note = narra::models::note::create_note_with_id(
        &harness.db,
        "idea1",
        narra::models::note::NoteCreate {
            title: "Plot Idea".to_string(),
            body: "What if Alice can fly?".to_string(),
        },
    )
    .await
    .unwrap();

    narra::models::note::attach_note(
        &harness.db,
        &note.id.key().to_string(),
        &format!("character:{}", alice.id.key()),
    )
    .await
    .unwrap();

    let export_service = ExportService::new(harness.db.clone());
    let exported = export_service.export_world().await.unwrap();

    assert_eq!(exported.notes.len(), 1);
    let n = &exported.notes[0];
    assert_eq!(n.title, "Plot Idea");
    assert_eq!(n.body, "What if Alice can fly?");
    assert_eq!(n.attach_to.len(), 1);
    assert!(n.attach_to[0].contains("character:"));
}

#[tokio::test]
async fn test_export_facts_with_links() {
    let harness = TestHarness::new().await;
    let entity_repo = SurrealEntityRepository::new(harness.db.clone());

    let alice = entity_repo
        .create_character(CharacterCreate {
            name: "Alice".to_string(),
            aliases: vec![],
            roles: vec![],
            ..Default::default()
        })
        .await
        .unwrap();

    let fact = narra::models::fact::create_fact_with_id(
        &harness.db,
        "gravity",
        narra::models::fact::FactCreate {
            title: "Gravity".to_string(),
            description: "Things fall down".to_string(),
            categories: vec![narra::models::fact::FactCategory::PhysicsMagic],
            enforcement_level: narra::models::fact::EnforcementLevel::Warning,
            scope: None,
        },
    )
    .await
    .unwrap();

    narra::models::fact::link_fact_to_entity(
        &harness.db,
        &fact.id.key().to_string(),
        &format!("character:{}", alice.id.key()),
        "manual",
        Some(0.9),
    )
    .await
    .unwrap();

    let export_service = ExportService::new(harness.db.clone());
    let exported = export_service.export_world().await.unwrap();

    assert_eq!(exported.facts.len(), 1);
    let f = &exported.facts[0];
    assert_eq!(f.title, "Gravity");
    assert_eq!(f.description, "Things fall down");
    assert!(f.categories.contains(&"physics_magic".to_string()));
    assert_eq!(f.enforcement_level.as_deref(), Some("warning"));
    assert_eq!(f.applies_to.len(), 1);
    assert!(f.applies_to[0].entity_id.contains("character:"));
    assert_eq!(f.applies_to[0].link_type, "manual");
    assert_eq!(f.applies_to[0].confidence, Some(0.9));
}
