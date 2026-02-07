mod common;

use common::harness::TestHarness;
use narra::mcp::types::{
    CharacterSpec, ConflictMode, EventSpec, FactLinkSpec, FactSpec, KnowledgeSpec, LocationSpec,
    NarraImport, NoteSpec, ParticipantSpec, RelationshipSpec, SceneSpec,
};
use narra::services::import::ImportService;
use std::sync::Arc;

use narra::embedding::{NoopEmbeddingService, StalenessManager};

fn test_staleness(harness: &TestHarness) -> Arc<StalenessManager> {
    let noop: Arc<dyn narra::embedding::EmbeddingService + Send + Sync> =
        Arc::new(NoopEmbeddingService::new());
    Arc::new(StalenessManager::new(harness.db.clone(), noop))
}

fn full_import() -> NarraImport {
    NarraImport {
        characters: vec![
            CharacterSpec {
                id: Some("alice".to_string()),
                name: "Alice".to_string(),
                role: Some("protagonist".to_string()),
                aliases: Some(vec!["The Shadow".to_string()]),
                description: Some("A brave adventurer".to_string()),
                profile: None,
            },
            CharacterSpec {
                id: Some("bob".to_string()),
                name: "Bob".to_string(),
                role: Some("antagonist".to_string()),
                aliases: None,
                description: None,
                profile: None,
            },
        ],
        locations: vec![
            LocationSpec {
                id: Some("kingdom".to_string()),
                name: "The Kingdom".to_string(),
                description: Some("A vast realm".to_string()),
                parent_id: None,
                loc_type: None,
            },
            LocationSpec {
                id: Some("castle".to_string()),
                name: "The Castle".to_string(),
                description: None,
                parent_id: Some("location:kingdom".to_string()),
                loc_type: None,
            },
        ],
        events: vec![EventSpec {
            id: Some("arrival".to_string()),
            title: "The Arrival".to_string(),
            description: Some("Alice arrives".to_string()),
            sequence: Some(10),
            date: None,
            date_precision: None,
        }],
        scenes: vec![SceneSpec {
            id: Some("homecoming".to_string()),
            title: "Homecoming".to_string(),
            event_id: "event:arrival".to_string(),
            location_id: "location:castle".to_string(),
            summary: Some("Alice returns home".to_string()),
            secondary_locations: vec![],
            participants: vec![ParticipantSpec {
                character_id: "alice".to_string(),
                role: "pov".to_string(),
                notes: None,
            }],
        }],
        relationships: vec![RelationshipSpec {
            from_character_id: "alice".to_string(),
            to_character_id: "bob".to_string(),
            rel_type: "rivalry".to_string(),
            subtype: None,
            label: Some("Political rivals".to_string()),
        }],
        knowledge: vec![KnowledgeSpec {
            character_id: "character:alice".to_string(),
            target_id: "character:bob".to_string(),
            fact: "Bob is secretly a spy".to_string(),
            certainty: "suspects".to_string(),
            method: Some("initial".to_string()),
            source_character_id: None,
            event_id: None,
        }],
        notes: vec![NoteSpec {
            id: Some("worldnote".to_string()),
            title: "World Notes".to_string(),
            body: "The kingdom is at war.".to_string(),
            attach_to: vec!["character:alice".to_string()],
        }],
        facts: vec![FactSpec {
            id: Some("no_flight".to_string()),
            title: "No Flying".to_string(),
            description: "No one can fly".to_string(),
            categories: vec!["physics_magic".to_string()],
            enforcement_level: Some("strict".to_string()),
            applies_to: vec![FactLinkSpec {
                entity_id: "character:alice".to_string(),
                link_type: "manual".to_string(),
                confidence: None,
            }],
        }],
    }
}

#[tokio::test]
async fn test_import_all_entity_types() {
    let harness = TestHarness::new().await;
    let staleness = test_staleness(&harness);
    let service = ImportService::new(harness.db.clone(), staleness);

    let import = full_import();
    let result = service
        .execute_import(import, ConflictMode::Error)
        .await
        .expect("import should succeed");

    assert_eq!(result.total_created, 10); // 2 char + 2 loc + 1 event + 1 scene + 1 rel + 1 knowledge + 1 note + 1 fact
    assert_eq!(result.total_errors, 0);
    assert_eq!(result.total_skipped, 0);

    // Verify counts per type
    let char_result = result
        .by_type
        .iter()
        .find(|t| t.entity_type == "character")
        .unwrap();
    assert_eq!(char_result.created, 2);

    let loc_result = result
        .by_type
        .iter()
        .find(|t| t.entity_type == "location")
        .unwrap();
    assert_eq!(loc_result.created, 2);

    let event_result = result
        .by_type
        .iter()
        .find(|t| t.entity_type == "event")
        .unwrap();
    assert_eq!(event_result.created, 1);

    let scene_result = result
        .by_type
        .iter()
        .find(|t| t.entity_type == "scene")
        .unwrap();
    assert_eq!(scene_result.created, 1);
}

#[tokio::test]
async fn test_import_dependency_ordering() {
    // Scenes reference events/locations created in same import
    let harness = TestHarness::new().await;
    let staleness = test_staleness(&harness);
    let service = ImportService::new(harness.db.clone(), staleness);

    let import = NarraImport {
        characters: vec![CharacterSpec {
            id: Some("hero".to_string()),
            name: "Hero".to_string(),
            role: None,
            aliases: None,
            description: None,
            profile: None,
        }],
        locations: vec![LocationSpec {
            id: Some("tavern".to_string()),
            name: "The Tavern".to_string(),
            description: None,
            parent_id: None,
            loc_type: None,
        }],
        events: vec![EventSpec {
            id: Some("meeting".to_string()),
            title: "The Meeting".to_string(),
            description: None,
            sequence: Some(1),
            date: None,
            date_precision: None,
        }],
        scenes: vec![SceneSpec {
            id: Some("intro".to_string()),
            title: "Introduction".to_string(),
            event_id: "event:meeting".to_string(),
            location_id: "location:tavern".to_string(),
            summary: None,
            secondary_locations: vec![],
            participants: vec![ParticipantSpec {
                character_id: "hero".to_string(),
                role: "pov".to_string(),
                notes: None,
            }],
        }],
        ..Default::default()
    };

    let result = service
        .execute_import(import, ConflictMode::Error)
        .await
        .expect("import should succeed");

    assert_eq!(result.total_errors, 0, "Errors: {:?}", result.by_type);
    assert_eq!(result.total_created, 4); // 1 char + 1 loc + 1 event + 1 scene
}

#[tokio::test]
async fn test_import_conflict_error() {
    let harness = TestHarness::new().await;
    let staleness = test_staleness(&harness);
    let service = ImportService::new(harness.db.clone(), staleness);

    // First import
    let import = NarraImport {
        characters: vec![
            CharacterSpec {
                id: Some("alice".to_string()),
                name: "Alice".to_string(),
                role: None,
                aliases: None,
                description: None,
                profile: None,
            },
            CharacterSpec {
                id: Some("bob".to_string()),
                name: "Bob".to_string(),
                role: None,
                aliases: None,
                description: None,
                profile: None,
            },
        ],
        ..Default::default()
    };

    let r1 = service
        .execute_import(import.clone(), ConflictMode::Error)
        .await
        .unwrap();
    assert_eq!(r1.total_created, 2);

    // Second import: duplicates + a new one
    let import2 = NarraImport {
        characters: vec![
            CharacterSpec {
                id: Some("alice".to_string()),
                name: "Alice 2".to_string(),
                role: None,
                aliases: None,
                description: None,
                profile: None,
            },
            CharacterSpec {
                id: Some("charlie".to_string()),
                name: "Charlie".to_string(),
                role: None,
                aliases: None,
                description: None,
                profile: None,
            },
        ],
        ..Default::default()
    };

    let r2 = service
        .execute_import(import2, ConflictMode::Error)
        .await
        .unwrap();
    assert_eq!(r2.total_created, 1); // Only charlie
    assert_eq!(r2.total_errors, 1); // alice is duplicate
}

#[tokio::test]
async fn test_import_conflict_skip() {
    let harness = TestHarness::new().await;
    let staleness = test_staleness(&harness);
    let service = ImportService::new(harness.db.clone(), staleness);

    let import = NarraImport {
        characters: vec![CharacterSpec {
            id: Some("alice".to_string()),
            name: "Alice".to_string(),
            role: None,
            aliases: None,
            description: None,
            profile: None,
        }],
        ..Default::default()
    };

    service
        .execute_import(import.clone(), ConflictMode::Error)
        .await
        .unwrap();

    // Re-import with skip
    let r2 = service
        .execute_import(import, ConflictMode::Skip)
        .await
        .unwrap();
    assert_eq!(r2.total_skipped, 1);
    assert_eq!(r2.total_created, 0);
    assert_eq!(r2.total_errors, 0);
}

#[tokio::test]
async fn test_import_conflict_update() {
    let harness = TestHarness::new().await;
    let staleness = test_staleness(&harness);
    let service = ImportService::new(harness.db.clone(), staleness);

    // Create initial
    let import = NarraImport {
        characters: vec![CharacterSpec {
            id: Some("alice".to_string()),
            name: "Alice Original".to_string(),
            role: Some("villager".to_string()),
            aliases: None,
            description: None,
            profile: None,
        }],
        ..Default::default()
    };

    service
        .execute_import(import, ConflictMode::Error)
        .await
        .unwrap();

    // Re-import with update
    let import2 = NarraImport {
        characters: vec![CharacterSpec {
            id: Some("alice".to_string()),
            name: "Alice Updated".to_string(),
            role: Some("hero".to_string()),
            aliases: None,
            description: None,
            profile: None,
        }],
        ..Default::default()
    };

    let r2 = service
        .execute_import(import2, ConflictMode::Update)
        .await
        .unwrap();
    assert_eq!(r2.total_updated, 1);
    assert_eq!(r2.total_created, 0);

    // Verify update took effect
    let alice = narra::models::character::get_character(&harness.db, "alice")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(alice.name, "Alice Updated");
}

#[tokio::test]
async fn test_import_scene_with_participants() {
    let harness = TestHarness::new().await;
    let staleness = test_staleness(&harness);
    let service = ImportService::new(harness.db.clone(), staleness);

    let import = NarraImport {
        characters: vec![
            CharacterSpec {
                id: Some("alice".to_string()),
                name: "Alice".to_string(),
                role: None,
                aliases: None,
                description: None,
                profile: None,
            },
            CharacterSpec {
                id: Some("bob".to_string()),
                name: "Bob".to_string(),
                role: None,
                aliases: None,
                description: None,
                profile: None,
            },
        ],
        locations: vec![LocationSpec {
            id: Some("hall".to_string()),
            name: "Great Hall".to_string(),
            description: None,
            parent_id: None,
            loc_type: None,
        }],
        events: vec![EventSpec {
            id: Some("feast".to_string()),
            title: "The Feast".to_string(),
            description: None,
            sequence: Some(1),
            date: None,
            date_precision: None,
        }],
        scenes: vec![SceneSpec {
            id: Some("dinner".to_string()),
            title: "The Dinner Scene".to_string(),
            event_id: "event:feast".to_string(),
            location_id: "location:hall".to_string(),
            summary: None,
            secondary_locations: vec![],
            participants: vec![
                ParticipantSpec {
                    character_id: "alice".to_string(),
                    role: "pov".to_string(),
                    notes: Some("Narrating".to_string()),
                },
                ParticipantSpec {
                    character_id: "bob".to_string(),
                    role: "supporting".to_string(),
                    notes: None,
                },
            ],
        }],
        ..Default::default()
    };

    let result = service
        .execute_import(import, ConflictMode::Error)
        .await
        .unwrap();
    assert_eq!(result.total_errors, 0, "Errors: {:?}", result.by_type);

    // Verify participants were created
    let participants = narra::models::scene::get_scene_participants(&harness.db, "dinner")
        .await
        .unwrap();
    assert_eq!(participants.len(), 2);

    let roles: Vec<&str> = participants.iter().map(|p| p.role.as_str()).collect();
    assert!(roles.contains(&"pov"));
    assert!(roles.contains(&"supporting"));
}

#[tokio::test]
async fn test_import_knowledge_entries() {
    let harness = TestHarness::new().await;
    let staleness = test_staleness(&harness);
    let service = ImportService::new(harness.db.clone(), staleness);

    // Need a character first
    let import = NarraImport {
        characters: vec![CharacterSpec {
            id: Some("alice".to_string()),
            name: "Alice".to_string(),
            role: None,
            aliases: None,
            description: None,
            profile: None,
        }],
        knowledge: vec![KnowledgeSpec {
            character_id: "character:alice".to_string(),
            target_id: "character:alice".to_string(),
            fact: "The secret password is 'moonlight'".to_string(),
            certainty: "knows".to_string(),
            method: Some("initial".to_string()),
            source_character_id: None,
            event_id: None,
        }],
        ..Default::default()
    };

    let result = service
        .execute_import(import, ConflictMode::Error)
        .await
        .unwrap();

    let know_result = result
        .by_type
        .iter()
        .find(|t| t.entity_type == "knowledge")
        .unwrap();
    assert_eq!(know_result.created, 1);
    assert!(
        know_result.errors.is_empty(),
        "Errors: {:?}",
        know_result.errors
    );

    // Verify knowledge was created
    let knowledge = narra::models::knowledge::get_character_knowledge(&harness.db, "alice")
        .await
        .unwrap();
    assert_eq!(knowledge.len(), 1);
    assert_eq!(knowledge[0].fact, "The secret password is 'moonlight'");

    // Verify knows edge was created
    let states = narra::models::knowledge::get_character_knowledge_states(&harness.db, "alice")
        .await
        .unwrap();
    assert_eq!(states.len(), 1);
}

#[tokio::test]
async fn test_import_notes_with_attachments() {
    let harness = TestHarness::new().await;
    let staleness = test_staleness(&harness);
    let service = ImportService::new(harness.db.clone(), staleness);

    let import = NarraImport {
        characters: vec![CharacterSpec {
            id: Some("alice".to_string()),
            name: "Alice".to_string(),
            role: None,
            aliases: None,
            description: None,
            profile: None,
        }],
        notes: vec![NoteSpec {
            id: Some("mynote".to_string()),
            title: "Research Notes".to_string(),
            body: "Some worldbuilding ideas".to_string(),
            attach_to: vec!["character:alice".to_string()],
        }],
        ..Default::default()
    };

    let result = service
        .execute_import(import, ConflictMode::Error)
        .await
        .unwrap();
    assert_eq!(result.total_errors, 0, "Errors: {:?}", result.by_type);

    // Verify note exists
    let note = narra::models::note::get_note(&harness.db, "mynote")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(note.title, "Research Notes");

    // Verify attachment
    let attachments = narra::models::note::get_note_attachments(&harness.db, "mynote")
        .await
        .unwrap();
    assert_eq!(attachments.len(), 1);
}

#[tokio::test]
async fn test_import_facts_with_links() {
    let harness = TestHarness::new().await;
    let staleness = test_staleness(&harness);
    let service = ImportService::new(harness.db.clone(), staleness);

    let import = NarraImport {
        characters: vec![CharacterSpec {
            id: Some("alice".to_string()),
            name: "Alice".to_string(),
            role: None,
            aliases: None,
            description: None,
            profile: None,
        }],
        facts: vec![FactSpec {
            id: Some("gravity".to_string()),
            title: "Normal Gravity".to_string(),
            description: "Gravity works normally".to_string(),
            categories: vec!["physics_magic".to_string()],
            enforcement_level: Some("warning".to_string()),
            applies_to: vec![FactLinkSpec {
                entity_id: "character:alice".to_string(),
                link_type: "manual".to_string(),
                confidence: Some(0.95),
            }],
        }],
        ..Default::default()
    };

    let result = service
        .execute_import(import, ConflictMode::Error)
        .await
        .unwrap();
    assert_eq!(result.total_errors, 0, "Errors: {:?}", result.by_type);

    // Verify fact exists
    let fact = narra::models::fact::get_fact(&harness.db, "gravity")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(fact.title, "Normal Gravity");

    // Verify applies_to link
    let apps = narra::models::fact::get_fact_applications(&harness.db, "gravity")
        .await
        .unwrap();
    assert_eq!(apps.len(), 1);
}

#[tokio::test]
async fn test_import_empty_document() {
    let harness = TestHarness::new().await;
    let staleness = test_staleness(&harness);
    let service = ImportService::new(harness.db.clone(), staleness);

    let result = service
        .execute_import(NarraImport::default(), ConflictMode::Error)
        .await
        .unwrap();

    assert_eq!(result.total_created, 0);
    assert_eq!(result.total_skipped, 0);
    assert_eq!(result.total_updated, 0);
    assert_eq!(result.total_errors, 0);
}
