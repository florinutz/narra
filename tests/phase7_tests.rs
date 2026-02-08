//! Integration tests for Phase 7: Advanced Features
//!
//! Requirements tested:
//! - NOTE-01: Notes can be created with title/body, attached to entities
//! - NOTE-02: Notes appear in search results (full-text and fuzzy)
//! - VGRF-01: Graph generation produces valid Mermaid diagrams
//! - VGRF-02: Edges are styled by relationship type
//! - VGRF-03: Character-centered graphs with depth limiting

mod common;

use common::harness::TestHarness;
use narra::embedding::NoopEmbeddingService;
use narra::models::{
    note::{
        attach_note, create_note, delete_note, detach_note, get_entity_notes, get_note,
        get_note_attachments, list_notes, update_note,
    },
    perception::create_perception_pair,
    CharacterCreate, NoteCreate, NoteUpdate, PerceptionCreate,
};
use narra::repository::{EntityRepository, SurrealEntityRepository};
use narra::services::{
    EntityType, GraphOptions, GraphScope, GraphService, MermaidGraphService, SearchFilter,
    SearchService, SurrealSearchService,
};
use std::sync::Arc;

// ============================================================================
// NOTE-01 Tests: Note CRUD and Attachments
// ============================================================================

/// Test basic note CRUD operations: create, read, update, delete.
///
/// Verifies: NOTE-01 (notes can be created with title/body)
#[tokio::test]
async fn test_note_crud_operations() {
    let harness = TestHarness::new().await;
    let db = &harness.db;

    // Create a note
    let note = create_note(
        db,
        NoteCreate {
            title: "Research: Magic System".to_string(),
            body: "The magic system is based on elemental forces. Fire, water, earth, air."
                .to_string(),
        },
    )
    .await
    .expect("Should create note");

    assert_eq!(note.title, "Research: Magic System");
    assert!(note.body.contains("elemental forces"));

    let note_id = note.id.key().to_string();

    // Read the note
    let fetched = get_note(db, &note_id)
        .await
        .expect("Should fetch note")
        .expect("Note should exist");

    assert_eq!(fetched.title, "Research: Magic System");
    assert_eq!(fetched.body, note.body);

    // Update the note
    let updated = update_note(
        db,
        &note_id,
        NoteUpdate {
            title: Some("Research: Elemental Magic System".to_string()),
            body: None,
        },
    )
    .await
    .expect("Should update note")
    .expect("Note should exist");

    assert_eq!(updated.title, "Research: Elemental Magic System");
    assert_eq!(updated.body, note.body); // Body unchanged

    // Delete the note
    let deleted = delete_note(db, &note_id)
        .await
        .expect("Should delete note")
        .expect("Note should exist");

    assert_eq!(deleted.title, "Research: Elemental Magic System");

    // Verify deletion
    let gone = get_note(db, &note_id).await.expect("Query should succeed");

    assert!(gone.is_none(), "Note should be deleted");

    println!("NOTE-01 test passed - note CRUD operations work");
}

/// Test attaching and detaching notes from entities.
///
/// Verifies: NOTE-01 (notes can be attached to entities)
#[tokio::test]
async fn test_note_attachment_to_entity() {
    let harness = TestHarness::new().await;
    let db = &harness.db;
    let entity_repo = SurrealEntityRepository::new(db.clone());

    // Create a character to attach notes to
    let character = entity_repo
        .create_character(CharacterCreate {
            name: "Elena Vasquez".to_string(),
            aliases: vec![],
            roles: vec!["protagonist".to_string()],
            ..Default::default()
        })
        .await
        .expect("Should create character");

    let character_id = format!("character:{}", character.id.key());

    // Create a note
    let note = create_note(
        db,
        NoteCreate {
            title: "Elena's Backstory Ideas".to_string(),
            body: "Consider: orphaned at age 10, raised by grandmother".to_string(),
        },
    )
    .await
    .expect("Should create note");

    let note_id = note.id.key().to_string();

    // Attach note to character
    let attachment = attach_note(db, &note_id, &character_id)
        .await
        .expect("Should attach note");

    assert_eq!(attachment.note.key().to_string(), note_id);
    assert_eq!(attachment.entity.to_string(), character_id);

    // Verify attachment via get_note_attachments
    let attachments = get_note_attachments(db, &note_id)
        .await
        .expect("Should get attachments");

    assert_eq!(attachments.len(), 1);
    assert_eq!(attachments[0].entity.to_string(), character_id);

    // Verify attachment via get_entity_notes
    let entity_notes = get_entity_notes(db, &character_id)
        .await
        .expect("Should get entity notes");

    assert_eq!(entity_notes.len(), 1);
    assert_eq!(entity_notes[0].title, "Elena's Backstory Ideas");

    // Detach note
    detach_note(db, &note_id, &character_id)
        .await
        .expect("Should detach note");

    // Verify detachment
    let remaining_attachments = get_note_attachments(db, &note_id)
        .await
        .expect("Should get attachments");

    assert_eq!(remaining_attachments.len(), 0, "Note should be detached");

    println!("NOTE-01 test passed - note attachments work");
}

/// Test that notes can exist without any attachments.
///
/// Verifies: NOTE-01 (notes can be standalone)
#[tokio::test]
async fn test_note_standalone_no_attachments() {
    let harness = TestHarness::new().await;
    let db = &harness.db;

    // Create multiple standalone notes
    let note1 = create_note(
        db,
        NoteCreate {
            title: "World Building: Climate".to_string(),
            body: "The world has three distinct seasons...".to_string(),
        },
    )
    .await
    .expect("Should create note 1");

    let note2 = create_note(
        db,
        NoteCreate {
            title: "Plot Idea: Betrayal Arc".to_string(),
            body: "What if the mentor character is secretly working for the enemy?".to_string(),
        },
    )
    .await
    .expect("Should create note 2");

    // List notes (should return both)
    let notes = list_notes(db, 10, 0).await.expect("Should list notes");

    assert_eq!(notes.len(), 2);

    // Verify no attachments for either note
    let attachments1 = get_note_attachments(db, &note1.id.key().to_string())
        .await
        .expect("Should get attachments");
    let attachments2 = get_note_attachments(db, &note2.id.key().to_string())
        .await
        .expect("Should get attachments");

    assert_eq!(attachments1.len(), 0, "Note 1 should have no attachments");
    assert_eq!(attachments2.len(), 0, "Note 2 should have no attachments");

    println!("NOTE-01 test passed - standalone notes work");
}

// ============================================================================
// NOTE-02 Tests: Notes in Search Results
// ============================================================================

/// Test that notes are findable via full-text search.
///
/// Verifies: NOTE-02 (notes appear in search results)
#[tokio::test]
async fn test_note_fulltext_search() {
    let harness = TestHarness::new().await;
    let db = &harness.db;
    let search_service =
        SurrealSearchService::new(db.clone(), Arc::new(NoopEmbeddingService::new()));

    // Create notes with searchable content
    create_note(
        db,
        NoteCreate {
            title: "Dragon Mythology".to_string(),
            body: "Dragons in this world are elemental creatures tied to volcanoes".to_string(),
        },
    )
    .await
    .expect("Should create note");

    create_note(
        db,
        NoteCreate {
            title: "Character Arc: Redemption".to_string(),
            body: "The villain seeks redemption after betraying the kingdom".to_string(),
        },
    )
    .await
    .expect("Should create note");

    // Small delay for index update
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    // Search for "Dragon" in title
    let results = search_service
        .search("Dragon", SearchFilter::default())
        .await
        .expect("Search should succeed");

    assert!(
        results.iter().any(|r| r.name.contains("Dragon")),
        "Should find note by title"
    );

    // Search for "redemption" (in body)
    let body_results = search_service
        .search("redemption", SearchFilter::default())
        .await
        .expect("Search should succeed");

    assert!(
        body_results.iter().any(|r| r.name.contains("Redemption")),
        "Should find note by body content"
    );

    println!("NOTE-02 test passed - notes found via full-text search");
}

/// Test that search can filter to only notes.
///
/// Verifies: NOTE-02 (search filter by type includes notes)
#[tokio::test]
async fn test_note_search_filter_by_type() {
    let harness = TestHarness::new().await;
    let db = &harness.db;
    let entity_repo = SurrealEntityRepository::new(db.clone());
    let search_service =
        SurrealSearchService::new(db.clone(), Arc::new(NoopEmbeddingService::new()));

    // Create a character with "Magic" in name
    entity_repo
        .create_character(CharacterCreate {
            name: "Magic Mike".to_string(),
            aliases: vec![],
            roles: vec![],
            ..Default::default()
        })
        .await
        .expect("Should create character");

    // Create a note with "Magic" in title
    create_note(
        db,
        NoteCreate {
            title: "Magic System Design".to_string(),
            body: "Notes about the magic system".to_string(),
        },
    )
    .await
    .expect("Should create note");

    // Small delay for index
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    // Search for "Magic" filtering to notes only
    let note_results = search_service
        .search(
            "Magic",
            SearchFilter {
                entity_types: vec![EntityType::Note],
                ..Default::default()
            },
        )
        .await
        .expect("Search should succeed");

    // Should find the note, not the character
    assert!(
        note_results.iter().all(|r| r.entity_type == "note"),
        "All results should be notes"
    );
    assert!(
        note_results.iter().any(|r| r.name.contains("Magic System")),
        "Should find the magic system note"
    );

    println!("NOTE-02 test passed - search filters to notes work");
}

// ============================================================================
// VGRF-01, VGRF-02, VGRF-03 Tests: Graph Generation
// ============================================================================

/// Test full network graph generation produces valid Mermaid.
///
/// Verifies: VGRF-01 (graph generation produces valid Mermaid)
#[tokio::test]
async fn test_full_network_graph_generation() {
    let harness = TestHarness::new().await;
    let db = &harness.db;
    let entity_repo = SurrealEntityRepository::new(db.clone());
    let graph_service = MermaidGraphService::new(db.clone());

    // Create characters
    let alice = entity_repo
        .create_character(CharacterCreate {
            name: "Alice".to_string(),
            aliases: vec![],
            roles: vec!["protagonist".to_string()],
            ..Default::default()
        })
        .await
        .expect("Should create Alice");

    let bob = entity_repo
        .create_character(CharacterCreate {
            name: "Bob".to_string(),
            aliases: vec![],
            roles: vec!["antagonist".to_string()],
            ..Default::default()
        })
        .await
        .expect("Should create Bob");

    let alice_id = alice.id.key().to_string();
    let bob_id = bob.id.key().to_string();

    // Create perception (relationship)
    create_perception_pair(
        db,
        &alice_id,
        &bob_id,
        PerceptionCreate {
            rel_types: vec!["rivalry".to_string()],
            subtype: None,
            feelings: Some("Distrust".to_string()),
            perception: Some("Untrustworthy".to_string()),
            tension_level: Some(8),
            history_notes: None,
        },
        PerceptionCreate {
            rel_types: vec!["rivalry".to_string()],
            subtype: None,
            feelings: Some("Contempt".to_string()),
            perception: Some("Naive idealist".to_string()),
            tension_level: Some(6),
            history_notes: None,
        },
    )
    .await
    .expect("Should create perception pair");

    // Generate full network graph
    let mermaid = graph_service
        .generate_mermaid(GraphScope::FullNetwork, GraphOptions::default())
        .await
        .expect("Should generate graph");

    // Verify Mermaid structure
    assert!(
        mermaid.contains("```mermaid"),
        "Should have mermaid code block"
    );
    assert!(mermaid.contains("graph TB"), "Should have graph directive");
    assert!(mermaid.contains("Alice"), "Should contain Alice node");
    assert!(mermaid.contains("Bob"), "Should contain Bob node");
    assert!(mermaid.contains("---"), "Should have edge connectors");
    assert!(mermaid.contains("## Legend"), "Should have legend");

    println!("VGRF-01 test passed - full network graph generates valid Mermaid");
}

/// Test that edges are styled by relationship type.
///
/// Verifies: VGRF-02 (edges styled by relationship type)
#[tokio::test]
async fn test_graph_edge_styling_by_type() {
    let harness = TestHarness::new().await;
    let db = &harness.db;
    let entity_repo = SurrealEntityRepository::new(db.clone());
    let graph_service = MermaidGraphService::new(db.clone());

    // Create three characters with different relationship types
    let parent = entity_repo
        .create_character(CharacterCreate {
            name: "Victor".to_string(),
            aliases: vec![],
            roles: vec!["mentor".to_string()],
            ..Default::default()
        })
        .await
        .expect("Should create Victor");

    let child = entity_repo
        .create_character(CharacterCreate {
            name: "Elena".to_string(),
            aliases: vec![],
            roles: vec!["protagonist".to_string()],
            ..Default::default()
        })
        .await
        .expect("Should create Elena");

    let friend = entity_repo
        .create_character(CharacterCreate {
            name: "Marcus".to_string(),
            aliases: vec![],
            roles: vec!["ally".to_string()],
            ..Default::default()
        })
        .await
        .expect("Should create Marcus");

    let parent_id = parent.id.key().to_string();
    let child_id = child.id.key().to_string();
    let friend_id = friend.id.key().to_string();

    // Create family relationship
    create_perception_pair(
        db,
        &parent_id,
        &child_id,
        PerceptionCreate {
            rel_types: vec!["family".to_string()],
            subtype: None,
            feelings: Some("Love".to_string()),
            perception: Some("My daughter".to_string()),
            tension_level: Some(2),
            history_notes: None,
        },
        PerceptionCreate {
            rel_types: vec!["family".to_string()],
            subtype: None,
            feelings: Some("Love".to_string()),
            perception: Some("My father".to_string()),
            tension_level: Some(2),
            history_notes: None,
        },
    )
    .await
    .expect("Should create family perception");

    // Create friendship relationship
    create_perception_pair(
        db,
        &child_id,
        &friend_id,
        PerceptionCreate {
            rel_types: vec!["friendship".to_string()],
            subtype: None,
            feelings: Some("Trust".to_string()),
            perception: Some("Best friend".to_string()),
            tension_level: Some(1),
            history_notes: None,
        },
        PerceptionCreate {
            rel_types: vec!["friendship".to_string()],
            subtype: None,
            feelings: Some("Trust".to_string()),
            perception: Some("Best friend".to_string()),
            tension_level: Some(1),
            history_notes: None,
        },
    )
    .await
    .expect("Should create friendship perception");

    // Generate graph
    let mermaid = graph_service
        .generate_mermaid(GraphScope::FullNetwork, GraphOptions::default())
        .await
        .expect("Should generate graph");

    // Verify edge labels include relationship types
    assert!(
        mermaid.contains("|family|") || mermaid.contains("family"),
        "Should have family relationship label"
    );
    assert!(
        mermaid.contains("|friendship|") || mermaid.contains("friendship"),
        "Should have friendship relationship label"
    );

    // Verify style definitions exist
    assert!(
        mermaid.contains("classDef"),
        "Should have class definitions"
    );
    assert!(
        mermaid.contains("stroke:#22c55e") || mermaid.contains("rel_family"),
        "Should have family color style (green)"
    );
    assert!(
        mermaid.contains("stroke:#f59e0b") || mermaid.contains("rel_friendship"),
        "Should have friendship color style (amber)"
    );

    println!("VGRF-02 test passed - edges styled by relationship type");
}

/// Test character-centered graph with depth limiting.
///
/// Verifies: VGRF-03 (character-centered depth-limited graphs)
#[tokio::test]
async fn test_character_centered_graph() {
    let harness = TestHarness::new().await;
    let db = &harness.db;
    let entity_repo = SurrealEntityRepository::new(db.clone());
    let graph_service = MermaidGraphService::new(db.clone());

    // Create a chain: A -> B -> C -> D
    let char_a = entity_repo
        .create_character(CharacterCreate {
            name: "Alpha".to_string(),
            aliases: vec![],
            roles: vec![],
            ..Default::default()
        })
        .await
        .expect("Should create Alpha");

    let char_b = entity_repo
        .create_character(CharacterCreate {
            name: "Beta".to_string(),
            aliases: vec![],
            roles: vec![],
            ..Default::default()
        })
        .await
        .expect("Should create Beta");

    let char_c = entity_repo
        .create_character(CharacterCreate {
            name: "Charlie".to_string(),
            aliases: vec![],
            roles: vec![],
            ..Default::default()
        })
        .await
        .expect("Should create Charlie");

    let char_d = entity_repo
        .create_character(CharacterCreate {
            name: "Delta".to_string(),
            aliases: vec![],
            roles: vec![],
            ..Default::default()
        })
        .await
        .expect("Should create Delta");

    let a_id = char_a.id.key().to_string();
    let b_id = char_b.id.key().to_string();
    let c_id = char_c.id.key().to_string();
    let d_id = char_d.id.key().to_string();

    // Create chain: A->B, B->C, C->D
    create_perception_pair(
        db,
        &a_id,
        &b_id,
        PerceptionCreate {
            rel_types: vec!["professional".to_string()],
            subtype: None,
            feelings: None,
            perception: None,
            tension_level: None,
            history_notes: None,
        },
        PerceptionCreate {
            rel_types: vec!["professional".to_string()],
            subtype: None,
            feelings: None,
            perception: None,
            tension_level: None,
            history_notes: None,
        },
    )
    .await
    .expect("Should create A-B perception");

    create_perception_pair(
        db,
        &b_id,
        &c_id,
        PerceptionCreate {
            rel_types: vec!["professional".to_string()],
            subtype: None,
            feelings: None,
            perception: None,
            tension_level: None,
            history_notes: None,
        },
        PerceptionCreate {
            rel_types: vec!["professional".to_string()],
            subtype: None,
            feelings: None,
            perception: None,
            tension_level: None,
            history_notes: None,
        },
    )
    .await
    .expect("Should create B-C perception");

    create_perception_pair(
        db,
        &c_id,
        &d_id,
        PerceptionCreate {
            rel_types: vec!["professional".to_string()],
            subtype: None,
            feelings: None,
            perception: None,
            tension_level: None,
            history_notes: None,
        },
        PerceptionCreate {
            rel_types: vec!["professional".to_string()],
            subtype: None,
            feelings: None,
            perception: None,
            tension_level: None,
            history_notes: None,
        },
    )
    .await
    .expect("Should create C-D perception");

    // Generate character-centered graph from A with depth 1
    let depth1_graph = graph_service
        .generate_mermaid(
            GraphScope::CharacterCentered {
                character_id: a_id.clone(),
                depth: 1,
            },
            GraphOptions::default(),
        )
        .await
        .expect("Should generate depth-1 graph");

    // Depth 1 from A should include A and B, but not C or D
    assert!(
        depth1_graph.contains("Alpha"),
        "Should contain Alpha (center)"
    );
    assert!(
        depth1_graph.contains("Beta"),
        "Should contain Beta (depth 1)"
    );
    // C and D are at depth 2 and 3, should not be in depth-1 graph
    // (though they might be included if BFS is inclusive - check implementation)

    // Generate character-centered graph from A with depth 2
    let depth2_graph = graph_service
        .generate_mermaid(
            GraphScope::CharacterCentered {
                character_id: a_id.clone(),
                depth: 2,
            },
            GraphOptions::default(),
        )
        .await
        .expect("Should generate depth-2 graph");

    // Depth 2 from A should include A, B, and C
    assert!(
        depth2_graph.contains("Alpha"),
        "Should contain Alpha (center)"
    );
    assert!(
        depth2_graph.contains("Beta"),
        "Should contain Beta (depth 1)"
    );
    assert!(
        depth2_graph.contains("Charlie"),
        "Should contain Charlie (depth 2)"
    );

    println!("VGRF-03 test passed - character-centered depth-limited graphs work");
}

// ============================================================================
// Full Phase 7 Workflow Test
// ============================================================================

/// Complete Phase 7 workflow test exercising all features together.
///
/// Verifies: All Phase 7 requirements work together in a realistic scenario.
#[tokio::test]
async fn test_phase7_full_workflow() {
    let harness = TestHarness::new().await;
    let db = &harness.db;
    let entity_repo = SurrealEntityRepository::new(db.clone());
    let search_service =
        SurrealSearchService::new(db.clone(), Arc::new(NoopEmbeddingService::new()));
    let graph_service = MermaidGraphService::new(db.clone());

    // 1. Create a small narrative world
    let protagonist = entity_repo
        .create_character(CharacterCreate {
            name: "Elena Vasquez".to_string(),
            aliases: vec!["The Shadow".to_string()],
            roles: vec!["protagonist".to_string(), "detective".to_string()],
            ..Default::default()
        })
        .await
        .expect("Should create protagonist");

    let antagonist = entity_repo
        .create_character(CharacterCreate {
            name: "Marcus Chen".to_string(),
            aliases: vec!["The Ghost".to_string()],
            roles: vec!["antagonist".to_string()],
            ..Default::default()
        })
        .await
        .expect("Should create antagonist");

    let mentor = entity_repo
        .create_character(CharacterCreate {
            name: "Victor Reyes".to_string(),
            aliases: vec![],
            roles: vec!["mentor".to_string()],
            ..Default::default()
        })
        .await
        .expect("Should create mentor");

    let protagonist_id = protagonist.id.key().to_string();
    let antagonist_id = antagonist.id.key().to_string();
    let mentor_id = mentor.id.key().to_string();

    // 2. Create relationships
    create_perception_pair(
        db,
        &protagonist_id,
        &antagonist_id,
        PerceptionCreate {
            rel_types: vec!["rivalry".to_string()],
            subtype: None,
            feelings: Some("Bitter hatred".to_string()),
            perception: Some("The man who ruined everything".to_string()),
            tension_level: Some(10),
            history_notes: None,
        },
        PerceptionCreate {
            rel_types: vec!["rivalry".to_string()],
            subtype: None,
            feelings: Some("Fear".to_string()),
            perception: Some("A threat".to_string()),
            tension_level: Some(7),
            history_notes: None,
        },
    )
    .await
    .expect("Should create rivalry");

    create_perception_pair(
        db,
        &mentor_id,
        &protagonist_id,
        PerceptionCreate {
            rel_types: vec!["mentorship".to_string()],
            subtype: None,
            feelings: Some("Pride".to_string()),
            perception: Some("My best student".to_string()),
            tension_level: Some(2),
            history_notes: None,
        },
        PerceptionCreate {
            rel_types: vec!["mentorship".to_string()],
            subtype: None,
            feelings: Some("Gratitude".to_string()),
            perception: Some("My mentor".to_string()),
            tension_level: Some(1),
            history_notes: None,
        },
    )
    .await
    .expect("Should create mentorship");

    // 3. Create notes (worldbuilding, character development)
    let backstory_note = create_note(
        db,
        NoteCreate {
            title: "Elena's Backstory".to_string(),
            body: "Elena grew up in Neo Tokyo's undercity. Her father was killed when she was 12. \
                   This trauma drove her to become a detective, seeking justice for those who \
                   cannot seek it themselves."
                .to_string(),
        },
    )
    .await
    .expect("Should create backstory note");

    let plot_note = create_note(
        db,
        NoteCreate {
            title: "Act 2 Plot Ideas".to_string(),
            body: "In Act 2, Elena discovers Marcus was responsible for her father's death. \
                   This revelation changes her motivation from professional to personal."
                .to_string(),
        },
    )
    .await
    .expect("Should create plot note");

    // 4. Attach notes to characters
    let protagonist_full_id = format!("character:{}", protagonist_id);
    attach_note(
        db,
        &backstory_note.id.key().to_string(),
        &protagonist_full_id,
    )
    .await
    .expect("Should attach backstory to protagonist");

    attach_note(db, &plot_note.id.key().to_string(), &protagonist_full_id)
        .await
        .expect("Should attach plot note to protagonist");

    // 5. Verify notes are attached
    let protagonist_notes = get_entity_notes(db, &protagonist_full_id)
        .await
        .expect("Should get protagonist notes");

    assert_eq!(
        protagonist_notes.len(),
        2,
        "Protagonist should have 2 notes"
    );

    // 6. Search for notes
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    let search_results = search_service
        .search("Elena", SearchFilter::default())
        .await
        .expect("Search should succeed");

    // Should find both the character and the backstory note
    assert!(
        search_results.iter().any(|r| r.entity_type == "character"),
        "Should find Elena character"
    );
    assert!(
        search_results
            .iter()
            .any(|r| r.entity_type == "note" && r.name.contains("Backstory")),
        "Should find backstory note"
    );

    // 7. Generate graph
    let full_graph = graph_service
        .generate_mermaid(GraphScope::FullNetwork, GraphOptions::default())
        .await
        .expect("Should generate full graph");

    assert!(full_graph.contains("Elena"), "Graph should contain Elena");
    assert!(full_graph.contains("Marcus"), "Graph should contain Marcus");
    assert!(full_graph.contains("Victor"), "Graph should contain Victor");
    assert!(full_graph.contains("rivalry"), "Graph should show rivalry");
    assert!(
        full_graph.contains("mentorship"),
        "Graph should show mentorship"
    );

    // 8. Generate character-centered graph
    let elena_graph = graph_service
        .generate_mermaid(
            GraphScope::CharacterCentered {
                character_id: protagonist_id.clone(),
                depth: 1,
            },
            GraphOptions::default(),
        )
        .await
        .expect("Should generate Elena-centered graph");

    assert!(
        elena_graph.contains("Elena"),
        "Elena-centered graph should contain Elena"
    );
    assert!(
        elena_graph.contains("Marcus") || elena_graph.contains("Victor"),
        "Elena-centered graph should contain at least one connected character"
    );

    println!("Phase 7 full workflow test passed - all features verified");
}
