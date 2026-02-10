//! Integration tests for session state and context management
//!
//! Verifies:
//! - Session state persists across restarts
//! - Startup context surfaces recent work
//! - Hot entity tracking works correctly
//! - MCP tools for session management

use chrono::{Duration, Utc};
use narra::db::connection::{init_db, DbConfig};
use narra::db::schema::apply_schema;
use narra::embedding::NoopEmbeddingService;
use narra::mcp::tools::SessionContextRequest;
use narra::mcp::NarraServer;
use narra::models::{CharacterCreate, LocationCreate};
use narra::repository::{EntityRepository, SurrealEntityRepository};
use narra::session::{
    generate_startup_context, PendingDecision, SessionStateManager, StartupVerbosity,
};
use std::sync::Arc;
use tempfile::TempDir;

#[tokio::test]
async fn test_session_state_persists_across_restarts() {
    let temp_dir = TempDir::new().unwrap();
    let session_path = temp_dir.path().join("session.json");

    // Create session, add state
    {
        let manager = SessionStateManager::load_or_create(&session_path).unwrap();

        // Pin entities
        manager.pin_entity("character:alice").await;
        manager.pin_entity("location:manor").await;

        // Record accesses
        manager.record_access("event:murder").await;
        manager.record_access("character:bob").await;
        manager.record_access("character:alice").await; // Alice should be most recent

        // Add pending decision
        let decision = PendingDecision {
            id: "decision-1".to_string(),
            description: "Kill off Bob?".to_string(),
            created_at: Utc::now(),
            entity_ids: vec!["character:bob".to_string()],
        };
        manager.add_pending_decision(decision).await;

        // Mark session end
        manager.mark_session_end().await;

        // Save state
        manager.save().await.unwrap();
    }

    // Drop and restart - simulate process restart
    {
        let manager = SessionStateManager::load_or_create(&session_path).unwrap();

        // Verify pinned entities preserved
        let pinned = manager.get_pinned().await;
        assert_eq!(pinned.len(), 2);
        assert!(pinned.contains(&"character:alice".to_string()));
        assert!(pinned.contains(&"location:manor".to_string()));

        // Verify recent accesses preserved in correct order
        let recent = manager.get_recent(10).await;
        assert_eq!(recent.len(), 3);
        assert_eq!(recent[0], "character:alice"); // Most recent
        assert_eq!(recent[1], "character:bob");
        assert_eq!(recent[2], "event:murder");

        // Verify pending decisions preserved
        let decisions = manager.get_pending_decisions().await;
        assert_eq!(decisions.len(), 1);
        assert_eq!(decisions[0].id, "decision-1");
        assert_eq!(decisions[0].description, "Kill off Bob?");

        // Verify session timestamp preserved
        let last_session = manager.get_last_session().await;
        assert!(last_session.is_some());

        println!("✓ Session state persisted across restart");
    }
}

#[tokio::test]
async fn test_startup_context_knows_last_work() {
    let temp_dir = TempDir::new().unwrap();
    let session_path = temp_dir.path().join("session.json");
    let db_path = temp_dir.path().join("test.db");

    // Setup database
    let db = Arc::new(
        init_db(
            &DbConfig::Embedded {
                path: Some(db_path.to_string_lossy().into_owned()),
            },
            temp_dir.path(),
        )
        .await
        .unwrap(),
    );
    apply_schema(&db).await.unwrap();
    let repo = SurrealEntityRepository::new(db.clone());

    // Create some entities
    let char1 = repo
        .create_character(CharacterCreate {
            name: "Elena".to_string(),
            aliases: vec![],
            roles: vec!["protagonist".to_string()],
            ..Default::default()
        })
        .await
        .unwrap();

    let char2 = repo
        .create_character(CharacterCreate {
            name: "Viktor".to_string(),
            aliases: vec![],
            roles: vec!["antagonist".to_string()],
            ..Default::default()
        })
        .await
        .unwrap();

    // Create session and record accesses
    let manager = SessionStateManager::load_or_create(&session_path).unwrap();
    let char1_id = format!("character:{}", char1.id.key());
    let char2_id = format!("character:{}", char2.id.key());

    manager.record_access(&char1_id).await;
    manager.record_access(&char2_id).await;
    manager.mark_session_end().await;

    // Generate startup context
    let startup_info = generate_startup_context(&manager, &db).await.unwrap();

    // Verify summary mentions recent work
    assert!(!startup_info.summary.is_empty());

    // Verify hot entities include characters we worked with
    assert!(!startup_info.hot_entities.is_empty());
    let entity_names: Vec<String> = startup_info
        .hot_entities
        .iter()
        .map(|e| e.name.clone())
        .collect();
    assert!(
        entity_names.contains(&"Elena".to_string()) || entity_names.contains(&"Viktor".to_string())
    );

    println!(
        "✓ Startup context knows last work: {}",
        startup_info.summary
    );
}

#[tokio::test]
async fn test_startup_surfaces_recent_context() {
    let temp_dir = TempDir::new().unwrap();
    let session_path = temp_dir.path().join("session.json");
    let db_path = temp_dir.path().join("test.db");

    // Setup database
    let db = Arc::new(
        init_db(
            &DbConfig::Embedded {
                path: Some(db_path.to_string_lossy().into_owned()),
            },
            temp_dir.path(),
        )
        .await
        .unwrap(),
    );
    apply_schema(&db).await.unwrap();
    let repo = SurrealEntityRepository::new(db.clone());

    // Create entities
    let char1 = repo
        .create_character(CharacterCreate {
            name: "Alice".to_string(),
            aliases: vec![],
            roles: vec!["hero".to_string()],
            ..Default::default()
        })
        .await
        .unwrap();

    let loc1 = repo
        .create_location(LocationCreate {
            name: "The Tower".to_string(),
            description: Some("A mysterious tower".to_string()),
            loc_type: "building".to_string(),
            parent: None,
        })
        .await
        .unwrap();

    // Create session with accesses
    let manager = SessionStateManager::load_or_create(&session_path).unwrap();
    manager
        .record_access(&format!("character:{}", char1.id.key()))
        .await;
    manager
        .record_access(&format!("location:{}", loc1.id.key()))
        .await;
    manager.mark_session_end().await;

    // Generate startup context
    let startup_info = generate_startup_context(&manager, &db).await.unwrap();

    // Verify "Last session" information present
    assert!(startup_info.last_session_ago.is_some());
    let time_ago = startup_info.last_session_ago.unwrap();
    assert!(time_ago.contains("just now") || time_ago.contains("ago"));

    // Verify hot entities listed
    assert!(!startup_info.hot_entities.is_empty());

    // Verify summary includes context
    assert!(!startup_info.summary.is_empty());

    println!(
        "✓ Recent context surfaced: last_session={}, {} hot entities",
        time_ago,
        startup_info.hot_entities.len()
    );
}

#[tokio::test]
async fn test_hot_entity_tracking() {
    let temp_dir = TempDir::new().unwrap();
    let session_path = temp_dir.path().join("session.json");
    let db_path = temp_dir.path().join("test.db");

    // Setup database
    let db = Arc::new(
        init_db(
            &DbConfig::Embedded {
                path: Some(db_path.to_string_lossy().into_owned()),
            },
            temp_dir.path(),
        )
        .await
        .unwrap(),
    );
    apply_schema(&db).await.unwrap();
    let repo = SurrealEntityRepository::new(db.clone());

    // Create entities A, B, C
    let char_a = repo
        .create_character(CharacterCreate {
            name: "A".to_string(),
            aliases: vec![],
            roles: vec![],
            ..Default::default()
        })
        .await
        .unwrap();

    let char_b = repo
        .create_character(CharacterCreate {
            name: "B".to_string(),
            aliases: vec![],
            roles: vec![],
            ..Default::default()
        })
        .await
        .unwrap();

    let char_c = repo
        .create_character(CharacterCreate {
            name: "C".to_string(),
            aliases: vec![],
            roles: vec![],
            ..Default::default()
        })
        .await
        .unwrap();

    // Create session and access: A, B, C, A, B, A
    let manager = SessionStateManager::load_or_create(&session_path).unwrap();
    let a_id = format!("character:{}", char_a.id.key());
    let b_id = format!("character:{}", char_b.id.key());
    let c_id = format!("character:{}", char_c.id.key());

    manager.record_access(&a_id).await;
    manager.record_access(&b_id).await;
    manager.record_access(&c_id).await;
    manager.record_access(&a_id).await;
    manager.record_access(&b_id).await;
    manager.record_access(&a_id).await;

    // Get recent accesses (acts as hot entity list)
    let recent = manager.get_recent(10).await;

    // Verify order: A (most recent), B, C
    assert_eq!(recent.len(), 3);
    assert_eq!(recent[0], a_id); // A accessed most recently
    assert_eq!(recent[1], b_id); // B second most recent
    assert_eq!(recent[2], c_id); // C least recent

    // Verify that hot entities from startup context match
    let startup_info = generate_startup_context(&manager, &db).await.unwrap();
    assert_eq!(startup_info.hot_entities.len(), 3);
    assert_eq!(startup_info.hot_entities[0].name, "A");
    assert_eq!(startup_info.hot_entities[1].name, "B");
    assert_eq!(startup_info.hot_entities[2].name, "C");

    println!(
        "✓ Hot entity tracking: {} entities in correct order",
        recent.len()
    );
}

#[tokio::test]
async fn test_startup_verbosity_by_time() {
    let temp_dir = TempDir::new().unwrap();
    let session_path = temp_dir.path().join("session.json");
    let db_path = temp_dir.path().join("test.db");

    // Setup database with data
    let db = Arc::new(
        init_db(
            &DbConfig::Embedded {
                path: Some(db_path.to_string_lossy().into_owned()),
            },
            temp_dir.path(),
        )
        .await
        .unwrap(),
    );
    apply_schema(&db).await.unwrap();
    let repo = SurrealEntityRepository::new(db.clone());

    repo.create_character(CharacterCreate {
        name: "Test Character".to_string(),
        aliases: vec![],
        roles: vec![],
        ..Default::default()
    })
    .await
    .unwrap();

    // Test Brief: last_session < 24 hours
    {
        let manager = SessionStateManager::load_or_create(&session_path).unwrap();
        manager.mark_session_end().await;
        manager.save().await.unwrap(); // Save before generating context

        let startup_info = generate_startup_context(&manager, &db).await.unwrap();
        assert_eq!(startup_info.verbosity, StartupVerbosity::Brief);
        println!("✓ Brief verbosity: {}", startup_info.summary);
    }

    // Test Standard: last_session 1-7 days (manually set timestamp)
    {
        // Manually set last_session to 3 days ago by modifying saved state
        let three_days_ago = Utc::now() - Duration::days(3);

        use std::fs;
        let mut state: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&session_path).unwrap()).unwrap();
        state["last_session"] = serde_json::json!(three_days_ago.to_rfc3339());
        fs::write(&session_path, serde_json::to_string_pretty(&state).unwrap()).unwrap();

        // Reload manager
        let manager = SessionStateManager::load_or_create(&session_path).unwrap();
        let startup_info = generate_startup_context(&manager, &db).await.unwrap();
        assert_eq!(startup_info.verbosity, StartupVerbosity::Standard);
        println!("✓ Standard verbosity: {}", startup_info.summary);
    }

    // Test Full: last_session > 7 days
    {
        let ten_days_ago = Utc::now() - Duration::days(10);

        use std::fs;
        let mut state: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&session_path).unwrap()).unwrap();
        state["last_session"] = serde_json::json!(ten_days_ago.to_rfc3339());
        fs::write(&session_path, serde_json::to_string_pretty(&state).unwrap()).unwrap();

        let manager = SessionStateManager::load_or_create(&session_path).unwrap();
        let startup_info = generate_startup_context(&manager, &db).await.unwrap();
        assert_eq!(startup_info.verbosity, StartupVerbosity::Full);
        println!("✓ Full verbosity: {}", startup_info.summary);
    }

    // Test NewWorld: no last_session + has data
    {
        let fresh_path = temp_dir.path().join("fresh_session.json");
        let manager = SessionStateManager::load_or_create(&fresh_path).unwrap();

        let startup_info = generate_startup_context(&manager, &db).await.unwrap();
        assert_eq!(startup_info.verbosity, StartupVerbosity::NewWorld);
        assert!(startup_info.world_overview.is_some());
        println!("✓ NewWorld verbosity: {}", startup_info.summary);
    }

    // Test EmptyWorld: no last_session + no data
    {
        let empty_db_path = temp_dir.path().join("empty.db");
        let empty_db = Arc::new(
            init_db(
                &DbConfig::Embedded {
                    path: Some(empty_db_path.to_string_lossy().into_owned()),
                },
                temp_dir.path(),
            )
            .await
            .unwrap(),
        );
        apply_schema(&empty_db).await.unwrap();

        let empty_session_path = temp_dir.path().join("empty_session.json");
        let manager = SessionStateManager::load_or_create(&empty_session_path).unwrap();

        let startup_info = generate_startup_context(&manager, &empty_db).await.unwrap();
        assert_eq!(startup_info.verbosity, StartupVerbosity::EmptyWorld);
        assert!(startup_info.summary.contains("empty"));
        println!("✓ EmptyWorld verbosity: {}", startup_info.summary);
    }
}

#[tokio::test]
async fn test_pending_decisions_surfaced() {
    let temp_dir = TempDir::new().unwrap();
    let session_path = temp_dir.path().join("session.json");
    let db_path = temp_dir.path().join("test.db");

    // Setup
    let db = Arc::new(
        init_db(
            &DbConfig::Embedded {
                path: Some(db_path.to_string_lossy().into_owned()),
            },
            temp_dir.path(),
        )
        .await
        .unwrap(),
    );
    apply_schema(&db).await.unwrap();

    // Create session with pending decisions
    let manager = SessionStateManager::load_or_create(&session_path).unwrap();

    let decision1 = PendingDecision {
        id: "decision-1".to_string(),
        description: "Should the protagonist survive?".to_string(),
        created_at: Utc::now(),
        entity_ids: vec!["character:hero".to_string()],
    };

    let decision2 = PendingDecision {
        id: "decision-2".to_string(),
        description: "Merge locations or keep separate?".to_string(),
        created_at: Utc::now() - Duration::hours(2),
        entity_ids: vec!["location:a".to_string(), "location:b".to_string()],
    };

    manager.add_pending_decision(decision1).await;
    manager.add_pending_decision(decision2).await;
    manager.mark_session_end().await;

    // Generate startup context
    let startup_info = generate_startup_context(&manager, &db).await.unwrap();

    // Verify pending decisions in response
    assert_eq!(startup_info.pending_decisions.len(), 2);

    let desc1 = &startup_info.pending_decisions[0].description;
    let desc2 = &startup_info.pending_decisions[1].description;
    assert!(desc1.contains("protagonist") || desc2.contains("protagonist"));
    assert!(desc1.contains("Merge") || desc2.contains("Merge"));

    // Verify decisions have age information
    assert!(!startup_info.pending_decisions[0].age.is_empty());
    assert!(!startup_info.pending_decisions[1].age.is_empty());

    println!(
        "✓ Pending decisions surfaced: {} decisions",
        startup_info.pending_decisions.len()
    );
}

#[tokio::test]
async fn test_get_session_context_mcp_tool() {
    let temp_dir = TempDir::new().unwrap();
    let session_path = temp_dir.path().join("session.json");
    let db_path = temp_dir.path().join("test.db");

    // Setup database with data
    let db = Arc::new(
        init_db(
            &DbConfig::Embedded {
                path: Some(db_path.to_string_lossy().into_owned()),
            },
            temp_dir.path(),
        )
        .await
        .unwrap(),
    );
    apply_schema(&db).await.unwrap();
    let repo = SurrealEntityRepository::new(db.clone());

    let char1 = repo
        .create_character(CharacterCreate {
            name: "Marie".to_string(),
            aliases: vec![],
            roles: vec!["detective".to_string()],
            ..Default::default()
        })
        .await
        .unwrap();

    // Setup session manager
    let session_manager = Arc::new(SessionStateManager::load_or_create(&session_path).unwrap());
    session_manager
        .record_access(&format!("character:{}", char1.id.key()))
        .await;
    session_manager.mark_session_end().await;

    // Create server
    let server = NarraServer::new(
        db.clone(),
        session_manager,
        Arc::new(NoopEmbeddingService::new()),
    )
    .await;

    // Call get_session_context tool
    let request = SessionContextRequest { force_full: false };
    let response = server
        .handle_get_session_context(rmcp::handler::server::wrapper::Parameters(request))
        .await
        .expect("get_session_context failed");

    // Verify response includes expected fields
    assert!(!response.verbosity.is_empty());
    assert!(!response.summary.is_empty());
    assert!(response.last_session_ago.is_some());

    // Verify hot entities present
    assert!(!response.hot_entities.is_empty());
    assert_eq!(response.hot_entities[0].name, "Marie");
    assert_eq!(response.hot_entities[0].entity_type, "character");

    println!(
        "✓ MCP tool get_session_context: verbosity={}, {} hot entities",
        response.verbosity,
        response.hot_entities.len()
    );
}

#[tokio::test]
async fn test_export_and_verify_structure() {
    use narra::mcp::tools::ExportRequest;

    let temp_dir = TempDir::new().unwrap();
    let session_path = temp_dir.path().join("session.json");
    let db_path = temp_dir.path().join("test.db");

    // Setup database with various entities
    let db = Arc::new(
        init_db(
            &DbConfig::Embedded {
                path: Some(db_path.to_string_lossy().into_owned()),
            },
            temp_dir.path(),
        )
        .await
        .unwrap(),
    );
    apply_schema(&db).await.unwrap();
    let repo = SurrealEntityRepository::new(db.clone());

    repo.create_character(CharacterCreate {
        name: "Export Test Character".to_string(),
        aliases: vec![],
        roles: vec![],
        ..Default::default()
    })
    .await
    .unwrap();

    repo.create_location(LocationCreate {
        name: "Export Test Location".to_string(),
        description: None,
        loc_type: "place".to_string(),
        parent: None,
    })
    .await
    .unwrap();

    // Setup session manager
    let session_manager = Arc::new(SessionStateManager::load_or_create(&session_path).unwrap());
    session_manager.pin_entity("character:test").await;

    // Create server
    let server = NarraServer::new(
        db.clone(),
        session_manager,
        Arc::new(NoopEmbeddingService::new()),
    )
    .await;

    // Export
    let export_path = temp_dir.path().join("export.yaml");
    let request = ExportRequest {
        output_path: Some(export_path.to_str().unwrap().to_string()),
    };

    let response = server
        .handle_export_world(rmcp::handler::server::wrapper::Parameters(request))
        .await
        .expect("export_world failed");

    // Verify file exists
    assert!(std::path::Path::new(&response.output_path).exists());

    // Parse and verify YAML structure as NarraImport
    let yaml_content = std::fs::read_to_string(&response.output_path).unwrap();
    assert!(yaml_content.starts_with("# Narra world export"));
    let import: narra::mcp::types::NarraImport = serde_yaml_ng::from_str(&yaml_content).unwrap();

    // Verify counts
    assert_eq!(response.summary.character_count, 1);
    assert_eq!(response.summary.location_count, 1);
    assert_eq!(import.characters.len(), 1);
    assert_eq!(import.locations.len(), 1);

    println!(
        "✓ Export structure valid: {} chars, {} locs",
        response.summary.character_count, response.summary.location_count
    );
}
