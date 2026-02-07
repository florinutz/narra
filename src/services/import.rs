use std::sync::Arc;

use surrealdb::engine::local::Db;
use surrealdb::{RecordId, Surreal};

use crate::embedding::StalenessManager;
use crate::mcp::types::{
    CharacterSpec, ConflictMode, EventSpec, FactSpec, ImportResult, ImportTypeResult,
    KnowledgeSpec, LocationSpec, NarraImport, NoteSpec, RelationshipSpec, SceneSpec,
};
use crate::models::character::{
    create_character, create_character_with_id, get_character, update_character, CharacterCreate,
    CharacterUpdate,
};
use crate::models::event::{
    create_event, create_event_with_id, get_event, update_event, EventCreate, EventUpdate,
};
use crate::models::fact::{
    create_fact, create_fact_with_id, get_fact, link_fact_to_entity, update_fact, FactCreate,
    FactUpdate,
};
use crate::models::knowledge::{
    create_knowledge, create_knowledge_state, CertaintyLevel, KnowledgeCreate,
    KnowledgeStateCreate, LearningMethod,
};
use crate::models::location::{
    create_location, create_location_with_id, get_location, update_location, LocationCreate,
    LocationUpdate,
};
use crate::models::note::{
    attach_note, create_note, create_note_with_id, get_note, update_note, NoteCreate, NoteUpdate,
};
use crate::models::relationship::{create_relationship, RelationshipCreate};
use crate::models::scene::{
    add_scene_participant, create_scene, create_scene_with_id, get_scene, update_scene,
    SceneCreate, SceneParticipantCreate, SceneUpdate,
};
use crate::NarraError;

pub struct ImportService {
    db: Arc<Surreal<Db>>,
    staleness_manager: Arc<StalenessManager>,
}

impl ImportService {
    pub fn new(db: Arc<Surreal<Db>>, staleness_manager: Arc<StalenessManager>) -> Self {
        Self {
            db,
            staleness_manager,
        }
    }

    pub async fn execute_import(
        &self,
        import: NarraImport,
        mode: ConflictMode,
    ) -> Result<ImportResult, NarraError> {
        let mut result = ImportResult::default();

        // Process in dependency order
        let char_result = self.import_characters(&import.characters, mode).await;
        let loc_result = self.import_locations(&import.locations, mode).await;
        let event_result = self.import_events(&import.events, mode).await;
        let scene_result = self.import_scenes(&import.scenes, mode).await;
        let rel_result = self.import_relationships(&import.relationships).await;
        let know_result = self.import_knowledge(&import.knowledge).await;
        let note_result = self.import_notes(&import.notes, mode).await;
        let fact_result = self.import_facts(&import.facts, mode).await;

        let type_results = vec![
            char_result,
            loc_result,
            event_result,
            scene_result,
            rel_result,
            know_result,
            note_result,
            fact_result,
        ];

        for tr in &type_results {
            result.total_created += tr.created;
            result.total_skipped += tr.skipped;
            result.total_updated += tr.updated;
            result.total_errors += tr.errors.len();
        }
        result.by_type = type_results;

        Ok(result)
    }

    async fn import_characters(
        &self,
        specs: &[CharacterSpec],
        mode: ConflictMode,
    ) -> ImportTypeResult {
        let mut result = ImportTypeResult {
            entity_type: "character".to_string(),
            ..Default::default()
        };

        for spec in specs {
            let create_data = CharacterCreate {
                name: spec.name.clone(),
                aliases: spec.aliases.clone().unwrap_or_default(),
                roles: spec
                    .role
                    .as_ref()
                    .map(|r| vec![r.clone()])
                    .unwrap_or_default(),
                profile: spec.profile.clone().unwrap_or_default(),
            };

            if let Some(ref id) = spec.id {
                // Check existence for conflict handling
                match get_character(&self.db, id).await {
                    Ok(Some(existing)) => match mode {
                        ConflictMode::Error => {
                            result
                                .errors
                                .push(format!("Character '{}' already exists", id));
                            continue;
                        }
                        ConflictMode::Skip => {
                            result.skipped += 1;
                            continue;
                        }
                        ConflictMode::Update => {
                            let update = CharacterUpdate {
                                name: Some(spec.name.clone()),
                                aliases: spec.aliases.clone(),
                                roles: spec.role.as_ref().map(|r| vec![r.clone()]),
                                profile: spec.profile.clone(),
                                updated_at: existing.updated_at,
                            };
                            match update_character(&self.db, id, update).await {
                                Ok(Some(_)) => {
                                    result.updated += 1;
                                    self.spawn_regen(&format!("character:{}", id), "character");
                                }
                                Ok(None) => {
                                    result
                                        .errors
                                        .push(format!("Character '{}' update returned None", id));
                                }
                                Err(e) => {
                                    result.errors.push(format!(
                                        "Failed to update character '{}': {}",
                                        id, e
                                    ));
                                }
                            }
                            continue;
                        }
                    },
                    Ok(None) => {}
                    Err(e) => {
                        result
                            .errors
                            .push(format!("Failed to check character '{}': {}", id, e));
                        continue;
                    }
                }

                match create_character_with_id(&self.db, id, create_data).await {
                    Ok(c) => {
                        result.created += 1;
                        self.spawn_regen(&c.id.to_string(), "character");
                    }
                    Err(e) => {
                        result
                            .errors
                            .push(format!("Failed to create character '{}': {}", id, e));
                    }
                }
            } else {
                match create_character(&self.db, create_data).await {
                    Ok(c) => {
                        result.created += 1;
                        self.spawn_regen(&c.id.to_string(), "character");
                    }
                    Err(e) => {
                        result
                            .errors
                            .push(format!("Failed to create character '{}': {}", spec.name, e));
                    }
                }
            }
        }

        result
    }

    async fn import_locations(
        &self,
        specs: &[LocationSpec],
        mode: ConflictMode,
    ) -> ImportTypeResult {
        let mut result = ImportTypeResult {
            entity_type: "location".to_string(),
            ..Default::default()
        };

        for spec in specs {
            let parent_record_id = match spec.parent_id.as_ref() {
                Some(pid) => match pid.parse::<RecordId>() {
                    Ok(rid) => Some(rid),
                    Err(e) => {
                        result.errors.push(format!(
                            "Invalid parent_id '{}' for '{}': {}",
                            pid, spec.name, e
                        ));
                        continue;
                    }
                },
                None => None,
            };

            let create_data = LocationCreate {
                name: spec.name.clone(),
                description: spec.description.clone(),
                loc_type: spec.loc_type.clone().unwrap_or_else(|| "place".to_string()),
                parent: parent_record_id.clone(),
            };

            if let Some(ref id) = spec.id {
                match get_location(&self.db, id).await {
                    Ok(Some(existing)) => match mode {
                        ConflictMode::Error => {
                            result
                                .errors
                                .push(format!("Location '{}' already exists", id));
                            continue;
                        }
                        ConflictMode::Skip => {
                            result.skipped += 1;
                            continue;
                        }
                        ConflictMode::Update => {
                            let update = LocationUpdate {
                                name: Some(spec.name.clone()),
                                description: Some(spec.description.clone()),
                                loc_type: None,
                                parent: parent_record_id.map(Some),
                                updated_at: existing.updated_at,
                            };
                            match update_location(&self.db, id, update).await {
                                Ok(Some(_)) => {
                                    result.updated += 1;
                                    self.spawn_regen(&format!("location:{}", id), "location");
                                }
                                Ok(None) => {
                                    result
                                        .errors
                                        .push(format!("Location '{}' update returned None", id));
                                }
                                Err(e) => {
                                    result
                                        .errors
                                        .push(format!("Failed to update location '{}': {}", id, e));
                                }
                            }
                            continue;
                        }
                    },
                    Ok(None) => {}
                    Err(e) => {
                        result
                            .errors
                            .push(format!("Failed to check location '{}': {}", id, e));
                        continue;
                    }
                }

                match create_location_with_id(&self.db, id, create_data).await {
                    Ok(loc) => {
                        result.created += 1;
                        self.spawn_regen(&loc.id.to_string(), "location");
                    }
                    Err(e) => {
                        result
                            .errors
                            .push(format!("Failed to create location '{}': {}", id, e));
                    }
                }
            } else {
                match create_location(&self.db, create_data).await {
                    Ok(loc) => {
                        result.created += 1;
                        self.spawn_regen(&loc.id.to_string(), "location");
                    }
                    Err(e) => {
                        result
                            .errors
                            .push(format!("Failed to create location '{}': {}", spec.name, e));
                    }
                }
            }
        }

        result
    }

    async fn import_events(&self, specs: &[EventSpec], mode: ConflictMode) -> ImportTypeResult {
        let mut result = ImportTypeResult {
            entity_type: "event".to_string(),
            ..Default::default()
        };

        for spec in specs {
            let parsed_date = match spec.date.as_ref() {
                Some(d) => match chrono::DateTime::parse_from_rfc3339(d) {
                    Ok(dt) => Some(dt.with_timezone(&chrono::Utc).into()),
                    Err(e) => {
                        result
                            .errors
                            .push(format!("Invalid date '{}' for '{}': {}", d, spec.title, e));
                        continue;
                    }
                },
                None => None,
            };

            let create_data = EventCreate {
                title: spec.title.clone(),
                description: spec.description.clone(),
                sequence: spec.sequence.unwrap_or(0) as i64,
                date: parsed_date,
                date_precision: spec.date_precision.clone(),
                duration_end: None,
            };

            if let Some(ref id) = spec.id {
                match get_event(&self.db, id).await {
                    Ok(Some(existing)) => match mode {
                        ConflictMode::Error => {
                            result.errors.push(format!("Event '{}' already exists", id));
                            continue;
                        }
                        ConflictMode::Skip => {
                            result.skipped += 1;
                            continue;
                        }
                        ConflictMode::Update => {
                            let update = EventUpdate {
                                title: Some(spec.title.clone()),
                                description: Some(spec.description.clone()),
                                sequence: spec.sequence.map(|s| s as i64),
                                date: None,
                                date_precision: None,
                                duration_end: None,
                                updated_at: existing.updated_at,
                            };
                            match update_event(&self.db, id, update).await {
                                Ok(Some(_)) => {
                                    result.updated += 1;
                                    self.spawn_regen(&format!("event:{}", id), "event");
                                }
                                Ok(None) => {
                                    result
                                        .errors
                                        .push(format!("Event '{}' update returned None", id));
                                }
                                Err(e) => {
                                    result
                                        .errors
                                        .push(format!("Failed to update event '{}': {}", id, e));
                                }
                            }
                            continue;
                        }
                    },
                    Ok(None) => {}
                    Err(e) => {
                        result
                            .errors
                            .push(format!("Failed to check event '{}': {}", id, e));
                        continue;
                    }
                }

                match create_event_with_id(&self.db, id, create_data).await {
                    Ok(ev) => {
                        result.created += 1;
                        self.spawn_regen(&ev.id.to_string(), "event");
                    }
                    Err(e) => {
                        result
                            .errors
                            .push(format!("Failed to create event '{}': {}", id, e));
                    }
                }
            } else {
                match create_event(&self.db, create_data).await {
                    Ok(ev) => {
                        result.created += 1;
                        self.spawn_regen(&ev.id.to_string(), "event");
                    }
                    Err(e) => {
                        result
                            .errors
                            .push(format!("Failed to create event '{}': {}", spec.title, e));
                    }
                }
            }
        }

        result
    }

    async fn import_scenes(&self, specs: &[SceneSpec], mode: ConflictMode) -> ImportTypeResult {
        let mut result = ImportTypeResult {
            entity_type: "scene".to_string(),
            ..Default::default()
        };

        for spec in specs {
            let event_rid: RecordId = match spec.event_id.parse() {
                Ok(rid) => rid,
                Err(e) => {
                    result.errors.push(format!(
                        "Invalid event_id '{}' for scene '{}': {}",
                        spec.event_id, spec.title, e
                    ));
                    continue;
                }
            };

            let location_rid: RecordId = match spec.location_id.parse() {
                Ok(rid) => rid,
                Err(e) => {
                    result.errors.push(format!(
                        "Invalid location_id '{}' for scene '{}': {}",
                        spec.location_id, spec.title, e
                    ));
                    continue;
                }
            };

            let secondary_locs: Vec<RecordId> = {
                let mut locs = Vec::new();
                let mut has_error = false;
                for sl in &spec.secondary_locations {
                    match sl.parse::<RecordId>() {
                        Ok(rid) => locs.push(rid),
                        Err(e) => {
                            result.errors.push(format!(
                                "Invalid secondary_location '{}' for scene '{}': {}",
                                sl, spec.title, e
                            ));
                            has_error = true;
                            break;
                        }
                    }
                }
                if has_error {
                    continue;
                }
                locs
            };

            let create_data = SceneCreate {
                title: spec.title.clone(),
                summary: spec.summary.clone(),
                event: event_rid,
                primary_location: location_rid,
                secondary_locations: secondary_locs,
            };

            let scene_id = if let Some(ref id) = spec.id {
                match get_scene(&self.db, id).await {
                    Ok(Some(existing)) => match mode {
                        ConflictMode::Error => {
                            result.errors.push(format!("Scene '{}' already exists", id));
                            continue;
                        }
                        ConflictMode::Skip => {
                            result.skipped += 1;
                            continue;
                        }
                        ConflictMode::Update => {
                            let update = SceneUpdate {
                                title: Some(spec.title.clone()),
                                summary: Some(spec.summary.clone()),
                                event: None,
                                primary_location: None,
                                secondary_locations: None,
                                updated_at: existing.updated_at,
                            };
                            match update_scene(&self.db, id, update).await {
                                Ok(Some(_)) => {
                                    result.updated += 1;
                                    self.spawn_regen(&format!("scene:{}", id), "scene");
                                }
                                Ok(None) => {
                                    result
                                        .errors
                                        .push(format!("Scene '{}' update returned None", id));
                                }
                                Err(e) => {
                                    result
                                        .errors
                                        .push(format!("Failed to update scene '{}': {}", id, e));
                                }
                            }
                            // Skip participant creation on update
                            continue;
                        }
                    },
                    Ok(None) => {}
                    Err(e) => {
                        result
                            .errors
                            .push(format!("Failed to check scene '{}': {}", id, e));
                        continue;
                    }
                }

                match create_scene_with_id(&self.db, id, create_data).await {
                    Ok(s) => {
                        result.created += 1;
                        self.spawn_regen(&s.id.to_string(), "scene");
                        s.id.key().to_string()
                    }
                    Err(e) => {
                        result
                            .errors
                            .push(format!("Failed to create scene '{}': {}", id, e));
                        continue;
                    }
                }
            } else {
                match create_scene(&self.db, create_data).await {
                    Ok(s) => {
                        result.created += 1;
                        self.spawn_regen(&s.id.to_string(), "scene");
                        s.id.key().to_string()
                    }
                    Err(e) => {
                        result
                            .errors
                            .push(format!("Failed to create scene '{}': {}", spec.title, e));
                        continue;
                    }
                }
            };

            // Add participants
            for p in &spec.participants {
                let participant_data = SceneParticipantCreate {
                    character_id: p.character_id.clone(),
                    scene_id: scene_id.clone(),
                    role: p.role.clone(),
                    notes: p.notes.clone(),
                };
                if let Err(e) = add_scene_participant(&self.db, participant_data).await {
                    result.errors.push(format!(
                        "Failed to add participant '{}' to scene '{}': {}",
                        p.character_id, scene_id, e
                    ));
                }
            }
        }

        result
    }

    async fn import_relationships(&self, specs: &[RelationshipSpec]) -> ImportTypeResult {
        let mut result = ImportTypeResult {
            entity_type: "relationship".to_string(),
            ..Default::default()
        };

        for spec in specs {
            // Check for existing relationship with same in/out/type
            let check_query = format!(
                "SELECT * FROM relates_to WHERE in = character:{} AND out = character:{} AND rel_type = $rel_type",
                spec.from_character_id, spec.to_character_id
            );
            let existing = self
                .db
                .query(&check_query)
                .bind(("rel_type", spec.rel_type.clone()))
                .await;

            if let Ok(mut resp) = existing {
                let rows: Vec<crate::models::relationship::Relationship> =
                    resp.take(0).unwrap_or_default();
                if !rows.is_empty() {
                    result.skipped += 1;
                    continue;
                }
            }

            let create_data = RelationshipCreate {
                from_character_id: spec.from_character_id.clone(),
                to_character_id: spec.to_character_id.clone(),
                rel_type: spec.rel_type.clone(),
                subtype: spec.subtype.clone(),
                label: spec.label.clone(),
            };

            match create_relationship(&self.db, create_data).await {
                Ok(rel) => {
                    result.created += 1;
                    self.spawn_regen(&rel.id.to_string(), "relates_to");
                }
                Err(e) => {
                    result.errors.push(format!(
                        "Failed to create relationship {} -> {}: {}",
                        spec.from_character_id, spec.to_character_id, e
                    ));
                }
            }
        }

        result
    }

    async fn import_knowledge(&self, specs: &[KnowledgeSpec]) -> ImportTypeResult {
        let mut result = ImportTypeResult {
            entity_type: "knowledge".to_string(),
            ..Default::default()
        };

        for spec in specs {
            // Step 1: Create knowledge entity
            let char_record_id: RecordId = match spec.character_id.parse() {
                Ok(rid) => rid,
                Err(e) => {
                    result.errors.push(format!(
                        "Invalid character_id '{}': {}",
                        spec.character_id, e
                    ));
                    continue;
                }
            };

            let knowledge_entity = match create_knowledge(
                &self.db,
                KnowledgeCreate {
                    character: char_record_id,
                    fact: spec.fact.clone(),
                },
            )
            .await
            {
                Ok(k) => k,
                Err(e) => {
                    result.errors.push(format!(
                        "Failed to create knowledge entity for '{}': {}",
                        spec.fact, e
                    ));
                    continue;
                }
            };

            // Step 2: Parse certainty and method
            let certainty_level = parse_certainty(&spec.certainty);
            let learning_method = spec
                .method
                .as_ref()
                .map(|m| parse_learning_method(m))
                .unwrap_or(LearningMethod::Initial);

            let knowledge_target = format!("knowledge:{}", knowledge_entity.id.key());
            let char_key = spec
                .character_id
                .split(':')
                .nth(1)
                .unwrap_or(&spec.character_id);

            // Step 3: Create knows edge
            let create = KnowledgeStateCreate {
                certainty: certainty_level,
                learning_method,
                source_character: spec.source_character_id.clone(),
                event: spec.event_id.clone(),
                premises: None,
                truth_value: if certainty_level == CertaintyLevel::BelievesWrongly {
                    Some(spec.fact.clone())
                } else {
                    None
                },
            };

            match create_knowledge_state(&self.db, char_key, &knowledge_target, create).await {
                Ok(_) => {
                    result.created += 1;
                    let knowledge_entity_id = knowledge_entity.id.to_string();
                    self.staleness_manager.spawn_regeneration(
                        knowledge_entity_id,
                        "knowledge".to_string(),
                        spec.event_id.clone(),
                    );
                }
                Err(e) => {
                    result.errors.push(format!(
                        "Failed to create knowledge state for '{}': {}",
                        spec.fact, e
                    ));
                }
            }
        }

        result
    }

    async fn import_notes(&self, specs: &[NoteSpec], mode: ConflictMode) -> ImportTypeResult {
        let mut result = ImportTypeResult {
            entity_type: "note".to_string(),
            ..Default::default()
        };

        for spec in specs {
            let create_data = NoteCreate {
                title: spec.title.clone(),
                body: spec.body.clone(),
            };

            let note_id = if let Some(ref id) = spec.id {
                match get_note(&self.db, id).await {
                    Ok(Some(_)) => match mode {
                        ConflictMode::Error => {
                            result.errors.push(format!("Note '{}' already exists", id));
                            continue;
                        }
                        ConflictMode::Skip => {
                            result.skipped += 1;
                            continue;
                        }
                        ConflictMode::Update => {
                            let update = NoteUpdate {
                                title: Some(spec.title.clone()),
                                body: Some(spec.body.clone()),
                            };
                            match update_note(&self.db, id, update).await {
                                Ok(Some(_)) => {
                                    result.updated += 1;
                                }
                                Ok(None) => {
                                    result
                                        .errors
                                        .push(format!("Note '{}' update returned None", id));
                                }
                                Err(e) => {
                                    result
                                        .errors
                                        .push(format!("Failed to update note '{}': {}", id, e));
                                }
                            }
                            continue;
                        }
                    },
                    Ok(None) => {}
                    Err(e) => {
                        result
                            .errors
                            .push(format!("Failed to check note '{}': {}", id, e));
                        continue;
                    }
                }

                match create_note_with_id(&self.db, id, create_data).await {
                    Ok(n) => {
                        result.created += 1;
                        n.id.key().to_string()
                    }
                    Err(e) => {
                        result
                            .errors
                            .push(format!("Failed to create note '{}': {}", id, e));
                        continue;
                    }
                }
            } else {
                match create_note(&self.db, create_data).await {
                    Ok(n) => {
                        result.created += 1;
                        n.id.key().to_string()
                    }
                    Err(e) => {
                        result
                            .errors
                            .push(format!("Failed to create note '{}': {}", spec.title, e));
                        continue;
                    }
                }
            };

            // Attach to entities
            for entity_id in &spec.attach_to {
                if let Err(e) = attach_note(&self.db, &note_id, entity_id).await {
                    result.errors.push(format!(
                        "Failed to attach note '{}' to '{}': {}",
                        note_id, entity_id, e
                    ));
                }
            }
        }

        result
    }

    async fn import_facts(&self, specs: &[FactSpec], mode: ConflictMode) -> ImportTypeResult {
        let mut result = ImportTypeResult {
            entity_type: "fact".to_string(),
            ..Default::default()
        };

        for spec in specs {
            let categories: Vec<crate::models::fact::FactCategory> = spec
                .categories
                .iter()
                .map(|c| match c.as_str() {
                    "physics_magic" => crate::models::fact::FactCategory::PhysicsMagic,
                    "social_cultural" => crate::models::fact::FactCategory::SocialCultural,
                    "technology" => crate::models::fact::FactCategory::Technology,
                    other => crate::models::fact::FactCategory::Custom(other.to_string()),
                })
                .collect();

            let enforcement = spec
                .enforcement_level
                .as_ref()
                .map(|e| match e.as_str() {
                    "informational" => crate::models::fact::EnforcementLevel::Informational,
                    "strict" => crate::models::fact::EnforcementLevel::Strict,
                    _ => crate::models::fact::EnforcementLevel::Warning,
                })
                .unwrap_or_default();

            let create_data = FactCreate {
                title: spec.title.clone(),
                description: spec.description.clone(),
                categories: categories.clone(),
                enforcement_level: enforcement,
                scope: None,
            };

            let fact_id = if let Some(ref id) = spec.id {
                match get_fact(&self.db, id).await {
                    Ok(Some(existing)) => match mode {
                        ConflictMode::Error => {
                            result.errors.push(format!("Fact '{}' already exists", id));
                            continue;
                        }
                        ConflictMode::Skip => {
                            result.skipped += 1;
                            continue;
                        }
                        ConflictMode::Update => {
                            let update = FactUpdate {
                                title: Some(spec.title.clone()),
                                description: Some(spec.description.clone()),
                                categories: Some(categories),
                                enforcement_level: Some(enforcement),
                                scope: None,
                                updated_at: existing.updated_at,
                            };
                            match update_fact(&self.db, id, update).await {
                                Ok(Some(_)) => {
                                    result.updated += 1;
                                }
                                Ok(None) => {
                                    result
                                        .errors
                                        .push(format!("Fact '{}' update returned None", id));
                                }
                                Err(e) => {
                                    result
                                        .errors
                                        .push(format!("Failed to update fact '{}': {}", id, e));
                                }
                            }
                            continue;
                        }
                    },
                    Ok(None) => {}
                    Err(e) => {
                        result
                            .errors
                            .push(format!("Failed to check fact '{}': {}", id, e));
                        continue;
                    }
                }

                match create_fact_with_id(&self.db, id, create_data).await {
                    Ok(f) => {
                        result.created += 1;
                        f.id.key().to_string()
                    }
                    Err(e) => {
                        result
                            .errors
                            .push(format!("Failed to create fact '{}': {}", id, e));
                        continue;
                    }
                }
            } else {
                match create_fact(&self.db, create_data).await {
                    Ok(f) => {
                        result.created += 1;
                        f.id.key().to_string()
                    }
                    Err(e) => {
                        result
                            .errors
                            .push(format!("Failed to create fact '{}': {}", spec.title, e));
                        continue;
                    }
                }
            };

            // Link to entities
            for link in &spec.applies_to {
                if let Err(e) = link_fact_to_entity(
                    &self.db,
                    &fact_id,
                    &link.entity_id,
                    &link.link_type,
                    link.confidence,
                )
                .await
                {
                    result.errors.push(format!(
                        "Failed to link fact '{}' to '{}': {}",
                        fact_id, link.entity_id, e
                    ));
                }
            }
        }

        result
    }

    fn spawn_regen(&self, entity_id: &str, entity_type: &str) {
        self.staleness_manager.spawn_regeneration(
            entity_id.to_string(),
            entity_type.to_string(),
            None,
        );
    }
}

fn parse_certainty(s: &str) -> CertaintyLevel {
    match s.to_lowercase().as_str() {
        "knows" | "certain" => CertaintyLevel::Knows,
        "suspects" => CertaintyLevel::Suspects,
        "believes_wrongly" | "wrong" => CertaintyLevel::BelievesWrongly,
        "uncertain" => CertaintyLevel::Uncertain,
        "assumes" => CertaintyLevel::Assumes,
        "denies" => CertaintyLevel::Denies,
        "forgotten" => CertaintyLevel::Forgotten,
        _ => CertaintyLevel::Uncertain,
    }
}

fn parse_learning_method(s: &str) -> LearningMethod {
    match s.to_lowercase().as_str() {
        "told" | "direct" => LearningMethod::Told,
        "overheard" => LearningMethod::Overheard,
        "witnessed" => LearningMethod::Witnessed,
        "discovered" => LearningMethod::Discovered,
        "deduced" => LearningMethod::Deduced,
        "read" => LearningMethod::Read,
        "remembered" => LearningMethod::Remembered,
        "initial" => LearningMethod::Initial,
        _ => LearningMethod::Discovered,
    }
}
