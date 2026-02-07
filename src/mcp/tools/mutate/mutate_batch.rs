use crate::mcp::types::{CharacterSpec, EventSpec, LocationSpec, RelationshipSpec};
use crate::mcp::{EntityResult, MutationResponse, NarraServer};
use crate::models::{CharacterCreate, EventCreate, LocationCreate};

impl NarraServer {
    pub(crate) async fn handle_batch_create_characters(
        &self,
        characters: Vec<CharacterSpec>,
    ) -> Result<MutationResponse, String> {
        use crate::models::character::{create_character, create_character_with_id};

        let count = characters.len();
        let mut entities = Vec::with_capacity(count);
        let mut errors: Vec<String> = Vec::new();

        for spec in characters {
            let create = CharacterCreate {
                name: spec.name.clone(),
                aliases: spec.aliases.unwrap_or_default(),
                roles: spec.role.map(|r| vec![r]).unwrap_or_default(),
                profile: spec.profile.unwrap_or_default(),
            };

            let result = if let Some(ref id) = spec.id {
                create_character_with_id(&self.db, id, create).await
            } else {
                create_character(&self.db, create).await
            };

            match result {
                Ok(character) => {
                    let entity_id = character.id.to_string();
                    self.staleness_manager.spawn_regeneration(
                        entity_id.clone(),
                        "character".to_string(),
                        None,
                    );
                    entities.push(EntityResult {
                        id: entity_id,
                        entity_type: "character".to_string(),
                        name: character.name,
                        content: String::new(),
                        confidence: Some(1.0),
                        last_modified: Some(character.updated_at.to_string()),
                    });
                }
                Err(e) => {
                    errors.push(format!("Failed to create '{}': {}", spec.name, e));
                }
            }
        }

        if entities.is_empty() && !errors.is_empty() {
            return Err(format!(
                "All {} characters failed: {}",
                count,
                errors.join("; ")
            ));
        }

        let mut hints = vec![format!("Created {}/{} characters", entities.len(), count)];
        if !errors.is_empty() {
            hints.push(format!("Errors: {}", errors.join("; ")));
        }

        let summary = EntityResult {
            id: String::new(),
            entity_type: "batch".to_string(),
            name: format!("Batch: {} characters", entities.len()),
            content: format!("Created {} characters", entities.len()),
            confidence: Some(1.0),
            last_modified: None,
        };

        Ok(MutationResponse {
            entity: summary,
            entities: Some(entities),
            impact: None,
            hints,
        })
    }

    pub(crate) async fn handle_batch_create_locations(
        &self,
        locations: Vec<LocationSpec>,
    ) -> Result<MutationResponse, String> {
        use crate::models::location::{create_location, create_location_with_id};
        use surrealdb::RecordId;

        let count = locations.len();
        let mut entities = Vec::with_capacity(count);
        let mut errors: Vec<String> = Vec::new();

        for spec in locations {
            let parent_record_id = match spec.parent_id.as_ref() {
                Some(id) => match id.parse::<RecordId>() {
                    Ok(rid) => Some(rid),
                    Err(e) => {
                        errors.push(format!(
                            "Invalid parent_id '{}' for '{}': {}",
                            id, spec.name, e
                        ));
                        continue;
                    }
                },
                None => None,
            };

            let create = LocationCreate {
                name: spec.name.clone(),
                description: spec.description,
                loc_type: "place".to_string(),
                parent: parent_record_id,
            };

            let result = if let Some(ref id) = spec.id {
                create_location_with_id(&self.db, id, create).await
            } else {
                create_location(&self.db, create).await
            };

            match result {
                Ok(location) => {
                    let entity_id = location.id.to_string();
                    self.staleness_manager.spawn_regeneration(
                        entity_id.clone(),
                        "location".to_string(),
                        None,
                    );
                    entities.push(EntityResult {
                        id: entity_id,
                        entity_type: "location".to_string(),
                        name: location.name,
                        content: String::new(),
                        confidence: Some(1.0),
                        last_modified: Some(location.updated_at.to_string()),
                    });
                }
                Err(e) => {
                    errors.push(format!("Failed to create '{}': {}", spec.name, e));
                }
            }
        }

        if entities.is_empty() && !errors.is_empty() {
            return Err(format!(
                "All {} locations failed: {}",
                count,
                errors.join("; ")
            ));
        }

        let mut hints = vec![format!("Created {}/{} locations", entities.len(), count)];
        if !errors.is_empty() {
            hints.push(format!("Errors: {}", errors.join("; ")));
        }

        let summary = EntityResult {
            id: String::new(),
            entity_type: "batch".to_string(),
            name: format!("Batch: {} locations", entities.len()),
            content: format!("Created {} locations", entities.len()),
            confidence: Some(1.0),
            last_modified: None,
        };

        Ok(MutationResponse {
            entity: summary,
            entities: Some(entities),
            impact: None,
            hints,
        })
    }

    pub(crate) async fn handle_batch_create_events(
        &self,
        events: Vec<EventSpec>,
    ) -> Result<MutationResponse, String> {
        use crate::models::event::{create_event, create_event_with_id};

        let count = events.len();
        let mut entities = Vec::with_capacity(count);
        let mut errors: Vec<String> = Vec::new();

        for spec in events {
            let parsed_date = match spec.date.as_ref() {
                Some(d) => match chrono::DateTime::parse_from_rfc3339(d) {
                    Ok(dt) => Some(dt.with_timezone(&chrono::Utc).into()),
                    Err(e) => {
                        errors.push(format!("Invalid date '{}' for '{}': {}", d, spec.title, e));
                        continue;
                    }
                },
                None => None,
            };

            let create = EventCreate {
                title: spec.title.clone(),
                description: spec.description,
                sequence: spec.sequence.unwrap_or(0) as i64,
                date: parsed_date,
                date_precision: spec.date_precision,
                duration_end: None,
            };

            let result = if let Some(ref id) = spec.id {
                create_event_with_id(&self.db, id, create).await
            } else {
                create_event(&self.db, create).await
            };

            match result {
                Ok(event) => {
                    let entity_id = event.id.to_string();
                    self.staleness_manager.spawn_regeneration(
                        entity_id.clone(),
                        "event".to_string(),
                        None,
                    );
                    entities.push(EntityResult {
                        id: entity_id,
                        entity_type: "event".to_string(),
                        name: event.title,
                        content: String::new(),
                        confidence: Some(1.0),
                        last_modified: Some(event.updated_at.to_string()),
                    });
                }
                Err(e) => {
                    errors.push(format!("Failed to create '{}': {}", spec.title, e));
                }
            }
        }

        if entities.is_empty() && !errors.is_empty() {
            return Err(format!(
                "All {} events failed: {}",
                count,
                errors.join("; ")
            ));
        }

        let mut hints = vec![format!("Created {}/{} events", entities.len(), count)];
        if !errors.is_empty() {
            hints.push(format!("Errors: {}", errors.join("; ")));
        }

        let summary = EntityResult {
            id: String::new(),
            entity_type: "batch".to_string(),
            name: format!("Batch: {} events", entities.len()),
            content: format!("Created {} events", entities.len()),
            confidence: Some(1.0),
            last_modified: None,
        };

        Ok(MutationResponse {
            entity: summary,
            entities: Some(entities),
            impact: None,
            hints,
        })
    }

    pub(crate) async fn handle_batch_create_relationships(
        &self,
        relationships: Vec<RelationshipSpec>,
    ) -> Result<MutationResponse, String> {
        use crate::models::relationship::{create_relationship, RelationshipCreate};

        let count = relationships.len();
        let mut entities = Vec::with_capacity(count);
        let mut errors: Vec<String> = Vec::new();

        for spec in relationships {
            let create = RelationshipCreate {
                from_character_id: spec.from_character_id.clone(),
                to_character_id: spec.to_character_id.clone(),
                rel_type: spec.rel_type.clone(),
                subtype: spec.subtype,
                label: spec.label,
            };

            match create_relationship(&self.db, create).await {
                Ok(rel) => {
                    let entity_id = rel.id.to_string();
                    // Mark relates_to edge for embedding generation
                    self.staleness_manager.spawn_regeneration(
                        entity_id.clone(),
                        "relates_to".to_string(),
                        None,
                    );
                    entities.push(EntityResult {
                        id: entity_id,
                        entity_type: "relationship".to_string(),
                        name: format!(
                            "{} -> {} ({})",
                            spec.from_character_id, spec.to_character_id, spec.rel_type
                        ),
                        content: String::new(),
                        confidence: Some(1.0),
                        last_modified: Some(rel.created_at.to_string()),
                    });
                }
                Err(e) => {
                    errors.push(format!(
                        "Failed to create {} -> {}: {}",
                        spec.from_character_id, spec.to_character_id, e
                    ));
                }
            }
        }

        if entities.is_empty() && !errors.is_empty() {
            return Err(format!(
                "All {} relationships failed: {}",
                count,
                errors.join("; ")
            ));
        }

        let mut hints = vec![format!(
            "Created {}/{} relationships",
            entities.len(),
            count
        )];
        if !errors.is_empty() {
            hints.push(format!("Errors: {}", errors.join("; ")));
        }

        let summary = EntityResult {
            id: String::new(),
            entity_type: "batch".to_string(),
            name: format!("Batch: {} relationships", entities.len()),
            content: format!("Created {} relationships", entities.len()),
            confidence: Some(1.0),
            last_modified: None,
        };

        Ok(MutationResponse {
            entity: summary,
            entities: Some(entities),
            impact: None,
            hints,
        })
    }
}
