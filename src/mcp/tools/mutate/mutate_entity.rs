use crate::mcp::{EntityResult, MutationResponse, NarraServer};
use crate::models::{CharacterCreate, EventCreate, LocationCreate, SceneCreate};

impl NarraServer {
    pub(crate) async fn handle_create_character(
        &self,
        id: Option<String>,
        name: String,
        role: Option<String>,
        aliases: Option<Vec<String>>,
        description: Option<String>,
        profile: Option<std::collections::HashMap<String, Vec<String>>>,
    ) -> Result<MutationResponse, String> {
        use crate::models::character::{create_character, create_character_with_id};

        // Build creation data for consistency check
        let creation_data = serde_json::json!({
            "name": name,
            "role": role,
            "aliases": aliases,
            "description": description,
        });

        // Check consistency BEFORE creation
        let consistency_result = self
            .consistency_service
            .check_entity_creation("character", &creation_data)
            .await
            .map_err(|e| format!("Consistency check failed: {}", e))?;

        let consistency_warnings = self.process_consistency_result(
            &consistency_result,
            &format!("Create character {}", name),
        )?;

        let create = CharacterCreate {
            name: name.clone(),
            aliases: aliases.unwrap_or_default(),
            roles: role.map(|r| vec![r]).unwrap_or_default(),
            profile: profile.unwrap_or_default(),
        };

        let character = if let Some(ref slug) = id {
            create_character_with_id(&self.db, slug, create).await
        } else {
            create_character(&self.db, create).await
        }
        .map_err(|e| format!("Failed to create character: {}", e))?;

        let entity_id = character.id.to_string();

        // Trigger async embedding generation for new entity
        self.staleness_manager
            .spawn_regeneration(entity_id.clone(), "character".to_string(), None);

        let result = EntityResult {
            id: entity_id.clone(),
            entity_type: "character".to_string(),
            name: character.name.clone(),
            content: format!("Created character: {}", character.name),
            confidence: Some(1.0),
            last_modified: Some(character.updated_at.to_string()),
        };

        // Add consistency warnings to hints
        let mut hints = vec![
            format!("Character '{}' created successfully", name),
            "Add relationships with other characters using record_knowledge or graph mutations"
                .to_string(),
        ];
        hints.extend(consistency_warnings);

        Ok(MutationResponse {
            entity: result,
            entities: None,
            impact: None, // No impact for new entity
            hints,
        })
    }

    pub(crate) async fn handle_create_location(
        &self,
        id: Option<String>,
        name: String,
        description: Option<String>,
        parent_id: Option<String>,
    ) -> Result<MutationResponse, String> {
        use crate::models::location::{create_location, create_location_with_id};
        use surrealdb::RecordId;

        // Build creation data for consistency check
        let creation_data = serde_json::json!({
            "name": name,
            "description": description,
            "parent_id": parent_id,
        });

        // Check consistency BEFORE creation
        let consistency_result = self
            .consistency_service
            .check_entity_creation("location", &creation_data)
            .await
            .map_err(|e| format!("Consistency check failed: {}", e))?;

        let consistency_warnings = self.process_consistency_result(
            &consistency_result,
            &format!("Create location {}", name),
        )?;

        let parent_record_id = parent_id
            .as_ref()
            .map(|id| id.parse::<RecordId>())
            .transpose()
            .map_err(|e| format!("Invalid parent_id: {}", e))?;

        let create = LocationCreate {
            name: name.clone(),
            description,
            loc_type: "place".to_string(), // Default location type
            parent: parent_record_id,
        };

        let location = if let Some(ref slug) = id {
            create_location_with_id(&self.db, slug, create).await
        } else {
            create_location(&self.db, create).await
        }
        .map_err(|e| format!("Failed to create location: {}", e))?;

        let entity_id = location.id.to_string();

        // Trigger async embedding generation for new entity
        self.staleness_manager
            .spawn_regeneration(entity_id.clone(), "location".to_string(), None);

        let result = EntityResult {
            id: entity_id,
            entity_type: "location".to_string(),
            name: location.name.clone(),
            content: format!("Created location: {}", location.name),
            confidence: Some(1.0),
            last_modified: Some(location.updated_at.to_string()),
        };

        // Add consistency warnings to hints
        let mut hints = vec![
            format!("Location '{}' created successfully", name),
            "Associate scenes with this location for narrative events".to_string(),
        ];
        hints.extend(consistency_warnings);

        Ok(MutationResponse {
            entity: result,
            entities: None,
            impact: None,
            hints,
        })
    }

    pub(crate) async fn handle_create_event(
        &self,
        id: Option<String>,
        title: String,
        description: Option<String>,
        sequence: Option<i32>,
        date: Option<String>,
        date_precision: Option<String>,
    ) -> Result<MutationResponse, String> {
        use crate::models::event::{create_event, create_event_with_id};

        // Build creation data for consistency check
        let creation_data = serde_json::json!({
            "title": title,
            "description": description,
            "sequence": sequence,
            "date": date,
            "date_precision": date_precision,
        });

        // Check consistency BEFORE creation
        let consistency_result = self
            .consistency_service
            .check_entity_creation("event", &creation_data)
            .await
            .map_err(|e| format!("Consistency check failed: {}", e))?;

        let consistency_warnings = self
            .process_consistency_result(&consistency_result, &format!("Create event {}", title))?;

        // Parse date if provided
        let parsed_date = date
            .as_ref()
            .map(|d| {
                chrono::DateTime::parse_from_rfc3339(d)
                    .map(|dt| dt.with_timezone(&chrono::Utc).into())
            })
            .transpose()
            .map_err(|e| format!("Invalid date format: {}", e))?;

        let create = EventCreate {
            title: title.clone(),
            description,
            sequence: sequence.unwrap_or(0) as i64,
            date: parsed_date,
            date_precision,
            duration_end: None,
        };

        let event = if let Some(ref slug) = id {
            create_event_with_id(&self.db, slug, create).await
        } else {
            create_event(&self.db, create).await
        }
        .map_err(|e| format!("Failed to create event: {}", e))?;

        let entity_id = event.id.to_string();

        // Trigger async embedding generation for new entity
        self.staleness_manager
            .spawn_regeneration(entity_id.clone(), "event".to_string(), None);

        let result = EntityResult {
            id: entity_id,
            entity_type: "event".to_string(),
            name: event.title.clone(),
            content: format!("Created event: {}", event.title),
            confidence: Some(1.0),
            last_modified: Some(event.updated_at.to_string()),
        };

        // Add consistency warnings to hints
        let mut hints = vec![
            format!("Event '{}' created successfully", title),
            "Add character involvement to track participation".to_string(),
        ];
        hints.extend(consistency_warnings);

        Ok(MutationResponse {
            entity: result,
            entities: None,
            impact: None,
            hints,
        })
    }

    pub(crate) async fn handle_create_scene(
        &self,
        title: String,
        summary: Option<String>,
        event_id: String,
        location_id: String,
    ) -> Result<MutationResponse, String> {
        use crate::models::scene::create_scene;
        use surrealdb::RecordId;

        // Build creation data for consistency check
        let creation_data = serde_json::json!({
            "title": title,
            "summary": summary,
            "event_id": event_id,
            "location_id": location_id,
        });

        // Check consistency BEFORE creation
        let consistency_result = self
            .consistency_service
            .check_entity_creation("scene", &creation_data)
            .await
            .map_err(|e| format!("Consistency check failed: {}", e))?;

        let consistency_warnings = self
            .process_consistency_result(&consistency_result, &format!("Create scene {}", title))?;

        let event_record_id = event_id
            .parse::<RecordId>()
            .map_err(|e| format!("Invalid event_id: {}", e))?;

        let location_record_id = location_id
            .parse::<RecordId>()
            .map_err(|e| format!("Invalid location_id: {}", e))?;

        let create = SceneCreate {
            title: title.clone(),
            summary,
            event: event_record_id,
            primary_location: location_record_id,
            secondary_locations: vec![],
        };

        let scene = create_scene(&self.db, create)
            .await
            .map_err(|e| format!("Failed to create scene: {}", e))?;

        let entity_id = scene.id.to_string();

        // Trigger async embedding generation for new entity
        self.staleness_manager
            .spawn_regeneration(entity_id.clone(), "scene".to_string(), None);

        let result = EntityResult {
            id: entity_id,
            entity_type: "scene".to_string(),
            name: scene.title.clone(),
            content: format!("Created scene: {}", scene.title),
            confidence: Some(1.0),
            last_modified: Some(scene.updated_at.to_string()),
        };

        // Add consistency warnings to hints
        let mut hints = vec![
            format!("Scene '{}' created successfully", title),
            "Add participants to track character involvement".to_string(),
        ];
        hints.extend(consistency_warnings);

        Ok(MutationResponse {
            entity: result,
            entities: None,
            impact: None,
            hints,
        })
    }

    pub(crate) async fn handle_update(
        &self,
        entity_id: &str,
        fields: serde_json::Value,
    ) -> Result<MutationResponse, String> {
        // Detect entity type from ID format (table:id)
        let entity_type = self.detect_entity_type(entity_id);

        // Check consistency and analyze impact in parallel (both are read-only pre-mutation checks)
        let (consistency_result, impact_analysis) = tokio::try_join!(
            async {
                self.consistency_service
                    .check_entity_mutation(entity_id, &fields)
                    .await
                    .map_err(|e| format!("Consistency check failed: {}", e))
            },
            async {
                self.impact_service
                    .analyze_impact(entity_id, "update", 3)
                    .await
                    .map_err(|e| format!("Impact analysis failed: {}", e))
            },
        )?;

        let consistency_warnings =
            self.process_consistency_result(&consistency_result, &format!("Update {}", entity_id))?;

        // Perform update based on entity type
        // Note: Update structs don't derive Deserialize, so we manually extract fields
        match entity_type.as_str() {
            "character" => {
                use crate::models::character::{update_character, CharacterUpdate};

                let update = CharacterUpdate {
                    name: fields
                        .get("name")
                        .and_then(|v| v.as_str())
                        .map(String::from),
                    aliases: fields
                        .get("aliases")
                        .and_then(|v| serde_json::from_value(v.clone()).ok()),
                    roles: fields
                        .get("roles")
                        .and_then(|v| serde_json::from_value(v.clone()).ok()),
                    profile: fields
                        .get("profile")
                        .and_then(|v| serde_json::from_value(v.clone()).ok()),
                    updated_at: chrono::Utc::now().into(),
                };

                update_character(&self.db, entity_id, update)
                    .await
                    .map_err(|e| format!("Failed to update character: {}", e))?;
            }
            "location" => {
                use crate::models::location::{update_location, LocationUpdate};

                let update = LocationUpdate {
                    name: fields
                        .get("name")
                        .and_then(|v| v.as_str())
                        .map(String::from),
                    description: fields
                        .get("description")
                        .map(|v| v.as_str().map(String::from)),
                    loc_type: fields
                        .get("loc_type")
                        .and_then(|v| v.as_str().map(String::from)),
                    parent: fields
                        .get("parent")
                        .map(|v| v.as_str().and_then(|s| s.parse().ok())),
                    updated_at: chrono::Utc::now().into(),
                };

                update_location(&self.db, entity_id, update)
                    .await
                    .map_err(|e| format!("Failed to update location: {}", e))?;
            }
            "event" => {
                use crate::models::event::{update_event, EventUpdate};

                let update = EventUpdate {
                    title: fields
                        .get("title")
                        .and_then(|v| v.as_str())
                        .map(String::from),
                    description: fields
                        .get("description")
                        .map(|v| v.as_str().map(String::from)),
                    sequence: fields.get("sequence").and_then(|v| v.as_i64()),
                    date: fields.get("date").map(|v| {
                        v.as_str().and_then(|s| {
                            chrono::DateTime::parse_from_rfc3339(s)
                                .ok()
                                .map(|dt| dt.with_timezone(&chrono::Utc).into())
                        })
                    }),
                    date_precision: fields
                        .get("date_precision")
                        .map(|v| v.as_str().map(String::from)),
                    duration_end: fields.get("duration_end").map(|v| {
                        v.as_str().and_then(|s| {
                            chrono::DateTime::parse_from_rfc3339(s)
                                .ok()
                                .map(|dt| dt.with_timezone(&chrono::Utc).into())
                        })
                    }),
                    updated_at: chrono::Utc::now().into(),
                };

                update_event(&self.db, entity_id, update)
                    .await
                    .map_err(|e| format!("Failed to update event: {}", e))?;
            }
            "scene" => {
                use crate::models::scene::{update_scene, SceneUpdate};

                let update = SceneUpdate {
                    title: fields
                        .get("title")
                        .and_then(|v| v.as_str())
                        .map(String::from),
                    summary: fields.get("summary").map(|v| v.as_str().map(String::from)),
                    event: fields
                        .get("event")
                        .and_then(|v| v.as_str().and_then(|s| s.parse().ok())),
                    primary_location: fields
                        .get("primary_location")
                        .and_then(|v| v.as_str().and_then(|s| s.parse().ok())),
                    secondary_locations: fields
                        .get("secondary_locations")
                        .and_then(|v| serde_json::from_value(v.clone()).ok()),
                    updated_at: chrono::Utc::now().into(),
                };

                update_scene(&self.db, entity_id, update)
                    .await
                    .map_err(|e| format!("Failed to update scene: {}", e))?;
            }
            _ => return Err(format!("Unknown entity type: {}", entity_type)),
        }

        // Invalidate summary cache for this entity
        self.summary_service.invalidate(entity_id).await;

        // Mark embedding stale and trigger regeneration for embeddable entity types
        if matches!(
            entity_type.as_str(),
            "character" | "location" | "event" | "scene" | "note" | "fact"
        ) {
            if let Err(e) = self.staleness_manager.mark_stale(entity_id).await {
                tracing::warn!("Failed to mark embedding stale for {}: {}", entity_id, e);
            }
            self.staleness_manager.spawn_regeneration(
                entity_id.to_string(),
                entity_type.clone(),
                None,
            );
        }

        // Mark annotations stale (emotion, etc.) so they are recomputed on next access
        if let Err(e) = self
            .staleness_manager
            .mark_annotations_stale(entity_id)
            .await
        {
            tracing::warn!("Failed to mark annotations stale for {}: {}", entity_id, e);
        }

        // If character name or roles changed, mark their relates_to edges stale
        if entity_type == "character"
            && (fields.get("name").is_some() || fields.get("roles").is_some())
        {
            let stale_query = format!(
                "UPDATE relates_to SET embedding_stale = true WHERE in = {} OR out = {}",
                entity_id, entity_id
            );
            if let Err(e) = self.db.query(&stale_query).await {
                tracing::warn!(
                    "Failed to mark relates_to edges stale for {}: {}",
                    entity_id,
                    e
                );
            }
        }

        // If updating a perceives edge, regenerate its perspective embedding
        if entity_type == "perceives" {
            if let Err(e) = self.staleness_manager.mark_stale(entity_id).await {
                tracing::warn!(
                    "Failed to mark perspective embedding stale for {}: {}",
                    entity_id,
                    e
                );
            }
            self.staleness_manager.spawn_regeneration(
                entity_id.to_string(),
                "perceives".to_string(),
                None,
            );
        }

        let result = EntityResult {
            id: entity_id.to_string(),
            entity_type: entity_type.clone(),
            name: self.extract_name_from_id(entity_id),
            content: format!("Updated {} successfully", entity_type),
            confidence: Some(1.0),
            last_modified: Some(chrono::Utc::now().to_rfc3339()),
        };

        // Include impact summary for high-severity changes
        let impact = if impact_analysis.has_protected_impact || impact_analysis.total_affected > 5 {
            Some(self.convert_impact(&impact_analysis))
        } else {
            None
        };

        let mut hints = vec![format!("{} updated", entity_type)];
        hints.extend(consistency_warnings);
        if impact_analysis.total_affected > 0 {
            hints.push(format!(
                "{} related entities may be affected",
                impact_analysis.total_affected
            ));
        }
        if !impact_analysis.warnings.is_empty() {
            hints.extend(impact_analysis.warnings.clone());
        }

        Ok(MutationResponse {
            entity: result,
            entities: None,
            impact,
            hints,
        })
    }

    pub(crate) async fn handle_delete(
        &self,
        entity_id: &str,
        hard: bool,
    ) -> Result<MutationResponse, String> {
        let entity_type = self.detect_entity_type(entity_id);

        // Analyze impact BEFORE deletion
        let impact_analysis = self
            .impact_service
            .analyze_impact(entity_id, "delete", 3)
            .await
            .map_err(|e| format!("Impact analysis failed: {}", e))?;

        // Warn if high-severity deletion (protected entities affected)
        if impact_analysis.has_protected_impact && !hard {
            let warnings: Vec<String> = impact_analysis.warnings.clone();
            return Err(format!(
                "CRITICAL: Deletion would affect protected entities. Warnings: {}. Use hard=true to force.",
                warnings.join("; ")
            ));
        }

        if hard {
            // Extract key from entity_id (format: "table:key")
            let entity_key = entity_id.split(':').nth(1).unwrap_or(entity_id);

            // Hard delete - remove from database
            match entity_type.as_str() {
                "character" => {
                    use crate::models::character::delete_character;
                    delete_character(&self.db, entity_key)
                        .await
                        .map_err(|e| format!("Failed to delete character: {}", e))?;
                }
                "location" => {
                    use crate::models::location::delete_location;
                    delete_location(&self.db, entity_key)
                        .await
                        .map_err(|e| format!("Failed to delete location: {}", e))?;
                }
                "event" => {
                    use crate::models::event::delete_event;
                    delete_event(&self.db, entity_key)
                        .await
                        .map_err(|e| format!("Failed to delete event: {}", e))?;
                }
                "scene" => {
                    use crate::models::scene::delete_scene;
                    delete_scene(&self.db, entity_key)
                        .await
                        .map_err(|e| format!("Failed to delete scene: {}", e))?;
                }
                _ => return Err(format!("Unknown entity type: {}", entity_type)),
            }
        } else {
            // Soft delete not implemented
            return Err(
                "Soft delete not yet implemented. Use hard=true to permanently delete.".to_string(),
            );
        }

        // Invalidate cache
        self.summary_service.invalidate(entity_id).await;

        // Mark related entities stale (their composite text may reference the deleted entity)
        if let Err(e) = self.staleness_manager.mark_related_stale(entity_id).await {
            tracing::warn!(
                "Failed to mark related embeddings stale for {}: {}",
                entity_id,
                e
            );
        }

        // Delete annotations for the deleted entity
        if let Err(e) =
            crate::models::annotation::delete_entity_annotations(&self.db, entity_id).await
        {
            tracing::warn!("Failed to delete annotations for {}: {}", entity_id, e);
        }

        let result = EntityResult {
            id: entity_id.to_string(),
            entity_type: entity_type.clone(),
            name: self.extract_name_from_id(entity_id),
            content: format!("Deleted {} (hard={})", entity_type, hard),
            confidence: Some(1.0),
            last_modified: Some(chrono::Utc::now().to_rfc3339()),
        };

        let impact = Some(self.convert_impact(&impact_analysis));

        let hints = vec![
            format!("{} deleted", entity_type),
            format!(
                "{} related entities affected",
                impact_analysis.total_affected
            ),
        ];

        Ok(MutationResponse {
            entity: result,
            entities: None,
            impact,
            hints,
        })
    }
}
