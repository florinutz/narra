use crate::mcp::{EntityResult, MutationResponse, NarraServer};

impl NarraServer {
    pub(crate) async fn handle_create_note(
        &self,
        title: String,
        body: String,
        attach_to: Option<Vec<String>>,
    ) -> Result<MutationResponse, String> {
        use crate::models::note::{attach_note, create_note, NoteCreate};

        let create = NoteCreate {
            title: title.clone(),
            body: body.clone(),
        };

        let note = create_note(&self.db, create)
            .await
            .map_err(|e| format!("Failed to create note: {}", e))?;

        let note_key = note.id.key().to_string();

        // Attach to entities if specified
        let mut attached_count = 0;
        if let Some(entity_ids) = attach_to {
            for entity_id in entity_ids {
                attach_note(&self.db, &note_key, &entity_id)
                    .await
                    .map_err(|e| format!("Failed to attach note to {}: {}", entity_id, e))?;
                attached_count += 1;
            }
        }

        let entity_id = note.id.to_string();

        // Trigger embedding generation for the new note
        if let Err(e) = self.staleness_manager.mark_stale(&entity_id).await {
            tracing::warn!("Failed to mark note embedding stale: {}", e);
        }
        self.staleness_manager
            .spawn_regeneration(entity_id.clone(), "note".to_string(), None);

        let result = EntityResult {
            id: entity_id.clone(),
            entity_type: "note".to_string(),
            name: note.title.clone(),
            content: format!("Created note: {}", note.title),
            confidence: Some(1.0),
            last_modified: Some(note.updated_at.to_string()),
        };

        let mut hints = vec![format!("Note '{}' created successfully", title)];
        if attached_count > 0 {
            hints.push(format!("Attached to {} entities", attached_count));
        } else {
            hints.push("Use attach_note to link this note to entities".to_string());
        }

        Ok(MutationResponse {
            entity: result,
            entities: None,
            impact: None,
            hints,
        })
    }

    pub(crate) async fn handle_attach_note(
        &self,
        note_id: String,
        entity_id: String,
    ) -> Result<MutationResponse, String> {
        use crate::models::note::attach_note;

        // Extract note key (handle both "note:xxx" and "xxx" formats)
        let note_key = note_id.split(':').next_back().unwrap_or(&note_id);

        attach_note(&self.db, note_key, &entity_id)
            .await
            .map_err(|e| format!("Failed to attach note: {}", e))?;

        let result = EntityResult {
            id: format!("note:{}", note_key),
            entity_type: "note_attachment".to_string(),
            name: format!("Attachment to {}", entity_id),
            content: format!("Attached note {} to {}", note_key, entity_id),
            confidence: Some(1.0),
            last_modified: Some(chrono::Utc::now().to_rfc3339()),
        };

        let hints = vec![
            format!("Note {} attached to {}", note_key, entity_id),
            "Use list_notes with entity_id to see notes for an entity".to_string(),
        ];

        Ok(MutationResponse {
            entity: result,
            entities: None,
            impact: None,
            hints,
        })
    }

    pub(crate) async fn handle_detach_note(
        &self,
        note_id: String,
        entity_id: String,
    ) -> Result<MutationResponse, String> {
        use crate::models::note::detach_note;

        // Extract note key (handle both "note:xxx" and "xxx" formats)
        let note_key = note_id.split(':').next_back().unwrap_or(&note_id);

        detach_note(&self.db, note_key, &entity_id)
            .await
            .map_err(|e| format!("Failed to detach note: {}", e))?;

        let result = EntityResult {
            id: format!("note:{}", note_key),
            entity_type: "note_attachment".to_string(),
            name: format!("Detached from {}", entity_id),
            content: format!("Detached note {} from {}", note_key, entity_id),
            confidence: Some(1.0),
            last_modified: Some(chrono::Utc::now().to_rfc3339()),
        };

        let hints = vec![format!("Note {} detached from {}", note_key, entity_id)];

        Ok(MutationResponse {
            entity: result,
            entities: None,
            impact: None,
            hints,
        })
    }
}
