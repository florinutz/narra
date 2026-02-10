use std::sync::Arc;

use serde::Deserialize;
use surrealdb::RecordId;

use crate::db::connection::NarraDb;

use crate::mcp::types::{
    CharacterSpec, EventSpec, FactLinkSpec, FactSpec, KnowledgeSpec, LocationSpec, NarraImport,
    NoteSpec, ParticipantSpec, RelationshipSpec, SceneSpec,
};
use crate::models::character::Character;
use crate::models::event::Event;
use crate::models::fact::{FactApplication, UniverseFact};
use crate::models::location::Location;
use crate::models::note::{Note, NoteAttachment};
use crate::models::relationship::Relationship;
use crate::models::scene::{Scene, SceneParticipant};
use crate::NarraError;

/// Row returned by the knowledge join query.
#[derive(Debug, Deserialize)]
struct KnowledgeExportRow {
    #[serde(rename = "in")]
    character: RecordId,
    #[serde(rename = "out")]
    target: RecordId,
    fact_text: Option<String>,
    certainty: String,
    learning_method: String,
    source_character: Option<RecordId>,
    event: Option<RecordId>,
}

pub struct ExportService {
    db: Arc<NarraDb>,
}

impl ExportService {
    pub fn new(db: Arc<NarraDb>) -> Self {
        Self { db }
    }

    pub async fn export_world(&self) -> Result<NarraImport, NarraError> {
        let characters = self.export_characters().await?;
        let locations = self.export_locations().await?;
        let events = self.export_events().await?;
        let scenes = self.export_scenes().await?;
        let relationships = self.export_relationships().await?;
        let knowledge = self.export_knowledge().await?;
        let notes = self.export_notes().await?;
        let facts = self.export_facts().await?;

        Ok(NarraImport {
            characters,
            locations,
            events,
            scenes,
            relationships,
            knowledge,
            notes,
            facts,
        })
    }

    async fn export_characters(&self) -> Result<Vec<CharacterSpec>, NarraError> {
        let characters: Vec<Character> = self.db.query("SELECT * FROM character").await?.take(0)?;

        Ok(characters
            .into_iter()
            .map(|c| {
                let profile = if c.profile.is_empty() {
                    None
                } else {
                    Some(c.profile)
                };
                CharacterSpec {
                    id: Some(c.id.key().to_string()),
                    name: c.name,
                    role: c.roles.into_iter().next(),
                    aliases: if c.aliases.is_empty() {
                        None
                    } else {
                        Some(c.aliases)
                    },
                    description: None, // Not stored on Character model
                    profile,
                }
            })
            .collect())
    }

    async fn export_locations(&self) -> Result<Vec<LocationSpec>, NarraError> {
        let locations: Vec<Location> = self.db.query("SELECT * FROM location").await?.take(0)?;

        Ok(locations
            .into_iter()
            .map(|l| LocationSpec {
                id: Some(l.id.key().to_string()),
                name: l.name,
                description: l.description,
                parent_id: l.parent.map(|p| p.to_string()),
                loc_type: Some(l.loc_type),
            })
            .collect())
    }

    async fn export_events(&self) -> Result<Vec<EventSpec>, NarraError> {
        let events: Vec<Event> = self
            .db
            .query("SELECT * FROM event ORDER BY sequence ASC")
            .await?
            .take(0)?;

        Ok(events
            .into_iter()
            .map(|e| EventSpec {
                id: Some(e.id.key().to_string()),
                title: e.title,
                description: e.description,
                sequence: Some(e.sequence as i32),
                date: e.date.map(|d| d.to_string()),
                date_precision: e.date_precision,
            })
            .collect())
    }

    async fn export_scenes(&self) -> Result<Vec<SceneSpec>, NarraError> {
        let scenes: Vec<Scene> = self.db.query("SELECT * FROM scene").await?.take(0)?;

        let mut specs = Vec::with_capacity(scenes.len());
        for scene in scenes {
            let scene_key = scene.id.key().to_string();
            let participants = self.get_scene_participants(&scene_key).await?;

            specs.push(SceneSpec {
                id: Some(scene_key),
                title: scene.title,
                event_id: scene.event.to_string(),
                location_id: scene.primary_location.to_string(),
                summary: scene.summary,
                secondary_locations: scene
                    .secondary_locations
                    .into_iter()
                    .map(|r| r.to_string())
                    .collect(),
                participants,
            });
        }

        Ok(specs)
    }

    async fn get_scene_participants(
        &self,
        scene_id: &str,
    ) -> Result<Vec<ParticipantSpec>, NarraError> {
        let participants: Vec<SceneParticipant> =
            crate::models::scene::get_scene_participants(&self.db, scene_id).await?;

        Ok(participants
            .into_iter()
            .map(|p| ParticipantSpec {
                character_id: p.character.to_string(),
                role: p.role,
                notes: p.notes,
            })
            .collect())
    }

    async fn export_relationships(&self) -> Result<Vec<RelationshipSpec>, NarraError> {
        let relationships: Vec<Relationship> =
            self.db.query("SELECT * FROM relates_to").await?.take(0)?;

        Ok(relationships
            .into_iter()
            .map(|r| RelationshipSpec {
                from_character_id: r.from_character.to_string(),
                to_character_id: r.to_character.to_string(),
                rel_type: r.rel_type,
                subtype: r.subtype,
                label: r.label,
            })
            .collect())
    }

    async fn export_knowledge(&self) -> Result<Vec<KnowledgeSpec>, NarraError> {
        let rows: Vec<KnowledgeExportRow> = self
            .db
            .query("SELECT *, out.fact AS fact_text FROM knows WHERE out.fact IS NOT NONE")
            .await?
            .take(0)?;

        Ok(rows
            .into_iter()
            .map(|row| {
                let source_character_id = row.source_character.map(|r| r.to_string());
                let event_id = row.event.map(|r| r.to_string());

                KnowledgeSpec {
                    character_id: row.character.to_string(),
                    target_id: row.target.to_string(),
                    fact: row.fact_text.unwrap_or_default(),
                    certainty: row.certainty,
                    method: Some(row.learning_method),
                    source_character_id,
                    event_id,
                }
            })
            .collect())
    }

    async fn export_notes(&self) -> Result<Vec<NoteSpec>, NarraError> {
        let notes: Vec<Note> = self
            .db
            .query("SELECT * FROM note ORDER BY created_at ASC")
            .await?
            .take(0)?;

        let mut specs = Vec::with_capacity(notes.len());
        for note in notes {
            let note_key = note.id.key().to_string();
            let attachments: Vec<NoteAttachment> =
                crate::models::note::get_note_attachments(&self.db, &note_key).await?;

            let attach_to: Vec<String> = attachments
                .into_iter()
                .map(|a| a.entity.to_string())
                .collect();

            specs.push(NoteSpec {
                id: Some(note_key),
                title: note.title,
                body: note.body,
                attach_to,
            });
        }

        Ok(specs)
    }

    async fn export_facts(&self) -> Result<Vec<FactSpec>, NarraError> {
        let facts: Vec<UniverseFact> = self
            .db
            .query("SELECT * FROM universe_fact ORDER BY created_at ASC")
            .await?
            .take(0)?;

        let mut specs = Vec::with_capacity(facts.len());
        for fact in facts {
            let fact_key = fact.id.key().to_string();
            let applications: Vec<FactApplication> =
                crate::models::fact::get_fact_applications(&self.db, &fact_key).await?;

            let applies_to: Vec<FactLinkSpec> = applications
                .into_iter()
                .map(|a| FactLinkSpec {
                    entity_id: a.entity.to_string(),
                    link_type: a.link_type,
                    confidence: a.confidence,
                })
                .collect();

            let categories: Vec<String> = fact
                .categories
                .iter()
                .map(|c| serde_json::to_value(c).ok())
                .filter_map(|v| v.and_then(|v| v.as_str().map(String::from)))
                .collect();

            let enforcement_level = match fact.enforcement_level {
                crate::models::fact::EnforcementLevel::Informational => {
                    Some("informational".to_string())
                }
                crate::models::fact::EnforcementLevel::Warning => Some("warning".to_string()),
                crate::models::fact::EnforcementLevel::Strict => Some("strict".to_string()),
            };

            specs.push(FactSpec {
                id: Some(fact_key),
                title: fact.title,
                description: fact.description,
                categories,
                enforcement_level,
                applies_to,
            });
        }

        Ok(specs)
    }
}
