//! Backfill embedding generation for existing entities.
//!
//! Generates embeddings for all entities that are missing them or have stale embeddings.

use std::collections::HashMap;
use std::sync::Arc;

use serde::Serialize;
use surrealdb::engine::local::Db;
use surrealdb::Surreal;
use tracing::info;

use crate::embedding::composite::{
    character_composite, event_composite, knowledge_composite, location_composite,
    perspective_composite, relationship_composite, scene_composite,
};
use crate::embedding::EmbeddingService;
use crate::models::{Character, Event, Location, Scene};
use crate::NarraError;

/// Statistics from a backfill operation.
#[derive(Debug, Clone, Default, Serialize)]
pub struct BackfillStats {
    pub total_entities: usize,
    pub embedded: usize,
    pub skipped: usize, // Already have valid embeddings
    pub failed: usize,
    pub entity_type_stats: HashMap<String, usize>,
}

/// Service for backfilling embeddings across all entities.
pub struct BackfillService {
    db: Arc<Surreal<Db>>,
    embedding_service: Arc<dyn EmbeddingService + Send + Sync>,
}

impl BackfillService {
    /// Create a new backfill service.
    pub fn new(
        db: Arc<Surreal<Db>>,
        embedding_service: Arc<dyn EmbeddingService + Send + Sync>,
    ) -> Self {
        Self {
            db,
            embedding_service,
        }
    }

    /// Embed a batch of texts and update their entities in the database.
    ///
    /// Handles the repeated pattern of: embed_batch → UPDATE each entity with embedding + composite_text.
    async fn embed_and_update_batch(
        &self,
        entity_ids: &[String],
        texts: &[String],
        entity_type: &str,
        stats: &mut BackfillStats,
    ) -> Result<(), NarraError> {
        match self.embedding_service.embed_batch(texts).await {
            Ok(embeddings) => {
                for (i, entity_id) in entity_ids.iter().enumerate() {
                    let update_query = format!(
                        "UPDATE {} SET embedding = $embedding, embedding_stale = false, composite_text = $composite_text",
                        entity_id
                    );

                    match self
                        .db
                        .query(&update_query)
                        .bind(("embedding", embeddings[i].clone()))
                        .bind(("composite_text", texts[i].clone()))
                        .await
                    {
                        Ok(_) => {
                            stats.embedded += 1;
                        }
                        Err(e) => {
                            tracing::warn!("Failed to update embedding for {}: {}", entity_id, e);
                            stats.failed += 1;
                        }
                    }
                }

                if stats.embedded.is_multiple_of(50) && stats.embedded > 0 {
                    info!(
                        "Backfilled {} {} entities so far",
                        stats.embedded, entity_type
                    );
                }
            }
            Err(e) => {
                tracing::warn!(
                    "Failed to generate embeddings for {} batch: {}",
                    entity_type,
                    e
                );
                stats.failed += entity_ids.len();
            }
        }
        Ok(())
    }

    /// Embed a single text and update its entity in the database.
    ///
    /// Handles the repeated pattern of: embed_text → UPDATE entity with embedding + composite_text.
    async fn embed_and_update_single(
        &self,
        entity_id: &str,
        text: &str,
        entity_type: &str,
        stats: &mut BackfillStats,
    ) -> Result<(), NarraError> {
        match self.embedding_service.embed_text(text).await {
            Ok(embedding) => {
                let update_query = format!(
                    "UPDATE {} SET embedding = $embedding, embedding_stale = false, composite_text = $composite_text",
                    entity_id
                );

                match self
                    .db
                    .query(&update_query)
                    .bind(("embedding", embedding))
                    .bind(("composite_text", text.to_string()))
                    .await
                {
                    Ok(_) => {
                        stats.embedded += 1;
                    }
                    Err(e) => {
                        tracing::warn!("Failed to update embedding for {}: {}", entity_id, e);
                        stats.failed += 1;
                    }
                }
            }
            Err(e) => {
                tracing::warn!(
                    "Failed to generate embedding for {} {}: {}",
                    entity_type,
                    entity_id,
                    e
                );
                stats.failed += 1;
            }
        }
        Ok(())
    }

    /// Backfill embeddings for all entity types.
    ///
    /// Processes characters, locations, events, and scenes.
    /// Skips entities that already have valid embeddings.
    ///
    /// # Returns
    ///
    /// BackfillStats with counts of processed, embedded, skipped, and failed entities.
    pub async fn backfill_all(&self) -> Result<BackfillStats, NarraError> {
        // Check embedding service is available
        if !self.embedding_service.is_available() {
            return Err(NarraError::Database(
                "Embedding service not available - cannot backfill".to_string(),
            ));
        }

        info!("Starting backfill for all entity types");

        let mut stats = BackfillStats::default();

        // Backfill each entity type
        for entity_type in &[
            "character",
            "location",
            "event",
            "scene",
            "knowledge",
            "perspective",
            "relationship",
        ] {
            let type_stats = self.backfill_type(entity_type).await?;
            stats.total_entities += type_stats.total_entities;
            stats.embedded += type_stats.embedded;
            stats.skipped += type_stats.skipped;
            stats.failed += type_stats.failed;
            stats
                .entity_type_stats
                .insert(entity_type.to_string(), type_stats.embedded);
        }

        info!(
            "Backfill complete: {} total, {} embedded, {} skipped, {} failed",
            stats.total_entities, stats.embedded, stats.skipped, stats.failed
        );

        Ok(stats)
    }

    /// Backfill embeddings for a single entity type.
    ///
    /// # Arguments
    ///
    /// * `entity_type` - One of: "character", "location", "event", "scene"
    ///
    /// # Returns
    ///
    /// BackfillStats for this entity type.
    pub async fn backfill_type(&self, entity_type: &str) -> Result<BackfillStats, NarraError> {
        // Check embedding service is available
        if !self.embedding_service.is_available() {
            return Err(NarraError::Database(
                "Embedding service not available - cannot backfill".to_string(),
            ));
        }

        info!("Backfilling {} entities", entity_type);

        let mut stats = BackfillStats::default();

        match entity_type {
            "character" => self.backfill_characters(&mut stats).await?,
            "location" => self.backfill_locations(&mut stats).await?,
            "event" => self.backfill_events(&mut stats).await?,
            "scene" => self.backfill_scenes(&mut stats).await?,
            "knowledge" => self.backfill_knowledge(&mut stats).await?,
            "perspective" => self.backfill_perspectives(&mut stats).await?,
            "relationship" => self.backfill_relationships(&mut stats).await?,
            _ => {
                return Err(NarraError::Database(format!(
                    "Unknown entity type for backfill: {}",
                    entity_type
                )))
            }
        }

        info!(
            "Backfilled {}: {} embedded, {} skipped, {} failed",
            entity_type, stats.embedded, stats.skipped, stats.failed
        );

        Ok(stats)
    }

    /// Backfill character embeddings.
    async fn backfill_characters(&self, stats: &mut BackfillStats) -> Result<(), NarraError> {
        let query = "SELECT * FROM character WHERE embedding IS NONE OR embedding_stale = true";
        let mut response = self.db.query(query).await?;
        let characters: Vec<Character> = response.take(0)?;

        stats.total_entities += characters.len();

        // Bulk pre-fetch all relationships and perceptions to avoid N+1
        let all_relationships = self.get_all_character_relationships().await?;
        let all_perceptions = self.get_all_character_perceptions().await?;

        for chunk in characters.chunks(50) {
            let mut texts = Vec::new();
            let mut ids = Vec::new();

            for character in chunk {
                let char_id = character.id.to_string();
                let relationships = all_relationships.get(&char_id).cloned().unwrap_or_default();
                let perceptions = all_perceptions.get(&char_id).cloned().unwrap_or_default();

                texts.push(character_composite(character, &relationships, &perceptions));
                ids.push(char_id);
            }

            self.embed_and_update_batch(&ids, &texts, "character", stats)
                .await?;
        }

        Ok(())
    }

    /// Backfill location embeddings.
    async fn backfill_locations(&self, stats: &mut BackfillStats) -> Result<(), NarraError> {
        let query = "SELECT * FROM location WHERE embedding IS NONE OR embedding_stale = true";
        let mut response = self.db.query(query).await?;
        let locations: Vec<Location> = response.take(0)?;

        stats.total_entities += locations.len();

        for chunk in locations.chunks(50) {
            let ids: Vec<String> = chunk.iter().map(|l| l.id.to_string()).collect();
            let texts: Vec<String> = chunk.iter().map(location_composite).collect();

            self.embed_and_update_batch(&ids, &texts, "location", stats)
                .await?;
        }

        Ok(())
    }

    /// Backfill event embeddings.
    async fn backfill_events(&self, stats: &mut BackfillStats) -> Result<(), NarraError> {
        let query = "SELECT * FROM event WHERE embedding IS NONE OR embedding_stale = true";
        let mut response = self.db.query(query).await?;
        let events: Vec<Event> = response.take(0)?;

        stats.total_entities += events.len();

        for chunk in events.chunks(50) {
            let ids: Vec<String> = chunk.iter().map(|e| e.id.to_string()).collect();
            let texts: Vec<String> = chunk.iter().map(event_composite).collect();

            self.embed_and_update_batch(&ids, &texts, "event", stats)
                .await?;
        }

        Ok(())
    }

    /// Backfill scene embeddings.
    async fn backfill_scenes(&self, stats: &mut BackfillStats) -> Result<(), NarraError> {
        // Query only the fields needed for composite text — avoid #[serde(flatten)]
        // which fails with SurrealDB's RecordId serialization.
        let query = r#"SELECT id, title, summary,
                              event.title AS event_title,
                              primary_location.name AS location_name
                       FROM scene
                       WHERE embedding IS NONE OR embedding_stale = true"#;
        let mut response = self.db.query(query).await?;

        #[derive(serde::Deserialize)]
        struct SceneForComposite {
            id: surrealdb::RecordId,
            title: String,
            summary: Option<String>,
            event_title: Option<String>,
            location_name: Option<String>,
        }

        let scenes: Vec<SceneForComposite> = response.take(0)?;

        stats.total_entities += scenes.len();

        for chunk in scenes.chunks(50) {
            let mut texts = Vec::new();
            let mut ids = Vec::new();

            for scene_ctx in chunk {
                let scene = Scene {
                    id: scene_ctx.id.clone(),
                    title: scene_ctx.title.clone(),
                    summary: scene_ctx.summary.clone(),
                    event: surrealdb::RecordId::from(("event", "placeholder")),
                    primary_location: surrealdb::RecordId::from(("location", "placeholder")),
                    secondary_locations: vec![],
                    created_at: surrealdb::Datetime::default(),
                    updated_at: surrealdb::Datetime::default(),
                };
                texts.push(scene_composite(
                    &scene,
                    scene_ctx.event_title.as_deref(),
                    scene_ctx.location_name.as_deref(),
                ));
                ids.push(scene_ctx.id.to_string());
            }

            self.embed_and_update_batch(&ids, &texts, "scene", stats)
                .await?;
        }

        Ok(())
    }

    /// Backfill knowledge embeddings.
    async fn backfill_knowledge(&self, stats: &mut BackfillStats) -> Result<(), NarraError> {
        // Query only the fields needed — avoid SELECT * which includes RecordId
        // fields that fail SurrealDB's serde deserialization.
        let query = r#"SELECT id, fact, character.name AS character_name
                       FROM knowledge
                       WHERE embedding IS NONE OR embedding_stale = true"#;
        let mut response = self.db.query(query).await?;

        #[derive(serde::Deserialize)]
        struct KnowledgeWithContext {
            id: surrealdb::RecordId,
            fact: String,
            character_name: Option<String>,
        }

        let knowledge_entities: Vec<KnowledgeWithContext> = response.take(0)?;

        stats.total_entities += knowledge_entities.len();

        // Bulk pre-fetch all knows edges to avoid N+1
        let edge_map = self.get_all_knows_edges().await?;

        for chunk in knowledge_entities.chunks(50) {
            let mut texts = Vec::new();
            let mut ids = Vec::new();

            for knowledge in chunk {
                let entity_id = knowledge.id.to_string();

                let edge = edge_map.get(&entity_id);
                let certainty = edge.and_then(|e| e.0.as_deref()).unwrap_or("knows");
                let learning_method = edge.and_then(|e| e.1.as_deref());

                texts.push(knowledge_composite(
                    &knowledge.fact,
                    knowledge.character_name.as_deref().unwrap_or("Someone"),
                    certainty,
                    learning_method,
                ));
                ids.push(entity_id);
            }

            self.embed_and_update_batch(&ids, &texts, "knowledge", stats)
                .await?;
        }

        Ok(())
    }

    /// Backfill perspective embeddings on perceives edges.
    async fn backfill_perspectives(&self, stats: &mut BackfillStats) -> Result<(), NarraError> {
        let query = r#"SELECT id, rel_types, subtype, feelings, perception, tension_level, history_notes,
                              in.name AS observer_name, out.name AS target_name,
                              type::string(in) AS observer_id, type::string(out) AS target_id
                       FROM perceives
                       WHERE embedding IS NONE OR embedding_stale = true"#;
        let mut response = self.db.query(query).await?;

        #[derive(serde::Deserialize)]
        struct PerceivesForBackfill {
            id: surrealdb::RecordId,
            rel_types: Vec<String>,
            subtype: Option<String>,
            feelings: Option<String>,
            perception: Option<String>,
            tension_level: Option<i32>,
            history_notes: Option<String>,
            observer_name: Option<String>,
            target_name: Option<String>,
            observer_id: Option<String>,
            target_id: Option<String>,
        }

        let perceives_edges: Vec<PerceivesForBackfill> = response.take(0)?;

        stats.total_entities += perceives_edges.len();

        // Use batch embedding instead of individual calls
        for chunk in perceives_edges.chunks(50) {
            let mut texts = Vec::new();
            let mut ids = Vec::new();

            for edge in chunk {
                let observer_name = edge.observer_name.as_deref().unwrap_or("Someone");
                let target_name = edge.target_name.as_deref().unwrap_or("Someone");

                let knowledge = if let (Some(ref obs_id), Some(ref tgt_name)) =
                    (&edge.observer_id, &edge.target_name)
                {
                    self.fetch_knowledge_about(obs_id, tgt_name).await
                } else {
                    vec![]
                };

                let shared_scenes = if let (Some(ref obs_id), Some(ref tgt_id)) =
                    (&edge.observer_id, &edge.target_id)
                {
                    self.fetch_shared_scenes(obs_id, tgt_id).await
                } else {
                    vec![]
                };

                texts.push(perspective_composite(
                    observer_name,
                    target_name,
                    &edge.rel_types,
                    edge.subtype.as_deref(),
                    edge.feelings.as_deref(),
                    edge.perception.as_deref(),
                    edge.tension_level,
                    edge.history_notes.as_deref(),
                    &knowledge,
                    &shared_scenes,
                ));
                ids.push(edge.id.to_string());
            }

            self.embed_and_update_batch(&ids, &texts, "perspective", stats)
                .await?;
        }

        Ok(())
    }

    /// Backfill relationship (relates_to) embeddings.
    async fn backfill_relationships(&self, stats: &mut BackfillStats) -> Result<(), NarraError> {
        let query = r#"SELECT id, rel_type, subtype, label,
                              in.name AS from_name, in.roles AS from_roles,
                              out.name AS to_name, out.roles AS to_roles
                       FROM relates_to
                       WHERE embedding IS NONE OR embedding_stale = true"#;
        let mut response = self.db.query(query).await?;

        #[derive(serde::Deserialize)]
        struct RelatesToForBackfill {
            id: surrealdb::RecordId,
            rel_type: String,
            subtype: Option<String>,
            label: Option<String>,
            from_name: Option<String>,
            from_roles: Option<Vec<String>>,
            to_name: Option<String>,
            to_roles: Option<Vec<String>>,
        }

        let edges: Vec<RelatesToForBackfill> = response.take(0)?;

        stats.total_entities += edges.len();

        for edge in &edges {
            let entity_id = edge.id.to_string();

            let composite = relationship_composite(
                edge.from_name.as_deref().unwrap_or("Unknown"),
                &edge.from_roles.clone().unwrap_or_default(),
                edge.to_name.as_deref().unwrap_or("Unknown"),
                &edge.to_roles.clone().unwrap_or_default(),
                &edge.rel_type,
                edge.subtype.as_deref(),
                edge.label.as_deref(),
            );

            self.embed_and_update_single(&entity_id, &composite, "relationship", stats)
                .await?;
        }

        Ok(())
    }

    /// Bulk-fetch all character relationships for composite text generation.
    async fn get_all_character_relationships(
        &self,
    ) -> Result<HashMap<String, Vec<(String, String)>>, NarraError> {
        let query = "SELECT type::string(out) AS char_id, rel_type, label FROM relates_to";
        let mut response = self.db.query(query).await?;

        #[derive(serde::Deserialize)]
        struct RelRecord {
            char_id: Option<String>,
            rel_type: String,
            label: Option<String>,
        }

        let records: Vec<RelRecord> = response.take(0).unwrap_or_default();
        let mut map: HashMap<String, Vec<(String, String)>> = HashMap::new();

        for r in records {
            if let Some(char_id) = r.char_id {
                let label = r.label.unwrap_or_else(|| r.rel_type.clone());
                map.entry(char_id).or_default().push((r.rel_type, label));
            }
        }

        Ok(map)
    }

    /// Bulk-fetch all character perceptions for composite text generation.
    async fn get_all_character_perceptions(
        &self,
    ) -> Result<HashMap<String, Vec<(String, String)>>, NarraError> {
        let query =
            "SELECT type::string(in) AS char_id, out.name AS target_name, perception FROM perceives";
        let mut response = self.db.query(query).await?;

        #[derive(serde::Deserialize)]
        struct PercRecord {
            char_id: Option<String>,
            target_name: Option<String>,
            perception: Option<String>,
        }

        let records: Vec<PercRecord> = response.take(0).unwrap_or_default();
        let mut map: HashMap<String, Vec<(String, String)>> = HashMap::new();

        for r in records {
            if let (Some(char_id), Some(name), Some(text)) =
                (r.char_id, r.target_name, r.perception)
            {
                map.entry(char_id).or_default().push((name, text));
            }
        }

        Ok(map)
    }

    /// Bulk-fetch all knows edges (certainty + learning_method) keyed by knowledge entity ID.
    /// Returns the most recent edge per knowledge entity.
    async fn get_all_knows_edges(
        &self,
    ) -> Result<HashMap<String, (Option<String>, Option<String>)>, NarraError> {
        let query = "SELECT type::string(out) AS knowledge_id, certainty, learning_method, learned_at FROM knows ORDER BY learned_at DESC";
        let mut response = self.db.query(query).await?;

        #[derive(serde::Deserialize)]
        struct KnowsRecord {
            knowledge_id: Option<String>,
            certainty: Option<String>,
            learning_method: Option<String>,
        }

        let records: Vec<KnowsRecord> = response.take(0).unwrap_or_default();
        let mut map: HashMap<String, (Option<String>, Option<String>)> = HashMap::new();

        for r in records {
            if let Some(kid) = r.knowledge_id {
                // First entry wins (most recent due to ORDER BY DESC)
                map.entry(kid).or_insert((r.certainty, r.learning_method));
            }
        }

        Ok(map)
    }

    /// Fetch observer's knowledge about a target character.
    async fn fetch_knowledge_about(
        &self,
        observer_id: &str,
        target_name: &str,
    ) -> Vec<(String, String)> {
        crate::embedding::queries::fetch_knowledge_about(&self.db, observer_id, target_name).await
    }

    /// Fetch scenes shared between two characters.
    async fn fetch_shared_scenes(
        &self,
        observer_id: &str,
        target_id: &str,
    ) -> Vec<(String, Option<String>)> {
        crate::embedding::queries::fetch_shared_scenes(&self.db, observer_id, target_id).await
    }
}
