//! Backfill embedding generation for existing entities.
//!
//! Generates embeddings for all entities that are missing them or have stale embeddings.

#![allow(clippy::needless_borrows_for_generic_args)]

use std::collections::HashMap;
use std::sync::Arc;

use crate::db::connection::NarraDb;
use serde::Serialize;
use tracing::info;

use crate::embedding::composite::{
    character_composite, event_composite, identity_composite, knowledge_composite,
    location_composite, narrative_composite, perspective_composite, psychology_composite,
    relationship_composite, scene_composite, social_composite,
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
    db: Arc<NarraDb>,
    embedding_service: Arc<dyn EmbeddingService + Send + Sync>,
}

impl BackfillService {
    /// Create a new backfill service.
    pub fn new(
        db: Arc<NarraDb>,
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

        // Backfill character facets (identity, psychology, social, narrative)
        let facet_stats = self.backfill_character_facets().await?;
        stats.total_entities += facet_stats.total_entities;
        stats.embedded += facet_stats.embedded;
        stats.skipped += facet_stats.skipped;
        stats.failed += facet_stats.failed;
        stats
            .entity_type_stats
            .insert("character_facets".to_string(), facet_stats.embedded);

        // Update world_meta with current embedding model info
        if stats.embedded > 0 {
            if let Err(e) =
                crate::init::update_embedding_metadata(&self.db, self.embedding_service.as_ref())
                    .await
            {
                tracing::warn!("Failed to update embedding metadata: {}", e);
            }
        }

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

    /// Backfill character facet embeddings (identity, psychology, social, narrative).
    ///
    /// Processes characters where any facet is stale or missing.
    /// Pre-fetches all necessary data in bulk to avoid N+1 queries.
    ///
    /// # Returns
    ///
    /// BackfillStats for facet embeddings.
    pub async fn backfill_character_facets(&self) -> Result<BackfillStats, NarraError> {
        // Check embedding service is available
        if !self.embedding_service.is_available() {
            return Err(NarraError::Database(
                "Embedding service not available - cannot backfill facets".to_string(),
            ));
        }

        info!("Backfilling character facet embeddings");

        let mut stats = BackfillStats::default();

        // Query characters where ANY facet is stale or missing
        let query = r#"
            SELECT * FROM character
            WHERE identity_embedding IS NONE OR identity_stale = true
               OR psychology_embedding IS NONE OR psychology_stale = true
               OR social_embedding IS NONE OR social_stale = true
               OR narrative_embedding IS NONE OR narrative_stale = true
        "#;
        let mut response = self.db.query(query).await?;
        let characters: Vec<Character> = response.take(0)?;

        stats.total_entities = characters.len();

        if characters.is_empty() {
            info!("No character facets need backfill");
            return Ok(stats);
        }

        info!("Backfilling facets for {} characters", characters.len());

        // Bulk pre-fetch all data needed for facet composites
        let all_relationships = self.get_all_character_relationships().await?;
        // Note: outbound perceptions not used for facets (only in general composite)
        let all_perceptions_in = self.get_all_character_inbound_perceptions().await?; // inbound (NEW)
        let all_scene_participation = self.get_all_character_scenes().await?;
        let all_knowledge = self.get_all_character_knowledge().await?;

        // Process characters in chunks of 50
        for chunk in characters.chunks(50) {
            // Build composites for each facet
            let mut identity_texts = Vec::new();
            let mut identity_ids = Vec::new();
            let mut identity_needs_update = Vec::new();

            let mut psychology_texts = Vec::new();
            let mut psychology_ids = Vec::new();
            let mut psychology_needs_update = Vec::new();

            let mut social_texts = Vec::new();
            let mut social_ids = Vec::new();
            let mut social_needs_update = Vec::new();

            let mut narrative_texts = Vec::new();
            let mut narrative_ids = Vec::new();
            let mut narrative_needs_update = Vec::new();

            for character in chunk {
                let char_id = character.id.to_string();

                // Check which facets need updates
                let needs_identity = {
                    let query_result = self
                        .db
                        .query(&format!(
                            "SELECT identity_embedding, identity_stale FROM {}",
                            char_id
                        ))
                        .await;

                    #[derive(serde::Deserialize)]
                    struct FacetCheck {
                        identity_embedding: Option<Vec<f32>>,
                        identity_stale: Option<bool>,
                    }

                    match query_result {
                        Ok(mut resp) => {
                            let check: Option<FacetCheck> = resp.take(0).unwrap_or(None);
                            check.is_none_or(|c| {
                                c.identity_embedding.is_none() || c.identity_stale.unwrap_or(false)
                            })
                        }
                        Err(_) => true,
                    }
                };

                if needs_identity {
                    identity_texts.push(identity_composite(character));
                    identity_ids.push(char_id.clone());
                    identity_needs_update.push(char_id.clone());
                }

                // Psychology facet
                let needs_psychology = {
                    let query_result = self
                        .db
                        .query(&format!(
                            "SELECT psychology_embedding, psychology_stale FROM {}",
                            char_id
                        ))
                        .await;

                    #[derive(serde::Deserialize)]
                    struct FacetCheck {
                        psychology_embedding: Option<Vec<f32>>,
                        psychology_stale: Option<bool>,
                    }

                    match query_result {
                        Ok(mut resp) => {
                            let check: Option<FacetCheck> = resp.take(0).unwrap_or(None);
                            check.is_none_or(|c| {
                                c.psychology_embedding.is_none()
                                    || c.psychology_stale.unwrap_or(false)
                            })
                        }
                        Err(_) => true,
                    }
                };

                if needs_psychology {
                    psychology_texts.push(psychology_composite(character));
                    psychology_ids.push(char_id.clone());
                    psychology_needs_update.push(char_id.clone());
                }

                // Social facet
                let needs_social = {
                    let query_result = self
                        .db
                        .query(&format!(
                            "SELECT social_embedding, social_stale FROM {}",
                            char_id
                        ))
                        .await;

                    #[derive(serde::Deserialize)]
                    struct FacetCheck {
                        social_embedding: Option<Vec<f32>>,
                        social_stale: Option<bool>,
                    }

                    match query_result {
                        Ok(mut resp) => {
                            let check: Option<FacetCheck> = resp.take(0).unwrap_or(None);
                            check.is_none_or(|c| {
                                c.social_embedding.is_none() || c.social_stale.unwrap_or(false)
                            })
                        }
                        Err(_) => true,
                    }
                };

                if needs_social {
                    let relationships =
                        all_relationships.get(&char_id).cloned().unwrap_or_default();
                    let perceptions_in = all_perceptions_in
                        .get(&char_id)
                        .cloned()
                        .unwrap_or_default();
                    social_texts.push(social_composite(character, &relationships, &perceptions_in));
                    social_ids.push(char_id.clone());
                    social_needs_update.push(char_id.clone());
                }

                // Narrative facet
                let needs_narrative = {
                    let query_result = self
                        .db
                        .query(&format!(
                            "SELECT narrative_embedding, narrative_stale FROM {}",
                            char_id
                        ))
                        .await;

                    #[derive(serde::Deserialize)]
                    struct FacetCheck {
                        narrative_embedding: Option<Vec<f32>>,
                        narrative_stale: Option<bool>,
                    }

                    match query_result {
                        Ok(mut resp) => {
                            let check: Option<FacetCheck> = resp.take(0).unwrap_or(None);
                            check.is_none_or(|c| {
                                c.narrative_embedding.is_none()
                                    || c.narrative_stale.unwrap_or(false)
                            })
                        }
                        Err(_) => true,
                    }
                };

                if needs_narrative {
                    let scenes = all_scene_participation
                        .get(&char_id)
                        .cloned()
                        .unwrap_or_default();
                    let knowledge = all_knowledge.get(&char_id).cloned().unwrap_or_default();
                    narrative_texts.push(narrative_composite(&character.name, &scenes, &knowledge));
                    narrative_ids.push(char_id.clone());
                    narrative_needs_update.push(char_id);
                }
            }

            // Embed each facet batch
            if !identity_texts.is_empty() {
                self.embed_and_update_facet_batch(
                    &identity_ids,
                    &identity_texts,
                    "identity",
                    &mut stats,
                )
                .await?;
            }

            if !psychology_texts.is_empty() {
                self.embed_and_update_facet_batch(
                    &psychology_ids,
                    &psychology_texts,
                    "psychology",
                    &mut stats,
                )
                .await?;
            }

            if !social_texts.is_empty() {
                self.embed_and_update_facet_batch(&social_ids, &social_texts, "social", &mut stats)
                    .await?;
            }

            if !narrative_texts.is_empty() {
                self.embed_and_update_facet_batch(
                    &narrative_ids,
                    &narrative_texts,
                    "narrative",
                    &mut stats,
                )
                .await?;
            }
        }

        info!(
            "Backfilled character facets: {} embedded, {} skipped, {} failed",
            stats.embedded, stats.skipped, stats.failed
        );

        Ok(stats)
    }

    /// Embed a batch of facet texts and update their entities in the database.
    async fn embed_and_update_facet_batch(
        &self,
        entity_ids: &[String],
        texts: &[String],
        facet: &str,
        stats: &mut BackfillStats,
    ) -> Result<(), NarraError> {
        match self.embedding_service.embed_batch(texts).await {
            Ok(embeddings) => {
                for (i, entity_id) in entity_ids.iter().enumerate() {
                    let update_query = format!(
                        "UPDATE {} SET {}_embedding = $embedding, {}_stale = false, {}_composite = $composite_text",
                        entity_id, facet, facet, facet
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
                            tracing::warn!(
                                "Failed to update {} facet for {}: {}",
                                facet,
                                entity_id,
                                e
                            );
                            stats.failed += 1;
                        }
                    }

                    // Create arc snapshot for this facet
                    if let Err(e) = self
                        .db
                        .query(
                            "CREATE arc_snapshot SET entity_id = $eid, entity_type = 'character', \
                             facet = $facet, embedding = $snap_embedding, delta_magnitude = NONE",
                        )
                        .bind((
                            "eid",
                            surrealdb::RecordId::from((
                                entity_id.split(':').next().unwrap_or("character"),
                                entity_id.split(':').nth(1).unwrap_or("unknown"),
                            )),
                        ))
                        .bind(("facet", facet.to_string()))
                        .bind(("snap_embedding", embeddings[i].clone()))
                        .await
                    {
                        tracing::warn!(
                            "Failed to create arc snapshot for {} facet {}: {}",
                            facet,
                            entity_id,
                            e
                        );
                    }
                }

                if stats.embedded.is_multiple_of(50) && stats.embedded > 0 {
                    info!("Backfilled {} character facets so far", stats.embedded);
                }
            }
            Err(e) => {
                tracing::warn!(
                    "Failed to generate embeddings for {} facet batch: {}",
                    facet,
                    e
                );
                stats.failed += entity_ids.len();
            }
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

    /// Bulk-fetch inbound perceptions for all characters (how others see them).
    /// Returns map of character_id -> Vec<(observer_name, perception_text)>
    async fn get_all_character_inbound_perceptions(
        &self,
    ) -> Result<HashMap<String, Vec<(String, String)>>, NarraError> {
        let query =
            "SELECT type::string(out) AS char_id, in.name AS observer_name, perception FROM perceives";
        let mut response = self.db.query(query).await?;

        #[derive(serde::Deserialize)]
        struct PercRecord {
            char_id: Option<String>,
            observer_name: Option<String>,
            perception: Option<String>,
        }

        let records: Vec<PercRecord> = response.take(0).unwrap_or_default();
        let mut map: HashMap<String, Vec<(String, String)>> = HashMap::new();

        for r in records {
            if let (Some(char_id), Some(observer), Some(text)) =
                (r.char_id, r.observer_name, r.perception)
            {
                map.entry(char_id).or_default().push((observer, text));
            }
        }

        Ok(map)
    }

    /// Bulk-fetch scene participation for all characters.
    /// Returns map of character_id -> Vec<(scene_title, optional_summary)>
    async fn get_all_character_scenes(
        &self,
    ) -> Result<HashMap<String, Vec<(String, Option<String>)>>, NarraError> {
        let query = r#"SELECT type::string(in) AS char_id, out.title AS scene_title,
                              out.summary AS scene_summary FROM participates_in"#;
        let mut response = self.db.query(query).await?;

        #[derive(serde::Deserialize)]
        struct SceneRecord {
            char_id: Option<String>,
            scene_title: Option<String>,
            scene_summary: Option<String>,
        }

        let records: Vec<SceneRecord> = response.take(0).unwrap_or_default();
        let mut map: HashMap<String, Vec<(String, Option<String>)>> = HashMap::new();

        for r in records {
            if let (Some(char_id), Some(title)) = (r.char_id, r.scene_title) {
                map.entry(char_id)
                    .or_default()
                    .push((title, r.scene_summary));
            }
        }

        Ok(map)
    }

    /// Bulk-fetch knowledge for all characters.
    /// Returns map of character_id -> Vec<(fact, certainty)>
    async fn get_all_character_knowledge(
        &self,
    ) -> Result<HashMap<String, Vec<(String, String)>>, NarraError> {
        let query = r#"SELECT type::string(in) AS char_id, out.fact AS fact, certainty
                       FROM knows WHERE out.fact IS NOT NONE ORDER BY learned_at DESC"#;
        let mut response = self.db.query(query).await?;

        #[derive(serde::Deserialize)]
        struct KnowledgeRecord {
            char_id: Option<String>,
            fact: Option<String>,
            certainty: Option<String>,
        }

        let records: Vec<KnowledgeRecord> = response.take(0).unwrap_or_default();
        let mut map: HashMap<String, Vec<(String, String)>> = HashMap::new();

        for r in records {
            if let (Some(char_id), Some(fact)) = (r.char_id, r.fact) {
                let certainty = r.certainty.unwrap_or_else(|| "knows".to_string());
                map.entry(char_id).or_default().push((fact, certainty));
            }
        }

        Ok(map)
    }
}
