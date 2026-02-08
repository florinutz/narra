//! Embedding staleness management.
//!
//! Tracks when entity embeddings need regeneration and handles background updates.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use surrealdb::engine::local::Db;
use surrealdb::Surreal;
use tokio::time::Instant;
use tracing::{error, info, warn};

use crate::embedding::composite::{
    character_composite, event_composite, knowledge_composite, location_composite,
    perspective_composite, relationship_composite, scene_composite,
};
use crate::embedding::EmbeddingService;
use crate::models::{Character, Event, Location, Scene};
use crate::utils::math::cosine_similarity;
use crate::NarraError;

/// Manages embedding staleness and regeneration.
///
/// Marks entities as stale when they change and spawns background tasks
/// to regenerate embeddings asynchronously.
/// Debounce window for regeneration spawns (seconds).
const REGENERATION_DEBOUNCE_SECS: u64 = 2;

pub struct StalenessManager {
    db: Arc<Surreal<Db>>,
    embedding_service: Arc<dyn EmbeddingService + Send + Sync>,
    /// Tracks in-flight regenerations to debounce rapid mutations on the same entity.
    in_flight: Arc<Mutex<HashMap<String, Instant>>>,
}

impl StalenessManager {
    /// Create a new staleness manager.
    ///
    /// # Arguments
    ///
    /// * `db` - Database connection
    /// * `embedding_service` - Embedding service for generating vectors
    pub fn new(
        db: Arc<Surreal<Db>>,
        embedding_service: Arc<dyn EmbeddingService + Send + Sync>,
    ) -> Self {
        Self {
            db,
            embedding_service,
            in_flight: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Mark an entity as stale (embedding needs regeneration).
    ///
    /// # Arguments
    ///
    /// * `entity_id` - Entity ID in format "table:key" (e.g., "character:alice")
    ///
    /// # Returns
    ///
    /// Ok(()) if entity was marked stale, Err if operation failed.
    pub async fn mark_stale(&self, entity_id: &str) -> Result<(), NarraError> {
        // Parse entity type from entity_id
        let parts: Vec<&str> = entity_id.split(':').collect();
        if parts.len() != 2 {
            return Err(NarraError::Database(format!(
                "Invalid entity_id format: {}",
                entity_id
            )));
        }

        // Update the entity to mark it stale
        let query = format!("UPDATE {} SET embedding_stale = true", entity_id);
        self.db.query(query).await.map_err(|e| {
            NarraError::Database(format!("Failed to mark {} stale: {}", entity_id, e))
        })?;

        info!("Marked {} as stale", entity_id);
        Ok(())
    }

    /// Mark related entities as stale (1-hop cascade).
    ///
    /// For characters, marks all directly connected characters as stale
    /// (via relates_to edges). For other entity types, this is a no-op.
    ///
    /// # Arguments
    ///
    /// * `entity_id` - Entity ID in format "table:key"
    ///
    /// # Returns
    ///
    /// Ok(()) if related entities were marked stale, Err if operation failed.
    pub async fn mark_related_stale(&self, entity_id: &str) -> Result<(), NarraError> {
        // Only cascade for characters (relationships are character-to-character)
        if !entity_id.starts_with("character:") {
            return Ok(());
        }

        // Find all connected characters via relates_to edges
        let query = format!(
            "SELECT ->relates_to->character.id AS related FROM {}",
            entity_id
        );

        let mut result =
            self.db.query(&query).await.map_err(|e| {
                NarraError::Database(format!("Failed to find related entities: {}", e))
            })?;

        #[derive(serde::Deserialize)]
        struct RelatedResult {
            related: Option<Vec<surrealdb::RecordId>>,
        }

        let related: Vec<RelatedResult> = result.take(0).map_err(|e| {
            NarraError::Database(format!("Failed to parse related entities: {}", e))
        })?;

        // Mark each related character as stale
        let mut count = 0;
        for r in related {
            if let Some(ids) = r.related {
                for id in ids {
                    let entity_id_str = id.to_string();
                    self.mark_stale(&entity_id_str).await?;
                    count += 1;
                }
            }
        }

        if count > 0 {
            info!("Marked {} related entities stale for {}", count, entity_id);
        }

        Ok(())
    }

    /// Check if a regeneration should be debounced.
    ///
    /// Returns true if the spawn should proceed, false if it should be skipped.
    /// On proceed, records the entity in the in-flight map.
    fn should_spawn(&self, entity_id: &str) -> bool {
        let mut map = self.in_flight.lock().unwrap_or_else(|e| e.into_inner());
        let now = Instant::now();
        if let Some(last) = map.get(entity_id) {
            if now.duration_since(*last).as_secs() < REGENERATION_DEBOUNCE_SECS {
                info!(
                    "Debounced regeneration for {} (spawned {}ms ago)",
                    entity_id,
                    now.duration_since(*last).as_millis()
                );
                return false;
            }
        }
        map.insert(entity_id.to_string(), now);

        // Prune stale entries to prevent unbounded growth
        if map.len() > 1000 {
            let cutoff = now - std::time::Duration::from_secs(REGENERATION_DEBOUNCE_SECS * 5);
            map.retain(|_, v| *v > cutoff);
        }
        true
    }

    /// Spawn background task to regenerate embedding for an entity.
    ///
    /// This is fire-and-forget: the entity operation returns immediately
    /// while embedding regeneration happens in the background.
    /// Rapid calls for the same entity within 2s are debounced.
    pub fn spawn_regeneration(
        &self,
        entity_id: String,
        entity_type: String,
        event_id: Option<String>,
    ) {
        if !self.should_spawn(&entity_id) {
            return;
        }

        let db = self.db.clone();
        let embedding_service = self.embedding_service.clone();
        let in_flight = Arc::clone(&self.in_flight);

        tokio::spawn(async move {
            let result = regenerate_embedding_internal(
                db,
                embedding_service,
                &entity_id,
                &entity_type,
                event_id,
            )
            .await;

            // Clear in-flight entry after completion
            if let Ok(mut map) = in_flight.lock() {
                map.remove(&entity_id);
            }

            if let Err(e) = result {
                error!("Failed to regenerate embedding for {}: {}", entity_id, e);
            }
        });
    }

    /// Regenerate embedding for an entity (synchronous, awaitable).
    ///
    /// Used during backfill operations where we want to wait for completion.
    ///
    /// # Arguments
    ///
    /// * `entity_id` - Entity ID in format "table:key"
    /// * `event_id` - Optional event ID to link to the arc snapshot
    ///
    /// # Returns
    ///
    /// Ok(()) if embedding was regenerated, Err if operation failed.
    pub async fn regenerate_embedding(
        &self,
        entity_id: &str,
        event_id: Option<String>,
    ) -> Result<(), NarraError> {
        // Parse entity type from entity_id
        let parts: Vec<&str> = entity_id.split(':').collect();
        if parts.len() != 2 {
            return Err(NarraError::Database(format!(
                "Invalid entity_id format: {}",
                entity_id
            )));
        }

        let entity_type = parts[0];

        regenerate_embedding_internal(
            self.db.clone(),
            self.embedding_service.clone(),
            entity_id,
            entity_type,
            event_id,
        )
        .await
    }
}

use super::queries::{fetch_knowledge_about, fetch_shared_scenes};

/// Internal function to regenerate embedding.
///
/// Shared by spawn_regeneration and regenerate_embedding.
async fn regenerate_embedding_internal(
    db: Arc<Surreal<Db>>,
    embedding_service: Arc<dyn EmbeddingService + Send + Sync>,
    entity_id: &str,
    entity_type: &str,
    event_id: Option<String>,
) -> Result<(), NarraError> {
    // Check if embedding service is available
    if !embedding_service.is_available() {
        return Err(NarraError::Database(
            "Embedding service is not available".to_string(),
        ));
    }

    // Parse entity_id into RecordId for parameterized queries
    let entity_ref = {
        let (table, key) = entity_id
            .split_once(':')
            .ok_or_else(|| NarraError::Validation(format!("Invalid entity ID: {}", entity_id)))?;
        surrealdb::RecordId::from((table, key))
    };

    // For arc-trackable types, fetch old embedding before regeneration
    let old_embedding: Option<Vec<f32>> = if matches!(
        entity_type,
        "character" | "knowledge" | "perceives" | "relates_to"
    ) {
        #[derive(serde::Deserialize)]
        struct EmbRow {
            embedding: Option<Vec<f32>>,
        }
        let mut result = db
            .query("SELECT embedding FROM ONLY $ref")
            .bind(("ref", entity_ref.clone()))
            .await
            .map_err(|e| NarraError::Database(format!("Failed to fetch old embedding: {}", e)))?;
        let row: Option<EmbRow> = result.take(0).unwrap_or(None);
        row.and_then(|r| r.embedding)
    } else {
        None
    };

    // Fetch entity from database
    let composite_text = match entity_type {
        "character" => {
            // Fetch character directly (no #[serde(flatten)] — SurrealDB RecordId
            // serialization is incompatible with serde's flatten buffering)
            let mut result = db
                .query("SELECT * FROM ONLY $ref")
                .bind(("ref", entity_ref.clone()))
                .await
                .map_err(|e| NarraError::Database(format!("Failed to fetch character: {}", e)))?;

            let character: Option<Character> = result
                .take(0)
                .map_err(|e| NarraError::Database(format!("Failed to parse character: {}", e)))?;

            let character = character.ok_or_else(|| {
                NarraError::Database(format!("Character not found: {}", entity_id))
            })?;

            // Fetch relationships separately
            let mut rel_result = db
                .query("SELECT ->relates_to->character.{name, relationship_type} AS relationships FROM ONLY $ref")
                .bind(("ref", entity_ref.clone()))
                .await
                .map_err(|e| {
                    NarraError::Database(format!("Failed to fetch relationships: {}", e))
                })?;

            #[derive(serde::Deserialize)]
            struct RelResult {
                relationships: Option<Vec<serde_json::Value>>,
            }

            let rel_data: Option<RelResult> = rel_result.take(0).unwrap_or(None);
            let relationships: Vec<(String, String)> = rel_data
                .and_then(|r| r.relationships)
                .unwrap_or_default()
                .iter()
                .filter_map(|r| {
                    let rel_type = r.get("relationship_type")?.as_str()?;
                    let name = r.get("name")?.as_str()?;
                    Some((rel_type.to_string(), name.to_string()))
                })
                .collect();

            // Fetch perceptions (how this character sees others)
            #[derive(serde::Deserialize)]
            struct PerceptionRecord {
                target_name: Option<String>,
                perception: Option<String>,
            }

            let mut perc_result = db
                .query("SELECT out.name AS target_name, perception FROM perceives WHERE in = $ref")
                .bind(("ref", entity_ref.clone()))
                .await
                .map_err(|e| NarraError::Database(format!("Failed to fetch perceptions: {}", e)))?;

            let perc_records: Vec<PerceptionRecord> = perc_result.take(0).unwrap_or_default();
            let perceptions: Vec<(String, String)> = perc_records
                .into_iter()
                .filter_map(|p| Some((p.target_name?, p.perception?)))
                .collect();

            character_composite(&character, &relationships, &perceptions)
        }
        "location" => {
            let mut locations: Vec<Location> = db
                .select(entity_id)
                .await
                .map_err(|e| NarraError::Database(format!("Failed to fetch location: {}", e)))?;

            let location = locations.pop().ok_or_else(|| {
                NarraError::Database(format!("Location not found: {}", entity_id))
            })?;

            location_composite(&location)
        }
        "event" => {
            let mut events: Vec<Event> = db
                .select(entity_id)
                .await
                .map_err(|e| NarraError::Database(format!("Failed to fetch event: {}", e)))?;

            let event = events
                .pop()
                .ok_or_else(|| NarraError::Database(format!("Event not found: {}", entity_id)))?;

            event_composite(&event)
        }
        "scene" => {
            // Fetch only the fields needed for composite text — avoid #[serde(flatten)]
            // which fails with SurrealDB's RecordId serialization.
            let mut result = db
                .query(
                    "SELECT id, title, summary, event.title AS event_title, \
                     primary_location.name AS location_name FROM ONLY $ref",
                )
                .bind(("ref", entity_ref.clone()))
                .await
                .map_err(|e| NarraError::Database(format!("Failed to fetch scene: {}", e)))?;

            #[derive(serde::Deserialize)]
            struct SceneForComposite {
                id: surrealdb::RecordId,
                title: String,
                summary: Option<String>,
                event_title: Option<String>,
                location_name: Option<String>,
            }

            let scene_data: Option<SceneForComposite> = result
                .take(0)
                .map_err(|e| NarraError::Database(format!("Failed to parse scene: {}", e)))?;

            let scene_data = scene_data
                .ok_or_else(|| NarraError::Database(format!("Scene not found: {}", entity_id)))?;

            let scene = Scene {
                id: scene_data.id,
                title: scene_data.title,
                summary: scene_data.summary,
                event: surrealdb::RecordId::from(("event", "placeholder")),
                primary_location: surrealdb::RecordId::from(("location", "placeholder")),
                secondary_locations: vec![],
                created_at: surrealdb::Datetime::default(),
                updated_at: surrealdb::Datetime::default(),
            };

            scene_composite(
                &scene,
                scene_data.event_title.as_deref(),
                scene_data.location_name.as_deref(),
            )
        }
        "knowledge" => {
            // Fetch only the fields needed — avoid SELECT * which includes RecordId
            // fields that fail SurrealDB's serde deserialization.
            let mut result = db
                .query("SELECT fact, character.name AS character_name FROM ONLY $ref")
                .bind(("ref", entity_ref.clone()))
                .await
                .map_err(|e| NarraError::Database(format!("Failed to fetch knowledge: {}", e)))?;

            #[derive(serde::Deserialize)]
            struct KnowledgeWithContext {
                fact: String,
                character_name: Option<String>,
            }

            let knowledge: Option<KnowledgeWithContext> = result
                .take(0)
                .map_err(|e| NarraError::Database(format!("Failed to parse knowledge: {}", e)))?;

            let knowledge = knowledge.ok_or_else(|| {
                NarraError::Database(format!("Knowledge not found: {}", entity_id))
            })?;

            // Fetch latest certainty and learning method from knows edge
            let mut edge_result = db
                .query(
                    "SELECT certainty, learning_method, learned_at FROM knows WHERE out = $ref ORDER BY learned_at DESC LIMIT 1",
                )
                .bind(("ref", entity_ref.clone()))
                .await
                .map_err(|e| {
                    NarraError::Database(format!("Failed to fetch knows edge: {}", e))
                })?;

            #[derive(serde::Deserialize)]
            struct KnowsEdge {
                certainty: Option<String>,
                learning_method: Option<String>,
            }

            let edge: Option<KnowsEdge> = edge_result
                .take(0)
                .map_err(|e| NarraError::Database(format!("Failed to parse knows edge: {}", e)))?;

            let certainty = edge
                .as_ref()
                .and_then(|e| e.certainty.as_deref())
                .unwrap_or("knows");
            let learning_method = edge.as_ref().and_then(|e| e.learning_method.as_deref());

            knowledge_composite(
                &knowledge.fact,
                knowledge.character_name.as_deref().unwrap_or("Someone"),
                certainty,
                learning_method,
            )
        }
        "relates_to" => {
            // Fetch relates_to edge data + connected character names/roles
            #[derive(serde::Deserialize)]
            struct RelatesToData {
                rel_type: String,
                subtype: Option<String>,
                label: Option<String>,
                from_name: Option<String>,
                from_roles: Option<Vec<String>>,
                to_name: Option<String>,
                to_roles: Option<Vec<String>>,
            }

            let mut result = db
                .query(
                    "SELECT rel_type, subtype, label, \
                     in.name AS from_name, in.roles AS from_roles, \
                     out.name AS to_name, out.roles AS to_roles \
                     FROM ONLY $ref",
                )
                .bind(("ref", entity_ref.clone()))
                .await
                .map_err(|e| NarraError::Database(format!("Failed to fetch relates_to: {}", e)))?;

            let data: Option<RelatesToData> = result
                .take(0)
                .map_err(|e| NarraError::Database(format!("Failed to parse relates_to: {}", e)))?;
            let data = data.ok_or_else(|| {
                NarraError::Database(format!("relates_to edge not found: {}", entity_id))
            })?;

            relationship_composite(
                data.from_name.as_deref().unwrap_or("Unknown"),
                &data.from_roles.unwrap_or_default(),
                data.to_name.as_deref().unwrap_or("Unknown"),
                &data.to_roles.unwrap_or_default(),
                &data.rel_type,
                data.subtype.as_deref(),
                data.label.as_deref(),
            )
        }
        "perceives" => {
            // Fetch perceives edge data + observer/target names
            #[derive(serde::Deserialize)]
            struct PerceivesData {
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

            let mut result = db
                .query(
                    "SELECT rel_types, subtype, feelings, perception, tension_level, history_notes, \
                     in.name AS observer_name, out.name AS target_name, \
                     type::string(in) AS observer_id, type::string(out) AS target_id FROM ONLY $ref",
                )
                .bind(("ref", entity_ref.clone()))
                .await
                .map_err(|e| NarraError::Database(format!("Failed to fetch perceives: {}", e)))?;

            let data: Option<PerceivesData> = result
                .take(0)
                .map_err(|e| NarraError::Database(format!("Failed to parse perceives: {}", e)))?;
            let data = data.ok_or_else(|| {
                NarraError::Database(format!("Perceives edge not found: {}", entity_id))
            })?;

            let observer_name_str = data.observer_name.as_deref().unwrap_or("Someone");
            let target_name_str = data.target_name.as_deref().unwrap_or("Someone");

            // Fetch observer's knowledge about target (text-match on target name)
            let knowledge = if let (Some(ref tgt_name), Some(ref obs_id)) =
                (&data.target_name, &data.observer_id)
            {
                fetch_knowledge_about(&db, obs_id, tgt_name).await
            } else {
                vec![]
            };

            // Fetch shared scenes
            let shared_scenes = if let (Some(ref obs_id), Some(ref tgt_id)) =
                (&data.observer_id, &data.target_id)
            {
                fetch_shared_scenes(&db, obs_id, tgt_id).await
            } else {
                vec![]
            };

            perspective_composite(
                observer_name_str,
                target_name_str,
                &data.rel_types,
                data.subtype.as_deref(),
                data.feelings.as_deref(),
                data.perception.as_deref(),
                data.tension_level,
                data.history_notes.as_deref(),
                &knowledge,
                &shared_scenes,
            )
        }
        _ => {
            return Err(NarraError::Database(format!(
                "Unknown entity type: {}",
                entity_type
            )));
        }
    };

    // No-op detection: skip re-embedding if composite text hasn't changed
    {
        #[derive(serde::Deserialize)]
        struct CompositeRow {
            composite_text: Option<String>,
        }
        let mut r = db
            .query("SELECT composite_text FROM ONLY $ref")
            .bind(("ref", entity_ref.clone()))
            .await
            .map_err(|e| NarraError::Database(format!("Failed to fetch composite_text: {}", e)))?;
        let stored_composite: Option<String> = r
            .take::<Option<CompositeRow>>(0)
            .unwrap_or(None)
            .and_then(|r| r.composite_text);

        if stored_composite.as_deref() == Some(&composite_text) {
            // Composite unchanged — clear stale flag without re-embedding
            db.query("UPDATE ONLY $ref SET embedding_stale = false")
                .bind(("ref", entity_ref.clone()))
                .await
                .map_err(|e| NarraError::Database(format!("Failed to clear stale flag: {}", e)))?;
            info!("Skipped re-embedding {} (composite unchanged)", entity_id);
            return Ok(());
        }
    }

    // Generate embedding
    let embedding = embedding_service
        .embed_text(&composite_text)
        .await
        .map_err(|e| NarraError::Database(format!("Failed to generate embedding: {}", e)))?;

    // Create arc snapshot for trackable types (character, knowledge, perceives, relates_to)
    if matches!(
        entity_type,
        "character" | "knowledge" | "perceives" | "relates_to"
    ) {
        let delta_magnitude = old_embedding
            .as_ref()
            .map(|old| 1.0 - cosine_similarity(old, &embedding));

        let event_ref: Option<surrealdb::RecordId> = match &event_id {
            Some(eid) if eid.starts_with("event:") => {
                let key = eid.strip_prefix("event:").unwrap_or(eid);
                Some(surrealdb::RecordId::from(("event", key)))
            }
            Some(eid) => Some(surrealdb::RecordId::from(("event", eid.as_str()))),
            None => None,
        };

        // Use friendly names as arc_snapshot entity_type for edge types
        let snapshot_entity_type = match entity_type {
            "perceives" => "perspective",
            "relates_to" => "relationship",
            other => other,
        };

        if let Err(e) = db
            .query(
                "CREATE arc_snapshot SET entity_id = $eid, entity_type = $entity_type, \
                 embedding = $snap_embedding, delta_magnitude = $delta_magnitude, event_id = $event_ref",
            )
            .bind(("eid", entity_ref.clone()))
            .bind(("entity_type", snapshot_entity_type.to_string()))
            .bind(("snap_embedding", embedding.clone()))
            .bind(("delta_magnitude", delta_magnitude))
            .bind(("event_ref", event_ref))
            .await
        {
            warn!("Failed to create arc snapshot for {}: {}", entity_id, e);
        }
    }

    // Update entity with new embedding + composite text
    db.query("UPDATE ONLY $ref SET embedding = $embedding, embedding_stale = false, composite_text = $composite_text")
        .bind(("ref", entity_ref))
        .bind(("embedding", embedding))
        .bind(("composite_text", composite_text.clone()))
        .await
        .map_err(|e| {
            NarraError::Database(format!(
                "Failed to update embedding for {}: {}",
                entity_id, e
            ))
        })?;

    info!(
        "Regenerated embedding for {} ({} chars)",
        entity_id,
        composite_text.len()
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Standalone debounce checker for testing — same logic as StalenessManager::should_spawn
    /// but without needing a DB connection.
    struct DebounceTracker {
        in_flight: Mutex<HashMap<String, Instant>>,
    }

    impl DebounceTracker {
        fn new() -> Self {
            Self {
                in_flight: Mutex::new(HashMap::new()),
            }
        }

        fn should_spawn(&self, entity_id: &str) -> bool {
            let mut map = self.in_flight.lock().unwrap();
            let now = Instant::now();
            if let Some(last) = map.get(entity_id) {
                if now.duration_since(*last).as_secs() < REGENERATION_DEBOUNCE_SECS {
                    return false;
                }
            }
            map.insert(entity_id.to_string(), now);
            true
        }
    }

    #[test]
    fn test_debounce_first_call_proceeds() {
        let tracker = DebounceTracker::new();
        assert!(tracker.should_spawn("character:alice"));
    }

    #[test]
    fn test_debounce_second_call_within_window_is_skipped() {
        let tracker = DebounceTracker::new();
        assert!(tracker.should_spawn("character:alice"));
        // Immediate second call — within 2s window
        assert!(!tracker.should_spawn("character:alice"));
    }

    #[test]
    fn test_debounce_different_entities_both_proceed() {
        let tracker = DebounceTracker::new();
        assert!(tracker.should_spawn("character:alice"));
        assert!(tracker.should_spawn("character:bob"));
    }

    #[test]
    fn test_debounce_after_window_expires() {
        let tracker = DebounceTracker::new();

        // Manually insert a timestamp in the past (beyond debounce window)
        {
            let mut map = tracker.in_flight.lock().unwrap();
            map.insert(
                "character:alice".to_string(),
                Instant::now() - std::time::Duration::from_secs(REGENERATION_DEBOUNCE_SECS + 1),
            );
        }

        // Should proceed since the window has expired
        assert!(tracker.should_spawn("character:alice"));
    }
}
