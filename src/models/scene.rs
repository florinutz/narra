//! Scene entity and participation tracking.
//!
//! Scenes are narrative units that happen at a specific point in time (event)
//! and location. They track character participation through the `participates_in`
//! edge table and event involvement through the `involved_in` edge table.

use crate::db::connection::NarraDb;
use serde::{Deserialize, Serialize};
use surrealdb::{Datetime, RecordId};

use crate::NarraError;

/// A scene in the narrative.
///
/// Scenes are anchored to:
/// - An `event` (when it happens in the timeline)
/// - A `primary_location` (where it takes place)
/// - Optional `secondary_locations` (other locations involved)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scene {
    pub id: RecordId,
    pub title: String,
    pub summary: Option<String>,
    pub event: RecordId,
    pub primary_location: RecordId,
    #[serde(default)]
    pub secondary_locations: Vec<RecordId>,
    pub created_at: Datetime,
    pub updated_at: Datetime,
}

/// Data for creating a new scene.
#[derive(Debug, Serialize)]
pub struct SceneCreate {
    pub title: String,
    pub summary: Option<String>,
    pub event: RecordId,
    pub primary_location: RecordId,
    #[serde(default)]
    pub secondary_locations: Vec<RecordId>,
}

/// Data for updating a scene.
#[derive(Debug, Serialize)]
pub struct SceneUpdate {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<Option<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub event: Option<RecordId>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub primary_location: Option<RecordId>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub secondary_locations: Option<Vec<RecordId>>,
    pub updated_at: Datetime,
}

/// Create a new scene in the database.
///
/// # Arguments
///
/// * `db` - Database connection
/// * `data` - Scene creation data
///
/// # Returns
///
/// The created scene with generated ID and timestamps.
pub async fn create_scene(db: &NarraDb, data: SceneCreate) -> Result<Scene, NarraError> {
    let result: Option<Scene> = db.create("scene").content(data).await?;
    result.ok_or_else(|| NarraError::Database("Failed to create scene".into()))
}

/// Create a new scene with a caller-specified ID.
pub async fn create_scene_with_id(
    db: &NarraDb,
    id: &str,
    data: SceneCreate,
) -> Result<Scene, NarraError> {
    let result: Option<Scene> = db.create(("scene", id)).content(data).await?;
    result.ok_or_else(|| NarraError::Database("Failed to create scene".into()))
}

/// Get a scene by ID.
///
/// # Arguments
///
/// * `db` - Database connection
/// * `id` - Scene ID (the key part, not the full RecordId)
///
/// # Returns
///
/// The scene if found, None otherwise.
pub async fn get_scene(db: &NarraDb, id: &str) -> Result<Option<Scene>, NarraError> {
    let result: Option<Scene> = db.select(("scene", id)).await?;
    Ok(result)
}

/// List all scenes.
///
/// # Arguments
///
/// * `db` - Database connection
///
/// # Returns
///
/// A vector of all scenes.
pub async fn list_scenes(db: &NarraDb) -> Result<Vec<Scene>, NarraError> {
    let result: Vec<Scene> = db.select("scene").await?;
    Ok(result)
}

/// Update a scene by ID.
///
/// # Arguments
///
/// * `db` - Database connection
/// * `id` - Scene ID (the key part, not the full RecordId)
/// * `data` - Fields to update
///
/// # Returns
///
/// The updated scene if found, None otherwise.
pub async fn update_scene(
    db: &NarraDb,
    id: &str,
    data: SceneUpdate,
) -> Result<Option<Scene>, NarraError> {
    let result: Option<Scene> = db.update(("scene", id)).merge(data).await?;
    Ok(result)
}

/// Delete a scene by ID.
///
/// # Arguments
///
/// * `db` - Database connection
/// * `id` - Scene ID (the key part, not the full RecordId)
///
/// # Returns
///
/// The deleted scene if found, None otherwise.
pub async fn delete_scene(db: &NarraDb, id: &str) -> Result<Option<Scene>, NarraError> {
    let result: Option<Scene> = db.delete(("scene", id)).await?;
    Ok(result)
}

/// Get all scenes at a specific event (timeline point).
///
/// # Arguments
///
/// * `db` - Database connection
/// * `event_id` - Event ID (the key part, not the full RecordId)
///
/// # Returns
///
/// A vector of scenes that occur at this event.
pub async fn get_scenes_at_event(db: &NarraDb, event_id: &str) -> Result<Vec<Scene>, NarraError> {
    let event_ref = RecordId::from(("event", event_id));
    let mut result = db
        .query("SELECT * FROM scene WHERE event = $event_ref")
        .bind(("event_ref", event_ref))
        .await?;
    let scenes: Vec<Scene> = result.take(0)?;
    Ok(scenes)
}

/// Get all scenes at a specific location.
///
/// Returns scenes where the location is either the primary or secondary location.
///
/// # Arguments
///
/// * `db` - Database connection
/// * `location_id` - Location ID (the key part, not the full RecordId)
///
/// # Returns
///
/// A vector of scenes that take place at this location.
pub async fn get_scenes_at_location(
    db: &NarraDb,
    location_id: &str,
) -> Result<Vec<Scene>, NarraError> {
    let loc_ref = RecordId::from(("location", location_id));
    let mut result = db
        .query("SELECT * FROM scene WHERE primary_location = $loc_ref OR $loc_ref IN secondary_locations")
        .bind(("loc_ref", loc_ref))
        .await?;
    let scenes: Vec<Scene> = result.take(0)?;
    Ok(scenes)
}

// ==============================================================================
// SCENE PARTICIPANTS (participates_in edge)
// ==============================================================================

/// A character's participation in a scene.
///
/// Uses SurrealDB's graph edge system via the RELATE statement.
/// The `in` and `out` fields are renamed from SurrealDB's reserved names.
///
/// Common roles include:
/// - pov (point-of-view character)
/// - protagonist
/// - antagonist
/// - supporting
/// - witness
/// - mentioned
///
/// Roles are user-extensible (not enforced by schema).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneParticipant {
    pub id: RecordId,
    #[serde(rename = "in")]
    pub character: RecordId,
    #[serde(rename = "out")]
    pub scene: RecordId,
    pub role: String,
    pub notes: Option<String>,
    pub created_at: Datetime,
}

/// Data for adding a scene participant.
#[derive(Debug)]
pub struct SceneParticipantCreate {
    pub character_id: String,
    pub scene_id: String,
    pub role: String,
    pub notes: Option<String>,
}

/// Add a character as a participant in a scene.
///
/// Uses SurrealDB's RELATE statement to create a graph edge.
///
/// # Arguments
///
/// * `db` - Database connection
/// * `data` - Participant creation data
///
/// # Returns
///
/// The created participation edge.
pub async fn add_scene_participant(
    db: &NarraDb,
    data: SceneParticipantCreate,
) -> Result<SceneParticipant, NarraError> {
    let query = format!(
        r#"RELATE character:{}->participates_in->scene:{} SET
            role = $role,
            notes = $notes"#,
        data.character_id, data.scene_id
    );

    let mut result = db
        .query(&query)
        .bind(("role", data.role))
        .bind(("notes", data.notes))
        .await?;

    let participant: Option<SceneParticipant> = result.take(0)?;
    participant.ok_or_else(|| NarraError::Database("Failed to add scene participant".into()))
}

/// Get all participants in a scene.
///
/// # Arguments
///
/// * `db` - Database connection
/// * `scene_id` - Scene ID (the key part, not the full RecordId)
///
/// # Returns
///
/// A vector of participation edges for this scene.
pub async fn get_scene_participants(
    db: &NarraDb,
    scene_id: &str,
) -> Result<Vec<SceneParticipant>, NarraError> {
    let query = format!(
        "SELECT * FROM participates_in WHERE out = scene:{}",
        scene_id
    );
    let mut result = db.query(&query).await?;
    let participants: Vec<SceneParticipant> = result.take(0)?;
    Ok(participants)
}

/// Get all scenes a character participates in.
///
/// # Arguments
///
/// * `db` - Database connection
/// * `character_id` - Character ID (the key part, not the full RecordId)
///
/// # Returns
///
/// A vector of participation edges for this character.
pub async fn get_character_scenes(
    db: &NarraDb,
    character_id: &str,
) -> Result<Vec<SceneParticipant>, NarraError> {
    let query = format!(
        "SELECT * FROM participates_in WHERE in = character:{}",
        character_id
    );
    let mut result = db.query(&query).await?;
    let participations: Vec<SceneParticipant> = result.take(0)?;
    Ok(participations)
}

/// Delete a scene participant by ID.
///
/// # Arguments
///
/// * `db` - Database connection
/// * `id` - Participation ID (the key part, not the full RecordId)
///
/// # Returns
///
/// The deleted participation if found, None otherwise.
pub async fn delete_scene_participant(
    db: &NarraDb,
    id: &str,
) -> Result<Option<SceneParticipant>, NarraError> {
    let result: Option<SceneParticipant> = db.delete(("participates_in", id)).await?;
    Ok(result)
}

// ==============================================================================
// EVENT INVOLVEMENT (involved_in edge)
// ==============================================================================

/// A character's involvement in a timeline event.
///
/// Uses SurrealDB's graph edge system via the RELATE statement.
/// The `in` and `out` fields are renamed from SurrealDB's reserved names.
///
/// Unlike scene participation (which tracks narrative presence),
/// event involvement tracks causal impact:
/// - Who was affected by this event?
/// - How did it impact them?
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Involvement {
    pub id: RecordId,
    #[serde(rename = "in")]
    pub character: RecordId,
    #[serde(rename = "out")]
    pub event: RecordId,
    pub role: Option<String>,
    pub impact: Option<String>,
    pub created_at: Datetime,
}

/// Data for adding event involvement.
#[derive(Debug)]
pub struct InvolvementCreate {
    pub character_id: String,
    pub event_id: String,
    pub role: Option<String>,
    pub impact: Option<String>,
}

/// Add a character's involvement in an event.
///
/// Uses SurrealDB's RELATE statement to create a graph edge.
///
/// # Arguments
///
/// * `db` - Database connection
/// * `data` - Involvement creation data
///
/// # Returns
///
/// The created involvement edge.
pub async fn add_event_involvement(
    db: &NarraDb,
    data: InvolvementCreate,
) -> Result<Involvement, NarraError> {
    let query = format!(
        r#"RELATE character:{}->involved_in->event:{} SET
            role = $role,
            impact = $impact"#,
        data.character_id, data.event_id
    );

    let mut result = db
        .query(&query)
        .bind(("role", data.role))
        .bind(("impact", data.impact))
        .await?;

    let involvement: Option<Involvement> = result.take(0)?;
    involvement.ok_or_else(|| NarraError::Database("Failed to add event involvement".into()))
}

/// Get all characters involved in an event.
///
/// # Arguments
///
/// * `db` - Database connection
/// * `event_id` - Event ID (the key part, not the full RecordId)
///
/// # Returns
///
/// A vector of involvement edges for this event.
pub async fn get_event_characters(
    db: &NarraDb,
    event_id: &str,
) -> Result<Vec<Involvement>, NarraError> {
    let event_ref = RecordId::from(("event", event_id));
    let mut result = db
        .query("SELECT * FROM involved_in WHERE out = $event_ref")
        .bind(("event_ref", event_ref))
        .await?;
    let involvements: Vec<Involvement> = result.take(0)?;
    Ok(involvements)
}

/// Get all events a character is involved in.
///
/// # Arguments
///
/// * `db` - Database connection
/// * `character_id` - Character ID (the key part, not the full RecordId)
///
/// # Returns
///
/// A vector of involvement edges for this character.
pub async fn get_character_events(
    db: &NarraDb,
    character_id: &str,
) -> Result<Vec<Involvement>, NarraError> {
    let char_ref = RecordId::from(("character", character_id));
    let mut result = db
        .query("SELECT * FROM involved_in WHERE in = $char_ref")
        .bind(("char_ref", char_ref))
        .await?;
    let involvements: Vec<Involvement> = result.take(0)?;
    Ok(involvements)
}

/// Delete an event involvement by ID.
///
/// # Arguments
///
/// * `db` - Database connection
/// * `id` - Involvement ID (the key part, not the full RecordId)
///
/// # Returns
///
/// The deleted involvement if found, None otherwise.
pub async fn delete_event_involvement(
    db: &NarraDb,
    id: &str,
) -> Result<Option<Involvement>, NarraError> {
    let result: Option<Involvement> = db.delete(("involved_in", id)).await?;
    Ok(result)
}
