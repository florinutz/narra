//! Freeform notes for worldbuilding ideas, research, and context.
//!
//! Notes can be standalone or attached to any entity (character, location,
//! event, scene, knowledge). Attachments are stored as graph edges in the
//! note_attachment table.

use serde::{Deserialize, Serialize};
use surrealdb::engine::local::Db;
use surrealdb::{Datetime, RecordId, Surreal};

use crate::NarraError;

/// A freeform note with optional entity attachments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Note {
    pub id: RecordId,
    pub title: String,
    pub body: String,
    pub created_at: Datetime,
    pub updated_at: Datetime,
}

/// Data for creating a new note.
#[derive(Debug, Clone, Serialize)]
pub struct NoteCreate {
    pub title: String,
    pub body: String,
}

/// Data for updating an existing note.
#[derive(Debug, Clone, Serialize)]
pub struct NoteUpdate {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body: Option<String>,
}

/// An attachment edge from note to any entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoteAttachment {
    pub id: RecordId,
    #[serde(rename = "in")]
    pub note: RecordId,
    #[serde(rename = "out")]
    pub entity: RecordId,
    pub attached_at: Datetime,
}

// ============================================================================
// Note CRUD Operations
// ============================================================================

/// Create a new note.
///
/// # Arguments
///
/// * `db` - Database connection
/// * `data` - Note creation data
///
/// # Returns
///
/// The created note with generated ID and timestamps.
pub async fn create_note(db: &Surreal<Db>, data: NoteCreate) -> Result<Note, NarraError> {
    let result: Option<Note> = db.create("note").content(data).await?;
    result.ok_or_else(|| NarraError::Database("Failed to create note".into()))
}

/// Create a new note with a caller-specified ID.
pub async fn create_note_with_id(
    db: &Surreal<Db>,
    id: &str,
    data: NoteCreate,
) -> Result<Note, NarraError> {
    let result: Option<Note> = db.create(("note", id)).content(data).await?;
    result.ok_or_else(|| NarraError::Database("Failed to create note".into()))
}

/// Get a note by ID.
///
/// # Arguments
///
/// * `db` - Database connection
/// * `id` - Note ID (the key part, not the full RecordId)
///
/// # Returns
///
/// The note if found, None otherwise.
pub async fn get_note(db: &Surreal<Db>, id: &str) -> Result<Option<Note>, NarraError> {
    let result: Option<Note> = db.select(("note", id)).await?;
    Ok(result)
}

/// Update a note by ID (partial update).
///
/// # Arguments
///
/// * `db` - Database connection
/// * `id` - Note ID (the key part, not the full RecordId)
/// * `data` - Fields to update
///
/// # Returns
///
/// The updated note if found, None otherwise.
pub async fn update_note(
    db: &Surreal<Db>,
    id: &str,
    data: NoteUpdate,
) -> Result<Option<Note>, NarraError> {
    let result: Option<Note> = db.update(("note", id)).merge(data).await?;
    Ok(result)
}

/// Delete a note by ID.
///
/// Note: This does NOT cascade to attachments. Use detach_note first
/// or delete attachments separately.
///
/// # Arguments
///
/// * `db` - Database connection
/// * `id` - Note ID (the key part, not the full RecordId)
///
/// # Returns
///
/// The deleted note if found, None otherwise.
pub async fn delete_note(db: &Surreal<Db>, id: &str) -> Result<Option<Note>, NarraError> {
    let result: Option<Note> = db.delete(("note", id)).await?;
    Ok(result)
}

/// List notes with pagination.
///
/// # Arguments
///
/// * `db` - Database connection
/// * `limit` - Maximum number of notes to return
/// * `offset` - Number of notes to skip
///
/// # Returns
///
/// A vector of notes ordered by created_at descending (newest first).
pub async fn list_notes(
    db: &Surreal<Db>,
    limit: usize,
    offset: usize,
) -> Result<Vec<Note>, NarraError> {
    let mut result = db
        .query("SELECT * FROM note ORDER BY created_at DESC LIMIT $limit START $offset")
        .bind(("limit", limit))
        .bind(("offset", offset))
        .await?;
    let notes: Vec<Note> = result.take(0)?;
    Ok(notes)
}

// ============================================================================
// Note Attachment Operations
// ============================================================================

/// Attach a note to any entity.
///
/// Uses SurrealDB RELATE to create a graph edge from note to entity.
///
/// # Arguments
///
/// * `db` - Database connection
/// * `note_id` - Note ID (key part only)
/// * `entity_id` - Full entity identifier (e.g., "character:alice", "location:castle")
///
/// # Returns
///
/// The created attachment edge.
pub async fn attach_note(
    db: &Surreal<Db>,
    note_id: &str,
    entity_id: &str,
) -> Result<NoteAttachment, NarraError> {
    let note_ref = RecordId::from(("note", note_id));
    let (table, key) = entity_id
        .split_once(':')
        .ok_or_else(|| NarraError::Validation(format!("Invalid entity ID: {}", entity_id)))?;
    let entity_ref = RecordId::from((table, key));

    let mut result = db
        .query("RELATE $from->note_attachment->$to")
        .bind(("from", note_ref))
        .bind(("to", entity_ref))
        .await?;
    let attachment: Option<NoteAttachment> = result.take(0)?;
    attachment.ok_or_else(|| NarraError::Database("Failed to create note attachment".into()))
}

/// Detach a note from a specific entity.
///
/// # Arguments
///
/// * `db` - Database connection
/// * `note_id` - Note ID (key part only)
/// * `entity_id` - Full entity identifier (e.g., "character:alice")
///
/// # Returns
///
/// Ok(()) if detachment succeeded (or edge didn't exist).
pub async fn detach_note(
    db: &Surreal<Db>,
    note_id: &str,
    entity_id: &str,
) -> Result<(), NarraError> {
    let note_ref = RecordId::from(("note", note_id));
    let (table, key) = entity_id
        .split_once(':')
        .ok_or_else(|| NarraError::Validation(format!("Invalid entity ID: {}", entity_id)))?;
    let entity_ref = RecordId::from((table, key));

    db.query("DELETE note_attachment WHERE in = $note_ref AND out = $entity_ref")
        .bind(("note_ref", note_ref))
        .bind(("entity_ref", entity_ref))
        .await?;
    Ok(())
}

/// Get all attachments for a note.
///
/// # Arguments
///
/// * `db` - Database connection
/// * `note_id` - Note ID (key part only)
///
/// # Returns
///
/// A vector of attachment edges for this note.
pub async fn get_note_attachments(
    db: &Surreal<Db>,
    note_id: &str,
) -> Result<Vec<NoteAttachment>, NarraError> {
    let note_ref = RecordId::from(("note", note_id));
    let mut result = db
        .query("SELECT * FROM note_attachment WHERE in = $note_ref")
        .bind(("note_ref", note_ref))
        .await?;
    let attachments: Vec<NoteAttachment> = result.take(0)?;
    Ok(attachments)
}

/// Get all notes attached to a specific entity.
///
/// # Arguments
///
/// * `db` - Database connection
/// * `entity_id` - Full entity identifier (e.g., "character:alice")
///
/// # Returns
///
/// A vector of notes attached to this entity.
pub async fn get_entity_notes(db: &Surreal<Db>, entity_id: &str) -> Result<Vec<Note>, NarraError> {
    let (table, key) = entity_id
        .split_once(':')
        .ok_or_else(|| NarraError::Validation(format!("Invalid entity ID: {}", entity_id)))?;
    let entity_ref = RecordId::from((table, key));

    let mut result = db
        .query("SELECT in.* AS note FROM note_attachment WHERE out = $entity_ref")
        .bind(("entity_ref", entity_ref))
        .await?;

    // Result comes back with note field containing the Note
    #[derive(Deserialize)]
    struct NoteWrapper {
        note: Note,
    }

    let wrappers: Vec<NoteWrapper> = result.take(0)?;
    Ok(wrappers.into_iter().map(|w| w.note).collect())
}
