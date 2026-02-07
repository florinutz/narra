use serde::{Deserialize, Serialize};
use surrealdb::engine::local::Db;
use surrealdb::{Datetime, RecordId, Surreal};

use crate::NarraError;

/// Event entity for timeline ordering.
///
/// Events use a hybrid ordering system:
/// - `sequence` provides relative ordering (always works)
/// - `date` provides optional absolute positioning
/// - `date_precision` indicates how precise the date is ("year", "month", "day")
/// - `duration_end` allows events to span time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    pub id: RecordId,
    pub title: String,
    pub description: Option<String>,
    pub sequence: i64,
    pub date: Option<Datetime>,
    pub date_precision: Option<String>,
    pub duration_end: Option<Datetime>,
    pub created_at: Datetime,
    pub updated_at: Datetime,
}

/// Data for creating a new event.
#[derive(Debug, Serialize)]
pub struct EventCreate {
    pub title: String,
    pub description: Option<String>,
    pub sequence: i64,
    pub date: Option<Datetime>,
    pub date_precision: Option<String>,
    pub duration_end: Option<Datetime>,
}

/// Data for updating an event.
#[derive(Debug, Serialize)]
pub struct EventUpdate {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<Option<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sequence: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub date: Option<Option<Datetime>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub date_precision: Option<Option<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_end: Option<Option<Datetime>>,
    pub updated_at: Datetime,
}

/// Create a new event in the database.
///
/// # Arguments
///
/// * `db` - Database connection
/// * `data` - Event creation data
///
/// # Returns
///
/// The created event with generated ID and timestamps.
pub async fn create_event(db: &Surreal<Db>, data: EventCreate) -> Result<Event, NarraError> {
    let result: Option<Event> = db.create("event").content(data).await?;
    result.ok_or_else(|| NarraError::Database("Failed to create event".into()))
}

/// Create a new event with a caller-specified ID.
pub async fn create_event_with_id(
    db: &Surreal<Db>,
    id: &str,
    data: EventCreate,
) -> Result<Event, NarraError> {
    let result: Option<Event> = db.create(("event", id)).content(data).await?;
    result.ok_or_else(|| NarraError::Database("Failed to create event".into()))
}

/// Get an event by ID.
///
/// # Arguments
///
/// * `db` - Database connection
/// * `id` - Event ID (the key part, not the full RecordId)
///
/// # Returns
///
/// The event if found, None otherwise.
pub async fn get_event(db: &Surreal<Db>, id: &str) -> Result<Option<Event>, NarraError> {
    let result: Option<Event> = db.select(("event", id)).await?;
    Ok(result)
}

/// List all events ordered by sequence number (ascending).
///
/// This is the primary way to retrieve events for timeline display,
/// ensuring consistent ordering regardless of creation order.
///
/// # Arguments
///
/// * `db` - Database connection
///
/// # Returns
///
/// A vector of events sorted by sequence number.
pub async fn list_events_ordered(db: &Surreal<Db>) -> Result<Vec<Event>, NarraError> {
    let mut result = db
        .query("SELECT * FROM event ORDER BY sequence ASC")
        .await?;
    let events: Vec<Event> = result.take(0)?;
    Ok(events)
}

/// Update an event by ID.
///
/// # Arguments
///
/// * `db` - Database connection
/// * `id` - Event ID (the key part, not the full RecordId)
/// * `data` - Fields to update
///
/// # Returns
///
/// The updated event if found, None otherwise.
pub async fn update_event(
    db: &Surreal<Db>,
    id: &str,
    data: EventUpdate,
) -> Result<Option<Event>, NarraError> {
    let result: Option<Event> = db.update(("event", id)).merge(data).await?;
    Ok(result)
}

/// Delete an event by ID.
///
/// # Arguments
///
/// * `db` - Database connection
/// * `id` - Event ID (the key part, not the full RecordId)
///
/// # Returns
///
/// The deleted event if found, None otherwise.
///
/// # Errors
///
/// Returns `ReferentialIntegrityViolation` if the event is referenced by other entities
/// and cannot be deleted due to ON DELETE REJECT constraints.
pub async fn delete_event(db: &Surreal<Db>, id: &str) -> Result<Option<Event>, NarraError> {
    match db.delete::<Option<Event>>(("event", id)).await {
        Ok(result) => Ok(result),
        Err(e) => {
            let err_msg = e.to_string();
            // Check for referential integrity violation from SurrealDB
            if err_msg.contains("REFERENCE")
                || err_msg.contains("Cannot delete")
                || err_msg.contains("REJECT")
            {
                return Err(NarraError::ReferentialIntegrityViolation {
                    entity_type: "event".to_string(),
                    entity_id: id.to_string(),
                    message: "This event is referenced by scenes or knowledge entries and cannot be deleted.".to_string(),
                });
            }
            Err(NarraError::from(e))
        }
    }
}

/// Response type for max sequence query.
#[derive(Debug, Deserialize)]
struct MaxSeqResult {
    max_seq: Option<i64>,
}

/// Get the next available sequence number.
///
/// Returns the current maximum sequence + 1, or 1 if no events exist.
/// Use this when adding events at the end of the timeline.
///
/// # Arguments
///
/// * `db` - Database connection
///
/// # Returns
///
/// The next sequence number to use.
pub async fn get_next_sequence(db: &Surreal<Db>) -> Result<i64, NarraError> {
    let mut result = db
        .query("SELECT math::max(sequence) AS max_seq FROM event GROUP ALL")
        .await?;
    let row: Option<MaxSeqResult> = result.take(0)?;
    match row {
        Some(r) => Ok(r.max_seq.unwrap_or(0) + 1),
        None => Ok(1),
    }
}

/// Insert an event between two existing sequence positions.
///
/// Calculates the midpoint between `after_seq` and `before_seq` and creates
/// the event with that sequence number. This allows inserting without
/// renumbering existing events.
///
/// # Arguments
///
/// * `db` - Database connection
/// * `data` - Event creation data (sequence field will be overwritten)
/// * `after_seq` - Sequence number of the event this should come after
/// * `before_seq` - Sequence number of the event this should come before
///
/// # Returns
///
/// The created event with calculated sequence number.
///
/// # Note
///
/// Uses integer division. For very tight insertions, consider renumbering.
pub async fn insert_event_between(
    db: &Surreal<Db>,
    mut data: EventCreate,
    after_seq: i64,
    before_seq: i64,
) -> Result<Event, NarraError> {
    // Calculate midpoint (integer division)
    let midpoint = (after_seq + before_seq) / 2;
    data.sequence = midpoint;
    create_event(db, data).await
}

/// Get all events that occurred before a reference event.
///
/// Uses the sequence number for ordering, not dates. Returns events
/// with a lower sequence number than the reference event.
///
/// # Arguments
///
/// * `db` - Database connection
/// * `event_id` - Event ID of the reference event (the key part, not the full RecordId)
///
/// # Returns
///
/// A vector of events with lower sequence numbers, ordered by sequence ascending.
///
/// # Errors
///
/// Returns `NarraError::NotFound` if the reference event doesn't exist.
pub async fn get_events_before(db: &Surreal<Db>, event_id: &str) -> Result<Vec<Event>, NarraError> {
    // First get the reference event to find its sequence number
    let reference = get_event(db, event_id).await?;
    let reference = reference.ok_or_else(|| NarraError::NotFound {
        entity_type: "event".to_string(),
        id: event_id.to_string(),
    })?;

    let mut result = db
        .query("SELECT * FROM event WHERE sequence < $seq ORDER BY sequence ASC")
        .bind(("seq", reference.sequence))
        .await?;
    let events: Vec<Event> = result.take(0)?;
    Ok(events)
}

/// Get all events that occurred after a reference event.
///
/// Uses the sequence number for ordering, not dates. Returns events
/// with a higher sequence number than the reference event.
///
/// # Arguments
///
/// * `db` - Database connection
/// * `event_id` - Event ID of the reference event (the key part, not the full RecordId)
///
/// # Returns
///
/// A vector of events with higher sequence numbers, ordered by sequence ascending.
///
/// # Errors
///
/// Returns `NarraError::NotFound` if the reference event doesn't exist.
pub async fn get_events_after(db: &Surreal<Db>, event_id: &str) -> Result<Vec<Event>, NarraError> {
    // First get the reference event to find its sequence number
    let reference = get_event(db, event_id).await?;
    let reference = reference.ok_or_else(|| NarraError::NotFound {
        entity_type: "event".to_string(),
        id: event_id.to_string(),
    })?;

    let mut result = db
        .query("SELECT * FROM event WHERE sequence > $seq ORDER BY sequence ASC")
        .bind(("seq", reference.sequence))
        .await?;
    let events: Vec<Event> = result.take(0)?;
    Ok(events)
}

/// Get events in a sequence range (inclusive).
///
/// Returns all events whose sequence number falls within the specified range.
///
/// # Arguments
///
/// * `db` - Database connection
/// * `from_seq` - Minimum sequence number (inclusive)
/// * `to_seq` - Maximum sequence number (inclusive)
///
/// # Returns
///
/// A vector of events in the range, ordered by sequence ascending.
pub async fn get_events_in_range(
    db: &Surreal<Db>,
    from_seq: i64,
    to_seq: i64,
) -> Result<Vec<Event>, NarraError> {
    let mut result = db
        .query(
            "SELECT * FROM event WHERE sequence >= $from AND sequence <= $to ORDER BY sequence ASC",
        )
        .bind(("from", from_seq))
        .bind(("to", to_seq))
        .await?;
    let events: Vec<Event> = result.take(0)?;
    Ok(events)
}
