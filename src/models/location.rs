use crate::db::connection::NarraDb;
use serde::{Deserialize, Serialize};
use surrealdb::{Datetime, RecordId};

use crate::NarraError;

/// Location entity as stored in database.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Location {
    pub id: RecordId,
    pub name: String,
    pub description: Option<String>,
    pub loc_type: String,
    pub parent: Option<RecordId>,
    pub created_at: Datetime,
    pub updated_at: Datetime,
}

/// Data for creating a new location.
#[derive(Debug, Serialize)]
pub struct LocationCreate {
    pub name: String,
    pub description: Option<String>,
    pub loc_type: String,
    pub parent: Option<RecordId>,
}

/// Data for updating a location.
#[derive(Debug, Serialize)]
pub struct LocationUpdate {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<Option<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub loc_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent: Option<Option<RecordId>>,
    pub updated_at: Datetime,
}

/// Create a new location in the database.
///
/// # Arguments
///
/// * `db` - Database connection
/// * `data` - Location creation data
///
/// # Returns
///
/// The created location with generated ID and timestamps.
pub async fn create_location(db: &NarraDb, data: LocationCreate) -> Result<Location, NarraError> {
    let result: Option<Location> = db.create("location").content(data).await?;
    result.ok_or_else(|| NarraError::Database("Failed to create location".into()))
}

/// Create a new location with a caller-specified ID.
pub async fn create_location_with_id(
    db: &NarraDb,
    id: &str,
    data: LocationCreate,
) -> Result<Location, NarraError> {
    let result: Option<Location> = db.create(("location", id)).content(data).await?;
    result.ok_or_else(|| NarraError::Database("Failed to create location".into()))
}

/// Get a location by ID.
///
/// # Arguments
///
/// * `db` - Database connection
/// * `id` - Location ID (the key part, not the full RecordId)
///
/// # Returns
///
/// The location if found, None otherwise.
pub async fn get_location(db: &NarraDb, id: &str) -> Result<Option<Location>, NarraError> {
    let result: Option<Location> = db.select(("location", id)).await?;
    Ok(result)
}

/// List all locations.
///
/// # Arguments
///
/// * `db` - Database connection
///
/// # Returns
///
/// A vector of all locations.
pub async fn list_locations(db: &NarraDb) -> Result<Vec<Location>, NarraError> {
    let result: Vec<Location> = db.select("location").await?;
    Ok(result)
}

/// Update a location by ID.
///
/// # Arguments
///
/// * `db` - Database connection
/// * `id` - Location ID (the key part, not the full RecordId)
/// * `data` - Fields to update
///
/// # Returns
///
/// The updated location if found, None otherwise.
pub async fn update_location(
    db: &NarraDb,
    id: &str,
    data: LocationUpdate,
) -> Result<Option<Location>, NarraError> {
    let result: Option<Location> = db.update(("location", id)).merge(data).await?;
    Ok(result)
}

/// Delete a location by ID.
///
/// # Arguments
///
/// * `db` - Database connection
/// * `id` - Location ID (the key part, not the full RecordId)
///
/// # Returns
///
/// The deleted location if found, None otherwise.
///
/// # Errors
///
/// Returns `ReferentialIntegrityViolation` if the location is referenced by other entities
/// (e.g., used as primary_location in scenes or as parent of other locations) and cannot be
/// deleted due to ON DELETE REJECT constraints.
pub async fn delete_location(db: &NarraDb, id: &str) -> Result<Option<Location>, NarraError> {
    match db.delete::<Option<Location>>(("location", id)).await {
        Ok(result) => Ok(result),
        Err(e) => {
            let err_msg = e.to_string();
            // Check for referential integrity violation from SurrealDB
            if err_msg.contains("REFERENCE")
                || err_msg.contains("Cannot delete")
                || err_msg.contains("REJECT")
            {
                return Err(NarraError::ReferentialIntegrityViolation {
                    entity_type: "location".to_string(),
                    entity_id: id.to_string(),
                    message: "This location is used as a primary location in scenes or as a parent of other locations and cannot be deleted.".to_string(),
                });
            }
            Err(NarraError::from(e))
        }
    }
}

/// Get all child locations of a parent location.
///
/// # Arguments
///
/// * `db` - Database connection
/// * `parent_id` - Parent location ID (the key part, not the full RecordId)
///
/// # Returns
///
/// A vector of child locations.
pub async fn get_children(db: &NarraDb, parent_id: &str) -> Result<Vec<Location>, NarraError> {
    let parent = RecordId::from(("location", parent_id));
    let mut result = db
        .query("SELECT * FROM location WHERE parent = $parent")
        .bind(("parent", parent))
        .await?;
    let children: Vec<Location> = result.take(0)?;
    Ok(children)
}
