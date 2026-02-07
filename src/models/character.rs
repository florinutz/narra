use serde::{Deserialize, Serialize};
use serde_with::skip_serializing_none;
use std::collections::HashMap;
use surrealdb::engine::local::Db;
use surrealdb::{Datetime, RecordId, Surreal};

use crate::NarraError;

/// Character entity as stored in database.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Character {
    pub id: RecordId,
    pub name: String,
    pub aliases: Vec<String>,
    pub roles: Vec<String>,
    /// Flexible character profile — keys are fiction-framework categories
    /// (e.g. "wound", "desire_conscious", "contradiction", "secret"),
    /// values are lists of entries for that category.
    #[serde(default)]
    pub profile: HashMap<String, Vec<String>>,
    pub created_at: Datetime,
    pub updated_at: Datetime,
}

/// Data for creating a new character.
#[derive(Debug, Default, Serialize)]
pub struct CharacterCreate {
    pub name: String,
    pub aliases: Vec<String>,
    pub roles: Vec<String>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub profile: HashMap<String, Vec<String>>,
}

/// Data for updating a character.
#[skip_serializing_none]
#[derive(Debug, Default, Serialize)]
pub struct CharacterUpdate {
    pub name: Option<String>,
    pub aliases: Option<Vec<String>>,
    pub roles: Option<Vec<String>>,
    pub profile: Option<HashMap<String, Vec<String>>>,
    pub updated_at: Datetime,
}

/// Create a new character in the database.
///
/// # Arguments
///
/// * `db` - Database connection
/// * `data` - Character creation data
///
/// # Returns
///
/// The created character with generated ID and timestamps.
pub async fn create_character(
    db: &Surreal<Db>,
    data: CharacterCreate,
) -> Result<Character, NarraError> {
    let result: Option<Character> = db.create("character").content(data).await?;
    result.ok_or_else(|| NarraError::Database("Failed to create character".into()))
}

/// Create a new character with a caller-specified ID.
pub async fn create_character_with_id(
    db: &Surreal<Db>,
    id: &str,
    data: CharacterCreate,
) -> Result<Character, NarraError> {
    let result: Option<Character> = db.create(("character", id)).content(data).await?;
    result.ok_or_else(|| NarraError::Database("Failed to create character".into()))
}

/// Get a character by ID.
///
/// # Arguments
///
/// * `db` - Database connection
/// * `id` - Character ID (the key part, not the full RecordId)
///
/// # Returns
///
/// The character if found, None otherwise.
pub async fn get_character(db: &Surreal<Db>, id: &str) -> Result<Option<Character>, NarraError> {
    let result: Option<Character> = db.select(("character", id)).await?;
    Ok(result)
}

/// List all characters.
///
/// # Arguments
///
/// * `db` - Database connection
///
/// # Returns
///
/// A vector of all characters.
pub async fn list_characters(db: &Surreal<Db>) -> Result<Vec<Character>, NarraError> {
    let result: Vec<Character> = db.select("character").await?;
    Ok(result)
}

/// Update a character by ID.
///
/// # Arguments
///
/// * `db` - Database connection
/// * `id` - Character ID (the key part, not the full RecordId)
/// * `data` - Fields to update
///
/// # Returns
///
/// The updated character if found, None otherwise.
pub async fn update_character(
    db: &Surreal<Db>,
    id: &str,
    data: CharacterUpdate,
) -> Result<Option<Character>, NarraError> {
    let result: Option<Character> = db.update(("character", id)).merge(data).await?;
    Ok(result)
}

/// Delete a character by ID.
///
/// # Arguments
///
/// * `db` - Database connection
/// * `id` - Character ID (the key part, not the full RecordId)
///
/// # Returns
///
/// The deleted character if found, None otherwise.
///
/// # Errors
///
/// Returns `ReferentialIntegrityViolation` if the character is referenced by other entities
/// (e.g., involved in historical events) and cannot be deleted due to ON DELETE REJECT constraints.
pub async fn delete_character(db: &Surreal<Db>, id: &str) -> Result<Option<Character>, NarraError> {
    match db.delete::<Option<Character>>(("character", id)).await {
        Ok(result) => Ok(result),
        Err(e) => {
            let err_msg = e.to_string();
            // Check for referential integrity violation from SurrealDB
            if err_msg.contains("REFERENCE")
                || err_msg.contains("Cannot delete")
                || err_msg.contains("REJECT")
            {
                return Err(NarraError::ReferentialIntegrityViolation {
                    entity_type: "character".to_string(),
                    entity_id: id.to_string(),
                    message: "This character is involved in historical events and cannot be deleted. Remove event participations first.".to_string(),
                });
            }
            Err(NarraError::from(e))
        }
    }
}
