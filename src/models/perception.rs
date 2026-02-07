//! Asymmetric relationship perceptions between characters.
//!
//! Each relationship is stored as TWO edges (A->B and B->A), allowing
//! each character to have independent feelings, perceptions, and tension
//! levels about the other. This models real-world asymmetric relationships
//! where what A knows about B differs from what B knows about A.

use serde::{Deserialize, Serialize};
use surrealdb::engine::local::Db;
use surrealdb::{Datetime, RecordId, Surreal};

use crate::NarraError;

/// A perception edge representing one character's view of another.
///
/// The `in` and `out` fields from SurrealDB's edge system are renamed
/// to `from_character` and `to_character` for clarity.
///
/// ## Relationship Types
///
/// Multiple relationship types are supported via `rel_types` (a set/vec):
/// - family (with subtypes: parent, child, sibling, spouse, etc.)
/// - romantic
/// - professional
/// - friendship
/// - rivalry
/// - mentorship
/// - alliance
///
/// A single perception can have multiple types (e.g., family + rivalry).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Perception {
    pub id: RecordId,
    #[serde(rename = "in")]
    pub from_character: RecordId,
    #[serde(rename = "out")]
    pub to_character: RecordId,
    /// Multiple relationship types (e.g., ["family", "professional"]).
    pub rel_types: Vec<String>,
    /// Subtype for more specific classification (e.g., "sibling" for family).
    pub subtype: Option<String>,
    /// How this character feels about the other.
    pub feelings: Option<String>,
    /// What this character believes/perceives about the other.
    pub perception: Option<String>,
    /// Tension level in this direction (0-10 scale suggested).
    pub tension_level: Option<i32>,
    /// Notes about the history of this relationship from this perspective.
    pub history_notes: Option<String>,
    pub created_at: Datetime,
    pub updated_at: Datetime,
}

/// Data for creating a new perception edge.
#[derive(Debug, Clone, Serialize)]
pub struct PerceptionCreate {
    pub rel_types: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subtype: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub feelings: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub perception: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tension_level: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub history_notes: Option<String>,
}

/// Data for updating a perception edge.
#[derive(Debug, Clone, Serialize)]
pub struct PerceptionUpdate {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rel_types: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subtype: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub feelings: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub perception: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tension_level: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub history_notes: Option<String>,
}

/// Create a single perception edge (one direction).
///
/// This creates A's perception of B. To create a bidirectional relationship,
/// also call this with arguments swapped, or use `create_perception_pair`.
///
/// # Arguments
///
/// * `db` - Database connection
/// * `from_character_id` - Character ID of the perceiver (key part only)
/// * `to_character_id` - Character ID of the perceived (key part only)
/// * `data` - Perception data
///
/// # Returns
///
/// The created perception edge.
pub async fn create_perception(
    db: &Surreal<Db>,
    from_character_id: &str,
    to_character_id: &str,
    data: PerceptionCreate,
) -> Result<Perception, NarraError> {
    let query = format!(
        r#"RELATE character:{}->perceives->character:{} SET
            rel_types = $rel_types,
            subtype = $subtype,
            feelings = $feelings,
            perception = $perception,
            tension_level = $tension_level,
            history_notes = $history_notes"#,
        from_character_id, to_character_id
    );

    let mut result = db
        .query(&query)
        .bind(("rel_types", data.rel_types))
        .bind(("subtype", data.subtype))
        .bind(("feelings", data.feelings))
        .bind(("perception", data.perception))
        .bind(("tension_level", data.tension_level))
        .bind(("history_notes", data.history_notes))
        .await?;

    let perception: Option<Perception> = result.take(0)?;
    perception.ok_or_else(|| NarraError::Database("Failed to create perception".into()))
}

/// Create a bidirectional relationship with TWO perception edges.
///
/// This is the primary way to establish a relationship between characters,
/// creating both A's view of B and B's view of A in a single operation.
///
/// # Arguments
///
/// * `db` - Database connection
/// * `from_char` - First character ID (key part only)
/// * `to_char` - Second character ID (key part only)
/// * `from_perspective` - How from_char perceives to_char
/// * `to_perspective` - How to_char perceives from_char
///
/// # Returns
///
/// A tuple of (from->to perception, to->from perception).
pub async fn create_perception_pair(
    db: &Surreal<Db>,
    from_char: &str,
    to_char: &str,
    from_perspective: PerceptionCreate,
    to_perspective: PerceptionCreate,
) -> Result<(Perception, Perception), NarraError> {
    // Create A's view of B
    let from_to = create_perception(db, from_char, to_char, from_perspective).await?;
    // Create B's view of A
    let to_from = create_perception(db, to_char, from_char, to_perspective).await?;

    Ok((from_to, to_from))
}

/// Get a specific perception (how from_char sees to_char).
///
/// # Arguments
///
/// * `db` - Database connection
/// * `from_char` - Character ID of the perceiver (key part only)
/// * `to_char` - Character ID of the perceived (key part only)
///
/// # Returns
///
/// The perception if found, None otherwise.
pub async fn get_perception(
    db: &Surreal<Db>,
    from_char: &str,
    to_char: &str,
) -> Result<Option<Perception>, NarraError> {
    let query = format!(
        "SELECT * FROM perceives WHERE in = character:{} AND out = character:{}",
        from_char, to_char
    );
    let mut result = db.query(&query).await?;
    let perceptions: Vec<Perception> = result.take(0)?;
    Ok(perceptions.into_iter().next())
}

/// Get all perceptions FROM a character (who they have views about).
///
/// Returns all perceptions where this character is the perceiver.
///
/// # Arguments
///
/// * `db` - Database connection
/// * `character_id` - Character ID (key part only)
///
/// # Returns
///
/// A vector of perceptions originating from this character.
pub async fn get_perceptions_from(
    db: &Surreal<Db>,
    character_id: &str,
) -> Result<Vec<Perception>, NarraError> {
    let query = format!(
        "SELECT * FROM perceives WHERE in = character:{}",
        character_id
    );
    let mut result = db.query(&query).await?;
    let perceptions: Vec<Perception> = result.take(0)?;
    Ok(perceptions)
}

/// Get all perceptions OF a character (how others see them).
///
/// Returns all perceptions where this character is the target.
///
/// # Arguments
///
/// * `db` - Database connection
/// * `character_id` - Character ID (key part only)
///
/// # Returns
///
/// A vector of perceptions targeting this character.
pub async fn get_perceptions_of(
    db: &Surreal<Db>,
    character_id: &str,
) -> Result<Vec<Perception>, NarraError> {
    let query = format!(
        "SELECT * FROM perceives WHERE out = character:{}",
        character_id
    );
    let mut result = db.query(&query).await?;
    let perceptions: Vec<Perception> = result.take(0)?;
    Ok(perceptions)
}

/// Get perceptions by relationship type.
///
/// Finds all perceptions that include the given type in their rel_types set.
///
/// # Arguments
///
/// * `db` - Database connection
/// * `rel_type` - Relationship type to search for
///
/// # Returns
///
/// A vector of perceptions with the given type.
pub async fn get_perceptions_by_type(
    db: &Surreal<Db>,
    rel_type: &str,
) -> Result<Vec<Perception>, NarraError> {
    let query = "SELECT * FROM perceives WHERE $rel_type IN rel_types";
    let mut result = db
        .query(query)
        .bind(("rel_type", rel_type.to_string()))
        .await?;
    let perceptions: Vec<Perception> = result.take(0)?;
    Ok(perceptions)
}

/// Update a perception by ID.
///
/// # Arguments
///
/// * `db` - Database connection
/// * `id` - Perception ID (the key part, not the full RecordId)
/// * `data` - Fields to update
///
/// # Returns
///
/// The updated perception if found, None otherwise.
pub async fn update_perception(
    db: &Surreal<Db>,
    id: &str,
    data: PerceptionUpdate,
) -> Result<Option<Perception>, NarraError> {
    let result: Option<Perception> = db.update(("perceives", id)).merge(data).await?;
    Ok(result)
}

/// Delete a perception by ID.
///
/// # Arguments
///
/// * `db` - Database connection
/// * `id` - Perception ID (the key part, not the full RecordId)
///
/// # Returns
///
/// The deleted perception if found, None otherwise.
pub async fn delete_perception(
    db: &Surreal<Db>,
    id: &str,
) -> Result<Option<Perception>, NarraError> {
    let result: Option<Perception> = db.delete(("perceives", id)).await?;
    Ok(result)
}
