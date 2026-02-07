use serde::{Deserialize, Serialize};
use surrealdb::engine::local::Db;
use surrealdb::{Datetime, RecordId, Surreal};

use crate::NarraError;

/// Relationship edge between characters.
///
/// Uses SurrealDB's graph edge system via the RELATE statement.
/// The `in` and `out` fields are renamed from SurrealDB's reserved names.
///
/// Relationship types include:
/// - family (with subtypes: parent, child, sibling, spouse, etc.)
/// - romantic
/// - professional
/// - friendship
/// - rivalry
/// - mentorship
/// - alliance
///
/// Types are user-extensible (not enforced by schema).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relationship {
    pub id: RecordId,
    #[serde(rename = "in")]
    pub from_character: RecordId,
    #[serde(rename = "out")]
    pub to_character: RecordId,
    pub rel_type: String,
    pub subtype: Option<String>,
    pub label: Option<String>,
    pub created_at: Datetime,
}

/// Data for creating a relationship.
#[derive(Debug)]
pub struct RelationshipCreate {
    pub from_character_id: String,
    pub to_character_id: String,
    pub rel_type: String,
    pub subtype: Option<String>,
    pub label: Option<String>,
}

/// Create a relationship between two characters.
///
/// Uses SurrealDB's RELATE statement to create a graph edge.
///
/// # Arguments
///
/// * `db` - Database connection
/// * `data` - Relationship creation data
///
/// # Returns
///
/// The created relationship.
pub async fn create_relationship(
    db: &Surreal<Db>,
    data: RelationshipCreate,
) -> Result<Relationship, NarraError> {
    let query = format!(
        r#"RELATE character:{}->relates_to->character:{} SET
            rel_type = $rel_type,
            subtype = $subtype,
            label = $label"#,
        data.from_character_id, data.to_character_id
    );

    let mut result = db
        .query(&query)
        .bind(("rel_type", data.rel_type))
        .bind(("subtype", data.subtype))
        .bind(("label", data.label))
        .await?;

    let rel: Option<Relationship> = result.take(0)?;
    rel.ok_or_else(|| NarraError::Database("Failed to create relationship".into()))
}

/// Get relationships originating from a character (outgoing).
///
/// # Arguments
///
/// * `db` - Database connection
/// * `character_id` - Character ID (the key part, not the full RecordId)
///
/// # Returns
///
/// A vector of relationships where this character is the source.
pub async fn get_relationships_from(
    db: &Surreal<Db>,
    character_id: &str,
) -> Result<Vec<Relationship>, NarraError> {
    let query = format!(
        "SELECT * FROM relates_to WHERE in = character:{}",
        character_id
    );
    let mut result = db.query(&query).await?;
    let rels: Vec<Relationship> = result.take(0)?;
    Ok(rels)
}

/// Get relationships pointing to a character (incoming).
///
/// # Arguments
///
/// * `db` - Database connection
/// * `character_id` - Character ID (the key part, not the full RecordId)
///
/// # Returns
///
/// A vector of relationships where this character is the target.
pub async fn get_relationships_to(
    db: &Surreal<Db>,
    character_id: &str,
) -> Result<Vec<Relationship>, NarraError> {
    let query = format!(
        "SELECT * FROM relates_to WHERE out = character:{}",
        character_id
    );
    let mut result = db.query(&query).await?;
    let rels: Vec<Relationship> = result.take(0)?;
    Ok(rels)
}

/// Get all relationships involving a character (bidirectional).
///
/// Returns relationships where the character is either the source or target.
/// This is the primary way to query "all connections" for a character.
///
/// # Arguments
///
/// * `db` - Database connection
/// * `character_id` - Character ID (the key part, not the full RecordId)
///
/// # Returns
///
/// A vector of all relationships involving this character.
pub async fn get_all_relationships(
    db: &Surreal<Db>,
    character_id: &str,
) -> Result<Vec<Relationship>, NarraError> {
    let query = format!(
        "SELECT * FROM relates_to WHERE in = character:{} OR out = character:{}",
        character_id, character_id
    );
    let mut result = db.query(&query).await?;
    let rels: Vec<Relationship> = result.take(0)?;
    Ok(rels)
}

/// Delete a relationship by ID.
///
/// # Arguments
///
/// * `db` - Database connection
/// * `id` - Relationship ID (the key part, not the full RecordId)
///
/// # Returns
///
/// The deleted relationship if found, None otherwise.
pub async fn delete_relationship(
    db: &Surreal<Db>,
    id: &str,
) -> Result<Option<Relationship>, NarraError> {
    let result: Option<Relationship> = db.delete(("relates_to", id)).await?;
    Ok(result)
}
