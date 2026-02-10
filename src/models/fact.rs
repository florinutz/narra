//! Universe facts for world rules and constraints.
//!
//! Facts are first-class entities that define world rules with flexible enforcement.
//! They can be linked to entities via the applies_to edge and scoped by temporal
//! or POV context.

use crate::db::connection::NarraDb;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_with::skip_serializing_none;
use surrealdb::{Datetime, RecordId};

use crate::NarraError;

// ============================================================================
// Category and Enforcement Types
// ============================================================================

/// Category of a universe fact for organization and filtering.
///
/// Serializes as strings for database compatibility.
/// Known categories: "physics_magic", "social_cultural", "technology"
/// Custom categories are any other string value.
#[derive(Debug, Clone, PartialEq, Eq, JsonSchema)]
pub enum FactCategory {
    /// Physics and magic system rules
    PhysicsMagic,
    /// Social and cultural norms
    SocialCultural,
    /// Technology constraints and capabilities
    Technology,
    /// User-defined category
    Custom(String),
}

impl Serialize for FactCategory {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            FactCategory::PhysicsMagic => serializer.serialize_str("physics_magic"),
            FactCategory::SocialCultural => serializer.serialize_str("social_cultural"),
            FactCategory::Technology => serializer.serialize_str("technology"),
            FactCategory::Custom(s) => serializer.serialize_str(s),
        }
    }
}

impl<'de> Deserialize<'de> for FactCategory {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Ok(match s.as_str() {
            "physics_magic" => FactCategory::PhysicsMagic,
            "social_cultural" => FactCategory::SocialCultural,
            "technology" => FactCategory::Technology,
            other => FactCategory::Custom(other.to_string()),
        })
    }
}

/// Enforcement level determining how violations are handled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "snake_case")]
pub enum EnforcementLevel {
    /// Violations logged but not shown to user
    Informational,
    /// Violations shown as warnings, operations allowed
    #[default]
    Warning,
    /// Violations shown, must edit fact or entity to resolve
    Strict,
}

impl EnforcementLevel {
    /// Returns true if this enforcement level should block the operation.
    pub fn should_block_operation(&self) -> bool {
        matches!(self, EnforcementLevel::Strict)
    }

    /// Returns true if this enforcement level should show a warning to the user.
    pub fn should_show_warning(&self) -> bool {
        matches!(self, EnforcementLevel::Warning | EnforcementLevel::Strict)
    }
}

// ============================================================================
// Scope Types
// ============================================================================

/// Temporal scope defining when a fact is valid.
///
/// Event references are stored as strings (e.g., "event:abc123") for JSON compatibility.
#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct TemporalScope {
    /// Event ID from which this fact becomes valid (e.g., "event:abc123")
    pub valid_from_event: Option<String>,
    /// Event ID at which this fact ends (e.g., "event:abc123")
    pub valid_until_event: Option<String>,
    /// Freeform description for user-defined temporal bounds
    pub freeform_description: Option<String>,
}

/// POV scope defining who the fact applies to.
///
/// Character/entity references are stored as strings (e.g., "character:alice") for JSON compatibility.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum PovScope {
    /// Applies to a specific character (ID as string, e.g., "character:alice")
    Character(String),
    /// Applies to a named group
    Group(String),
    /// Applies globally except to listed characters (IDs as strings)
    ExceptCharacters(Vec<String>),
}

/// Combined scope for temporal and POV filtering.
#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct FactScope {
    /// Temporal bounds (AND logic with POV)
    pub temporal: Option<TemporalScope>,
    /// POV-specific scope (AND logic with temporal)
    pub pov: Option<PovScope>,
}

// ============================================================================
// Fact Entity Types
// ============================================================================

/// A universe fact defining world rules.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniverseFact {
    pub id: RecordId,
    pub title: String,
    pub description: String,
    pub categories: Vec<FactCategory>,
    pub enforcement_level: EnforcementLevel,
    pub scope: Option<FactScope>,
    pub created_at: Datetime,
    pub updated_at: Datetime,
}

/// Data for creating a new fact.
#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactCreate {
    pub title: String,
    pub description: String,
    #[serde(default)]
    pub categories: Vec<FactCategory>,
    #[serde(default)]
    pub enforcement_level: EnforcementLevel,
    pub scope: Option<FactScope>,
}

/// Data for updating an existing fact.
#[skip_serializing_none]
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FactUpdate {
    pub title: Option<String>,
    pub description: Option<String>,
    pub categories: Option<Vec<FactCategory>>,
    pub enforcement_level: Option<EnforcementLevel>,
    pub scope: Option<FactScope>,
    pub updated_at: Datetime,
}

/// An application edge from fact to any entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactApplication {
    pub id: RecordId,
    #[serde(rename = "in")]
    pub fact: RecordId,
    #[serde(rename = "out")]
    pub entity: RecordId,
    pub link_type: String,
    pub confidence: Option<f32>,
    pub created_at: Datetime,
}

// ============================================================================
// Fact CRUD Operations
// ============================================================================

/// Create a new universe fact.
///
/// # Arguments
///
/// * `db` - Database connection
/// * `data` - Fact creation data
///
/// # Returns
///
/// The created fact with generated ID and timestamps.
pub async fn create_fact(db: &NarraDb, data: FactCreate) -> Result<UniverseFact, NarraError> {
    let result: Option<UniverseFact> = db.create("universe_fact").content(data).await?;
    result.ok_or_else(|| NarraError::Database("Failed to create fact".into()))
}

/// Create a new universe fact with a caller-specified ID.
pub async fn create_fact_with_id(
    db: &NarraDb,
    id: &str,
    data: FactCreate,
) -> Result<UniverseFact, NarraError> {
    let result: Option<UniverseFact> = db.create(("universe_fact", id)).content(data).await?;
    result.ok_or_else(|| NarraError::Database("Failed to create fact".into()))
}

/// Get a fact by ID.
///
/// # Arguments
///
/// * `db` - Database connection
/// * `id` - Fact ID (the key part, not the full RecordId)
///
/// # Returns
///
/// The fact if found, None otherwise.
pub async fn get_fact(db: &NarraDb, id: &str) -> Result<Option<UniverseFact>, NarraError> {
    let result: Option<UniverseFact> = db.select(("universe_fact", id)).await?;
    Ok(result)
}

/// List all facts.
///
/// # Arguments
///
/// * `db` - Database connection
///
/// # Returns
///
/// A vector of all facts ordered by creation time descending.
pub async fn list_facts(db: &NarraDb) -> Result<Vec<UniverseFact>, NarraError> {
    let mut result = db
        .query("SELECT * FROM universe_fact ORDER BY created_at DESC")
        .await?;
    let facts: Vec<UniverseFact> = result.take(0)?;
    Ok(facts)
}

/// Update a fact by ID (partial update).
///
/// # Arguments
///
/// * `db` - Database connection
/// * `id` - Fact ID (the key part, not the full RecordId)
/// * `data` - Fields to update
///
/// # Returns
///
/// The updated fact if found, None otherwise.
pub async fn update_fact(
    db: &NarraDb,
    id: &str,
    data: FactUpdate,
) -> Result<Option<UniverseFact>, NarraError> {
    let result: Option<UniverseFact> = db.update(("universe_fact", id)).merge(data).await?;
    Ok(result)
}

/// Delete a fact by ID.
///
/// Note: This does NOT cascade to applies_to edges. Delete applications separately.
///
/// # Arguments
///
/// * `db` - Database connection
/// * `id` - Fact ID (the key part, not the full RecordId)
///
/// # Returns
///
/// The deleted fact if found, None otherwise.
pub async fn delete_fact(db: &NarraDb, id: &str) -> Result<Option<UniverseFact>, NarraError> {
    let result: Option<UniverseFact> = db.delete(("universe_fact", id)).await?;
    Ok(result)
}

// ============================================================================
// Fact Application Operations
// ============================================================================

/// Link a fact to an entity.
///
/// Uses SurrealDB RELATE to create a graph edge from fact to entity.
///
/// # Arguments
///
/// * `db` - Database connection
/// * `fact_id` - Fact ID (key part only)
/// * `entity_id` - Full entity identifier (e.g., "character:alice", "location:castle")
/// * `link_type` - Type of link ("manual" or "inferred")
/// * `confidence` - Confidence score for inferred links (optional)
///
/// # Returns
///
/// The created application edge.
pub async fn link_fact_to_entity(
    db: &NarraDb,
    fact_id: &str,
    entity_id: &str,
    link_type: &str,
    confidence: Option<f32>,
) -> Result<FactApplication, NarraError> {
    let query = format!(
        "RELATE universe_fact:{}->applies_to->{} SET link_type = $link_type, confidence = $confidence",
        fact_id, entity_id
    );

    let link_type_owned = link_type.to_string();
    let mut result = db
        .query(&query)
        .bind(("link_type", link_type_owned))
        .bind(("confidence", confidence))
        .await?;

    let app: Option<FactApplication> = result.take(0)?;
    app.ok_or_else(|| NarraError::Database("Failed to link fact to entity".into()))
}

/// Unlink a fact from an entity.
///
/// # Arguments
///
/// * `db` - Database connection
/// * `fact_id` - Fact ID (key part only)
/// * `entity_id` - Full entity identifier (e.g., "character:alice")
///
/// # Returns
///
/// Ok(()) if the unlinking succeeded (or edge didn't exist).
pub async fn unlink_fact_from_entity(
    db: &NarraDb,
    fact_id: &str,
    entity_id: &str,
) -> Result<(), NarraError> {
    let query = format!(
        "DELETE applies_to WHERE in = universe_fact:{} AND out = {}",
        fact_id, entity_id
    );
    db.query(&query).await?;
    Ok(())
}

/// Get all entities linked to a fact.
///
/// # Arguments
///
/// * `db` - Database connection
/// * `fact_id` - Fact ID (key part only)
///
/// # Returns
///
/// A vector of application edges for this fact.
pub async fn get_fact_applications(
    db: &NarraDb,
    fact_id: &str,
) -> Result<Vec<FactApplication>, NarraError> {
    let query = format!(
        "SELECT * FROM applies_to WHERE in = universe_fact:{}",
        fact_id
    );
    let mut result = db.query(&query).await?;
    let apps: Vec<FactApplication> = result.take(0)?;
    Ok(apps)
}

/// Get all facts linked to an entity.
///
/// # Arguments
///
/// * `db` - Database connection
/// * `entity_id` - Full entity identifier (e.g., "character:alice")
///
/// # Returns
///
/// A vector of facts linked to this entity.
pub async fn get_entity_facts(
    db: &NarraDb,
    entity_id: &str,
) -> Result<Vec<UniverseFact>, NarraError> {
    let query = format!(
        r#"SELECT in.* AS fact FROM applies_to WHERE out = {}"#,
        entity_id
    );

    let mut result = db.query(&query).await?;

    #[derive(Deserialize)]
    struct FactWrapper {
        fact: UniverseFact,
    }

    let wrappers: Vec<FactWrapper> = result.take(0)?;
    Ok(wrappers.into_iter().map(|w| w.fact).collect())
}

/// Get count of facts linked to an entity.
///
/// Optimized for summary display - returns just the count, not full facts.
///
/// # Arguments
///
/// * `db` - Database connection
/// * `entity_id` - Full entity identifier (e.g., "character:alice")
///
/// # Returns
///
/// The number of facts linked to this entity.
pub async fn get_fact_count_for_entity(db: &NarraDb, entity_id: &str) -> Result<usize, NarraError> {
    let query = format!(
        "SELECT count() AS count FROM applies_to WHERE out = {} GROUP ALL",
        entity_id
    );

    let mut result = db.query(&query).await?;

    #[derive(Deserialize)]
    struct CountResult {
        count: usize,
    }

    let counts: Vec<CountResult> = result.take(0)?;
    Ok(counts.first().map(|c| c.count).unwrap_or(0))
}

/// Get all entity IDs linked to a fact.
///
/// # Arguments
///
/// * `db` - Database connection
/// * `fact_id` - Fact ID (key part only)
///
/// # Returns
///
/// A vector of entity IDs (as strings) linked to this fact.
pub async fn get_entities_for_fact(db: &NarraDb, fact_id: &str) -> Result<Vec<String>, NarraError> {
    let query = format!(
        "SELECT out FROM applies_to WHERE in = universe_fact:{}",
        fact_id
    );

    let mut result = db.query(&query).await?;

    #[derive(Deserialize)]
    struct OutWrapper {
        out: RecordId,
    }

    let wrappers: Vec<OutWrapper> = result.take(0)?;
    Ok(wrappers.into_iter().map(|w| w.out.to_string()).collect())
}
