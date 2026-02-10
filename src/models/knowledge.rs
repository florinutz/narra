use std::collections::HashSet;

use crate::db::connection::NarraDb;
use serde::{Deserialize, Serialize};
use surrealdb::{Datetime, RecordId};

use crate::models::event::get_event;
use crate::NarraError;

/// Certainty level for character knowledge.
///
/// Categorical levels from CONTEXT.md decisions:
/// - Knows: Certain, accurate knowledge
/// - Suspects: Uncertain, may be true
/// - BelievesWrongly: Certain but incorrect (requires truth_value field)
/// - Uncertain: Low confidence
/// - Assumes: Inference without evidence
/// - Denies: Actively rejects
/// - Forgotten: Previously knew, no longer remembers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum CertaintyLevel {
    #[default]
    Knows,
    Suspects,
    BelievesWrongly,
    Uncertain,
    Assumes,
    Denies,
    Forgotten,
}

/// Learning method for how knowledge was acquired.
///
/// From CONTEXT.md decisions, plus Initial for pre-story knowledge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LearningMethod {
    Told,
    Overheard,
    Witnessed,
    Discovered,
    Deduced,
    Read,
    Remembered,
    Initial, // Pre-story knowledge (no event required)
}

/// Knowledge state - what a character knows about a fact.
///
/// Note: Phase 1 is simple string fact. Phase 3 will add certainty, provenance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Knowledge {
    pub id: RecordId,
    pub character: RecordId, // record<character> reference
    pub fact: String,        // What they know
    pub created_at: Datetime,
}

/// A knowledge state edge: what a character knows about a fact.
///
/// Uses append-only pattern: new edges are created when certainty changes,
/// preserving history. Query with learned_at to get state at any point in time.
///
/// ## Provenance
///
/// Every knowledge state records:
/// - `learning_method`: How the character learned (told, overheard, etc.)
/// - `source_character`: Who told/showed them (if applicable)
/// - `event`: Where/when they learned (required except for Initial method)
/// - `premises`: For deductions, which knowledge pieces were combined
///
/// ## Notes on OUT field
///
/// The `out` field references the target - either:
/// - A knowledge record ID (for structured facts from Phase 1)
/// - A character record ID (for "knows about person")
///   Use the appropriate query pattern based on your use case.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeState {
    pub id: RecordId,
    #[serde(rename = "in")]
    pub character: RecordId, // Who knows (character:id)
    #[serde(rename = "out")]
    pub target: RecordId, // What they know about (knowledge:id or character:id)
    pub certainty: CertaintyLevel,
    pub learning_method: LearningMethod,
    pub source_character: Option<RecordId>, // Who told them (if applicable)
    pub event: Option<RecordId>,            // Event where learned
    pub premises: Option<Vec<RecordId>>,    // For deductions: source knowledge IDs
    pub truth_value: Option<String>,        // For BelievesWrongly: the actual truth
    pub learned_at: Datetime,
    pub created_at: Datetime,
    pub updated_at: Datetime,
}

/// Data for creating a new knowledge state.
#[derive(Debug, Clone, Serialize)]
pub struct KnowledgeStateCreate {
    pub certainty: CertaintyLevel,
    pub learning_method: LearningMethod,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_character: Option<String>, // Character ID (key only)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub event: Option<String>, // Event ID (key only)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub premises: Option<Vec<String>>, // knows edge IDs
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truth_value: Option<String>, // For BelievesWrongly
}

impl Default for KnowledgeStateCreate {
    fn default() -> Self {
        Self {
            certainty: CertaintyLevel::Knows,
            learning_method: LearningMethod::Initial,
            source_character: None,
            event: None,
            premises: None,
            truth_value: None,
        }
    }
}

/// Data for creating knowledge.
#[derive(Debug, Serialize)]
pub struct KnowledgeCreate {
    pub character: RecordId,
    pub fact: String,
}

/// Create a new knowledge entry in the database.
///
/// # Arguments
///
/// * `db` - Database connection
/// * `data` - Knowledge creation data
///
/// # Returns
///
/// The created knowledge with generated ID and timestamp.
pub async fn create_knowledge(
    db: &NarraDb,
    data: KnowledgeCreate,
) -> Result<Knowledge, NarraError> {
    let result: Option<Knowledge> = db.create("knowledge").content(data).await?;
    result.ok_or_else(|| NarraError::Database("Failed to create knowledge".into()))
}

/// Get a knowledge entry by ID.
///
/// # Arguments
///
/// * `db` - Database connection
/// * `id` - Knowledge ID (the key part, not the full RecordId)
///
/// # Returns
///
/// The knowledge if found, None otherwise.
pub async fn get_knowledge(db: &NarraDb, id: &str) -> Result<Option<Knowledge>, NarraError> {
    let result: Option<Knowledge> = db.select(("knowledge", id)).await?;
    Ok(result)
}

/// Get all knowledge entries for a specific character.
///
/// # Arguments
///
/// * `db` - Database connection
/// * `character_id` - Character ID (the key part, not the full RecordId)
///
/// # Returns
///
/// A vector of all facts the character knows.
pub async fn get_character_knowledge(
    db: &NarraDb,
    character_id: &str,
) -> Result<Vec<Knowledge>, NarraError> {
    let character = RecordId::from(("character", character_id));
    let mut result = db
        .query("SELECT * FROM knowledge WHERE character = $character")
        .bind(("character", character))
        .await?;
    let knowledge: Vec<Knowledge> = result.take(0)?;
    Ok(knowledge)
}

/// Delete a knowledge entry by ID.
///
/// # Arguments
///
/// * `db` - Database connection
/// * `id` - Knowledge ID (the key part, not the full RecordId)
///
/// # Returns
///
/// The deleted knowledge if found, None otherwise.
pub async fn delete_knowledge(db: &NarraDb, id: &str) -> Result<Option<Knowledge>, NarraError> {
    let result: Option<Knowledge> = db.delete(("knowledge", id)).await?;
    Ok(result)
}

/// Search knowledge entries by fact content.
///
/// # Arguments
///
/// * `db` - Database connection
/// * `fact_contains` - Text to search for within facts
///
/// # Returns
///
/// A vector of knowledge entries whose fact contains the search term.
pub async fn search_knowledge_by_fact(
    db: &NarraDb,
    fact_contains: &str,
) -> Result<Vec<Knowledge>, NarraError> {
    let search = fact_contains.to_string();
    let mut result = db
        .query("SELECT * FROM knowledge WHERE fact CONTAINS $search")
        .bind(("search", search))
        .await?;
    let knowledge: Vec<Knowledge> = result.take(0)?;
    Ok(knowledge)
}

/// Create knowledge by character ID (string) instead of RecordId.
///
/// This is a convenience function for easier API usage.
///
/// # Arguments
///
/// * `db` - Database connection
/// * `character_id` - Character ID (the key part, not the full RecordId)
/// * `fact` - The fact the character knows
///
/// # Returns
///
/// The created knowledge entry.
pub async fn record_character_knows(
    db: &NarraDb,
    character_id: &str,
    fact: &str,
) -> Result<Knowledge, NarraError> {
    let record_id = RecordId::from(("character", character_id));
    create_knowledge(
        db,
        KnowledgeCreate {
            character: record_id,
            fact: fact.to_string(),
        },
    )
    .await
}

// ============================================================================
// KnowledgeState CRUD (Phase 3 - knows edge)
// ============================================================================

/// Create a knowledge state edge linking character to target.
///
/// Target can be:
/// - A knowledge record ID: `create_knowledge_state(db, "alice", "knowledge:secret", data)`
/// - A character ID: `create_knowledge_state(db, "alice", "character:bob", data)` (knows about person)
///
/// ## Validation
///
/// - Non-Initial learning methods require an event
/// - BelievesWrongly requires truth_value
///
/// # Arguments
///
/// * `db` - Database connection
/// * `character_id` - Character who knows (key part only)
/// * `target` - Full target identifier (e.g., "knowledge:secret" or "character:bob")
/// * `data` - Knowledge state data
///
/// # Returns
///
/// The created knowledge state edge.
pub async fn create_knowledge_state(
    db: &NarraDb,
    character_id: &str,
    target: &str,
    data: KnowledgeStateCreate,
) -> Result<KnowledgeState, NarraError> {
    // Validation
    if data.learning_method != LearningMethod::Initial && data.event.is_none() {
        return Err(NarraError::Validation(
            "Non-initial knowledge requires an event".into(),
        ));
    }
    if data.certainty == CertaintyLevel::BelievesWrongly && data.truth_value.is_none() {
        return Err(NarraError::Validation(
            "BelievesWrongly requires truth_value".into(),
        ));
    }

    // Build optional references
    let source_ref = data
        .source_character
        .as_ref()
        .map(|id| RecordId::from(("character", id.as_str())));
    let event_ref = data
        .event
        .as_ref()
        .map(|id| RecordId::from(("event", id.as_str())));
    let premises_ref: Option<Vec<RecordId>> = data.premises.as_ref().map(|ids| {
        ids.iter()
            .map(|id| RecordId::from(("knows", id.as_str())))
            .collect()
    });

    // When knowledge is tied to an event, use the event's timestamp for learned_at
    // This ensures temporal queries work correctly ("what did X know at event Y")
    let learned_at_clause = if data.event.is_some() {
        "$event.created_at"
    } else {
        "time::now()"
    };

    let query = format!(
        r#"RELATE character:{}->knows->{} SET
            certainty = $certainty,
            learning_method = $method,
            source_character = $source,
            event = $event,
            premises = $premises,
            truth_value = $truth_value,
            learned_at = {}"#,
        character_id, target, learned_at_clause
    );

    let mut result = db
        .query(&query)
        .bind(("certainty", data.certainty))
        .bind(("method", data.learning_method))
        .bind(("source", source_ref))
        .bind(("event", event_ref))
        .bind(("premises", premises_ref))
        .bind(("truth_value", data.truth_value))
        .await?;

    let knowledge: Option<KnowledgeState> = result.take(0)?;
    knowledge.ok_or_else(|| NarraError::Database("Failed to create knowledge state".into()))
}

/// Get all knowledge states for a character (what they know).
///
/// Returns all current and historical knowledge states. Use learned_at
/// to filter for point-in-time queries.
///
/// # Arguments
///
/// * `db` - Database connection
/// * `character_id` - Character ID (key part only)
///
/// # Returns
///
/// All knowledge states where this character is the knower, ordered by learned_at desc.
pub async fn get_character_knowledge_states(
    db: &NarraDb,
    character_id: &str,
) -> Result<Vec<KnowledgeState>, NarraError> {
    let query = format!(
        "SELECT * FROM knows WHERE in = character:{} ORDER BY learned_at DESC",
        character_id
    );
    let mut result = db.query(&query).await?;
    let states: Vec<KnowledgeState> = result.take(0)?;
    Ok(states)
}

/// Get all characters who know about a specific target.
///
/// Fact-centric query: "Who knows about X?"
///
/// # Arguments
///
/// * `db` - Database connection
/// * `target` - Full target identifier (e.g., "knowledge:secret" or "character:bob")
///
/// # Returns
///
/// All knowledge states referencing this target, ordered by learned_at asc.
pub async fn get_fact_knowers(
    db: &NarraDb,
    target: &str,
) -> Result<Vec<KnowledgeState>, NarraError> {
    let query = format!(
        "SELECT * FROM knows WHERE out = {} ORDER BY learned_at ASC",
        target
    );
    let mut result = db.query(&query).await?;
    let states: Vec<KnowledgeState> = result.take(0)?;
    Ok(states)
}

/// Update certainty by creating a new edge (append-only pattern).
///
/// This preserves history - the old edge remains, new edge has updated certainty.
/// Temporal queries will see the certainty evolution.
///
/// # Arguments
///
/// * `db` - Database connection
/// * `character_id` - Character ID (key part only)
/// * `target` - Full target identifier
/// * `new_certainty` - The new certainty level
/// * `event_id` - Event where certainty changed (required)
///
/// # Returns
///
/// The new knowledge state edge with updated certainty.
pub async fn update_knowledge_certainty(
    db: &NarraDb,
    character_id: &str,
    target: &str,
    new_certainty: CertaintyLevel,
    event_id: &str,
) -> Result<KnowledgeState, NarraError> {
    create_knowledge_state(
        db,
        character_id,
        target,
        KnowledgeStateCreate {
            certainty: new_certainty,
            learning_method: LearningMethod::Discovered, // Generic for certainty changes
            event: Some(event_id.to_string()),
            ..Default::default()
        },
    )
    .await
}

/// Delete a knowledge state edge by ID.
///
/// Note: In append-only pattern, you typically don't delete history.
/// Use this for corrections/admin purposes only.
///
/// # Arguments
///
/// * `db` - Database connection
/// * `id` - Knowledge state edge ID (key part only)
///
/// # Returns
///
/// The deleted knowledge state if found.
pub async fn delete_knowledge_state(
    db: &NarraDb,
    id: &str,
) -> Result<Option<KnowledgeState>, NarraError> {
    let result: Option<KnowledgeState> = db.delete(("knows", id)).await?;
    Ok(result)
}

// ============================================================================
// Temporal Knowledge Queries (03-03)
// ============================================================================

/// Get a character's knowledge state at a specific event.
///
/// Returns what the character knew at the time of the event. Uses the event's
/// timestamp to filter knowledge edges learned before/at that time.
///
/// ## Deduplication
///
/// Since we use append-only edges, a character may have multiple edges for
/// the same target (certainty history). This returns only the most recent
/// state for each target as of the reference event.
///
/// # Arguments
///
/// * `db` - Database connection
/// * `character_id` - Character ID (key part only)
/// * `event_id` - Reference event ID (key part only)
///
/// # Returns
///
/// Knowledge states as of the event time, one per target (most recent certainty).
///
/// # Errors
///
/// Returns `NarraError::NotFound` if the reference event doesn't exist.
pub async fn get_knowledge_at_event(
    db: &NarraDb,
    character_id: &str,
    event_id: &str,
) -> Result<Vec<KnowledgeState>, NarraError> {
    // Get reference event for its timestamp
    let reference = get_event(db, event_id).await?;
    let reference = reference.ok_or_else(|| NarraError::NotFound {
        entity_type: "event".to_string(),
        id: event_id.to_string(),
    })?;

    // Query knowledge learned at or before this event
    // Note: We use event's created_at as the reference timestamp
    // Knowledge with event = this event is also included (learned during event)
    let query = format!(
        r#"SELECT * FROM knows
           WHERE in = character:{}
             AND learned_at <= $time
           ORDER BY out, learned_at DESC"#,
        character_id
    );

    let mut result = db
        .query(&query)
        .bind(("time", reference.created_at))
        .await?;

    let all_states: Vec<KnowledgeState> = result.take(0)?;

    // Deduplicate: keep only the most recent state per target
    // Since ordered by learned_at DESC, first occurrence for each target is most recent
    Ok(deduplicate_by_target(all_states))
}

/// Deduplicate knowledge states by target, keeping most recent per target.
fn deduplicate_by_target(states: Vec<KnowledgeState>) -> Vec<KnowledgeState> {
    let mut seen = HashSet::new();
    states
        .into_iter()
        .filter(|s| seen.insert(s.target.to_string()))
        .collect()
}

/// Get the certainty history for a specific character-target pair.
///
/// Returns all knowledge edges for this pair, ordered chronologically.
/// Shows how certainty evolved over time (suspected -> knew -> forgot).
///
/// # Arguments
///
/// * `db` - Database connection
/// * `character_id` - Character ID (key part only)
/// * `target` - Full target identifier (e.g., "knowledge:secret")
///
/// # Returns
///
/// All knowledge state edges for this pair, ordered by learned_at ascending.
pub async fn get_knowledge_history(
    db: &NarraDb,
    character_id: &str,
    target: &str,
) -> Result<Vec<KnowledgeState>, NarraError> {
    let query = format!(
        r#"SELECT * FROM knows
           WHERE in = character:{} AND out = {}
           ORDER BY learned_at ASC"#,
        character_id, target
    );

    let mut result = db.query(&query).await?;
    let history: Vec<KnowledgeState> = result.take(0)?;
    Ok(history)
}

/// Get the current (most recent) knowledge state for a character-target pair.
///
/// Convenience function when you only need the latest certainty level,
/// not the full history.
///
/// # Arguments
///
/// * `db` - Database connection
/// * `character_id` - Character ID (key part only)
/// * `target` - Full target identifier
///
/// # Returns
///
/// The most recent knowledge state, or None if character has never known this.
pub async fn get_current_knowledge(
    db: &NarraDb,
    character_id: &str,
    target: &str,
) -> Result<Option<KnowledgeState>, NarraError> {
    let history = get_knowledge_history(db, character_id, target).await?;
    Ok(history.into_iter().last()) // Last = most recent (ascending order)
}

// ============================================================================
// Provenance Queries (03-04)
// ============================================================================

/// Response type for transmission chain queries.
#[derive(Debug, Clone, Deserialize)]
pub struct KnowledgeTransmission {
    pub character_id: String,
    pub source_id: Option<String>,
    pub learning_method: LearningMethod,
    pub certainty: CertaintyLevel,
    pub learned_at: Datetime,
    pub event_id: Option<String>,
}

/// Get the transmission chain for a specific fact.
///
/// Shows how knowledge spread: who learned from whom, in what order.
/// Enables answering "How did this information spread through the story?"
///
/// # Arguments
///
/// * `db` - Database connection
/// * `target` - Full target identifier (e.g., "knowledge:secret")
///
/// # Returns
///
/// Transmission chain ordered by learned_at ascending (earliest first).
pub async fn get_transmission_chain(
    db: &NarraDb,
    target: &str,
) -> Result<Vec<KnowledgeTransmission>, NarraError> {
    let query = format!(
        r#"SELECT
            string::concat(in.tb, ':', in.id) AS character_id,
            IF source_character != NONE THEN string::concat(source_character.tb, ':', source_character.id) ELSE NONE END AS source_id,
            learning_method,
            certainty,
            learned_at,
            IF event != NONE THEN string::concat(event.tb, ':', event.id) ELSE NONE END AS event_id
           FROM knows
           WHERE out = {}
           ORDER BY learned_at ASC"#,
        target
    );

    let mut result = db.query(&query).await?;
    let chain: Vec<KnowledgeTransmission> = result.take(0)?;
    Ok(chain)
}

/// Response type for conflict detection.
#[derive(Debug, Clone, Deserialize)]
pub struct KnowledgeConflict {
    pub target: String,
    pub conflicting_states: Vec<ConflictingState>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ConflictingState {
    pub character_id: String,
    pub certainty: CertaintyLevel,
    pub truth_value: Option<String>,
}

/// Find knowledge conflicts where characters believe contradictory facts.
///
/// A conflict occurs when:
/// - At least one character has certainty = BelievesWrongly (explicit contradiction)
///
/// This implementation identifies BelievesWrongly states as potential conflicts.
/// Full conflict detection (comparing truth values across multiple characters
/// who "know" contradictory facts) is deferred per RESEARCH.md open questions.
///
/// # Arguments
///
/// * `db` - Database connection
///
/// # Returns
///
/// List of conflicts, each with the target and conflicting knowledge states.
pub async fn find_knowledge_conflicts(db: &NarraDb) -> Result<Vec<KnowledgeConflict>, NarraError> {
    // Find all BelievesWrongly states - these are explicit contradictions
    // Group by target to identify conflicting knowledge
    let query = r#"
        SELECT
            string::concat(out.tb, ':', out.id) AS target,
            string::concat(in.tb, ':', in.id) AS character_id,
            certainty,
            truth_value
        FROM knows
        WHERE certainty = 'believes_wrongly'
        ORDER BY target
    "#;

    let mut result = db.query(query).await?;

    #[derive(Debug, Deserialize)]
    struct WrongBeliefRow {
        target: String,
        character_id: String,
        certainty: CertaintyLevel,
        truth_value: Option<String>,
    }

    let wrong_beliefs: Vec<WrongBeliefRow> = result.take(0)?;

    // Group by target
    let mut conflicts_map: std::collections::HashMap<String, Vec<ConflictingState>> =
        std::collections::HashMap::new();

    for row in wrong_beliefs {
        let state = ConflictingState {
            character_id: row.character_id,
            certainty: row.certainty,
            truth_value: row.truth_value,
        };
        conflicts_map.entry(row.target).or_default().push(state);
    }

    // Convert to Vec<KnowledgeConflict>
    let conflicts: Vec<KnowledgeConflict> = conflicts_map
        .into_iter()
        .map(|(target, conflicting_states)| KnowledgeConflict {
            target,
            conflicting_states,
        })
        .collect();

    Ok(conflicts)
}

/// Get characters who could have told a character about something.
///
/// Based on scene co-presence: who was in the same scene as the character
/// when the knowledge was acquired?
///
/// # Arguments
///
/// * `db` - Database connection
/// * `knowledge_state_id` - Knowledge edge ID (key part only)
///
/// # Returns
///
/// List of character IDs who were present at the learning event.
pub async fn get_possible_sources(
    db: &NarraDb,
    knowledge_state_id: &str,
) -> Result<Vec<String>, NarraError> {
    // Get the knowledge state to find its event
    let state: Option<KnowledgeState> = db.select(("knows", knowledge_state_id)).await?;
    let state = state.ok_or_else(|| NarraError::NotFound {
        entity_type: "knowledge_state".to_string(),
        id: knowledge_state_id.to_string(),
    })?;

    // If no event, no scene-based sources
    let event_id = match state.event {
        Some(ref e) => e.key().to_string(),
        None => return Ok(vec![]),
    };

    // Find all scenes at this event, then all participants
    // Using participates_in edge from Phase 2
    let query = r#"
        SELECT DISTINCT string::concat(in.tb, ':', in.id) AS character_id
        FROM participates_in
        WHERE out.event.id = $event_id
    "#;

    let event_ref = RecordId::from(("event", event_id.as_str()));
    let mut result = db.query(query).bind(("event_id", event_ref)).await?;

    #[derive(Deserialize)]
    struct CharId {
        character_id: String,
    }

    let participants: Vec<CharId> = result.take(0)?;

    // Filter out the character themselves
    // RecordId::to_string() gives "table:key" format
    let character_key = state.character.to_string();
    Ok(participants
        .into_iter()
        .map(|p| p.character_id)
        .filter(|id| id != &character_key)
        .collect())
}
