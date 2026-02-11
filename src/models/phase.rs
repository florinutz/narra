//! Persisted narrative phases from temporal-semantic clustering.
//!
//! Phases are detected via `TemporalService::detect_phases()` and optionally
//! saved to SurrealDB as `phase:phase_0`, `phase:phase_1`, etc. with
//! `belongs_to_phase` membership edges for instant loading.

use crate::db::connection::NarraDb;
use crate::NarraError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use surrealdb::{Datetime, RecordId};

/// A persisted narrative phase.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Phase {
    pub id: RecordId,
    pub name: String,
    pub label: String,
    pub phase_order: i64,
    pub sequence_range_min: Option<i64>,
    pub sequence_range_max: Option<i64>,
    pub entity_type_counts: HashMap<String, serde_json::Value>,
    pub weights_content: f64,
    pub weights_neighborhood: f64,
    pub weights_temporal: f64,
    pub member_count: i64,
    pub created_at: Datetime,
    pub updated_at: Datetime,
}

/// Data for creating a new phase.
#[derive(Debug, Clone, Serialize)]
pub struct PhaseCreate {
    pub name: String,
    pub label: String,
    pub phase_order: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sequence_range_min: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sequence_range_max: Option<i64>,
    pub entity_type_counts: HashMap<String, serde_json::Value>,
    pub weights_content: f64,
    pub weights_neighborhood: f64,
    pub weights_temporal: f64,
    pub member_count: i64,
}

/// A membership edge: entity belongs to a phase.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseMembership {
    pub id: RecordId,
    #[serde(rename = "in")]
    pub entity: RecordId,
    #[serde(rename = "out")]
    pub phase: RecordId,
    pub entity_type: String,
    pub entity_name: String,
    pub centrality: f64,
    pub sequence_position: Option<f64>,
}

// ============================================================================
// Phase CRUD Operations
// ============================================================================

/// Create a phase with an explicit ID (e.g., "phase_0").
pub async fn create_phase_with_id(
    db: &NarraDb,
    id: &str,
    data: PhaseCreate,
) -> Result<Phase, NarraError> {
    let result: Option<Phase> = db.create(("phase", id)).content(data).await?;
    result.ok_or_else(|| NarraError::Database("Failed to create phase".into()))
}

/// Get a single phase by ID.
pub async fn get_phase(db: &NarraDb, id: &str) -> Result<Option<Phase>, NarraError> {
    let result: Option<Phase> = db.select(("phase", id)).await?;
    Ok(result)
}

/// List all phases ordered by phase_order.
pub async fn list_phases(db: &NarraDb) -> Result<Vec<Phase>, NarraError> {
    let mut result = db
        .query("SELECT * FROM phase ORDER BY phase_order ASC")
        .await?;
    let phases: Vec<Phase> = result.take(0)?;
    Ok(phases)
}

/// Delete all phases and membership edges (wipe before re-detection).
pub async fn delete_all_phases(db: &NarraDb) -> Result<usize, NarraError> {
    // Count before deleting
    let mut count_resp = db.query("SELECT count() FROM phase GROUP ALL").await?;

    #[derive(Deserialize)]
    struct CountResult {
        count: i64,
    }

    let counts: Vec<CountResult> = count_resp.take(0).unwrap_or_default();
    let count = counts.first().map(|c| c.count as usize).unwrap_or(0);

    // Delete edges first, then phases
    db.query("DELETE FROM belongs_to_phase").await?;
    db.query("DELETE FROM phase").await?;

    Ok(count)
}

/// Create a membership edge from entity to phase.
pub async fn create_membership(
    db: &NarraDb,
    entity_id: &str,
    phase_id: &str,
    entity_type: &str,
    entity_name: &str,
    centrality: f64,
    sequence_position: Option<f64>,
) -> Result<PhaseMembership, NarraError> {
    let (entity_table, entity_key) = entity_id
        .split_once(':')
        .ok_or_else(|| NarraError::Validation(format!("Invalid entity ID: {}", entity_id)))?;
    let entity_ref = RecordId::from((entity_table, entity_key));
    let phase_ref = RecordId::from(("phase", phase_id));

    let mut result = db
        .query(
            "RELATE $from->belongs_to_phase->$to SET \
             entity_type = $entity_type, \
             entity_name = $entity_name, \
             centrality = $centrality, \
             sequence_position = $seq_pos",
        )
        .bind(("from", entity_ref))
        .bind(("to", phase_ref))
        .bind(("entity_type", entity_type.to_string()))
        .bind(("entity_name", entity_name.to_string()))
        .bind(("centrality", centrality))
        .bind(("seq_pos", sequence_position))
        .await?;

    let membership: Option<PhaseMembership> = result.take(0)?;
    membership.ok_or_else(|| NarraError::Database("Failed to create phase membership edge".into()))
}

/// Get all members of a phase.
pub async fn get_phase_members(
    db: &NarraDb,
    phase_id: &str,
) -> Result<Vec<PhaseMembership>, NarraError> {
    let phase_ref = RecordId::from(("phase", phase_id));
    let mut result = db
        .query("SELECT * FROM belongs_to_phase WHERE out = $phase_ref ORDER BY centrality DESC")
        .bind(("phase_ref", phase_ref))
        .await?;
    let members: Vec<PhaseMembership> = result.take(0)?;
    Ok(members)
}
