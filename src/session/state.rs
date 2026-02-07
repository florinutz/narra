use crate::error::NarraError;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;

/// A pending decision from impact analysis that requires user input.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingDecision {
    /// Unique identifier for the decision
    pub id: String,
    /// Human-readable description of the decision
    pub description: String,
    /// When the decision was created
    pub created_at: DateTime<Utc>,
    /// Entity IDs affected by this decision
    pub entity_ids: Vec<String>,
}

/// Session state that persists across process restarts.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SessionState {
    /// When the last session ended (None for new session)
    pub last_session: Option<DateTime<Utc>>,
    /// Explicitly pinned entity IDs
    pub pinned_entities: Vec<String>,
    /// Recent entity accesses (most recent first, max 100)
    pub recent_accesses: Vec<String>,
    /// Pending decisions from impact analysis
    pub pending_decisions: Vec<PendingDecision>,
}

/// Manages session state persistence to disk.
pub struct SessionStateManager {
    /// Path to the session state JSON file
    state_path: PathBuf,
    /// Current session state
    state: Arc<RwLock<SessionState>>,
}

impl SessionStateManager {
    /// Load session state from disk or create a new default state.
    pub fn load_or_create(path: &Path) -> Result<Self, NarraError> {
        let state = if path.exists() {
            // Load existing state
            let json = std::fs::read_to_string(path).map_err(|e| {
                NarraError::Database(format!("Failed to read session state: {}", e))
            })?;

            serde_json::from_str(&json).map_err(|e| {
                NarraError::Database(format!("Failed to parse session state: {}", e))
            })?
        } else {
            // Create default state
            SessionState::default()
        };

        Ok(Self {
            state_path: path.to_path_buf(),
            state: Arc::new(RwLock::new(state)),
        })
    }

    /// Persist current state to disk.
    pub async fn save(&self) -> Result<(), NarraError> {
        let state = self.state.read().await;
        let json = serde_json::to_string_pretty(&*state).map_err(|e| {
            NarraError::Database(format!("Failed to serialize session state: {}", e))
        })?;

        // Ensure parent directory exists
        if let Some(parent) = self.state_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                NarraError::Database(format!("Failed to create session directory: {}", e))
            })?;
        }

        std::fs::write(&self.state_path, json)
            .map_err(|e| NarraError::Database(format!("Failed to write session state: {}", e)))?;

        Ok(())
    }

    /// Record an entity access.
    ///
    /// Moves entity to the front if already present, otherwise adds it.
    /// Maintains max 100 entries.
    pub async fn record_access(&self, entity_id: &str) {
        let mut state = self.state.write().await;

        // Remove if already present
        state.recent_accesses.retain(|e| e != entity_id);

        // Add to front
        state.recent_accesses.insert(0, entity_id.to_string());

        // Keep only last 100
        if state.recent_accesses.len() > 100 {
            state.recent_accesses.truncate(100);
        }
    }

    /// Pin an entity.
    pub async fn pin_entity(&self, entity_id: &str) {
        let mut state = self.state.write().await;

        // Only add if not already pinned
        if !state.pinned_entities.contains(&entity_id.to_string()) {
            state.pinned_entities.push(entity_id.to_string());
        }
    }

    /// Unpin an entity.
    pub async fn unpin_entity(&self, entity_id: &str) {
        let mut state = self.state.write().await;
        state.pinned_entities.retain(|e| e != entity_id);
    }

    /// Get list of pinned entities.
    pub async fn get_pinned(&self) -> Vec<String> {
        let state = self.state.read().await;
        state.pinned_entities.clone()
    }

    /// Get recent entity accesses.
    ///
    /// Returns up to `limit` most recent accesses.
    pub async fn get_recent(&self, limit: usize) -> Vec<String> {
        let state = self.state.read().await;
        state.recent_accesses.iter().take(limit).cloned().collect()
    }

    /// Add a pending decision.
    pub async fn add_pending_decision(&self, decision: PendingDecision) {
        let mut state = self.state.write().await;
        state.pending_decisions.push(decision);
    }

    /// Resolve a pending decision by removing it.
    pub async fn resolve_pending_decision(&self, id: &str) {
        let mut state = self.state.write().await;
        state.pending_decisions.retain(|d| d.id != id);
    }

    /// Get all pending decisions.
    pub async fn get_pending_decisions(&self) -> Vec<PendingDecision> {
        let state = self.state.read().await;
        state.pending_decisions.clone()
    }

    /// Mark the end of a session.
    ///
    /// Sets last_session to current time.
    pub async fn mark_session_end(&self) {
        let mut state = self.state.write().await;
        state.last_session = Some(Utc::now());
    }

    /// Get the last session timestamp.
    pub async fn get_last_session(&self) -> Option<DateTime<Utc>> {
        let state = self.state.read().await;
        state.last_session
    }
}
